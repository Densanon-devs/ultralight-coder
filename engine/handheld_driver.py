"""
Handheld driver — interleaved per-step execution with prior-state injection.

Different from `engine/architect_agent.py`:

  ARCHITECT (existing):
    plan once -> spawn N sub-agents in parallel-context-isolation.
    Each sub-agent sees: goal + step text + (lite) prior step titles.
    Each sub-agent has FRESH transcript. They share workspace but
    don't share intent. Cross-file consistency dies (Goal 1.5 of the
    2026-04-26 walkthrough proved this catastrophically).

  HANDHELD (new):
    plan once -> drive N sub-agents SEQUENTIALLY.
    Each sub-agent sees: goal + plan + ACTUAL deliverable summaries
    from EVERY prior step (which files they wrote, what tests ran,
    what errors fired and got fixed). Each sub-agent still has its own
    transcript so context windows stay manageable, but the system-prompt
    extra carries forward what matters. Trades VRAM-per-step for
    cross-step consistency.

  Use this when:
    - The project is large enough to exceed a flat agent's context
      window OR iteration budget.
    - Goal has cross-file dependencies (later step needs to know what
      types/functions earlier steps defined).
    - The 14B has been failing at the integrated task but each step is
      individually within its grasp.

  Don't use this when:
    - The project is small (< 4 files) — flat agent is faster.
    - The plan can't be enumerated up front — handheld is plan-driven.

Schema of the plan (JSON, the model's first response):

    {
      "goal": "<one-line restatement>",
      "steps": [
        {
          "n": 1,
          "title": "<short imperative>",
          "files": ["pomodoro/timer.py"],
          "description": "<what to do>",
          "success_criteria": "<how to verify>"
        },
        ...
      ],
      "final_verification": "Run pytest -q"
    }
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

from engine.agent import Agent, AgentEvent, AgentResult
from engine.agent_memory import AgentMemory
from engine.agent_tools import ToolRegistry, ToolCall, ToolResult

logger = logging.getLogger(__name__)


# ── Plan schema ─────────────────────────────────────────────────────


@dataclass
class Step:
    n: int
    title: str
    files: list[str] = field(default_factory=list)
    description: str = ""
    success_criteria: str = ""


@dataclass
class Plan:
    goal: str
    steps: list[Step] = field(default_factory=list)
    final_verification: str = "Run pytest -q"

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        steps = [
            Step(
                n=int(s.get("n", i + 1)),
                title=str(s.get("title", "")),
                files=list(s.get("files", []) or []),
                description=str(s.get("description", "")),
                success_criteria=str(s.get("success_criteria", "")),
            )
            for i, s in enumerate(data.get("steps", []))
        ]
        return cls(
            goal=str(data.get("goal", "")),
            steps=steps,
            final_verification=str(data.get("final_verification", "Run pytest -q")),
        )


# ── Plan parsing ────────────────────────────────────────────────────


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```")
_BARE_JSON_RE = re.compile(r"(\{(?:[^{}]|\{[^{}]*\})*\})", re.DOTALL)


def parse_plan(text: str) -> Optional[Plan]:
    """Extract a Plan from the model's freeform response. Tries JSON in a
    code fence first, then bare JSON anywhere. Returns None if no parseable
    JSON is found OR if the JSON has no `steps` array."""
    if not text:
        return None
    candidates: list[str] = []
    for m in _JSON_FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    for m in _BARE_JSON_RE.finditer(text):
        candidates.append(m.group(1))
    for raw in candidates:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if "steps" not in data or not isinstance(data["steps"], list):
            continue
        if not data["steps"]:
            continue
        return Plan.from_dict(data)
    return None


# ── Planner prompt ──────────────────────────────────────────────────


_PLANNER_SYSTEM = """You are a project planning assistant. Given a high-level
goal, produce a structured JSON plan with concrete atomic steps.

OUTPUT FORMAT — emit ONE JSON object inside a ```json code fence. Schema:

```json
{
  "goal": "<one-line restatement>",
  "steps": [
    {
      "n": 1,
      "title": "<short imperative, ~6 words max>",
      "files": ["path/to/file.py"],
      "description": "<2-3 sentence concrete description: what the step writes, what classes/functions it defines, what shape the output takes>",
      "success_criteria": "<how to verify this step is done — e.g. 'file imports cleanly' or 'pytest tests/test_X.py passes'>"
    }
  ],
  "final_verification": "<single command or short description, e.g. 'Run pytest -q'>"
}
```

RULES:
- 4-12 steps total. Atomic steps — each touches 1-2 files.
- Order: shared types/dataclasses first, then storage/persistence, then
  business logic, then CLI/entrypoint, then tests, then verification.
- Each step's description must be specific enough that another agent
  could execute it WITHOUT re-reading the goal — it carries the contract
  for that step.
- `files` must be the file paths the step actually writes/edits. The
  driver uses this for coverage tracking.
- success_criteria is what THE NEXT STEP can rely on — e.g. "Timer
  class is importable from pomodoro.timer with start/stop/elapsed".

Return ONLY the JSON object inside the ```json fence. No prose before
or after."""


def _build_planner_prompt(goal: str) -> str:
    """Build a chat-formatted prompt for the planner turn. Uses the
    same ChatML format the rest of the system uses."""
    return (
        "<|im_start|>system\n"
        + _PLANNER_SYSTEM
        + "\n<|im_end|>\n<|im_start|>user\n"
        + goal
        + "\n<|im_end|>\n<|im_start|>assistant\n"
    )


# ── Step prompt builder ─────────────────────────────────────────────


def _build_step_extra(plan: Plan, step: Step, prior_summaries: list[str]) -> str:
    """The system_prompt_extra each step's sub-agent gets. Carries:
      - the original goal (so the model never loses sight of intent)
      - the full plan (so the model knows where it is in the sequence)
      - actual deliverable summaries from every prior step
      - the current step's full spec
      - explicit "do ONLY this step" instruction
    """
    plan_lines = [f"  {s.n}. {s.title} ({', '.join(s.files) or '-'})" for s in plan.steps]
    prior_block = ""
    if prior_summaries:
        prior_block = (
            "\n\nPRIOR STEPS COMPLETED (most recent last):\n"
            + "\n".join(prior_summaries)
            + "\n"
        )
    return (
        f"PROJECT GOAL: {plan.goal}\n\n"
        f"FULL PLAN:\n" + "\n".join(plan_lines) + "\n"
        f"{prior_block}\n"
        f"YOUR CURRENT STEP ({step.n}/{len(plan.steps)}): {step.title}\n"
        f"  files: {', '.join(step.files) or '(not specified)'}\n"
        f"  description: {step.description}\n"
        f"  success criteria: {step.success_criteria}\n\n"
        f"IMPORTANT:\n"
        f"  - Do ONLY this step. Don't write code that belongs to a later step.\n"
        f"  - Use the prior-step summaries above as ground truth for what "
        f"already exists. If you need to use a class/function defined by an "
        f"earlier step, the summary tells you its name and shape — IMPORT it, "
        f"don't redefine it.\n"
        f"  - End with a short plain-text confirmation summarizing what you "
        f"actually wrote (file paths + key symbol names) so the next step "
        f"has accurate state."
    )


def _extract_public_api(workspace: Optional[Path], file_path: str) -> str:
    """Read a Python file and extract its top-level public API: classes
    (with __init__ signature) and functions (with their signatures).
    Returns a one-line string suitable for embedding in step summaries.

    The whole point: the next step's sub-agent needs to KNOW the
    signatures of what prior steps wrote, otherwise it makes up
    different contracts (Timer(duration_sec=10) vs Timer(clock=...)).
    """
    if workspace is None:
        return ""
    try:
        target = (workspace / file_path).resolve()
        # Stay within workspace
        if workspace.resolve() not in target.parents and target != workspace.resolve():
            return ""
        if not target.exists() or target.suffix != ".py":
            return ""
        source = target.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    try:
        import ast
        tree = ast.parse(source)
    except SyntaxError:
        return "(syntax error)"

    parts: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            # Class signature: __init__'s args plus public method names
            # with their signatures. Private methods (starting _, except
            # __init__) skipped.
            init_sig = "()"
            methods: list[str] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == "__init__":
                        init_sig = "(" + _ast_args(child.args) + ")"
                    elif not child.name.startswith("_"):
                        methods.append(f"def {child.name}({_ast_args(child.args)})")
            class_line = f"class {node.name}{init_sig}"
            if methods:
                class_line += " {" + "; ".join(methods) + "}"
            parts.append(class_line)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue  # skip private
            parts.append(f"def {node.name}({_ast_args(node.args)})")
    if not parts:
        return ""
    return "; ".join(parts)


def _ast_args(args) -> str:
    """Render an ast.arguments node as a readable signature string.
    Skips `self` for class methods. Drops type annotations (the next
    step needs the SHAPE not the types) but emits `=...` for arguments
    that have a default, so the step knows which are optional."""
    out: list[str] = []
    n_defaults = len(args.defaults)
    n_args = len(args.args)
    first_default_idx = n_args - n_defaults
    for i, arg in enumerate(args.args):
        if i == 0 and arg.arg == "self":
            continue
        if i >= first_default_idx:
            out.append(f"{arg.arg}=...")
        else:
            out.append(arg.arg)
    if args.vararg:
        out.append(f"*{args.vararg.arg}")
    for arg in args.kwonlyargs:
        out.append(f"{arg.arg}=...")
    if args.kwarg:
        out.append(f"**{args.kwarg.arg}")
    return ", ".join(out)


def _summarize_step_result(step: Step, result: AgentResult, workspace: Optional[Path] = None) -> str:
    """Distill a step's AgentResult into a multi-line summary the next
    step can rely on. Format:

        Step N (title): files_written
          path/file.py: class Foo(arg=...); def bar(x, y=...)

    The API extraction is the key fix vs naive file-path-only summaries:
    later steps need the SIGNATURES of what's been written, not just
    the paths, otherwise they make up different contracts.
    """
    files_written: list[str] = []
    for call, res in zip(result.tool_calls, result.tool_results):
        if not getattr(res, "success", False):
            continue
        cname = getattr(call, "name", "")
        if cname not in ("write_file", "edit_file", "insert_at_line"):
            continue
        cargs = getattr(call, "arguments", {}) or {}
        if not isinstance(cargs, dict):
            continue
        path = cargs.get("path") or "?"
        if path not in files_written:
            files_written.append(path)

    if not files_written:
        files_part = "(no files written)"
    else:
        files_part = ", ".join(files_written)

    api_lines: list[str] = []
    for path in files_written:
        api = _extract_public_api(workspace, path)
        if api:
            api_lines.append(f"      {path}: {api}")

    answer_excerpt = (result.final_answer or "").strip().replace("\n", " ")[:240]
    summary = f"  Step {step.n} ({step.title}): {files_part}."
    if api_lines:
        summary += "\n" + "\n".join(api_lines)
    summary += f"\n      confirmation: {answer_excerpt}"
    return summary


# ── HandheldDriver ──────────────────────────────────────────────────


@dataclass
class HandheldResult:
    plan: Optional[Plan]
    final_answer: str
    iterations: int
    stop_reason: str
    wall_time: float
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    step_summaries: list[str] = field(default_factory=list)
    verification_summary: str = ""


class HandheldDriver:
    """Plan-then-drive pattern for large-project scaffolding on small
    models. See module docstring for design rationale and contrast with
    architect_agent."""

    def __init__(
        self,
        model: Any,
        registry: ToolRegistry,
        system_prompt_extra: str = "",
        workspace_root: Optional[Path] = None,
        memory: Optional[AgentMemory] = None,
        auto_verify_python: bool = True,
        max_iterations_per_step: int = 8,
        max_wall_time: float = 1800.0,  # 30 min default — large projects
        max_tokens_per_turn: int = 1024,
        temperature: Optional[float] = 0.1,
        repeat_penalty: Optional[float] = 1.15,
        confirm_risky: Optional[Callable[[ToolCall], bool]] = None,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
        planner_max_tokens: int = 1500,
    ) -> None:
        self.model = model
        self.registry = registry
        self.system_prompt_extra = system_prompt_extra
        self.workspace_root = Path(workspace_root) if workspace_root is not None else None
        self.memory = memory
        self.auto_verify_python = auto_verify_python
        self.max_iterations_per_step = max(1, int(max_iterations_per_step))
        self.max_wall_time = float(max_wall_time)
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.confirm_risky = confirm_risky
        self._emit = on_event or (lambda _e: None)
        self.planner_max_tokens = int(planner_max_tokens)

    def run(self, goal: str) -> HandheldResult:
        start = time.monotonic()

        # PHASE 1: PLAN
        plan_prompt = _build_planner_prompt(goal)
        try:
            plan_text = self.model.generate(
                plan_prompt,
                max_tokens=self.planner_max_tokens,
                temperature=self.temperature if self.temperature is not None else 0.1,
                repeat_penalty=self.repeat_penalty if self.repeat_penalty is not None else 1.15,
                stop=["<|im_end|>", "<|im_start|>"],
            )
        except Exception as exc:
            logger.exception("planner call failed")
            return HandheldResult(
                plan=None, final_answer=f"[planner error: {exc}]",
                iterations=0, stop_reason="planner_error",
                wall_time=time.monotonic() - start,
            )

        plan = parse_plan(plan_text or "")
        if plan is None:
            return HandheldResult(
                plan=None,
                final_answer=("[planner produced no parseable JSON plan]\n"
                              + (plan_text or "")[:600]),
                iterations=0, stop_reason="plan_unparseable",
                wall_time=time.monotonic() - start,
            )

        plan.goal = plan.goal or goal  # fill default if planner omitted
        self._emit(AgentEvent(
            "iteration", 0,
            f"handheld planned {len(plan.steps)} step(s): "
            + " | ".join(s.title[:40] for s in plan.steps)
        ))

        # PHASE 2: PER-STEP EXECUTION
        all_calls: list[ToolCall] = []
        all_results: list[ToolResult] = []
        transcript_merged: list[dict] = [{"role": "user", "content": goal}]
        prior_summaries: list[str] = []
        last_stop_reason = "answered"

        for step in plan.steps:
            if time.monotonic() - start > self.max_wall_time:
                last_stop_reason = "wall_time"
                break

            extra = (
                self.system_prompt_extra + "\n\n" + _build_step_extra(plan, step, prior_summaries)
                if self.system_prompt_extra
                else _build_step_extra(plan, step, prior_summaries)
            )
            remaining = self.max_wall_time - (time.monotonic() - start)
            per_step_wall = max(60.0, remaining / max(1, len(plan.steps) - step.n + 1))

            worker = Agent(
                model=self.model,
                registry=self.registry,
                system_prompt_extra=extra,
                workspace_root=self.workspace_root,
                memory=self.memory,
                auto_verify_python=self.auto_verify_python,
                # Sub-agents must produce mutating action — their step is a
                # write/edit, not just a read.
                require_mutating_action=True,
                # The driver runs its own end-pass verification, so per-step
                # sub-agents shouldn't try to sweep goal tokens themselves.
                enable_goal_token_sweep=False,
                max_iterations=self.max_iterations_per_step,
                max_wall_time=per_step_wall,
                max_tokens_per_turn=self.max_tokens_per_turn,
                temperature=self.temperature,
                repeat_penalty=self.repeat_penalty,
                confirm_risky=self.confirm_risky,
                on_event=self._emit,
            )
            self._emit(AgentEvent(
                "iteration", step.n,
                f"step {step.n}/{len(plan.steps)}: {step.title}"
            ))
            try:
                step_result = worker.run(step.description)
            except Exception:
                logger.exception("handheld step %d raised", step.n)
                last_stop_reason = "exception"
                break

            all_calls.extend(step_result.tool_calls)
            all_results.extend(step_result.tool_results)
            summary = _summarize_step_result(step, step_result, workspace=self.workspace_root)
            prior_summaries.append(summary)
            transcript_merged.append({
                "role": "assistant",
                "content": f"[step {step.n}/{len(plan.steps)}: {step.title}] {summary}",
            })
            last_stop_reason = step_result.stop_reason

        # PHASE 3: FINAL VERIFICATION (optional pytest run)
        verification_summary = ""
        if (
            self.workspace_root
            and self.workspace_root.exists()
            and time.monotonic() - start < self.max_wall_time
        ):
            verification_summary = self._run_final_verification(plan)

        total_wall = time.monotonic() - start
        final_msg = f"Handheld driver completed {len(prior_summaries)}/{len(plan.steps)} steps."
        if verification_summary:
            final_msg += f" {verification_summary}"
        return HandheldResult(
            plan=plan,
            final_answer=final_msg,
            iterations=len(prior_summaries),
            stop_reason=last_stop_reason,
            wall_time=total_wall,
            tool_calls=all_calls,
            tool_results=all_results,
            transcript=transcript_merged,
            step_summaries=prior_summaries,
            verification_summary=verification_summary,
        )

    def _run_final_verification(self, plan: Plan) -> str:
        """Run the plan's final_verification. Same posture as
        ArchitectAgent._final_test_pass — uses the registry's run_tests
        tool, surfaces a short pass/fail summary."""
        if not self.workspace_root:
            return ""
        try:
            run_tests_tool = self.registry.get("run_tests")
            if run_tests_tool is None:
                return ""
            result = run_tests_tool.function(path=".", runner="pytest")
        except Exception as exc:
            return f"(verification raised: {exc})"
        if isinstance(result, dict):
            passed = result.get("passed")
            exit_code = result.get("exit_code")
            stdout = str(result.get("stdout", ""))[:240]
            if passed is True or exit_code == 0:
                return "Tests pass."
            return f"Tests FAIL (exit={exit_code}): {stdout}"
        return f"Verification ran: {str(result)[:200]}"
