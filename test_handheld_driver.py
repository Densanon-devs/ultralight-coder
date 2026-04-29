"""
Tests for engine/handheld_driver.py — plan + per-step driver with
prior-state injection.

Schema, planner parsing, and step-prompt assembly are unit-testable
without loading a model. Live model behavior is tested via a separate
hand-driven walkthrough (not in this file).
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE = ROOT.parent / "densanon-core"
if _CORE.exists() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from engine.handheld_driver import (
    Plan, Step, parse_plan, _build_step_extra,
    _summarize_step_result, HandheldDriver, HandheldResult,
)
from engine.agent_tools import ToolRegistry, ToolSchema


# ── Plan schema ───────────────────────────────────────────────────


def test_plan_from_dict_basic():
    p = Plan.from_dict({
        "goal": "build a thing",
        "steps": [
            {"n": 1, "title": "step one", "files": ["a.py"], "description": "do A", "success_criteria": "passes"},
            {"n": 2, "title": "step two", "files": ["b.py"], "description": "do B", "success_criteria": "passes"},
        ],
        "final_verification": "Run pytest -q",
    })
    assert p.goal == "build a thing"
    assert len(p.steps) == 2
    assert p.steps[0].title == "step one"
    assert p.steps[0].files == ["a.py"]
    assert p.final_verification == "Run pytest -q"


def test_plan_from_dict_fills_n_when_missing():
    p = Plan.from_dict({
        "goal": "x",
        "steps": [{"title": "first"}, {"title": "second"}],
    })
    assert p.steps[0].n == 1
    assert p.steps[1].n == 2


# ── Plan parsing from model output ────────────────────────────────


def test_parse_plan_extracts_json_from_fence():
    text = """Here's the plan:

```json
{
  "goal": "build a tiny CLI",
  "steps": [
    {"n": 1, "title": "create main", "files": ["main.py"], "description": "main module"}
  ],
  "final_verification": "Run pytest"
}
```

Let me know if this looks right."""
    p = parse_plan(text)
    assert p is not None
    assert p.goal == "build a tiny CLI"
    assert len(p.steps) == 1


def test_parse_plan_extracts_bare_json():
    text = """{"goal": "x", "steps": [{"n": 1, "title": "one"}], "final_verification": "Y"}"""
    p = parse_plan(text)
    assert p is not None
    assert len(p.steps) == 1


def test_parse_plan_returns_none_on_no_json():
    p = parse_plan("Sorry, I can't generate a plan for this.")
    assert p is None


def test_parse_plan_returns_none_on_empty_steps():
    text = '{"goal": "x", "steps": []}'
    p = parse_plan(text)
    assert p is None


def test_parse_plan_returns_none_on_invalid_json():
    text = "{ broken json"
    p = parse_plan(text)
    assert p is None


def test_parse_plan_handles_plain_json_no_fence():
    text = '{"goal": "x", "steps": [{"n": 1, "title": "step a"}], "final_verification": "Y"}'
    p = parse_plan(text)
    assert p is not None
    assert p.steps[0].title == "step a"


# ── Step extra prompt ─────────────────────────────────────────────


def test_step_extra_carries_full_plan():
    plan = Plan(goal="build x", steps=[
        Step(1, "make a", ["a.py"], "step a desc", "a passes"),
        Step(2, "make b", ["b.py"], "step b desc", "b passes"),
        Step(3, "test it", ["test.py"], "tests", "all green"),
    ])
    extra = _build_step_extra(plan, plan.steps[1], prior_summaries=[
        "  Step 1 (make a): a.py. Confirmation: wrote ClassA"
    ])
    assert "PROJECT GOAL" in extra
    assert "FULL PLAN" in extra
    assert "1. make a" in extra
    assert "2. make b" in extra
    assert "3. test it" in extra
    assert "PRIOR STEPS COMPLETED" in extra
    assert "Step 1 (make a)" in extra
    assert "step b desc" in extra
    assert "Do ONLY this step" in extra


def test_step_extra_skips_prior_block_when_empty():
    plan = Plan(goal="x", steps=[Step(1, "first", ["a.py"], "do A", "ok")])
    extra = _build_step_extra(plan, plan.steps[0], prior_summaries=[])
    assert "PRIOR STEPS COMPLETED" not in extra


def test_step_extra_says_import_dont_redefine():
    """The prompt must explicitly tell the sub-agent to import prior-step
    classes — this is the key fix for architect's cross-file duplication."""
    plan = Plan(goal="x", steps=[Step(1, "a", ["a.py"]), Step(2, "b", ["b.py"])])
    extra = _build_step_extra(plan, plan.steps[1], prior_summaries=["  Step 1 (a): a.py"])
    assert "IMPORT it, don't redefine" in extra


# ── Step result summarization ─────────────────────────────────────


def test_summarize_extracts_files_written():
    step = Step(1, "make storage", ["storage.py"], "...", "...")
    result = SimpleNamespace(
        final_answer="Storage module created with load_bookmarks and save_bookmarks.",
        tool_calls=[
            SimpleNamespace(name="write_file", arguments={"path": "storage.py", "content": ["..."]}),
        ],
        tool_results=[SimpleNamespace(success=True, content="Wrote 100 chars")],
    )
    summary = _summarize_step_result(step, result)
    assert "storage.py" in summary
    assert "Step 1" in summary
    assert "make storage" in summary


def test_summarize_includes_public_api_signatures():
    """The bottleneck observed in the 2026-04-26 pomodoro live test:
    step summaries said 'wrote timer.py' but didn't include Timer's
    __init__ signature, so the next step's sub-agent invented a
    different signature. Fix: AST-extract public class/function
    signatures and inject them into the summary."""
    workspace = Path(tempfile.mkdtemp())
    (workspace / "timer.py").write_text(
        "from time import monotonic\n"
        "class Timer:\n"
        "    def __init__(self, clock=monotonic):\n"
        "        self.clock = clock\n"
        "    def start(self, duration_sec):\n"
        "        pass\n"
        "    def is_done(self):\n"
        "        return False\n"
    )
    step = Step(1, "make timer", ["timer.py"], "...", "...")
    result = SimpleNamespace(
        final_answer="Timer class added.",
        tool_calls=[
            SimpleNamespace(name="write_file", arguments={"path": "timer.py", "content": ["..."]}),
        ],
        tool_results=[SimpleNamespace(success=True, content="Wrote 100 chars")],
    )
    summary = _summarize_step_result(step, result, workspace=workspace)
    # Class signature should appear so subsequent steps know the contract
    assert "class Timer" in summary
    assert "clock=" in summary  # the optional kwarg
    # Public methods should appear too
    assert "def start" in summary or "start" in summary


def test_summarize_skips_private_functions():
    workspace = Path(tempfile.mkdtemp())
    (workspace / "x.py").write_text(
        "def public_one(a):\n    pass\n"
        "def _private_helper(b):\n    pass\n"
    )
    step = Step(1, "x", ["x.py"], "...", "...")
    result = SimpleNamespace(
        final_answer="ok",
        tool_calls=[SimpleNamespace(name="write_file", arguments={"path": "x.py"})],
        tool_results=[SimpleNamespace(success=True, content="ok")],
    )
    summary = _summarize_step_result(step, result, workspace=workspace)
    assert "public_one" in summary
    assert "_private_helper" not in summary


def test_summarize_handles_missing_workspace():
    """When workspace is None or file doesn't exist, fall back to
    file-path-only summary (no API extraction)."""
    step = Step(1, "x", ["x.py"], "...", "...")
    result = SimpleNamespace(
        final_answer="ok",
        tool_calls=[SimpleNamespace(name="write_file", arguments={"path": "x.py"})],
        tool_results=[SimpleNamespace(success=True, content="ok")],
    )
    summary = _summarize_step_result(step, result, workspace=None)
    assert "x.py" in summary  # path still there, just no signatures


def test_summarize_handles_no_files_written():
    step = Step(1, "explore", [], "...", "...")
    result = SimpleNamespace(
        final_answer="No write needed.",
        tool_calls=[SimpleNamespace(name="read_file", arguments={"path": "x.py"})],
        tool_results=[SimpleNamespace(success=True, content="(file)")],
    )
    summary = _summarize_step_result(step, result)
    assert "no files written" in summary


def test_summarize_skips_failed_writes():
    step = Step(1, "x", ["x.py"], "...", "...")
    result = SimpleNamespace(
        final_answer="...",
        tool_calls=[SimpleNamespace(name="write_file", arguments={"path": "x.py"})],
        tool_results=[SimpleNamespace(success=False, content="", error="parse error")],
    )
    summary = _summarize_step_result(step, result)
    assert "no files written" in summary


# ── HandheldDriver shape ──────────────────────────────────────────


def test_driver_planner_error_returns_handheld_result():
    """If the model raises during planning, return a sensible HandheldResult
    instead of crashing."""
    class _BadModel:
        def generate(self, *a, **k):
            raise RuntimeError("model exploded")
    driver = HandheldDriver(model=_BadModel(), registry=ToolRegistry())
    result = driver.run("build something")
    assert isinstance(result, HandheldResult)
    assert result.plan is None
    assert result.stop_reason == "planner_error"


def test_driver_unparseable_plan_returns_handheld_result():
    """If the model returns prose with no JSON, return a sensible result."""
    class _ProseModel:
        def generate(self, *a, **k):
            return "I would need more details to make a plan."
    driver = HandheldDriver(model=_ProseModel(), registry=ToolRegistry())
    result = driver.run("build something")
    assert result.plan is None
    assert result.stop_reason == "plan_unparseable"


def test_driver_runs_steps_when_plan_parses(tmp_path=None):
    """End-to-end with a stub model that returns a plan, then no-ops on
    each step turn (sub-agent will fail to mutate; we just check the
    driver wires the steps through)."""
    plan_json = json.dumps({
        "goal": "build it",
        "steps": [
            {"n": 1, "title": "make a", "files": ["a.py"], "description": "make a"},
            {"n": 2, "title": "make b", "files": ["b.py"], "description": "make b"},
        ],
        "final_verification": "Run pytest -q",
    })

    plan_text = "```json\n" + plan_json + "\n```"

    calls_seen = {"n": 0}
    class _StubModel:
        def generate(self, *a, **k):
            calls_seen["n"] += 1
            if calls_seen["n"] == 1:
                # Planner turn
                return plan_text
            # Sub-agent turns — return empty (sub-agent will fail to
            # produce mutating action; that's fine for this test).
            return ""

    workspace = Path(tempfile.mkdtemp())
    driver = HandheldDriver(
        model=_StubModel(),
        registry=ToolRegistry(),
        workspace_root=workspace,
        max_iterations_per_step=1,
        max_wall_time=30.0,
    )
    result = driver.run("build it")
    assert result.plan is not None
    assert len(result.plan.steps) == 2
    # Driver should have attempted both steps (even if they didn't produce
    # work due to the stub model)
    assert "Handheld driver completed" in result.final_answer


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
