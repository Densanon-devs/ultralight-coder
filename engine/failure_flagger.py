"""
Failure flagger — analyzes a finished AgentResult and emits structured
FailureRecord entries for known 14B failure patterns observed during
Phase 14 benchmarking.

Used together with `engine/yaml_augmentor_builder.py` to close the loop:
  agent runs -> failure detected -> YAML augmentor synthesized -> next
  similar query gets the right in-context example.

The flagger is INTENTIONALLY NARROW. It only tags patterns that are:
  (a) observed empirically in bench logs (today: bench_qwen3_*, bench_gpt55_*)
  (b) cleanly recoverable with a YAML demonstration
  (c) NOT already covered by an existing _auto_apply_* heuristic on Agent

Categories shipped today (each maps to a YAML augmentor template):
  json_quote_leak       — Python single-quoted strings inside JSON content
  fstring_nested_quote  — Python 3.10/3.11 nested-quote f-string trap
  missing_import        — write_file/edit_file produced unreferenced names
  premature_bail        — model said "done" while stderr held a Traceback
  stuck_repeat_loop     — agent.py emitted stuck_repeat ≥1 time

Adding a new category: append to FAILURE_CATEGORIES + a detector function.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    from engine.agent_tools import ToolCall, ToolResult  # full-package import
except ImportError:  # pragma: no cover — running as `python engine/failure_flagger.py`
    from agent_tools import ToolCall, ToolResult


# ── Failure record ──────────────────────────────────────────────────


@dataclass
class FailureRecord:
    category: str          # one of FAILURE_CATEGORIES
    iteration: int         # 1-indexed iteration where the failure surfaced
    tool_name: str         # the tool that produced the failure (or "" for prose)
    error_excerpt: str     # one-line summary of the error
    triggering_args: dict[str, Any] = field(default_factory=dict)
    file_path: str | None = None
    suggested_fix: str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "iteration": self.iteration,
            "tool_name": self.tool_name,
            "error_excerpt": self.error_excerpt[:500],
            "triggering_args": _abridge_args(self.triggering_args),
            "file_path": self.file_path,
            "suggested_fix": self.suggested_fix,
        }


FAILURE_CATEGORIES = (
    "json_quote_leak",
    "fstring_nested_quote",
    "missing_import",
    "premature_bail",
    "stuck_repeat_loop",
    # Added 2026-04-25 (B4):
    "stale_anchor_edit",      # edit_file `old_string not found` 2+ times same file
    "cwd_assumption",         # run_bash command assumed wrong cwd / path
    "test_import_path",       # test file can't import its target module
    # Added 2026-04-26 (handheld walkthrough):
    "incomplete_deliverable", # final answer with N of M numbered requirements skipped
    "narration_without_action",  # "I will now emit..." prose ends in final answer
    # Added 2026-04-26 (re-walkthrough — model fixated on cli.py, ignored storage.py):
    "unaddressed_file_in_goal",
)


_ARG_PREVIEW = 200


def _abridge_args(args: dict) -> dict:
    """Trim long string fields so the FailureRecord stays human-readable."""
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > _ARG_PREVIEW:
            out[k] = v[:_ARG_PREVIEW] + f" ...<+{len(v) - _ARG_PREVIEW} chars>"
        elif isinstance(v, list):
            joined = sum(len(s) for s in v if isinstance(s, str))
            if joined > _ARG_PREVIEW:
                out[k] = f"<list[{len(v)}], {joined} chars total>"
            else:
                out[k] = v
        else:
            out[k] = v
    return out


# ── Detectors ───────────────────────────────────────────────────────


def _detect_json_quote_leak(call: ToolCall, result: ToolResult) -> bool:
    """The 14B/32B Hermes JSON ceiling: Python `'foo'` inside JSON content
    array emits `Expecting ',' delimiter` or `Unterminated string` errors."""
    if result.success:
        return False
    err = (result.error or "").lower()
    return (
        "expecting ',' delimiter" in err
        or "unterminated string starting at" in err
        or "bare json tool call failed to parse" in err
    )


def _detect_fstring_trap(call: ToolCall, result: ToolResult) -> bool:
    """f-string nested-quote SyntaxError on Python 3.10/3.11."""
    if result.success:
        return False
    # str() guard: ToolResult.content can be a dict for tools like run_bash
    # ({'stdout': ..., 'stderr': ...}) — concatenating directly with + " "
    # raises TypeError. str-cast first so all detectors operate on strings.
    msg = (str(result.content or "") + " " + str(result.error or "")).lower()
    return (
        ("f-string" in msg or "f'" in msg)
        and ("expecting '}'" in msg or "syntaxerror" in msg)
    )


def _detect_missing_import(call: ToolCall, result: ToolResult) -> bool:
    """auto_verify reported `references undefined names: <name>` after a write."""
    msg = str(result.content or "") + " " + str(result.error or "")
    return "references undefined names" in msg


def _detect_premature_bail(
    final_answer: str, last_tool_result: ToolResult | None
) -> bool:
    """Two shapes — the harness should catch both:
      A. Model claimed 'task complete' / 'all tests pass' / etc. but
         the LAST tool result carried Traceback / SyntaxError /
         AssertionError. (Original detector.)
      B. Model gave up with 'I cannot complete this task' / 'I apologize' /
         'cannot complete' prose. The 2026-04-26 re-walkthrough's Goal 2
         hit this: model bailed at iter 4 despite a clear pre_finish_check
         directive telling it exactly what edit to emit. Last result was
         a synthetic pre_finish_check, not a Traceback — so shape A
         missed it. Shape B fires regardless of last_tool_result."""
    if not final_answer:
        return False
    fa_lower = final_answer.lower()

    # Shape B — give-up prose. Fires unconditionally on these markers.
    give_up_signals = (
        "i cannot complete",
        "i can't complete",
        "i cannot finish",
        "i can't finish",
        "i apologize",
        "unable to complete",
        "could not complete",
        "i'm unable to",
        "cannot proceed",
    )
    if any(s in fa_lower for s in give_up_signals):
        return True

    # Shape A — original detector. Requires Traceback in last result.
    bail_signals = ("task is complete", "task is done", "all tests pass",
                    "successfully built", "implementation complete")
    if not any(s in fa_lower for s in bail_signals):
        return False
    if last_tool_result is None:
        return False
    last_text = (
        (str(last_tool_result.content) if last_tool_result.content else "")
        + " "
        + (last_tool_result.error or "")
    )
    return (
        "Traceback" in last_text
        or "SyntaxError" in last_text
        or "AssertionError" in last_text
    )


def _detect_stuck_repeat(result: ToolResult) -> bool:
    """The agent loop's stuck_repeat watchdog fired."""
    err = (result.error or "").lower()
    return "stuck_repeat" in err or "stuck-repeat" in err


def _detect_stale_anchor(call: ToolCall, result: ToolResult) -> bool:
    """edit_file failed because `old_string` no longer matches the file —
    common after the model has already mutated the file but kept its old
    mental model of the contents."""
    if getattr(result, "success", False):
        return False
    err = (getattr(result, "error", "") or "").lower()
    cname = getattr(call, "name", "") if call else ""
    return cname == "edit_file" and "old_string not found" in err


def _detect_cwd_assumption(call: ToolCall, result: ToolResult) -> bool:
    """run_bash failed because the command assumed a path/cwd that
    doesn't exist (e.g. `python tests/test_x.py` when tests/ isn't there)."""
    cname = getattr(call, "name", "") if call else ""
    if cname != "run_bash":
        return False
    content = str(getattr(result, "content", "") or "")
    err = str(getattr(result, "error", "") or "")
    blob = (content + " " + err).lower()
    # Common Python signals when cwd is wrong
    if "no such file or directory" in blob:
        return True
    if "[errno 2]" in blob:
        return True
    if "filenotfounderror" in blob:
        return True
    if "can't open file" in blob:  # python -m / file path failures
        return True
    return False


# Mutating tool names — used by deliverable-counting detectors below.
_MUTATING_TOOLS = frozenset({
    "write_file", "edit_file", "insert_at_line", "edit_html", "apply_patch",
    "rename_symbol", "run_bash", "run_tests",
})

# Numbered-requirement pattern (1.  / 2)  / 3. / etc). Matches either
# at start of line OR after whitespace within a line — goals from CLI
# invocations often have all requirements on one line because the
# shell argument is one big string.
_NUMBERED_REQ_RE = re.compile(r"(?:^|\s)(\d+[.)])\s", re.MULTILINE)

# Imperative mutation verbs in goal text — distinguishes "do X" goals from
# "explain X" / "show X" / "list X" goals where 0 mutations is correct.
_IMPERATIVE_VERBS = (
    "add", "create", "build", "implement", "write", "scaffold",
    "fix", "modify", "change", "update", "edit", "remove", "delete",
    "rename", "refactor", "extend",
)

# Future-tense intent markers that mean the model TALKED about acting
# but didn't actually act — caught by narration_without_action.
_NARRATION_PATTERNS = (
    "i will now",
    "i'll now",
    "i'll proceed to",
    "i will proceed to",
    "next, i will",
    "next i will",
    "i plan to",
    "i'm going to",
    "i am going to",
    "i'll emit",
    "i will emit",
    "i'll add",
    "i will add",
    "let me now",
    "let me proceed",
    "let me start by",
    "i'll begin by",
)


def _count_mutating_calls(calls: list, results: list) -> int:
    """How many real (non-synthetic) mutating tool calls actually executed
    successfully in this run? Used by incomplete_deliverable."""
    if not calls or not results:
        return 0
    n = 0
    call_cursor = 0
    for res in results:
        rname = getattr(res, "name", "") or ""
        if rname in _SYNTHETIC_RESULT_NAMES:
            continue
        if call_cursor >= len(calls):
            break
        call = calls[call_cursor]
        call_cursor += 1
        cname = getattr(call, "name", "") or ""
        if cname in _MUTATING_TOOLS and getattr(res, "success", False):
            n += 1
    return n


def _detect_incomplete_deliverable(goal: str, result: object) -> bool:
    """Goal lists N numbered requirements but the run made significantly
    fewer mutating tool calls. Caught the Goal 1 retry of the 2026-04-26
    walkthrough where 5 numbered requirements got 3 mutating calls and
    test_bookmarks.py + run_tests were silently skipped.

    Heuristic: requirements_count >= 3 AND mutating_calls < requirements_count // 2."""
    if not goal:
        return False
    requirements = len(_NUMBERED_REQ_RE.findall(goal))
    if requirements < 3:
        return False
    g_lower = goal.lower()
    if not any(v in g_lower for v in _IMPERATIVE_VERBS):
        return False
    calls = list(getattr(result, "tool_calls", []) or [])
    results = list(getattr(result, "tool_results", []) or [])
    mutating = _count_mutating_calls(calls, results)
    # Coverage threshold: fire when MORE THAN ONE numbered requirement
    # was skipped. Single-requirement misses happen on legit edge cases
    # (e.g. one numbered item is a meta-instruction); skipping 2+ is a
    # clear pattern. With N reqs: fires when mutating < N - 1.
    return mutating < (requirements - 1)


# Filename pattern for goal scanning — matches names like
# "storage.py", "cli.py", "src/handlers.py", "tests/test_x.py".
_GOAL_FILENAME_RE = re.compile(
    r"\b([\w/\\.-]+\.(?:py|js|ts|jsx|tsx|go|rs|java|cs|rb|sh|html|yaml|yml|json|toml))\b",
    re.IGNORECASE,
)


def _detect_unaddressed_file_in_goal(goal: str, result: object) -> bool:
    """Goal explicitly names specific files but the agent never wrote
    or edited some of them. The 2026-04-26 re-walkthrough's Goal 1.5
    hit this: 'Fix three bugs in storage.py and cli.py' — model edited
    cli.py 5 times and never touched storage.py.

    Distinct from incomplete_deliverable (which counts numbered items):
    fires even when the call count is high but the wrong file got the
    attention. The signal is COVERAGE OF NAMED FILES, not deliverable count.
    """
    if not goal:
        return False
    g_lower = goal.lower()
    # Must be an action goal, not a read/explain goal
    if not any(v in g_lower for v in _IMPERATIVE_VERBS):
        return False

    # Files the goal names — only count distinct ones with directory
    # prefixes stripped. Goal text often says "storage.py" and "cli.py"
    # without paths.
    mentioned_raw = {m.group(1).lower() for m in _GOAL_FILENAME_RE.finditer(goal)}
    if len(mentioned_raw) < 2:
        return False  # single-file goals don't have coverage to check

    # Ignore non-actionable mention contexts: bookmarks.json (a runtime
    # artifact), conftest.py (rare), filenames in negative phrasing
    # ("DO NOT modify A or B"). Strip the obvious ones.
    # Match the FULL clause after the verb (up to ~80 chars or end of
    # sentence), then extract every filename pattern from that span —
    # handles "do not modify A or B" / "do not edit X, Y, or Z".
    # Match the whole clause after the verb. We allow `.` because
    # filenames contain periods (`storage.py`); we stop on `!`, `?`,
    # newline, or two periods in a row (sentence boundary signal).
    # Length cap prevents a runaway match swallowing the rest of the
    # goal.
    do_not_clause_re = re.compile(
        r"do not (?:modify|edit|touch|change|alter)\s+([^!?\n]{0,160})",
        re.IGNORECASE,
    )
    excluded: set[str] = set()
    for m in do_not_clause_re.finditer(goal):
        clause = m.group(1)
        for fm in _GOAL_FILENAME_RE.finditer(clause):
            excluded.add(fm.group(1).lower())
    excluded |= {"bookmarks.json"}

    mentioned = {f for f in mentioned_raw if f not in excluded}
    if len(mentioned) < 2:
        return False

    calls = list(getattr(result, "tool_calls", []) or [])
    touched: set[str] = set()
    for call in calls:
        cname = getattr(call, "name", "") or ""
        if cname not in _MUTATING_TOOLS:
            continue
        cargs = getattr(call, "arguments", {}) or {}
        if not isinstance(cargs, dict):
            continue
        path = cargs.get("path") or ""
        if not path:
            continue
        # Match by basename — the goal usually says "cli.py" but the
        # mutating call's path could be "src/cli.py" or "./cli.py".
        base = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
        touched.add(base)

    unaddressed = mentioned - touched
    return bool(unaddressed)


def _detect_narration_without_action(goal: str, result: object) -> bool:
    """Final answer contains future-tense intent prose AND no mutating
    tool was called this run AND the goal asks for action. The 14B
    sometimes emits 'I will now emit...' as final-answer prose without
    actually emitting any tool call — Goal 2 of the 2026-04-26 walkthrough.

    Distinct from premature_bail (which checks stderr Traceback)."""
    if not goal:
        return False
    g_lower = goal.lower()
    if not any(v in g_lower for v in _IMPERATIVE_VERBS):
        return False
    final_answer = str(getattr(result, "final_answer", "") or "").lower()
    if not final_answer:
        return False
    if not any(p in final_answer for p in _NARRATION_PATTERNS):
        return False
    calls = list(getattr(result, "tool_calls", []) or [])
    results = list(getattr(result, "tool_results", []) or [])
    mutating = _count_mutating_calls(calls, results)
    return mutating == 0


def _detect_test_import_path(call: ToolCall, result: ToolResult) -> bool:
    """A test file failed because it can't import its target module. Usually
    a PYTHONPATH/cwd issue — pytest needs `__init__.py` or `conftest.py`
    for sibling-module imports to work, or the test must be run from the
    project root."""
    cname = getattr(call, "name", "") if call else ""
    if cname not in ("run_tests", "run_bash"):
        return False
    content = str(getattr(result, "content", "") or "")
    err = str(getattr(result, "error", "") or "")
    blob = content + " " + err
    has_modulenotfound = "ModuleNotFoundError" in blob or "ImportError" in blob
    # Heuristic: if it's a test runner AND the missing module looks like a
    # local module (lowercase, no dots, common test-target name), flag it.
    if has_modulenotfound and ("test_" in blob.lower() or "tests/" in blob):
        return True
    return False


# ── Suggested-fix templates ─────────────────────────────────────────


_FIX_TEMPLATES = {
    "json_quote_leak": (
        "Use the array form for `content` in write_file: every line is its "
        "own element with double-quoted string. Inner Python strings can stay "
        "single-quoted naturally with no JSON escape contortions."
    ),
    "fstring_nested_quote": (
        "Switch the outer f-string to double quotes and keep inner literals "
        "single-quoted (or extract the value before the f-string). Pattern: "
        "`f'{x if y else \\'X\\'}'` → `f\"{x if y else 'X'}\"`."
    ),
    "missing_import": (
        "Prepend the missing import to the file using "
        "edit_file with empty old_string. Example: "
        "`{\"path\": \"X.py\", \"old_string\": \"\", \"new_string\": "
        "\"from Y import Z\\n\"}`."
    ),
    "premature_bail": (
        "Re-read the failing file or re-run the failing command before "
        "claiming completion. A final answer is only correct when the most "
        "recent tool result was clean — no Traceback, no SyntaxError, no "
        "AssertionError."
    ),
    "stuck_repeat_loop": (
        "When stuck_repeat fires, switch tools or arguments. Re-read the "
        "file with read_file to refresh state, then attempt a different "
        "anchor or smaller change."
    ),
    "stale_anchor_edit": (
        "edit_file's `old_string` doesn't match because the file changed "
        "since you last read it. Re-read the file with read_file to see "
        "its CURRENT state, then build old_string from the actual current "
        "content (a SHORT unique substring works best)."
    ),
    "cwd_assumption": (
        "The shell command failed because of a path/cwd assumption. Use "
        "list_dir or glob to confirm the file exists at the expected path "
        "BEFORE running the command. Prefer absolute or workspace-rooted "
        "paths over relative ones, especially for python file invocations."
    ),
    "test_import_path": (
        "Test failed to import its target module. Common fix: ensure the "
        "target and tests live in the same directory, OR add an empty "
        "conftest.py at the project root, OR run pytest from the project "
        "root (not from tests/). When in doubt, use `run_tests` (which "
        "auto-detects the right invocation) instead of `run_bash python -m pytest`."
    ),
    "incomplete_deliverable": (
        "The goal listed N numbered requirements but the run finished with "
        "significantly fewer mutating tool calls. Re-read the goal and "
        "address EVERY numbered item before declaring complete. Common "
        "miss: the goal asks to (1) write code, (2) write tests, (3) run "
        "tests — agent does (1) and stops. Run tests is required when "
        "the goal mentions 'run tests' or 'verify they pass'."
    ),
    "narration_without_action": (
        "Model emitted future-tense intent prose ('I will now emit...', "
        "'Next, I will...') and stopped instead of emitting the actual "
        "tool_call block. The harness treated the prose as the final "
        "answer. Always emit the tool_call block IMMEDIATELY after the "
        "narration sentence — same turn, no delay."
    ),
    "unaddressed_file_in_goal": (
        "Goal explicitly named multiple files (e.g. 'storage.py and cli.py') "
        "but you only edited some of them. Model fixated on one file and "
        "skipped the other(s). Before declaring done, list every file the "
        "goal named and confirm each was actually written or edited. If "
        "the goal says 'fix storage.py and cli.py', BOTH must show up in "
        "your tool_call paths."
    ),
}


# ── Public API ──────────────────────────────────────────────────────


# Tool-result names that are SYNTHETIC — appended by Agent without a paired
# tool_call. We detect them by name so the flagger walks results
# independently of calls. Confirmed against engine/agent.py 2026-04-25.
_SYNTHETIC_RESULT_NAMES = frozenset({
    "auto_verify",      # appended after every successful write_file/edit_file
    "parse_error",      # appended when JSON tool-call parsing fails
    "stuck_repeat",     # appended when the same call repeats 3+ times
    "truncation",       # appended on transcript truncation events
})


def flag(result: Any, goal: str = "") -> list[FailureRecord]:
    """Walk result.tool_calls + result.tool_results and the final_answer;
    emit FailureRecord entries for each detected pattern.

    The agent's tool_results list contains BOTH real call results AND
    synthetic results (auto_verify, parse_error, stuck_repeat, truncation)
    appended with no matching tool_call. We walk results independently and
    track the most-recent real call for "what triggered this failure"
    attribution. zip(calls, results) was wrong — synthetic results carry
    most of the failure signal we care about (auto_verify catches the
    SyntaxError and missing-import cases; parse_error catches JSON quote
    leaks).
    """
    calls = list(getattr(result, "tool_calls", []) or [])
    results = list(getattr(result, "tool_results", []) or [])
    final_answer = str(getattr(result, "final_answer", "") or "")

    records: list[FailureRecord] = []

    # Walk results in order; track the most-recent REAL call for blame
    # attribution. Real calls in tool_calls are exactly the non-synthetic
    # tool_results; we walk both lists in parallel using a separate
    # "call cursor" that advances only on real (non-synthetic) results.
    #
    # Iteration accounting:
    #   - Real call's result -> iteration = call_cursor (1-indexed)
    #   - Synthetic AFTER a real call -> attributed to that call's iter
    #   - Synthetic BEFORE first real call (orphan parse_error) -> 0
    #     so downstream consumers can distinguish "failure happened before
    #     iter 1's recovery" from "failure during iter 1".
    call_cursor = 0
    iteration = 0

    for res in results:
        rname = getattr(res, "name", "") or ""
        is_synthetic = rname in _SYNTHETIC_RESULT_NAMES
        if not is_synthetic and call_cursor < len(calls):
            current_call = calls[call_cursor]
            call_cursor += 1
            iteration += 1
            attributed_iter = iteration
        elif is_synthetic:
            current_call = calls[call_cursor - 1] if call_cursor > 0 else None
            # Orphans (synthetic before any real call) get iter 0 so
            # recovery detection treats the next real call as the fix.
            attributed_iter = iteration if call_cursor > 0 else 0
        else:
            current_call = None
            attributed_iter = iteration

        cname = getattr(current_call, "name", "") if current_call else ""
        cargs = getattr(current_call, "arguments", {}) if current_call else {}
        if not isinstance(cargs, dict):
            cargs = {}
        path = cargs.get("path") if cargs else None

        if _detect_json_quote_leak(current_call, res):
            records.append(FailureRecord(
                category="json_quote_leak",
                iteration=attributed_iter, tool_name=cname,
                error_excerpt=(getattr(res, "error", "") or "")[:300],
                triggering_args=cargs, file_path=path,
                suggested_fix=_FIX_TEMPLATES["json_quote_leak"],
            ))
        if _detect_fstring_trap(current_call, res):
            records.append(FailureRecord(
                category="fstring_nested_quote",
                iteration=attributed_iter, tool_name=cname,
                error_excerpt=((getattr(res, "content", "") or "")
                               + (getattr(res, "error", "") or ""))[:300],
                triggering_args=cargs, file_path=path,
                suggested_fix=_FIX_TEMPLATES["fstring_nested_quote"],
            ))
        if _detect_missing_import(current_call, res):
            records.append(FailureRecord(
                category="missing_import",
                iteration=attributed_iter, tool_name=cname,
                error_excerpt=((getattr(res, "content", "") or "")
                               + (getattr(res, "error", "") or ""))[:300],
                triggering_args=cargs, file_path=path,
                suggested_fix=_FIX_TEMPLATES["missing_import"],
            ))
        if _detect_stuck_repeat(res):
            records.append(FailureRecord(
                category="stuck_repeat_loop",
                iteration=attributed_iter, tool_name=cname,
                error_excerpt=(getattr(res, "error", "") or "")[:300],
                triggering_args=cargs, file_path=path,
                suggested_fix=_FIX_TEMPLATES["stuck_repeat_loop"],
            ))
        if _detect_stale_anchor(current_call, res):
            records.append(FailureRecord(
                category="stale_anchor_edit",
                iteration=attributed_iter, tool_name=cname,
                error_excerpt=(getattr(res, "error", "") or "")[:300],
                triggering_args=cargs, file_path=path,
                suggested_fix=_FIX_TEMPLATES["stale_anchor_edit"],
            ))
        if _detect_cwd_assumption(current_call, res):
            records.append(FailureRecord(
                category="cwd_assumption",
                iteration=attributed_iter, tool_name=cname,
                error_excerpt=((getattr(res, "content", "") or "")
                               + (getattr(res, "error", "") or ""))[:300],
                triggering_args=cargs, file_path=path,
                suggested_fix=_FIX_TEMPLATES["cwd_assumption"],
            ))
        if _detect_test_import_path(current_call, res):
            records.append(FailureRecord(
                category="test_import_path",
                iteration=attributed_iter, tool_name=cname,
                error_excerpt=((getattr(res, "content", "") or "")
                               + (getattr(res, "error", "") or ""))[:300],
                triggering_args=cargs, file_path=path,
                suggested_fix=_FIX_TEMPLATES["test_import_path"],
            ))

    # Premature bail: only checked once at the end
    last_result = results[-1] if results else None
    if _detect_premature_bail(final_answer, last_result):
        records.append(FailureRecord(
            category="premature_bail",
            iteration=len(results) or 1,
            tool_name="",
            error_excerpt=(final_answer or "")[:300],
            triggering_args={},
            file_path=None,
            suggested_fix=_FIX_TEMPLATES["premature_bail"],
        ))

    # Whole-run detectors that depend on the goal text + run shape:
    if _detect_incomplete_deliverable(goal, result):
        records.append(FailureRecord(
            category="incomplete_deliverable",
            iteration=len(results) or 1,
            tool_name="",
            error_excerpt=(final_answer or "")[:300],
            triggering_args={"goal_excerpt": (goal or "")[:300]},
            file_path=None,
            suggested_fix=_FIX_TEMPLATES["incomplete_deliverable"],
        ))
    if _detect_narration_without_action(goal, result):
        records.append(FailureRecord(
            category="narration_without_action",
            iteration=len(results) or 1,
            tool_name="",
            error_excerpt=(final_answer or "")[:300],
            triggering_args={"goal_excerpt": (goal or "")[:300]},
            file_path=None,
            suggested_fix=_FIX_TEMPLATES["narration_without_action"],
        ))
    if _detect_unaddressed_file_in_goal(goal, result):
        records.append(FailureRecord(
            category="unaddressed_file_in_goal",
            iteration=len(results) or 1,
            tool_name="",
            error_excerpt=(final_answer or "")[:300],
            triggering_args={"goal_excerpt": (goal or "")[:300]},
            file_path=None,
            suggested_fix=_FIX_TEMPLATES["unaddressed_file_in_goal"],
        ))

    return records


def summarize(records: Iterable[FailureRecord]) -> dict:
    """One-line histogram of failure categories — handy for /flag-errors output."""
    counts: dict[str, int] = {}
    for r in records:
        counts[r.category] = counts.get(r.category, 0) + 1
    return counts


# ── Smoke test ──────────────────────────────────────────────────────


if __name__ == "__main__":
    # Lightweight smoke check using stub objects shaped like AgentResult.
    class _Stub:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)

    def _call(name, **args): return _Stub(name=name, arguments=args)
    def _result(success, content="", error=""): return _Stub(success=success, content=content, error=error)

    res = _Stub(
        final_answer="Task is complete. CLI works as expected.",
        tool_calls=[
            _call("write_file", path="cli.py", content=["bad"]),
            _call("write_file", path="cli.py", content=["fixed"]),
            _call("run_bash", command="python cli.py add buy-milk"),
        ],
        tool_results=[
            _result(False, error="bare JSON tool call failed to parse: Expecting ',' delimiter at line 4 column 158"),
            _result(False, content="syntax OK but cli.py references undefined names: json"),
            _result(True, content="{'stderr': 'Traceback (most recent call last):...'}"),
        ],
        stop_reason="answered",
    )
    flags = flag(res, goal="Build a todo CLI")
    print(f"Flagged {len(flags)} failure(s):")
    for f in flags:
        print(f"  [{f.category}] iter={f.iteration} tool={f.tool_name} -> {f.error_excerpt[:80]!r}")
    print(f"Summary: {summarize(flags)}")
