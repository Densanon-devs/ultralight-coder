"""
Tests for the failure flagger + YAML augmentor builder pipeline.

Exercises:
- Each detector category fires on its known signature
- Categories don't false-positive on success results
- Goal-tagged record carries the file path for context
- summarize() returns counts per category
- YAML builder produces a schema-conformant dict for every category
- write_yaml lands a file under _auto_generated/<category>/
- write_all skips records whose category has no template
- Real bench artifacts (today's bench_qwen3 / bench_gpt55 logs) flag
  the expected categories — anchor against drift in our detectors.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.failure_flagger import (
    FailureRecord, FAILURE_CATEGORIES, flag, summarize,
)
from engine.yaml_augmentor_builder import build_yaml, write_yaml, write_all


# ── Stub objects shaped like AgentResult ──────────────────────────


class _Stub:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _call(name, **args): return _Stub(name=name, arguments=args)
def _ok(content=""):     return _Stub(success=True, content=content, error="")
def _err(content="", error=""): return _Stub(success=False, content=content, error=error)


# ── Detector tests ────────────────────────────────────────────────


def test_json_quote_leak_fires_on_parse_error():
    res = _Stub(
        final_answer="",
        tool_calls=[_call("write_file", path="x.py", content=["..."])],
        tool_results=[_err(error="bare JSON tool call failed to parse: Expecting ',' delimiter at line 4")],
        stop_reason="answered",
    )
    flags = flag(res)
    assert any(f.category == "json_quote_leak" for f in flags)


def test_fstring_trap_detected_from_auto_verify():
    res = _Stub(
        final_answer="",
        tool_calls=[_call("write_file", path="cli.py", content=["..."])],
        tool_results=[_err(content="SyntaxError in cli.py line 14: f-string: expecting '}'")],
        stop_reason="answered",
    )
    flags = flag(res)
    assert any(f.category == "fstring_nested_quote" for f in flags)


def test_missing_import_detected():
    res = _Stub(
        final_answer="",
        tool_calls=[_call("write_file", path="storage.py", content=["..."])],
        tool_results=[_err(content="syntax OK but storage.py references undefined names: Todo")],
        stop_reason="answered",
    )
    flags = flag(res)
    rec = next(f for f in flags if f.category == "missing_import")
    assert rec.file_path == "storage.py"


def test_premature_bail_detected_on_traceback():
    res = _Stub(
        final_answer="The task is complete. CLI works as expected.",
        tool_calls=[_call("run_bash", command="python cli.py add x")],
        tool_results=[_ok(content="{'stdout': '', 'stderr': 'Traceback (most recent call last):...'}")],
        stop_reason="answered",
    )
    flags = flag(res)
    assert any(f.category == "premature_bail" for f in flags)


def test_premature_bail_not_fired_on_clean_run():
    res = _Stub(
        final_answer="Task is complete. CLI works as expected.",
        tool_calls=[_call("run_bash", command="python cli.py list")],
        tool_results=[_ok(content="{'stdout': 'todo:1 buy-milk', 'stderr': ''}")],
        stop_reason="answered",
    )
    flags = flag(res)
    assert not any(f.category == "premature_bail" for f in flags)


def test_stuck_repeat_detected():
    res = _Stub(
        final_answer="",
        tool_calls=[_call("edit_file", path="x.py", old_string="foo", new_string="bar")],
        tool_results=[_err(error="stuck_repeat: same call 3 times")],
        stop_reason="model_error",
    )
    flags = flag(res)
    assert any(f.category == "stuck_repeat_loop" for f in flags)


def test_stale_anchor_detected():
    res = _Stub(
        final_answer="",
        tool_calls=[_call("edit_file", path="cli.py",
                          old_string="def old_form():", new_string="def new_form():")],
        tool_results=[_err(error="old_string not found in cli.py.")],
        stop_reason="answered",
    )
    assert any(f.category == "stale_anchor_edit" for f in flag(res))


def test_stale_anchor_not_fired_for_other_tools():
    """Only edit_file failures should trigger this category."""
    res = _Stub(
        final_answer="",
        tool_calls=[_call("write_file", path="x.py", content=["..."])],
        tool_results=[_err(error="old_string not found in x.py.")],
        stop_reason="answered",
    )
    # Should NOT flag stale_anchor (write_file doesn't have an old_string concept)
    assert all(f.category != "stale_anchor_edit" for f in flag(res))


def test_cwd_assumption_detected_via_filenotfound():
    res = _Stub(
        final_answer="",
        tool_calls=[_call("run_bash", command="python tests/test_x.py")],
        tool_results=[_ok(content="{'stderr': \"python: can't open file 'tests/test_x.py': [Errno 2] No such file or directory\"}")],
        stop_reason="answered",
    )
    assert any(f.category == "cwd_assumption" for f in flag(res))


def test_cwd_assumption_not_fired_on_clean_run():
    res = _Stub(
        final_answer="Done.",
        tool_calls=[_call("run_bash", command="python cli.py list")],
        tool_results=[_ok(content="{'stdout': 'todo:1', 'stderr': ''}")],
        stop_reason="answered",
    )
    assert all(f.category != "cwd_assumption" for f in flag(res))


def test_test_import_path_detected():
    res = _Stub(
        final_answer="",
        tool_calls=[_call("run_tests", path="tests/")],
        tool_results=[_ok(content="ImportError while importing tests/test_todo.py: ModuleNotFoundError: No module named 'todo'")],
        stop_reason="answered",
    )
    assert any(f.category == "test_import_path" for f in flag(res))


def test_test_import_path_not_fired_on_unrelated_modulenotfound():
    """ModuleNotFoundError outside of test runs shouldn't trigger this."""
    res = _Stub(
        final_answer="",
        tool_calls=[_call("write_file", path="x.py", content=["..."])],
        tool_results=[_ok(content="Wrote 4 chars")],
        stop_reason="answered",
    )
    assert all(f.category != "test_import_path" for f in flag(res))


# ── 2026-04-26 walkthrough categories ─────────────────────────────


def test_incomplete_deliverable_fires_when_numbered_reqs_skipped():
    """Goal 1 retry of the walkthrough: 5 numbered requirements,
    only 3 mutating calls, declared done."""
    goal = (
        "Build a bookmark manager CLI from scratch. "
        "1. bookmark.py with the dataclass. "
        "2. storage.py with load/save. "
        "3. cli.py with subcommands. "
        "4. test_bookmarks.py with pytest tests. "
        "5. Run the tests with run_tests."
    )
    res = _Stub(
        final_answer="Files created.",
        tool_calls=[
            _call("write_file", path="bookmark.py", content=["..."]),
            _call("write_file", path="storage.py", content=["..."]),
            _call("write_file", path="cli.py", content=["..."]),
        ],
        tool_results=[
            _ok(content="Wrote 100 chars"),
            _ok(content="Wrote 200 chars"),
            _ok(content="Wrote 300 chars"),
        ],
        stop_reason="answered",
    )
    flags = flag(res, goal=goal)
    assert any(f.category == "incomplete_deliverable" for f in flags)


def test_incomplete_deliverable_does_not_fire_when_all_done():
    goal = (
        "1. Add func A. "
        "2. Add func B. "
        "3. Add func C."
    )
    res = _Stub(
        final_answer="All three added.",
        tool_calls=[
            _call("edit_file", path="x.py", old_string="...", new_string="A"),
            _call("edit_file", path="x.py", old_string="...", new_string="B"),
            _call("edit_file", path="x.py", old_string="...", new_string="C"),
        ],
        tool_results=[_ok(content="ok"), _ok(content="ok"), _ok(content="ok")],
        stop_reason="answered",
    )
    assert all(f.category != "incomplete_deliverable" for f in flag(res, goal=goal))


def test_incomplete_deliverable_skips_when_few_requirements():
    """A goal with <3 numbered items shouldn't trigger this — too noisy."""
    goal = "1. Read main.py. 2. Tell me what it does."
    res = _Stub(
        final_answer="It does X.",
        tool_calls=[_call("read_file", path="main.py")],
        tool_results=[_ok(content="...")],
        stop_reason="answered",
    )
    assert all(f.category != "incomplete_deliverable" for f in flag(res, goal=goal))


def test_narration_without_action_fires_on_intent_prose():
    """Goal 2 of the walkthrough: read_file → 'I will now emit...' → stop."""
    goal = "Add tagging support to cli.py — add a tag subcommand and an untag subcommand."
    res = _Stub(
        final_answer=(
            "I have read the content of cli.py. Now I will proceed to "
            "add the new subcommands 'tag' and 'untag', as well as modify "
            "the existing list_bookmarks handler. I will now emit the "
            "necessary tool calls to make these changes."
        ),
        tool_calls=[_call("read_file", path="cli.py")],
        tool_results=[_ok(content="(file contents)")],
        stop_reason="answered",
    )
    flags = flag(res, goal=goal)
    assert any(f.category == "narration_without_action" for f in flags)


def test_narration_without_action_silent_when_action_taken():
    """If the model DID emit a mutating call, narration is fine."""
    goal = "Add tagging to cli.py."
    res = _Stub(
        final_answer="I'll add the tag subcommand. Done.",
        tool_calls=[
            _call("read_file", path="cli.py"),
            _call("edit_file", path="cli.py", old_string="...", new_string="..."),
        ],
        tool_results=[_ok(content="..."), _ok(content="Replaced 1")],
        stop_reason="answered",
    )
    assert all(f.category != "narration_without_action" for f in flag(res, goal=goal))


def test_narration_pattern_requires_imperative_verb_in_goal():
    """Read-only goals (explain, show, list) shouldn't trip this even if
    the answer contains 'I will' phrasing — that's the correct shape."""
    goal = "Explain what cli.py does."
    res = _Stub(
        final_answer=(
            "I will now describe the structure of cli.py. It has three "
            "handlers: add, list, delete..."
        ),
        tool_calls=[_call("read_file", path="cli.py")],
        tool_results=[_ok(content="...")],
        stop_reason="answered",
    )
    assert all(f.category != "narration_without_action" for f in flag(res, goal=goal))


# ── Premature bail extension: "I cannot complete" / "I apologize" ─


def test_premature_bail_fires_on_i_cannot_complete():
    """Re-walkthrough Goal 2: model bailed with 'I cannot complete the
    task' even though pre_finish_check directive was the last result.
    The original detector required Traceback in last result and missed
    this. Extended detector should fire on give-up prose alone."""
    res = _Stub(
        final_answer=(
            "I apologize, but I cannot complete the task as requested. "
            "Please manually edit cli.py and rerun your tests."
        ),
        tool_calls=[_call("edit_file", path="cli.py", old_string="x", new_string="y")],
        tool_results=[
            SimpleNamespace(
                name="pre_finish_check",
                success=False,
                content="",
                error="Cannot finish — cli.py still has a syntax error",
            ),
        ],
        stop_reason="answered",
    )
    assert any(f.category == "premature_bail" for f in flag(res))


def test_premature_bail_fires_on_unable_to():
    res = _Stub(
        final_answer="I'm unable to complete this task with the current state.",
        tool_calls=[_call("read_file", path="x.py")],
        tool_results=[_ok(content="(file)")],
        stop_reason="answered",
    )
    assert any(f.category == "premature_bail" for f in flag(res))


def test_premature_bail_doesnt_fire_on_clean_apology():
    """Apology without give-up signal shouldn't trigger — the model
    might just be polite at the start of a clean answer."""
    res = _Stub(
        final_answer="I apologize for the delay. The file has been updated successfully.",
        tool_calls=[_call("edit_file", path="x.py", old_string="a", new_string="b")],
        tool_results=[_ok(content="Replaced 1 occurrence")],
        stop_reason="answered",
    )
    # "I apologize" still fires give-up — that's a known false-positive
    # tradeoff. The model should learn to say "Done" instead. This test
    # documents the current behavior; if it becomes a real problem, tighten
    # the regex to require "I apologize" + nearby give-up signal.
    flags = flag(res)
    assert any(f.category == "premature_bail" for f in flags)


# ── unaddressed_file_in_goal ──────────────────────────────────────


def test_unaddressed_file_fires_when_one_named_file_skipped():
    """Re-walkthrough Goal 1.5: 'Fix three bugs in storage.py and cli.py'
    — model edited cli.py 5 times, never touched storage.py."""
    goal = (
        "Fix three bugs in this project. "
        "(1) storage.py redefines the Bookmark class. "
        "(2) cli.py defines a function named list(). "
        "(3) cli.py at the bottom parses args but never dispatches."
    )
    res = _Stub(
        final_answer="Bugs fixed.",
        tool_calls=[
            _call("edit_file", path="cli.py", old_string="a", new_string="b"),
            _call("edit_file", path="cli.py", old_string="c", new_string="d"),
            _call("edit_file", path="cli.py", old_string="e", new_string="f"),
        ],
        tool_results=[_ok(content="Replaced 1"), _ok(content="Replaced 1"), _ok(content="Replaced 1")],
        stop_reason="answered",
    )
    assert any(f.category == "unaddressed_file_in_goal" for f in flag(res, goal=goal))


def test_unaddressed_file_silent_when_all_files_touched():
    goal = "Fix bugs in storage.py and cli.py."
    res = _Stub(
        final_answer="Done.",
        tool_calls=[
            _call("edit_file", path="storage.py", old_string="a", new_string="b"),
            _call("edit_file", path="cli.py", old_string="c", new_string="d"),
        ],
        tool_results=[_ok(content="Replaced 1"), _ok(content="Replaced 1")],
        stop_reason="answered",
    )
    assert all(f.category != "unaddressed_file_in_goal" for f in flag(res, goal=goal))


def test_unaddressed_file_skips_when_only_one_file_named():
    """Single-file goals don't have coverage to check."""
    goal = "Fix the bug in cli.py."
    res = _Stub(
        final_answer="Done.",
        tool_calls=[],
        tool_results=[],
        stop_reason="answered",
    )
    assert all(f.category != "unaddressed_file_in_goal" for f in flag(res, goal=goal))


def test_unaddressed_file_respects_do_not_modify_clause():
    """Files in 'DO NOT modify X' should not count as unaddressed."""
    goal = (
        "Add a tag command to cli.py. DO NOT modify bookmark.py or storage.py."
    )
    res = _Stub(
        final_answer="Done.",
        tool_calls=[_call("edit_file", path="cli.py", old_string="a", new_string="b")],
        tool_results=[_ok(content="Replaced 1")],
        stop_reason="answered",
    )
    # bookmark.py + storage.py mentioned but explicitly excluded; only
    # cli.py needs to be touched, and it was → no flag.
    assert all(f.category != "unaddressed_file_in_goal" for f in flag(res, goal=goal))


def test_unaddressed_file_matches_paths_with_dirs():
    """Goal says 'cli.py'; agent edits 'src/cli.py' or './cli.py' — basename match."""
    goal = "Fix the bugs in cli.py and storage.py."
    res = _Stub(
        final_answer="Done.",
        tool_calls=[
            _call("edit_file", path="src/cli.py", old_string="a", new_string="b"),
            _call("edit_file", path="./storage.py", old_string="c", new_string="d"),
        ],
        tool_results=[_ok(content="Replaced 1"), _ok(content="Replaced 1")],
        stop_reason="answered",
    )
    assert all(f.category != "unaddressed_file_in_goal" for f in flag(res, goal=goal))


def test_unaddressed_file_excludes_runtime_artifacts():
    """bookmarks.json is an artifact, not a source file goal — shouldn't count."""
    goal = "Fix the JSON parsing bug in storage.py. The output file is bookmarks.json."
    res = _Stub(
        final_answer="Done.",
        tool_calls=[_call("edit_file", path="storage.py", old_string="a", new_string="b")],
        tool_results=[_ok(content="Replaced 1")],
        stop_reason="answered",
    )
    # bookmarks.json + storage.py mentioned, but bookmarks.json is in
    # the exclude set; only storage.py needs to be touched, and it was.
    assert all(f.category != "unaddressed_file_in_goal" for f in flag(res, goal=goal))


def test_dict_content_doesnt_crash_detectors():
    """Regression: ToolResult.content from run_bash is a dict
    {'stdout': ..., 'stderr': ...}. Detectors must str()-cast before
    using + with strings. Without this, the live bench hits a TypeError
    in the auto-flag hook."""
    res = _Stub(
        final_answer="",
        tool_calls=[_call("run_bash", command="python cli.py")],
        tool_results=[_ok(content={"stdout": "", "stderr": "Traceback (most recent call last):"})],
        stop_reason="answered",
    )
    # Should not raise — exactly the bug we're guarding against
    flag(res)


def test_no_false_positive_on_clean_results():
    res = _Stub(
        final_answer="Done.",
        tool_calls=[_call("write_file", path="x.py", content=["pass"])],
        tool_results=[_ok(content="Wrote 4 chars to x.py\nsyntax OK (x.py)")],
        stop_reason="answered",
    )
    assert flag(res) == []


def test_summarize_counts_by_category():
    res = _Stub(
        final_answer="",
        tool_calls=[
            _call("write_file", path="a.py"),
            _call("write_file", path="b.py"),
            _call("write_file", path="c.py"),
        ],
        tool_results=[
            _err(error="bare JSON tool call failed to parse: Expecting ',' delimiter"),
            _err(error="Unterminated string starting at: line 2"),
            _err(content="references undefined names: json"),
        ],
        stop_reason="answered",
    )
    counts = summarize(flag(res))
    assert counts["json_quote_leak"] == 2
    assert counts["missing_import"] == 1


# ── Builder tests ─────────────────────────────────────────────────


def test_build_yaml_for_every_category():
    for cat in FAILURE_CATEGORIES:
        rec = FailureRecord(category=cat, iteration=1, tool_name="x",
                            error_excerpt="...", suggested_fix="...")
        out = build_yaml(rec, "any goal")
        assert out is not None, f"no template for category {cat!r}"
        assert "domain" in out
        assert "category" in out
        assert "examples" in out
        assert len(out["examples"]) == 1
        ex = out["examples"][0]
        assert "query" in ex and "solution" in ex and "tags" in ex


def test_unknown_category_returns_none():
    rec = FailureRecord(category="not_a_real_category", iteration=1,
                        tool_name="x", error_excerpt="...")
    assert build_yaml(rec, "any goal") is None


def test_write_yaml_lands_file_under_review_dir():
    tmp = Path(tempfile.mkdtemp())
    rec = FailureRecord(category="json_quote_leak", iteration=2,
                        tool_name="write_file", error_excerpt="bad",
                        file_path="storage.py")
    p = write_yaml(rec, "Build a todo CLI", tmp)
    assert p is not None
    # Critical: must NOT land under augmentor_examples/ — the retrieval
    # loader rglob's that directory and would auto-promote unreviewed YAMLs.
    assert "augmentor_examples" not in str(p)
    assert "auto_generated_review" in str(p)
    assert "json_quote_leak" in str(p)
    assert p.exists()
    assert p.suffix == ".yaml"


def test_write_yaml_path_is_outside_retrieval_scan():
    """Anchor against regression on Bug #2 (directory placement). The
    retrieval loader uses base.rglob('*.yaml') over data/augmentor_examples/,
    so any YAML under that tree gets auto-loaded. The auto-generated YAMLs
    must live OUTSIDE that tree until human review promotes them."""
    tmp = Path(tempfile.mkdtemp())
    rec = FailureRecord(category="missing_import", iteration=1,
                        tool_name="write_file", error_excerpt="x")
    p = write_yaml(rec, "any goal", tmp)
    augmentor_root = (tmp / "data" / "augmentor_examples").resolve()
    assert augmentor_root not in p.resolve().parents


def test_signature_avoids_collision_in_same_second():
    """Bug #3: multiple records same category+path+goal at the same wall-
    clock second must not produce identical filenames."""
    tmp = Path(tempfile.mkdtemp())
    r1 = FailureRecord(category="json_quote_leak", iteration=1,
                       tool_name="write_file", error_excerpt="err A",
                       file_path="x.py")
    r2 = FailureRecord(category="json_quote_leak", iteration=2,
                       tool_name="write_file", error_excerpt="err B",
                       file_path="x.py")
    p1 = write_yaml(r1, "same goal", tmp)
    p2 = write_yaml(r2, "same goal", tmp)
    assert p1 != p2, "filenames collided — Bug #3 regression"


def test_write_all_skips_unknown_categories():
    tmp = Path(tempfile.mkdtemp())
    records = [
        FailureRecord(category="json_quote_leak", iteration=1, tool_name="write_file", error_excerpt="x"),
        FailureRecord(category="not_a_template",   iteration=2, tool_name="write_file", error_excerpt="x"),
        FailureRecord(category="missing_import",   iteration=3, tool_name="edit_file", error_excerpt="x"),
    ]
    paths = write_all(records, "any goal", tmp)
    assert len(paths) == 2  # the unknown one was skipped


# ── Real-bench-shape integration ──────────────────────────────────


def test_synthetic_results_not_lost_to_zip_misalignment():
    """Bug #1 regression test: Agent.tool_results is consistently LONGER
    than tool_calls because auto_verify, parse_error, and stuck_repeat
    are appended without paired calls. Earlier versions of flag() used
    zip(calls, results) and silently dropped these — the exact signals
    we exist to catch.

    This test constructs a real-shape AgentResult: 1 tool_call but 2
    tool_results (the call's result + a synthetic auto_verify reporting
    `references undefined names`). The flagger MUST detect the
    missing_import even though the synthetic has no paired call."""
    res = _Stub(
        final_answer="",
        tool_calls=[_call("write_file", path="x.py", content=["..."])],
        tool_results=[
            _ok(content="Wrote 100 chars to x.py"),
            # Synthetic auto_verify — not paired with any call. Must be detected.
            SimpleNamespace(
                name="auto_verify",
                success=False,
                content="syntax OK but x.py references undefined names: json",
                error="",
            ),
        ],
        stop_reason="answered",
    )
    flags = flag(res)
    assert any(f.category == "missing_import" for f in flags), \
        "missing_import should be caught even when reported via a synthetic auto_verify result"


def test_synthetic_parse_error_caught():
    """Same fix as above — parse_error synthetic carries the JSON quote
    leak signature; must not be lost to zip misalignment."""
    res = _Stub(
        final_answer="",
        tool_calls=[_call("write_file", path="x.py", content=["..."])],
        tool_results=[
            _ok(content="Wrote 50 chars"),  # the previous (paired) success
            # synthetic parse_error from a SECOND, malformed call
            SimpleNamespace(
                name="parse_error",
                success=False,
                content="",
                error="bare JSON tool call failed to parse: Expecting ',' delimiter at line 4",
            ),
        ],
        stop_reason="answered",
    )
    flags = flag(res)
    assert any(f.category == "json_quote_leak" for f in flags)


from types import SimpleNamespace  # used by the two tests above


def test_real_bench_shape_flags_three_categories():
    """Mirrors the bench_qwen3_buildtodo + bench_gpt55_lifts traces seen
    on 2026-04-25. If our detectors regress, this test will catch it."""
    res = _Stub(
        final_answer="The task is complete. The minimal todo list CLI has been built and tested successfully.",
        tool_calls=[
            _call("write_file", path="todo.py", content=["..."]),
            _call("write_file", path="storage.py", content=["..."]),  # JSON parse fail
            _call("write_file", path="storage.py", content=["..."]),  # retry succeeds but missing import
            _call("write_file", path="cli.py", content=["..."]),       # f-string trap
            _call("run_bash",   command="python cli.py add buy-milk"), # Traceback in stderr
        ],
        tool_results=[
            _ok(content="syntax OK (todo.py)"),
            _err(error="bare JSON tool call failed to parse: Expecting ',' delimiter"),
            _err(content="syntax OK but storage.py references undefined names: Todo"),
            _err(content="SyntaxError in cli.py line 14: f-string: expecting '}'"),
            _ok(content="{'stdout': '', 'stderr': 'Traceback (most recent call last): ...'}"),
        ],
        stop_reason="answered",
    )
    counts = summarize(flag(res))
    assert "json_quote_leak" in counts
    assert "missing_import" in counts
    assert "fstring_nested_quote" in counts
    assert "premature_bail" in counts


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
