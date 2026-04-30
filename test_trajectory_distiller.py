"""
Tests for engine.trajectory_distiller.

Exercises:
- detect_intent finds the right label for common goals
- detect_language picks up file extensions and keywords
- distill converts AgentResult-shape into a Trajectory
- distill abridges large argument blobs
- distill records files_touched only for mutating tools
- save writes successful runs to trajectory_examples/, failures to trajectory_review/
- save+load round-trips correctly
- match returns ranked trajectories for similar queries
- match prefers same-intent + same-language hits
- match returns empty list when there are no stored examples
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.trajectory_distiller import (
    Trajectory, detect_intent, detect_language, distill, save, load, match,
)


# ── intent / language detection ────────────────────────────────────

def test_detect_intent_codegen():
    assert detect_intent("Add a divide function to calc.py") == "codegen"
    assert detect_intent("Implement a JSON parser") == "codegen"


def test_detect_intent_refactor():
    assert detect_intent("Rename foo to bar across the project") == "refactor"


def test_detect_intent_debug():
    assert detect_intent("Fix the bug in the login handler") == "debug"


def test_detect_intent_unknown():
    assert detect_intent("hmm") == "other"


def test_detect_language_python():
    assert detect_language("Add a divide function to calc.py") == "python"


def test_detect_language_javascript():
    assert detect_language("Refactor reducer.js") == "javascript"


def test_detect_language_typescript():
    assert detect_language("Add the User interface to types.ts") == "typescript"


def test_detect_language_rust():
    assert detect_language("Implement parse() in lib.rs") == "rust"


def test_detect_language_unknown():
    assert detect_language("clean up") == "other"


# ── distillation ───────────────────────────────────────────────────

class _Stub:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _stub_call(name, **args):
    return _Stub(name=name, arguments=args)


def test_distill_basic():
    result = _Stub(
        final_answer="Done.",
        iterations=3,
        stop_reason="answered",
        wall_time=5.0,
        tool_calls=[
            _stub_call("read_file", path="calc.py"),
            _stub_call("edit_file", path="calc.py", old_string="def add", new_string="def divide"),
        ],
    )
    traj = distill(result, "Add a divide function to calc.py")
    assert traj.success
    assert traj.intent == "codegen"
    assert traj.language == "python"
    assert traj.files_touched == ["calc.py"]
    assert len(traj.tool_calls) == 2


def test_distill_failure():
    result = _Stub(
        final_answer="",
        iterations=20,
        stop_reason="max_iterations",
        wall_time=600.0,
        tool_calls=[],
    )
    traj = distill(result, "Some goal")
    assert not traj.success


def test_distill_abridges_long_args():
    big = "x" * 5000
    result = _Stub(
        final_answer="Done.",
        iterations=1,
        stop_reason="answered",
        wall_time=1.0,
        tool_calls=[_stub_call("write_file", path="big.py", content=big)],
    )
    traj = distill(result, "Write a big file")
    args = traj.tool_calls[0]["args"]
    assert "5000 chars" in args["content"]
    assert big not in args["content"]


def test_distill_files_touched_excludes_reads():
    result = _Stub(
        final_answer="Done.",
        iterations=2,
        stop_reason="answered",
        wall_time=1.0,
        tool_calls=[
            _stub_call("read_file", path="a.py"),
            _stub_call("grep", pattern="foo", path="b.py"),
            _stub_call("edit_file", path="c.py", old_string="x", new_string="y"),
        ],
    )
    traj = distill(result, "edit c.py")
    assert traj.files_touched == ["c.py"]


# ── persistence ────────────────────────────────────────────────────

def test_save_success_goes_to_examples():
    tmp = Path(tempfile.mkdtemp())
    traj = Trajectory(
        goal="add foo", success=True, iterations=1,
        stop_reason="answered", wall_time=1.0, final_answer="ok",
        intent="codegen", language="python",
    )
    p = save(traj, tmp)
    assert "trajectory_examples" in str(p)
    assert p.exists()


def test_save_failure_goes_to_review():
    tmp = Path(tempfile.mkdtemp())
    traj = Trajectory(
        goal="something hard", success=False, iterations=20,
        stop_reason="max_iterations", wall_time=600.0, final_answer="",
        intent="debug", language="python",
    )
    p = save(traj, tmp)
    assert "trajectory_review" in str(p)
    assert p.exists()


def test_save_load_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    traj = Trajectory(
        goal="add divide to calc.py", success=True, iterations=4,
        stop_reason="answered", wall_time=12.3, final_answer="Done.",
        tool_calls=[{"name": "edit_file", "args": {"path": "calc.py"}}],
        files_touched=["calc.py"], intent="codegen", language="python",
    )
    p = save(traj, tmp)
    loaded = load(p)
    assert loaded.goal == traj.goal
    assert loaded.tool_calls == traj.tool_calls
    assert loaded.files_touched == traj.files_touched
    assert loaded.intent == traj.intent


# ── retrieval ──────────────────────────────────────────────────────

def test_match_empty_returns_empty():
    tmp = Path(tempfile.mkdtemp())
    assert match("anything", tmp) == []


def test_match_finds_similar_goal():
    tmp = Path(tempfile.mkdtemp())
    save(Trajectory(
        goal="add divide function to calculator.py",
        success=True, iterations=3, stop_reason="answered", wall_time=5,
        final_answer="ok", intent="codegen", language="python",
    ), tmp)
    save(Trajectory(
        goal="rename foo to bar across project",
        success=True, iterations=5, stop_reason="answered", wall_time=8,
        final_answer="ok", intent="refactor", language="python",
    ), tmp)
    hits = match("add subtract function to calculator.py", tmp)
    assert len(hits) >= 1
    # The codegen-add-function trajectory should rank above refactor-rename
    top_score, top_traj = hits[0]
    assert "divide" in top_traj.goal


def test_match_intent_bonus():
    tmp = Path(tempfile.mkdtemp())
    save(Trajectory(
        goal="thing widget gadget",
        success=True, iterations=1, stop_reason="answered", wall_time=1,
        final_answer="ok", intent="codegen", language="python",
    ), tmp)
    save(Trajectory(
        goal="thing widget gadget",  # same tokens
        success=True, iterations=1, stop_reason="answered", wall_time=1,
        final_answer="ok", intent="refactor", language="python",
    ), tmp)
    hits = match("add thing widget gadget", tmp)  # intent=codegen
    assert len(hits) == 2
    # The codegen one should rank higher because of intent bonus
    assert hits[0][1].intent == "codegen"


def test_match_excludes_failed_runs():
    tmp = Path(tempfile.mkdtemp())
    save(Trajectory(
        goal="add divide", success=False, iterations=1,
        stop_reason="model_error", wall_time=1, final_answer="",
        intent="codegen", language="python",
    ), tmp)
    # Failed runs go to trajectory_review/, not trajectory_examples/, so
    # match should find nothing.
    assert match("add divide", tmp) == []


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
