"""
Tests for B6 architect-mode auto-routing.

Exercises:
- Single-file edits do NOT trigger architect mode
- "build a todo cli" triggers architect mode via phrase match
- 3+ file tokens in one goal triggers architect mode
- Filename count below threshold doesn't trigger
- --architect force_on overrides everything
- --no-architect force_off overrides everything
- Negation phrasing in the goal disables architect
- _build_architect_agent raises if flat agent has no _ulcagent_ctx
- _build_architect_agent returns an ArchitectAgent reusing the flat agent's
  registry/model/memory
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from ulcagent import _should_use_architect, _build_architect_agent


# ── Heuristic ───────────────────────────────────────────────────

def test_single_file_edit_no_trigger():
    assert not _should_use_architect("Add a divide function to calculator.py")


def test_simple_fix_no_trigger():
    assert not _should_use_architect("Fix the off-by-one in paginate")


def test_phrase_build_a_triggers():
    assert _should_use_architect("Build a todo CLI app")


def test_phrase_scaffold_triggers():
    assert _should_use_architect("Scaffold a FastAPI service with auth")


def test_phrase_implement_a_triggers():
    assert _should_use_architect("Implement a simple key-value store")


def test_three_file_tokens_trigger():
    assert _should_use_architect(
        "Touch up todo.py, storage.py, and cli.py — minor cleanup"
    )


def test_two_file_tokens_no_trigger():
    assert not _should_use_architect("Update server.py and client.py")


def test_one_file_token_no_trigger():
    assert not _should_use_architect("Read main.py and tell me what it does")


def test_force_on_overrides_simple_goal():
    assert _should_use_architect("simple", force_on=True)


def test_force_off_overrides_scaffold_phrase():
    assert not _should_use_architect("Build a todo CLI", force_off=True)


def test_negation_disables_architect():
    assert not _should_use_architect("Build a todo CLI but don't use architect mode")


def test_force_off_beats_force_on():
    # If both flags somehow appear, force_off wins (safer default)
    assert not _should_use_architect("Build a todo", force_on=True, force_off=True)


def test_mixed_filenames_count():
    # Diverse extensions should still all count
    goal = "I want changes to a.py, b.js, c.go — keep them small"
    assert _should_use_architect(goal)


def test_dotfile_paths_count():
    goal = "Update src/foo.py, lib/bar.py, tests/test_foo.py"
    assert _should_use_architect(goal)


def test_case_insensitive_phrases():
    assert _should_use_architect("BUILD A todo CLI")
    assert _should_use_architect("Scaffold a project")


# ── Fix/debug suppression (2026-04-26 walkthrough finding #2) ────


def test_fix_three_bugs_suppresses_architect_even_with_3_files():
    # The exact catastrophe shape from Goal 1.5 of the handheld walkthrough.
    g = ("Fix three bugs in this project. (1) storage.py redefines Bookmark. "
         "(2) cli.py shadows list. (3) cli.py never dispatches via args.func. "
         "Run the tests in test_bookmarks.py after.")
    assert not _should_use_architect(g)


def test_rename_across_files_suppresses_architect():
    g = "Rename foo to bar across server.py, client.py, models.py, and tests/test_foo.py"
    assert not _should_use_architect(g)


def test_refactor_suppresses_architect():
    g = "Refactor the auth flow across server.py, middleware.py, and config.py"
    assert not _should_use_architect(g)


def test_broken_signal_suppresses_architect():
    g = "The login flow is broken across auth.py, user.py, and session.py — figure out why"
    assert not _should_use_architect(g)


def test_remove_duplicate_suppresses_architect():
    g = "Remove the duplicate Bookmark class in storage.py — bookmark.py and cli.py already use it"
    assert not _should_use_architect(g)


def test_scaffold_keywords_still_win_when_no_fix_signal():
    # A genuine multi-file scaffold should still trigger architect.
    g = "Build a todo CLI with todo.py, storage.py, cli.py, and test_todo.py"
    assert _should_use_architect(g)


def test_force_on_still_overrides_suppression():
    # If the user explicitly asks for architect, give it to them.
    assert _should_use_architect("Fix bugs in a.py b.py c.py", force_on=True)


def test_negation_takes_priority_over_suppression():
    # --no-architect already off, suppression doesn't matter, returns False.
    g = "Fix bugs across a.py, b.py, c.py — don't use architect mode"
    assert not _should_use_architect(g)


# ── Builder ─────────────────────────────────────────────────────

class _StubFlat:
    """Bare flat-agent stand-in carrying the build context."""
    def __init__(self, ctx):
        self._ulcagent_ctx = ctx


def _bare_ctx():
    """Minimal context — enough for ArchitectAgent.__init__ to succeed."""
    from engine.agent_tools import ToolRegistry
    return {
        "model": object(),
        "registry": ToolRegistry(),
        "system_prompt_extra": "",
        "workspace_root": Path("."),
        "memory": None,
        "max_tokens_per_turn": 1024,
        "temperature": 0.1,
        "confirm_risky": None,
        "on_event": None,
    }


def test_build_architect_requires_ctx():
    class NoCtx:
        pass
    try:
        _build_architect_agent(NoCtx())
    except RuntimeError as e:
        assert "_ulcagent_ctx" in str(e)
        return
    assert False, "expected RuntimeError"


def test_build_architect_returns_architect_agent():
    from engine.architect_agent import ArchitectAgent
    flat = _StubFlat(_bare_ctx())
    arch = _build_architect_agent(flat)
    assert isinstance(arch, ArchitectAgent)


def test_build_architect_shares_registry():
    flat = _StubFlat(_bare_ctx())
    arch = _build_architect_agent(flat)
    assert arch.registry is flat._ulcagent_ctx["registry"]
    assert arch.model is flat._ulcagent_ctx["model"]


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
