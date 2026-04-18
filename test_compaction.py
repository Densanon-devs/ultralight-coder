"""
Tests for Agent._maybe_compact_transcript.

Exercises:
- No-op when under budget.
- Tool turn bodies get replaced when over budget.
- User goal turn is never touched.
- Recent turns are never touched.
- Multiple compactions across iterations don't corrupt structure.
- Assistant prose collapses to tool_call blocks on extreme overflow.
- Disabled budget (=0) is a no-op.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.agent import Agent
from engine.agent_tools import ToolRegistry


class _Stub:
    def generate(self, *_a, **_k): return ""


def _make(budget=1000, keep=4):
    return Agent(
        model=_Stub(),
        registry=ToolRegistry(),
        context_char_budget=budget,
        compact_keep_recent=keep,
    )


def _total(a: Agent) -> int:
    return sum(len(t.get("content") or "") for t in a._transcript)


def test_under_budget_is_noop():
    a = _make(budget=10000)
    a._transcript = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
        {"role": "tool", "content": "small result"},
    ]
    before = [dict(t) for t in a._transcript]
    a._maybe_compact_transcript()
    assert a._transcript == before
    assert a._compactions == 0


def test_over_budget_elides_old_tool_turns():
    a = _make(budget=500, keep=2)
    big_tool = "X" * 800
    a._transcript = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "<tool_call>{\"name\":\"read_file\"}</tool_call>"},
        {"role": "tool", "content": big_tool},
        {"role": "assistant", "content": "<tool_call>{\"name\":\"edit_file\"}</tool_call>"},
        {"role": "tool", "content": "recent tool result"},
    ]
    a._maybe_compact_transcript()
    assert a._compactions == 1
    # Oldest tool body got elided
    assert "elided" in a._transcript[2]["content"]
    assert "X" * 100 not in a._transcript[2]["content"]
    # Recent tool content untouched (it's inside the last 2 turns)
    assert a._transcript[4]["content"] == "recent tool result"
    # User goal untouched
    assert a._transcript[0]["content"] == "do the thing"


def test_goal_turn_never_touched_even_if_huge():
    a = _make(budget=200, keep=2)
    huge_goal = "Goal text: " + "Z" * 10000
    a._transcript = [
        {"role": "user", "content": huge_goal},
        {"role": "tool", "content": "Y" * 500},
        {"role": "tool", "content": "fresh"},
    ]
    a._maybe_compact_transcript()
    assert a._transcript[0]["content"] == huge_goal  # goal preserved


def test_recent_tool_turn_never_touched():
    a = _make(budget=300, keep=3)
    a._transcript = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "content": "old " * 200},
        {"role": "assistant", "content": "middle"},
        {"role": "tool", "content": "middle tool " * 50},
        {"role": "tool", "content": "newest keep me"},
    ]
    a._maybe_compact_transcript()
    # Index 2,3,4 are the last 3 — all preserved
    assert a._transcript[4]["content"] == "newest keep me"
    assert "middle tool" in a._transcript[3]["content"]


def test_disabled_budget_never_compacts():
    a = _make(budget=0)
    a._transcript = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "content": "X" * 100000},
        {"role": "tool", "content": "more"},
    ]
    before = [dict(t) for t in a._transcript]
    a._maybe_compact_transcript()
    assert a._transcript == before
    assert a._compactions == 0


def test_short_tool_turns_not_elided():
    # Bodies under 200 chars shouldn't be touched — eliding them saves
    # less than the marker adds.
    a = _make(budget=100, keep=2)
    a._transcript = [
        {"role": "user", "content": "goal"},
        {"role": "tool", "content": "short one"},  # 9 chars
        {"role": "tool", "content": "Y" * 300},   # big, will get elided
        {"role": "tool", "content": "keep me"},
    ]
    a._maybe_compact_transcript()
    assert a._transcript[1]["content"] == "short one"  # unchanged
    # Index 2 is within keep=2 window (len=4), so NOT elided either
    # (keep_from = max(1, 4-2) = 2, loop range(1, 2) only touches idx 1)
    # So only index 1 would be a candidate — but it's short → no elision.
    # This case: no compaction fires.
    assert a._compactions == 0


def test_extreme_overflow_also_elides_assistant_prose():
    # If eliding tool bodies isn't enough, assistant turns should get
    # reduced to their tool_call blocks.
    a = _make(budget=100, keep=2)
    tool_call_block = '<tool_call>\n{"name":"read_file","arguments":{"path":"a.py"}}\n</tool_call>'
    a._transcript = [
        {"role": "user", "content": "goal"},
        {
            "role": "assistant",
            "content": "Long monologue about the plan " * 100 + "\n" + tool_call_block,
        },
        {"role": "tool", "content": "T" * 2000},
        {"role": "tool", "content": "recent"},
    ]
    a._maybe_compact_transcript()
    # First compaction elides tool bodies (index 2 is in "keep recent 2" zone
    # so it's kept; but there are no OLDER tool turns, so the first pass
    # can't help). The fallback then elides the assistant prose.
    # Wait — with keep=2 and len=4, keep_from = max(1, 4-2) = 2. range(1,2)
    # = only index 1 (the assistant). So compaction only considers
    # assistant elision (there are no old tool turns to elide first).
    # Outcome: assistant got collapsed to just its tool_call block.
    assert "tool_call" in a._transcript[1]["content"]
    assert "monologue" not in a._transcript[1]["content"]


def test_multiple_compactions_do_not_corrupt_structure():
    # Simulate multiple iterations each pushing a big tool result.
    a = _make(budget=1000, keep=2)
    a._transcript = [{"role": "user", "content": "goal"}]
    for i in range(6):
        a._transcript.append({"role": "assistant", "content": f"<tool_call>{{\"i\":{i}}}</tool_call>"})
        a._transcript.append({"role": "tool", "content": f"result-{i}: " + ("B" * 500)})
        a._maybe_compact_transcript()
    # Old tool results got elided; the most recent one should still have its B's
    assert "B" * 100 in a._transcript[-1]["content"]
    # Old ones should be markers
    assert "elided" in a._transcript[2]["content"]
    # Goal preserved
    assert a._transcript[0]["content"] == "goal"
    # Multiple compactions fired
    assert a._compactions >= 1


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith("test_")]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
