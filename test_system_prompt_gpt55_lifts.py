"""
Regression-guard tests for the GPT-5.5 prompt-lift experiment.

EXPERIMENT RESULT 2026-04-25: ALL 5 lifts hurt the Qwen 2.5 Coder 14B.

Two sequential bench runs against build_todo_cli (which was 100% deterministic
on the pre-experiment baseline):
  - Tier 1 + Tier 2:  FAIL @ iter 19, 12 calls, 341s. Model bailed twice
                      with "task is complete" while stderr held a Traceback.
                      Initial hypothesis: Tier 2 (escape hatch) at fault.
  - Tier 1 only:      FAIL @ iter 5, 2 calls, 24s. Model emitted PLAN
                      narrative + "I'll do X" prose with NO tool call
                      following. PLAN preamble + ONE-ACTION rule together
                      pushed model into narrative-only turns.

Conclusion: GPT-5.5 prompt patterns do NOT transfer cleanly to the 14B.
This validates OpenAI's own caveat in the guide: "treat it as a new model
family to tune for, not a drop-in replacement." The 14B was already tuned
against its specific behavior in the existing _DEFAULT_SYSTEM; bolting on
frontier-model patterns is a net regression.

These tests exist to PREVENT FUTURE RE-INTRODUCTION of these patterns
without an A/B that explicitly rebuts the 2026-04-25 finding.

Bench artifacts kept on disk:
  - bench_gpt55_lifts.json  (Tier 1+2 result)
  - bench_tier1_only.json    (Tier 1 only result)
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.agent import _DEFAULT_SYSTEM


def test_plan_preamble_NOT_present():
    # Tier 1.1 (PLAN preamble) was reverted — pushed model into narrative-only turns
    assert "BEFORE YOUR FIRST TOOL CALL" not in _DEFAULT_SYSTEM
    # Note: the literal substring "PLAN:" may still appear in unrelated
    # contexts; the marker we check is the section header.


def test_one_action_rule_NOT_present():
    # Tier 1.2 (ONE ACTION PER TURN) was reverted with PLAN — together they
    # caused the iter-5 narrative-without-tool-call failure
    assert "ONE ACTION PER TURN" not in _DEFAULT_SYSTEM


def test_empty_result_recovery_NOT_present():
    # Tier 1.3 (EMPTY-RESULT RECOVERY) was reverted alongside the others.
    # Standalone it's probably benign, but couldn't isolate cleanly without
    # another full bench. Re-introduce only after a deterministic A/B.
    assert "EMPTY-RESULT RECOVERY" not in _DEFAULT_SYSTEM


def test_tier2_rules_NOT_present():
    # Tier 2.1 (UNCERTAINTY ESCAPE HATCH) and Tier 2.2 (TOOL BUDGET AWARENESS)
    # caused premature "task complete" bails — model read the escape hatch
    # as license to declare done with caveats.
    assert "UNCERTAINTY ESCAPE HATCH" not in _DEFAULT_SYSTEM
    assert "TOOL BUDGET AWARENESS" not in _DEFAULT_SYSTEM


def test_existing_anchors_still_present():
    # Regression guard: the new sections must NOT have replaced any of the
    # battle-tested 14B-ceiling rules from feedback_14b_tool_call_ceilings.md
    for anchor in (
        "READING IS NOT THE GOAL",
        "DO NOT REPEAT A FAILING CALL",
        "stuck_repeat",
        "F-STRING NESTED QUOTES TRAP",
        "FIX SYNTAX ERRORS BEFORE DOING ANYTHING ELSE",
        "FOR RENAMES, USE GREP FIRST",
        "FOR CIRCULAR IMPORTS",
        "auto_verify",
    ):
        assert anchor in _DEFAULT_SYSTEM, f"missing pre-existing rule: {anchor!r}"


def test_prompt_size_stays_reasonable():
    # The 5 lifts shouldn't bloat the prompt past ~16k chars (~5k tokens).
    # Hard ceiling here so we notice if a future lift goes overboard.
    assert len(_DEFAULT_SYSTEM) < 16000, f"prompt grew to {len(_DEFAULT_SYSTEM)} chars"


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
