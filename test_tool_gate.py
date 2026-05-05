"""Unit tests for engine.tool_gate — G-STEP confidence gate."""

from __future__ import annotations

import pytest

from engine.tool_gate import (
    EXPLORATION_TOOLS,
    ACTION_TOOLS,
    GateState,
    ToolGateConfig,
    check,
)


# ── Disabled gate is a no-op ────────────────────────────────────


def test_disabled_gate_allows_everything():
    state = GateState()
    config = ToolGateConfig(enabled=False)
    decision = check("read_file", {"path": "absolute_garbage"}, state, config)
    assert decision.allow


# ── Lexical-anchor check ────────────────────────────────────────


def test_anchored_path_in_goal_passes():
    state = GateState(seen_corpus="fix the bug in src/foo.py")
    config = ToolGateConfig(enabled=True)
    decision = check("read_file", {"path": "src/foo.py"}, state, config)
    assert decision.allow, decision.reason


def test_anchored_path_in_prior_observation_passes():
    state = GateState()
    state.record_observation("Listed: src/, tests/, README.md")
    config = ToolGateConfig(enabled=True)
    # 'tests/' surfaced in a prior list_dir — reading 'tests/test_x.py' is anchored.
    decision = check("read_file", {"path": "tests/test_x.py"}, state, config)
    assert decision.allow, decision.reason


def test_invented_path_with_no_anchor_is_gated():
    state = GateState(seen_corpus="fix the bug in src/foo.py")
    config = ToolGateConfig(enabled=True)
    decision = check(
        "read_file", {"path": "totally/unrelated/quux.py"}, state, config
    )
    assert not decision.allow
    assert "no lexical anchor" in decision.reason


def test_short_token_below_min_length_does_not_anchor():
    # 'a' and 'b' are below the 4-char default min_anchor_length
    # so neither anchors the call. The path itself ('a/b/x') also
    # has no chars >=4 except 'a/b/x' which itself is not in the
    # corpus.
    state = GateState(seen_corpus="a b")
    config = ToolGateConfig(enabled=True, min_anchor_length=4)
    decision = check("read_file", {"path": "a/b/x"}, state, config)
    assert not decision.allow


# ── Exploration cap ─────────────────────────────────────────────


def test_consecutive_exploration_under_cap_passes():
    state = GateState(seen_corpus="src/foo.py src/bar.py src/baz.py")
    config = ToolGateConfig(enabled=True, max_consecutive_exploration=4)
    state.consecutive_exploration = 3
    decision = check("read_file", {"path": "src/foo.py"}, state, config)
    assert decision.allow


def test_consecutive_exploration_at_cap_is_gated():
    state = GateState(seen_corpus="src/foo.py")
    config = ToolGateConfig(enabled=True, max_consecutive_exploration=4)
    state.consecutive_exploration = 4
    decision = check("read_file", {"path": "src/foo.py"}, state, config)
    assert not decision.allow
    assert "doom-loop" in decision.reason


def test_action_tool_resets_exploration_streak():
    state = GateState()
    state.consecutive_exploration = 7
    state.record_call("write_file")
    assert state.consecutive_exploration == 0


def test_exploration_call_increments_streak():
    state = GateState()
    state.record_call("read_file")
    state.record_call("list_dir")
    assert state.consecutive_exploration == 2


# ── Action tools are never gated ────────────────────────────────


@pytest.mark.parametrize("tool", sorted(ACTION_TOOLS))
def test_action_tools_are_never_gated_even_under_pathological_state(tool):
    state = GateState(seen_corpus="")  # empty corpus would gate exploration
    state.consecutive_exploration = 100  # over any reasonable cap
    config = ToolGateConfig(enabled=True)
    decision = check(tool, {"path": "/anything"}, state, config)
    assert decision.allow, f"{tool} was gated; action tools must always pass"


# ── Rejection message format ────────────────────────────────────


def test_rejection_message_names_the_tool_and_explains_self():
    state = GateState(seen_corpus="hello")
    config = ToolGateConfig(enabled=True)
    decision = check(
        "read_file", {"path": "totally/unrelated/quux.py"}, state, config
    )
    msg = decision.rejection_message("read_file")
    assert "read_file" in msg
    assert "confidence gate" in msg
    assert "well-formed" in msg
