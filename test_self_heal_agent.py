"""Integration test: self_heal classifier wired into the live Agent loop.

Drives the real Agent via the existing StubModel pattern from agent.py's
own smoke tests. Triggers two consecutive STALE_ANCHOR failures
(edit_file with bogus old_string) and asserts the synthetic
`self_heal_diagnose` ToolResult fires.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.agent import Agent
from engine.agent_builtins import build_default_registry


class _StubModel:
    """Minimal canned-response model — same shape the agent's own smoke
    tests use (see __main__ block at the bottom of engine/agent.py)."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int, stop, **kwargs):
        self.prompts.append(prompt)
        if self.calls >= len(self.responses):
            self.calls += 1
            return "(stub exhausted)"
        r = self.responses[self.calls]
        self.calls += 1
        return r


def test_two_stale_anchor_fires_self_heal_diagnose():
    """Two edit_file calls with non-matching old_string back-to-back
    should trigger self_heal_diagnose on iteration 2's tool turn."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "x.py"
        target.write_text("print('hello')\n", encoding="utf-8")

        reg = build_default_registry(tmp)
        # Iter 1: bogus edit (stale_anchor #1)
        # Iter 2: also bogus edit, different bogus old_string so the
        #         existing stuck_repeat guard doesn't fire (it triggers
        #         on identical args 3 times).
        # Iter 3: just answer "ok" so we exit cleanly.
        responses = [
            '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old_string": "definitely-not-here-1", "new_string": "world"}}</tool_call>',
            '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old_string": "definitely-not-here-2", "new_string": "world"}}</tool_call>',
            "ok done",
        ]
        model = _StubModel(responses)
        # Explicit enable_self_heal=True — the default flipped to False
        # after the 2026-05-05 A/B showed 0 firings on standard + stress
        # benches. The layer remains opt-in for stress scenarios and for
        # smaller models with weaker tool-call discipline.
        agent = Agent(
            model, reg, workspace_root=Path(tmp),
            max_iterations=5, enable_self_heal=True,
        )
        result = agent.run("Edit x.py to say world.")

    # The synthetic diagnose result must appear in tool_results history.
    diagnose_results = [r for r in result.tool_results if r.name == "self_heal_diagnose"]
    assert len(diagnose_results) == 1, [r.name for r in result.tool_results]
    assert "PAUSE" in diagnose_results[0].error
    assert "stale_anchor" in diagnose_results[0].error.lower()

    # Result counter mirrors it.
    assert result.self_heals == 1

    # The diagnose message must reach the model in iteration 3's prompt
    # (i.e. after the 2nd failure, before the model's 3rd response).
    assert "self_heal_diagnose" in model.prompts[2] or "PAUSE" in model.prompts[2]


def test_clean_run_does_not_fire_self_heal():
    """Happy-path run with no failures records self_heals=0 and emits no
    self_heal_diagnose results."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "x.py"
        target.write_text("print('hello')\n", encoding="utf-8")

        reg = build_default_registry(tmp)
        responses = [
            '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>',
            "Done — file says hello.",
        ]
        model = _StubModel(responses)
        agent = Agent(model, reg, workspace_root=Path(tmp), max_iterations=5)
        result = agent.run("Read x.py.")

    assert result.self_heals == 0
    assert not any(r.name == "self_heal_diagnose" for r in result.tool_results)


def test_default_is_off_post_2026_05_05():
    """As of 2026-05-05 the Agent default for enable_self_heal flipped
    to False after the standard + stress A/B recorded 0 firings across
    31 bench runs. Regression-protect that default so a later refactor
    doesn't silently re-enable a known-quiet layer."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "x.py"
        target.write_text("print('hello')\n", encoding="utf-8")

        reg = build_default_registry(tmp)
        # Same 2-streak pattern that DID fire when enable_self_heal=True
        # was the default. With the new default, no firing.
        responses = [
            '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old_string": "definitely-not-here-1", "new_string": "world"}}</tool_call>',
            '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old_string": "definitely-not-here-2", "new_string": "world"}}</tool_call>',
            "ok done",
        ]
        model = _StubModel(responses)
        # NOTE: no enable_self_heal kwarg — exercising the default.
        agent = Agent(model, reg, workspace_root=Path(tmp), max_iterations=5)
        result = agent.run("Edit x.py to say world.")

    assert result.self_heals == 0, "default should be OFF post-2026-05-05"
    assert not any(r.name == "self_heal_diagnose" for r in result.tool_results)


def test_enable_self_heal_false_skips_injection_entirely():
    """When the Agent is constructed with enable_self_heal=False, no
    self_heal_diagnose ToolResult is ever produced — even when a 2-streak
    same-class failure pattern is present. Used to A/B-measure the layer."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "x.py"
        target.write_text("print('hello')\n", encoding="utf-8")

        reg = build_default_registry(tmp)
        responses = [
            '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old_string": "definitely-not-here-1", "new_string": "world"}}</tool_call>',
            '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old_string": "definitely-not-here-2", "new_string": "world"}}</tool_call>',
            "ok done",
        ]
        model = _StubModel(responses)
        agent = Agent(
            model, reg, workspace_root=Path(tmp),
            max_iterations=5,
            enable_self_heal=False,
        )
        result = agent.run("Edit x.py to say world.")

    assert result.self_heals == 0
    assert not any(r.name == "self_heal_diagnose" for r in result.tool_results)


def test_alternating_failure_classes_does_not_fire():
    """A SyntaxError followed by a STALE_ANCHOR (different classes) must
    NOT trigger the diagnose injection — the streak resets between them."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "x.py"
        target.write_text("print('hello')\n", encoding="utf-8")

        reg = build_default_registry(tmp)
        # Iter 1: write_file with a deliberate Python SyntaxError → triggers
        #         auto_verify failure → SYNTAX_ERROR class.
        # Iter 2: edit_file with a non-matching old_string → STALE_ANCHOR class.
        # Different classes — should NOT inject diagnose.
        responses = [
            '<tool_call>{"name": "write_file", "arguments": {"path": "x.py", "content": "def broken(:\\n    pass\\n"}}</tool_call>',
            '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "old_string": "definitely-not-here", "new_string": "world"}}</tool_call>',
            "ok done",
        ]
        model = _StubModel(responses)
        # enable_self_heal=True so this test exercises the streak-reset
        # behavior; default flipped to False after the 2026-05-05 A/B.
        agent = Agent(
            model, reg, workspace_root=Path(tmp),
            max_iterations=5, enable_self_heal=True,
        )
        result = agent.run("Test alternating failures.")

    assert result.self_heals == 0
    assert not any(r.name == "self_heal_diagnose" for r in result.tool_results)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
