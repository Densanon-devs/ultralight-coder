"""
Tests for parallel tool execution in the Agent main loop.

- When the model emits 2+ read-only tool calls in a single turn, they
  execute concurrently via ThreadPoolExecutor.
- When ANY call in the batch is a mutating tool (write/edit/run_bash),
  the whole batch runs serially to preserve model-intended ordering.
- Parallel batches preserve input order in the output (pool.map).

Approach: use a stub model that emits a chosen sequence of tool-call
blocks, register slow no-op tools that sleep(0.5s), then check that the
total wall time for 3 parallel calls is well under 1.5s (serial sum).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.agent import Agent, _PARALLELIZABLE_TOOLS
from engine.agent_tools import ToolRegistry, ToolSchema, ToolResult


# ── test fixtures ─────────────────────────────────────────────────


class _ScriptedModel:
    """Returns a fixed sequence of responses, one per generate() call."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, *_a, **_k) -> str:
        if self.calls >= len(self.responses):
            return "done."
        r = self.responses[self.calls]
        self.calls += 1
        return r


def _slow_read_reg() -> tuple[ToolRegistry, dict]:
    """Registry with a `read_file` and a `grep` that each sleep 0.5s.
    Returns the registry and a dict the tool appends to so we can inspect
    call order."""
    log: list[tuple[str, float]] = []
    reg = ToolRegistry()

    def _mk_handler(name: str):
        def _h(**kw):
            t0 = time.monotonic()
            time.sleep(0.5)
            log.append((name, t0))
            return f"{name}-result"
        return _h

    for tool_name in ("read_file", "grep", "list_dir", "glob"):
        reg.register(ToolSchema(
            name=tool_name,
            description=f"{tool_name} (slow test stub)",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            function=_mk_handler(tool_name),
            category="file",
        ))

    # Also a slow write_file that IS mutating
    reg.register(ToolSchema(
        name="write_file",
        description="write_file (slow test stub)",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        function=_mk_handler("write_file"),
        category="file",
    ))
    return reg, {"log": log}


# ── parallelization decision table ────────────────────────────────


def test_parallelizable_tools_list_is_read_only():
    # Sanity: the frozenset doesn't accidentally include a mutating tool
    for name in _PARALLELIZABLE_TOOLS:
        assert name in ("read_file", "list_dir", "glob", "grep"), f"unexpected: {name}"


# ── timing test: 3 parallel read-only calls ───────────────────────


def _build_tool_call_block(calls: list[dict]) -> str:
    """Produce a model response that emits N `<tool_call>` blocks."""
    import json
    blocks = []
    for c in calls:
        blocks.append(f"<tool_call>\n{json.dumps(c)}\n</tool_call>")
    return "\n".join(blocks)


def test_three_parallel_reads_run_concurrently():
    reg, state = _slow_read_reg()
    tool_batch = _build_tool_call_block([
        {"name": "read_file", "arguments": {"path": "a.py"}},
        {"name": "read_file", "arguments": {"path": "b.py"}},
        {"name": "read_file", "arguments": {"path": "c.py"}},
    ])
    # After the batch, model gives its final answer (no tool calls) so
    # the loop exits cleanly.
    model = _ScriptedModel([tool_batch, "all files read, done."])
    agent = Agent(
        model=model, registry=reg, max_iterations=5,
        auto_verify_python=False, enable_goal_token_sweep=False,
    )
    t0 = time.monotonic()
    result = agent.run("read three files")
    wall = time.monotonic() - t0
    # 3 * 0.5s serial = 1.5s, parallel should be ~0.5s. Allow 1.0s budget.
    assert wall < 1.0, f"expected <1.0s (parallel), got {wall:.2f}s (probably serial)"
    # All 3 calls logged
    assert len(state["log"]) == 3, f"expected 3 call log entries, got {len(state['log'])}"
    # And they overlapped in time (start times within 0.2s of each other)
    starts = sorted(t for _, t in state["log"])
    spread = starts[-1] - starts[0]
    assert spread < 0.2, f"calls didn't actually overlap; start spread = {spread:.2f}s"


def test_mutating_call_in_batch_forces_serial():
    reg, state = _slow_read_reg()
    # Mix read_file + write_file in one batch → must go serial
    tool_batch = _build_tool_call_block([
        {"name": "read_file", "arguments": {"path": "a.py"}},
        {"name": "write_file", "arguments": {"path": "b.py"}},
        {"name": "read_file", "arguments": {"path": "c.py"}},
    ])
    model = _ScriptedModel([tool_batch, "done."])
    agent = Agent(
        model=model, registry=reg, max_iterations=5,
        auto_verify_python=False, enable_goal_token_sweep=False,
    )
    t0 = time.monotonic()
    agent.run("read-write-read")
    wall = time.monotonic() - t0
    # 3 * 0.5s serial = 1.5s, should be AT LEAST 1.3s if truly serial
    assert wall >= 1.3, f"expected serial (>=1.3s), got {wall:.2f}s — batch was parallelized despite write_file"
    assert len(state["log"]) == 3


def test_single_call_batch_doesnt_use_pool():
    # Single call should not go through the pool (min batch size is 2).
    reg, state = _slow_read_reg()
    tool_batch = _build_tool_call_block([
        {"name": "read_file", "arguments": {"path": "only.py"}},
    ])
    model = _ScriptedModel([tool_batch, "done."])
    agent = Agent(
        model=model, registry=reg, max_iterations=5,
        auto_verify_python=False, enable_goal_token_sweep=False,
    )
    t0 = time.monotonic()
    agent.run("single read")
    wall = time.monotonic() - t0
    # Just one call = 0.5s
    assert 0.4 <= wall < 1.0, f"single call wall time unexpected: {wall:.2f}s"
    assert len(state["log"]) == 1


def test_parallel_results_match_call_order():
    # Verify that when pool.map reorders internal scheduling, the
    # results[] list still matches the order of the requested calls.
    reg = ToolRegistry()
    def _mk(name, delay):
        def _h(**kw):
            time.sleep(delay)
            return f"result-from-{name}"
        return _h
    # First call is slow, second is fast — if the pool returned in
    # completion order we'd see second before first. pool.map preserves
    # input order, so we should get first first.
    reg.register(ToolSchema(name="read_file", description="slow",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        function=_mk("slow", 0.6), category="file",
    ))
    reg.register(ToolSchema(name="grep", description="fast",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        function=_mk("fast", 0.1), category="search",
    ))
    tool_batch = _build_tool_call_block([
        {"name": "read_file", "arguments": {"path": "a"}},
        {"name": "grep", "arguments": {"path": "b"}},
    ])
    model = _ScriptedModel([tool_batch, "done."])
    agent = Agent(
        model=model, registry=reg, max_iterations=5,
        auto_verify_python=False, enable_goal_token_sweep=False,
    )
    result = agent.run("slow then fast")
    # Inspect the tool_results captured on AgentResult
    # First result should be the slow one (call-order, not completion-order)
    assert len(result.tool_results) >= 2, f"expected 2+ tool_results, got {len(result.tool_results)}"
    first, second = result.tool_results[0], result.tool_results[1]
    assert first.content == "result-from-slow", f"first result should be slow: got {first.content}"
    assert second.content == "result-from-fast", f"second result should be fast: got {second.content}"


if __name__ == "__main__":
    import inspect
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith("test_")]
    passed = failed = 0
    for name, fn in fns:
        try:
            t0 = time.monotonic()
            fn()
            print(f"  PASS  {name}  ({time.monotonic()-t0:.2f}s)")
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
