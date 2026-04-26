"""Tests for the MCP-client scaffold.

The scaffold is interface-only — the JSON-RPC stdio client isn't wired
yet. These tests pin the contract:

  1. Empty `mcp_servers` list = clean no-op (the default code path
     every existing ulcagent run takes).
  2. Non-empty list raises NotImplementedError with the configured
     server names in the message, so the gap is loud not silent.
  3. The `_parse_mcp_arg` helper handles both `--mcp foo,bar` and
     `--mcp=foo,bar` forms.
  4. The `_BUILTIN_SERVERS` shortcut table includes `densa-deck`.

When activation lands, replace these with actual subprocess + JSON-RPC
exercises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_empty_mcp_servers_is_noop():
    """The default `mcp_servers=None` / `[]` path must be a clean no-op
    so existing ulcagent runs aren't affected by the scaffold landing."""
    from engine.agent_tools import ToolRegistry
    from engine.mcp_adapter import register_mcp_tools

    reg = ToolRegistry()
    # None and empty list both must work without raising.
    register_mcp_tools(reg, [])
    register_mcp_tools(reg, [], tool_pack=None)
    register_mcp_tools(reg, [], tool_pack=["search_cards"])
    # Registry should still be empty afterwards.
    assert reg.status()["total"] == 0


def test_non_empty_mcp_servers_raises_with_named_servers():
    """Passing a real server name today must raise NotImplementedError —
    the activation TODO hasn't been done. Error message must name the
    requested servers so the user sees what triggered the failure."""
    from engine.agent_tools import ToolRegistry
    from engine.mcp_adapter import register_mcp_tools

    reg = ToolRegistry()
    with pytest.raises(NotImplementedError) as exc_info:
        register_mcp_tools(reg, ["densa-deck"])
    assert "densa-deck" in str(exc_info.value)


def test_is_active_returns_false_today():
    """The activation gate. Flip to True after the JSON-RPC client lands
    AND the `mcp` SDK is installed."""
    from engine.mcp_adapter import is_active
    assert is_active() is False


def test_resolve_servers_resolves_builtin_shortcut():
    from engine.mcp_adapter import resolve_servers

    configs = resolve_servers(["densa-deck"])
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.name == "densa-deck"
    assert cfg.command == "densa-deck"
    assert cfg.args == ["mcp", "serve"]


def test_resolve_servers_unknown_raises_with_helpful_message():
    from engine.mcp_adapter import resolve_servers

    with pytest.raises(ValueError) as exc_info:
        resolve_servers(["does-not-exist"])
    msg = str(exc_info.value)
    assert "does-not-exist" in msg
    # Built-in list should be surfaced so the user knows what works.
    assert "densa-deck" in msg


def test_parse_mcp_arg_default_empty():
    from ulcagent import _parse_mcp_arg
    assert _parse_mcp_arg(["ulcagent"]) == []
    assert _parse_mcp_arg(["ulcagent", "do", "thing"]) == []


def test_parse_mcp_arg_space_form():
    from ulcagent import _parse_mcp_arg
    assert _parse_mcp_arg(["ulcagent", "--mcp", "densa-deck"]) == ["densa-deck"]
    assert _parse_mcp_arg(["ulcagent", "--mcp", "a,b,c"]) == ["a", "b", "c"]


def test_parse_mcp_arg_equals_form():
    from ulcagent import _parse_mcp_arg
    assert _parse_mcp_arg(["ulcagent", "--mcp=densa-deck"]) == ["densa-deck"]
    assert _parse_mcp_arg(["ulcagent", "--mcp=a,b"]) == ["a", "b"]


def test_parse_mcp_arg_strips_whitespace():
    from ulcagent import _parse_mcp_arg
    assert _parse_mcp_arg(["ulcagent", "--mcp", "a, b , c"]) == ["a", "b", "c"]


def test_build_default_registry_passes_through_mcp_servers():
    """Smoke: the ulcagent registry builder accepts and forwards the new
    parameter. Empty list = no exception. Non-empty list = the
    NotImplementedError bubbles up so the gap is visible."""
    import tempfile
    from engine.agent_builtins import build_default_registry

    with tempfile.TemporaryDirectory() as tmp:
        # Empty list: clean no-op, registry built with builtins as usual.
        reg = build_default_registry(tmp, mcp_servers=[])
        assert reg.status()["total"] >= 8  # at least the core builtins

        # Non-empty list: NotImplementedError surfaces during the build,
        # not silently swallowed — proves the hook is wired.
        with pytest.raises(NotImplementedError):
            build_default_registry(tmp, mcp_servers=["densa-deck"])
