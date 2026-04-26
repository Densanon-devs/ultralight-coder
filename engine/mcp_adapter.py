"""MCP (Model Context Protocol) client adapter — SCAFFOLD ONLY.

This module is the future home of ulcagent's MCP-client surface. It lets
the agent mount tools from external MCP servers (Densa Deck, GitHub,
Postgres, Filesystem, etc.) into the same `ToolRegistry` the builtin
tools live in.

**Status (2026-04-26):** Scaffold + interface. The actual JSON-RPC stdio
client and `tools/list` → `ToolRegistry.register` translation are NOT
wired yet. The `mcp` SDK is NOT a dependency. This file exists so:

  1. The interface is locked in and reviewed before we add the
     dependency, so when we DO turn it on the surface area is what
     we already designed.
  2. `--mcp <server>` CLI plumbing in `main.py` / `ulcagent.py` can
     point at a stable function name that returns a clear "not
     wired yet" error today.
  3. The `feedback_tool_count_regression.md` constraint (14B drops
     97.6% → 85.7% when the registry crosses ~10 tools) is encoded
     in the design — see `register_mcp_tools.tool_pack` arg below.

When we activate this:
  - `pip install mcp` (or pin to a specific version in pyproject.toml)
  - Implement `_launch_server` to spawn the configured subprocess and
    speak JSON-RPC over its stdio (use `mcp.client.stdio.stdio_client`)
  - Implement `_translate_tool` to convert an MCP `Tool` (name,
    description, inputSchema) into a `ToolSchema` whose `function`
    field invokes `session.call_tool(name, arguments)` and returns the
    result content
  - Optionally cache server processes so the same `--mcp densa-deck`
    invocation across REPL turns reuses the spawned subprocess
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.agent_tools import ToolRegistry


@dataclass
class McpServerConfig:
    """One MCP server's launch config.

    `command` + `args` is what the AI client uses (Claude desktop's
    `claude_desktop_config.json` format). `env` is an optional dict of
    extra env vars to inject into the subprocess.

    Built-in shortcuts (resolved at activation time):
      "densa-deck" -> {command: "densa-deck", args: ["mcp", "serve"]}

    Anything not in the shortcut table is treated as a literal command.
    """

    name: str
    command: str
    args: list[str]
    env: Optional[dict[str, str]] = None


# Built-in shortcuts so users can pass simple names like `--mcp densa-deck`
# without writing a config file. Add entries here as we ship MCP servers
# from sister projects.
_BUILTIN_SERVERS: dict[str, McpServerConfig] = {
    "densa-deck": McpServerConfig(
        name="densa-deck",
        command="densa-deck",
        args=["mcp", "serve"],
    ),
    # Future entries:
    # "d-brief":    McpServerConfig(...),
    # "densabooks": McpServerConfig(...),
}


def resolve_servers(server_specs: list[str]) -> list[McpServerConfig]:
    """Resolve user-facing server specs into McpServerConfig objects.

    Each spec is either a built-in shortcut name (`densa-deck`) or a path
    to a JSON config file. For now the JSON-config path is unimplemented;
    only the built-in shortcuts resolve.

    Returns: list of resolved configs. Unknown specs raise ValueError so
    the CLI can surface a clear error instead of silently dropping them.
    """
    resolved = []
    for spec in server_specs:
        if spec in _BUILTIN_SERVERS:
            resolved.append(_BUILTIN_SERVERS[spec])
            continue
        # Future: parse spec as path-to-JSON-config and load it.
        raise ValueError(
            f"Unknown MCP server: {spec!r}. "
            f"Built-in shortcuts: {sorted(_BUILTIN_SERVERS)}. "
            f"(Custom JSON config files: not yet supported.)"
        )
    return resolved


def register_mcp_tools(
    registry: ToolRegistry,
    server_specs: list[str],
    tool_pack: Optional[list[str]] = None,
) -> None:
    """Mount tools from the given MCP servers into `registry`.

    Args:
      registry: the agent's tool registry — MCP-discovered tools are
        added alongside the builtin tools.
      server_specs: list of server names / config paths. Empty list =
        no-op. This is the no-MCP code path the harness exercises by
        default; nothing here can fail or import optional deps when the
        list is empty.
      tool_pack: optional whitelist of tool names to actually expose.
        Necessary because the 14B models in this repo regress hard when
        the tool-count crosses ~10 (see `feedback_tool_count_regression.md`
        in user memory). Pass e.g. `["search_cards", "analyze_deck",
        "run_goldfish"]` to mount only the three you actually need for
        the session. Default `None` = mount everything the server
        advertises (only safe with very small servers).

    Raises:
      NotImplementedError if `server_specs` is non-empty — the JSON-RPC
      stdio client isn't wired yet. The empty-list case is a clean no-op
      so the rest of the harness can call this unconditionally.
    """
    if not server_specs:
        return
    # ---- Activation TODO ---- #
    # When we wire this up:
    #   1. Resolve server_specs to McpServerConfig objects via
    #      resolve_servers().
    #   2. For each config, spawn the subprocess using mcp's
    #      stdio_client + ClientSession.
    #   3. Call session.initialize() then session.list_tools().
    #   4. Filter by tool_pack if supplied.
    #   5. For each MCP Tool, build a ToolSchema whose function field
    #      calls session.call_tool(name, arguments) and returns the
    #      result content as a string.
    #   6. Track session lifecycles so the harness can cleanly terminate
    #      subprocesses when the agent loop exits.
    raise NotImplementedError(
        "MCP-client support is scaffolded but not wired yet. "
        "Track activation in engine/mcp_adapter.py docstring. "
        f"Requested servers: {server_specs}"
    )


def is_active() -> bool:
    """Return True if the MCP-client implementation is wired (i.e. the
    `mcp` SDK is installed AND the activation TODO has been done).

    Used by `--mcp <flag>` argparse handlers to decide whether to fail
    loudly with "not wired yet" or proceed.
    """
    # When activation lands, flip this to actually probe `import mcp`.
    return False
