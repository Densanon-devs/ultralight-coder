# MCP (Model Context Protocol) — scaffold + activation plan

## Status (2026-04-26)

The MCP-client surface is **scaffolded but not wired**. The interface,
CLI plumbing, and registry hook are all in place; the JSON-RPC stdio
client itself is the one piece left to write before users can run

```bash
ulcagent --mcp densa-deck "build me a Selesnya tokens deck under $100"
```

and have the agent drive the Densa Deck engine through MCP tool calls.

Today, passing `--mcp <anything>` raises `NotImplementedError` so the gap
is loud, not silent. Without `--mcp`, ulcagent's behavior is identical
to before this scaffold — same builtin-only registry, same benchmarks,
same default tool count.

## Why MCP

[MCP](https://modelcontextprotocol.io/) is Anthropic's open standard for
AI clients to call tools on local servers. One protocol, many servers
— Densa Deck, GitHub, Postgres, Filesystem, your own custom ones.
Once ulcagent speaks MCP-client, every MCP server you have on disk
becomes an extension of the agent's tool set, with no per-server
adapter code.

It's also how the densanon-devs portfolio cross-pollinates: each
product (Densa Deck, D-Brief, DensaBooks) ships its own MCP server,
and ulcagent mounts whichever ones the user opts in to via `--mcp`.

## What's already in place

- **`engine/mcp_adapter.py`** — interface stubs for `register_mcp_tools`,
  `resolve_servers`, `is_active`, plus a `_BUILTIN_SERVERS` table for
  shortcut names like `densa-deck`. The activation TODO is documented
  inline.
- **`engine/agent_builtins.py::build_default_registry`** — accepts
  `mcp_servers` and `mcp_tool_pack` parameters. Empty/None = no-op
  (default and the only code path tests exercise today).
- **`ulcagent.py::_parse_mcp_arg`** — parses `--mcp foo,bar` from argv
  and threads it through `_build_agent` → `build_default_registry`.
- **`main.py`** — `--mcp` flag added to argparse; `run_agent_fast`
  accepts `mcp_servers`; both `--agent` and `/agent` REPL paths
  forward the value.

## What activation looks like

Three concrete steps, no surprises:

1. **Add the dep.** `pip install mcp` (or pin in `pyproject.toml` under
   an optional extras group).
2. **Implement `register_mcp_tools` in `engine/mcp_adapter.py`.** The
   docstring already lists the six steps — spawn subprocess, init
   session, list tools, filter by `tool_pack`, translate each MCP
   `Tool` into a `ToolSchema`, track lifecycles. Estimate: half a day
   if the SDK doesn't surprise.
3. **Flip `is_active()` to actually probe `import mcp`.** Done.

## Tool-count regression caveat

The [`feedback_tool_count_regression.md`](../../) memory found that the
14B models in this repo regress hard when the registry crosses ~10 tools
(97.6% → 85.7% on the agent benchmark). Mounting an MCP server like
Densa Deck adds ~17 free-tier tools, which would push the lean registry
well past that line.

The `mcp_tool_pack` parameter on `build_default_registry` is the
mitigation: pass an explicit whitelist and only those tools land in the
registry. E.g. for an MTG deckbuilding session:

```python
build_default_registry(
    workspace,
    mcp_servers=["densa-deck"],
    mcp_tool_pack=["search_cards", "analyze_deck", "run_goldfish"],
)
```

When activation lands, `--mcp` from the CLI should default to a curated
3-5 tool pack per server, with a `--mcp-all` opt-in for users who want
the full surface.

## Built-in shortcuts

Defined in `engine/mcp_adapter._BUILTIN_SERVERS`. Today:

| Shortcut | Command | Source repo |
|---|---|---|
| `densa-deck` | `densa-deck mcp serve` | [densa-deck](https://github.com/Densanon-devs/densa-deck) |

Add new entries here as sister projects ship MCP servers.

## Testing

`tests/test_mcp_adapter_scaffold.py` verifies the no-MCP-servers branch
is a clean no-op and that passing a non-empty list raises
`NotImplementedError` with the configured server names included in
the message. When activation lands, those tests will be extended to
exercise the actual JSON-RPC layer with a stub server.
