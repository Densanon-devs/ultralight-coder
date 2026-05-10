"""Registry-level tests for the enable_web gate.

Verifies that web tools are NOT in the registry by default and ARE
registered (with risky=True) when enable_web=True.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from engine.agent_builtins import build_default_registry


def _build(enable_web: bool):
    with tempfile.TemporaryDirectory() as tmp:
        return build_default_registry(Path(tmp), enable_web=enable_web)


class TestEnableWebGate:
    def test_default_no_web_tools(self):
        reg = _build(enable_web=False)
        names = {t.name for t in reg.enabled_tools()}
        assert "web_search" not in names
        assert "fetch_url" not in names

    def test_enable_web_adds_both_tools(self):
        reg = _build(enable_web=True)
        names = {t.name for t in reg.enabled_tools()}
        assert "web_search" in names
        assert "fetch_url" in names

    def test_web_tools_marked_risky(self):
        reg = _build(enable_web=True)
        ws = reg.get("web_search")
        fu = reg.get("fetch_url")
        assert ws is not None and ws.risky is True
        assert fu is not None and fu.risky is True

    def test_web_tools_orthogonal_to_extended(self):
        # enable_web=True with extended_tools=False should still register web tools.
        with tempfile.TemporaryDirectory() as tmp:
            reg = build_default_registry(
                Path(tmp), enable_web=True, extended_tools=False
            )
        names = {t.name for t in reg.enabled_tools()}
        assert "web_search" in names
        assert "fetch_url" in names
        # And rename_symbol (an extended-only tool) should NOT be present.
        assert "rename_symbol" not in names

    def test_web_tools_have_correct_category(self):
        reg = _build(enable_web=True)
        assert reg.get("web_search").category == "web"
        assert reg.get("fetch_url").category == "web"

    def test_web_search_schema_required_query(self):
        reg = _build(enable_web=True)
        params = reg.get("web_search").parameters
        assert "query" in params["required"]

    def test_fetch_url_schema_required_url(self):
        reg = _build(enable_web=True)
        params = reg.get("fetch_url").parameters
        assert "url" in params["required"]
