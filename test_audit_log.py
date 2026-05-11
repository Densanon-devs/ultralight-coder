"""Tests for engine/audit_log.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.audit_log import AuditLog, AUDIT_SUBDIR


class TestAuditLogBasics:
    def test_for_workspace_returns_none_without_audit_dir(self, tmp_path):
        ws = tmp_path / "ad-hoc-workspace"
        ws.mkdir()
        assert AuditLog.for_workspace(ws) is None

    def test_for_workspace_constructs_when_audit_dir_present(self, tmp_path):
        ws = tmp_path / "engagement"
        (ws / AUDIT_SUBDIR).mkdir(parents=True)
        log = AuditLog.for_workspace(ws)
        assert log is not None
        assert log.log_dir == ws / AUDIT_SUBDIR

    def test_log_call_writes_jsonl(self, tmp_path):
        ws = tmp_path / "engagement"
        (ws / AUDIT_SUBDIR).mkdir(parents=True)
        log = AuditLog(ws)
        log.log_call("web_search", args={"query": "test query"}, result="result body")

        # Find the daily file
        files = list((ws / AUDIT_SUBDIR).glob("*.jsonl"))
        assert len(files) == 1, f"expected one jsonl, got {files}"
        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tool"] == "web_search"
        assert entry["args"] == {"query": "test query"}
        assert entry["result_chars"] == len("result body")
        assert entry["error"] is None

    def test_multiple_calls_append(self, tmp_path):
        ws = tmp_path / "engagement"
        (ws / AUDIT_SUBDIR).mkdir(parents=True)
        log = AuditLog(ws)
        for i in range(5):
            log.log_call("fetch_url", args={"url": f"https://example.com/{i}"})
        files = list((ws / AUDIT_SUBDIR).glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_long_result_truncated_in_preview(self, tmp_path):
        ws = tmp_path / "engagement"
        (ws / AUDIT_SUBDIR).mkdir(parents=True)
        log = AuditLog(ws)
        big = "X" * 5000
        log.log_call("fetch_url", args={"url": "x"}, result=big)
        files = list((ws / AUDIT_SUBDIR).glob("*.jsonl"))
        entry = json.loads(files[0].read_text(encoding="utf-8").strip())
        # result_chars reflects original size
        assert entry["result_chars"] == 5000
        # preview is capped + indicates truncation
        assert "more chars" in entry["result_preview"]
        assert len(entry["result_preview"]) < 1000

    def test_error_field_logged(self, tmp_path):
        ws = tmp_path / "engagement"
        (ws / AUDIT_SUBDIR).mkdir(parents=True)
        log = AuditLog(ws)
        log.log_call("run_bash", args={"command": "false"}, error="exit code 1")
        files = list((ws / AUDIT_SUBDIR).glob("*.jsonl"))
        entry = json.loads(files[0].read_text(encoding="utf-8").strip())
        assert entry["error"] == "exit code 1"

    def test_extra_field_logged(self, tmp_path):
        ws = tmp_path / "engagement"
        (ws / AUDIT_SUBDIR).mkdir(parents=True)
        log = AuditLog(ws)
        log.log_call("web_search", args={"query": "x"},
                     extra={"engagement": "Acme Corp", "iteration": 3})
        files = list((ws / AUDIT_SUBDIR).glob("*.jsonl"))
        entry = json.loads(files[0].read_text(encoding="utf-8").strip())
        assert entry["extra"]["engagement"] == "Acme Corp"
        assert entry["extra"]["iteration"] == 3

    def test_audit_failure_does_not_raise(self, tmp_path, monkeypatch):
        """Audit logging must never break the agent loop."""
        ws = tmp_path / "engagement"
        (ws / AUDIT_SUBDIR).mkdir(parents=True)
        log = AuditLog(ws)

        # Force an exception during write by replacing log_dir with a file (not dir)
        bad_path = tmp_path / "not_a_dir"
        bad_path.write_text("blocking file")
        log.log_dir = bad_path  # writes to <bad_path>/<date>.jsonl will fail

        # Should NOT raise even though the underlying write fails
        log.log_call("test_tool", args={})
