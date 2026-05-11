"""Tests for engine/engagement_scaffold.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.engagement_scaffold import create_engagement, _slugify


class TestSlugify:
    def test_simple(self):
        assert _slugify("Acme Corp") == "acme-corp"

    def test_punctuation_stripped(self):
        assert _slugify("Acme, Inc.") == "acme-inc"

    def test_collapses_dashes(self):
        assert _slugify("foo  bar  baz") == "foo-bar-baz"

    def test_unicode_safe(self):
        # Non-ascii chars become dashes (ASCII-only output)
        assert _slugify("café") == "caf"


class TestCreateEngagement:
    def test_creates_directory_tree(self, tmp_path):
        eng = create_engagement("Acme Corp", parent_dir=tmp_path)
        assert eng == tmp_path / "acme-corp"
        for sub in ("scope", "evidence", "findings", "findings/ARCHIVE",
                    "tools", "audit", "report"):
            assert (eng / sub).is_dir(), f"{sub} missing"

    def test_creates_template_files(self, tmp_path):
        eng = create_engagement("Acme Corp", parent_dir=tmp_path)
        for f in (".ulcagent", "README.md",
                  "scope/sow.md", "scope/targets.txt", "scope/out_of_scope.txt",
                  "findings/findings.json", "report/README.md"):
            assert (eng / f).exists(), f"{f} missing"

    def test_findings_json_initialized_to_empty_array(self, tmp_path):
        eng = create_engagement("Acme Corp", parent_dir=tmp_path)
        data = json.loads((eng / "findings" / "findings.json").read_text(encoding="utf-8"))
        assert data == []

    def test_ulcagent_file_includes_client_name(self, tmp_path):
        eng = create_engagement("Acme Corp", parent_dir=tmp_path)
        ulc = (eng / ".ulcagent").read_text(encoding="utf-8")
        assert "Acme Corp" in ulc
        assert "AUTHORIZED" in ulc
        assert "[aliases]" in ulc

    def test_sow_includes_client_and_date(self, tmp_path):
        eng = create_engagement("Acme Corp", parent_dir=tmp_path,
                                start_date="2026-08-15")
        sow = (eng / "scope" / "sow.md").read_text(encoding="utf-8")
        assert "Acme Corp" in sow
        assert "2026-08-15" in sow

    def test_gitkeeps_present_in_empty_dirs(self, tmp_path):
        eng = create_engagement("Acme Corp", parent_dir=tmp_path)
        for sub in ("evidence", "findings/ARCHIVE", "tools", "audit"):
            assert (eng / sub / ".gitkeep").exists(), f"missing .gitkeep in {sub}"

    def test_existing_dir_raises(self, tmp_path):
        create_engagement("Acme Corp", parent_dir=tmp_path)
        with pytest.raises(FileExistsError):
            create_engagement("Acme Corp", parent_dir=tmp_path)

    def test_empty_client_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="required"):
            create_engagement("", parent_dir=tmp_path)
        with pytest.raises(ValueError, match="required"):
            create_engagement("   ", parent_dir=tmp_path)

    def test_unparseable_client_rejected(self, tmp_path):
        # Pure punctuation produces an empty slug
        with pytest.raises(ValueError, match="slug"):
            create_engagement("...", parent_dir=tmp_path)

    def test_audit_dir_is_present_for_audit_log(self, tmp_path):
        """Engagement scaffold must create the audit dir so AuditLog.for_workspace
        auto-engages inside it."""
        from engine.audit_log import AuditLog
        eng = create_engagement("Acme Corp", parent_dir=tmp_path)
        assert AuditLog.for_workspace(eng) is not None
