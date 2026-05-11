"""Tests for engine/agent_augmentor.py — lightweight keyword-matched
augmentor injection for the agent path.

The Phase 13 baseline (28/28 on Python codegen with agent path bypassing
augmentors) MUST be preserved. The key invariant: plain-Python goals with
no agentic-domain keyword match return empty injection.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.agent_augmentor import build_agent_augmentor


@pytest.fixture
def repo_examples_dir():
    """Path to the real augmentor library in this repo."""
    return Path(__file__).parent / "data" / "augmentor_examples"


class TestNoMatchReturnsEmpty:
    """Phase 13 baseline preservation — the cornerstone invariant."""

    def test_plain_python_codegen_returns_empty(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        assert aug("Write a function that reverses a string") == ""

    def test_local_rename_returns_empty(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        assert aug("Rename function compute to compute_total in calculator.py") == ""

    def test_fix_syntax_error_returns_empty(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        assert aug("There's a Python syntax error in broken.py. Find and fix it.") == ""

    def test_refactor_returns_empty(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        assert aug("Refactor the for-loop in process.py to use a list comprehension") == ""

    def test_empty_goal_returns_empty(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        assert aug("") == ""
        assert aug("   ") == ""


class TestSecurityGoalsFire:
    def test_arp_scan_triggers(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        out = aug("Build an ARP scanner using scapy that finds all hosts on /24")
        assert out, "expected non-empty injection for ARP/scapy goal"
        assert "network_arp_scan" in out
        assert "scapy" in out.lower()

    def test_port_scan_triggers(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        out = aug("Write a port scanner that does TCP connect against the top 100 ports")
        assert out
        assert "port_scan" in out.lower()

    def test_mifare_dump_triggers(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        out = aug("Build a MIFARE Classic dumper using nfcpy")
        assert out
        assert "mifare" in out.lower() or "nfcpy" in out.lower()

    def test_pentest_keyword_triggers(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        out = aug("Write a pentest tool that does port scanning and reports findings")
        assert out, "pentest + port scan should match"


class TestWebResearchGoalsFire:
    def test_web_search_keyword_triggers(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        out = aug("Use web_search and fetch_url to find docs and write demo.py")
        assert out
        assert "web_research_to_file" in out

    def test_look_up_phrase_triggers(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        out = aug("Look up the latest pydantic v2 syntax and save to pydantic_demo.py")
        assert out
        assert "web_research_to_file" in out


class TestMaxExamplesCap:
    def test_respects_max_examples(self, repo_examples_dir):
        aug = build_agent_augmentor(examples_dir=repo_examples_dir, max_examples=1)
        out = aug("Build an ARP scanner using scapy AND a port scanner")
        # At most 1 ### Pattern: heading
        assert out.count("### Pattern:") <= 1


class TestFailureSafety:
    def test_missing_examples_dir_returns_empty(self, tmp_path):
        """If the examples dir doesn't exist, augmentor returns empty (no crash)."""
        aug = build_agent_augmentor(examples_dir=tmp_path / "nonexistent")
        assert aug("Build an ARP scanner using scapy") == ""

    def test_empty_examples_dir_returns_empty(self, tmp_path):
        """Empty (but existing) dir returns empty."""
        empty_root = tmp_path / "data" / "augmentor_examples"
        empty_root.mkdir(parents=True)
        aug = build_agent_augmentor(examples_dir=empty_root)
        # Even though keyword matches, no examples available — return empty
        assert aug("Build an ARP scanner using scapy") == ""


class TestCacheReuse:
    def test_repeated_calls_use_cache(self, repo_examples_dir):
        """Two calls with same goal should be deterministic (cache works)."""
        aug = build_agent_augmentor(examples_dir=repo_examples_dir)
        out1 = aug("Build an ARP scanner using scapy")
        out2 = aug("Build an ARP scanner using scapy")
        assert out1 == out2
        assert out1, "expected non-empty"
