"""
Tests for engine.destructive_command_gate + the Agent._execute_call
hook that wires it in. Covers the pattern matcher in isolation and the
end-to-end Agent dispatch behavior (fail-safe denial, callback
approve/deny, non-bash-tool no-op, non-destructive-bash no-op).
"""

from __future__ import annotations

import pytest

from engine.agent import Agent
from engine.agent_tools import ToolCall, ToolRegistry, ToolResult, ToolSchema
from engine.destructive_command_gate import (
    DestructiveMatch,
    check_command,
    format_warning,
)


# ── Pattern matcher: true positives ─────────────────────────────


@pytest.mark.parametrize(
    "command, expected_id",
    [
        # The canonical May 2026 incident command (with stray ~/)
        ("rm -rf tests/ patches/ plan/ ~/", "rm_recursive"),
        ("rm -rf tests/ patches/ plan/ ~/", "rm_force"),
        # Unix recursive remove variants
        ("rm -r /tmp/foo", "rm_recursive"),
        ("rm -R /tmp/foo", "rm_recursive"),
        ("rm --recursive /tmp/foo", "rm_recursive"),
        ("rm -rf /tmp/foo", "rm_force"),
        ("rm -f important.lock", "rm_force"),
        # PowerShell
        ("Remove-Item -Recurse C:\\foo", "remove_item_recurse"),
        ("Remove-Item -Recurse -Force foo", "remove_item_recurse"),
        ("Remove-Item -Force foo", "remove_item_force"),
        # Windows cmd
        ("del /s /q .", "del_recursive"),
        ("del /S /Q *.tmp", "del_recursive"),
        ("rd /s /q build", "rd_recursive"),
        ("rmdir /s /q dist", "rd_recursive"),
        # Formatting / wiping
        ("format C:", "format_drive"),
        ("FORMAT D: /q", "format_drive"),
        ("mkfs.ext4 /dev/sda1", "mkfs"),
        ("shred -uvz important.txt", "shred"),
        ("dd if=/dev/zero of=/dev/sda bs=1M", "dd_to_device"),
        # find variants
        ("find . -name '*.tmp' -delete", "find_delete"),
        ("find /var -mtime +30 -exec rm {} \\;", "find_exec_rm"),
        # git destructive
        ("git reset --hard origin/main", "git_reset_hard"),
        ("git clean -fd", "git_clean_force"),
        ("git clean --force", "git_clean_force"),
        ("git push --force origin main", "git_push_force"),
        ("git push -f origin main", "git_push_force"),
        ("git checkout .", "git_checkout_dot"),
        ("git restore .", "git_checkout_dot"),
        ("git branch -D feature-x", "git_branch_delete"),
        # SQL
        ("DROP TABLE users", "sql_drop"),
        ("drop database production", "sql_drop"),
        ("TRUNCATE TABLE audit_log", "sql_truncate"),
        ("DELETE FROM users", "sql_unscoped_delete"),
        ("UPDATE users SET role='admin'", "sql_unscoped_delete"),
        # Process / OS
        ("kill -9 1234", "kill_all"),
        ("killall node", "kill_all"),
        ("pkill -9 -f server", "kill_all"),
        ("shutdown -h now", "shutdown_reboot"),
        ("reboot", "shutdown_reboot"),
        ("poweroff", "shutdown_reboot"),
    ],
)
def test_destructive_pattern_matches(command, expected_id):
    matches = check_command(command)
    matched_ids = {m.pattern_id for m in matches}
    assert expected_id in matched_ids, (
        f"command {command!r} should have matched {expected_id}, "
        f"matched: {matched_ids}"
    )


# ── Pattern matcher: true negatives (no false positives) ─────────


@pytest.mark.parametrize(
    "command",
    [
        # Safe single-file removes
        "rm foo.txt",
        "rm temp.log",
        # Reading or copying with home expansion (NOT destructive)
        "cat ~/.bashrc",
        "cp foo.txt ~/Documents/",
        "echo $HOME",
        "ls ~/",
        # Listing system directories
        "ls /etc/hosts",
        "cat /var/log/syslog",
        "ls C:\\Windows",
        # Safe git operations
        'git commit -m "fix bug"',
        "git status",
        "git diff",
        "git push origin feature",
        "git checkout main",
        "git checkout -b feature/foo",
        "git reset HEAD~1",  # soft reset is safe
        "git clean -n",  # dry run, no -f
        # Safe SQL
        "DELETE FROM users WHERE id = 5",
        "UPDATE users SET name='alice' WHERE id=5",
        "SELECT * FROM users",
        "INSERT INTO users (name) VALUES ('alice')",
        # Process inspection (not killing)
        "ps aux",
        "kill 1234",  # graceful kill, no -9
        # find variants without -delete or -exec rm
        "find . -name '*.py'",
        "find . -type f -print",
        # Common build commands
        "pytest -v",
        "npm install",
        "python -m pytest",
        "go build ./...",
        "cargo build --release",
        # Edge cases
        "",
        "echo hello",
        "# rm -rf is in a comment",  # NOTE: still matches — see test below
    ],
)
def test_safe_commands_do_not_match(command):
    matches = check_command(command)
    # The "comment" case is an intentional false positive — pattern
    # matchers don't parse shell syntax. We flag this explicitly
    # in the dedicated test below and accept the false-positive cost
    # for the safety win.
    if command == "# rm -rf is in a comment":
        return
    assert matches == [], (
        f"command {command!r} should be safe, but matched: "
        f"{[m.pattern_id for m in matches]}"
    )


def test_destructive_match_in_comment_is_known_false_positive():
    """We don't parse shell syntax — a destructive pattern in a comment
    still matches. Documented as an accepted tradeoff."""
    matches = check_command("# rm -rf is in a comment")
    assert any(m.pattern_id in {"rm_recursive", "rm_force"} for m in matches)


# ── Multi-pattern hits ───────────────────────────────────────────


def test_canonical_incident_matches_multiple_patterns():
    """The May 2026 r/ClaudeAI incident: `rm -rf tests/ patches/ plan/ ~/`.
    Should match at minimum rm_recursive AND rm_force.
    """
    matches = check_command("rm -rf tests/ patches/ plan/ ~/")
    matched_ids = {m.pattern_id for m in matches}
    assert "rm_recursive" in matched_ids
    assert "rm_force" in matched_ids


# ── Type safety / null safety ────────────────────────────────────


def test_empty_string_returns_empty():
    assert check_command("") == []


def test_none_returns_empty():
    # check_command should tolerate non-string input gracefully
    assert check_command(None) == []  # type: ignore[arg-type]


def test_format_warning_includes_all_matches():
    cmd = "rm -rf /tmp/foo"
    matches = check_command(cmd)
    warning = format_warning(cmd, matches)
    assert "DESTRUCTIVE COMMAND DETECTED" in warning
    assert cmd in warning
    for m in matches:
        assert m.pattern_id in warning
        assert m.description in warning


# ── End-to-end: Agent._execute_call gate behavior ────────────────


def _make_test_registry() -> ToolRegistry:
    """Build a registry with a stubbed run_bash that records calls
    instead of actually executing subprocess."""
    reg = ToolRegistry()
    executed: list[str] = []

    def fake_bash(command: str, **kwargs):
        executed.append(command)
        return {"stdout": "ok", "stderr": "", "exit_code": 0}

    reg.register(
        ToolSchema(
            name="run_bash",
            description="Run a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "cwd": {"type": "string"},
                },
                "required": ["command"],
            },
            function=fake_bash,
            risky=True,
        )
    )
    # Attach the recorder so tests can inspect what actually ran
    reg._executed = executed  # type: ignore[attr-defined]
    return reg


class _StubModel:
    """Minimal model stub. Agent._execute_call doesn't touch the model,
    so this only needs to exist to satisfy Agent.__init__."""

    def generate(self, prompt, max_tokens, stop=None, **kwargs):
        return ""


def _bash_call(command: str) -> ToolCall:
    """Build a minimal ToolCall for run_bash. The `raw` field is unused
    by _execute_call so we pass a placeholder."""
    return ToolCall(
        name="run_bash",
        arguments={"command": command},
        raw=f"<tool_call>{{...}}</tool_call>",
    )


def _agent(
    registry: ToolRegistry,
    *,
    confirm_risky=None,
    confirm_destructive=None,
) -> Agent:
    return Agent(
        model=_StubModel(),
        registry=registry,
        confirm_risky=confirm_risky,
        confirm_destructive=confirm_destructive,
    )


def test_safe_bash_command_executes_normally():
    reg = _make_test_registry()
    agent = _agent(reg, confirm_risky=lambda _c: True)
    call = _bash_call("ls -la")
    result = agent._execute_call(call)
    assert result.success, result.error
    assert reg._executed == ["ls -la"]  # type: ignore[attr-defined]


def test_destructive_command_refused_with_no_callback():
    """Fail-safe: if confirm_destructive is None, destructive commands
    are refused outright. This is the critical invariant — even with
    auto-approve enabled (confirm_risky always True), the gate still
    refuses.
    """
    reg = _make_test_registry()
    agent = _agent(reg, confirm_risky=lambda _c: True, confirm_destructive=None)
    call = _bash_call("rm -rf /tmp/foo")
    result = agent._execute_call(call)
    assert not result.success
    assert "Destructive shell command refused" in result.error
    assert "rm_recursive" in result.error
    # The subprocess stub must NOT have been called
    assert reg._executed == []  # type: ignore[attr-defined]


def test_destructive_command_refused_when_callback_denies():
    reg = _make_test_registry()
    denies = lambda _c, _m: False
    agent = _agent(reg, confirm_risky=lambda _c: True, confirm_destructive=denies)
    call = _bash_call("rm -rf /tmp/foo")
    result = agent._execute_call(call)
    assert not result.success
    assert "denied destructive command" in result.error.lower()
    assert reg._executed == []  # type: ignore[attr-defined]


def test_destructive_command_executes_when_callback_approves():
    reg = _make_test_registry()
    approves = lambda _c, _m: True
    agent = _agent(reg, confirm_risky=lambda _c: True, confirm_destructive=approves)
    call = _bash_call("rm -rf /tmp/foo")
    result = agent._execute_call(call)
    assert result.success, result.error
    assert reg._executed == ["rm -rf /tmp/foo"]  # type: ignore[attr-defined]


def test_callback_raising_treats_as_denied():
    reg = _make_test_registry()

    def boom(_c, _m):
        raise RuntimeError("oops")

    agent = _agent(reg, confirm_risky=lambda _c: True, confirm_destructive=boom)
    call = _bash_call("rm -rf /tmp/foo")
    result = agent._execute_call(call)
    assert not result.success
    assert reg._executed == []  # type: ignore[attr-defined]


def test_auto_approve_risky_does_not_bypass_destructive_gate():
    """The core security invariant. confirm_risky=(lambda _c: True) is
    how `--yes` / auto-approve mode is implemented. The destructive
    gate MUST still fire even when confirm_risky rubber-stamps.
    """
    reg = _make_test_registry()
    # confirm_risky auto-approves (simulating --yes), but
    # confirm_destructive is None → fail-safe refusal.
    agent = _agent(reg, confirm_risky=lambda _c: True, confirm_destructive=None)
    call = _bash_call("rm -rf ~/")
    result = agent._execute_call(call)
    assert not result.success
    assert "Destructive shell command refused" in result.error
    assert reg._executed == []  # type: ignore[attr-defined]


def test_non_bash_tools_skip_destructive_gate():
    """The gate only applies to run_bash. Other tools (read_file,
    write_file, etc.) are not gated even if their arguments happen to
    contain destructive-looking strings (e.g. writing a file whose
    contents mention `rm -rf`)."""
    reg = ToolRegistry()
    captured = {}

    def fake_write(path: str, content: str, **kwargs):
        captured["path"] = path
        captured["content"] = content
        return {"ok": True}

    reg.register(
        ToolSchema(
            name="write_file",
            description="Write a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            function=fake_write,
            risky=False,
        )
    )
    agent = _agent(reg)
    call = ToolCall(
        name="write_file",
        arguments={"path": "notes.md", "content": "Don't run `rm -rf` ever"},
        raw="<tool_call>{}</tool_call>",
    )
    result = agent._execute_call(call)
    assert result.success, result.error
    assert captured["path"] == "notes.md"


def test_missing_command_argument_is_safe():
    """If the model omits the `command` arg, the gate should not crash."""
    reg = _make_test_registry()
    agent = _agent(reg, confirm_risky=lambda _c: True)
    call = ToolCall(
        name="run_bash", arguments={}, raw="<tool_call>{}</tool_call>"
    )  # malformed
    # The gate sees command="" and lets it through; downstream
    # registry execution may itself fail validation, but that's a
    # separate concern. We're only asserting the gate doesn't crash.
    result = agent._execute_call(call)
    # Either the gate passed it through (success or downstream error),
    # but no exception escapes _execute_call.
    assert result is not None
