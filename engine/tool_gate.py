"""G-STEP-style pre-dispatch confidence gate for tool calls.

Adapted from arXiv 2605.00136 ("Are Tools All We Need? Unveiling the
Tool-Use Tax in LLM Agents"). The paper's Factorized Intervention
Framework decomposes a tool call's cost into formatting cost,
protocol overhead, and execution gain — and shows that under semantic
distractors the protocol overhead routinely *exceeds* the gain, so
tool use makes performance worse than no tool use. G-STEP is the
paper's lightweight inference-time gate that suppresses calls likely
to lose to that tax.

Why this lives next to ``tool_validator``: the validator catches
*malformed* calls (wrong shape); this gate catches *unproductive*
calls (correctly-shaped but expected to add no information). They
compose — the gate fires first, the validator second, the function
third.

Confidence model (no logit access required, so the gate is cheap and
loader-agnostic):

1. **Lexical anchor.** Read-only "exploration" tools (``read_file``,
   ``list_dir``, ``glob``, ``grep``) need their key argument
   (``path`` or ``pattern``) to either appear in the goal text or
   appear in some prior tool result. A path the model invents from
   nowhere is the canonical tool-use-tax case.

2. **Exploration cap.** A run of consecutive exploration calls
   without an intervening write/edit/test is a doom-loop signal. The
   gate caps the number of consecutive exploration calls per turn
   window — once the cap is hit, further exploration calls are
   gated until the model takes an action.

The gate is **opt-in** via ``ToolGateConfig(enabled=True)`` and is
intentionally permissive — it never gates write/edit/test/run calls.
The lean 10-tool ULC path already hits 100% on the agentic bench
(see CLAUDE.md → "Bench checker hygiene"); this gate is targeted at
the *extended* 21-tool path that drops to ~86% (per
``feedback_tool_count_regression.md``), where the protocol overhead
on rarely-used tools is the suspected cause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Tools that read state without modifying it. Exploration is fine in
# moderation but is the high-risk surface for tool-use tax: cheap to
# emit, expensive in protocol overhead, often produces no new info.
EXPLORATION_TOOLS: frozenset[str] = frozenset(
    {"read_file", "list_dir", "glob", "grep", "find_definition", "find_references"}
)

# Tools whose execution is the agent making progress on the goal. The
# gate never gates these.
ACTION_TOOLS: frozenset[str] = frozenset(
    {
        "write_file",
        "edit_file",
        "insert_at_line",
        "run_bash",
        "run_tests",
        "run_python",
        "git_commit",
    }
)


@dataclass
class ToolGateConfig:
    """Opt-in configuration for the G-STEP gate."""

    enabled: bool = False
    # Maximum number of consecutive exploration calls before the gate
    # demands the model take action. 4 is conservative — even a goal
    # that requires reading several files before editing usually wraps
    # in 3-4 exploration calls.
    max_consecutive_exploration: int = 4
    # Minimum prefix length when matching tool args against the goal /
    # prior observations. Smaller = more permissive, larger = stricter.
    # 4 chars catches things like "test", "calc", short module names
    # without falsely matching English filler.
    min_anchor_length: int = 4


@dataclass
class GateState:
    """Mutable per-run state. The Agent owns one instance per task run."""

    consecutive_exploration: int = 0
    # Lowercase concatenation of the goal text + every prior tool
    # result. The gate checks tool args against this corpus.
    seen_corpus: str = ""

    def record_observation(self, text: str) -> None:
        """Append a tool result (or any prior observation) to the seen
        corpus so future calls can anchor against it."""
        if text:
            self.seen_corpus += "\n" + text.lower()

    def record_call(self, tool_name: str) -> None:
        if tool_name in EXPLORATION_TOOLS:
            self.consecutive_exploration += 1
        elif tool_name in ACTION_TOOLS:
            # Any action call resets the exploration streak. The model
            # made progress; let it explore again afterwards.
            self.consecutive_exploration = 0
        # Tools outside both sets (e.g. ``remember``, ``plan``) are
        # neither penalized nor reset — they're orthogonal.


@dataclass
class GateDecision:
    """Result of a gate check."""

    allow: bool
    reason: str = ""

    def rejection_message(self, tool_name: str) -> str:
        """Human-and-model-readable rejection. Format mirrors
        tool_validator.format_rejection so the model's recovery prompt
        is consistent across both gates."""
        return (
            f"Tool call rejected: {tool_name}\n"
            f"  Reason: {self.reason}\n"
            f"  This is a confidence gate, not a syntax error — the call is "
            f"well-formed but appears unlikely to make progress. Either "
            f"justify why the call is needed (cite the goal text or a "
            f"prior observation that motivates it), take a write/edit/test "
            f"action, or emit your final answer if the goal is already met."
        )


# Two-stage tokenizer:
#  - _PATH_RE captures whole path-like tokens (so an exact echo of
#    "src/foo.py" wins by direct substring match)
#  - _SEG_RE then splits each path token on path separators so a path
#    can also anchor on any of its segments ("tests/test_x.py" anchors
#    if the corpus mentions "tests/" — the listing format list_dir
#    itself emits)
_PATH_RE = re.compile(r"[A-Za-z0-9_./\\-]+")
_SEG_RE = re.compile(r"[/\\.\-_]+")


def _key_argument(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Return the argument value most worth checking for an anchor.
    None means the gate skips the lexical-anchor check for this tool
    (insufficient signal to judge)."""
    if tool_name in {"read_file", "list_dir"}:
        return arguments.get("path")
    if tool_name == "glob":
        return arguments.get("pattern")
    if tool_name == "grep":
        # grep's pattern is the search term; checking it against the
        # corpus would penalize fishing expeditions, which is fine.
        return arguments.get("pattern")
    if tool_name in {"find_definition", "find_references"}:
        return arguments.get("name") or arguments.get("symbol")
    return None


def _has_lexical_anchor(value: str, corpus: str, min_length: int) -> bool:
    """Return True if any token of *value* of length >= min_length
    appears in *corpus*. Paths like ``a/b/foo.py`` decompose into
    ``a``, ``b``, ``foo``, ``py`` (etc.) and any one of them anchoring
    is enough."""
    if not value or not corpus:
        return False
    value_lower = value.lower()
    # Direct substring win is the common case (model echoed the path
    # verbatim). Bail early.
    if value_lower in corpus:
        return True
    # Path tokens first, then split each on / \ . - _ to surface
    # individual segments that may have anchored independently.
    for path_token in _PATH_RE.findall(value_lower):
        if len(path_token) >= min_length and path_token in corpus:
            return True
        for seg in _SEG_RE.split(path_token):
            if len(seg) >= min_length and seg in corpus:
                return True
    return False


def check(
    tool_name: str,
    arguments: dict[str, Any],
    state: GateState,
    config: ToolGateConfig,
) -> GateDecision:
    """The single entry point. Cheap, no I/O, no model calls."""
    if not config.enabled:
        return GateDecision(allow=True)

    # Action tools are never gated — the gate is about suppressing
    # *unproductive* exploration, not blocking progress.
    if tool_name in ACTION_TOOLS:
        return GateDecision(allow=True)

    # Cap consecutive exploration. Off-by-one note: the call we're
    # about to gate is NOT yet recorded in state, so we compare
    # against >= cap (the next call would push past it).
    if (
        tool_name in EXPLORATION_TOOLS
        and state.consecutive_exploration >= config.max_consecutive_exploration
    ):
        return GateDecision(
            allow=False,
            reason=(
                f"reached the cap of {config.max_consecutive_exploration} "
                "consecutive exploration calls without an intervening write, "
                "edit, or test. This is the doom-loop signal — take a "
                "concrete action with the information already gathered, or "
                "emit your final answer."
            ),
        )

    # Lexical-anchor check on the key argument.
    key_arg = _key_argument(tool_name, arguments)
    if key_arg is not None and isinstance(key_arg, str):
        if not _has_lexical_anchor(key_arg, state.seen_corpus, config.min_anchor_length):
            return GateDecision(
                allow=False,
                reason=(
                    f"argument {key_arg!r} has no lexical anchor in the goal "
                    "text or any prior observation. Either quote the goal "
                    "fragment that motivates this call, or call list_dir / "
                    "glob first to discover what actually exists in the "
                    "workspace before guessing at a path."
                ),
            )

    return GateDecision(allow=True)
