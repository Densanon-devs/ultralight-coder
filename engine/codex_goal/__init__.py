"""Codex-style /goal persistent objective loop.

Adapted from OpenAI Codex CLI 0.128.0's /goal pattern. The loop wraps
`agent.run()`: after each final answer, a continuation prompt asks the
model whether the goal is fully achieved. If not, the loop re-enters
the agent with a continue_session prompt that nudges the model to pick
the next concrete sub-task. Token budget is the single hard exit.

The loop logic is implemented entirely as prompt strings — no special
runtime state. See `prompts.py`.
"""

from .loop import GoalResult, run_goal_loop

__all__ = ["GoalResult", "run_goal_loop"]
