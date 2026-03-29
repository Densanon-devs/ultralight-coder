"""
KV Cache Manager — Phase 3 Performance

Optimizes inference by tracking and reusing the KV (Key-Value) cache
across conversation turns.

The Problem:
    Every turn, the model processes the ENTIRE prompt from scratch:
    system prompt + module context + memory + conversation + user input.
    But the system prompt and module context rarely change between turns.
    Re-processing them wastes ~30-50% of inference time.

The Solution:
    Track which prefix of the prompt is identical to the last turn.
    Tell the model to reuse the KV cache for that prefix and only
    process the new/changed tokens.

How it works with llama-cpp-python:
    llama-cpp-python automatically reuses KV cache when you call
    model() with a prompt that shares a prefix with the previous call.
    But it only works if the prefix is EXACTLY identical byte-for-byte.

    This module ensures that:
    1. The "stable prefix" (system + modules) is assembled identically
       each turn when the active modules haven't changed
    2. We track prefix length so the fusion layer can optimize assembly
    3. We detect when the cache should be invalidated (module change,
       LoRA swap, etc.)
    4. We report cache hit rates for monitoring

Context window management:
    As conversation grows, we need to manage the context window.
    This module tracks token usage and signals when the conversation
    should be compressed or truncated.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheState:
    """Current state of the KV cache."""
    # The prefix that is currently cached
    cached_prefix: str = ""
    cached_prefix_hash: str = ""
    cached_prefix_tokens: int = 0

    # What produced this prefix
    active_modules: list[str] = field(default_factory=list)
    active_lora: Optional[str] = None

    # Stats
    last_prompt_tokens: int = 0
    last_new_tokens: int = 0        # Tokens that needed processing (not cached)

    # Timestamp
    created_at: float = 0.0


class KVCacheManager:
    """
    Manages KV cache reuse across conversation turns.

    The manager doesn't directly manipulate the KV cache (that's handled
    by llama-cpp-python internally). Instead, it ensures prompt assembly
    produces identical prefixes when possible, and tracks metrics.

    Usage:
        kv = KVCacheManager(context_length=2048)

        # Before assembly
        prefix = kv.get_stable_prefix(system_prompt, module_context)
        hit = kv.would_hit(prefix)

        # After generation
        kv.update(full_prompt, active_modules, active_lora)

        # Check if context window is getting full
        if kv.should_compress():
            # Trigger conversation compression
    """

    def __init__(self, context_length: int = 2048, max_generation_tokens: int = 512):
        self.context_length = context_length
        self.max_generation_tokens = max_generation_tokens
        self._state = CacheState()
        self._token_counter = lambda text: len(text) // 4  # Default estimate

        # Stats
        self._total_turns = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_tokens_saved = 0
        self._invalidation_reasons: list[str] = []

    def set_token_counter(self, counter):
        """Set the token counting function (from BaseModel)."""
        self._token_counter = counter

    # ── Prefix Management ─────────────────────────────────────

    def build_stable_prefix(
        self,
        system_prompt: str,
        module_context: str = "",
        system_knowledge: str = "",
    ) -> str:
        """
        Build the "stable prefix" — the part of the prompt that
        doesn't change between turns (when modules stay the same).

        This should be called by the fusion layer to ensure byte-exact
        prefix matching for KV cache reuse.
        """
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        if system_knowledge:
            parts.append(system_knowledge)
        if module_context:
            parts.append(module_context)

        return "\n\n".join(parts)

    def would_hit(
        self,
        prefix: str,
        active_modules: list[str],
        active_lora: Optional[str] = None,
    ) -> bool:
        """
        Check if using this prefix would result in a KV cache hit.

        A hit means the model can skip re-processing the prefix tokens.
        """
        prefix_hash = self._hash(prefix)

        if (prefix_hash == self._state.cached_prefix_hash
                and active_modules == self._state.active_modules
                and active_lora == self._state.active_lora):
            return True

        return False

    def update(
        self,
        full_prompt: str,
        prefix: str,
        active_modules: list[str],
        active_lora: Optional[str] = None,
    ):
        """
        Update cache state after a generation.
        Call this after every model inference.
        """
        self._total_turns += 1
        prefix_hash = self._hash(prefix)
        prefix_tokens = self._token_counter(prefix)
        full_tokens = self._token_counter(full_prompt)

        was_hit = (prefix_hash == self._state.cached_prefix_hash
                   and active_modules == self._state.active_modules
                   and active_lora == self._state.active_lora)

        if was_hit:
            self._cache_hits += 1
            new_tokens = full_tokens - prefix_tokens
            self._total_tokens_saved += prefix_tokens
            logger.debug(
                f"KV cache HIT: reused {prefix_tokens} tokens, "
                f"processed {new_tokens} new tokens"
            )
        else:
            self._cache_misses += 1
            new_tokens = full_tokens

            # Log reason for miss
            reasons = []
            if prefix_hash != self._state.cached_prefix_hash:
                reasons.append("prefix_changed")
            if active_modules != self._state.active_modules:
                reasons.append(f"modules_changed({self._state.active_modules} → {active_modules})")
            if active_lora != self._state.active_lora:
                reasons.append(f"lora_changed({self._state.active_lora} → {active_lora})")

            reason_str = ", ".join(reasons) if reasons else "first_turn"
            self._invalidation_reasons.append(reason_str)
            logger.debug(f"KV cache MISS: {reason_str} — processing all {full_tokens} tokens")

        # Update state
        self._state = CacheState(
            cached_prefix=prefix,
            cached_prefix_hash=prefix_hash,
            cached_prefix_tokens=prefix_tokens,
            active_modules=list(active_modules),
            active_lora=active_lora,
            last_prompt_tokens=full_tokens,
            last_new_tokens=new_tokens,
            created_at=time.time(),
        )

    def invalidate(self, reason: str = "manual"):
        """Force-invalidate the cache (e.g., after /clear or mode switch)."""
        logger.info(f"KV cache invalidated: {reason}")
        self._invalidation_reasons.append(reason)
        self._state = CacheState()

    # ── Context Window Management ─────────────────────────────

    def available_tokens(self) -> int:
        """How many tokens are available for prompt + generation."""
        return self.context_length

    def prompt_budget(self) -> int:
        """Max tokens for the prompt (context - generation reserve)."""
        return self.context_length - self.max_generation_tokens

    def should_compress(self, current_prompt_tokens: Optional[int] = None) -> bool:
        """
        Check if the conversation should be compressed/truncated.

        Returns True if the prompt is using >80% of available budget.
        The engine should respond by summarizing older conversation
        turns into long-term memory.
        """
        tokens = current_prompt_tokens or self._state.last_prompt_tokens
        budget = self.prompt_budget()
        usage_ratio = tokens / budget if budget > 0 else 1.0
        return usage_ratio > 0.80

    def context_pressure(self) -> float:
        """
        Return context window pressure as 0.0-1.0.
        0.0 = plenty of room, 1.0 = context window full.
        """
        budget = self.prompt_budget()
        if budget <= 0:
            return 1.0
        return min(1.0, self._state.last_prompt_tokens / budget)

    # ── Stats ─────────────────────────────────────────────────

    def hit_rate(self) -> float:
        """Cache hit rate as 0.0-1.0."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def tokens_saved(self) -> int:
        """Total tokens saved by cache hits."""
        return self._total_tokens_saved

    def status(self) -> dict:
        """Full cache status for display."""
        total = self._cache_hits + self._cache_misses
        return {
            "turns": self._total_turns,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": f"{self.hit_rate():.1%}",
            "tokens_saved": self._total_tokens_saved,
            "cached_prefix_tokens": self._state.cached_prefix_tokens,
            "last_prompt_tokens": self._state.last_prompt_tokens,
            "context_pressure": f"{self.context_pressure():.1%}",
            "active_modules": self._state.active_modules,
            "active_lora": self._state.active_lora,
            "recent_invalidations": self._invalidation_reasons[-5:],
        }

    def _hash(self, text: str) -> str:
        """Fast hash for prefix comparison."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
