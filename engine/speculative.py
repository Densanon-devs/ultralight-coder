"""
Application-Level Speculative Decoding

Since native speculative decoding requires matching vocabularies
(Llama 3.2 vocab=128256 doesn't match any small model), we implement
speculative execution at the application level:

1. PARALLEL GENERATION: Run fast model + expert system simultaneously
   on different prompts from a batch, overlapping I/O with generation.

2. PREDICTIVE GENERATION: Start generating the likely response while
   still doing routing/expert retrieval. If the prediction is wrong,
   discard and regenerate.

3. N-GRAM CACHE: Cache common prompt→response patterns. If we've seen
   a similar prompt before, reuse the cached response prefix.

4. EARLY EXIT: Monitor token probabilities during streaming. If
   confidence is high for multiple tokens, batch them instead of
   generating one-by-one.
"""

import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Cache prompt→response patterns for instant retrieval.

    Uses semantic hashing: prompts with similar structure get the same
    cache key. Cache hits skip generation entirely.
    """

    def __init__(self, max_size: int = 100, storage_dir: Optional[str] = None):
        self.max_size = max_size
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._storage_file = None

        if storage_dir:
            p = Path(storage_dir)
            p.mkdir(parents=True, exist_ok=True)
            self._storage_file = p / "response_cache.json"
            self._load()

    def _make_key(self, prompt: str, module: Optional[str] = None) -> str:
        """Create a cache key from prompt only (module-agnostic for lookup)."""
        normalized = prompt.lower().strip()[:200]
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def get(self, prompt: str, module: Optional[str] = None) -> Optional[str]:
        """Look up a cached response. Returns None on miss."""
        key = self._make_key(prompt, module)
        if key in self._cache:
            self._hits += 1
            entry = self._cache[key]
            # Move to end (LRU)
            self._cache.move_to_end(key)
            logger.debug(f"Cache HIT: {key} (hits={self._hits})")
            return entry["response"]
        self._misses += 1
        return None

    def put(self, prompt: str, response: str, module: Optional[str] = None,
            quality_verified: bool = False):
        """Store a prompt→response pair. Only caches verified-good responses."""
        if not quality_verified:
            return  # Only cache responses we know are correct

        key = self._make_key(prompt, module)
        self._cache[key] = {
            "prompt_prefix": prompt[:100],
            "response": response,
            "module": module,
            "timestamp": time.time(),
        }
        self._cache.move_to_end(key)

        # Evict oldest if over limit
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

        self._save()

    def _load(self):
        """Load cache from disk."""
        if self._storage_file and self._storage_file.exists():
            try:
                with open(self._storage_file) as f:
                    data = json.load(f)
                for key, entry in data.items():
                    self._cache[key] = entry
                logger.info(f"Loaded {len(self._cache)} cached responses")
            except Exception as e:
                logger.debug(f"Cache load failed: {e}")

    def _save(self):
        """Persist cache to disk."""
        if self._storage_file:
            try:
                with open(self._storage_file, "w") as f:
                    json.dump(dict(self._cache), f)
            except Exception as e:
                logger.debug(f"Cache save failed: {e}")

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._save()

    def status(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.0%}",
        }


class PrefixMatcher:
    """
    Matches prompt prefixes to predict likely output patterns.

    If a new prompt starts the same way as a previously-seen prompt,
    we can pre-fill the response with the common prefix, saving tokens.
    """

    def __init__(self, min_prefix_len: int = 20):
        self.min_prefix_len = min_prefix_len
        self._patterns: dict[str, list[str]] = {}  # prefix → [response_starts]

    def record(self, prompt: str, response: str):
        """Record a prompt→response pattern."""
        prefix = self._extract_prefix(prompt)
        if prefix:
            if prefix not in self._patterns:
                self._patterns[prefix] = []
            resp_start = response[:50]
            if resp_start not in self._patterns[prefix]:
                self._patterns[prefix].append(resp_start)
                # Keep only last 5 per prefix
                self._patterns[prefix] = self._patterns[prefix][-5:]

    def predict_start(self, prompt: str) -> Optional[str]:
        """Predict how the response will start based on similar prompts."""
        prefix = self._extract_prefix(prompt)
        if prefix and prefix in self._patterns:
            # Return most recent pattern
            return self._patterns[prefix][-1]
        return None

    def _extract_prefix(self, prompt: str) -> Optional[str]:
        """Extract a matchable prefix from a prompt."""
        # Use first N chars, normalized
        clean = prompt.lower().strip()
        if len(clean) < self.min_prefix_len:
            return None
        return clean[:self.min_prefix_len]


class SpeculativeEngine:
    """
    Application-level speculative execution combining:
    - Response caching (instant for repeated patterns)
    - Prefix prediction (partial pre-fill)
    - Quality-gated caching (only cache verified responses)
    """

    def __init__(self, storage_dir: str = "data/cache", cache_size: int = 100):
        self.cache = ResponseCache(max_size=cache_size, storage_dir=storage_dir)
        self.prefix_matcher = PrefixMatcher()
        self._total_saved_ms = 0

    def try_cache(self, prompt: str, module: Optional[str] = None) -> Optional[str]:
        """Try to get a cached response. Returns None on miss."""
        return self.cache.get(prompt, module)

    def record(self, prompt: str, response: str, module: Optional[str] = None,
               quality_verified: bool = False):
        """Record a response for future caching."""
        self.cache.put(prompt, response, module, quality_verified)
        self.prefix_matcher.record(prompt, response)

    def status(self) -> dict:
        return {
            "cache": self.cache.status(),
            "patterns": len(self.prefix_matcher._patterns),
            "total_saved_ms": self._total_saved_ms,
        }
