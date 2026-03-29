"""
Shared Embedder Pool — Single sentence-transformer instance.

Before: Memory, classifier, and micro-adapters each loaded their own
sentence-transformer model = 3x RAM, 3x startup time (~10s total).

Now: All components share one instance via get_embedder().
Cuts startup to ~4s and saves ~500MB RAM.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_embedders: dict[str, object] = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get or create a shared sentence-transformer instance.
    Thread-safe. Returns the same instance for the same model name.
    """
    with _lock:
        if model_name in _embedders:
            return _embedders[model_name]

        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(model_name)
            _embedders[model_name] = embedder
            logger.info(f"Loaded shared embedder: {model_name}")
            return embedder
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return None
