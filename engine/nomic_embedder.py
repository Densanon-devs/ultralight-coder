"""nomic-embed-text-v2-moe adapter for ulcagent's project indexing.

Adapted from AMD GAIA 0.17.5 — they use ``nomic-embed-text-v2-moe-GGUF``
for FAISS-backed semantic code search and validated it on a real
codebase (973 files → 24,349 chunks in the GAIA repo itself). GGUF
format means it loads via llama-cpp-python (already a dependency for
ulcagent's coder model), so adopting it here doesn't add a new
runtime requirement.

Why prefer nomic over the current sentence-transformer baseline:
- Quality at 8K context (vs 512 for `all-MiniLM-L6-v2`) — long code
  blocks no longer truncate in indexing.
- Code-aware contrastive pre-training pulls semantically related
  symbols together better than the general-purpose MiniLM.
- Validated at real codebase scale (24K+ chunks) by AMD.
- Stays in-process, GGUF-format, llama-cpp-python compatible — no
  PyTorch dependency, no torch CUDA contention with the coder model.

API contract: this adapter exposes the *same* surface
``ProjectIndex.init_embedder`` already calls — ``.encode(list[str],
normalize_embeddings: bool, show_progress_bar: bool, batch_size: int)
-> np.ndarray`` — so swapping it in is config-only, no caller changes.

Critical detail for nomic models: queries and indexed passages take
*different* role prefixes (``search_query: ...`` vs ``search_document:
...``) per the model's contrastive training. The adapter applies the
correct prefix automatically based on whether the call is going
through the index path (passage) or the search path (query).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Default model file. Override via config.
DEFAULT_MODEL_FILENAME = "nomic-embed-text-v2-moe.Q5_K_M.gguf"

# Nomic role prefixes from the model card.
# https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-gguf
QUERY_PREFIX = "search_query: "
PASSAGE_PREFIX = "search_document: "


class NomicGGUFEmbedder:
    """Loader-agnostic embedder over nomic-embed-text-v2-moe-GGUF.

    Wraps llama-cpp-python's ``Llama(... embedding=True)`` to produce
    normalized 768-d embeddings on a list of strings.

    Defers the heavyweight import of ``llama_cpp`` until ``load()`` is
    called — keeps test imports light and lets the project_context
    module import this adapter without forcing the GGUF model into
    memory at module load.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        seed: int = 0,
        verbose: bool = False,
        # Test seam: callers (or unit tests) may inject a pre-built
        # llama-cpp-python instance to avoid loading the real GGUF.
        # The injected object only needs to expose ``embed(text: str)
        # -> list[float] | list[list[float]]``.
        _llm_factory=None,
    ) -> None:
        self.model_path = str(model_path)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_threads = n_threads
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self._llm_factory = _llm_factory
        self._llm = None
        self._dim: Optional[int] = None
        # role pulled from the surrounding context — see use_passage_role()
        # / use_query_role(). Defaults to passage since indexing is the
        # heavy/slow path.
        self._role_prefix: str = PASSAGE_PREFIX

    # ── Lifecycle ────────────────────────────────────────────────

    def load(self) -> None:
        """Construct the underlying llama-cpp-python instance.

        Idempotent. The first call pays the model-load cost; later
        calls are no-ops.
        """
        if self._llm is not None:
            return

        if self._llm_factory is not None:
            # Test path or custom-loader injection.
            self._llm = self._llm_factory()
        else:
            # Real path: import lazily so test runs that mock the
            # adapter don't pay the llama-cpp-python import cost.
            from llama_cpp import Llama
            kwargs = {
                "model_path": self.model_path,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "embedding": True,
                "seed": self.seed,
                "verbose": self.verbose,
            }
            if self.n_threads is not None:
                kwargs["n_threads"] = int(self.n_threads)
            self._llm = Llama(**kwargs)

        # Probe one embedding to fix the output dimension.
        probe = self._llm.embed("nomic-embed-text-v2-moe probe")
        if isinstance(probe, list) and probe and isinstance(probe[0], list):
            # Some llama-cpp-python builds return list[list[float]]
            # for embed() even on a single string.
            probe_vec = probe[0]
        else:
            probe_vec = probe
        self._dim = len(probe_vec)
        logger.info(
            "Loaded nomic embedder: dim=%d, n_ctx=%d, gpu_layers=%d",
            self._dim, self.n_ctx, self.n_gpu_layers,
        )

    # ── Role switch ──────────────────────────────────────────────

    def use_passage_role(self) -> None:
        """All subsequent encode() calls treat inputs as indexed passages."""
        self._role_prefix = PASSAGE_PREFIX

    def use_query_role(self) -> None:
        """All subsequent encode() calls treat inputs as search queries."""
        self._role_prefix = QUERY_PREFIX

    # ── Encode ───────────────────────────────────────────────────

    def encode(
        self,
        texts: Iterable[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Embed *texts*. Mirrors the sentence-transformers API used by
        ProjectIndex so it's a drop-in.

        ``batch_size`` is honored as a hint — llama-cpp-python's embed()
        is single-input today, so we loop. Kept as a parameter to
        preserve the call-site signature.
        """
        if self._llm is None:
            self.load()

        items = list(texts)
        n = len(items)
        if n == 0:
            return np.zeros((0, self._dim or 0), dtype=np.float32)

        prefix = self._role_prefix
        out = np.empty((n, self._dim), dtype=np.float32)
        for i, text in enumerate(items):
            payload = prefix + (text or "")
            vec = self._llm.embed(payload)
            # llama-cpp-python may return list[float] OR list[list[float]]
            # depending on build; normalize either to a 1-D array.
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[0]
            out[i] = arr

        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            # Avoid divide-by-zero on degenerate (all-zero) embeddings.
            norms = np.where(norms == 0.0, 1.0, norms)
            out = out / norms

        return out

    # ── Introspection ────────────────────────────────────────────

    @property
    def dimension(self) -> Optional[int]:
        return self._dim


def build_default(model_path: str | Path, **kwargs) -> NomicGGUFEmbedder:
    """Convenience constructor used from project_context to keep the
    call site short."""
    return NomicGGUFEmbedder(model_path, **kwargs)
