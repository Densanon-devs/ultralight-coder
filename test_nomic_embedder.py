"""Unit tests for engine.nomic_embedder.

Use a stub _llm_factory so tests run with no GGUF model present and
no llama-cpp-python initialization. The stub just returns deterministic
vectors based on the input string, so we can verify shape + role
prefix + normalization behavior without a real embedder.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.nomic_embedder import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    NomicGGUFEmbedder,
    build_default,
)


class _StubLLM:
    """Tiny stand-in for llama_cpp.Llama with embed() only.

    Records every text it was called on (so tests can inspect the role
    prefix that was applied) and returns a deterministic 8-d vector
    derived from the input length + first character."""

    DIM = 8

    def __init__(self) -> None:
        self.calls: list[str] = []
        # Simulate the variant where embed() returns list[list[float]]
        # — that's the path our adapter must also handle. Toggleable
        # per-test to also exercise the list[float] path.
        self.return_nested = False

    def embed(self, text: str):
        self.calls.append(text)
        # Deterministic vector: first 8 floats from a hash of the text.
        h = abs(hash(text))
        vec = [float((h >> (i * 4)) & 0xF) for i in range(self.DIM)]
        if self.return_nested:
            return [vec]
        return vec


def _make_embedder(stub: _StubLLM | None = None) -> NomicGGUFEmbedder:
    stub = stub or _StubLLM()
    emb = NomicGGUFEmbedder(
        model_path="/no/such/path.gguf",
        n_gpu_layers=0,
        _llm_factory=lambda: stub,
    )
    return emb


# ── Lifecycle ────────────────────────────────────────────────────


def test_load_is_idempotent():
    stub = _StubLLM()
    constructed = []
    emb = NomicGGUFEmbedder(
        model_path="/x.gguf",
        _llm_factory=lambda: (constructed.append(1) or stub),
    )
    emb.load()
    emb.load()
    emb.load()
    # Factory should only fire once.
    assert constructed == [1]
    # Probe must have set the dimension.
    assert emb.dimension == _StubLLM.DIM


def test_dimension_inferred_from_probe():
    emb = _make_embedder()
    emb.load()
    assert emb.dimension == 8


def test_handles_nested_embed_return_shape():
    """Some llama-cpp-python builds return list[list[float]] from embed()
    even on a single string. The adapter must normalize either path."""
    stub = _StubLLM()
    stub.return_nested = True
    emb = _make_embedder(stub)
    out = emb.encode(["hello"])
    assert out.shape == (1, 8)


# ── Encode shape + role ──────────────────────────────────────────


def test_encode_returns_2d_array_with_correct_shape():
    emb = _make_embedder()
    out = emb.encode(["a", "b", "c"])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 8)
    assert out.dtype == np.float32


def test_encode_empty_input_returns_zero_rows():
    emb = _make_embedder()
    emb.load()  # so dimension is known
    out = emb.encode([])
    assert out.shape == (0, 8)


def test_default_role_is_passage():
    stub = _StubLLM()
    emb = _make_embedder(stub)
    emb.encode(["hello world"])
    # First call was the load() probe; second was our encode payload.
    assert stub.calls[-1] == PASSAGE_PREFIX + "hello world"


def test_use_query_role_switches_prefix():
    stub = _StubLLM()
    emb = _make_embedder(stub)
    emb.use_query_role()
    emb.encode(["how do I import json"])
    assert stub.calls[-1] == QUERY_PREFIX + "how do I import json"


def test_use_passage_role_switches_back():
    stub = _StubLLM()
    emb = _make_embedder(stub)
    emb.use_query_role()
    emb.encode(["q"])
    assert stub.calls[-1].startswith(QUERY_PREFIX)
    emb.use_passage_role()
    emb.encode(["p"])
    assert stub.calls[-1].startswith(PASSAGE_PREFIX)


def test_none_or_empty_strings_get_prefix_only():
    stub = _StubLLM()
    emb = _make_embedder(stub)
    emb.encode(["", ""])
    # Two encode calls after the probe.
    assert stub.calls[-2] == PASSAGE_PREFIX
    assert stub.calls[-1] == PASSAGE_PREFIX


# ── Normalization ────────────────────────────────────────────────


def test_normalize_default_true_produces_unit_vectors():
    emb = _make_embedder()
    out = emb.encode(["alpha", "beta", "gamma"])
    norms = np.linalg.norm(out, axis=1)
    for n in norms:
        assert math.isclose(n, 1.0, abs_tol=1e-5) or n == 0.0


def test_normalize_false_preserves_raw_values():
    stub = _StubLLM()
    emb = _make_embedder(stub)
    raw = emb.encode(["alpha"], normalize_embeddings=False)
    norm = emb.encode(["alpha"], normalize_embeddings=True)
    # Raw should have norm > 1 in general (stub returns ints 0-15);
    # normalized vector should sit on the unit sphere.
    assert np.linalg.norm(raw[0]) >= 1.0 - 1e-5
    assert math.isclose(np.linalg.norm(norm[0]), 1.0, abs_tol=1e-5)


def test_zero_embedding_does_not_divide_by_zero():
    """If the underlying model emits an all-zero vector for some input,
    normalization must not crash with NaN."""
    class _ZeroLLM:
        def embed(self, text):
            return [0.0] * 8
    emb = NomicGGUFEmbedder(model_path="/x", _llm_factory=lambda: _ZeroLLM())
    out = emb.encode(["x"])
    assert not np.isnan(out).any()
    # All zeros stays all zeros after the safe-divide.
    assert np.all(out == 0.0)


# ── Determinism ──────────────────────────────────────────────────


def test_same_input_same_output():
    stub = _StubLLM()
    emb = _make_embedder(stub)
    a = emb.encode(["repeatable"])
    b = emb.encode(["repeatable"])
    assert np.allclose(a, b)


# ── Factory ──────────────────────────────────────────────────────


def test_build_default_returns_embedder_instance():
    emb = build_default("/x.gguf", _llm_factory=lambda: _StubLLM())
    assert isinstance(emb, NomicGGUFEmbedder)


# ── Show-progress / batch-size are accepted as kwargs ─────────────


def test_show_progress_and_batch_size_kwargs_accepted():
    """ProjectIndex calls .encode(...) with these kwargs; the adapter
    must accept them silently for drop-in compatibility."""
    emb = _make_embedder()
    out = emb.encode(["a", "b"], show_progress_bar=True, batch_size=64)
    assert out.shape == (2, 8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
