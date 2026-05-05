"""Integration test: NomicGGUFEmbedder swapped into ProjectIndex.

Verifies the role-switch hooks fire correctly: the index path uses
PASSAGE_PREFIX (default), the search path flips to QUERY_PREFIX, and
flips back to passage after the search call returns.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from engine.config import ProjectContextConfig
from engine.nomic_embedder import NomicGGUFEmbedder, PASSAGE_PREFIX, QUERY_PREFIX
from engine.project_context import ProjectIndex


class _StubLLM:
    DIM = 8

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str):
        self.calls.append(text)
        h = abs(hash(text))
        return [float((h >> (i * 4)) & 0xF) for i in range(self.DIM)]


def _make_indexed_project(tmpdir: Path, embedder: NomicGGUFEmbedder) -> ProjectIndex:
    """Create a tiny project, hand it to ProjectIndex, and run index_directory."""
    src = tmpdir / "src"
    src.mkdir()
    (src / "a.py").write_text("def add(x, y):\n    return x + y\n", encoding="utf-8")
    (src / "b.py").write_text("def greet(name):\n    return f'hi {name}'\n", encoding="utf-8")

    cfg = ProjectContextConfig(
        enabled=True,
        storage_dir=str(tmpdir / "_index"),
        top_k=2,
        similarity_threshold=-1.0,  # accept all hits in test
        max_chunk_lines=40,
    )
    idx = ProjectIndex(cfg)
    idx.init_embedder(embedder)
    return idx, str(src)


def test_search_uses_query_role_then_restores_passage():
    """Index → search → assert call sequence shows passage prefixes
    during indexing and a query prefix on the search call, then
    passage again afterward."""
    pytest.importorskip("faiss", reason="FAISS not installed in this environment")
    stub = _StubLLM()
    embedder = NomicGGUFEmbedder(
        model_path="/x.gguf", _llm_factory=lambda: stub
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        idx, src_root = _make_indexed_project(tmp_p, embedder)
        idx.index_directory(src_root)

        # Mark the boundary so we can split the call log cleanly.
        boundary = len(stub.calls)

        # Run a search.
        idx.search("how do I add two numbers")

        post_search_calls = stub.calls[boundary:]
        # The search itself should have emitted exactly one query-prefixed
        # call (the user query).
        query_prefixed = [c for c in post_search_calls if c.startswith(QUERY_PREFIX)]
        assert len(query_prefixed) == 1, post_search_calls
        assert query_prefixed[0] == QUERY_PREFIX + "how do I add two numbers"

        # After search returns, role should be reset to passage.
        # Encode another doc-style payload directly via the embedder.
        embedder.encode(["something else"])
        # The most recent call should now have the passage prefix again.
        assert stub.calls[-1] == PASSAGE_PREFIX + "something else"


def test_indexing_uses_passage_role_throughout():
    """Every encode() call during index_directory() must be passage-prefixed."""
    pytest.importorskip("faiss", reason="FAISS not installed in this environment")
    stub = _StubLLM()
    embedder = NomicGGUFEmbedder(
        model_path="/x.gguf", _llm_factory=lambda: stub
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        idx, src_root = _make_indexed_project(tmp_p, embedder)
        # Snapshot the call boundary AFTER init_embedder's probe call.
        boundary = len(stub.calls)
        idx.index_directory(src_root)

        index_calls = stub.calls[boundary:]
        # All index calls go via passage role.
        non_passage = [c for c in index_calls if not c.startswith(PASSAGE_PREFIX)]
        assert non_passage == [], non_passage


def test_search_role_restored_even_on_exception():
    """If the embedder raises during search, the role must still be
    restored to passage so subsequent indexing isn't poisoned."""
    pytest.importorskip("faiss", reason="FAISS not installed in this environment")

    class _BoomLLM:
        def __init__(self):
            self.calls = 0

        def embed(self, text):
            self.calls += 1
            if self.calls == 1:
                # First call is the probe — succeed so load() returns clean.
                return [0.1] * 8
            # Every other call (including indexing chunks) succeeds...
            # but the search call will raise.
            if "search_query: " in text:
                raise RuntimeError("nope")
            return [0.2] * 8

    boom = _BoomLLM()
    embedder = NomicGGUFEmbedder(
        model_path="/x.gguf", _llm_factory=lambda: boom
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        idx, src_root = _make_indexed_project(tmp_p, embedder)
        idx.index_directory(src_root)

        with pytest.raises(RuntimeError):
            idx.search("anything")

        # Even after the exception, role should be back at passage.
        assert embedder._role_prefix == PASSAGE_PREFIX


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
