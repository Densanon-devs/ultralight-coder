"""
Project Context — Codebase Indexing and Retrieval

Indexes a user's project directory so the model can reference relevant code
when answering questions. Uses the shared sentence-transformer embedder and
FAISS for fast semantic search.

Flow:
    1. Walk the project directory, respecting ignore patterns
    2. Chunk each file into logical blocks (by blank-line boundaries)
    3. Embed all chunks with the shared sentence-transformer
    4. Store in a FAISS IndexFlatIP for cosine similarity search
    5. On each query, retrieve top-k relevant chunks
    6. Format as a "project" section for prompt injection via FusionLayer

Persists the index to disk so re-indexing is only needed when files change.
"""

import fnmatch
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from engine.config import ProjectContextConfig

logger = logging.getLogger(__name__)

_faiss = None
_FAISS_AVAILABLE = False


def _ensure_faiss():
    global _faiss, _FAISS_AVAILABLE
    if _FAISS_AVAILABLE:
        return True
    try:
        import faiss as _f
        _faiss = _f
        _FAISS_AVAILABLE = True
        return True
    except ImportError:
        return False


@dataclass
class CodeChunk:
    """A chunk of code from the project."""
    file_path: str       # Relative path from project root
    start_line: int      # 1-based
    end_line: int        # 1-based, inclusive
    content: str         # The actual code text
    language: str = ""   # Detected from extension

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeChunk":
        return cls(**data)

    def format_for_prompt(self) -> str:
        return f"# {self.file_path}:{self.start_line}-{self.end_line}\n{self.content}"


# Extension → language name mapping
_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".java": "java", ".cs": "csharp", ".rb": "ruby",
    ".kt": "kotlin", ".swift": "swift", ".sql": "sql",
    ".sh": "bash", ".bash": "bash",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".toml": "toml", ".md": "markdown", ".txt": "text",
}


class ProjectIndex:
    """
    Indexes a project directory for semantic code retrieval.

    Thread-safe. Persists index and metadata to disk.
    """

    def __init__(self, config: ProjectContextConfig):
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._chunks: list[CodeChunk] = []
        self._index = None
        self._embedder = None
        self._dim: int = 0
        self._lock = threading.RLock()

        self._project_root: Optional[str] = None
        self._indexed_at: float = 0
        self._file_count: int = 0

        # File paths
        self._metadata_file = self.storage_dir / "project_chunks.json"
        self._index_file = self.storage_dir / "project.faiss"
        self._state_file = self.storage_dir / "project_state.json"

        self._load_state()
        self._load_index()

    def init_embedder(self, embedder):
        """Set the shared embedder instance."""
        self._embedder = embedder
        if embedder:
            dummy = embedder.encode(["hello"], normalize_embeddings=True)
            self._dim = dummy.shape[1]

    def _load_state(self):
        """Load project state (root path, index time)."""
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    state = json.load(f)
                self._project_root = state.get("project_root")
                self._indexed_at = state.get("indexed_at", 0)
                self._file_count = state.get("file_count", 0)
            except Exception as e:
                logger.warning(f"Failed to load project state: {e}")

    def _save_state(self):
        """Save project state."""
        try:
            with open(self._state_file, "w") as f:
                json.dump({
                    "project_root": self._project_root,
                    "indexed_at": self._indexed_at,
                    "file_count": self._file_count,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save project state: {e}")

    def _load_index(self):
        """Load persisted chunks and FAISS index."""
        if not self._metadata_file.exists():
            return

        try:
            with open(self._metadata_file, "r") as f:
                raw = json.load(f)
            self._chunks = [CodeChunk.from_dict(c) for c in raw]
        except Exception as e:
            logger.warning(f"Failed to load project chunks: {e}")
            self._chunks = []
            return

        if _ensure_faiss() and self._index_file.exists() and self._chunks:
            try:
                self._index = _faiss.read_index(str(self._index_file))
                if self._index.ntotal != len(self._chunks):
                    logger.warning("Index/chunk mismatch, will need re-index")
                    self._index = None
                else:
                    logger.info(f"Loaded project index: {self._index.ntotal} chunks from {self._project_root}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS project index: {e}")
                self._index = None

    def _save_index(self):
        """Persist chunks and FAISS index."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump([c.to_dict() for c in self._chunks], f)
        except Exception as e:
            logger.error(f"Failed to save project chunks: {e}")

        if _ensure_faiss() and self._index is not None:
            try:
                _faiss.write_index(self._index, str(self._index_file))
            except Exception as e:
                logger.error(f"Failed to save project FAISS index: {e}")

    # ── Indexing ──

    def _should_ignore(self, path: Path, root: Path) -> bool:
        """Check if a path matches any ignore pattern."""
        rel = str(path.relative_to(root))
        name = path.name
        for pattern in self.config.ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
            if fnmatch.fnmatch(rel, pattern):
                return True
            # Check if any parent directory matches
            for part in path.relative_to(root).parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False

    def _chunk_file(self, filepath: Path, root: Path) -> list[CodeChunk]:
        """Split a file into logical chunks by blank-line boundaries."""
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        rel_path = str(filepath.relative_to(root)).replace("\\", "/")
        ext = filepath.suffix.lower()
        language = _EXT_LANG.get(ext, "")

        lines = text.split("\n")
        if len(lines) <= self.config.max_chunk_lines:
            # Small file — single chunk
            content = text.strip()
            if not content:
                return []
            return [CodeChunk(
                file_path=rel_path,
                start_line=1,
                end_line=len(lines),
                content=content,
                language=language,
            )]

        # Split at blank-line boundaries, respecting max chunk size
        chunks = []
        chunk_start = 0
        max_lines = self.config.max_chunk_lines
        overlap = self.config.overlap_lines

        while chunk_start < len(lines):
            chunk_end = min(chunk_start + max_lines, len(lines))

            # Try to break at a blank line near the end
            if chunk_end < len(lines):
                best_break = chunk_end
                for i in range(chunk_end, max(chunk_start + max_lines // 2, chunk_start), -1):
                    if i < len(lines) and lines[i].strip() == "":
                        best_break = i + 1
                        break
                chunk_end = best_break

            content = "\n".join(lines[chunk_start:chunk_end]).strip()
            if content:
                chunks.append(CodeChunk(
                    file_path=rel_path,
                    start_line=chunk_start + 1,
                    end_line=chunk_end,
                    content=content,
                    language=language,
                ))

            chunk_start = chunk_end - overlap
            if chunk_start >= len(lines) - overlap:
                break

        return chunks

    def index_directory(self, project_root: str) -> dict:
        """Index a project directory. Returns stats dict."""
        if not self._embedder:
            return {"error": "Embedder not initialized"}

        if not _ensure_faiss():
            return {"error": "FAISS not available"}

        root = Path(project_root)
        if not root.is_dir():
            return {"error": f"Not a directory: {project_root}"}

        start = time.monotonic()
        logger.info(f"Indexing project: {root}")

        # Collect all eligible files
        all_chunks: list[CodeChunk] = []
        file_count = 0
        skipped = 0

        for filepath in sorted(root.rglob("*")):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in self.config.file_extensions:
                continue
            if self._should_ignore(filepath, root):
                skipped += 1
                continue
            if filepath.stat().st_size > self.config.max_file_size_kb * 1024:
                skipped += 1
                continue

            chunks = self._chunk_file(filepath, root)
            all_chunks.extend(chunks)
            file_count += 1

        if not all_chunks:
            return {"error": "No eligible files found", "skipped": skipped}

        # Embed all chunks
        texts = [c.content for c in all_chunks]
        logger.info(f"Embedding {len(texts)} chunks from {file_count} files...")
        embeddings = self._embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
            batch_size=64,
        )

        # Build FAISS index
        with self._lock:
            self._index = _faiss.IndexFlatIP(self._dim)
            self._index.add(embeddings.astype(np.float32))
            self._chunks = all_chunks
            self._project_root = str(root)
            self._indexed_at = time.time()
            self._file_count = file_count

            self._save_index()
            self._save_state()

        elapsed = time.monotonic() - start
        stats = {
            "project_root": str(root),
            "files": file_count,
            "chunks": len(all_chunks),
            "skipped": skipped,
            "time": round(elapsed, 2),
        }
        logger.info(f"Project indexed: {stats}")
        return stats

    # ── Retrieval ──

    def search(self, query: str, top_k: Optional[int] = None) -> list[CodeChunk]:
        """Find the most relevant code chunks for a query."""
        with self._lock:
            if not self._embedder or self._index is None or self._index.ntotal == 0:
                return []

            k = min(top_k or self.config.top_k, self._index.ntotal)
            query_vec = self._embedder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False,
            )

            scores, indices = self._index.search(query_vec.astype(np.float32), k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                if score < self.config.similarity_threshold:
                    continue
                results.append(self._chunks[idx])

            return results

    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """Search and format results for prompt injection."""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        lines = ["[Relevant project code:]"]
        for chunk in results:
            lines.append(chunk.format_for_prompt())
        return "\n\n".join(lines)

    # ── Management ──

    def clear(self):
        """Clear the project index."""
        with self._lock:
            self._chunks = []
            self._index = None
            self._project_root = None
            self._indexed_at = 0
            self._file_count = 0

            for f in [self._metadata_file, self._index_file, self._state_file]:
                if f.exists():
                    f.unlink()

    def status(self) -> dict:
        """Get project index status."""
        return {
            "indexed": self._index is not None and self._index.ntotal > 0,
            "project_root": self._project_root,
            "files": self._file_count,
            "chunks": len(self._chunks),
            "indexed_at": self._indexed_at,
        }

    @property
    def is_indexed(self) -> bool:
        return self._index is not None and self._index.ntotal > 0
