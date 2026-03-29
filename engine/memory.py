"""
Memory Module — First-Class Memory System

Three tiers of memory:
1. Short-Term: Current conversation buffer (trimmed intelligently)
2. Long-Term: Persistent facts/interactions stored as embeddings
3. System: Static knowledge (world rules, NPC lore, etc.)

Phase 1-3: Simple JSON-based long-term store with keyword matching.
Phase 4:   FAISS-based semantic search with sentence-transformer embeddings.
           Configurable via backend: "simple" | "faiss"
           Graceful fallback to simple if FAISS/sentence-transformers not installed.

Memory is what makes a small model feel intelligent across sessions.
Without it, every conversation starts from zero.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from engine.config import MemoryConfig

# Lazy imports for optional FAISS dependencies
_faiss = None
_FAISS_AVAILABLE = False

def _ensure_faiss():
    """Lazy-load FAISS. Returns True if available."""
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

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory record."""
    content: str
    timestamp: float
    source: str = "conversation"  # conversation, user, system
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "source": self.source,
            "tags": self.tags,
            "importance": self.importance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "conversation"),
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
        )


class ShortTermMemory:
    """
    Conversation buffer. Keeps recent turns in a sliding window.
    Trims oldest turns when the buffer exceeds limits.
    """

    def __init__(self, max_turns: int = 20, max_tokens: int = 1024):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.turns: list[dict] = []

    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        self._trim()

    def _trim(self):
        """Keep turns within limits."""
        # Trim by turn count
        while len(self.turns) > self.max_turns:
            self.turns.pop(0)

        # Trim by approximate token count
        total_chars = sum(len(t["content"]) for t in self.turns)
        approx_tokens = total_chars // 4
        while approx_tokens > self.max_tokens and len(self.turns) > 2:
            removed = self.turns.pop(0)
            total_chars -= len(removed["content"])
            approx_tokens = total_chars // 4

    def get_context(self) -> str:
        """Format conversation history for prompt injection."""
        if not self.turns:
            return ""

        lines = []
        for turn in self.turns:
            role = turn["role"].capitalize()
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def get_turns(self) -> list[dict]:
        """Get raw turn data."""
        return list(self.turns)

    def clear(self):
        """Clear conversation buffer."""
        self.turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self.turns)


class LongTermMemory:
    """
    Persistent memory store.

    v1: Simple JSON-based storage with keyword search.
    v2: FAISS vector store with sentence-transformer embeddings.

    Stores facts, user preferences, important interactions,
    and anything the system should remember across sessions.
    """

    def __init__(self, storage_dir: str, top_k: int = 5, similarity_threshold: float = 0.5):
        self.storage_dir = Path(storage_dir)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self._memories: list[MemoryEntry] = []
        self._storage_file = self.storage_dir / "long_term.json"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Load memories from disk."""
        if self._storage_file.exists():
            try:
                with open(self._storage_file, "r") as f:
                    data = json.load(f)
                self._memories = [MemoryEntry.from_dict(m) for m in data]
                logger.info(f"Loaded {len(self._memories)} long-term memories")
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
                self._memories = []
        else:
            logger.info("No existing long-term memories found")

    def _save(self):
        """Persist memories to disk."""
        try:
            with open(self._storage_file, "w") as f:
                json.dump(
                    [m.to_dict() for m in self._memories],
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")

    def store(self, content: str, source: str = "conversation",
              tags: Optional[list[str]] = None, importance: float = 0.5,
              metadata: Optional[dict] = None):
        """
        Store a new memory.

        Args:
            content: The memory content
            source: Where this came from (conversation, user, system)
            tags: Searchable tags
            importance: 0.0-1.0 importance score
            metadata: Additional key-value data
        """
        entry = MemoryEntry(
            content=content,
            timestamp=time.time(),
            source=source,
            tags=tags or [],
            importance=importance,
            metadata=metadata or {},
        )
        self._memories.append(entry)
        self._save()
        logger.debug(f"Stored memory: {content[:60]}...")

    def search(self, query: str, top_k: Optional[int] = None) -> list[MemoryEntry]:
        """
        Search memories by keyword matching.

        v1: Simple keyword overlap scoring.
        v2: Will use cosine similarity on embeddings.

        Args:
            query: Search query
            top_k: Override number of results

        Returns:
            List of matching MemoryEntry objects, sorted by relevance.
        """
        k = top_k or self.top_k
        query_words = set(query.lower().split())

        scored = []
        for memory in self._memories:
            memory_words = set(memory.content.lower().split())
            memory_tags = set(t.lower() for t in memory.tags)

            # Score: keyword overlap + tag match + importance boost
            word_overlap = len(query_words & memory_words)
            tag_overlap = len(query_words & memory_tags) * 2  # tags worth more
            importance_boost = memory.importance * 0.5

            if word_overlap > 0 or tag_overlap > 0:
                total_score = word_overlap + tag_overlap + importance_boost
                # Normalize roughly to 0-1
                max_possible = len(query_words) * 3 + 1
                normalized = min(total_score / max_possible, 1.0)

                if normalized >= self.similarity_threshold:
                    scored.append((normalized, memory))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results = [mem for _, mem in scored[:k]]
        logger.debug(f"Memory search '{query[:40]}' returned {len(results)} results")
        return results

    def get_context(self, query: str) -> str:
        """
        Search and format memories for prompt injection.
        """
        results = self.search(query)
        if not results:
            return ""

        lines = ["[Recalled memories:]"]
        for mem in results:
            tag_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
            lines.append(f"- {mem.content}{tag_str}")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._memories)

    def clear(self):
        """Clear all memories (use carefully)."""
        self._memories.clear()
        self._save()


class FAISSLongTermMemory:
    """
    Phase 4: Vector-based long-term memory using FAISS + sentence-transformers.

    Embeds all memories with a lightweight sentence-transformer model,
    stores vectors in a FAISS IndexFlatIP (cosine similarity on normalized vectors),
    and persists both the index and metadata to disk.

    Same interface as LongTermMemory so they're interchangeable.
    Thread-safe for async pipeline compatibility.
    """

    def __init__(self, storage_dir: str, top_k: int = 5,
                 similarity_threshold: float = 0.5,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.storage_dir = Path(storage_dir)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self._embedding_model_name = embedding_model

        self._memories: list[MemoryEntry] = []
        self._index = None  # FAISS index
        self._embedder = None  # SentenceTransformer
        self._dim: int = 0  # Embedding dimension
        self._lock = threading.RLock()

        # File paths
        self._metadata_file = self.storage_dir / "long_term.json"
        self._index_file = self.storage_dir / "faiss.index"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._init_embedder()
        self._load()

    def _init_embedder(self):
        """Initialize the sentence-transformer embedding model (shared instance)."""
        from engine.embedder import get_embedder
        self._embedder = get_embedder(self._embedding_model_name)
        dummy = self._embedder.encode(["hello"], normalize_embeddings=True)
        self._dim = dummy.shape[1]
        logger.info(
            f"FAISS memory initialized: model={self._embedding_model_name}, "
            f"dim={self._dim}"
        )

    def _build_empty_index(self):
        """Create a fresh FAISS index."""
        self._index = _faiss.IndexFlatIP(self._dim)

    def _load(self):
        """Load metadata + FAISS index from disk."""
        # Load metadata (memory entries)
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r") as f:
                    data = json.load(f)
                self._memories = [MemoryEntry.from_dict(m) for m in data]
                logger.info(f"Loaded {len(self._memories)} long-term memories (FAISS)")
            except Exception as e:
                logger.error(f"Failed to load memory metadata: {e}")
                self._memories = []

        # Load or rebuild FAISS index
        if self._index_file.exists() and self._memories:
            try:
                self._index = _faiss.read_index(str(self._index_file))
                if self._index.ntotal != len(self._memories):
                    logger.warning(
                        f"Index/metadata mismatch ({self._index.ntotal} vs "
                        f"{len(self._memories)}), rebuilding index"
                    )
                    self._rebuild_index()
                else:
                    logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}, rebuilding")
                self._rebuild_index()
        elif self._memories:
            logger.info("No FAISS index found, building from existing memories")
            self._rebuild_index()
        else:
            self._build_empty_index()

    def _rebuild_index(self):
        """Re-embed all memories and build a fresh FAISS index."""
        self._build_empty_index()
        if not self._memories:
            return

        texts = [m.content for m in self._memories]
        embeddings = self._embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        self._index.add(embeddings.astype(np.float32))
        self._save_index()
        logger.info(f"Rebuilt FAISS index with {len(texts)} vectors")

    def _save(self):
        """Persist metadata to disk."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump([m.to_dict() for m in self._memories], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory metadata: {e}")

    def _save_index(self):
        """Persist FAISS index to disk."""
        try:
            _faiss.write_index(self._index, str(self._index_file))
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def store(self, content: str, source: str = "conversation",
              tags: Optional[list[str]] = None, importance: float = 0.5,
              metadata: Optional[dict] = None):
        """Store a new memory with its embedding."""
        with self._lock:
            entry = MemoryEntry(
                content=content,
                timestamp=time.time(),
                source=source,
                tags=tags or [],
                importance=importance,
                metadata=metadata or {},
            )
            self._memories.append(entry)

            # Embed and add to index
            embedding = self._embedder.encode(
                [content], normalize_embeddings=True, show_progress_bar=False,
            )
            self._index.add(embedding.astype(np.float32))

            self._save()
            self._save_index()
            logger.debug(f"Stored memory (FAISS): {content[:60]}...")

    def search(self, query: str, top_k: Optional[int] = None) -> list[MemoryEntry]:
        """
        Semantic search over memories using cosine similarity.

        Embeds the query, searches the FAISS index, filters by
        similarity threshold, and applies importance boosting.
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []

            k = min(top_k or self.top_k, self._index.ntotal)
            query_vec = self._embedder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False,
            )

            # FAISS search returns (distances, indices)
            # With IndexFlatIP on normalized vectors, distance = cosine similarity
            scores, indices = self._index.search(query_vec.astype(np.float32), k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:  # FAISS returns -1 for missing results
                    continue

                memory = self._memories[idx]
                # Combine semantic similarity with importance
                boosted_score = score * 0.8 + memory.importance * 0.2

                if boosted_score >= self.similarity_threshold:
                    results.append((boosted_score, memory))

            # Sort by boosted score descending
            results.sort(key=lambda x: x[0], reverse=True)

            matched = [mem for _, mem in results]
            logger.debug(
                f"FAISS search '{query[:40]}' returned {len(matched)} results "
                f"(top score: {results[0][0]:.3f})" if results else
                f"FAISS search '{query[:40]}' returned 0 results"
            )
            return matched

    def get_context(self, query: str) -> str:
        """Search and format memories for prompt injection."""
        results = self.search(query)
        if not results:
            return ""

        lines = ["[Recalled memories:]"]
        for mem in results:
            tag_str = f" [{', '.join(mem.tags)}]" if mem.tags else ""
            lines.append(f"- {mem.content}{tag_str}")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._memories)

    def clear(self):
        """Clear all memories and the FAISS index."""
        with self._lock:
            self._memories.clear()
            self._build_empty_index()
            self._save()
            self._save_index()


class SystemMemory:
    """
    Static knowledge store. Loaded from text/YAML files.
    Used for world rules, NPC backstories, static lore, etc.

    Not searchable in the same way — injected based on
    active NPC profile or system mode.
    """

    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = Path(knowledge_dir)
        self._knowledge: dict[str, str] = {}

        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Load all knowledge files."""
        if not self.knowledge_dir.exists():
            return

        for file_path in self.knowledge_dir.glob("*.txt"):
            key = file_path.stem
            self._knowledge[key] = file_path.read_text(encoding="utf-8")
            logger.info(f"Loaded system knowledge: {key}")

        for file_path in self.knowledge_dir.glob("*.yaml"):
            key = file_path.stem
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            self._knowledge[key] = yaml.dump(data, default_flow_style=False)
            logger.info(f"Loaded system knowledge: {key}")

    def get(self, key: str) -> Optional[str]:
        """Get a specific knowledge entry."""
        return self._knowledge.get(key)

    def get_all(self) -> dict[str, str]:
        """Get all knowledge entries."""
        return dict(self._knowledge)

    @property
    def available_keys(self) -> list[str]:
        return list(self._knowledge.keys())


@dataclass
class CompressedState:
    """A compressed summary of past conversation(s)."""
    summary_sentences: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: dict[str, int] = field(default_factory=dict)
    turns_compressed: int = 0
    total_compressions: int = 0
    last_compressed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "summary_sentences": self.summary_sentences,
            "topics": self.topics,
            "entities": self.entities,
            "turns_compressed": self.turns_compressed,
            "total_compressions": self.total_compressions,
            "last_compressed": self.last_compressed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompressedState":
        return cls(
            summary_sentences=data.get("summary_sentences", []),
            topics=data.get("topics", []),
            entities=data.get("entities", {}),
            turns_compressed=data.get("turns_compressed", 0),
            total_compressions=data.get("total_compressions", 0),
            last_compressed=data.get("last_compressed", 0.0),
        )


class ConversationCompressor:
    """
    Phase 4: Compresses conversation history into persistent summaries.

    Uses extractive summarization (no LLM required):
    - TF-IDF scoring to find the most representative sentences
    - Keyword frequency analysis for topic extraction
    - Entity tracking for important terms

    The compressed state persists to disk and is restored on startup,
    giving the model context about previous sessions without needing
    to replay the full conversation history.
    """

    def __init__(self, storage_dir: str, max_summary_sentences: int = 10,
                 max_topics: int = 8):
        self.storage_dir = Path(storage_dir)
        self.max_summary_sentences = max_summary_sentences
        self.max_topics = max_topics
        self._state_file = self.storage_dir / "compressed_state.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.state: CompressedState = self._load()

    def _load(self) -> CompressedState:
        """Load compressed state from disk."""
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    data = json.load(f)
                state = CompressedState.from_dict(data)
                if state.turns_compressed > 0:
                    logger.info(
                        f"Loaded compressed state: {state.turns_compressed} turns, "
                        f"{len(state.summary_sentences)} sentences, "
                        f"{len(state.topics)} topics"
                    )
                return state
            except Exception as e:
                logger.error(f"Failed to load compressed state: {e}")
        return CompressedState()

    def _save(self):
        """Persist compressed state to disk."""
        try:
            with open(self._state_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save compressed state: {e}")

    def compress(self, turns: list[dict], keep_recent: int = 4) -> tuple[CompressedState, list[dict]]:
        """
        Compress older conversation turns into a summary.

        Args:
            turns: Full list of conversation turns
            keep_recent: Number of recent turns to keep uncompressed

        Returns:
            (updated CompressedState, remaining uncompressed turns)
        """
        if len(turns) <= keep_recent:
            return self.state, turns

        # Split: compress older turns, keep recent
        to_compress = turns[:-keep_recent] if keep_recent > 0 else turns
        to_keep = turns[-keep_recent:] if keep_recent > 0 else []

        # Extract all text from turns to compress
        all_text = []
        for turn in to_compress:
            content = turn.get("content", "")
            if content.strip():
                all_text.append(f"{turn.get('role', 'unknown')}: {content}")

        if not all_text:
            return self.state, to_keep

        # Extract key sentences via TF-IDF scoring
        sentences = self._extract_sentences(all_text)
        key_sentences = self._score_and_select(sentences)

        # Extract topics and entities
        topics = self._extract_topics(all_text)
        entities = self._extract_entities(all_text)

        # Merge with existing compressed state
        merged_sentences = self.state.summary_sentences + key_sentences
        # Keep only the most recent max_summary_sentences
        if len(merged_sentences) > self.max_summary_sentences:
            merged_sentences = merged_sentences[-self.max_summary_sentences:]

        merged_topics = list(dict.fromkeys(self.state.topics + topics))[:self.max_topics]

        merged_entities = dict(self.state.entities)
        for entity, count in entities.items():
            merged_entities[entity] = merged_entities.get(entity, 0) + count
        # Keep top entities by frequency
        if len(merged_entities) > 20:
            sorted_entities = sorted(merged_entities.items(), key=lambda x: x[1], reverse=True)
            merged_entities = dict(sorted_entities[:20])

        self.state = CompressedState(
            summary_sentences=merged_sentences,
            topics=merged_topics,
            entities=merged_entities,
            turns_compressed=self.state.turns_compressed + len(to_compress),
            total_compressions=self.state.total_compressions + 1,
            last_compressed=time.time(),
        )
        self._save()

        logger.info(
            f"Compressed {len(to_compress)} turns into {len(key_sentences)} sentences. "
            f"Total compressed: {self.state.turns_compressed} turns"
        )

        return self.state, to_keep

    def _extract_sentences(self, texts: list[str]) -> list[str]:
        """Split text into individual sentences."""
        import re
        sentences = []
        for text in texts:
            # Split on sentence boundaries
            parts = re.split(r'(?<=[.!?])\s+', text)
            for part in parts:
                part = part.strip()
                if len(part) > 15:  # Skip very short fragments
                    sentences.append(part)
        return sentences

    def _score_and_select(self, sentences: list[str]) -> list[str]:
        """Score sentences by TF-IDF importance and select the top ones."""
        if len(sentences) <= self.max_summary_sentences:
            return sentences

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Score each sentence by sum of its TF-IDF values
            scores = tfidf_matrix.sum(axis=1).A1  # Dense array

            # Get top-K sentence indices, preserving original order
            top_indices = sorted(
                sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                [:self.max_summary_sentences]
            )

            return [sentences[i] for i in top_indices]

        except ImportError:
            # Fallback: just take evenly spaced sentences
            step = max(1, len(sentences) // self.max_summary_sentences)
            return sentences[::step][:self.max_summary_sentences]

    def _extract_topics(self, texts: list[str]) -> list[str]:
        """Extract the most important topics from the conversation."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            # Average TF-IDF across all documents
            avg_scores = tfidf_matrix.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-self.max_topics:][::-1]

            return [feature_names[i] for i in top_indices]

        except ImportError:
            # Fallback: simple word frequency
            from collections import Counter
            words = " ".join(texts).lower().split()
            stop = {"the", "a", "an", "is", "was", "are", "were", "be", "been",
                    "i", "you", "we", "they", "he", "she", "it", "my", "your",
                    "to", "of", "in", "for", "on", "with", "at", "by", "from",
                    "and", "or", "but", "not", "this", "that", "user:", "assistant:"}
            filtered = [w for w in words if w not in stop and len(w) > 2]
            return [w for w, _ in Counter(filtered).most_common(self.max_topics)]

    def _extract_entities(self, texts: list[str]) -> dict[str, int]:
        """Extract frequently mentioned entities (capitalized terms, code identifiers)."""
        import re
        from collections import Counter

        entities = Counter()
        for text in texts:
            # Find capitalized words (potential proper nouns/entities)
            caps = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
            entities.update(caps)

            # Find code-like identifiers (snake_case, camelCase)
            code_ids = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text)  # snake_case
            entities.update(code_ids)
            camel = re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', text)  # camelCase
            entities.update(camel)

        # Filter out low-frequency and very common terms
        return {k: v for k, v in entities.items() if v >= 2}

    def format_for_prompt(self) -> str:
        """Format compressed state for injection into the prompt."""
        if not self.state.summary_sentences and not self.state.topics:
            return ""

        lines = ["[Previous session context:]"]

        if self.state.topics:
            lines.append(f"Topics discussed: {', '.join(self.state.topics)}")

        if self.state.entities:
            top_entities = sorted(self.state.entities.items(), key=lambda x: x[1], reverse=True)[:10]
            entity_str = ", ".join(f"{k}" for k, v in top_entities)
            lines.append(f"Key terms: {entity_str}")

        if self.state.summary_sentences:
            lines.append("Summary of prior conversation:")
            for sentence in self.state.summary_sentences:
                lines.append(f"  - {sentence}")

        return "\n".join(lines)

    def clear(self):
        """Clear compressed state."""
        self.state = CompressedState()
        self._save()

    def status(self) -> dict:
        """Get compressor status."""
        return {
            "turns_compressed": self.state.turns_compressed,
            "summary_sentences": len(self.state.summary_sentences),
            "topics": self.state.topics,
            "total_compressions": self.state.total_compressions,
            "has_state": bool(self.state.summary_sentences or self.state.topics),
        }


class MemorySystem:
    """
    Unified memory interface. Coordinates all memory tiers + compression.
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.enabled = config.enabled

        # Initialize tiers
        self.short_term = ShortTermMemory(
            max_turns=config.short_term.max_turns,
            max_tokens=config.short_term.max_tokens,
        )

        self.long_term = None
        if config.long_term.enabled:
            backend = config.long_term.backend

            if backend == "faiss":
                if _ensure_faiss():
                    self.long_term = FAISSLongTermMemory(
                        storage_dir=config.long_term.storage_dir,
                        top_k=config.long_term.top_k,
                        similarity_threshold=config.long_term.similarity_threshold,
                        embedding_model=config.long_term.embedding_model,
                    )
                    logger.info("Long-term memory: FAISS backend")
                else:
                    logger.warning(
                        "FAISS backend requested but faiss-cpu or sentence-transformers "
                        "not installed. Falling back to simple keyword backend."
                    )
                    backend = "simple"

            if backend != "faiss":
                self.long_term = LongTermMemory(
                    storage_dir=config.long_term.storage_dir,
                    top_k=config.long_term.top_k,
                    similarity_threshold=config.long_term.similarity_threshold,
                )
                logger.info("Long-term memory: simple keyword backend")

        self.system = None
        if config.system.enabled:
            self.system = SystemMemory(
                knowledge_dir=config.system.knowledge_dir,
            )

        # Phase 4: Conversation compressor
        self.compressor = ConversationCompressor(
            storage_dir=config.long_term.storage_dir if config.long_term.enabled else "data/memory",
            max_summary_sentences=config.compressor.max_summary_sentences,
            max_topics=config.compressor.max_topics,
        )

    def add_interaction(self, role: str, content: str):
        """Record a conversation turn in short-term memory."""
        if not self.enabled:
            return
        self.short_term.add_turn(role, content)

    def remember(self, content: str, **kwargs):
        """Store something in long-term memory."""
        if self.long_term:
            self.long_term.store(content, **kwargs)

    def compress(self, keep_recent: int = 4) -> str:
        """
        Compress older conversation turns into persistent summary.
        Returns formatted summary for confirmation.
        """
        turns = self.short_term.get_turns()
        state, remaining = self.compressor.compress(turns, keep_recent=keep_recent)

        # Replace short-term memory with only the kept turns
        self.short_term.turns = remaining

        return self.compressor.format_for_prompt()

    def recall(self, query: str) -> dict[str, str]:
        """
        Retrieve relevant memories from all tiers.

        Returns dict with keys:
          - "conversation": recent conversation history
          - "long_term": relevant long-term memories
          - "compressed": compressed state from prior sessions
        """
        context = {}

        # Compressed state from prior sessions/compressions
        compressed = self.compressor.format_for_prompt()
        if compressed:
            context["compressed"] = compressed

        # Short-term: always include conversation
        context["conversation"] = self.short_term.get_context()

        # Long-term: search for relevant memories
        if self.long_term:
            context["long_term"] = self.long_term.get_context(query)

        return context

    def get_system_knowledge(self, key: str) -> Optional[str]:
        """Get specific system knowledge by key."""
        if self.system:
            return self.system.get(key)
        return None

    def status(self) -> dict:
        """Get memory system status."""
        return {
            "enabled": self.enabled,
            "short_term_turns": self.short_term.turn_count,
            "long_term_count": self.long_term.count if self.long_term else 0,
            "system_keys": self.system.available_keys if self.system else [],
            "compressor": self.compressor.status(),
        }
