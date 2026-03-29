"""
Micro-Adapter Engine — Phase 4: On-the-Fly Behavioral Adapters

Instead of full LoRA fine-tuning (too expensive for consumer hardware),
micro-adapters are lightweight behavioral profiles generated automatically
from interaction patterns.

How it works:
1. Every interaction is recorded (prompt, response, modules, feedback)
2. Periodically, interactions are clustered by semantic similarity
   (k-means on sentence-transformer embeddings)
3. Each cluster becomes a MicroAdapter with:
   - A context injection prompt (synthesized from representative interactions)
   - Style parameters (temperature/max_tokens tuned to cluster characteristics)
   - Module affinity weights (which modules this cluster tends to use)
4. On each new prompt, the engine finds the nearest adapter and applies it
5. Adapters persist to disk and improve as more interactions accumulate

This gives the system "learned behavior" without actual weight modification.
The base model stays frozen — adapters modify the prompt and generation params.
"""

import json
import logging
import re
import time
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InteractionRecord:
    """A recorded interaction for adapter learning."""
    prompt: str
    response: str
    modules: list[str]
    feedback: Optional[str] = None  # "good", "bad", None
    response_length: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "modules": self.modules,
            "feedback": self.feedback,
            "response_length": self.response_length,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InteractionRecord":
        return cls(
            prompt=data["prompt"],
            response=data.get("response", ""),
            modules=data.get("modules", []),
            feedback=data.get("feedback"),
            response_length=data.get("response_length", 0),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class MicroAdapter:
    """
    A lightweight behavioral adapter generated from interaction clusters.

    Applied by injecting context_prompt into the fusion layer and
    adjusting generation parameters via style_params.
    """
    name: str
    cluster_id: int
    context_prompt: str               # Injected into prompt to steer behavior
    style_params: dict                # temperature, max_tokens adjustments
    module_affinities: dict[str, float]  # module -> affinity weight (0-1)
    representative_prompts: list[str] # Example prompts from this cluster
    interaction_count: int = 0
    quality_score: float = 0.0        # Average quality from feedback
    centroid: Optional[list[float]] = None  # Cluster centroid embedding

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cluster_id": self.cluster_id,
            "context_prompt": self.context_prompt,
            "style_params": self.style_params,
            "module_affinities": self.module_affinities,
            "representative_prompts": self.representative_prompts,
            "interaction_count": self.interaction_count,
            "quality_score": self.quality_score,
            "centroid": self.centroid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MicroAdapter":
        return cls(
            name=data["name"],
            cluster_id=data["cluster_id"],
            context_prompt=data["context_prompt"],
            style_params=data.get("style_params", {}),
            module_affinities=data.get("module_affinities", {}),
            representative_prompts=data.get("representative_prompts", []),
            interaction_count=data.get("interaction_count", 0),
            quality_score=data.get("quality_score", 0.0),
            centroid=data.get("centroid"),
        )


class MicroAdapterEngine:
    """
    Generates and manages micro-adapters from interaction patterns.

    Lifecycle:
        1. record_interaction() — called after each turn
        2. generate_adapters() — called when enough new data (auto or /adapters generate)
        3. select_adapter() — called during routing to find best adapter
        4. apply() — modifies generation params and injects context
    """

    def __init__(self, storage_dir: str, embedding_model: str = "all-MiniLM-L6-v2",
                 min_cluster_size: int = 5, max_adapters: int = 8,
                 regenerate_interval: int = 20):
        self.storage_dir = Path(storage_dir)
        self._embedding_model_name = embedding_model
        self.min_cluster_size = min_cluster_size
        self.max_adapters = max_adapters
        self.regenerate_interval = regenerate_interval

        self._interactions: list[InteractionRecord] = []
        self._adapters: list[MicroAdapter] = []
        self._embedder = None
        self._interactions_since_regen = 0
        self._lock = threading.RLock()

        # File paths
        self._interactions_file = self.storage_dir / "adapter_interactions.json"
        self._adapters_file = self.storage_dir / "micro_adapters.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._init_embedder()
        self._load()

    def _init_embedder(self):
        """Initialize sentence-transformer (shared instance with FAISS/classifier)."""
        try:
            from engine.embedder import get_embedder
            self._embedder = get_embedder(self._embedding_model_name)
            if self._embedder:
                logger.info(f"Micro-adapter embedder: {self._embedding_model_name} (shared)")
            else:
                logger.warning("Failed to get shared embedder — micro-adapters disabled")
        except ImportError:
            logger.warning("sentence-transformers not available — micro-adapters disabled")

    def _load(self):
        """Load interactions and adapters from disk."""
        if self._interactions_file.exists():
            try:
                with open(self._interactions_file, "r") as f:
                    data = json.load(f)
                self._interactions = [InteractionRecord.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self._interactions)} adapter interactions")
            except Exception as e:
                logger.error(f"Failed to load adapter interactions: {e}")

        if self._adapters_file.exists():
            try:
                with open(self._adapters_file, "r") as f:
                    data = json.load(f)
                self._adapters = [MicroAdapter.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self._adapters)} micro-adapters")
            except Exception as e:
                logger.error(f"Failed to load micro-adapters: {e}")

    def _save_interactions(self):
        """Persist interactions to disk."""
        try:
            with open(self._interactions_file, "w") as f:
                json.dump([i.to_dict() for i in self._interactions], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save adapter interactions: {e}")

    def _save_adapters(self):
        """Persist adapters to disk."""
        try:
            with open(self._adapters_file, "w") as f:
                json.dump([a.to_dict() for a in self._adapters], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save micro-adapters: {e}")

    # ── Interaction Recording ─────────────────────────────────

    def record_interaction(self, prompt: str, response: str,
                           modules: list[str], feedback: Optional[str] = None):
        """Record an interaction for future adapter generation."""
        with self._lock:
            record = InteractionRecord(
                prompt=prompt,
                response=response,
                modules=modules,
                feedback=feedback,
                response_length=len(response),
                timestamp=time.time(),
            )
            self._interactions.append(record)
            self._interactions_since_regen += 1
            self._save_interactions()

            logger.debug(
                f"Recorded adapter interaction: '{prompt[:40]}...' "
                f"modules={modules} (total: {len(self._interactions)})"
            )

            # Auto-regenerate if enough new data
            if (self._interactions_since_regen >= self.regenerate_interval
                    and len(self._interactions) >= self.min_cluster_size * 2):
                logger.info("Auto-regenerating micro-adapters...")
                self.generate_adapters()

    def rate_last(self, feedback: str):
        """Rate the most recent interaction."""
        with self._lock:
            if self._interactions:
                self._interactions[-1].feedback = feedback
                self._save_interactions()

    # ── Adapter Generation ────────────────────────────────────

    def generate_adapters(self) -> dict:
        """
        Analyze recorded interactions and generate micro-adapters.

        Steps:
        1. Embed all interaction prompts
        2. Cluster with k-means (auto-select k)
        3. For each cluster, build a MicroAdapter
        4. Save adapters
        """
        with self._lock:
            if self._embedder is None:
                return {"error": "sentence-transformers not available"}

            # Filter to usable interactions (not rated "bad")
            usable = [i for i in self._interactions if i.feedback != "bad"]
            if len(usable) < self.min_cluster_size * 2:
                return {
                    "error": f"Need at least {self.min_cluster_size * 2} interactions "
                             f"(have {len(usable)})",
                    "interactions": len(usable),
                }

            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score

                # Step 1: Embed all prompts
                prompts = [i.prompt for i in usable]
                embeddings = self._embedder.encode(
                    prompts, normalize_embeddings=True, show_progress_bar=False,
                )

                # Step 2: Auto-select k (2 to max_adapters)
                max_k = min(self.max_adapters, len(usable) // self.min_cluster_size)
                max_k = max(max_k, 2)

                best_k = 2
                best_score = -1

                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    # Check all clusters meet minimum size
                    cluster_sizes = Counter(labels)
                    if min(cluster_sizes.values()) < self.min_cluster_size:
                        continue
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k

                # Final clustering with best k
                kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(embeddings)

                # Step 3: Build adapters from clusters
                self._adapters = []
                for cluster_id in range(best_k):
                    cluster_mask = labels == cluster_id
                    cluster_interactions = [usable[i] for i in range(len(usable)) if cluster_mask[i]]
                    cluster_embeddings = embeddings[cluster_mask]

                    adapter = self._build_adapter(
                        cluster_id=cluster_id,
                        interactions=cluster_interactions,
                        centroid=kmeans.cluster_centers_[cluster_id],
                        embeddings=cluster_embeddings,
                    )
                    self._adapters.append(adapter)

                self._save_adapters()
                self._interactions_since_regen = 0

                stats = {
                    "adapters_generated": len(self._adapters),
                    "interactions_used": len(usable),
                    "best_k": best_k,
                    "silhouette_score": round(best_score, 3),
                    "adapter_names": [a.name for a in self._adapters],
                }
                logger.info(f"Generated micro-adapters: {stats}")
                return stats

            except ImportError:
                return {"error": "scikit-learn not installed"}
            except Exception as e:
                logger.error(f"Adapter generation failed: {e}", exc_info=True)
                return {"error": str(e)}

    def _build_adapter(self, cluster_id: int, interactions: list[InteractionRecord],
                       centroid: np.ndarray, embeddings: np.ndarray) -> MicroAdapter:
        """Build a single micro-adapter from a cluster of interactions."""

        # Derive adapter name from dominant topics
        name = self._derive_name(interactions)

        # Find most representative prompts (closest to centroid)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_indices = distances.argsort()[:5]
        representative_prompts = [interactions[i].prompt for i in closest_indices]

        # Compute module affinities
        module_counts = Counter()
        for interaction in interactions:
            module_counts.update(interaction.modules)
        total = len(interactions)
        module_affinities = {
            mod: round(count / total, 3)
            for mod, count in module_counts.most_common()
        }

        # Derive style parameters from interaction characteristics
        style_params = self._derive_style(interactions)

        # Generate context injection prompt
        context_prompt = self._generate_context_prompt(
            name, interactions, module_affinities, representative_prompts,
        )

        # Quality score from feedback
        rated = [i for i in interactions if i.feedback is not None]
        if rated:
            good = sum(1 for i in rated if i.feedback == "good")
            quality_score = good / len(rated)
        else:
            quality_score = 0.5  # Neutral default

        return MicroAdapter(
            name=name,
            cluster_id=cluster_id,
            context_prompt=context_prompt,
            style_params=style_params,
            module_affinities=module_affinities,
            representative_prompts=representative_prompts,
            interaction_count=len(interactions),
            quality_score=round(quality_score, 3),
            centroid=centroid.tolist(),
        )

    def _derive_name(self, interactions: list[InteractionRecord]) -> str:
        """Derive a human-readable name from cluster interactions."""
        # Combine all prompts, extract top keywords
        all_text = " ".join(i.prompt for i in interactions).lower()
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        stop = {"the", "and", "for", "that", "this", "with", "from", "are",
                "was", "were", "been", "have", "has", "had", "not", "but",
                "what", "how", "can", "you", "your", "will", "would", "could",
                "should", "does", "write", "create", "make", "help", "want",
                "need", "please", "using", "use"}
        filtered = [w for w in words if w not in stop]
        top = Counter(filtered).most_common(3)

        if top:
            return "_".join(w for w, _ in top)
        return f"cluster_{interactions[0].modules[0] if interactions[0].modules else 'general'}"

    def _derive_style(self, interactions: list[InteractionRecord]) -> dict:
        """Derive generation style parameters from interaction patterns."""
        # Analyze response lengths to determine preferred verbosity
        lengths = [i.response_length for i in interactions if i.response_length > 0]
        avg_length = sum(lengths) / len(lengths) if lengths else 300

        # Analyze module diversity
        all_modules = [m for i in interactions for m in i.modules]
        unique_modules = len(set(all_modules))

        # Derive temperature: more diverse topics → slightly higher temperature
        # More focused (single module) → lower temperature
        if unique_modules <= 1:
            temp_adjust = -0.1  # More focused → cooler
        elif unique_modules >= 3:
            temp_adjust = 0.05  # More diverse → slightly warmer
        else:
            temp_adjust = 0.0

        # Derive max_tokens from average response length
        # Rough: 1 token ≈ 4 chars
        token_estimate = int(avg_length / 4)
        max_tokens_adjust = max(-128, min(256, token_estimate - 512))

        return {
            "temperature_adjust": round(temp_adjust, 2),
            "max_tokens_adjust": max_tokens_adjust,
            "avg_response_length": int(avg_length),
        }

    def _generate_context_prompt(self, name: str, interactions: list[InteractionRecord],
                                  module_affinities: dict, representative_prompts: list[str]) -> str:
        """Generate a context injection prompt for this adapter."""
        # Build a behavioral description from the cluster
        lines = [f"[Adapter: {name}]"]

        # Describe the dominant interaction pattern
        if module_affinities:
            top_modules = list(module_affinities.keys())[:3]
            lines.append(f"This conversation involves: {', '.join(top_modules)}")

        # Add representative examples as behavioral anchors
        lines.append("Typical requests in this context:")
        for prompt in representative_prompts[:3]:
            # Truncate long prompts
            short = prompt[:100] + "..." if len(prompt) > 100 else prompt
            lines.append(f"  - {short}")

        # Style guidance based on feedback
        good_interactions = [i for i in interactions if i.feedback == "good"]
        if good_interactions:
            # Extract patterns from positively-rated responses
            avg_good_len = sum(i.response_length for i in good_interactions) / len(good_interactions)
            if avg_good_len > 500:
                lines.append("User prefers detailed, thorough responses.")
            elif avg_good_len < 150:
                lines.append("User prefers concise, brief responses.")

        return "\n".join(lines)

    # ── Adapter Selection ─────────────────────────────────────

    def select_adapter(self, prompt: str) -> Optional[MicroAdapter]:
        """
        Find the best micro-adapter for a given prompt.

        Embeds the prompt and finds the nearest adapter centroid
        using cosine similarity.
        """
        if not self._adapters or self._embedder is None:
            return None

        with self._lock:
            # Embed the prompt
            prompt_embedding = self._embedder.encode(
                [prompt], normalize_embeddings=True, show_progress_bar=False,
            )[0]

            # Find nearest adapter centroid
            best_adapter = None
            best_similarity = -1

            for adapter in self._adapters:
                if adapter.centroid is None:
                    continue
                centroid = np.array(adapter.centroid)
                # Cosine similarity (vectors are normalized)
                similarity = float(np.dot(prompt_embedding, centroid))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_adapter = adapter

            # Only return if similarity is meaningful (> 0.3)
            if best_adapter and best_similarity > 0.3:
                logger.debug(
                    f"Selected adapter '{best_adapter.name}' "
                    f"(similarity={best_similarity:.3f})"
                )
                return best_adapter

            return None

    def apply(self, adapter: MicroAdapter, base_temperature: float,
              base_max_tokens: int) -> dict:
        """
        Apply adapter modifications to generation parameters.

        Returns dict with adjusted parameters:
            - temperature: adjusted temperature
            - max_tokens: adjusted max tokens
            - context_prompt: text to inject into prompt
        """
        temp = base_temperature + adapter.style_params.get("temperature_adjust", 0)
        temp = max(0.1, min(1.5, temp))  # Clamp

        max_tok = base_max_tokens + adapter.style_params.get("max_tokens_adjust", 0)
        max_tok = max(64, min(2048, max_tok))  # Clamp

        return {
            "temperature": round(temp, 2),
            "max_tokens": max_tok,
            "context_prompt": adapter.context_prompt,
            "adapter_name": adapter.name,
            "module_affinities": adapter.module_affinities,
        }

    # ── Status & Info ─────────────────────────────────────────

    @property
    def adapter_count(self) -> int:
        return len(self._adapters)

    @property
    def interaction_count(self) -> int:
        return len(self._interactions)

    def status(self) -> dict:
        """Get micro-adapter engine status."""
        return {
            "adapters": len(self._adapters),
            "interactions": len(self._interactions),
            "since_regen": self._interactions_since_regen,
            "regen_interval": self.regenerate_interval,
            "adapter_names": [a.name for a in self._adapters],
            "adapter_details": [
                {
                    "name": a.name,
                    "interactions": a.interaction_count,
                    "quality": a.quality_score,
                    "modules": list(a.module_affinities.keys()),
                }
                for a in self._adapters
            ],
        }

    def list_adapters(self) -> list[dict]:
        """List all adapters with details."""
        return [
            {
                "name": a.name,
                "cluster_id": a.cluster_id,
                "interaction_count": a.interaction_count,
                "quality_score": a.quality_score,
                "module_affinities": a.module_affinities,
                "representative_prompts": a.representative_prompts[:3],
                "style": a.style_params,
            }
            for a in self._adapters
        ]

    def clear(self):
        """Clear all interactions and adapters."""
        with self._lock:
            self._interactions.clear()
            self._adapters.clear()
            self._interactions_since_regen = 0
            self._save_interactions()
            self._save_adapters()
