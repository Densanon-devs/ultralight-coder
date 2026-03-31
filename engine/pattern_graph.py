"""
Pattern Graph — Dependency-aware retrieval for augmentor examples.

Loads a YAML graph of category relationships (depends_on, related_to) and uses
it to expand retrieval beyond flat cosine similarity. When a query matches
category A, the graph pulls in A's prerequisites and related categories,
producing a focused set of examples that covers the structural knowledge
the model needs.

Runs in parallel with the existing flat retrieval — not a replacement.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraphEdge:
    """An edge in the pattern graph."""
    target: str
    weight: float
    edge_type: str  # "depends_on" or "related_to"


@dataclass
class GraphNode:
    """A node in the pattern graph."""
    category: str
    description: str
    edges: list[GraphEdge] = field(default_factory=list)


class PatternGraph:
    """Directed graph of pattern category relationships.

    Loaded from data/pattern_graph.yaml. Used to expand example retrieval
    beyond flat similarity by traversing dependency and affinity edges.
    """

    def __init__(self, yaml_path: str = "data/pattern_graph.yaml"):
        self.nodes: dict[str, GraphNode] = {}
        self._load(yaml_path)

    def _load(self, yaml_path: str):
        """Parse YAML into nodes and edges."""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed, pattern graph unavailable")
            return

        path = Path(yaml_path)
        if not path.exists():
            logger.warning(f"Pattern graph not found at {path}")
            return

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "nodes" not in data:
            logger.warning(f"Invalid pattern graph format in {path}")
            return

        # First pass: create all nodes
        for cat_name, node_data in data["nodes"].items():
            self.nodes[cat_name] = GraphNode(
                category=cat_name,
                description=node_data.get("description", ""),
            )

        # Second pass: build edges (validate targets exist)
        for cat_name, node_data in data["nodes"].items():
            node = self.nodes[cat_name]

            for dep in node_data.get("depends_on", []):
                target = dep["category"]
                if target not in self.nodes:
                    logger.warning(f"Graph edge {cat_name} -> {target}: target not found, skipping")
                    continue
                node.edges.append(GraphEdge(
                    target=target,
                    weight=dep.get("weight", 0.5),
                    edge_type="depends_on",
                ))

            for rel in node_data.get("related_to", []):
                target = rel["category"]
                if target not in self.nodes:
                    logger.warning(f"Graph edge {cat_name} -> {target}: target not found, skipping")
                    continue
                node.edges.append(GraphEdge(
                    target=target,
                    weight=rel.get("weight", 0.5),
                    edge_type="related_to",
                ))

        logger.info(
            f"Pattern graph loaded: {len(self.nodes)} nodes, "
            f"{sum(len(n.edges) for n in self.nodes.values())} edges"
        )

    def get_dependencies(self, category: str, depth: int = 1) -> list[tuple[str, float]]:
        """BFS along depends_on edges up to `depth` hops.

        Returns [(category, combined_weight), ...] sorted by weight descending.
        """
        if category not in self.nodes:
            return []

        visited: dict[str, float] = {}
        queue: deque[tuple[str, float, int]] = deque()

        # Seed from the starting node's depends_on edges
        for edge in self.nodes[category].edges:
            if edge.edge_type == "depends_on":
                queue.append((edge.target, edge.weight, 1))

        while queue:
            target, weight, d = queue.popleft()
            if target in visited:
                visited[target] = max(visited[target], weight)
                continue
            visited[target] = weight

            if d < depth and target in self.nodes:
                for edge in self.nodes[target].edges:
                    if edge.edge_type == "depends_on" and edge.target not in visited:
                        queue.append((edge.target, weight * edge.weight, d + 1))

        return sorted(visited.items(), key=lambda x: x[1], reverse=True)

    def get_related(self, category: str, depth: int = 1) -> list[tuple[str, float]]:
        """BFS along related_to edges up to `depth` hops.

        Returns [(category, combined_weight), ...] sorted by weight descending.
        """
        if category not in self.nodes:
            return []

        visited: dict[str, float] = {}
        queue: deque[tuple[str, float, int]] = deque()

        for edge in self.nodes[category].edges:
            if edge.edge_type == "related_to":
                queue.append((edge.target, edge.weight, 1))

        while queue:
            target, weight, d = queue.popleft()
            if target in visited:
                visited[target] = max(visited[target], weight)
                continue
            visited[target] = weight

            if d < depth and target in self.nodes:
                for edge in self.nodes[target].edges:
                    if edge.edge_type == "related_to" and edge.target not in visited:
                        queue.append((edge.target, weight * edge.weight, d + 1))

        return sorted(visited.items(), key=lambda x: x[1], reverse=True)

    def get_neighborhood(self, categories: list[str],
                         max_depth: int = 1,
                         max_categories: int = 4) -> list[str]:
        """Expand seed categories via graph traversal.

        Algorithm:
        1. Start with seed categories
        2. Always include depends_on targets (prerequisites)
        3. Add related_to targets ranked by weight
        4. Cap at max_categories total

        Returns ordered list: seeds first, then prerequisites, then related.
        """
        result = list(categories)
        seen = set(categories)

        # Collect all prerequisites (always include these)
        deps: list[tuple[str, float]] = []
        for cat in categories:
            for target, weight in self.get_dependencies(cat, depth=max_depth):
                if target not in seen:
                    deps.append((target, weight))

        # Deduplicate deps, keep highest weight
        dep_map: dict[str, float] = {}
        for target, weight in deps:
            dep_map[target] = max(dep_map.get(target, 0), weight)

        # Add deps sorted by weight
        for target, _ in sorted(dep_map.items(), key=lambda x: x[1], reverse=True):
            if len(result) >= max_categories:
                break
            result.append(target)
            seen.add(target)

        # Fill remaining slots with related_to
        if len(result) < max_categories:
            related: dict[str, float] = {}
            for cat in categories:
                for target, weight in self.get_related(cat, depth=max_depth):
                    if target not in seen:
                        related[target] = max(related.get(target, 0), weight)

            for target, _ in sorted(related.items(), key=lambda x: x[1], reverse=True):
                if len(result) >= max_categories:
                    break
                result.append(target)
                seen.add(target)

        return result

    def categories(self) -> list[str]:
        """Return all known category names."""
        return list(self.nodes.keys())


def graph_retrieve_examples(
    query: str,
    embedder,
    example_embeddings: np.ndarray,
    examples: list,
    graph: PatternGraph,
    failure_patterns: Optional[dict] = None,
    top_k: int = 3,
    seed_threshold: float = 0.40,
    max_seed_categories: int = 2,
    max_expanded_categories: int = 4,
) -> list:
    """Graph-enhanced example retrieval.

    Steps:
    1. Check failure patterns first (same as flat — keeps the safety net)
    2. Embed query, compute similarities to all examples
    3. Identify top seed categories from similarity scores
    4. Expand via graph neighborhood (prerequisites + related)
    5. Filter examples to expanded category set
    6. Pick top_k by similarity from the filtered set

    Args:
        query: User's input prompt
        embedder: Sentence-transformers encoder
        example_embeddings: Pre-computed example embeddings (N x D)
        examples: List of SolvedExample objects
        graph: PatternGraph instance
        failure_patterns: Optional dict of category -> trigger keywords
        top_k: Max examples to return
        seed_threshold: Min similarity to consider a category as seed
        max_seed_categories: Max seed categories to extract
        max_expanded_categories: Max total categories after graph expansion

    Returns:
        List of SolvedExample objects (up to top_k)
    """
    if not embedder or example_embeddings is None or len(examples) == 0:
        return examples[:top_k]

    # Step 0: Failure-aware routing (reuse existing safety net)
    if failure_patterns:
        forced = _check_failure_patterns_graph(
            query, examples, embedder, example_embeddings, failure_patterns
        )
        if forced:
            return forced[:top_k]

    # Step 1: Embed and score
    query_vec = embedder.encode([query], normalize_embeddings=True)
    sims = np.dot(example_embeddings, query_vec.T).flatten()
    ranked = sims.argsort()[::-1]

    # Step 2: Identify seed categories
    # Find the best similarity per category
    cat_best: dict[str, float] = {}
    for idx in ranked:
        cat = examples[idx].category
        if cat and cat not in cat_best:
            cat_best[cat] = float(sims[idx])
            if len(cat_best) >= max_seed_categories * 2:
                break  # enough candidates scanned

    # Filter to those above threshold, take top N
    seeds = sorted(
        [(cat, score) for cat, score in cat_best.items() if score >= seed_threshold],
        key=lambda x: x[1],
        reverse=True,
    )[:max_seed_categories]

    if not seeds:
        return []

    seed_names = [cat for cat, _ in seeds]

    # Step 3: Expand via graph
    expanded = graph.get_neighborhood(
        seed_names,
        max_depth=1,
        max_categories=max_expanded_categories,
    )
    expanded_set = set(expanded)

    logger.debug(
        f"Graph retrieval: seeds={seed_names} -> expanded={expanded} "
        f"(+{len(expanded) - len(seed_names)} from graph)"
    )

    # Step 4: Filter examples to expanded categories
    # Step 5: Pick top_k by similarity from filtered set
    selected = []
    for idx in ranked:
        if len(selected) >= top_k:
            break
        ex = examples[idx]
        if ex.category in expanded_set and sims[idx] >= seed_threshold:
            selected.append(ex)

    return selected


def _check_failure_patterns_graph(
    query: str,
    examples: list,
    embedder,
    example_embeddings: np.ndarray,
    failure_patterns: dict,
) -> list:
    """Check failure patterns (shared logic with flat retrieval)."""
    q = query.lower()
    matched_categories: list[str] = []

    for category, triggers in failure_patterns.items():
        if any(trigger in q for trigger in triggers):
            matched_categories.append(category)

    if not matched_categories:
        return []

    # Collect candidates from matched categories
    candidates: dict[str, list[tuple[int, object]]] = {cat: [] for cat in matched_categories}
    for i, ex in enumerate(examples):
        if ex.category in candidates:
            candidates[ex.category].append((i, ex))

    # Pick best per category using similarity
    forced = []
    query_vec = embedder.encode([query], normalize_embeddings=True)
    sims = np.dot(example_embeddings, query_vec.T).flatten()

    for cat in matched_categories:
        if not candidates[cat]:
            continue
        best_idx, best_ex = max(candidates[cat], key=lambda x: sims[x[0]])
        forced.append(best_ex)

    return forced
