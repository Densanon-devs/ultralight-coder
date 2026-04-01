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
    seed_threshold: float = 0.35,
    max_seed_categories: int = 3,
    max_expanded_categories: int = 5,
) -> list:
    """Graph-enhanced example retrieval with confidence-gated expansion.

    Key insight from benchmarks: graph expansion helps weak models on composite
    tasks but hurts on simple single-pattern tasks by injecting noise. Solution:
    only expand when the query looks composite (multiple categories close in
    similarity). Single-pattern queries get focused retrieval from just the
    seed category + its direct dependencies.

    Steps:
    1. Check failure patterns first (with graph-aware expansion)
    2. Embed query, compute similarities to all examples
    3. Identify seed categories from similarity scores
    4. Decide expansion strategy based on seed confidence gap:
       - Strong single seed (dominant category) → minimal expansion (deps only)
       - Multiple close seeds (composite task) → full graph expansion
    5. Filter examples to expanded category set
    6. Pick top_k by similarity from the filtered set

    Returns:
        List of SolvedExample objects (up to top_k)
    """
    if not embedder or example_embeddings is None or len(examples) == 0:
        return examples[:top_k]

    # Step 0: Failure-aware routing with graph expansion
    if failure_patterns:
        forced = _check_failure_patterns_with_graph(
            query, examples, embedder, example_embeddings, failure_patterns,
            graph, top_k,
        )
        if forced:
            return forced

    # Step 1: Embed and score
    query_vec = embedder.encode([query], normalize_embeddings=True)
    sims = np.dot(example_embeddings, query_vec.T).flatten()
    ranked = sims.argsort()[::-1]

    # Step 2: Identify seed categories
    cat_best: dict[str, float] = {}
    for idx in ranked:
        cat = examples[idx].category
        if cat and cat not in cat_best:
            cat_best[cat] = float(sims[idx])
            if len(cat_best) >= max_seed_categories * 3:
                break

    # Filter to those above threshold
    seeds = sorted(
        [(cat, score) for cat, score in cat_best.items() if score >= seed_threshold],
        key=lambda x: x[1],
        reverse=True,
    )[:max_seed_categories]

    if not seeds:
        return []

    seed_names = [cat for cat, _ in seeds]

    # Step 3: Confidence-gated expansion
    # If the top seed dominates (big gap to #2), this is a single-pattern task.
    # Only expand to direct dependencies, skip related_to.
    # If seeds are close together, it's composite — do full expansion.
    if len(seeds) >= 2:
        gap = seeds[0][1] - seeds[1][1]
    else:
        gap = 1.0  # single seed = treat as dominant

    if gap > 0.08:
        # Dominant single category — deps + top 1 related per seed
        expanded = list(seed_names)
        seen = set(expanded)
        for cat in seed_names:
            for dep, weight in graph.get_dependencies(cat, depth=1):
                if dep not in seen:
                    expanded.append(dep)
                    seen.add(dep)
            # Add strongest related (provides usage context)
            related = graph.get_related(cat, depth=1)
            if related and related[0][0] not in seen:
                expanded.append(related[0][0])
                seen.add(related[0][0])
        # Cap at seeds + 3 max
        expanded = expanded[:len(seed_names) + 3]
        mode = "focused"
    else:
        # Composite task — full graph expansion
        expanded = graph.get_neighborhood(
            seed_names,
            max_depth=1,
            max_categories=max_expanded_categories,
        )
        mode = "expanded"

    expanded_set = set(expanded)

    logger.debug(
        f"Graph retrieval [{mode}]: seeds={seeds[:3]} -> expanded={expanded} "
        f"(gap={gap:.3f}, +{len(expanded) - len(seed_names)} from graph)"
    )

    # Step 4+5: Filter and select
    # Seed categories use full threshold, expanded categories use lower
    expanded_threshold = seed_threshold * 0.65
    seed_set = set(seed_names)
    selected = []
    for idx in ranked:
        if len(selected) >= top_k:
            break
        ex = examples[idx]
        if ex.category not in expanded_set:
            continue
        threshold = seed_threshold if ex.category in seed_set else expanded_threshold
        if sims[idx] >= threshold:
            selected.append(ex)

    return selected


def graph_rerank_examples(
    query: str,
    embedder,
    example_embeddings: np.ndarray,
    examples: list,
    graph: PatternGraph,
    failure_patterns: Optional[dict] = None,
    top_k: int = 2,
    flat_candidates: int = 5,
    min_sim_best: float = 0.50,
) -> list:
    """Graph-as-reranker: flat retrieval for candidates, graph for scoring.

    Instead of using the graph to *expand* the candidate set (which adds noise),
    use flat similarity to retrieve a broad candidate pool, then use graph
    structural coherence to *rerank* and filter down to 1-2 high-confidence picks.

    Steps:
    1. Check failure patterns first (same as flat — no graph expansion here)
    2. Flat-retrieve top `flat_candidates` examples by cosine similarity
    3. For each candidate, compute a graph coherence score:
       - Does its category appear in the graph neighborhood of other candidates?
       - Does it have depends_on edges to/from other candidate categories?
       - Higher score = more structurally coherent with the query's pattern family
    4. Rerank by: (0.7 * similarity) + (0.3 * graph_coherence)
    5. Return top_k (default 2) from the reranked list

    Key insight: graph intelligence without graph-driven expansion. The graph
    filters *out* noise rather than bringing *in* neighbors.
    """
    if not embedder or example_embeddings is None or len(examples) == 0:
        return examples[:top_k]

    # Step 0: Failure-aware routing (flat-style, no graph expansion)
    if failure_patterns:
        q = query.lower()
        matched_categories: list[str] = []
        for category, triggers in failure_patterns.items():
            if any(trigger in q for trigger in triggers):
                matched_categories.append(category)
        if matched_categories:
            # Pick best per matched category by similarity, sorted by relevance
            query_vec = embedder.encode([query], normalize_embeddings=True)
            sims = np.dot(example_embeddings, query_vec.T).flatten()
            scored = []
            for cat in matched_categories:
                cat_examples = [(i, ex) for i, ex in enumerate(examples) if ex.category == cat]
                if cat_examples:
                    best_idx, best_ex = max(cat_examples, key=lambda x: sims[x[0]])
                    scored.append((best_ex, float(sims[best_idx])))
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored:
                return [ex for ex, _ in scored][:top_k]

    # Step 1: Embed and flat-rank
    query_vec = embedder.encode([query], normalize_embeddings=True)
    sims = np.dot(example_embeddings, query_vec.T).flatten()
    ranked = sims.argsort()[::-1]

    # Gate: reject if best match is too weak
    if sims[ranked[0]] < min_sim_best:
        return []

    # Step 2: Collect top flat_candidates
    candidates = []
    for idx in ranked[:flat_candidates * 2]:  # oversample to allow threshold filtering
        if sims[idx] < min_sim_best * 0.8:  # don't consider very weak matches
            break
        candidates.append((int(idx), float(sims[idx])))
        if len(candidates) >= flat_candidates:
            break

    if not candidates:
        return []

    # Step 3: Compute graph coherence for each candidate
    candidate_categories = [examples[idx].category for idx, _ in candidates]
    unique_cats = list(set(candidate_categories))

    # Build a "neighborhood set" from the top candidate's category
    top_cat = examples[candidates[0][0]].category
    neighborhood = set()
    if top_cat:
        neighborhood.add(top_cat)
        for dep, _ in graph.get_dependencies(top_cat, depth=1):
            neighborhood.add(dep)
        for rel, _ in graph.get_related(top_cat, depth=1):
            neighborhood.add(rel)

    reranked = []
    for idx, sim_score in candidates:
        cat = examples[idx].category
        # Graph coherence: is this candidate's category in the neighborhood?
        coherence = 0.0
        if cat in neighborhood:
            coherence = 1.0
        elif cat:
            # Check if this candidate has *any* graph connection to the top category
            cat_deps = {d for d, _ in graph.get_dependencies(cat, depth=1)}
            cat_rels = {r for r, _ in graph.get_related(cat, depth=1)}
            if top_cat in cat_deps:
                coherence = 0.8  # depends on the top pattern
            elif top_cat in cat_rels:
                coherence = 0.5  # related to the top pattern
            # Also check if it connects to any other candidate's category
            elif cat_deps & set(unique_cats):
                coherence = 0.4
            elif cat_rels & set(unique_cats):
                coherence = 0.3

        combined = 0.7 * sim_score + 0.3 * coherence
        reranked.append((idx, combined, sim_score, coherence))

    # Sort by combined score
    reranked.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        f"Graph rerank: top_cat={top_cat}, neighborhood={len(neighborhood)} cats, "
        f"candidates={len(candidates)} -> reranked top: "
        f"{[(examples[idx].category, f'{comb:.3f}={sim:.3f}+{coh:.3f}') for idx, comb, sim, coh in reranked[:top_k]]}"
    )

    return [examples[idx] for idx, _, _, _ in reranked[:top_k]]


def graph_plan_examples(
    query: str,
    embedder,
    example_embeddings: np.ndarray,
    examples: list,
    graph: PatternGraph,
    failure_patterns: Optional[dict] = None,
    min_sim_best: float = 0.50,
) -> list:
    """Graph-as-planner: graph identifies subpattern needs, injects only 1 example.

    The graph's role is purely analytical — it identifies what conceptual
    subpatterns are involved in the query. But instead of injecting examples
    for each subpattern (which causes noise), it picks only the single most
    decisive concrete example.

    Steps:
    1. Check failure patterns first (flat-style)
    2. Embed query, identify the top seed category
    3. Use graph to identify the full pattern family (deps + related)
    4. Among ALL examples in the pattern family, pick the single one with
       highest similarity to the query
    5. Return exactly 1 example (or 0 if nothing passes threshold)

    Key insight: graph expands the *search space* for the best single match,
    but never injects more than one example. This gives the model the single
    most authoritative blueprint without overcrowding its context.
    """
    if not embedder or example_embeddings is None or len(examples) == 0:
        return examples[:1]

    # Step 0: Failure-aware routing (flat-style, no graph expansion)
    if failure_patterns:
        q = query.lower()
        matched_categories: list[str] = []
        for category, triggers in failure_patterns.items():
            if any(trigger in q for trigger in triggers):
                matched_categories.append(category)
        if matched_categories:
            query_vec = embedder.encode([query], normalize_embeddings=True)
            sims = np.dot(example_embeddings, query_vec.T).flatten()
            # Even with failure patterns, pick only the single best
            best_sim = -1.0
            best_ex = None
            for cat in matched_categories:
                for i, ex in enumerate(examples):
                    if ex.category == cat and sims[i] > best_sim:
                        best_sim = sims[i]
                        best_ex = ex
            if best_ex:
                return [best_ex]

    # Step 1: Embed and find seed category
    query_vec = embedder.encode([query], normalize_embeddings=True)
    sims = np.dot(example_embeddings, query_vec.T).flatten()
    ranked = sims.argsort()[::-1]

    if sims[ranked[0]] < min_sim_best:
        return []

    # Step 2: Identify seed category from the best flat match
    seed_cat = examples[ranked[0]].category

    # Step 3: Graph-expand the search space (but NOT the injection set)
    search_cats = {seed_cat}
    if seed_cat:
        for dep, _ in graph.get_dependencies(seed_cat, depth=1):
            search_cats.add(dep)
        for rel, _ in graph.get_related(seed_cat, depth=1):
            search_cats.add(rel)

    # Step 4: Find the single best example across the entire pattern family
    best_idx = -1
    best_sim = -1.0
    for idx in ranked:
        ex = examples[idx]
        if ex.category in search_cats and sims[idx] > best_sim:
            best_idx = int(idx)
            best_sim = float(sims[idx])
            break  # ranked is sorted, so first match in family is the best

    if best_idx < 0 or best_sim < min_sim_best:
        return []

    logger.debug(
        f"Graph plan: seed={seed_cat}, search_space={len(search_cats)} cats, "
        f"selected={examples[best_idx].category} (sim={best_sim:.3f})"
    )

    # Return exactly 1
    return [examples[best_idx]]


def _check_failure_patterns_with_graph(
    query: str,
    examples: list,
    embedder,
    example_embeddings: np.ndarray,
    failure_patterns: dict,
    graph: PatternGraph,
    top_k: int,
) -> list:
    """Failure-aware routing with confidence-gated graph expansion.

    Single matched category → focused: only add direct dependencies (prevents
    noise on simple single-pattern tasks like Timer).
    Multiple matched categories → expanded: full graph neighborhood (composite
    tasks like "router with rate limiting").
    """
    q = query.lower()
    matched_categories: list[str] = []

    for category, triggers in failure_patterns.items():
        if any(trigger in q for trigger in triggers):
            matched_categories.append(category)

    if not matched_categories:
        return []

    # Confidence-gated expansion based on number of matches
    if len(matched_categories) == 1:
        # Single match → focused: deps + top 1 related
        expanded = list(matched_categories)
        seen = set(expanded)
        for dep, weight in graph.get_dependencies(matched_categories[0], depth=1):
            if dep not in seen:
                expanded.append(dep)
                seen.add(dep)
        # Add the single strongest related_to (provides usage context)
        related = graph.get_related(matched_categories[0], depth=1)
        if related and related[0][0] not in seen:
            expanded.append(related[0][0])
        mode = "focused"
    else:
        # Multiple matches → full graph expansion
        expanded = graph.get_neighborhood(
            matched_categories,
            max_depth=1,
            max_categories=max(top_k + 1, len(matched_categories) + 2),
        )
        mode = "expanded"

    expanded_set = set(expanded)

    logger.debug(
        f"Failure-aware graph [{mode}]: matched={matched_categories} -> expanded={expanded}"
    )

    # Collect candidates from expanded categories
    candidates: dict[str, list[tuple[int, object]]] = {cat: [] for cat in expanded_set}
    for i, ex in enumerate(examples):
        if ex.category in candidates:
            candidates[ex.category].append((i, ex))

    # Pick best per category using similarity, prioritizing matched categories first
    query_vec = embedder.encode([query], normalize_embeddings=True)
    sims = np.dot(example_embeddings, query_vec.T).flatten()

    forced = []
    # First: matched categories (the failure patterns)
    for cat in matched_categories:
        if not candidates.get(cat):
            continue
        best_idx, best_ex = max(candidates[cat], key=lambda x: sims[x[0]])
        forced.append(best_ex)

    # Then: graph-expanded categories (prerequisites/related)
    # Apply similarity threshold — don't inject low-relevance expanded examples
    expanded_min_sim = 0.35
    for cat in expanded:
        if cat in matched_categories:
            continue  # already added
        if not candidates.get(cat):
            continue
        if len(forced) >= top_k:
            break
        best_idx, best_ex = max(candidates[cat], key=lambda x: sims[x[0]])
        if sims[best_idx] < expanded_min_sim:
            logger.debug(f"Failure-aware graph: skipping {cat} (best sim={sims[best_idx]:.3f} < {expanded_min_sim})")
            continue
        forced.append(best_ex)

    return forced[:top_k]
