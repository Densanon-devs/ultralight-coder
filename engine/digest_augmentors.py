"""
Digest Curation Augmentors — Multi-pass on-device digest curation.

Uses the augmentor system to make a small (0.5B) GGUF model curate news
digests through three narrowly-scoped passes:
  Pass 1: Selection — pick 8-10 best articles, assign categories
  Pass 2: Takeaways — write 2-4 bullets per selected article
  Pass 3: Highlights — write 1-2 sentence intro summarizing themes

Each pass has its own GBNF grammar, YAML example bank, and verifier.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.augmentors import Augmentor, SolvedExample, AugmentorResult  # KEEP - unique to ultralight-coder
from densanon.core.example_loader import load_domain_examples, to_solved_examples

logger = logging.getLogger(__name__)

# ── Failure-Aware Routing for Curation ──────────────────────

CURATION_FAILURE_PATTERNS: dict[str, list[str]] = {
    "curation_selection": [
        "select articles", "curate", "pick the best", "choose articles",
        "digest selection", "select the", "most important articles",
        "assign categories", "assign each a category",
    ],
    "curation_takeaway_research": [
        "takeaway", "key points", "bullet points", "insights for",
        "takeaways for this article", "write 2-4 takeaway",
        "arxiv", "research paper", "abstract:",
    ],
    "curation_takeaway_industry": [
        "takeaway", "key points", "bullet points",
        "product launch", "funding", "acquisition", "layoff",
        "industry news", "model release",
    ],
    "curation_takeaway_tools": [
        "takeaway", "key points", "bullet points",
        "library release", "framework update", "tool release",
        "maintenance release", "api change",
    ],
    "curation_takeaway_community": [
        "takeaway", "key points", "bullet points",
        "upvotes", "comments", "reddit", "hacker news",
        "show hn", "community",
    ],
    "curation_takeaway_general": [
        "takeaway", "key points", "bullet points",
    ],
    "curation_highlights": [
        "highlights intro", "summarize themes", "digest summary",
        "opening paragraph", "today's digest", "write a 1-2 sentence",
    ],
}


# ── Verifiers ───────────────────────────────────────────────

def verify_selection(response: str, query: str) -> tuple[bool, str]:
    """Verify article selection output."""
    try:
        data = json.loads(response.strip())
    except json.JSONDecodeError:
        return False, "Response is not valid JSON."

    if "selected" not in data:
        return False, "Missing 'selected' key."

    selected = data["selected"]
    if not isinstance(selected, list):
        return False, "'selected' must be a list."

    if len(selected) < 4:
        return False, f"Only {len(selected)} articles selected, need at least 4."

    if len(selected) > 15:
        return False, f"{len(selected)} articles selected, max is 15."

    # Check for required fields
    for i, item in enumerate(selected):
        if "index" not in item:
            return False, f"Item {i} missing 'index'."
        if "category" not in item:
            return False, f"Item {i} missing 'category'."
        if not isinstance(item["index"], int):
            return False, f"Item {i} 'index' must be integer."

    # Check for duplicate indices
    indices = [item["index"] for item in selected]
    if len(set(indices)) != len(indices):
        return False, "Duplicate article indices."

    # Check category diversity (at least 2 categories when 6+ articles)
    categories = set(item["category"] for item in selected)
    if len(selected) >= 6 and len(categories) < 2:
        return False, "Only one category used. Need at least 2 for diversity."

    # No single category should exceed 70% of selections
    from collections import Counter
    cat_counts = Counter(item["category"] for item in selected)
    max_count = max(cat_counts.values())
    if max_count / len(selected) > 0.70:
        dominant = cat_counts.most_common(1)[0][0]
        return False, f"Category '{dominant}' has {max_count}/{len(selected)} articles (>70%)."

    return True, ""


def verify_takeaways(response: str, query: str) -> tuple[bool, str]:
    """Verify takeaway generation output."""
    try:
        data = json.loads(response.strip())
    except json.JSONDecodeError:
        return False, "Response is not valid JSON."

    if "takeaways" not in data:
        return False, "Missing 'takeaways' key."

    takeaways = data["takeaways"]
    if not isinstance(takeaways, list):
        return False, "'takeaways' must be a list."

    if len(takeaways) < 2:
        return False, f"Only {len(takeaways)} takeaways, need at least 2."

    if len(takeaways) > 5:
        return False, f"{len(takeaways)} takeaways is too many, max 5."

    for i, t in enumerate(takeaways):
        if not isinstance(t, str):
            return False, f"Takeaway {i} is not a string."
        if len(t) < 15:
            return False, f"Takeaway {i} is too short ({len(t)} chars)."
        if len(t) > 500:
            return False, f"Takeaway {i} is too long ({len(t)} chars)."

    # No meta-language
    meta_patterns = ["as an ai", "here are the takeaways", "i'll provide",
                     "let me summarize", "in summary,"]
    response_lower = response.lower()
    for pattern in meta_patterns:
        if pattern in response_lower:
            return False, f"Contains meta-language: '{pattern}'."

    return True, ""


def verify_highlights(response: str, query: str) -> tuple[bool, str]:
    """Verify highlights intro output."""
    try:
        data = json.loads(response.strip())
    except json.JSONDecodeError:
        return False, "Response is not valid JSON."

    if "highlights_intro" not in data:
        return False, "Missing 'highlights_intro' key."

    intro = data["highlights_intro"]
    if not isinstance(intro, str):
        return False, "'highlights_intro' must be a string."

    if len(intro) < 40:
        return False, f"Intro too short ({len(intro)} chars)."

    if len(intro) > 800:
        return False, f"Intro too long ({len(intro)} chars)."

    # No meta-language
    meta_patterns = ["as an ai", "here is the", "i'll write", "let me"]
    intro_lower = intro.lower()
    for pattern in meta_patterns:
        if pattern in intro_lower:
            return False, f"Contains meta-language: '{pattern}'."

    return True, ""


# ── Grammar Loading ─────────────────────────────────────────

def load_grammar(grammar_path: str) -> Optional[str]:
    """Load a GBNF grammar file and return as string."""
    path = Path(grammar_path)
    if not path.exists():
        logger.warning(f"Grammar file not found: {path}")
        return None
    return path.read_text(encoding="utf-8")


# ── DigestAugmentorRouter ──────────────────────────────────

class DigestAugmentorRouter:
    """Manages the three curation augmentors (selection, takeaway, highlights).

    Loads YAML examples from data/augmentor_examples/curation/,
    applies GBNF grammars, and exposes per-pass processing methods.
    """

    def __init__(self,
                 examples_dir: str = "data/augmentor_examples/curation",
                 grammars_dir: str = "data/digest_grammars",
                 max_retries: int = 2):
        self.examples_dir = examples_dir
        self.grammars_dir = grammars_dir
        self.max_retries = max_retries
        self._embedder = None

        # Load grammars
        self.selection_grammar = load_grammar(f"{grammars_dir}/selection.gbnf")
        self.takeaways_grammar = load_grammar(f"{grammars_dir}/takeaways.gbnf")
        self.highlights_grammar = load_grammar(f"{grammars_dir}/highlights.gbnf")

        # Load examples and build augmentors
        self._build_augmentors()

    def _build_augmentors(self):
        """Load YAML examples and create the three augmentor instances."""
        raw = load_domain_examples(
            str(Path(self.examples_dir).parent),
            Path(self.examples_dir).name,
        )
        all_examples = to_solved_examples(raw)

        # Split by category
        selection_ex = [e for e in all_examples
                        if e.category in ("curation_selection", "curation_selection_edge")]
        takeaway_ex = [e for e in all_examples
                       if e.category.startswith("curation_takeaway")]
        highlights_ex = [e for e in all_examples
                         if e.category == "curation_highlights"]

        logger.info(f"Digest augmentors: {len(selection_ex)} selection, "
                     f"{len(takeaway_ex)} takeaway, {len(highlights_ex)} highlights examples")

        # Selection augmentor
        self.selection = Augmentor(
            name="digest_selection",
            system_context=(
                "You are a news digest curator. Select the most important and "
                "diverse articles from the list. Skip low-quality, promotional, "
                "or redundant content. Assign each to the most fitting category."
            ),
            examples=selection_ex,
            verifier=verify_selection,
            grammar_str=self.selection_grammar,
            max_examples=1,  # rerank1 mode — one example for 0.5B
            max_retries=self.max_retries,
        )

        # Takeaway augmentor
        self.takeaway = Augmentor(
            name="digest_takeaway",
            system_context=(
                "You write concise, insightful takeaway bullets for news articles. "
                "Each takeaway should explain what happened, why it matters, and "
                "what the implications are. Be specific, not generic."
            ),
            examples=takeaway_ex,
            verifier=verify_takeaways,
            grammar_str=self.takeaways_grammar,
            max_examples=1,
            max_retries=self.max_retries,
        )

        # Highlights augmentor
        self.highlights = Augmentor(
            name="digest_highlights",
            system_context=(
                "You write a 1-2 sentence highlights intro for a daily news digest. "
                "Identify the key themes and most notable items from the selected "
                "articles. Be specific about what makes today's digest notable."
            ),
            examples=highlights_ex,
            verifier=verify_highlights,
            grammar_str=self.highlights_grammar,
            max_examples=1,
            max_retries=self.max_retries,
        )

    def init_embeddings(self, embedder):
        """Initialize embeddings for all three augmentors."""
        self._embedder = embedder
        self.selection.init_embeddings(embedder)
        self.takeaway.init_embeddings(embedder)
        self.highlights.init_embeddings(embedder)
        logger.info("Digest augmentor embeddings initialized")

    def process_selection(self, compact_articles: str, model,
                          chat_format: str = "chatml",
                          categories: Optional[list[str]] = None) -> dict:
        """Pass 1: Select best articles and assign categories.

        Args:
            compact_articles: Numbered article list in compact format
            model: BaseModel instance
            chat_format: Chat template format
            categories: Available categories (for the prompt)

        Returns:
            Parsed selection dict: {"selected": [{"index": N, "category": "..."}]}
        """
        cat_str = ", ".join(categories) if categories else "research, industry, tools, community"
        query = (
            f"Select the 8-10 most important articles and assign each a category.\n"
            f"Categories: {cat_str}\n"
            f"Articles:\n{compact_articles}"
        )
        return self._run_pass(self.selection, query, model, chat_format)

    def process_takeaways(self, title: str, source: str, summary: str,
                          model, chat_format: str = "chatml") -> dict:
        """Pass 2: Generate takeaways for a single article.

        Returns:
            Parsed takeaways dict: {"takeaways": ["...", "...", "..."]}
        """
        query = (
            f"Write 2-4 takeaway bullets for this article:\n"
            f"Title: \"{title}\"\n"
            f"Source: {source}\n"
            f"Summary: {summary}"
        )
        return self._run_pass(self.takeaway, query, model, chat_format)

    def process_highlights(self, selected_titles: list[dict], model,
                           chat_format: str = "chatml") -> dict:
        """Pass 3: Generate highlights intro from selected article titles.

        Args:
            selected_titles: List of {"title": str, "category": str}

        Returns:
            Parsed highlights dict: {"highlights_intro": "..."}
        """
        lines = [f"- \"{t['title']}\" ({t['category']})" for t in selected_titles]
        query = (
            f"Write a 1-2 sentence highlights intro for today's digest.\n"
            f"Selected articles:\n" + "\n".join(lines)
        )
        return self._run_pass(self.highlights, query, model, chat_format)

    def _run_pass(self, augmentor: Augmentor, query: str, model,
                  chat_format: str) -> dict:
        """Execute a single augmentor pass with grammar, verify, retry."""
        from llama_cpp import LlamaGrammar

        # Build grammar object if available
        grammar = None
        if augmentor.grammar_str:
            try:
                grammar = LlamaGrammar.from_string(augmentor.grammar_str)
            except Exception as e:
                logger.warning(f"Grammar load failed for {augmentor.name}: {e}")

        prompt = augmentor.build_prompt(query, chat_format)
        gen_kwargs = {"max_tokens": 512, "temperature": 0.3}
        if grammar:
            gen_kwargs["grammar"] = grammar

        for attempt in range(1, augmentor.max_retries + 2):  # +1 for initial attempt
            response = model.generate(prompt, **gen_kwargs)

            if not response or not response.strip():
                logger.warning(f"{augmentor.name} attempt {attempt}: empty response")
                continue

            # Verify
            passed, error_hint = augmentor.verify(response, query)
            if passed:
                try:
                    return json.loads(response.strip())
                except json.JSONDecodeError:
                    error_hint = "Response is not valid JSON despite passing verification."

            logger.info(f"{augmentor.name} attempt {attempt} failed: {error_hint}")

            # Retry with feedback
            if attempt <= augmentor.max_retries:
                prompt = augmentor.build_retry_prompt(query, response, error_hint, chat_format)
                # Recreate grammar for retry
                if augmentor.grammar_str:
                    try:
                        grammar = LlamaGrammar.from_string(augmentor.grammar_str)
                        gen_kwargs["grammar"] = grammar
                    except Exception:
                        pass

        # All attempts failed — return best-effort parse or empty
        logger.error(f"{augmentor.name}: all {augmentor.max_retries + 1} attempts failed")
        try:
            return json.loads(response.strip())
        except Exception:
            return self._fallback(augmentor.name)

    def _fallback(self, augmentor_name: str) -> dict:
        """Return safe fallback when all attempts fail."""
        if "selection" in augmentor_name:
            return {"selected": []}
        elif "takeaway" in augmentor_name:
            return {"takeaways": ["No takeaways generated."]}
        elif "highlights" in augmentor_name:
            return {"highlights_intro": "Today's digest covers recent developments across the field."}
        return {}


def format_articles_compact(articles: list[dict],
                            max_summary_chars: int = 100) -> str:
    """Format pending articles into compact numbered list for Pass 1.

    Args:
        articles: List of article dicts from pending_digest.json
        max_summary_chars: Max chars per summary excerpt

    Returns:
        Numbered list string, one article per line.
    """
    lines = []
    for i, art in enumerate(articles):
        cat = art.get("category", "general")
        title = art.get("title", "Untitled")
        source = art.get("source_name", "Unknown")
        summary = art.get("summary", "").strip()

        # Truncate summary
        if len(summary) > max_summary_chars:
            summary = summary[:max_summary_chars].rsplit(" ", 1)[0] + "..."
        elif not summary:
            summary = "(no summary)"

        # Add score for Reddit posts
        score_str = ""
        if "score" in art and art["score"]:
            score_str = f" [{art['score']} pts]"

        lines.append(f"{i}. [{cat}] \"{title}\" ({source}){score_str} — {summary}")

    return "\n".join(lines)
