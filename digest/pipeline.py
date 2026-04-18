"""
Digest Curation Pipeline — Multi-pass on-device LLM curation.

Orchestrates the three-pass curation flow:
  1. Fetch articles from sources (RSS, Reddit, GitHub)
  2. Deduplicate against seen articles
  3. Pass 1: Select best articles + assign categories
  4. Pass 2: Generate takeaways per selected article
  5. Pass 3: Generate highlights intro
  6. Assemble into digest_content.json
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add densanon-core to sys.path for shared modules
_CORE_ROOT = Path(__file__).resolve().parent.parent.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from digest.config_loader import load_topic_config, discover_topics
from digest.assembler import assemble_digest, save_digest
from engine.digest_augmentors import (
    DigestAugmentorRouter, format_articles_compact,
)

logger = logging.getLogger(__name__)


class DigestPipeline:
    """Orchestrates the full digest curation pipeline."""

    def __init__(self, model, digest_router: DigestAugmentorRouter,
                 topics_dir: str = "digest/topics",
                 chat_format: str = "chatml"):
        self.model = model
        self.router = digest_router
        self.topics_dir = topics_dir
        self.chat_format = chat_format

    def fetch(self, topic_id: str) -> str:
        """Fetch articles for a topic, deduplicate, save pending_digest.json.

        Returns path to pending_digest.json.
        """
        from densanon.core.feed_fetcher import fetch_all
        from densanon.core.dedup import load_seen, save_seen, prune_old, filter_new

        config = load_topic_config(self.topics_dir, topic_id)
        topic_dir = Path(self.topics_dir) / topic_id

        print(f"\n{'='*60}")
        print(f"Fetching: {config.get('digest_name', topic_id)}")
        print(f"{'='*60}")

        # Fetch from all sources
        all_articles = fetch_all(config)

        # Dedup
        seen_path = str(topic_dir / "seen_articles.json")
        seen_db = load_seen(seen_path)
        seen_db = prune_old(seen_db)
        new_articles, seen_db = filter_new(all_articles, seen_db)

        # Save staged seen DB (promote after successful curation)
        staged_path = str(topic_dir / "seen_articles_staged.json")
        save_seen(staged_path, seen_db)

        # Save pending digest
        max_pending = config.get("max_pending_articles", 20)
        pending = new_articles[:max_pending]

        # Trim summaries for file size
        for art in pending:
            if art.get("summary") and len(art["summary"]) > 300:
                art["summary"] = art["summary"][:300].rsplit(" ", 1)[0] + "..."

        pending_path = str(topic_dir / "pending_digest.json")
        with open(pending_path, "w", encoding="utf-8") as f:
            json.dump({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "article_count": len(pending),
                "total_new": len(new_articles),
                "articles": pending,
            }, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(pending)} articles to {pending_path}")
        return pending_path

    def curate(self, topic_id: str,
               pending_path: Optional[str] = None) -> dict:
        """Run the 3-pass curation pipeline on pending articles.

        Args:
            topic_id: Digest topic identifier
            pending_path: Path to pending_digest.json (auto-discovers if None)

        Returns:
            Final digest_content dict.
        """
        config = load_topic_config(self.topics_dir, topic_id)
        topic_dir = Path(self.topics_dir) / topic_id

        if pending_path is None:
            pending_path = str(topic_dir / "pending_digest.json")

        # Load pending articles
        with open(pending_path, encoding="utf-8") as f:
            pending = json.load(f)
        articles = pending.get("articles", [])

        if not articles:
            print(f"No articles to curate for {topic_id}")
            return {"highlights_intro": "", "sections": []}

        categories = list(config.get("categories", {}).keys())

        print(f"\n{'='*60}")
        print(f"Curating: {config.get('digest_name', topic_id)} ({len(articles)} articles)")
        print(f"{'='*60}")

        start = time.time()

        # ── Pass 1: Selection ────────────────────────────────
        print("\n[Pass 1] Selecting articles...")
        compact = format_articles_compact(articles)
        selection = self.router.process_selection(
            compact, self.model, self.chat_format, categories
        )
        selected = selection.get("selected", [])
        print(f"  Selected {len(selected)} articles")

        # ── Pass 2: Takeaways ────────────────────────────────
        print("\n[Pass 2] Generating takeaways...")
        takeaways_map: dict[int, dict] = {}
        for i, sel in enumerate(selected):
            idx = sel["index"]
            if idx < 0 or idx >= len(articles):
                continue
            art = articles[idx]
            print(f"  [{i+1}/{len(selected)}] {art.get('title', 'Untitled')[:60]}...")
            result = self.router.process_takeaways(
                title=art.get("title", ""),
                source=art.get("source_name", ""),
                summary=art.get("summary", ""),
                model=self.model,
                chat_format=self.chat_format,
            )
            takeaways_map[idx] = result

        # ── Pass 3: Highlights ───────────────────────────────
        print("\n[Pass 3] Generating highlights intro...")
        selected_titles = []
        for sel in selected:
            idx = sel["index"]
            if idx < 0 or idx >= len(articles):
                continue
            selected_titles.append({
                "title": articles[idx].get("title", ""),
                "category": sel["category"],
            })
        highlights = self.router.process_highlights(
            selected_titles, self.model, self.chat_format
        )

        elapsed = time.time() - start
        print(f"\nCuration complete in {elapsed:.1f}s")

        # ── Assemble ─────────────────────────────────────────
        digest = assemble_digest(selection, articles, takeaways_map,
                                 highlights, config)

        # Save
        output_path = str(topic_dir / "digest_content.json")
        save_digest(digest, output_path)
        print(f"Saved digest to {output_path}")

        # Promote staged seen DB
        staged = topic_dir / "seen_articles_staged.json"
        if staged.exists():
            import shutil
            shutil.move(str(staged), str(topic_dir / "seen_articles.json"))
            print("Promoted seen_articles_staged.json")

        return digest

    def fetch_and_curate(self, topic_id: str) -> dict:
        """Full pipeline: fetch articles, then curate."""
        pending_path = self.fetch(topic_id)
        return self.curate(topic_id, pending_path)

    def run_all(self, curate_only: bool = False) -> dict[str, dict]:
        """Run pipeline for all discovered topics.

        Returns:
            {topic_id: digest_content_dict}
        """
        topics = discover_topics(self.topics_dir)
        results = {}
        for topic in topics:
            try:
                if curate_only:
                    results[topic] = self.curate(topic)
                else:
                    results[topic] = self.fetch_and_curate(topic)
            except Exception as e:
                logger.error(f"Failed to process {topic}: {e}")
                print(f"\n[ERROR] {topic}: {e}")
        return results
