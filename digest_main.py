#!/usr/bin/env python3
"""
Digest CLI — Fetch and curate news digests using on-device LLM.

Usage:
    python digest_main.py ai                 # Fetch + curate AI digest
    python digest_main.py --curate ai        # Curate from existing pending
    python digest_main.py --fetch ai         # Fetch only (save pending)
    python digest_main.py --all              # All topics, fetch + curate
    python digest_main.py --all --curate     # All topics, curate only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="On-device digest curation")
    parser.add_argument("topic", nargs="?", help="Topic ID (e.g., ai, robotics)")
    parser.add_argument("--all", action="store_true", help="Process all topics")
    parser.add_argument("--fetch", action="store_true", help="Fetch only (no curation)")
    parser.add_argument("--curate", action="store_true", help="Curate only (from existing pending)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.topic and not args.all:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s"
    )

    # Load config
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    digest_config = config.get("digest", {})
    topics_dir = digest_config.get("topics_dir", "digest/topics")
    grammars_dir = digest_config.get("grammars_dir", "data/digest_grammars")
    examples_dir = digest_config.get("examples_dir", "data/augmentor_examples/curation")

    # Fetch-only mode doesn't need the model
    if args.fetch:
        from digest.config_loader import load_topic_config, discover_topics
        from digest.sources import fetch_all
        from digest.dedup import load_seen, save_seen, prune_old, filter_new
        from datetime import datetime, timezone
        import json

        topics = discover_topics(topics_dir) if args.all else [args.topic]
        for topic_id in topics:
            tc = load_topic_config(topics_dir, topic_id)
            topic_dir = Path(topics_dir) / topic_id

            print(f"\nFetching: {tc.get('digest_name', topic_id)}")
            all_articles = fetch_all(tc)

            seen_path = str(topic_dir / "seen_articles.json")
            seen_db = load_seen(seen_path)
            seen_db = prune_old(seen_db)
            new_articles, seen_db = filter_new(all_articles, seen_db)

            save_seen(str(topic_dir / "seen_articles_staged.json"), seen_db)

            max_pending = tc.get("max_pending_articles", 20)
            pending = new_articles[:max_pending]
            for art in pending:
                if art.get("summary") and len(art["summary"]) > 300:
                    art["summary"] = art["summary"][:300].rsplit(" ", 1)[0] + "..."

            pending_path = str(topic_dir / "pending_digest.json")
            with open(pending_path, "w", encoding="utf-8") as fp:
                json.dump({
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "article_count": len(pending),
                    "total_new": len(new_articles),
                    "articles": pending,
                }, fp, indent=2, ensure_ascii=False)
            print(f"Saved {len(pending)} articles to {pending_path}")
        return

    # Curation mode — needs model
    model_path = args.model or config.get("base_model", {}).get("path", "")
    if not model_path or not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Use --model <path> or set base_model.path in config.yaml")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    start = time.time()

    from engine.base_model import BaseModel
    from engine.config import Config

    cfg = Config(args.config)
    # Apply --model override
    if args.model:
        cfg.base_model.path = args.model
    model = BaseModel(cfg.base_model)
    model.load()
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Initialize digest augmentors
    from engine.digest_augmentors import DigestAugmentorRouter

    router = DigestAugmentorRouter(
        examples_dir=examples_dir,
        grammars_dir=grammars_dir,
    )

    # Initialize embeddings for example retrieval
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        router.init_embeddings(embedder)
        print("Embeddings initialized")
    except ImportError:
        print("sentence-transformers not installed — using first-match examples")

    # Build pipeline
    chat_format = config.get("fusion", {}).get("chat_format", "chatml")
    from digest.pipeline import DigestPipeline

    pipeline = DigestPipeline(
        model=model,
        digest_router=router,
        topics_dir=topics_dir,
        chat_format=chat_format,
    )

    # Run
    if args.all:
        pipeline.run_all(curate_only=args.curate)
    elif args.curate:
        pipeline.curate(args.topic)
    else:
        pipeline.fetch_and_curate(args.topic)


if __name__ == "__main__":
    main()
