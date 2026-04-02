"""Deduplication: track seen articles and filter to new ones only."""

import json
import os
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, urlunparse


def normalize_url(url):
    """Normalize a URL for dedup comparison."""
    parsed = urlparse(url.strip().rstrip("/"))
    # Remove fragments and query params for matching
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/"),
        "", "", ""
    ))
    return normalized


def load_seen(db_path):
    """Load the seen articles database."""
    if not os.path.exists(db_path):
        return {}
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_seen(db_path, seen_db):
    """Save the seen articles database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(seen_db, f, indent=2, ensure_ascii=False)


def prune_old(seen_db, days=90):
    """Remove entries older than N days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    pruned = {
        url: data for url, data in seen_db.items()
        if data.get("first_seen", "") > cutoff
    }
    removed = len(seen_db) - len(pruned)
    if removed > 0:
        print(f"  Pruned {removed} old entries from seen database")
    return pruned


def filter_new(articles, seen_db):
    """Filter articles to only those not yet seen. Returns (new_articles, updated_seen_db)."""
    new_articles = []
    for article in articles:
        norm_url = normalize_url(article["url"])
        if norm_url not in seen_db:
            new_articles.append(article)
            seen_db[norm_url] = {
                "title": article["title"],
                "source": article["source_name"],
                "first_seen": datetime.now(timezone.utc).isoformat(),
                "category": article["category"]
            }

    print(f"  {len(new_articles)} new articles (out of {len(articles)} total)")
    return new_articles, seen_db
