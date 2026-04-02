"""Fetcher modules for each source type."""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import json
import re


def fetch_rss(source, headers, lookback_hours=48, max_items=10):
    """Fetch articles from an RSS/Atom feed."""
    articles = []
    try:
        feed = feedparser.parse(source["url"], agent=headers["User-Agent"])
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        for entry in feed.entries[:max_items * 2]:  # fetch extra, filter by date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

            # Extract a cleaner title and summary
            title = entry.get("title", "Untitled").strip()
            title = re.sub(r"\s+", " ", title)  # collapse whitespace

            summary = entry.get("summary", entry.get("description", ""))
            # Strip HTML tags from summary
            if summary:
                summary = BeautifulSoup(summary, "html.parser").get_text(separator=" ")
                summary = re.sub(r"\s+", " ", summary).strip()
                summary = summary[:1000]  # cap length

            link = entry.get("link", "")

            articles.append({
                "title": title,
                "url": link,
                "summary": summary,
                "source_name": source["name"],
                "category": source["category"],
                "published": published.isoformat() if published else None,
                "fetched_at": datetime.now(timezone.utc).isoformat()
            })

            if len(articles) >= max_items:
                break

    except Exception as e:
        print(f"  [WARN] Failed to fetch RSS {source['name']}: {e}")

    return articles


def fetch_reddit(source, headers, max_posts=5):
    """Fetch top posts from a subreddit using JSON API (no auth)."""
    articles = []
    try:
        url = f"https://www.reddit.com/r/{source['subreddit']}/hot.json?limit={max_posts * 2}"
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for post in data.get("data", {}).get("children", []):
            p = post["data"]
            # Skip stickied/pinned posts
            if p.get("stickied"):
                continue
            # Only include posts with decent engagement
            if p.get("score", 0) < 10:
                continue

            created = datetime.fromtimestamp(p["created_utc"], tz=timezone.utc)
            selftext = p.get("selftext", "")[:500]

            articles.append({
                "title": p.get("title", "Untitled"),
                "url": f"https://reddit.com{p.get('permalink', '')}",
                "summary": selftext if selftext else f"Score: {p.get('score', 0)} | Comments: {p.get('num_comments', 0)}",
                "source_name": source["name"],
                "category": source["category"],
                "published": created.isoformat(),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "score": p.get("score", 0),
                "num_comments": p.get("num_comments", 0)
            })

            if len(articles) >= max_posts:
                break

    except Exception as e:
        print(f"  [WARN] Failed to fetch Reddit {source['subreddit']}: {e}")

    return articles


def fetch_github_releases(source, headers):
    """Fetch recent releases/events from a GitHub org."""
    articles = []
    try:
        # Get public repos for the org
        repos_url = f"https://api.github.com/orgs/{source['org']}/repos?sort=updated&per_page=10"
        resp = requests.get(repos_url, headers=headers, timeout=15)
        resp.raise_for_status()
        repos = resp.json()

        cutoff = datetime.now(timezone.utc) - timedelta(hours=72)

        for repo in repos:
            repo_name = repo["full_name"]
            # Check for recent releases
            rel_url = f"https://api.github.com/repos/{repo_name}/releases?per_page=3"
            try:
                rel_resp = requests.get(rel_url, headers=headers, timeout=10)
                if rel_resp.status_code == 200:
                    for release in rel_resp.json():
                        published = datetime.fromisoformat(release["published_at"].replace("Z", "+00:00"))
                        if published > cutoff:
                            body = release.get("body", "")[:500]
                            articles.append({
                                "title": f"{repo_name}: {release.get('name', release.get('tag_name', 'New Release'))}",
                                "url": release.get("html_url", ""),
                                "summary": body,
                                "source_name": source["name"],
                                "category": source["category"],
                                "published": published.isoformat(),
                                "fetched_at": datetime.now(timezone.utc).isoformat()
                            })
            except Exception:
                continue

    except Exception as e:
        print(f"  [WARN] Failed to fetch GitHub {source['org']}: {e}")

    return articles


def fetch_all(config):
    """Fetch from all configured sources. Returns list of article dicts."""
    all_articles = []
    lookback = config.get("digest_lookback_hours", 48)
    headers = {
        "User-Agent": config.get("user_agent", "DensanonDigest/1.0 (https://densanon.com; admin@densanon.com)")
    }

    print("Fetching RSS feeds...")
    for source in config["sources"].get("rss", []):
        max_items = config.get("max_articles_per_source", 10)
        if "arxiv" in source["url"].lower():
            max_items = config.get("arxiv_max_per_feed", 5)
        print(f"  -> {source['name']}")
        articles = fetch_rss(source, headers, lookback_hours=lookback, max_items=max_items)
        all_articles.extend(articles)
        print(f"     Found {len(articles)} articles")

    print("Fetching Reddit...")
    for source in config["sources"].get("reddit", []):
        print(f"  -> {source['name']}")
        articles = fetch_reddit(source, headers, max_posts=config.get("max_reddit_posts", 5))
        all_articles.extend(articles)
        print(f"     Found {len(articles)} posts")

    print("Fetching GitHub releases...")
    for source in config["sources"].get("github", []):
        print(f"  -> {source['name']}")
        articles = fetch_github_releases(source, headers)
        all_articles.extend(articles)
        print(f"     Found {len(articles)} releases")

    print(f"\nTotal raw articles: {len(all_articles)}")
    return all_articles
