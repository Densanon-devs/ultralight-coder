"""Digest topic configuration loader."""

import json
from pathlib import Path


def load_topic_config(topics_dir: str, topic_id: str) -> dict:
    """Load a single topic's config.json."""
    path = Path(topics_dir) / topic_id / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"No config.json for topic '{topic_id}' at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_topics(topics_dir: str) -> list[str]:
    """Find all topic IDs (subdirectories with config.json)."""
    base = Path(topics_dir)
    if not base.exists():
        return []
    topics = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            topics.append(d.name)
    return topics
