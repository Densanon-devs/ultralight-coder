"""Assemble multi-pass curation outputs into final digest_content.json."""

import json
from pathlib import Path


def assemble_digest(selection: dict, articles: list[dict],
                    takeaways_map: dict[int, dict],
                    highlights: dict, config: dict) -> dict:
    """Combine selection + takeaways + highlights into digest_content.json format.

    Args:
        selection: {"selected": [{"index": N, "category": "..."}]}
        articles: Original article list from pending_digest.json
        takeaways_map: {article_index: {"takeaways": [...]}}
        highlights: {"highlights_intro": "..."}
        config: Topic config.json

    Returns:
        dict matching digest_content.json schema.
    """
    categories = config.get("categories", {})

    # Group selected articles by category
    sections_map: dict[str, list] = {}
    for sel in selection.get("selected", []):
        idx = sel["index"]
        cat = sel["category"]

        if idx < 0 or idx >= len(articles):
            continue

        art = articles[idx]
        takeaway_data = takeaways_map.get(idx, {"takeaways": []})

        article_entry = {
            "title": art.get("title", "Untitled"),
            "source": art.get("source_name", "Unknown"),
            "url": art.get("url", ""),
            "takeaways": takeaway_data.get("takeaways", []),
        }

        if cat not in sections_map:
            sections_map[cat] = []
        sections_map[cat].append(article_entry)

    # Build sections in config-defined order
    sections = []
    for cat_key, cat_label in categories.items():
        if cat_key in sections_map:
            sections.append({
                "name": cat_label,
                "category": cat_key,
                "articles": sections_map[cat_key],
            })

    # Add any categories not in config
    for cat_key, articles_list in sections_map.items():
        if cat_key not in categories:
            sections.append({
                "name": cat_key.replace("_", " ").title(),
                "category": cat_key,
                "articles": articles_list,
            })

    return {
        "highlights_intro": highlights.get("highlights_intro", ""),
        "sections": sections,
    }


def save_digest(digest_data: dict, output_path: str):
    """Write digest_content.json."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(digest_data, f, indent=2, ensure_ascii=False)
