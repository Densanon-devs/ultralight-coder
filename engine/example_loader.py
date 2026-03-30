"""
Augmentor Example Loader — Loads SolvedExamples from YAML files.

Reads domain-organized YAML files from data/augmentor_examples/ and converts
them to SolvedExample objects for the augmentor system.

Directory structure:
    data/augmentor_examples/
        pattern/decorator.yaml
        algorithm/tree.yaml
        text/glob_match.yaml
        ...

YAML format:
    domain: pattern
    category: pattern_decorator
    examples:
      - query: |
            Write a decorator...
        solution: |
            ```python
            ...
            ```
        tags: [decorator, functools]
        difficulty: medium
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, list]] = {}  # path -> (mtime, examples)


def _load_yaml():
    """Import yaml, trying PyYAML first."""
    try:
        import yaml
        return yaml
    except ImportError:
        logger.warning("PyYAML not installed. Install with: pip install pyyaml")
        return None


def load_examples_from_file(filepath: Path) -> list[dict]:
    """Load examples from a single YAML file. Returns list of dicts."""
    yaml = _load_yaml()
    if yaml is None:
        return []

    filepath = Path(filepath)
    if not filepath.exists():
        return []

    # Check cache
    mtime = filepath.stat().st_mtime
    cache_key = str(filepath)
    if cache_key in _cache and _cache[cache_key][0] == mtime:
        return _cache[cache_key][1]

    try:
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "examples" not in data:
            return []

        category = data.get("category", "")
        examples = []
        for ex in data["examples"]:
            examples.append({
                "query": ex.get("query", "").strip(),
                "solution": ex.get("solution", "").strip(),
                "category": ex.get("category", category),
                "tags": ex.get("tags", []),
                "difficulty": ex.get("difficulty", "medium"),
            })

        _cache[cache_key] = (mtime, examples)
        return examples

    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return []


def load_all_examples(base_dir: str = "data/augmentor_examples") -> list[dict]:
    """Load all examples from all YAML files under base_dir."""
    base = Path(base_dir)
    if not base.exists():
        logger.info(f"No augmentor examples directory at {base}")
        return []

    all_examples = []
    for yaml_file in sorted(base.rglob("*.yaml")):
        examples = load_examples_from_file(yaml_file)
        all_examples.extend(examples)
        if examples:
            logger.debug(f"Loaded {len(examples)} examples from {yaml_file.name}")

    logger.info(f"Loaded {len(all_examples)} augmentor examples from {base}")
    return all_examples


def load_domain_examples(base_dir: str, domain: str) -> list[dict]:
    """Load examples from a specific domain subdirectory."""
    domain_dir = Path(base_dir) / domain
    if not domain_dir.exists():
        return []

    all_examples = []
    for yaml_file in sorted(domain_dir.glob("*.yaml")):
        all_examples.extend(load_examples_from_file(yaml_file))
    return all_examples


def to_solved_examples(raw_examples: list[dict]) -> list:
    """Convert raw dicts to SolvedExample objects."""
    from engine.augmentors import SolvedExample
    return [
        SolvedExample(
            query=ex["query"],
            solution=ex["solution"],
            category=ex.get("category", ""),
        )
        for ex in raw_examples
        if ex.get("query") and ex.get("solution")
    ]


def load_and_convert(base_dir: str = "data/augmentor_examples") -> list:
    """Load all YAML examples and return as SolvedExample objects."""
    raw = load_all_examples(base_dir)
    return to_solved_examples(raw)
