#!/usr/bin/env python3
"""Quick verification test for Android/Termux setup.

Run this before the full pipeline to catch issues early.
Tests each component independently so you know exactly what broke.
"""

import sys
import os

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
errors = 0


def check(name, fn):
    global errors
    try:
        result = fn()
        if result:
            print(f"  [{PASS}] {name}")
        else:
            print(f"  [{FAIL}] {name}")
            errors += 1
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e}")
        errors += 1


def check_warn(name, fn):
    try:
        result = fn()
        if result:
            print(f"  [{PASS}] {name}")
        else:
            print(f"  [{WARN}] {name} (optional)")
    except Exception as e:
        print(f"  [{WARN}] {name}: {e} (optional)")


print("=" * 50)
print("  Digest Engine — Android Test")
print("=" * 50)

# ── 1. Core imports ──────────────────────────────────────
print("\n1. Core imports")
check("yaml", lambda: __import__("yaml") and True)
check("numpy", lambda: __import__("numpy") and True)
check("feedparser", lambda: __import__("feedparser") and True)
check("requests", lambda: __import__("requests") and True)
check("bs4", lambda: __import__("bs4") and True)

# ── 2. LLM inference ────────────────────────────────────
print("\n2. LLM inference")
check("llama_cpp", lambda: __import__("llama_cpp") and True)

# ── 3. Augmentor system ─────────────────────────────────
print("\n3. Augmentor system")


def test_augmentors():
    from engine.digest_augmentors import DigestAugmentorRouter
    r = DigestAugmentorRouter()
    total = len(r.selection.examples) + len(r.takeaway.examples) + len(r.highlights.examples)
    print(f"       {total} examples loaded")
    return total >= 70


check("augmentor loading", test_augmentors)


def test_grammars():
    from engine.digest_augmentors import DigestAugmentorRouter
    r = DigestAugmentorRouter()
    return all([r.selection_grammar, r.takeaways_grammar, r.highlights_grammar])


check("GBNF grammars", test_grammars)

# ── 4. Verifiers ────────────────────────────────────────
print("\n4. Verifiers")


def test_verify_selection():
    from engine.digest_augmentors import verify_selection
    good = '{"selected": [{"index": 0, "category": "research"}, {"index": 3, "category": "tools"}, {"index": 5, "category": "community"}, {"index": 7, "category": "industry"}, {"index": 9, "category": "research"}, {"index": 11, "category": "tools"}]}'
    ok, _ = verify_selection(good, "")
    return ok


def test_verify_takeaways():
    from engine.digest_augmentors import verify_takeaways
    good = '{"takeaways": ["First point about the article.", "Second point about implications.", "Third point about context."]}'
    ok, _ = verify_takeaways(good, "")
    return ok


def test_verify_highlights():
    from engine.digest_augmentors import verify_highlights
    good = '{"highlights_intro": "Today\'s digest covers major advances in chip manufacturing and open-source model compression, with Intel and Nvidia making billion-dollar moves."}'
    ok, _ = verify_highlights(good, "")
    return ok


check("verify_selection", test_verify_selection)
check("verify_takeaways", test_verify_takeaways)
check("verify_highlights", test_verify_highlights)

# ── 5. Model ────────────────────────────────────────────
print("\n5. Model file")
from pathlib import Path

models = list(Path("models").glob("*.gguf")) if Path("models").exists() else []
if models:
    for m in models:
        size_mb = m.stat().st_size / (1024 * 1024)
        print(f"  [{PASS}] {m.name} ({size_mb:.0f} MB)")
else:
    print(f"  [{FAIL}] No .gguf files in models/")
    errors += 1

# ── 6. Quick model load test ────────────────────────────
print("\n6. Model inference test")


def test_model_load():
    if not models:
        return False
    smallest = min(models, key=lambda p: p.stat().st_size)
    print(f"       Loading {smallest.name}...")
    from llama_cpp import Llama
    model = Llama(
        model_path=str(smallest),
        n_ctx=512,
        n_gpu_layers=0,  # CPU only on Android
        n_threads=4,
        verbose=False,
    )
    out = model("Hello", max_tokens=5)
    text = out["choices"][0]["text"]
    print(f"       Generated: {repr(text[:50])}")
    del model
    return len(text) > 0


check("model loads and generates", test_model_load)

# ── 7. Optional: embeddings ─────────────────────────────
print("\n7. Sentence embeddings (optional)")
check_warn("sentence_transformers", lambda: __import__("sentence_transformers") and True)

# ── 8. Topic configs ────────────────────────────────────
print("\n8. Topic configs")
from digest.config_loader import discover_topics

topics = discover_topics("digest/topics")
if topics:
    print(f"  [{PASS}] Topics: {', '.join(topics)}")
else:
    print(f"  [{FAIL}] No topics found in digest/topics/")
    errors += 1

# ── Summary ─────────────────────────────────────────────
print("\n" + "=" * 50)
if errors == 0:
    print(f"  All checks passed! Ready to run:")
    print(f"    python digest_main.py --fetch ai")
    print(f"    python digest_main.py --curate ai")
else:
    print(f"  {errors} check(s) failed. Fix issues above first.")
print("=" * 50)
sys.exit(errors)
