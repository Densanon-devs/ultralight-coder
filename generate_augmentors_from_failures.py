#!/usr/bin/env python3
"""
Automated Augmentor Generation from Benchmark Failures

Reads a benchmark results JSON, finds the failing queries, and uses a larger
local model (Llama 3.2 3B default) to draft candidate YAML augmentor examples.
Each candidate passes through gates before acceptance:

  1. Schema gate — must parse as valid YAML in the augmentor_examples format
  2. Isolation gate (optional) — target model with the new example injected
     must now pass the originally-failing query
  3. Regression gate (manual) — run the full benchmark to confirm no drop

Usage:
    python generate_augmentors_from_failures.py \\
        --results benchmark_realworld_results.json \\
        --generator models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \\
        --target models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf

    # Schema-only (fast, unsafe):
    python generate_augmentors_from_failures.py --results X.json --no-isolate

    # Dry-run (no generation, just show which failures would be processed):
    python generate_augmentors_from_failures.py --results X.json --dry-run

Output lands in data/augmentor_examples/generated/ — quarantined from the
hand-curated library until you review each one and move it to its domain
directory.

Design notes:
- The generator is a Tier-up: a larger model writing examples for a smaller
  one. Llama 3.2 3B is the default because it's already in the stack and
  produces YAML reliably. Post-BIOS Phase 13, you can swap in Qwen 14B.
- Schema gate is REQUIRED. Isolation is default-on but can be disabled for
  faster human-review loops. Regression is left manual because it re-runs
  the full benchmark (minutes) and belongs in a separate command.
- Candidates carry metadata (source failure query, generator model, gate
  results) as a top-level `_meta:` block in the YAML so you can trace how
  each example was born.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ── Data shapes ─────────────────────────────────────────────────────────────


@dataclass
class Failure:
    """A single failing benchmark result."""
    query: str
    domain: str
    missing: list[str]
    unwanted: list[str]
    score: float
    lines: int


@dataclass
class Candidate:
    """A generated augmentor example with gate results."""
    failure: Failure
    yaml_text: str
    parsed: Optional[dict]
    schema_ok: bool = False
    isolation_ok: Optional[bool] = None  # None = not tested
    schema_error: str = ""
    isolation_error: str = ""
    output_path: Optional[Path] = None


# ── Loading failures ────────────────────────────────────────────────────────


def load_failures(results_path: Path) -> list[Failure]:
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    if "results" not in data:
        raise ValueError(f"{results_path} has no 'results' field (not a benchmark JSON?)")

    failures = []
    for r in data["results"]:
        if r.get("passed"):
            continue
        failures.append(Failure(
            query=r.get("query", ""),
            domain=r.get("domain", "general"),
            missing=r.get("missing", []),
            unwanted=r.get("unwanted", []),
            score=r.get("score", 0.0),
            lines=r.get("lines", 0),
        ))
    return failures


# ── Generator prompt ────────────────────────────────────────────────────────


GENERATOR_SYSTEM = """You are an expert Python developer writing few-shot examples for a small code-generation model.
Your examples teach the small model to produce correct, complete code on prompts it previously failed.
Reply with ONLY a YAML block in the exact schema given. No commentary, no markdown fences around the YAML."""


def build_generator_prompt(failure: Failure) -> str:
    """Build the structured prompt that asks the generator to produce one YAML example."""
    must = ", ".join(failure.missing) if failure.missing else "(none)"
    must_not = ", ".join(failure.unwanted) if failure.unwanted else "(none)"

    return f"""A small Python code-generation assistant just failed the following prompt. Generate ONE high-quality few-shot example that would help it succeed next time.

FAILING PROMPT:
{failure.query}

DOMAIN: {failure.domain}

REQUIREMENTS:
- The solution MUST contain: {must}
- The solution MUST NOT contain: {must_not}
- Minimum {max(failure.lines + 3, 6)} lines of real code
- Python 3.10+ idioms
- Include a brief "Key:" note after the code block explaining the key pattern

Produce your answer in this EXACT YAML schema (indentation matters):

domain: {failure.domain}
category: {failure.domain}_generated
examples:
  - query: |
      <rephrase the failing prompt as a clear, specific instruction>
    solution: |
      ```python
      <complete working Python code that meets all requirements>
      ```

      Key: <one sentence explaining the core pattern>
    tags: [<tag1>, <tag2>, <tag3>]
    difficulty: medium

Reply with the YAML block only. Do not wrap it in markdown fences."""


# ── Generator model wrapper ─────────────────────────────────────────────────


class Generator:
    """Loads a larger local model and uses it to draft augmentor examples."""

    def __init__(self, model_path: Path, gpu_layers: int = 99, context: int = 4096):
        from llama_cpp import Llama
        logger.info(f"Loading generator model: {model_path.name}")
        self.llama = Llama(
            model_path=str(model_path),
            n_ctx=context,
            n_gpu_layers=gpu_layers,
            n_batch=512,
            verbose=False,
        )
        self.model_name = model_path.name

    def generate(self, failure: Failure, max_tokens: int = 1024) -> str:
        user = build_generator_prompt(failure)
        prompt = (
            f"<|im_start|>system\n{GENERATOR_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        out = self.llama(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            stop=["<|im_end|>", "</s>"],
            echo=False,
        )
        return out["choices"][0]["text"].strip()


# ── Schema gate ─────────────────────────────────────────────────────────────


def strip_yaml_fences(text: str) -> str:
    """Tolerate models that wrap YAML in ```yaml fences despite instructions.

    Uses rsplit on the closing fence because augmentor solution blocks
    contain their own ```python ... ``` fences — naively splitting on the
    first ``` would truncate inside the solution body.
    """
    text = text.strip()
    if text.startswith("```yaml"):
        text = text[len("```yaml"):]
    elif text.startswith("```"):
        text = text[3:]
        if text.startswith("yaml"):
            text = text[4:]
    else:
        return text
    if "```" in text:
        text = text.rsplit("```", 1)[0]
    return text.strip()


def schema_gate(candidate: Candidate) -> None:
    """Populate candidate.schema_ok and candidate.parsed in place."""
    import yaml

    text = strip_yaml_fences(candidate.yaml_text)

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as e:
        candidate.schema_error = f"YAML parse failed: {e}"
        return

    if not isinstance(parsed, dict):
        candidate.schema_error = f"Expected mapping at root, got {type(parsed).__name__}"
        return

    if "examples" not in parsed or not isinstance(parsed["examples"], list) or not parsed["examples"]:
        candidate.schema_error = "Missing or empty 'examples' list"
        return

    for i, ex in enumerate(parsed["examples"]):
        if not isinstance(ex, dict):
            candidate.schema_error = f"examples[{i}] is not a mapping"
            return
        if not ex.get("query") or not ex.get("solution"):
            candidate.schema_error = f"examples[{i}] missing query or solution"
            return

    if "domain" not in parsed:
        parsed["domain"] = candidate.failure.domain
    if "category" not in parsed:
        parsed["category"] = f"{candidate.failure.domain}_generated"

    candidate.parsed = parsed
    candidate.schema_ok = True


# ── Isolation gate ──────────────────────────────────────────────────────────


class IsolationTester:
    """
    Loads the target model once and tests whether a failing query passes
    when the candidate YAML is injected into the augmentor library.
    """

    def __init__(self, target_path: Path, gpu_layers: int = 99, context: int = 4096):
        from llama_cpp import Llama
        from benchmark_exec import detect_chat_format

        logger.info(f"Loading target model for isolation tests: {target_path.name}")
        self.model_size_mb = target_path.stat().st_size / (1024 * 1024)
        self.chat_format = detect_chat_format(str(target_path))
        self.llama = Llama(
            model_path=str(target_path),
            n_ctx=context,
            n_gpu_layers=gpu_layers,
            n_batch=512,
            verbose=False,
        )

    def test(self, candidate: Candidate, temp_yaml_dir: Path) -> None:
        """Run the failing query through target+augmentors with the candidate injected."""
        import yaml as yamllib
        from engine.augmentors import AugmentorRouter
        from engine.embedder import get_embedder
        from benchmark_realworld import extract_code, check_query, RealWorldQuery

        domain = candidate.parsed.get("domain", candidate.failure.domain)
        category = candidate.parsed.get("category", f"{domain}_generated")
        temp_file = temp_yaml_dir / domain / f"{category}.yaml"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, "w", encoding="utf-8") as f:
            yamllib.safe_dump(candidate.parsed, f, sort_keys=False, allow_unicode=True)

        try:
            router = AugmentorRouter(yaml_dir=str(temp_yaml_dir))
            embedder = get_embedder()
            router.init_embeddings(embedder)
            router.use_auto_augmentors(self.model_size_mb)

            aug = router.select_augmentor(candidate.failure.query, "code_gen")
            if aug is None:
                candidate.isolation_error = "router returned no augmentor"
                candidate.isolation_ok = False
                return

            prompt = aug.build_prompt(candidate.failure.query, self.chat_format)
            out = self.llama(
                prompt,
                max_tokens=512,
                temperature=0.2,
                stop=["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"],
                echo=False,
            )
            response = out["choices"][0]["text"].strip()
            code = extract_code(response)

            query_obj = RealWorldQuery(
                query=candidate.failure.query,
                domain=candidate.failure.domain,
                must_contain=candidate.failure.missing,
                must_not_contain=candidate.failure.unwanted,
                min_lines=max(candidate.failure.lines, 3),
            )
            checks = check_query(code, query_obj)
            candidate.isolation_ok = bool(checks["passed"])
            if not candidate.isolation_ok:
                candidate.isolation_error = (
                    f"target still failed: missing={checks['must_contain_fail']} "
                    f"unwanted={checks['must_not_contain_fail']} "
                    f"lines={checks['line_count']}"
                )
        finally:
            try:
                temp_file.unlink()
            except OSError:
                pass


# ── Writer ──────────────────────────────────────────────────────────────────


def write_accepted(candidate: Candidate, output_dir: Path, timestamp: str) -> Path:
    """Persist an accepted candidate to the quarantine directory with metadata."""
    import yaml as yamllib

    domain = candidate.parsed.get("domain", candidate.failure.domain)
    category = candidate.parsed.get("category", f"{domain}_generated")
    out_dir = output_dir / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_query = "".join(c if c.isalnum() else "_" for c in candidate.failure.query[:40]).strip("_")
    filename = f"{category}__{safe_query}__{timestamp}.yaml"
    path = out_dir / filename

    doc = dict(candidate.parsed)
    doc["_meta"] = {
        "generated_at": timestamp,
        "source_failure": candidate.failure.query,
        "source_domain": candidate.failure.domain,
        "source_missing": candidate.failure.missing,
        "source_unwanted": candidate.failure.unwanted,
        "schema_ok": candidate.schema_ok,
        "isolation_ok": candidate.isolation_ok,
        "isolation_error": candidate.isolation_error,
        "quarantine": True,
    }

    with open(path, "w", encoding="utf-8") as f:
        yamllib.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

    return path


# ── Orchestrator ────────────────────────────────────────────────────────────


def run(args) -> int:
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}", file=sys.stderr)
        return 1

    failures = load_failures(results_path)
    print(f"Loaded {len(failures)} failures from {results_path}")

    if not failures:
        print("No failures to process — nothing to do.")
        return 0

    if args.max_failures and len(failures) > args.max_failures:
        print(f"Limiting to first {args.max_failures} failures")
        failures = failures[:args.max_failures]

    if args.dry_run:
        print("\nDry run — showing failures that would be processed:")
        for i, f in enumerate(failures, 1):
            print(f"  [{i}] domain={f.domain} query={f.query[:70]!r}")
            if f.missing:
                print(f"      missing: {f.missing}")
            if f.unwanted:
                print(f"      unwanted: {f.unwanted}")
        return 0

    generator_path = Path(args.generator)
    if not generator_path.exists():
        print(f"Generator model not found: {generator_path}", file=sys.stderr)
        return 2
    generator = Generator(generator_path, gpu_layers=args.gpu_layers)

    tester = None
    if not args.no_isolate:
        target_path = Path(args.target)
        if not target_path.exists():
            print(f"Target model not found: {target_path}", file=sys.stderr)
            return 3
        tester = IsolationTester(target_path, gpu_layers=args.gpu_layers)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_isolation_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidates: list[Candidate] = []

    for i, failure in enumerate(failures, 1):
        print(f"\n[{i}/{len(failures)}] {failure.query[:70]!r}")
        print(f"  domain={failure.domain} missing={failure.missing} unwanted={failure.unwanted}")

        print("  generating...", end=" ", flush=True)
        start = time.monotonic()
        yaml_text = generator.generate(failure)
        elapsed = time.monotonic() - start
        print(f"{elapsed:.1f}s")

        cand = Candidate(failure=failure, yaml_text=yaml_text, parsed=None)
        schema_gate(cand)
        if not cand.schema_ok:
            print(f"  SCHEMA FAIL: {cand.schema_error}")
            candidates.append(cand)
            continue
        print("  SCHEMA OK")

        if tester is not None:
            print("  isolation test...", end=" ", flush=True)
            start = time.monotonic()
            try:
                tester.test(cand, temp_dir)
            except Exception as e:
                cand.isolation_ok = False
                cand.isolation_error = f"exception during isolation: {e}"
            elapsed = time.monotonic() - start
            status = "OK" if cand.isolation_ok else f"FAIL ({cand.isolation_error})"
            print(f"{elapsed:.1f}s — {status}")

        if cand.schema_ok and (cand.isolation_ok is not False):
            cand.output_path = write_accepted(cand, output_dir, f"{timestamp}_{i:03d}")
            print(f"  ACCEPTED: {cand.output_path.relative_to(PROJECT_ROOT)}")

        candidates.append(cand)

    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    # ── Summary ────────────────────────────────────────────────────────────
    accepted = [c for c in candidates if c.output_path]
    schema_fails = [c for c in candidates if not c.schema_ok]
    isolation_fails = [c for c in candidates if c.schema_ok and c.isolation_ok is False]

    print(f"\n{'=' * 60}")
    print(f"  Total failures processed: {len(candidates)}")
    print(f"  Schema-failed:  {len(schema_fails)}")
    print(f"  Isolation-failed: {len(isolation_fails)}")
    print(f"  Accepted:       {len(accepted)}")
    print(f"  Output dir:     {output_dir.relative_to(PROJECT_ROOT)}")
    if accepted:
        print("\n  Next step: review each accepted YAML, then run the full benchmark")
        print("             to verify no regression before moving out of quarantine.")

    return 0 if accepted else 4


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Generate augmentor examples from benchmark failures")
    p.add_argument("--results", required=True, help="Benchmark JSON (from benchmark_realworld.py)")
    p.add_argument("--generator", default="models/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
                   help="Larger model that drafts YAML examples (default: Qwen 2.5 Coder 3B, same family as typical benchmark targets)")
    p.add_argument("--target", default="models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
                   help="Small target model used for isolation testing")
    p.add_argument("--output", default="data/augmentor_examples/generated",
                   help="Where to save accepted candidates (quarantine dir)")
    p.add_argument("--gpu-layers", type=int, default=99)
    p.add_argument("--max-failures", type=int, default=0,
                   help="Limit to first N failures (0 = all)")
    p.add_argument("--no-isolate", action="store_true",
                   help="Skip isolation gate (schema-only, faster but unsafe)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print failures that would be processed, then exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
