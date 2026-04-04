#!/usr/bin/env python3
"""
Multi-Agent Architecture — Systematic Benchmark

Runs each test prompt N times, validates outputs, and produces
averaged scores with variance. Tests:
  1. Syntax: does compile() pass?
  2. Imports: do all imports resolve?
  3. Structure: are all expected classes/functions present?
  4. Exec: does exec() succeed (with mocked externals)?
"""

import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

PROMPTS = [
    {
        "name": "Task Queue",
        "prompt": "build a task queue with sqlite backend, worker threads, and a cli interface",
    },
    {
        "name": "REST API Auth",
        "prompt": "build a REST API with user registration, password hashing, token-based authentication, and a protected endpoint that returns user profile data",
    },
    {
        "name": "Web Scraper",
        "prompt": "build a web scraper that fetches pages with rate limiting, parses HTML with BeautifulSoup, stores results in a SQLite database, and exports to CSV",
    },
    {
        "name": "Chat Server",
        "prompt": "build a chat server with asyncio, supporting multiple rooms, user nicknames, message history stored in memory, and a command system for /join /leave /list /nick",
    },
    {
        "name": "File Sync",
        "prompt": "build a file synchronization tool that watches a directory for changes using filesystem events, computes file checksums, maintains a sync manifest in JSON, and copies changed files to a destination",
    },
    {
        "name": "Plugin System",
        "prompt": "build a plugin system with a plugin loader that discovers plugins from a directory, a plugin registry, a hook system for pre/post events, and a plugin manager with enable/disable/reload",
    },
    {
        "name": "Data Pipeline",
        "prompt": "build a data pipeline with a CSV reader that streams rows, a transformer that applies column mappings and type conversions, a validator that checks constraints, and a writer that outputs to JSON lines format",
    },
]


@dataclass
class RunResult:
    prompt_name: str
    run: int
    total_time: float
    plan_time: float
    build_time: float
    assemble_time: float
    num_subtasks: int
    num_sections: int
    output_lines: int
    output_chars: int
    # Validation
    syntax_ok: bool
    syntax_error: str = ""
    imports_ok: bool = False
    import_errors: list = field(default_factory=list)
    classes_found: list = field(default_factory=list)
    functions_found: list = field(default_factory=list)
    exec_ok: bool = False
    exec_error: str = ""


def check_syntax(code: str) -> tuple[bool, str]:
    """Check if code compiles."""
    try:
        compile(code, "<test>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} (line {e.lineno})"


def check_imports(code: str) -> tuple[bool, list[str]]:
    """Check if all imports can be resolved."""
    errors = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            parts = stripped.split()
            if len(parts) < 2:
                continue
            mod = parts[1].split(".")[0]
            if not mod:
                continue
            try:
                __import__(mod)
            except (ImportError, ValueError):
                errors.append(mod)
    return len(errors) == 0, errors


def find_definitions(code: str) -> tuple[list[str], list[str]]:
    """Find all top-level class and function definitions."""
    classes = re.findall(r'^class\s+(\w+)', code, re.MULTILINE)
    functions = re.findall(r'^def\s+(\w+)', code, re.MULTILINE)
    return classes, functions


def check_exec(code: str) -> tuple[bool, str]:
    """Try to exec the code (minus the main call) with mocked externals."""
    # Remove if __name__ block to avoid running main
    lines = code.split("\n")
    filtered = []
    skip = False
    for line in lines:
        if line.strip().startswith("if __name__"):
            skip = True
            continue
        if skip:
            if line and not line[0].isspace() and line.strip():
                skip = False
                filtered.append(line)
            continue
        filtered.append(line)

    test_code = "\n".join(filtered)
    try:
        env = {"__builtins__": __builtins__, "__name__": "__exec_test__"}
        exec(compile(test_code, "<exec_test>", "exec"), env)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run_benchmark(runs_per_prompt: int = 3):
    from engine.architect import MultiAgentOrchestrator

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n  Multi-Agent Benchmark — {len(PROMPTS)} prompts x {runs_per_prompt} runs")
    print(f"  {'='*60}\n")

    # Initialize once
    print("  Loading models...", end=" ", flush=True)
    orch = MultiAgentOrchestrator()
    orch.initialize()
    print("done.\n")

    all_results: list[RunResult] = []

    for pi, pinfo in enumerate(PROMPTS):
        name = pinfo["name"]
        prompt = pinfo["prompt"]
        print(f"  [{pi+1}/{len(PROMPTS)}] {name}")

        for run in range(runs_per_prompt):
            start = time.monotonic()
            result = orch.process(prompt)
            elapsed = time.monotonic() - start

            code = result.code or ""
            lines = len(code.strip().split("\n")) if code.strip() else 0

            # Validations
            syn_ok, syn_err = check_syntax(code)
            imp_ok, imp_errs = check_imports(code) if syn_ok else (False, [])
            classes, funcs = find_definitions(code)
            ex_ok, ex_err = check_exec(code) if syn_ok else (False, "syntax error")

            rr = RunResult(
                prompt_name=name,
                run=run + 1,
                total_time=round(result.total_time, 1),
                plan_time=round(result.plan_time, 1),
                build_time=round(result.build_time, 1),
                assemble_time=round(result.assemble_time, 1),
                num_subtasks=len(result.plan.subtasks),
                num_sections=code.count("# -- "),
                output_lines=lines,
                output_chars=len(code),
                syntax_ok=syn_ok,
                syntax_error=syn_err,
                imports_ok=imp_ok,
                import_errors=imp_errs,
                classes_found=classes,
                functions_found=funcs,
                exec_ok=ex_ok,
                exec_error=ex_err,
            )
            all_results.append(rr)

            status = []
            status.append("SYN:OK" if syn_ok else f"SYN:FAIL({syn_err[:30]})")
            status.append("IMP:OK" if imp_ok else f"IMP:FAIL({imp_errs})")
            status.append("EXEC:OK" if ex_ok else f"EXEC:FAIL({ex_err[:40]})")
            print(f"    run {run+1}: {rr.total_time}s | {lines}L | {' | '.join(status)}")

        print()

    # Save raw results
    out_path = PROJECT_ROOT / "bench_multiagent_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"  Raw results saved to {out_path.name}")

    # Summary
    print(f"\n  {'='*60}")
    print(f"  SUMMARY — {len(PROMPTS)} prompts x {runs_per_prompt} runs = {len(all_results)} total")
    print(f"  {'='*60}\n")

    header = f"  {'Test':<20} {'Avg Time':>8} {'StdDev':>7} {'Syntax':>7} {'Import':>7} {'Exec':>7} {'Avg Lines':>9}"
    print(header)
    print(f"  {'-'*len(header.strip())}")

    totals = {"syntax": 0, "imports": 0, "exec": 0, "runs": 0}

    for pinfo in PROMPTS:
        name = pinfo["name"]
        runs = [r for r in all_results if r.prompt_name == name]
        times = [r.total_time for r in runs]
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
        syn_rate = sum(1 for r in runs if r.syntax_ok) / len(runs)
        imp_rate = sum(1 for r in runs if r.imports_ok) / len(runs)
        exec_rate = sum(1 for r in runs if r.exec_ok) / len(runs)
        avg_lines = sum(r.output_lines for r in runs) / len(runs)

        totals["syntax"] += sum(1 for r in runs if r.syntax_ok)
        totals["imports"] += sum(1 for r in runs if r.imports_ok)
        totals["exec"] += sum(1 for r in runs if r.exec_ok)
        totals["runs"] += len(runs)

        print(f"  {name:<20} {avg_time:>7.1f}s {std_time:>6.1f}s {syn_rate:>6.0%} {imp_rate:>6.0%} {exec_rate:>6.0%} {avg_lines:>9.0f}")

    n = totals["runs"]
    print(f"  {'-'*len(header.strip())}")
    print(f"  {'OVERALL':<20} {'':>8} {'':>7} {totals['syntax']/n:>6.0%} {totals['imports']/n:>6.0%} {totals['exec']/n:>6.0%}")

    # Failure analysis
    failures = [r for r in all_results if not r.exec_ok]
    if failures:
        print(f"\n  Exec failures ({len(failures)}/{n}):")
        for r in failures:
            err = r.exec_error or r.syntax_error
            print(f"    {r.prompt_name} run {r.run}: {err[:80]}")

    print()


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_benchmark(runs)
