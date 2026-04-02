#!/usr/bin/env python3
"""
Multi-Agent Code Assistant — Architect + Workers + Assembler

The 3B model plans, the 0.5B models build, the 3B model stitches.

Usage:
    python multi_agent.py "build a web scraper with database storage and cli interface"
    python multi_agent.py --interactive
    python multi_agent.py --architect-model models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Code Assistant")
    parser.add_argument("prompt", nargs="?", help="The coding task to decompose and build")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--architect-model", default="models/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
                        help="Large model for planning/assembly")
    parser.add_argument("--worker-model", default="models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
                        help="Small model for code generation")
    parser.add_argument("--workers", type=int, default=1,
                        help="Max parallel workers (1 = sequential)")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--context-length", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from engine.architect import MultiAgentOrchestrator

    print()
    print("  Multi-Agent Code Assistant")
    print(f"  Architect: {Path(args.architect_model).stem}")
    print(f"  Worker:    {Path(args.worker_model).stem}")
    print(f"  Pipeline:  ARCHITECT (3B) -> WORKERS (0.5B) -> ASSEMBLER (3B)")
    print()

    orchestrator = MultiAgentOrchestrator(
        architect_model_path=args.architect_model,
        worker_model_path=args.worker_model,
        max_workers=args.workers,
        gpu_layers=args.gpu_layers,
        context_length=args.context_length,
    )

    print("  Loading models...")
    orchestrator.initialize()
    print("  Ready!\n")

    if args.interactive or not args.prompt:
        # Interactive mode
        print("  Type a complex coding task. Type 'quit' to exit.")
        print("  Example: build a task queue with sqlite backend, worker threads, and cli interface\n")

        while True:
            try:
                prompt = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break

            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                print("  Goodbye!")
                break

            result = orchestrator.process(prompt)

            print(f"\n{'='*70}")
            print(f"  FINAL CODE ({len(result.code.split(chr(10)))} lines)")
            print(f"{'='*70}")
            print()
            print(result.code)
            print()
            print(f"{'='*70}")
            print(f"  Pipeline: plan={result.plan_time:.1f}s "
                  f"build={result.build_time:.1f}s "
                  f"assemble={result.assemble_time:.1f}s "
                  f"total={result.total_time:.1f}s")
            print(f"{'='*70}\n")
    else:
        # Single prompt mode
        result = orchestrator.process(args.prompt)

        print(f"\n{'='*70}")
        print(f"  FINAL CODE ({len(result.code.split(chr(10)))} lines)")
        print(f"{'='*70}")
        print()
        print(result.code)
        print()
        print(f"{'='*70}")
        print(f"  Pipeline: plan={result.plan_time:.1f}s "
              f"build={result.build_time:.1f}s "
              f"assemble={result.assemble_time:.1f}s "
              f"total={result.total_time:.1f}s")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
