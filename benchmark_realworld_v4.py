#!/usr/bin/env python3
"""
Real-World Benchmark V4 — Deep Gaps, Niche Patterns, Production Scenarios

100 queries targeting areas we haven't tested:
- Type hints / generics
- Functional programming
- Protocols / ABCs
- Performance patterns
- Config management
- Logging / monitoring
- Data validation edge cases
- Production hardening patterns

Usage:
    python benchmark_realworld_v4.py
    python benchmark_realworld_v4.py --check-routing
    python benchmark_realworld_v4.py --model models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_realworld import RealWorldQuery, extract_code, check_query, check_routing_only, run_benchmark


def build_deep_gap_queries() -> list[RealWorldQuery]:
    """100 queries targeting unexplored territory."""
    return [
        # ── Functional Programming (12) ──
        RealWorldQuery("write a compose function that chains multiple functions together", "general",
                       must_contain=["def compose", "return"], min_lines=3),
        RealWorldQuery("create a curry function that turns f(a,b,c) into f(a)(b)(c)", "general",
                       must_contain=["def curry", "return"], min_lines=4),
        RealWorldQuery("implement a pipe function that passes data through a series of transforms", "general",
                       must_contain=["def pipe", "return"], min_lines=3),
        RealWorldQuery("write a memoize decorator using a dictionary cache", "pattern",
                       must_contain=["def ", "cache"], min_lines=5),
        RealWorldQuery("create a partial application function without using functools.partial", "general",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("write a function that flattens an arbitrarily nested list", "general",
                       must_contain=["def flatten", "return"], min_lines=4),
        RealWorldQuery("implement map filter and reduce from scratch without builtins", "general",
                       must_contain=["def "], min_lines=8),
        RealWorldQuery("write a trampoline function for tail-call optimization", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("create a lazy evaluation wrapper that delays computation until needed", "general",
                       must_contain=["class ", "def "], min_lines=5),
        RealWorldQuery("write a function that groups items by a key function like itertools.groupby", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("implement a maybe monad for handling None values in a chain", "general",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("write a transducer that composes map and filter without intermediate lists", "general",
                       must_contain=["def ", "return"], min_lines=5),

        # ── Type Hints / Protocols (10) ──
        RealWorldQuery("write a generic stack class with type hints using TypeVar", "general",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("create a Protocol class that defines a Comparable interface", "general",
                       must_contain=["class ", "def "], min_lines=4),
        RealWorldQuery("write a function with overloaded signatures using typing.overload", "general",
                       must_contain=["overload", "def "], min_lines=6),
        RealWorldQuery("create a TypedDict for a json api response", "general",
                       must_contain=["TypedDict", "class "], min_lines=4),
        RealWorldQuery("write a generic function that works with any iterable and returns a list", "general",
                       must_contain=["def ", "Iterable"], min_lines=3),
        RealWorldQuery("implement a runtime type checker decorator that validates argument types", "pattern",
                       must_contain=["def ", "type"], min_lines=6),
        RealWorldQuery("create a dataclass with custom validators using __post_init__", "general",
                       must_contain=["dataclass", "def "], min_lines=6),
        RealWorldQuery("write a type-safe event system using generics", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("create a Union type handler that dispatches based on the actual type", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("write a recursive type hint for a tree structure like a json value", "general",
                       must_contain=["Union", "Dict"], min_lines=3),

        # ── Performance Patterns (10) ──
        RealWorldQuery("write a lru cache decorator that limits memory by number of entries", "pattern",
                       must_contain=["def ", "cache"], min_lines=6),
        RealWorldQuery("create an object pool that reuses expensive objects instead of creating new ones", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write a bloom filter for fast approximate set membership testing", "algorithm",
                       must_contain=["class ", "def add", "def "], min_lines=8),
        RealWorldQuery("implement a ring buffer with fixed size that overwrites old entries", "algorithm",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("write a debounce function that delays execution until calls stop", "pattern",
                       must_contain=["def ", "time"], min_lines=5),
        RealWorldQuery("create a throttle function that limits how often a function can run", "pattern",
                       must_contain=["def ", "time"], min_lines=5),
        RealWorldQuery("write a lazy property that computes once and caches the result", "pattern",
                       must_contain=["class ", "def "], min_lines=5),
        RealWorldQuery("implement a skip list for O(log n) search in a sorted collection", "algorithm",
                       must_contain=["class ", "def "], min_lines=10),
        RealWorldQuery("write a connection pool with max size and timeout on acquire", "pattern",
                       must_contain=["class ", "def acquire"], min_lines=8),
        RealWorldQuery("create a batch accumulator that flushes when size or time threshold is hit", "pattern",
                       must_contain=["class ", "def "], min_lines=8),

        # ── Config / Environment (8) ──
        RealWorldQuery("write a settings class that loads from environment variables with defaults", "cli",
                       must_contain=["os.environ", "class "], min_lines=5),
        RealWorldQuery("create a config system that merges yaml file, env vars, and cli args in priority order", "cli",
                       must_contain=["def ", "config"], min_lines=6),
        RealWorldQuery("write a secrets manager that reads encrypted values from a file", "general",
                       must_contain=["def ", "secret"], min_lines=5),
        RealWorldQuery("build a feature toggle system with percentage-based rollouts", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("create a dotenv file parser that loads key=value pairs into os.environ", "cli",
                       must_contain=["def ", "os.environ"], min_lines=4),
        RealWorldQuery("write a config watcher that reloads when the config file changes", "cli",
                       must_contain=["def ", "watch"], min_lines=6),
        RealWorldQuery("build a hierarchical config that inherits values from parent configs", "general",
                       must_contain=["class ", "def get"], min_lines=6),
        RealWorldQuery("create an app config with validation that fails fast on missing required values", "general",
                       must_contain=["class ", "def "], min_lines=5),

        # ── Logging / Monitoring (8) ──
        RealWorldQuery("write a structured json logger that outputs one json object per log line", "cli",
                       must_contain=["json", "log"], min_lines=5),
        RealWorldQuery("create a metrics collector that tracks count, sum, and average", "general",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("write a function timer context manager that logs slow functions", "pattern",
                       must_contain=["def ", "time"], min_lines=5),
        RealWorldQuery("build a request id middleware that adds a unique id to every request", "web",
                       must_contain=["def ", "uuid"], min_lines=5),
        RealWorldQuery("create a log aggregator that groups log lines by request id", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("write a decorator that logs function entry exit and exceptions", "pattern",
                       must_contain=["def ", "wrapper"], min_lines=6),
        RealWorldQuery("build a simple alerting system that sends notifications when metrics exceed thresholds", "general",
                       must_contain=["def ", "threshold"], min_lines=6),
        RealWorldQuery("create a ring buffer for keeping the last N log entries in memory", "general",
                       must_contain=["class ", "def "], min_lines=6),

        # ── Data Validation Edge Cases (8) ──
        RealWorldQuery("write a recursive json schema validator that handles nested objects and arrays", "data",
                       must_contain=["def validate", "return"], min_lines=8),
        RealWorldQuery("create a form validator with chained rules like required().min(3).max(100)", "data",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write a function that sanitizes html input to prevent xss", "general",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("build a data coercion function that converts strings to appropriate python types", "data",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("create a diff function that shows what changed between two dicts", "data",
                       must_contain=["def diff", "return"], min_lines=5),
        RealWorldQuery("write a deepcompare function that handles nested dicts lists and sets", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("build a data masking function that redacts sensitive fields in nested dicts", "data",
                       must_contain=["def ", "mask"], min_lines=5),
        RealWorldQuery("create a schema migration function that transforms old data format to new format", "data",
                       must_contain=["def ", "return"], min_lines=5),

        # ── Networking Deep (8) ──
        RealWorldQuery("write a tcp chat room where multiple clients can send messages to each other", "web",
                       must_contain=["socket", "def "], min_lines=10),
        RealWorldQuery("create a simple http proxy that forwards requests to a backend server", "web",
                       must_contain=["def ", "socket"], min_lines=8),
        RealWorldQuery("write a udp packet sender and receiver", "web",
                       must_contain=["socket", "UDP"], min_lines=6),
        RealWorldQuery("build a port scanner that checks if ports are open on a host", "web",
                       must_contain=["socket", "def "], min_lines=5),
        RealWorldQuery("create a simple dns resolver that sends udp queries", "web",
                       must_contain=["socket", "def "], min_lines=6),
        RealWorldQuery("write an http server from scratch using only sockets", "web",
                       must_contain=["socket", "HTTP"], min_lines=10),
        RealWorldQuery("build a websocket handshake handler that upgrades an http connection", "web",
                       must_contain=["def ", "websocket"], min_lines=6),
        RealWorldQuery("create a connection multiplexer that handles multiple sockets with select", "web",
                       must_contain=["select", "socket"], min_lines=8),

        # ── Metaprogramming (8) ──
        RealWorldQuery("write a class decorator that adds comparison methods based on a key", "general",
                       must_contain=["def ", "class"], min_lines=6),
        RealWorldQuery("create a metaclass that enforces all methods have docstrings", "general",
                       must_contain=["class ", "Meta"], min_lines=6),
        RealWorldQuery("write a function that creates a new class dynamically with given attributes", "general",
                       must_contain=["class", "return"], min_lines=4),
        RealWorldQuery("build a mixin system where classes can compose behavior from multiple mixins", "general",
                       must_contain=["class "], min_lines=8),
        RealWorldQuery("create a property factory that generates getters and setters with validation", "general",
                       must_contain=["def ", "property"], min_lines=5),
        RealWorldQuery("write an auto-repr mixin that generates __repr__ from instance attributes", "general",
                       must_contain=["__repr__", "class "], min_lines=3),
        RealWorldQuery("implement a frozen class decorator that prevents attribute modification after init", "general",
                       must_contain=["def ", "class"], min_lines=5),
        RealWorldQuery("create an abstract base class with abstract methods and a register mechanism", "general",
                       must_contain=["ABC", "abstract"], min_lines=6),

        # ── Production Hardening (12) ──
        RealWorldQuery("write a graceful shutdown handler that finishes in-flight requests before stopping", "cli",
                       must_contain=["def ", "shutdown"], min_lines=6),
        RealWorldQuery("create a health check that verifies database connectivity and disk space", "cli",
                       must_contain=["def ", "health"], min_lines=5),
        RealWorldQuery("write an exponential backoff with jitter for retrying failed operations", "pattern",
                       must_contain=["def ", "retry"], min_lines=5),
        RealWorldQuery("build a bulkhead pattern that limits concurrent access to a resource", "pattern",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("create a dead letter queue for messages that fail processing", "pattern",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("write a request deduplication system using idempotency keys", "web",
                       must_contain=["def ", "key"], min_lines=5),
        RealWorldQuery("build a token bucket rate limiter with configurable rate and burst", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("create a canary deployment checker that compares error rates", "general",
                       must_contain=["def ", "error"], min_lines=5),
        RealWorldQuery("write a distributed lock using a file-based mechanism", "general",
                       must_contain=["def ", "lock"], min_lines=5),
        RealWorldQuery("build a worker pool with configurable concurrency and error handling", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("create a lease-based resource manager with automatic expiry", "pattern",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("write a chaos monkey function that randomly kills processes for testing resilience", "general",
                       must_contain=["def ", "random"], min_lines=5),

        # ── Algorithms Deep (8) ──
        RealWorldQuery("implement dijkstra's shortest path algorithm", "algorithm",
                       must_contain=["def ", "return"], min_lines=8),
        RealWorldQuery("write an a-star pathfinding algorithm on a 2d grid", "algorithm",
                       must_contain=["def ", "heuristic"], min_lines=10),
        RealWorldQuery("create a red-black tree with insert and rebalance", "algorithm",
                       must_contain=["class ", "def "], min_lines=10),
        RealWorldQuery("implement knuth-morris-pratt string matching algorithm", "algorithm",
                       must_contain=["def ", "return"], min_lines=8),
        RealWorldQuery("write a huffman encoding and decoding system", "algorithm",
                       must_contain=["class ", "def "], min_lines=10),
        RealWorldQuery("create a disjoint set (union-find) with path compression", "algorithm",
                       must_contain=["class ", "find", "union"], min_lines=6),
        RealWorldQuery("implement a b-tree with insert and search operations", "algorithm",
                       must_contain=["class ", "def "], min_lines=10),
        RealWorldQuery("write a consistent hashing ring for distributed systems", "algorithm",
                       must_contain=["class ", "def "], min_lines=8),

        # ── Miscellaneous Hard (8) ──
        RealWorldQuery("write a python debugger that hooks into sys.settrace", "general",
                       must_contain=["settrace", "def "], min_lines=6),
        RealWorldQuery("create a sandboxed exec that limits what code can do", "general",
                       must_contain=["exec", "def "], min_lines=5),
        RealWorldQuery("write a simple bytecode interpreter for a stack-based language", "algorithm",
                       must_contain=["def ", "stack"], min_lines=10),
        RealWorldQuery("build a task dependency resolver using topological sort", "algorithm",
                       must_contain=["def ", "return"], min_lines=6),
        RealWorldQuery("create a finite automaton that matches a simple pattern", "algorithm",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write an incremental parser that processes data as it arrives in chunks", "general",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("build a promise/future implementation with then and catch", "pattern",
                       must_contain=["class ", "then"], min_lines=8),
        RealWorldQuery("create a reactive stream with map filter and subscribe operators", "pattern",
                       must_contain=["class ", "subscribe"], min_lines=8),
    ]


def main():
    parser = argparse.ArgumentParser(description="Real-World Benchmark V4 — Deep Gaps")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--quick", action="store_true", help="Run 20 queries only")
    parser.add_argument("--check-routing", action="store_true", help="Check routing only")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=str, default="benchmark_realworld_v4_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    queries = build_deep_gap_queries()
    print(f"  Set 4 (deep gaps + niche patterns): {len(queries)} queries")

    if args.quick:
        by_domain = {}
        for q in queries:
            by_domain.setdefault(q.domain, []).append(q)
        queries = []
        for domain_queries in by_domain.values():
            queries.extend(domain_queries[:2])
        print(f"  Quick mode: {len(queries)} queries")

    if args.check_routing:
        print(f"\n  Routing Check ({len(queries)} queries)")
        print(f"  {'=' * 60}")
        check_routing_only(queries)
        return

    if args.model:
        model_path = Path(args.model)
    else:
        models = sorted(Path("models").glob("*.gguf"), key=lambda p: p.stat().st_size)
        if not models:
            print("No models found.")
            sys.exit(1)
        model_path = models[0]

    data = run_benchmark(model_path, queries, args.gpu_layers, args.threads)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
