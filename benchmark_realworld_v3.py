#!/usr/bin/env python3
"""
Real-World Benchmark V3 — Edge Cases, Architecture, and Weird Prompts

100 queries designed to break the system:
- Vague/ambiguous prompts
- Multi-concept mashups
- Architecture patterns the model hasn't seen
- Deliberately tricky phrasing
- Non-Python-looking requests that need Python answers
- Things beginners actually type

Usage:
    python benchmark_realworld_v3.py
    python benchmark_realworld_v3.py --check-routing
    python benchmark_realworld_v3.py --model models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_realworld import RealWorldQuery, extract_code, check_query, check_routing_only, run_benchmark


def build_edge_case_queries() -> list[RealWorldQuery]:
    """100 queries that push boundaries — edge cases, architecture, weird phrasing."""
    return [
        # ── Vague / Beginner Prompts (15) ──
        # These are how real beginners actually type — no technical terms
        RealWorldQuery("make something that remembers things", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("i want to read a file and do something with each line", "general",
                       must_contain=["open", "for"], min_lines=3),
        RealWorldQuery("how do i make my code faster", "general",
                       must_contain=["def "], min_lines=3),
        RealWorldQuery("thing that takes a list and removes the duplicates", "general",
                       must_contain=["def ", "return"], min_lines=2),
        RealWorldQuery("make a thing that goes through a website and grabs stuff", "general",
                       must_contain=["def "], min_lines=4),
        RealWorldQuery("i keep getting an error when i try to open a file that doesnt exist", "general",
                       must_contain=["try", "except"], min_lines=3),
        RealWorldQuery("whats the best way to store user settings in python", "general",
                       must_contain=["def "], min_lines=3),
        RealWorldQuery("help me make a login system", "general",
                       must_contain=["def ", "password"], min_lines=5),
        RealWorldQuery("i need to send data between two scripts", "general",
                       must_contain=["def "], min_lines=4),
        RealWorldQuery("make python wait for 5 seconds then do something", "general",
                       must_contain=["time", "sleep"], min_lines=2),
        RealWorldQuery("how to make a list of lists and access stuff in it", "general",
                       must_contain=["["], min_lines=2),
        RealWorldQuery("i want to count how many times something appears", "general",
                       must_contain=["def ", "return"], min_lines=2),
        RealWorldQuery("write code that keeps asking for input until you type quit", "general",
                       must_contain=["while", "input"], min_lines=3),
        RealWorldQuery("turn a json string into something i can use in python", "general",
                       must_contain=["json"], min_lines=2),
        RealWorldQuery("how to save a python object to a file and load it later", "general",
                       must_contain=["def "], min_lines=3),

        # ── Multi-Concept Mashups (15) ──
        # Queries that combine 2-3 concepts the model needs to handle together
        RealWorldQuery("write an async sqlite database class with connection pooling", "database",
                       must_contain=["async", "sqlite"], min_lines=8),
        RealWorldQuery("create a FastAPI endpoint that reads from sqlite and returns paginated json", "web",
                       must_contain=["def ", "sqlite"], min_lines=8),
        RealWorldQuery("build a decorator that caches results in sqlite with a ttl", "pattern",
                       must_contain=["def ", "sqlite3", "cache"], min_lines=8),
        RealWorldQuery("write a thread-safe LRU cache that persists to disk on shutdown", "pattern",
                       must_contain=["class ", "Lock"], min_lines=10),
        RealWorldQuery("create a cli tool that monitors a directory and runs pytest when files change", "cli",
                       must_contain=["def ", "watch"], min_lines=8),
        RealWorldQuery("build a rate-limited async http client with retry and logging", "web",
                       must_contain=["async", "def "], min_lines=8),
        RealWorldQuery("write a pub-sub system that uses sqlite for message persistence", "pattern",
                       must_contain=["class ", "publish"], min_lines=8),
        RealWorldQuery("create a dataclass that validates fields and serializes to json", "data",
                       must_contain=["dataclass", "json"], min_lines=6),
        RealWorldQuery("build a middleware chain that logs requests, checks auth, and rate limits", "web",
                       must_contain=["def ", "middleware"], min_lines=8),
        RealWorldQuery("write an async generator that reads a csv file in chunks", "async",
                       must_contain=["async", "yield"], min_lines=5),
        RealWorldQuery("create a state machine that persists its state to a json file", "pattern",
                       must_contain=["class ", "state"], min_lines=8),
        RealWorldQuery("build a test fixture that spins up a fastapi test server with a test database", "testing",
                       must_contain=["def ", "test"], min_lines=5),
        RealWorldQuery("write a generic repository class that works with any dataclass model", "database",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("create a pipeline that reads json, validates it, transforms it, and writes csv", "data",
                       must_contain=["def ", "json", "csv"], min_lines=6),
        RealWorldQuery("build a command that takes a git diff and generates a changelog entry", "cli",
                       must_contain=["def "], min_lines=5),

        # ── Architecture Patterns (15) ──
        # Design patterns and architectural concepts
        RealWorldQuery("implement the strategy pattern for different sorting algorithms", "pattern",
                       must_contain=["class ", "def sort"], min_lines=8),
        RealWorldQuery("write a dependency injection container in python", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("create a command pattern with undo support", "pattern",
                       must_contain=["class ", "undo"], min_lines=8),
        RealWorldQuery("build an abstract factory for creating different database connections", "pattern",
                       must_contain=["class ", "def create"], min_lines=8),
        RealWorldQuery("implement the chain of responsibility pattern for request handling", "pattern",
                       must_contain=["class ", "handle"], min_lines=8),
        RealWorldQuery("write a specification pattern for filtering objects", "pattern",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("create an event sourcing system where state is derived from events", "pattern",
                       must_contain=["class ", "event"], min_lines=8),
        RealWorldQuery("build a CQRS pattern with separate read and write models", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("implement a circuit breaker that tracks failures and opens after threshold", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write a saga pattern for coordinating distributed transactions", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("create a unit of work pattern for managing database transactions", "pattern",
                       must_contain=["class ", "commit"], min_lines=8),
        RealWorldQuery("build a hexagonal architecture adapter for an external api", "pattern",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("implement the visitor pattern for traversing an ast", "pattern",
                       must_contain=["class ", "visit"], min_lines=8),
        RealWorldQuery("write a builder pattern for constructing complex objects step by step", "pattern",
                       must_contain=["class ", "build"], min_lines=8),
        RealWorldQuery("create a mediator that coordinates communication between components", "pattern",
                       must_contain=["class ", "def "], min_lines=8),

        # ── Tricky / Edge Case Code (15) ──
        # Things that trip up small models
        RealWorldQuery("write a function that deep copies a nested dict without using copy module", "general",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("implement __eq__ __hash__ and __repr__ for a custom class", "general",
                       must_contain=["__eq__", "__hash__", "__repr__"], min_lines=6),
        RealWorldQuery("write a metaclass that automatically registers all subclasses", "general",
                       must_contain=["class ", "Meta"], min_lines=6),
        RealWorldQuery("create a descriptor that validates type and range on attribute set", "pattern",
                       must_contain=["__set__", "class "], min_lines=6),
        RealWorldQuery("write a function that accepts *args and **kwargs and forwards them correctly", "general",
                       must_contain=["*args", "**kwargs"], min_lines=3),
        RealWorldQuery("implement __enter__ and __exit__ that work with nested with statements", "pattern",
                       must_contain=["__enter__", "__exit__"], min_lines=5),
        RealWorldQuery("write a generator that yields fibonacci numbers infinitely", "general",
                       must_contain=["def ", "yield"], min_lines=4),
        RealWorldQuery("create a class that uses __slots__ for memory efficiency", "general",
                       must_contain=["__slots__"], min_lines=4),
        RealWorldQuery("write a function decorator that preserves the original function signature", "pattern",
                       must_contain=["wraps", "def "], min_lines=5),
        RealWorldQuery("implement __getattr__ to create a proxy object that logs all method calls", "general",
                       must_contain=["__getattr__", "class "], min_lines=6),
        RealWorldQuery("write a coroutine-based state machine using yield and send", "general",
                       must_contain=["yield", "send"], min_lines=6),
        RealWorldQuery("create a weakref callback that cleans up when an object is garbage collected", "general",
                       must_contain=["weakref", "def "], min_lines=4),
        RealWorldQuery("write a context variable that works across async tasks", "async",
                       must_contain=["contextvars", "def "], min_lines=4),
        RealWorldQuery("implement a custom exception with extra attributes and proper pickling", "general",
                       must_contain=["class ", "Exception"], min_lines=5),
        RealWorldQuery("write a function that patches itself to cache its first result", "general",
                       must_contain=["def "], min_lines=4),

        # ── System / Low Level (10) ──
        RealWorldQuery("write a memory-mapped file reader for processing huge files", "general",
                       must_contain=["mmap", "def "], min_lines=5),
        RealWorldQuery("create a signal handler for graceful shutdown on ctrl-c", "cli",
                       must_contain=["signal", "def "], min_lines=4),
        RealWorldQuery("write a function that monitors memory usage of the current process", "cli",
                       must_contain=["def "], min_lines=4),
        RealWorldQuery("build a simple profiler that measures function execution time", "general",
                       must_contain=["def ", "time"], min_lines=5),
        RealWorldQuery("create a file lock using fcntl or msvcrt for cross-platform locking", "cli",
                       must_contain=["def ", "lock"], min_lines=5),
        RealWorldQuery("write a daemon process that runs in the background", "cli",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("implement a simple garbage collector using reference counting", "general",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write a ctypes wrapper for calling a c function from python", "general",
                       must_contain=["ctypes", "def "], min_lines=4),
        RealWorldQuery("create a shared memory region between two python processes", "general",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("write a custom importer that loads python modules from a zip file", "general",
                       must_contain=["class ", "import"], min_lines=6),

        # ── Unusual Requests (10) ──
        RealWorldQuery("make a quine - a python program that prints its own source code", "general",
                       must_contain=["print"], min_lines=1),
        RealWorldQuery("write python that generates valid python code for a given function spec", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("create a brainfuck interpreter in python", "general",
                       must_contain=["def ", "while"], min_lines=8),
        RealWorldQuery("write a function that converts an integer to roman numerals", "algorithm",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("build a truth table generator for boolean expressions", "algorithm",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("write a sudoku solver using backtracking", "algorithm",
                       must_contain=["def ", "solve"], min_lines=8),
        RealWorldQuery("create a markov chain text generator from a corpus", "algorithm",
                       must_contain=["def ", "return"], min_lines=6),
        RealWorldQuery("write a diff algorithm that produces unified diff output", "algorithm",
                       must_contain=["def ", "diff"], min_lines=5),
        RealWorldQuery("build a simple lisp interpreter with eval and apply", "algorithm",
                       must_contain=["def ", "eval"], min_lines=8),
        RealWorldQuery("create a genetic algorithm that evolves a string to match a target", "algorithm",
                       must_contain=["def ", "fitness"], min_lines=8),

        # ── Real Project Scenarios (20) ──
        RealWorldQuery("i have a csv with columns name,email,age - write code to find all users over 30 and export to a new csv", "data",
                       must_contain=["csv", "30"], min_lines=5),
        RealWorldQuery("my api returns nested json - write a function to flatten it into a table-like format", "data",
                       must_contain=["def ", "flatten"], min_lines=5),
        RealWorldQuery("write a migration script that adds a column to an existing sqlite table safely", "database",
                       must_contain=["sqlite3", "ALTER"], min_lines=4),
        RealWorldQuery("create a caching decorator that invalidates after n seconds", "pattern",
                       must_contain=["def ", "cache"], min_lines=6),
        RealWorldQuery("write a function that retries an http request with exponential backoff and jitter", "web",
                       must_contain=["def ", "retry"], min_lines=6),
        RealWorldQuery("build a simple orm that maps python classes to sqlite tables automatically", "database",
                       must_contain=["class ", "sqlite3"], min_lines=10),
        RealWorldQuery("write a background job processor that picks up tasks from a database queue", "pattern",
                       must_contain=["def ", "while"], min_lines=8),
        RealWorldQuery("create a feature flag system that reads flags from a config file", "general",
                       must_contain=["def ", "flag"], min_lines=5),
        RealWorldQuery("write a data anonymizer that replaces names and emails with fake data", "data",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("build a webhook receiver that validates signatures and processes events", "web",
                       must_contain=["def ", "signature"], min_lines=6),
        RealWorldQuery("write a connection string parser for database urls like postgres://user:pass@host/db", "database",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("create a batch processor that handles items in groups of n with error recovery", "pattern",
                       must_contain=["def ", "batch"], min_lines=6),
        RealWorldQuery("write a simple ab test framework that assigns users to variants", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("build a circuit breaker decorator with configurable threshold and reset time", "pattern",
                       must_contain=["def ", "class "], min_lines=8),
        RealWorldQuery("write a multitenancy wrapper that prefixes all database queries with tenant id", "database",
                       must_contain=["def ", "tenant"], min_lines=5),
        RealWorldQuery("create an api versioning system using url prefixes", "web",
                       must_contain=["def ", "version"], min_lines=5),
        RealWorldQuery("write a health check endpoint that tests database connectivity and returns status", "web",
                       must_contain=["def ", "health"], min_lines=5),
        RealWorldQuery("build a simple etl pipeline that extracts from csv transforms dates and loads to sqlite", "data",
                       must_contain=["csv", "sqlite3"], min_lines=8),
        RealWorldQuery("write a request validator middleware that checks required headers", "web",
                       must_contain=["def ", "header"], min_lines=5),
        RealWorldQuery("create an idempotency key system for preventing duplicate api requests", "web",
                       must_contain=["def ", "idempoten"], min_lines=5),
    ]


def main():
    parser = argparse.ArgumentParser(description="Real-World Benchmark V3 — Edge Cases")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--quick", action="store_true", help="Run 20 queries only")
    parser.add_argument("--check-routing", action="store_true", help="Check routing only")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=str, default="benchmark_realworld_v3_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    queries = build_edge_case_queries()
    print(f"  Set 3 (edge cases + architecture): {len(queries)} queries")

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

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    data = run_benchmark(model_path, queries, args.gpu_layers, args.threads)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
