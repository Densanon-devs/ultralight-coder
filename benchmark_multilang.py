#!/usr/bin/env python3
"""
Multi-Language Benchmark — 120 queries across 12 languages.

Tests whether the augmentor system produces correct code in the right language
for natural, varied prompts. 10 queries per language, each with must_contain
markers that verify both language correctness and code quality.

Usage:
    python benchmark_multilang.py                          # Run all
    python benchmark_multilang.py --lang javascript go     # Specific languages
    python benchmark_multilang.py --check-routing           # Just check routing (no model)
    python benchmark_multilang.py --model models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_realworld import RealWorldQuery, extract_code, check_query, run_benchmark


def build_multilang_queries(languages=None) -> list[RealWorldQuery]:
    """120 queries across 12 languages — 10 per language.

    Each query is a realistic ask in that language with markers that confirm:
    1. Code is in the right language (language-specific keywords)
    2. Code is structurally correct (min lines, required constructs)
    """
    all_queries = {}

    # ── Python (10) — baseline, should stay rock solid ──
    all_queries["python"] = [
        RealWorldQuery("Write a Python function that finds all anagrams in a list of words", "python",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("Write a Python decorator that measures execution time of a function", "python",
                       must_contain=["def ", "wrapper", "time"], min_lines=5),
        RealWorldQuery("Write a Python class that implements a priority queue using heapq", "python",
                       must_contain=["class ", "heapq"], min_lines=6),
        RealWorldQuery("Write a Python async function that fetches multiple URLs concurrently", "python",
                       must_contain=["async ", "await"], min_lines=5),
        RealWorldQuery("Write a Python context manager for a database transaction", "python",
                       must_contain=[("def ", "class "),
                                     ("__enter__", "contextmanager", "@contextmanager")],
                       min_lines=5),
        RealWorldQuery("Write a Python generator that yields fibonacci numbers", "python",
                       must_contain=["def ", "yield"], min_lines=4),
        RealWorldQuery("Write a Python function that validates a JSON schema", "python",
                       must_contain=["def ", "json"], min_lines=4),
        RealWorldQuery("Write a Python dataclass with validation in __post_init__", "python",
                       must_contain=["dataclass", "__post_init__"], min_lines=5),
        RealWorldQuery("Write a Python function that merges two sorted lists into one sorted list", "python",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("Write a Python class that implements the observer pattern", "python",
                       must_contain=["class ", "notify"], min_lines=8),
    ]

    # ── JavaScript (10) ──
    all_queries["javascript"] = [
        RealWorldQuery("Write a JavaScript function that deep merges two objects", "javascript",
                       must_contain=["function ", "return"], min_lines=5),
        RealWorldQuery("Write a JavaScript async function that retries a fetch request 3 times", "javascript",
                       must_contain=["async ", "fetch"], min_lines=5),
        RealWorldQuery("Write a JavaScript class for a publish-subscribe event system", "javascript",
                       must_contain=["class ", "subscribe"], min_lines=8),
        RealWorldQuery("Write a JavaScript function that flattens a deeply nested array", "javascript",
                       must_contain=["function ", "return"], min_lines=3),
        RealWorldQuery("Write a JavaScript Promise.all implementation from scratch", "javascript",
                       must_contain=["function ", "promise"], min_lines=6),
        RealWorldQuery("Write a JavaScript function that converts a flat array into a tree structure", "javascript",
                       must_contain=["function ", "return"], min_lines=5),
        RealWorldQuery("Write a JavaScript middleware pattern for processing requests", "javascript",
                       must_contain=["function ", "next"], min_lines=5),
        RealWorldQuery("Write a JavaScript function that implements memoization with a cache", "javascript",
                       must_contain=["function ", "cache"], min_lines=4),
        RealWorldQuery("Write a JavaScript class that implements a simple router with path matching", "javascript",
                       must_contain=["class ", "route"], min_lines=6),
        RealWorldQuery("Write a JavaScript function that validates a credit card number using Luhn algorithm", "javascript",
                       must_contain=["function ", "return"], min_lines=5),
    ]

    # ── TypeScript (10) ──
    all_queries["typescript"] = [
        RealWorldQuery("Write a TypeScript generic function that removes duplicates from an array", "typescript",
                       must_contain=["function ", "return"], min_lines=3),
        RealWorldQuery("Write a TypeScript interface for a REST API response with pagination", "typescript",
                       must_contain=["interface "], min_lines=4),
        RealWorldQuery("Write a TypeScript generic class for an observable value with subscribers", "typescript",
                       must_contain=["class "], min_lines=8),
        RealWorldQuery("Write a TypeScript discriminated union for different shape types with an area function", "typescript",
                       must_contain=["type ", "area"], min_lines=6),
        RealWorldQuery("Write a TypeScript generic function that creates a type-safe dictionary", "typescript",
                       must_contain=["function ", "return"], min_lines=4),
        RealWorldQuery("Write a TypeScript decorator that logs method calls with parameters", "typescript",
                       must_contain=["function ", "return"], min_lines=5),
        RealWorldQuery("Write a TypeScript utility type that makes all nested properties optional", "typescript",
                       must_contain=["type "], min_lines=3),
        RealWorldQuery("Write a TypeScript async function that fetches data with retry logic and timeout", "typescript",
                       must_contain=["async ", "fetch"], min_lines=6),
        RealWorldQuery("Write a TypeScript class that implements a simple dependency injection container", "typescript",
                       must_contain=["class "], min_lines=8),
        RealWorldQuery("Write a TypeScript generic linked list with type-safe nodes", "typescript",
                       must_contain=["class ", "next"], min_lines=8),
    ]

    # ── Go (10) ──
    all_queries["go"] = [
        RealWorldQuery("Write a Go function that reverses a slice of any type using generics", "go",
                       must_contain=["func ", "return"], min_lines=4),
        RealWorldQuery("Write a Go HTTP handler that accepts JSON POST requests and validates required fields", "go",
                       must_contain=["func ", "http"], min_lines=6),
        RealWorldQuery("Write a Go goroutine-based pipeline with three stages connected by channels", "go",
                       must_contain=["func ", "chan "], min_lines=8),
        RealWorldQuery("Write a Go function that reads a CSV file and returns a slice of structs", "go",
                       must_contain=["func ", "csv"], min_lines=6),
        RealWorldQuery("Write a Go interface for a cache with Get, Set, and Delete methods", "go",
                       must_contain=["interface ", "Get"], min_lines=4),
        RealWorldQuery("Write a Go function that walks a directory tree and returns all file paths matching a pattern", "go",
                       must_contain=["func ", "filepath"], min_lines=5),
        RealWorldQuery("Write a Go context-aware HTTP client with timeout and cancellation", "go",
                       must_contain=["func ", "context"], min_lines=5),
        RealWorldQuery("Write a Go struct with methods that implements the Stringer interface", "go",
                       must_contain=["func ", "String()"], min_lines=4),
        RealWorldQuery("Write a Go error type that wraps another error with additional context", "go",
                       must_contain=["func ", "Error()"], min_lines=4),
        RealWorldQuery("Write a Go function that merges multiple sorted channels into one sorted output channel", "go",
                       must_contain=["func ", "chan "], min_lines=8),
    ]

    # ── Rust (10) ──
    all_queries["rust"] = [
        RealWorldQuery("Write a Rust function that counts word frequency in a string using HashMap", "rust",
                       must_contain=["fn ", "HashMap"], min_lines=4),
        RealWorldQuery("Write a Rust struct for a linked list with push and pop methods", "rust",
                       must_contain=["struct ", ("impl ", "impl<", "impl(")], min_lines=8),
        RealWorldQuery("Write a Rust enum for a binary tree with insert and contains methods", "rust",
                       must_contain=["enum ", ("impl ", "impl<", "impl(")], min_lines=8),
        RealWorldQuery("Write a Rust function that reads lines from a file and returns a Vec<String>", "rust",
                       must_contain=["fn ", "Result"], min_lines=4),
        RealWorldQuery("Write a Rust trait for serialization with to_json and from_json methods", "rust",
                       must_contain=["trait ", "fn "], min_lines=5),
        RealWorldQuery("Write a Rust function that filters and maps a vector using iterator chains", "rust",
                       must_contain=["fn ", ".filter(", ".map("], min_lines=3),
        RealWorldQuery("Write a Rust struct that implements Iterator for a custom range with step", "rust",
                       must_contain=[("impl ", "impl<"), "Iterator", "next"], min_lines=6),
        RealWorldQuery("Write a Rust error enum with Display and From implementations for conversion", "rust",
                       must_contain=["enum ", ("impl ", "impl<"), "Display"], min_lines=8),
        RealWorldQuery("Write a Rust generic function that finds the minimum value in a slice", "rust",
                       must_contain=["fn ", "Option"], min_lines=3),
        RealWorldQuery("Write a Rust function that spawns threads to process items in parallel and collects results", "rust",
                       must_contain=["fn ", ("thread", "spawn", "rayon", "std::thread")], min_lines=4),
    ]

    # ── SQL (10) ──
    all_queries["sql"] = [
        RealWorldQuery("Write a SQL query that finds the second highest salary in each department", "sql",
                       must_contain=["SELECT", "FROM"], min_lines=3),
        RealWorldQuery("Write SQL to create a users table with email uniqueness and timestamps", "sql",
                       must_contain=["CREATE TABLE", "email"], min_lines=3),
        RealWorldQuery("Write a SQL query using a CTE to calculate running totals by month", "sql",
                       must_contain=["SELECT", "FROM"], min_lines=4),
        RealWorldQuery("Write a SQL query that finds customers who ordered every product in a category", "sql",
                       must_contain=["SELECT", "FROM", "GROUP BY"], min_lines=4),
        RealWorldQuery("Write SQL to create a view that shows order summaries with customer names", "sql",
                       must_contain=["CREATE", "VIEW", "JOIN"], min_lines=3),
        RealWorldQuery("Write a SQL query that detects gaps in a sequence of IDs", "sql",
                       must_contain=["SELECT", "FROM"], min_lines=3),
        RealWorldQuery("Write SQL to update prices with a 10% increase for products not sold in 30 days", "sql",
                       must_contain=["UPDATE", "SET"], min_lines=3),
        RealWorldQuery("Write a SQL query that pivots rows into columns showing monthly sales per product", "sql",
                       must_contain=["SELECT", "CASE"], min_lines=4),
        RealWorldQuery("Write SQL to delete duplicate rows keeping only the one with the lowest ID", "sql",
                       must_contain=["DELETE", "FROM"], min_lines=3),
        RealWorldQuery("Write a SQL query with a self-join to find employees and their managers", "sql",
                       must_contain=["SELECT", "JOIN"], min_lines=3),
    ]

    # ── C (10) ──
    all_queries["c"] = [
        RealWorldQuery("Write a C function that reverses a string in place using pointers", "c",
                       must_contain=["void ", "char"], min_lines=4),
        RealWorldQuery("Write a C function that implements binary search on a sorted array", "c",
                       must_contain=["int ", "return"], min_lines=5),
        RealWorldQuery("Write a C struct for a dynamic array with push and get functions", "c",
                       must_contain=["struct ", "malloc"], min_lines=8),
        RealWorldQuery("Write a C function that merges two sorted arrays into a new sorted array", "c",
                       must_contain=["int ", "malloc"], min_lines=6),
        RealWorldQuery("Write a C function that counts the occurrences of each character in a string", "c",
                       must_contain=["char", "int"], min_lines=4),
        RealWorldQuery("Write a C function that reads a file line by line using fgets", "c",
                       must_contain=["FILE", ("fgets", "getline", "fgetc")], min_lines=5),
        RealWorldQuery("Write a C implementation of a stack using a linked list", "c",
                       must_contain=["struct ", "push"], min_lines=8),
        RealWorldQuery("Write a C function that splits a string by a delimiter", "c",
                       must_contain=["char"], min_lines=5),
        RealWorldQuery("Write a C function that sorts an array using quicksort", "c",
                       must_contain=["void ", "int"], min_lines=8),
        RealWorldQuery("Write a C function that computes the factorial of a number recursively", "c",
                       must_contain=["int ", "return"], min_lines=3),
    ]

    # ── Java (10) ──
    all_queries["java"] = [
        RealWorldQuery("Write a Java method that checks if a string is a palindrome", "java",
                       must_contain=["public ", "return"], min_lines=3),
        RealWorldQuery("Write a Java class that implements a thread-safe singleton pattern", "java",
                       must_contain=["class ", "private ", "static"], min_lines=5),
        RealWorldQuery("Write a Java method using streams to find the most frequent word in a list", "java",
                       must_contain=["stream()", "return"], min_lines=3),
        RealWorldQuery("Write a Java generic class for a bounded buffer with put and take methods", "java",
                       must_contain=["class ", "synchronized"], min_lines=8),
        RealWorldQuery("Write a Java interface for a repository with CRUD methods and implement it", "java",
                       must_contain=["interface ", "class "], min_lines=8),
        RealWorldQuery("Write a Java method that reads a file and counts word frequencies using a HashMap", "java",
                       must_contain=["HashMap", "return"], min_lines=5),
        RealWorldQuery("Write a Java enum with fields, a constructor, and a lookup method", "java",
                       must_contain=["enum ", "return"], min_lines=5),
        RealWorldQuery("Write a Java CompletableFuture chain that fetches data, transforms it, and handles errors", "java",
                       must_contain=["CompletableFuture"], min_lines=4),
        RealWorldQuery("Write a Java method that flattens a nested List of Lists into a single list", "java",
                       must_contain=["List", "return"], min_lines=3),
        RealWorldQuery("Write a Java class that implements Comparable for sorting by multiple fields", "java",
                       must_contain=["class ", "compareTo"], min_lines=6),
    ]

    # ── C# (10) ──
    all_queries["csharp"] = [
        RealWorldQuery("Write a C# method that groups a list of objects by a property using LINQ", "csharp",
                       must_contain=["GroupBy", "return"], min_lines=3),
        RealWorldQuery("Write a C# async method that downloads multiple files in parallel", "csharp",
                       must_contain=["async ", "Task"], min_lines=5),
        RealWorldQuery("Write a C# generic class that implements a stack with push, pop, and peek", "csharp",
                       must_contain=["class ", "Push"], min_lines=6),
        RealWorldQuery("Write a C# record with computed properties and custom equality", "csharp",
                       must_contain=["record "], min_lines=4),
        RealWorldQuery("Write a C# extension method that converts a string to title case", "csharp",
                       must_contain=["static ", "this string"], min_lines=3),
        RealWorldQuery("Write a C# interface for a repository pattern with generic CRUD methods", "csharp",
                       must_contain=["interface ", ("Task", "IEnumerable", "void", "IQueryable")], min_lines=4),
        RealWorldQuery("Write a C# method that parses a CSV string into a list of dictionaries", "csharp",
                       must_contain=["List", "Dictionary"], min_lines=5),
        RealWorldQuery("Write a C# class that implements IDisposable for managing unmanaged resources", "csharp",
                       must_contain=["class ", "Dispose"], min_lines=6),
        RealWorldQuery("Write a C# LINQ query that joins two lists and selects specific fields", "csharp",
                       must_contain=["join ", "select "], min_lines=3),
        RealWorldQuery("Write a C# method that validates an object using custom validation attributes", "csharp",
                       must_contain=["class ", "Validate"], min_lines=5),
    ]

    # ── Ruby (10) ──
    all_queries["ruby"] = [
        RealWorldQuery("Write a Ruby method that finds all permutations of an array", "ruby",
                       must_contain=["def ", "end"], min_lines=3),
        RealWorldQuery("Write a Ruby class with comparable, initialize, and sorting support", "ruby",
                       must_contain=["class ", "def ", "end"], min_lines=6),
        RealWorldQuery("Write a Ruby method that scrapes a webpage title using net/http", "ruby",
                       must_contain=["require ", "def ", "end"], min_lines=4),
        RealWorldQuery("Write a Ruby module with class methods and instance methods mixed in", "ruby",
                       must_contain=["module ", "def ", "end"], min_lines=6),
        RealWorldQuery("Write a Ruby method that reads a CSV file and returns an array of hashes", "ruby",
                       must_contain=["require ", "CSV", "end"], min_lines=3),
        RealWorldQuery("Write a Ruby class that implements each and includes Enumerable", "ruby",
                       must_contain=["class ", "Enumerable", "each", "end"], min_lines=6),
        RealWorldQuery("Write a Ruby method that converts a hash to query string parameters", "ruby",
                       must_contain=["def ", "end"], min_lines=3),
        RealWorldQuery("Write a Ruby method that retries a block up to N times on exception", "ruby",
                       must_contain=["def ", "rescue", "end"], min_lines=5),
        RealWorldQuery("Write a Ruby class for a simple HTTP API client with get and post methods", "ruby",
                       must_contain=["class ", "def ", "end"], min_lines=8),
        RealWorldQuery("Write a Ruby method that deep merges two nested hashes", "ruby",
                       must_contain=["def ", "merge", "end"], min_lines=4),
    ]

    # ── Bash (10) ──
    all_queries["bash"] = [
        RealWorldQuery("Write a bash script that finds and deletes all files older than 30 days", "bash",
                       must_contain=["find ", "-mtime"], min_lines=2),
        RealWorldQuery("Write a bash function that checks if a port is open on a host", "bash",
                       must_contain=["()"], min_lines=3),
        RealWorldQuery("Write a bash script that monitors disk usage and sends an alert above 90%", "bash",
                       must_contain=["df ", "if"], min_lines=4),
        RealWorldQuery("Write a bash script that renames all .txt files in a directory to .md", "bash",
                       must_contain=["for ", "mv "], min_lines=3),
        RealWorldQuery("Write a bash function that extracts a specific field from a CSV file", "bash",
                       must_contain=["()"], min_lines=3),
        RealWorldQuery("Write a bash script that creates a compressed archive of changed git files", "bash",
                       must_contain=["git ", "tar "], min_lines=3),
        RealWorldQuery("Write a bash script with proper argument parsing using getopts", "bash",
                       must_contain=["getopts", "case"], min_lines=5),
        RealWorldQuery("Write a bash function that validates an IP address format", "bash",
                       must_contain=["()"], min_lines=3),
        RealWorldQuery("Write a bash script that watches a directory for new files and processes them", "bash",
                       must_contain=[("while", "inotifywait", "fswatch", "for "),
                                     ("do", "done", "process", "handle")],
                       min_lines=2),
        RealWorldQuery("Write a bash script that generates a report of the top 10 largest files in a directory", "bash",
                       must_contain=["sort", "head"], min_lines=2),
    ]

    # ── Kotlin (10) ──
    all_queries["kotlin"] = [
        RealWorldQuery("Write a Kotlin data class with validation in the init block", "kotlin",
                       must_contain=["data class", "init"], min_lines=4),
        RealWorldQuery("Write a Kotlin sealed class hierarchy for API response states", "kotlin",
                       must_contain=["sealed ", "class "], min_lines=5),
        RealWorldQuery("Write a Kotlin extension function on List that returns the second largest element", "kotlin",
                       must_contain=["fun ", "return"], min_lines=3),
        RealWorldQuery("Write a Kotlin coroutine function that fetches data from two APIs in parallel", "kotlin",
                       must_contain=["suspend ", "async"], min_lines=4),
        RealWorldQuery("Write a Kotlin function that groups a list of items by a selector and counts them", "kotlin",
                       must_contain=["fun ", "groupBy"], min_lines=3),
        RealWorldQuery("Write a Kotlin inline function with reified type parameter for JSON parsing", "kotlin",
                       must_contain=["inline ", "reified"], min_lines=3),
        RealWorldQuery("Write a Kotlin class that implements a simple LRU cache using LinkedHashMap", "kotlin",
                       must_contain=["class ", "LinkedHashMap"], min_lines=5),
        RealWorldQuery("Write a Kotlin function that reads a file and returns its lines, handling errors", "kotlin",
                       must_contain=["fun ", "return"], min_lines=3),
        RealWorldQuery("Write a Kotlin object that implements a thread-safe singleton with lazy initialization", "kotlin",
                       must_contain=["object ", "lazy"], min_lines=3),
        RealWorldQuery("Write a Kotlin flow that emits items with a delay between each", "kotlin",
                       must_contain=["fun ", "flow"], min_lines=3),
    ]

    # ── Swift (10) ──
    all_queries["swift"] = [
        RealWorldQuery("Write a Swift struct with Codable conformance for JSON serialization", "swift",
                       must_contain=["struct ", "Codable"], min_lines=3),
        RealWorldQuery("Write a Swift generic function that finds the first element matching a predicate", "swift",
                       must_contain=["func ", "return"], min_lines=3),
        RealWorldQuery("Write a Swift enum with associated values and a computed property", "swift",
                       must_contain=["enum ", "case "], min_lines=5),
        RealWorldQuery("Write a Swift protocol for a data store with async methods", "swift",
                       must_contain=["protocol ", "func "], min_lines=4),
        RealWorldQuery("Write a Swift class that implements a simple observer pattern with closures", "swift",
                       must_contain=["class ", "func "], min_lines=6),
        RealWorldQuery("Write a Swift extension on Array that adds a method to chunk into groups of N", "swift",
                       must_contain=["extension ", "func "], min_lines=4),
        RealWorldQuery("Write a Swift async function that downloads data from a URL with error handling", "swift",
                       must_contain=["func ", "async", "throws"], min_lines=4),
        RealWorldQuery("Write a Swift struct that conforms to Comparable for custom sorting", "swift",
                       must_contain=["struct ", "Comparable"], min_lines=4),
        RealWorldQuery("Write a Swift property wrapper that clamps a value between a min and max", "swift",
                       must_contain=["struct ", "wrappedValue"], min_lines=5),
        RealWorldQuery("Write a Swift generic stack using an array with push, pop, and peek", "swift",
                       must_contain=["struct ", "mutating"], min_lines=6),
    ]

    # Filter by requested languages
    if languages:
        filtered = []
        for lang in languages:
            lang_lower = lang.lower()
            if lang_lower in all_queries:
                filtered.extend(all_queries[lang_lower])
            else:
                print(f"  Warning: unknown language '{lang}', skipping")
        return filtered

    # Return all
    result = []
    for lang in ["python", "javascript", "typescript", "go", "rust", "sql",
                 "c", "java", "csharp", "ruby", "bash", "kotlin", "swift"]:
        if lang in all_queries:
            result.extend(all_queries[lang])
    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-Language Benchmark")
    parser.add_argument("--model", type=str,
                        default="models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
                        help="Model path")
    parser.add_argument("--lang", nargs="+", help="Languages to test (default: all)")
    parser.add_argument("--check-routing", action="store_true",
                        help="Check routing only, no model needed")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    queries = build_multilang_queries(args.lang)
    print(f"\n  Multi-Language Benchmark: {len(queries)} queries")
    if args.lang:
        print(f"  Languages: {', '.join(args.lang)}")
    else:
        print(f"  Languages: all 12")

    if args.check_routing:
        from benchmark_realworld import check_routing_only
        check_routing_only(queries)
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n  Model not found: {model_path}")
        sys.exit(1)

    results = run_benchmark(
        model_path, queries,
        gpu_layers=args.gpu_layers,
        threads=args.threads,
    )

    # Save results
    out_file = f"bench_multilang_{model_path.stem}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    # Per-language breakdown
    print(f"\n  === Per-Language Results ===")
    lang_stats = {}
    for r in results["results"]:
        domain = r["domain"]
        if domain not in lang_stats:
            lang_stats[domain] = {"pass": 0, "fail": 0}
        if r["passed"]:
            lang_stats[domain]["pass"] += 1
        else:
            lang_stats[domain]["fail"] += 1

    for lang in sorted(lang_stats):
        s = lang_stats[lang]
        total = s["pass"] + s["fail"]
        pct = s["pass"] / total * 100
        bar = "#" * s["pass"] + "." * s["fail"]
        print(f"    {lang:12s} {s['pass']:2d}/{total:2d} ({pct:5.1f}%)  [{bar}]")

    total_pass = sum(s["pass"] for s in lang_stats.values())
    total = sum(s["pass"] + s["fail"] for s in lang_stats.values())
    print(f"\n    {'TOTAL':12s} {total_pass}/{total} ({total_pass / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
