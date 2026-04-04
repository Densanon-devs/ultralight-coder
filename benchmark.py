#!/usr/bin/env python3
"""
Ultralite Code Assistant — Comprehensive Model Benchmark

Tests GGUF models on code-specific tasks across 4 categories:
  1. Code Generation (10 tests)
  2. Debugging (8 tests)
  3. Code Review (6 tests)
  4. Explanation (6 tests)

Each model is tested WITH and WITHOUT the augmentor system to measure
augmentor impact. Results are saved as JSON and a summary text file.

Usage:
    python benchmark.py                      # Run all models
    python benchmark.py --model <path>       # Run a specific model
    python benchmark.py --category code_gen  # Run one category only
    python benchmark.py --quick              # 1 test per category (smoke test)
    python benchmark.py --no-augmentors       # Skip augmentor-enabled runs
    python benchmark.py --max-tokens 256     # Override generation length
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Chat-format detection from filename
# ---------------------------------------------------------------------------

CHAT_FORMAT_RULES = [
    # (filename substring -> chat_format)
    ("qwen",      "chatml"),
    ("smollm",    "chatml"),
    ("llama-3",   "llama3"),
    ("Llama-3",   "llama3"),
    ("phi-3",     "phi3"),
    ("Phi-3",     "phi3"),
    ("phi3",      "phi3"),
    ("tinyllama",  "alpaca"),
    ("TinyLlama",  "alpaca"),
]


def detect_chat_format(model_path: str) -> str:
    """Detect the correct chat_format from a model filename."""
    name = Path(model_path).name.lower()
    for substring, fmt in CHAT_FORMAT_RULES:
        if substring.lower() in name:
            return fmt
    # Fallback: chatml is the safest default for modern models
    return "chatml"


# ---------------------------------------------------------------------------
# Prompt builder (mirrors augmentor system's _wrap_augmentor_chat)
# ---------------------------------------------------------------------------

def wrap_chat(system: str, user: str, chat_format: str) -> str:
    """Wrap system+user into a chat-formatted prompt."""
    if chat_format == "chatml":
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif chat_format == "phi3":
        return (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user}<|end|>\n"
            f"<|assistant|>\n"
        )
    elif chat_format == "llama3":
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif chat_format == "alpaca":
        return (
            f"### System:\n{system}\n\n"
            f"### Instruction:\n{user}\n\n"
            f"### Response:\n"
        )
    else:
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Result from a single test."""
    test_id: str
    category: str
    prompt: str
    response: str
    passed: bool
    failure_reason: str = ""
    response_time: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    augmentor_used: bool = False
    augmentor_name: str = ""
    augmentor_attempts: int = 0


@dataclass
class ModelResult:
    """Aggregated results for one model under one mode (augmentor/no-augmentor)."""
    model_name: str
    model_path: str
    model_size_mb: float
    chat_format: str
    augmentors_enabled: bool
    total_tests: int = 0
    total_passed: int = 0
    quality_score: float = 0.0
    category_scores: dict = field(default_factory=dict)
    avg_tokens_per_second: float = 0.0
    avg_response_time: float = 0.0
    total_time: float = 0.0
    tests: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkTest:
    """A single benchmark test case."""
    test_id: str
    category: str           # code_gen, debug, review, explain
    augmentor_hint: str     # module_hint for AugmentorRouter
    prompt: str
    verify: "callable"      # fn(response, prompt) -> (bool, str)


# ── Verification helpers ──────────────────────────────────────

def _has_any(text: str, terms: list[str], case_sensitive: bool = False) -> bool:
    """Check if text contains any of the given terms."""
    t = text if case_sensitive else text.lower()
    for term in terms:
        check = term if case_sensitive else term.lower()
        if check in t:
            return True
    return False


def _has_all(text: str, terms: list[str], case_sensitive: bool = False) -> bool:
    """Check if text contains all of the given terms."""
    t = text if case_sensitive else text.lower()
    for term in terms:
        check = term if case_sensitive else term.lower()
        if check not in t:
            return False
    return True


def _has_function_def(text: str) -> bool:
    """Check for a function or class definition."""
    return bool(re.search(r'\bdef\s+\w+|class\s+\w+|function\s+\w+|const\s+\w+\s*=', text))


def _has_code_block(text: str) -> bool:
    """Check for a markdown code block or bare function definition."""
    return "```" in text or _has_function_def(text)


def _has_return(text: str) -> bool:
    """Check for a return statement or output mechanism."""
    return _has_any(text, ["return ", "return\n", "print(", "yield ", "console.log"])


def _min_length(text: str, n: int) -> bool:
    """Check minimum response length."""
    return len(text.strip()) >= n


# ── Code Generation Verifiers ────────────────────────────────

def verify_fibonacci(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block or function definition")
    if not _has_any(resp, ["def ", "function "]):
        checks.append("No function definition found")
    if not _has_return(resp):
        checks.append("No return statement")
    if not _has_any(resp, ["fib", "fibonacci"]):
        checks.append("Function name should reference fibonacci")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_merge_sorted(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block")
    if not _has_any(resp, ["def ", "function "]):
        checks.append("No function definition")
    if not _has_return(resp):
        checks.append("No return statement")
    # Should handle two lists/arrays
    if not _has_any(resp, ["while", "for", "if"]):
        checks.append("Expected iteration or comparison logic")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_is_prime(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block")
    if not _has_any(resp, ["def ", "function "]):
        checks.append("No function definition")
    if not _has_return(resp):
        checks.append("No return statement")
    if not _has_any(resp, ["prime", "bool", "true", "false"]):
        checks.append("Should reference primality or return bool")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_flatten(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block")
    if not _has_any(resp, ["def ", "function "]):
        checks.append("No function definition")
    if not _has_return(resp):
        checks.append("No return statement")
    # Flatten typically needs recursion or isinstance check
    if not _has_any(resp, ["isinstance", "list", "iter", "recursive", "flatten", "extend", "append"]):
        checks.append("Should handle nested structure (isinstance/recursion)")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_longest_common_prefix(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block")
    if not _has_any(resp, ["def ", "function "]):
        checks.append("No function definition")
    if not _has_return(resp):
        checks.append("No return statement")
    if not _has_any(resp, ["prefix", "common", "str"]):
        checks.append("Should reference prefix/common/string operations")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_lru_cache(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block")
    if not _has_any(resp, ["def ", "class ", "function "]):
        checks.append("No function/class definition")
    if not _has_any(resp, ["get", "put", "set", "cache", "dict", "ordereddict", "capacity"]):
        checks.append("Should implement get/put with cache eviction logic")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_roman_to_int(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block")
    if not _has_any(resp, ["def ", "function "]):
        checks.append("No function definition")
    if not _has_return(resp):
        checks.append("No return statement")
    # Should have a mapping of roman numerals
    if not _has_any(resp, ["i", "v", "x", "l", "c", "d", "m", "dict", "map", "1000", "500", "100"]):
        checks.append("Should contain roman numeral value mapping")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_permutations(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block")
    if not _has_any(resp, ["def ", "function "]):
        checks.append("No function definition")
    if not _has_return(resp):
        checks.append("No return statement")
    if not _has_any(resp, ["permut", "recursive", "backtrack", "swap", "itertools"]):
        checks.append("Should use recursion/backtracking or itertools")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_email_regex(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block")
    if not _has_any(resp, ["def ", "function "]):
        checks.append("No function definition")
    if not _has_any(resp, ["re.", "regex", "re.match", "re.search", "re.fullmatch", "pattern", r"\@", "@"]):
        checks.append("Should use regex with @ pattern")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_matrix_multiply(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _has_code_block(resp):
        checks.append("Missing code block")
    if not _has_any(resp, ["def ", "function "]):
        checks.append("No function definition")
    if not _has_return(resp):
        checks.append("No return statement")
    if not _has_any(resp, ["for", "range", "zip", "numpy", "np.", "dot", "matmul", "sum"]):
        checks.append("Should contain loop or matrix operation logic")
    if checks:
        return False, "; ".join(checks)
    return True, ""


# ── Debugging Verifiers ──────────────────────────────────────

def _verify_debug_generic(resp: str, required_terms: list[str], fix_description: str) -> tuple[bool, str]:
    """Generic debug verifier: checks for diagnosis + fix."""
    checks = []
    r = resp.lower()

    has_diagnosis = _has_any(r, [
        "because", "cause", "reason", "the issue", "the problem", "the bug",
        "the error", "happens when", "occurs when", "due to", "off-by-one",
        "out of bounds", "index", "mutable", "default", "scope", "infinite",
        "keyerror", "typeerror", "iteration", "recursion", "base case",
    ])
    if not has_diagnosis:
        checks.append("Should identify root cause")

    has_fix = _has_code_block(resp) or _has_any(r, ["fix", "change", "replace", "should be", "instead"])
    if not has_fix:
        checks.append("Should show a fix")

    # Check for the specific required terms
    if required_terms and not _has_any(r, required_terms):
        checks.append(f"Should mention: {fix_description}")

    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_debug_off_by_one(resp: str, _prompt: str) -> tuple[bool, str]:
    return _verify_debug_generic(resp,
        ["off-by-one", "off by one", "index", "range", "len", "- 1", "-1", "bound"],
        "off-by-one or index boundary issue")


def verify_debug_mutable_default(resp: str, _prompt: str) -> tuple[bool, str]:
    return _verify_debug_generic(resp,
        ["mutable", "default", "none", "shared", "[]"],
        "mutable default argument")


def verify_debug_scope(resp: str, _prompt: str) -> tuple[bool, str]:
    return _verify_debug_generic(resp,
        ["scope", "unbound", "local", "global", "variable", "assign", "before"],
        "variable scope / UnboundLocalError")


def verify_debug_infinite_loop(resp: str, _prompt: str) -> tuple[bool, str]:
    return _verify_debug_generic(resp,
        ["infinite", "never", "condition", "update", "increment", "decrement", "loop", "while", "break"],
        "infinite loop cause")


def verify_debug_keyerror(resp: str, _prompt: str) -> tuple[bool, str]:
    return _verify_debug_generic(resp,
        ["key", "keyerror", "not in", "get(", "missing", "exist", "default"],
        "KeyError / missing key")


def verify_debug_typeerror(resp: str, _prompt: str) -> tuple[bool, str]:
    return _verify_debug_generic(resp,
        ["type", "str", "int", "convert", "cast", "f-string", "format", "typeerror"],
        "type mismatch")


def verify_debug_list_modification(resp: str, _prompt: str) -> tuple[bool, str]:
    return _verify_debug_generic(resp,
        ["modify", "iteration", "iterating", "copy", "list", "remove", "changing", "mutate", "new list", "comprehension"],
        "modifying list during iteration")


def verify_debug_recursion_base(resp: str, _prompt: str) -> tuple[bool, str]:
    return _verify_debug_generic(resp,
        ["base case", "base_case", "recursion", "recursive", "return", "stop", "condition", "termina", "== 0", "== 1", "<= 0", "<= 1"],
        "incorrect/missing recursion base case")


# ── Code Review Verifiers ────────────────────────────────────

def verify_review_sql_injection(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    r = resp.lower()
    if not _has_any(r, ["sql injection", "injection", "parameterize", "placeholder", "sanitize", "escape", "?", "%s", "execute("]):
        checks.append("Should identify SQL injection vulnerability")
    if not _has_any(r, ["fix", "safe", "secure", "parameterize", "placeholder", "```"]):
        checks.append("Should suggest parameterized queries")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_review_resource_leak(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    r = resp.lower()
    if not _has_any(r, ["close", "leak", "with", "context manager", "resource", "open", "handle"]):
        checks.append("Should identify unclosed resource / file handle leak")
    if not _has_any(r, ["with ", "close()", "context", "finally", "```"]):
        checks.append("Should suggest using 'with' statement or explicit close")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_review_performance(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    r = resp.lower()
    if not _has_any(r, ["o(n", "complexity", "slow", "performance", "efficient", "optimize", "quadratic", "cubic", "nested loop"]):
        checks.append("Should identify performance / complexity issue")
    if not _has_any(r, ["improve", "better", "faster", "fix", "optimize", "set", "hash", "dict", "```"]):
        checks.append("Should suggest a more efficient approach")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_review_race_condition(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    r = resp.lower()
    if not _has_any(r, ["race", "thread", "concurrent", "shared", "mutable", "lock", "mutex", "synchroni", "atomic", "safe"]):
        checks.append("Should identify race condition / thread safety issue")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_review_hardcoded_creds(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    r = resp.lower()
    if not _has_any(r, ["hardcoded", "hard-coded", "credential", "password", "secret", "plain", "sensitive", "security", "env", "environment", "config"]):
        checks.append("Should identify hardcoded credentials as a security risk")
    if not _has_any(r, ["environment", "env", "config", "vault", "secret", "variable", "```"]):
        checks.append("Should suggest externalized configuration")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_review_input_validation(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    r = resp.lower()
    if not _has_any(r, ["valid", "check", "sanitize", "input", "missing", "none", "null", "empty", "type", "boundary", "negative"]):
        checks.append("Should identify missing input validation")
    if checks:
        return False, "; ".join(checks)
    return True, ""


# ── Explanation Verifiers ────────────────────────────────────

def verify_explain_decorator(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _min_length(resp, 80):
        checks.append("Explanation too short (< 80 chars)")
    if not _has_any(resp, ["@", "wrapper", "wraps", "function", "def "]):
        checks.append("Should mention @ syntax or wrapper pattern")
    if not _has_code_block(resp):
        checks.append("Should include a code example")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_explain_deepcopy(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _min_length(resp, 80):
        checks.append("Explanation too short")
    if not _has_any(resp, ["shallow", "deep", "copy", "reference", "nested", "independent"]):
        checks.append("Should explain difference between shallow/deep copy")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_explain_yield(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _min_length(resp, 80):
        checks.append("Explanation too short")
    if not _has_any(resp, ["generator", "yield", "iterator", "lazy", "next", "iteration", "pause"]):
        checks.append("Should mention generator/iterator concept")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_explain_async(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _min_length(resp, 80):
        checks.append("Explanation too short")
    if not _has_any(resp, ["async", "await", "coroutine", "asynchronous", "event loop", "concurrent", "non-blocking"]):
        checks.append("Should explain async/await mechanism")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_explain_closure(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _min_length(resp, 80):
        checks.append("Explanation too short")
    if not _has_any(resp, ["closure", "enclosed", "outer", "inner", "variable", "scope", "capture", "function"]):
        checks.append("Should explain closure concept (inner function capturing outer scope)")
    if checks:
        return False, "; ".join(checks)
    return True, ""


def verify_explain_gil(resp: str, _prompt: str) -> tuple[bool, str]:
    checks = []
    if not _min_length(resp, 80):
        checks.append("Explanation too short")
    if not _has_any(resp, ["global interpreter lock", "gil", "thread", "cpython", "lock", "one thread", "single"]):
        checks.append("Should explain GIL (Global Interpreter Lock)")
    if checks:
        return False, "; ".join(checks)
    return True, ""


# ---------------------------------------------------------------------------
# Test suite definition
# ---------------------------------------------------------------------------

def build_test_suite() -> list[BenchmarkTest]:
    """Build the complete 30-test benchmark suite."""
    tests = []

    # ── Code Generation (10) ─────────────────────────────────

    tests.append(BenchmarkTest(
        test_id="cg01_fibonacci",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a Python function to compute the nth Fibonacci number. Handle edge cases like n=0 and n=1.",
        verify=verify_fibonacci,
    ))

    tests.append(BenchmarkTest(
        test_id="cg02_merge_sorted",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a Python function that merges two sorted lists into a single sorted list without using built-in sort.",
        verify=verify_merge_sorted,
    ))

    tests.append(BenchmarkTest(
        test_id="cg03_is_prime",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a Python function to check if a number is prime. Return True if prime, False otherwise. Handle edge cases (0, 1, negative numbers).",
        verify=verify_is_prime,
    ))

    tests.append(BenchmarkTest(
        test_id="cg04_flatten",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a Python function to flatten a nested list of arbitrary depth. Example: [1, [2, [3, 4]], 5] -> [1, 2, 3, 4, 5]",
        verify=verify_flatten,
    ))

    tests.append(BenchmarkTest(
        test_id="cg05_longest_prefix",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a Python function to find the longest common prefix string among a list of strings. Return empty string if no common prefix.",
        verify=verify_longest_common_prefix,
    ))

    tests.append(BenchmarkTest(
        test_id="cg06_lru_cache",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a simple LRU cache class in Python with get(key) and put(key, value) methods. It should have a fixed capacity and evict the least recently used item when full.",
        verify=verify_lru_cache,
    ))

    tests.append(BenchmarkTest(
        test_id="cg07_roman_to_int",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a Python function to convert a Roman numeral string (like 'XIV') to an integer. Handle subtractive cases like IV=4, IX=9.",
        verify=verify_roman_to_int,
    ))

    tests.append(BenchmarkTest(
        test_id="cg08_permutations",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a Python function to find all permutations of a string. Return a list of strings.",
        verify=verify_permutations,
    ))

    tests.append(BenchmarkTest(
        test_id="cg09_email_regex",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a Python function to validate an email address using regex. Return True if the email is valid.",
        verify=verify_email_regex,
    ))

    tests.append(BenchmarkTest(
        test_id="cg10_matrix_multiply",
        category="code_gen",
        augmentor_hint="code_gen",
        prompt="Write a Python function to multiply two matrices (2D lists). Handle dimension mismatch with a ValueError.",
        verify=verify_matrix_multiply,
    ))

    # ── Debugging (8) ────────────────────────────────────────

    tests.append(BenchmarkTest(
        test_id="db01_off_by_one",
        category="debug",
        augmentor_hint="debugger",
        prompt=(
            "Fix this code. It throws IndexError:\n\n"
            "```python\n"
            "def get_pairs(lst):\n"
            "    pairs = []\n"
            "    for i in range(len(lst)):\n"
            "        pairs.append((lst[i], lst[i+1]))\n"
            "    return pairs\n"
            "```\n\n"
            "What's the bug and how do you fix it?"
        ),
        verify=verify_debug_off_by_one,
    ))

    tests.append(BenchmarkTest(
        test_id="db02_mutable_default",
        category="debug",
        augmentor_hint="debugger",
        prompt=(
            "This function behaves strangely — calling it multiple times accumulates results:\n\n"
            "```python\n"
            "def add_item(item, items=[]):\n"
            "    items.append(item)\n"
            "    return items\n"
            "\n"
            "print(add_item('a'))  # ['a']\n"
            "print(add_item('b'))  # ['a', 'b']  ← unexpected!\n"
            "```\n\n"
            "What's the bug and how do you fix it?"
        ),
        verify=verify_debug_mutable_default,
    ))

    tests.append(BenchmarkTest(
        test_id="db03_scope",
        category="debug",
        augmentor_hint="debugger",
        prompt=(
            "This code raises UnboundLocalError:\n\n"
            "```python\n"
            "counter = 0\n"
            "\n"
            "def increment():\n"
            "    counter = counter + 1\n"
            "    return counter\n"
            "```\n\n"
            "What's the bug and how do you fix it?"
        ),
        verify=verify_debug_scope,
    ))

    tests.append(BenchmarkTest(
        test_id="db04_infinite_loop",
        category="debug",
        augmentor_hint="debugger",
        prompt=(
            "This code hangs and never terminates:\n\n"
            "```python\n"
            "def countdown(n):\n"
            "    while n > 0:\n"
            "        print(n)\n"
            "    return 'done'\n"
            "```\n\n"
            "What's the bug and how do you fix it?"
        ),
        verify=verify_debug_infinite_loop,
    ))

    tests.append(BenchmarkTest(
        test_id="db05_keyerror",
        category="debug",
        augmentor_hint="debugger",
        prompt=(
            "This code crashes with KeyError:\n\n"
            "```python\n"
            "def count_words(text):\n"
            "    counts = {}\n"
            "    for word in text.split():\n"
            "        counts[word] += 1\n"
            "    return counts\n"
            "```\n\n"
            "What's the bug and how do you fix it?"
        ),
        verify=verify_debug_keyerror,
    ))

    tests.append(BenchmarkTest(
        test_id="db06_typeerror",
        category="debug",
        augmentor_hint="debugger",
        prompt=(
            "This code throws TypeError:\n\n"
            "```python\n"
            "def greet(name, age):\n"
            "    return 'Hello ' + name + ', you are ' + age + ' years old'\n"
            "\n"
            "print(greet('Alice', 30))\n"
            "```\n\n"
            "What's the bug and how do you fix it?"
        ),
        verify=verify_debug_typeerror,
    ))

    tests.append(BenchmarkTest(
        test_id="db07_list_modification",
        category="debug",
        augmentor_hint="debugger",
        prompt=(
            "This code skips some elements and gives wrong results:\n\n"
            "```python\n"
            "def remove_evens(numbers):\n"
            "    for num in numbers:\n"
            "        if num % 2 == 0:\n"
            "            numbers.remove(num)\n"
            "    return numbers\n"
            "\n"
            "print(remove_evens([1, 2, 3, 4, 5, 6]))  # Expected [1, 3, 5]\n"
            "```\n\n"
            "What's the bug and how do you fix it?"
        ),
        verify=verify_debug_list_modification,
    ))

    tests.append(BenchmarkTest(
        test_id="db08_recursion_base",
        category="debug",
        augmentor_hint="debugger",
        prompt=(
            "This recursive function causes maximum recursion depth exceeded:\n\n"
            "```python\n"
            "def factorial(n):\n"
            "    return n * factorial(n - 1)\n"
            "```\n\n"
            "What's the bug and how do you fix it?"
        ),
        verify=verify_debug_recursion_base,
    ))

    # ── Code Review (6) ──────────────────────────────────────

    tests.append(BenchmarkTest(
        test_id="cr01_sql_injection",
        category="review",
        augmentor_hint="code_review",
        prompt=(
            "Review this code for security issues:\n\n"
            "```python\n"
            "import sqlite3\n"
            "\n"
            "def get_user(username):\n"
            "    conn = sqlite3.connect('users.db')\n"
            "    cursor = conn.cursor()\n"
            "    query = f\"SELECT * FROM users WHERE name = '{username}'\"\n"
            "    cursor.execute(query)\n"
            "    return cursor.fetchone()\n"
            "```"
        ),
        verify=verify_review_sql_injection,
    ))

    tests.append(BenchmarkTest(
        test_id="cr02_resource_leak",
        category="review",
        augmentor_hint="code_review",
        prompt=(
            "Review this code:\n\n"
            "```python\n"
            "def process_files(paths):\n"
            "    results = []\n"
            "    for path in paths:\n"
            "        f = open(path, 'r')\n"
            "        data = f.read()\n"
            "        results.append(len(data))\n"
            "    return results\n"
            "```"
        ),
        verify=verify_review_resource_leak,
    ))

    tests.append(BenchmarkTest(
        test_id="cr03_performance",
        category="review",
        augmentor_hint="code_review",
        prompt=(
            "Review this code for performance:\n\n"
            "```python\n"
            "def has_duplicates(lst):\n"
            "    for i in range(len(lst)):\n"
            "        for j in range(i + 1, len(lst)):\n"
            "            if lst[i] == lst[j]:\n"
            "                return True\n"
            "    return False\n"
            "```"
        ),
        verify=verify_review_performance,
    ))

    tests.append(BenchmarkTest(
        test_id="cr04_race_condition",
        category="review",
        augmentor_hint="code_review",
        prompt=(
            "Review this code used in a multi-threaded web server:\n\n"
            "```python\n"
            "user_count = 0\n"
            "\n"
            "def handle_request(request):\n"
            "    global user_count\n"
            "    user_count += 1\n"
            "    request_id = user_count\n"
            "    # process request...\n"
            "    return f'Request {request_id} processed'\n"
            "```"
        ),
        verify=verify_review_race_condition,
    ))

    tests.append(BenchmarkTest(
        test_id="cr05_hardcoded_creds",
        category="review",
        augmentor_hint="code_review",
        prompt=(
            "Review this code:\n\n"
            "```python\n"
            "import smtplib\n"
            "\n"
            "def send_alert(message):\n"
            "    server = smtplib.SMTP('smtp.gmail.com', 587)\n"
            "    server.starttls()\n"
            "    server.login('admin@company.com', 'SuperSecret123!')\n"
            "    server.sendmail('admin@company.com', 'team@company.com', message)\n"
            "    server.quit()\n"
            "```"
        ),
        verify=verify_review_hardcoded_creds,
    ))

    tests.append(BenchmarkTest(
        test_id="cr06_input_validation",
        category="review",
        augmentor_hint="code_review",
        prompt=(
            "Review this code:\n\n"
            "```python\n"
            "def calculate_average(numbers):\n"
            "    total = sum(numbers)\n"
            "    return total / len(numbers)\n"
            "\n"
            "def get_element(lst, index):\n"
            "    return lst[index]\n"
            "```"
        ),
        verify=verify_review_input_validation,
    ))

    # ── Explanation (6) ──────────────────────────────────────

    tests.append(BenchmarkTest(
        test_id="ex01_decorator",
        category="explain",
        augmentor_hint="explainer",
        prompt="Explain what a decorator does in Python. Include a simple example.",
        verify=verify_explain_decorator,
    ))

    tests.append(BenchmarkTest(
        test_id="ex02_deepcopy",
        category="explain",
        augmentor_hint="explainer",
        prompt="Explain the difference between a shallow copy and a deep copy in Python. When would you use each?",
        verify=verify_explain_deepcopy,
    ))

    tests.append(BenchmarkTest(
        test_id="ex03_yield",
        category="explain",
        augmentor_hint="explainer",
        prompt="Explain what `yield` does in Python and how generators work. Show a simple example.",
        verify=verify_explain_yield,
    ))

    tests.append(BenchmarkTest(
        test_id="ex04_async",
        category="explain",
        augmentor_hint="explainer",
        prompt="Explain async/await in Python. When should you use it and what problem does it solve?",
        verify=verify_explain_async,
    ))

    tests.append(BenchmarkTest(
        test_id="ex05_closure",
        category="explain",
        augmentor_hint="explainer",
        prompt="Explain what a closure is in Python. Give a concrete example.",
        verify=verify_explain_closure,
    ))

    tests.append(BenchmarkTest(
        test_id="ex06_gil",
        category="explain",
        augmentor_hint="explainer",
        prompt="Explain the GIL (Global Interpreter Lock) in Python. Why does it exist and how does it affect multithreading?",
        verify=verify_explain_gil,
    ))

    return tests


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models() -> list[Path]:
    """Find all GGUF model files in both model directories."""
    search_dirs = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT.parent / "plug-in-intelligence-engine" / "models",
    ]

    models = []
    for d in search_dirs:
        if d.exists():
            for f in sorted(d.glob("*.gguf")):
                if f.is_file() and f.stat().st_size > 1_000_000:  # Skip tiny/corrupt files
                    models.append(f)

    # Deduplicate by filename (in case same model in both dirs)
    seen = set()
    unique = []
    for m in models:
        if m.name not in seen:
            seen.add(m.name)
            unique.append(m)

    return unique


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs the full benchmark suite across models."""

    def __init__(
        self,
        tests: list[BenchmarkTest],
        max_tokens: int = 512,
        temperature: float = 0.2,
        gpu_layers: int = 99,
        threads: int = 8,
        context_length: int = 4096,
        batch_size: int = 512,
    ):
        self.tests = tests
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.context_length = context_length
        self.batch_size = batch_size
        self.results: list[ModelResult] = []

    def load_model(self, model_path: Path, chat_format: str):
        """Load a GGUF model via llama-cpp-python directly."""
        from llama_cpp import Llama

        logger.info(f"Loading model: {model_path.name}")
        model = Llama(
            model_path=str(model_path),
            n_ctx=self.context_length,
            n_gpu_layers=self.gpu_layers,
            n_threads=self.threads,
            n_batch=self.batch_size,
            verbose=False,
        )
        return model

    def generate(self, model, prompt: str) -> tuple[str, float, int]:
        """Generate a response. Returns (text, elapsed_seconds, tokens_generated)."""
        stop_seqs = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]

        start = time.monotonic()
        output = model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop_seqs,
            echo=False,
        )
        elapsed = time.monotonic() - start

        text = output["choices"][0]["text"].strip()
        usage = output.get("usage", {})
        tokens = usage.get("completion_tokens", max(1, len(text) // 4))

        return text, elapsed, tokens

    def run_test(
        self,
        model,
        test: BenchmarkTest,
        chat_format: str,
        augmentor_router: Optional["AugmentorRouter"] = None,
    ) -> TestResult:
        """Run a single test, optionally through the augmentor system."""
        result = TestResult(
            test_id=test.test_id,
            category=test.category,
            prompt=test.prompt,
            response="",
            passed=False,
            augmentor_used=augmentor_router is not None,
        )

        try:
            if augmentor_router is not None:
                # Run through augmentor system
                from engine.augmentors import AugmentorRouter as _AR
                from engine.base_model import BaseModel as _BM

                # Create a minimal shim that wraps the raw llama model
                # so AugmentorRouter.process() can call model.generate() and
                # model.count_tokens()
                shim = _ModelShim(model, self.max_tokens, self.temperature)

                augmentor_result = augmentor_router.process(
                    query=test.prompt,
                    model=shim,
                    chat_format=chat_format,
                    module_hint=test.augmentor_hint,
                    gen_kwargs={
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                )

                if augmentor_result is not None:
                    result.response = augmentor_result.response
                    result.augmentor_name = augmentor_result.augmentor_name
                    result.augmentor_attempts = augmentor_result.attempts
                    result.response_time = shim.total_time
                    result.tokens_generated = shim.total_tokens
                else:
                    # Augmentor router returned None, fall back to direct
                    prompt = wrap_chat(
                        "You are a coding assistant. Write clean, working code.",
                        test.prompt,
                        chat_format,
                    )
                    text, elapsed, tokens = self.generate(model, prompt)
                    result.response = text
                    result.response_time = elapsed
                    result.tokens_generated = tokens
            else:
                # Direct generation (no augmentor system)
                prompt = wrap_chat(
                    "You are a coding assistant. Write clean, working code.",
                    test.prompt,
                    chat_format,
                )
                text, elapsed, tokens = self.generate(model, prompt)
                result.response = text
                result.response_time = elapsed
                result.tokens_generated = tokens

        except Exception as e:
            result.response = f"[ERROR: {e}]"
            result.failure_reason = f"Generation error: {e}"
            logger.error(f"  ERROR in {test.test_id}: {e}")
            return result

        # Verify
        try:
            passed, reason = test.verify(result.response, test.prompt)
            result.passed = passed
            result.failure_reason = reason
        except Exception as e:
            result.passed = False
            result.failure_reason = f"Verification error: {e}"

        # Calculate tokens/second
        if result.response_time > 0 and result.tokens_generated > 0:
            result.tokens_per_second = result.tokens_generated / result.response_time

        return result

    def run_model(
        self,
        model_path: Path,
        run_augmentors: bool = True,
        run_no_augmentors: bool = True,
    ) -> list[ModelResult]:
        """Run the full test suite on a single model. Returns results for each mode."""
        chat_format = detect_chat_format(str(model_path))
        model_name = model_path.stem
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"  Size: {model_size_mb:.1f} MB | Format: {chat_format}")
        print(f"  Tests: {len(self.tests)} | Max tokens: {self.max_tokens}")
        print(f"{'='*70}")

        # Load model once
        try:
            model = self.load_model(model_path, chat_format)
        except Exception as e:
            print(f"  FAILED TO LOAD: {e}")
            logger.error(f"Failed to load {model_name}: {e}")
            return []

        model_results = []

        # --- Run WITHOUT augmentors ---
        if run_no_augmentors:
            print(f"\n  --- Mode: NO AUGMENTORS ---")
            mr = self._run_suite(
                model, model_name, str(model_path), model_size_mb,
                chat_format, augmentors_enabled=False, augmentor_router=None,
            )
            model_results.append(mr)
            self.results.append(mr)

        # --- Run WITH augmentors ---
        if run_augmentors:
            print(f"\n  --- Mode: WITH AUGMENTORS ---")
            from engine.augmentors import AugmentorRouter
            augmentor_router = AugmentorRouter()
            mr = self._run_suite(
                model, model_name, str(model_path), model_size_mb,
                chat_format, augmentors_enabled=True, augmentor_router=augmentor_router,
            )
            model_results.append(mr)
            self.results.append(mr)

        # Unload model
        del model
        gc.collect()
        print(f"\n  Model unloaded.")

        return model_results

    def _run_suite(
        self,
        model,
        model_name: str,
        model_path: str,
        model_size_mb: float,
        chat_format: str,
        augmentors_enabled: bool,
        augmentor_router,
    ) -> ModelResult:
        """Run all tests under one mode and aggregate results."""
        mr = ModelResult(
            model_name=model_name,
            model_path=model_path,
            model_size_mb=model_size_mb,
            chat_format=chat_format,
            augmentors_enabled=augmentors_enabled,
        )

        suite_start = time.monotonic()
        all_tps = []
        all_times = []
        category_counts: dict[str, dict] = {}

        for i, test in enumerate(self.tests, 1):
            tag = "AUGMENTOR" if augmentors_enabled else "DIRECT"
            print(f"  [{tag}] ({i}/{len(self.tests)}) {test.test_id}...", end=" ", flush=True)

            tr = self.run_test(
                model, test, chat_format,
                augmentor_router=augmentor_router if augmentors_enabled else None,
            )

            status = "PASS" if tr.passed else "FAIL"
            tps_str = f"{tr.tokens_per_second:.1f} tok/s" if tr.tokens_per_second > 0 else "N/A"
            print(f"{status} ({tr.response_time:.1f}s, {tps_str})")

            if not tr.passed and tr.failure_reason:
                print(f"         Reason: {tr.failure_reason}")

            mr.tests.append(asdict(tr))
            mr.total_tests += 1
            if tr.passed:
                mr.total_passed += 1

            if tr.tokens_per_second > 0:
                all_tps.append(tr.tokens_per_second)
            if tr.response_time > 0:
                all_times.append(tr.response_time)

            # Category tracking
            cat = test.category
            if cat not in category_counts:
                category_counts[cat] = {"total": 0, "passed": 0}
            category_counts[cat]["total"] += 1
            if tr.passed:
                category_counts[cat]["passed"] += 1

        # Aggregate
        mr.total_time = time.monotonic() - suite_start
        mr.quality_score = (mr.total_passed / mr.total_tests * 100) if mr.total_tests > 0 else 0
        mr.avg_tokens_per_second = (sum(all_tps) / len(all_tps)) if all_tps else 0
        mr.avg_response_time = (sum(all_times) / len(all_times)) if all_times else 0

        for cat, counts in category_counts.items():
            pct = (counts["passed"] / counts["total"] * 100) if counts["total"] > 0 else 0
            mr.category_scores[cat] = {
                "passed": counts["passed"],
                "total": counts["total"],
                "score": round(pct, 1),
            }

        # Print category summary
        tag = "AUGMENTOR" if augmentors_enabled else "DIRECT"
        print(f"\n  [{tag}] Results for {model_name}:")
        print(f"    Overall: {mr.total_passed}/{mr.total_tests} ({mr.quality_score:.1f}%)")
        for cat, scores in mr.category_scores.items():
            print(f"    {cat:12s}: {scores['passed']}/{scores['total']} ({scores['score']}%)")
        print(f"    Avg speed: {mr.avg_tokens_per_second:.1f} tok/s")
        print(f"    Avg time:  {mr.avg_response_time:.1f}s")
        print(f"    Total time: {mr.total_time:.1f}s")

        return mr


class _ModelShim:
    """
    Minimal wrapper around a raw llama-cpp model object to satisfy
    the AugmentorRouter.process() interface, which expects:
      - model.generate(prompt, **kwargs) -> str
      - model.count_tokens(text) -> int
    """

    def __init__(self, llama_model, default_max_tokens: int, default_temperature: float):
        self._model = llama_model
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self.total_time = 0.0
        self.total_tokens = 0

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None,
                 **kwargs) -> str:
        stop_seqs = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]
        mt = max_tokens or self._default_max_tokens
        temp = temperature or self._default_temperature

        # Strip grammar if present (llama_cpp handles it)
        gen_kwargs = {
            "max_tokens": mt,
            "temperature": temp,
            "stop": stop_seqs,
            "echo": False,
        }
        if "grammar" in kwargs and kwargs["grammar"] is not None:
            gen_kwargs["grammar"] = kwargs["grammar"]

        start = time.monotonic()
        output = self._model(prompt, **gen_kwargs)
        elapsed = time.monotonic() - start

        text = output["choices"][0]["text"].strip()
        usage = output.get("usage", {})
        tokens = usage.get("completion_tokens", max(1, len(text) // 4))

        self.total_time += elapsed
        self.total_tokens += tokens

        return text

    def count_tokens(self, text: str) -> int:
        try:
            tokens = self._model.tokenize(text.encode("utf-8"))
            return len(tokens)
        except Exception:
            return len(text) // 4


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary_table(results: list[ModelResult]):
    """Print a formatted comparison table of all results."""
    if not results:
        print("\nNo results to display.")
        return

    print("\n")
    print("=" * 110)
    print("BENCHMARK SUMMARY")
    print("=" * 110)

    # Header
    header = (
        f"{'Model':<35s} {'Mode':<8s} {'Score':>6s} "
        f"{'CodeGen':>8s} {'Debug':>8s} {'Review':>8s} {'Explain':>8s} "
        f"{'tok/s':>7s} {'Avg(s)':>7s} {'Total':>7s}"
    )
    print(header)
    print("-" * 110)

    # Sort by model name, then augmentors_enabled
    sorted_results = sorted(results, key=lambda r: (r.model_name, r.augmentors_enabled))

    for mr in sorted_results:
        mode = "AUGMENTOR" if mr.augmentors_enabled else "DIRECT"
        cg = mr.category_scores.get("code_gen", {}).get("score", 0)
        db = mr.category_scores.get("debug", {}).get("score", 0)
        cr = mr.category_scores.get("review", {}).get("score", 0)
        ex = mr.category_scores.get("explain", {}).get("score", 0)

        # Truncate long model names
        name = mr.model_name
        if len(name) > 34:
            name = name[:31] + "..."

        line = (
            f"{name:<35s} {mode:<8s} {mr.quality_score:>5.1f}% "
            f"{cg:>7.1f}% {db:>7.1f}% {cr:>7.1f}% {ex:>7.1f}% "
            f"{mr.avg_tokens_per_second:>7.1f} {mr.avg_response_time:>7.1f} {mr.total_time:>6.0f}s"
        )
        print(line)

    print("-" * 110)

    # Augmentor impact analysis
    print("\nAUGMENTOR SYSTEM IMPACT:")
    print("-" * 60)

    model_names = set(r.model_name for r in results)
    for name in sorted(model_names):
        direct = [r for r in results if r.model_name == name and not r.augmentors_enabled]
        augmented = [r for r in results if r.model_name == name and r.augmentors_enabled]
        if direct and augmented:
            d = direct[0]
            e = augmented[0]
            delta = e.quality_score - d.quality_score
            sign = "+" if delta >= 0 else ""
            short_name = name if len(name) <= 40 else name[:37] + "..."
            print(f"  {short_name:<42s} {d.quality_score:>5.1f}% -> {e.quality_score:>5.1f}% ({sign}{delta:>+5.1f}%)")

    print()


def save_json_results(results: list[ModelResult], output_path: Path):
    """Save detailed results to JSON."""
    data = {
        "benchmark_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_models": len(set(r.model_name for r in results)),
        "total_tests_per_model": results[0].total_tests if results else 0,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    print(f"Detailed results saved to: {output_path}")


def save_text_summary(results: list[ModelResult], output_path: Path):
    """Save a human-readable summary to text file."""
    lines = []
    lines.append("=" * 80)
    lines.append("ULTRALITE CODE ASSISTANT — BENCHMARK RESULTS")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Ranking
    lines.append("RANKING (by quality score, augmentor mode):")
    lines.append("-" * 60)

    augmentor_results = sorted(
        [r for r in results if r.augmentors_enabled],
        key=lambda r: r.quality_score,
        reverse=True,
    )
    direct_results = sorted(
        [r for r in results if not r.augmentors_enabled],
        key=lambda r: r.quality_score,
        reverse=True,
    )

    if augmentor_results:
        lines.append("")
        lines.append("  With Augmentor System:")
        for i, r in enumerate(augmentor_results, 1):
            lines.append(
                f"    {i}. {r.model_name} — {r.quality_score:.1f}% "
                f"({r.total_passed}/{r.total_tests}) "
                f"[{r.avg_tokens_per_second:.1f} tok/s, {r.model_size_mb:.0f}MB]"
            )

    if direct_results:
        lines.append("")
        lines.append("  Without Augmentor System:")
        for i, r in enumerate(direct_results, 1):
            lines.append(
                f"    {i}. {r.model_name} — {r.quality_score:.1f}% "
                f"({r.total_passed}/{r.total_tests}) "
                f"[{r.avg_tokens_per_second:.1f} tok/s, {r.model_size_mb:.0f}MB]"
            )

    # Per-model detail
    lines.append("")
    lines.append("=" * 80)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 80)

    sorted_all = sorted(results, key=lambda r: (r.model_name, r.augmentors_enabled))
    for mr in sorted_all:
        mode = "WITH AUGMENTORS" if mr.augmentors_enabled else "WITHOUT AUGMENTORS"
        lines.append("")
        lines.append(f"--- {mr.model_name} ({mode}) ---")
        lines.append(f"  Quality: {mr.quality_score:.1f}% ({mr.total_passed}/{mr.total_tests})")
        lines.append(f"  Speed:   {mr.avg_tokens_per_second:.1f} tok/s avg")
        lines.append(f"  Time:    {mr.avg_response_time:.1f}s avg, {mr.total_time:.0f}s total")
        lines.append(f"  Format:  {mr.chat_format}")
        lines.append(f"  Size:    {mr.model_size_mb:.0f} MB")
        lines.append("")

        for cat, scores in mr.category_scores.items():
            lines.append(f"  {cat:12s}: {scores['passed']}/{scores['total']} ({scores['score']}%)")

        lines.append("")
        lines.append("  Failed tests:")
        failed = [t for t in mr.tests if not t["passed"]]
        if not failed:
            lines.append("    (none)")
        else:
            for t in failed:
                lines.append(f"    - {t['test_id']}: {t['failure_reason']}")

    # Augmentor impact
    lines.append("")
    lines.append("=" * 80)
    lines.append("AUGMENTOR SYSTEM IMPACT")
    lines.append("=" * 80)
    lines.append("")

    model_names = sorted(set(r.model_name for r in results))
    for name in model_names:
        direct = [r for r in results if r.model_name == name and not r.augmentors_enabled]
        augmented = [r for r in results if r.model_name == name and r.augmentors_enabled]
        if direct and augmented:
            d = direct[0]
            e = augmented[0]
            delta = e.quality_score - d.quality_score
            lines.append(f"  {name}:")
            lines.append(f"    Direct: {d.quality_score:.1f}% -> Augmentor: {e.quality_score:.1f}% (delta: {delta:+.1f}%)")

            # Per-category delta
            for cat in d.category_scores:
                ds = d.category_scores.get(cat, {}).get("score", 0)
                es = e.category_scores.get(cat, {}).get("score", 0)
                cd = es - ds
                lines.append(f"      {cat:12s}: {ds:.1f}% -> {es:.1f}% ({cd:+.1f}%)")
            lines.append("")

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Summary saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GGUF models on code tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmark.py                          # Full benchmark\n"
            "  python benchmark.py --model models/foo.gguf  # Single model\n"
            "  python benchmark.py --category code_gen      # One category\n"
            "  python benchmark.py --quick                  # Smoke test\n"
            "  python benchmark.py --no-augmentors           # Skip augmentor runs\n"
        ),
    )
    parser.add_argument("--model", type=str, help="Path to a specific GGUF model to test")
    parser.add_argument("--category", type=str, choices=["code_gen", "debug", "review", "explain"],
                        help="Run only one test category")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 test per category")
    parser.add_argument("--no-augmentors", action="store_true", help="Skip augmentor-enabled runs")
    parser.add_argument("--no-direct", action="store_true", help="Skip direct (no-augmentor) runs")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--gpu-layers", type=int, default=99, help="GPU layers to offload (default: 99)")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (default: 8)")
    parser.add_argument("--context-length", type=int, default=4096, help="Context length (default: 4096)")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT), help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build test suite
    all_tests = build_test_suite()

    # Filter by category
    if args.category:
        all_tests = [t for t in all_tests if t.category == args.category]
        print(f"Filtered to category: {args.category} ({len(all_tests)} tests)")

    # Quick mode: 1 test per category
    if args.quick:
        seen_cats = set()
        quick_tests = []
        for t in all_tests:
            if t.category not in seen_cats:
                quick_tests.append(t)
                seen_cats.add(t.category)
        all_tests = quick_tests
        print(f"Quick mode: {len(all_tests)} tests (1 per category)")

    if not all_tests:
        print("No tests to run.")
        sys.exit(1)

    # Discover models
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            sys.exit(1)
        models = [model_path]
    else:
        models = discover_models()

    if not models:
        print("No GGUF models found in models/ or ../plug-in-intelligence-engine/models/")
        print("Download a model first: python download_model.py")
        sys.exit(1)

    print(f"\nUltralite Code Assistant — Benchmark")
    print(f"Models: {len(models)}")
    print(f"Tests:  {len(all_tests)} per model")
    print(f"Modes:  {'direct' if not args.no_direct else ''}"
          f"{'+ augmentor' if not args.no_augmentors else ''}")
    print(f"Config: max_tokens={args.max_tokens}, temp={args.temperature}, "
          f"gpu_layers={args.gpu_layers}, threads={args.threads}")
    print(f"\nModels to benchmark:")
    for m in models:
        size_mb = m.stat().st_size / (1024 * 1024)
        fmt = detect_chat_format(str(m))
        print(f"  - {m.name} ({size_mb:.0f} MB, {fmt})")

    # Create runner
    runner = BenchmarkRunner(
        tests=all_tests,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_layers=args.gpu_layers,
        threads=args.threads,
        context_length=args.context_length,
    )

    # Run benchmark
    total_start = time.monotonic()

    for model_path in models:
        try:
            runner.run_model(
                model_path,
                run_augmentors=not args.no_augmentors,
                run_no_augmentors=not args.no_direct,
            )
        except Exception as e:
            print(f"\n  FATAL ERROR with {model_path.name}: {e}")
            logger.exception(f"Fatal error benchmarking {model_path.name}")

    total_elapsed = time.monotonic() - total_start

    # Output
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE — {total_elapsed:.0f}s total")
    print(f"{'='*70}")

    print_summary_table(runner.results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_results.json"
    txt_path = output_dir / "benchmark_output.txt"

    save_json_results(runner.results, json_path)
    save_text_summary(runner.results, txt_path)

    print(f"\nTotal wall time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
