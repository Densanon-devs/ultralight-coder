#!/usr/bin/env python3
"""
Execution-Based Benchmark — Actually runs the generated code.

Unlike benchmark.py which checks structure ("does it have a function def?"),
this benchmark extracts the code, executes it, and runs test cases.

Each test:
  1. Prompts the model to write a function
  2. Extracts the code from the response
  3. Executes it in a sandboxed namespace
  4. Runs test assertions against it
  5. Scores pass/fail based on actual correctness

Usage:
    python benchmark_exec.py                          # Top 3 models
    python benchmark_exec.py --model models/foo.gguf  # Specific model
    python benchmark_exec.py --all                    # All models
    python benchmark_exec.py --quick                  # 5 tests only
"""

import argparse
import gc
import json
import logging
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from io import StringIO
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("bench_exec")


# ---------------------------------------------------------------------------
# Chat format detection (same as benchmark.py)
# ---------------------------------------------------------------------------

def detect_chat_format(model_path: str) -> str:
    name = Path(model_path).name.lower()
    for sub, fmt in [("qwen","chatml"),("smollm","chatml"),("llama-3","llama3"),
                     ("Llama-3","llama3"),("phi-3","phi3"),("Phi-3","phi3"),
                     ("tinyllama","alpaca"),("TinyLlama","alpaca")]:
        if sub.lower() in name:
            return fmt
    return "chatml"


def wrap_chat(system: str, user: str, fmt: str) -> str:
    if fmt == "chatml":
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    elif fmt == "llama3":
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    elif fmt == "phi3":
        return f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
    elif fmt == "alpaca":
        return f"### System:\n{system}\n\n### Instruction:\n{user}\n\n### Response:\n"
    return f"System: {system}\n\nUser: {user}\n\nAssistant:"


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(response: str) -> str:
    """Extract Python code from a model response."""
    # Try markdown code blocks first
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        return "\n".join(blocks)

    # Try to find function/class definitions
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            in_code = True
        if in_code:
            # Stop at blank line after dedent, or at explanation text
            if stripped == "" and code_lines and not code_lines[-1].strip() == "":
                # Check if next non-empty line is code
                continue
            if stripped and not stripped.startswith(("#", " ", "\t", "def ", "class ",
                "import ", "from ", "return", "if ", "else", "elif", "for ", "while ",
                "try:", "except", "finally", "with ", "raise", "yield", "pass",
                "break", "continue", "assert", "@", ")", "]", "}", "'''", '"""')):
                if not any(c in stripped for c in ["=", "(", "[", "{"]):
                    break
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    # Last resort: return everything
    return response


def safe_exec(code: str, timeout: float = 5.0) -> tuple[dict, str]:
    """Execute code in an isolated namespace. Returns (namespace, error_str)."""
    namespace = {"__builtins__": __builtins__}
    try:
        exec(code, namespace)
        return namespace, ""
    except Exception as e:
        return namespace, f"{type(e).__name__}: {e}"


def run_tests(namespace: dict, test_code: str) -> tuple[int, int, list[str]]:
    """Run test assertions against the namespace. Returns (passed, total, errors)."""
    # Split test code into individual assertions
    tests = [t.strip() for t in test_code.strip().split("\n") if t.strip() and not t.strip().startswith("#")]
    passed = 0
    total = len(tests)
    errors = []

    for test in tests:
        try:
            exec(test, namespace)
            passed += 1
        except AssertionError as e:
            errors.append(f"FAIL: {test} -> {e}")
        except Exception as e:
            errors.append(f"ERROR: {test} -> {type(e).__name__}: {e}")

    return passed, total, errors


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

@dataclass
class ExecTest:
    test_id: str
    prompt: str
    test_code: str  # assertions to run against the generated code
    func_name: str  # expected function name (for extraction hints)


def build_exec_tests() -> list[ExecTest]:
    tests = []

    # ── Basic Functions ──────────────────────────────────────

    tests.append(ExecTest(
        test_id="e01_fibonacci",
        prompt="Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number. fibonacci(0)=0, fibonacci(1)=1, fibonacci(2)=1, fibonacci(10)=55.",
        func_name="fibonacci",
        test_code="""
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(5) == 5
assert fibonacci(10) == 55
assert fibonacci(15) == 610
""",
    ))

    tests.append(ExecTest(
        test_id="e02_is_prime",
        prompt="Write a Python function called `is_prime(n)` that returns True if n is prime, False otherwise. Handle 0, 1, negative numbers, and 2.",
        func_name="is_prime",
        test_code="""
assert is_prime(2) == True
assert is_prime(3) == True
assert is_prime(4) == False
assert is_prime(17) == True
assert is_prime(1) == False
assert is_prime(0) == False
assert is_prime(-5) == False
assert is_prime(97) == True
assert is_prime(100) == False
""",
    ))

    tests.append(ExecTest(
        test_id="e03_reverse_string",
        prompt="Write a Python function called `reverse_string(s)` that returns the reversed string.",
        func_name="reverse_string",
        test_code="""
assert reverse_string("hello") == "olleh"
assert reverse_string("") == ""
assert reverse_string("a") == "a"
assert reverse_string("abcd") == "dcba"
""",
    ))

    tests.append(ExecTest(
        test_id="e04_flatten",
        prompt="Write a Python function called `flatten(lst)` that flattens a nested list of arbitrary depth. Example: flatten([1, [2, [3, 4]], 5]) returns [1, 2, 3, 4, 5].",
        func_name="flatten",
        test_code="""
assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
assert flatten([]) == []
assert flatten([1, 2, 3]) == [1, 2, 3]
assert flatten([[1], [2], [3]]) == [1, 2, 3]
assert flatten([[[1]], [[2]], [[3]]]) == [1, 2, 3]
""",
    ))

    tests.append(ExecTest(
        test_id="e05_word_count",
        prompt="Write a Python function called `word_count(text)` that returns a dictionary mapping each word (lowercased) to its frequency. Split on whitespace.",
        func_name="word_count",
        test_code="""
assert word_count("hello world hello") == {"hello": 2, "world": 1}
assert word_count("") == {}
assert word_count("one") == {"one": 1}
assert word_count("The the THE") == {"the": 3}
""",
    ))

    # ── Algorithms ───────────────────────────────────────────

    tests.append(ExecTest(
        test_id="e06_binary_search",
        prompt="Write a Python function called `binary_search(arr, target)` that returns the index of target in sorted array arr, or -1 if not found.",
        func_name="binary_search",
        test_code="""
assert binary_search([1, 3, 5, 7, 9], 5) == 2
assert binary_search([1, 3, 5, 7, 9], 1) == 0
assert binary_search([1, 3, 5, 7, 9], 9) == 4
assert binary_search([1, 3, 5, 7, 9], 4) == -1
assert binary_search([], 1) == -1
assert binary_search([1], 1) == 0
""",
    ))

    tests.append(ExecTest(
        test_id="e07_merge_sorted",
        prompt="Write a Python function called `merge_sorted(a, b)` that merges two sorted lists into one sorted list. Do not use built-in sort.",
        func_name="merge_sorted",
        test_code="""
assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_sorted([], [1, 2]) == [1, 2]
assert merge_sorted([1, 2], []) == [1, 2]
assert merge_sorted([], []) == []
assert merge_sorted([1, 1], [1, 1]) == [1, 1, 1, 1]
assert merge_sorted([5], [1]) == [1, 5]
""",
    ))

    tests.append(ExecTest(
        test_id="e08_two_sum",
        prompt="Write a Python function called `two_sum(nums, target)` that returns the indices of two numbers that add up to target. Return a list of 2 indices. Assume exactly one solution exists.",
        func_name="two_sum",
        test_code="""
result = two_sum([2, 7, 11, 15], 9)
assert sorted(result) == [0, 1]
result = two_sum([3, 2, 4], 6)
assert sorted(result) == [1, 2]
result = two_sum([3, 3], 6)
assert sorted(result) == [0, 1]
""",
    ))

    tests.append(ExecTest(
        test_id="e09_max_subarray",
        prompt="Write a Python function called `max_subarray(nums)` that returns the largest sum of a contiguous subarray (Kadane's algorithm). Example: max_subarray([-2,1,-3,4,-1,2,1,-5,4]) returns 6.",
        func_name="max_subarray",
        test_code="""
assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray([1]) == 1
assert max_subarray([-1]) == -1
assert max_subarray([5, 4, -1, 7, 8]) == 23
assert max_subarray([-2, -1]) == -1
""",
    ))

    tests.append(ExecTest(
        test_id="e10_roman_to_int",
        prompt="Write a Python function called `roman_to_int(s)` that converts a Roman numeral string to an integer. Handle subtractive cases: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900.",
        func_name="roman_to_int",
        test_code="""
assert roman_to_int("III") == 3
assert roman_to_int("IV") == 4
assert roman_to_int("IX") == 9
assert roman_to_int("XLII") == 42
assert roman_to_int("MCMXCIV") == 1994
assert roman_to_int("MMXXVI") == 2026
""",
    ))

    # ── Data Structures ──────────────────────────────────────

    tests.append(ExecTest(
        test_id="e11_stack",
        prompt="Write a Python class called `Stack` with methods: push(item), pop() that returns the item, peek() that returns top without removing, is_empty() that returns bool, and __len__. Raise IndexError on pop/peek when empty.",
        func_name="Stack",
        test_code="""
s = Stack()
assert s.is_empty() == True
assert len(s) == 0
s.push(1)
s.push(2)
s.push(3)
assert len(s) == 3
assert s.peek() == 3
assert s.pop() == 3
assert s.pop() == 2
assert len(s) == 1
assert s.is_empty() == False
assert s.pop() == 1
assert s.is_empty() == True
try:
    s.pop()
    assert False, "Should have raised IndexError"
except IndexError:
    pass
""",
    ))

    tests.append(ExecTest(
        test_id="e12_matrix_multiply",
        prompt="Write a Python function called `matrix_multiply(a, b)` that multiplies two matrices (2D lists). Raise ValueError if dimensions don't match.",
        func_name="matrix_multiply",
        test_code="""
assert matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
assert matrix_multiply([[1, 0], [0, 1]], [[5, 6], [7, 8]]) == [[5, 6], [7, 8]]
assert matrix_multiply([[2]], [[3]]) == [[6]]
try:
    matrix_multiply([[1, 2]], [[1, 2]])
    assert False, "Should have raised ValueError"
except ValueError:
    pass
""",
    ))

    # ── String Processing ────────────────────────────────────

    tests.append(ExecTest(
        test_id="e13_palindrome",
        prompt="Write a Python function called `is_palindrome(s)` that returns True if the string is a palindrome, ignoring case and non-alphanumeric characters. Example: is_palindrome('A man, a plan, a canal: Panama') returns True.",
        func_name="is_palindrome",
        test_code="""
assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("race a car") == False
assert is_palindrome("") == True
assert is_palindrome("a") == True
assert is_palindrome("Was it a car or a cat I saw?") == True
""",
    ))

    tests.append(ExecTest(
        test_id="e14_longest_common_prefix",
        prompt="Write a Python function called `longest_common_prefix(strs)` that returns the longest common prefix among a list of strings. Return '' if no common prefix.",
        func_name="longest_common_prefix",
        test_code="""
assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
assert longest_common_prefix(["dog", "racecar", "car"]) == ""
assert longest_common_prefix(["interspecies", "interstellar", "interstate"]) == "inters"
assert longest_common_prefix([]) == ""
assert longest_common_prefix(["alone"]) == "alone"
""",
    ))

    tests.append(ExecTest(
        test_id="e15_caesar_cipher",
        prompt="Write a Python function called `caesar_cipher(text, shift)` that applies Caesar cipher. Only shift a-z and A-Z, leave other characters unchanged. Wrap around: z+1=a.",
        func_name="caesar_cipher",
        test_code="""
assert caesar_cipher("abc", 1) == "bcd"
assert caesar_cipher("xyz", 3) == "abc"
assert caesar_cipher("Hello, World!", 13) == "Uryyb, Jbeyq!"
assert caesar_cipher("abc", 0) == "abc"
assert caesar_cipher("ABC", 26) == "ABC"
""",
    ))

    # ── Math/Logic ───────────────────────────────────────────

    tests.append(ExecTest(
        test_id="e16_gcd",
        prompt="Write a Python function called `gcd(a, b)` that returns the greatest common divisor using Euclid's algorithm.",
        func_name="gcd",
        test_code="""
assert gcd(12, 8) == 4
assert gcd(54, 24) == 6
assert gcd(7, 13) == 1
assert gcd(0, 5) == 5
assert gcd(100, 100) == 100
""",
    ))

    tests.append(ExecTest(
        test_id="e17_power",
        prompt="Write a Python function called `power(base, exp)` that computes base**exp for non-negative integer exp. Do not use the ** operator or pow(). Use fast exponentiation (repeated squaring).",
        func_name="power",
        test_code="""
assert power(2, 10) == 1024
assert power(3, 0) == 1
assert power(5, 1) == 5
assert power(2, 20) == 1048576
assert power(1, 100) == 1
assert power(7, 3) == 343
""",
    ))

    tests.append(ExecTest(
        test_id="e18_unique_paths",
        prompt="Write a Python function called `unique_paths(m, n)` that returns the number of unique paths from top-left to bottom-right of an m x n grid, moving only right or down.",
        func_name="unique_paths",
        test_code="""
assert unique_paths(3, 7) == 28
assert unique_paths(3, 2) == 3
assert unique_paths(1, 1) == 1
assert unique_paths(1, 5) == 1
assert unique_paths(7, 3) == 28
""",
    ))

    # ── Practical Utilities ──────────────────────────────────

    tests.append(ExecTest(
        test_id="e19_valid_parentheses",
        prompt="Write a Python function called `valid_parentheses(s)` that returns True if the string has valid matching parentheses/brackets/braces. Example: '([{}])' is valid, '([)]' is not.",
        func_name="valid_parentheses",
        test_code="""
assert valid_parentheses("()") == True
assert valid_parentheses("()[]{}") == True
assert valid_parentheses("(]") == False
assert valid_parentheses("([)]") == False
assert valid_parentheses("{[]}") == True
assert valid_parentheses("") == True
assert valid_parentheses("((()))") == True
""",
    ))

    tests.append(ExecTest(
        test_id="e20_fizzbuzz",
        prompt="Write a Python function called `fizzbuzz(n)` that returns a list of strings from 1 to n. For multiples of 3, use 'Fizz'. For multiples of 5, use 'Buzz'. For multiples of both, use 'FizzBuzz'. Otherwise, use the number as a string.",
        func_name="fizzbuzz",
        test_code="""
result = fizzbuzz(15)
assert result[0] == "1"
assert result[2] == "Fizz"
assert result[4] == "Buzz"
assert result[14] == "FizzBuzz"
assert len(result) == 15
assert result[5] == "Fizz"
assert result[9] == "Buzz"
""",
    ))

    return tests


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

TOP_MODELS = [
    "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",                      # local
    "../plug-in-intelligence-engine/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
    "../plug-in-intelligence-engine/models/SmolLM2-135M-Instruct-Q4_K_M.gguf",
]

def discover_models(all_models: bool = False) -> list[Path]:
    if all_models:
        dirs = [PROJECT_ROOT / "models", PROJECT_ROOT.parent / "plug-in-intelligence-engine" / "models"]
        models = []
        for d in dirs:
            if d.exists():
                models.extend(sorted(d.glob("*.gguf")))
        return models

    # Default: top performers from structural benchmark
    found = []
    for p in TOP_MODELS:
        path = PROJECT_ROOT / p
        if path.exists():
            found.append(path)
    return found


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ExecTestResult:
    test_id: str
    prompt: str
    response: str
    extracted_code: str
    exec_error: str
    tests_passed: int
    tests_total: int
    test_errors: list[str]
    response_time: float
    score: float  # tests_passed / tests_total

@dataclass
class ExecModelResult:
    model_name: str
    model_path: str
    model_size_mb: float
    chat_format: str
    total_score: float  # average of test scores
    tests_run: int
    perfect_count: int  # tests with 100% assertions passing
    avg_response_time: float
    avg_tokens_per_second: float
    results: list[ExecTestResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExecBenchmarkRunner:
    def __init__(self, tests: list[ExecTest], max_tokens: int = 512,
                 temperature: float = 0.2, gpu_layers: int = 99,
                 threads: int = 8, context_length: int = 4096):
        self.tests = tests
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.context_length = context_length
        self.all_results: list[ExecModelResult] = []

    def load_model(self, model_path: Path):
        from llama_cpp import Llama
        return Llama(
            model_path=str(model_path),
            n_ctx=self.context_length,
            n_gpu_layers=self.gpu_layers,
            n_threads=self.threads,
            n_batch=512,
            verbose=False,
        )

    def generate(self, model, prompt: str) -> tuple[str, float, int]:
        fmt = detect_chat_format(str(model.model_path))
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]
        start = time.monotonic()
        output = model(prompt, max_tokens=self.max_tokens, temperature=self.temperature,
                       stop=stop, echo=False)
        elapsed = time.monotonic() - start
        text = output["choices"][0]["text"].strip()
        tokens = output.get("usage", {}).get("completion_tokens", max(1, len(text) // 4))
        return text, elapsed, tokens

    def run_test(self, model, test: ExecTest, chat_format: str) -> ExecTestResult:
        system = (
            "Write a Python function as requested. Return ONLY the code in a ```python block. "
            "No explanation needed. The function must be complete and runnable."
        )
        prompt = wrap_chat(system, test.prompt, chat_format)

        # Generate
        response, elapsed, tokens = self.generate(model, prompt)
        tps = tokens / elapsed if elapsed > 0 else 0

        # Extract code
        code = extract_code(response)

        # Execute
        namespace, exec_error = safe_exec(code)

        # Check function exists
        if not exec_error and test.func_name not in namespace:
            exec_error = f"Function '{test.func_name}' not found in generated code"

        # Run tests
        if exec_error:
            return ExecTestResult(
                test_id=test.test_id, prompt=test.prompt, response=response,
                extracted_code=code, exec_error=exec_error,
                tests_passed=0, tests_total=test.test_code.strip().count("assert"),
                test_errors=[exec_error], response_time=elapsed,
                score=0.0,
            )

        passed, total, errors = run_tests(namespace, test.test_code)

        return ExecTestResult(
            test_id=test.test_id, prompt=test.prompt, response=response,
            extracted_code=code, exec_error="",
            tests_passed=passed, tests_total=total,
            test_errors=errors, response_time=elapsed,
            score=passed / total if total > 0 else 0.0,
        )

    def run_model(self, model_path: Path) -> ExecModelResult:
        chat_format = detect_chat_format(str(model_path))
        model_name = model_path.stem
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"  Size: {model_size_mb:.0f} MB | Format: {chat_format} | Tests: {len(self.tests)}")
        print(f"{'='*70}")

        try:
            model = self.load_model(model_path)
        except Exception as e:
            print(f"  FAILED TO LOAD: {e}")
            return None

        mr = ExecModelResult(
            model_name=model_name, model_path=str(model_path),
            model_size_mb=model_size_mb, chat_format=chat_format,
            total_score=0, tests_run=0, perfect_count=0,
            avg_response_time=0, avg_tokens_per_second=0,
        )

        all_scores = []
        all_times = []
        all_tps = []

        for i, test in enumerate(self.tests, 1):
            print(f"  ({i}/{len(self.tests)}) {test.test_id}...", end=" ", flush=True)

            tr = self.run_test(model, test, chat_format)
            mr.results.append(tr)
            all_scores.append(tr.score)
            all_times.append(tr.response_time)
            mr.tests_run += 1

            if tr.exec_error:
                print(f"EXEC_ERR ({tr.response_time:.1f}s)")
                print(f"         {tr.exec_error[:100]}")
            elif tr.score == 1.0:
                tps = len(tr.response.split()) / tr.response_time if tr.response_time > 0 else 0
                print(f"PERFECT {tr.tests_passed}/{tr.tests_total} ({tr.response_time:.1f}s)")
                mr.perfect_count += 1
            else:
                print(f"PARTIAL {tr.tests_passed}/{tr.tests_total} ({tr.score:.0%}) ({tr.response_time:.1f}s)")
                for err in tr.test_errors[:2]:
                    print(f"         {err[:100]}")

        # Aggregate
        mr.total_score = sum(all_scores) / len(all_scores) * 100 if all_scores else 0
        mr.avg_response_time = sum(all_times) / len(all_times) if all_times else 0

        print(f"\n  RESULTS: {model_name}")
        print(f"    Execution Score: {mr.total_score:.1f}%")
        print(f"    Perfect tests:   {mr.perfect_count}/{mr.tests_run}")
        print(f"    Avg time:        {mr.avg_response_time:.1f}s")

        # Unload
        del model
        gc.collect()

        self.all_results.append(mr)
        return mr


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(results: list[ExecModelResult]):
    print(f"\n{'='*70}")
    print("EXECUTION BENCHMARK — FINAL RANKINGS")
    print(f"{'='*70}\n")

    sorted_r = sorted(results, key=lambda r: (-r.total_score, -r.perfect_count))

    print(f"  {'Model':<45s} {'Score':>6s} {'Perfect':>8s} {'Size':>7s} {'Time':>6s}")
    print(f"  {'-'*45} {'-'*6} {'-'*8} {'-'*7} {'-'*6}")

    for r in sorted_r:
        print(f"  {r.model_name:<45s} {r.total_score:>5.1f}% "
              f"{r.perfect_count:>3d}/{r.tests_run:<3d}  "
              f"{r.model_size_mb:>5.0f}MB {r.avg_response_time:>5.1f}s")

    print()


def save_results(results: list[ExecModelResult], path: Path):
    data = {
        "benchmark": "execution",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Execution-based code benchmark")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--all", action="store_true", help="Test all available models")
    parser.add_argument("--quick", action="store_true", help="Quick: first 5 tests only")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    tests = build_exec_tests()
    if args.quick:
        tests = tests[:5]
        print(f"Quick mode: {len(tests)} tests")

    if args.model:
        models = [Path(args.model)]
    elif args.all:
        models = discover_models(all_models=True)
    else:
        models = discover_models(all_models=False)

    if not models:
        print("No models found.")
        sys.exit(1)

    print(f"\nExecution Benchmark — {len(tests)} tests, {len(models)} models")
    for m in models:
        print(f"  - {m.name} ({m.stat().st_size / (1024*1024):.0f} MB)")

    runner = ExecBenchmarkRunner(
        tests=tests, max_tokens=args.max_tokens, temperature=args.temperature,
        gpu_layers=args.gpu_layers, threads=args.threads,
    )

    for model_path in models:
        try:
            runner.run_model(model_path)
        except Exception as e:
            print(f"\n  FATAL: {model_path.name}: {e}")
            traceback.print_exc()

    print_summary(runner.all_results)
    save_results(runner.all_results, PROJECT_ROOT / "benchmark_exec_results.json")


if __name__ == "__main__":
    main()
