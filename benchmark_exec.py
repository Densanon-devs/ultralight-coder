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
    # Parse test code into executable blocks.
    # Multi-line blocks (try/except, if/else, for, while, with) are grouped
    # together and executed as a single unit.
    raw_lines = test_code.strip().split("\n")

    blocks: list[str] = []
    current_block: list[str] = []
    in_multiline = False

    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if in_multiline:
            current_block.append(line)
            # A multiline block ends when we hit a non-indented line that is
            # NOT a continuation keyword (except, elif, else, finally).
            if stripped and not line[0].isspace() and not stripped.startswith(
                ("except", "elif", "else:", "else ", "finally")
            ):
                # This line is a new top-level statement — flush the block
                # without this line, then start fresh.
                blocks.append("\n".join(current_block[:-1]))
                current_block = [line]
                in_multiline = stripped.endswith(":") and stripped.startswith(
                    ("try:", "if ", "for ", "while ", "with ")
                )
            elif not line[0].isspace() and stripped.startswith(
                ("except", "elif", "else:", "else ", "finally")
            ):
                # Continuation of a compound statement — stay in multiline.
                pass
            else:
                # Indented line — still inside the block.
                pass
        else:
            # Check if this line starts a compound statement
            if stripped.endswith(":") and stripped.startswith(
                ("try:", "if ", "for ", "while ", "with ")
            ):
                if current_block:
                    blocks.append("\n".join(current_block))
                current_block = [line]
                in_multiline = True
            else:
                current_block.append(line)

    # Flush remaining
    if current_block:
        if in_multiline:
            blocks.append("\n".join(current_block))
        else:
            # Each remaining single line is its own block
            for line in current_block:
                blocks.append(line.strip())

    # Count total test assertions across all blocks
    total = sum(b.count("assert ") for b in blocks)
    passed = 0
    errors = []

    for block in blocks:
        block_asserts = block.count("assert ")
        try:
            exec(block, namespace)
            passed += block_asserts
        except AssertionError as e:
            # For single-assert blocks, report the exact line. For multi-line
            # blocks, report a summary.
            first_line = block.split("\n")[0].strip()
            errors.append(f"FAIL: {first_line} -> {e}")
        except Exception as e:
            first_line = block.split("\n")[0].strip()
            errors.append(f"ERROR: {first_line} -> {type(e).__name__}: {e}")

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

    # ── Algorithm Weaknesses (models fail these) ─────────────

    tests.append(ExecTest(
        test_id="e21_fibonacci_memo",
        prompt="Write a Python function called `fibonacci_memo(n)` that returns the nth Fibonacci number using memoization. It must handle large n efficiently. fibonacci_memo(0)=0, fibonacci_memo(1)=1, fibonacci_memo(30)=832040.",
        func_name="fibonacci_memo",
        test_code="""
assert fibonacci_memo(0) == 0
assert fibonacci_memo(1) == 1
assert fibonacci_memo(2) == 1
assert fibonacci_memo(10) == 55
assert fibonacci_memo(30) == 832040
assert fibonacci_memo(40) == 102334155
""",
    ))

    tests.append(ExecTest(
        test_id="e22_kadane_negative",
        prompt="Write a Python function called `max_subarray_neg(nums)` that returns the largest sum of a contiguous subarray using Kadane's algorithm. It MUST handle all-negative arrays correctly by returning the largest (least negative) element. Example: max_subarray_neg([-3, -2, -5]) returns -2.",
        func_name="max_subarray_neg",
        test_code="""
assert max_subarray_neg([-3, -2, -5]) == -2
assert max_subarray_neg([-1]) == -1
assert max_subarray_neg([-10, -7, -3, -8]) == -3
assert max_subarray_neg([1, 2, 3]) == 6
assert max_subarray_neg([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray_neg([5]) == 5
""",
    ))

    tests.append(ExecTest(
        test_id="e23_roman_to_int_hard",
        prompt="Write a Python function called `roman_to_int_hard(s)` that converts a Roman numeral string to an integer. Handle ALL subtractive cases: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900. Example: roman_to_int_hard('MCMXCIX') returns 1999.",
        func_name="roman_to_int_hard",
        test_code="""
assert roman_to_int_hard("MCMXCIX") == 1999
assert roman_to_int_hard("IV") == 4
assert roman_to_int_hard("IX") == 9
assert roman_to_int_hard("XL") == 40
assert roman_to_int_hard("XC") == 90
assert roman_to_int_hard("CD") == 400
assert roman_to_int_hard("CM") == 900
""",
    ))

    tests.append(ExecTest(
        test_id="e24_matrix_transpose",
        prompt="Write a Python function called `transpose(matrix)` that returns the transpose of a 2D list (matrix). The matrix may be non-square. Example: transpose([[1,2,3],[4,5,6]]) returns [[1,4],[2,5],[3,6]].",
        func_name="transpose",
        test_code="""
assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
assert transpose([[1]]) == [[1]]
assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
assert transpose([[1, 2, 3]]) == [[1], [2], [3]]
assert transpose([[1], [2], [3]]) == [[1, 2, 3]]
""",
    ))

    tests.append(ExecTest(
        test_id="e25_bubble_sort",
        prompt="Write a Python function called `bubble_sort(arr)` that sorts a list of numbers in ascending order using the bubble sort algorithm. Return the sorted list. Do not use built-in sort.",
        func_name="bubble_sort",
        test_code="""
assert bubble_sort([5, 3, 1, 4, 2]) == [1, 2, 3, 4, 5]
assert bubble_sort([]) == []
assert bubble_sort([1]) == [1]
assert bubble_sort([3, 2, 1]) == [1, 2, 3]
assert bubble_sort([1, 2, 3]) == [1, 2, 3]
assert bubble_sort([5, 5, 3, 3, 1]) == [1, 3, 3, 5, 5]
""",
    ))

    # ── String/Data Processing (edge case heavy) ────────────

    tests.append(ExecTest(
        test_id="e26_anagram_check",
        prompt="Write a Python function called `is_anagram(s1, s2)` that returns True if s1 and s2 are anagrams of each other. Ignore case and spaces. Example: is_anagram('listen', 'silent') returns True.",
        func_name="is_anagram",
        test_code="""
assert is_anagram("listen", "silent") == True
assert is_anagram("hello", "world") == False
assert is_anagram("Astronomer", "Moon starer") == True
assert is_anagram("", "") == True
assert is_anagram("a", "a") == True
assert is_anagram("ab", "ba") == True
assert is_anagram("abc", "abd") == False
""",
    ))

    tests.append(ExecTest(
        test_id="e27_run_length_encode",
        prompt="Write a Python function called `run_length_encode(s)` that performs run-length encoding on a string. Example: run_length_encode('aaabbc') returns '3a2b1c'. Each group is count followed by the character.",
        func_name="run_length_encode",
        test_code="""
assert run_length_encode("aaabbc") == "3a2b1c"
assert run_length_encode("") == ""
assert run_length_encode("a") == "1a"
assert run_length_encode("aaa") == "3a"
assert run_length_encode("abcd") == "1a1b1c1d"
assert run_length_encode("aabbcc") == "2a2b2c"
""",
    ))

    tests.append(ExecTest(
        test_id="e28_title_case",
        prompt="Write a Python function called `title_case(s)` that converts a string to title case: the first letter of each word is capitalized, the rest are lowercase. Split on whitespace. Example: title_case('hello world') returns 'Hello World'.",
        func_name="title_case",
        test_code="""
assert title_case("hello world") == "Hello World"
assert title_case("HELLO WORLD") == "Hello World"
assert title_case("") == ""
assert title_case("a") == "A"
assert title_case("hello") == "Hello"
assert title_case("foo bar baz") == "Foo Bar Baz"
""",
    ))

    # ── Practical Coding Tasks ──────────────────────────────

    tests.append(ExecTest(
        test_id="e29_remove_duplicates",
        prompt="Write a Python function called `remove_duplicates(lst)` that removes duplicates from a list while preserving the original order. Example: remove_duplicates([1, 2, 2, 3, 1]) returns [1, 2, 3].",
        func_name="remove_duplicates",
        test_code="""
assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]
assert remove_duplicates([]) == []
assert remove_duplicates([1, 1, 1]) == [1]
assert remove_duplicates([1, 2, 3]) == [1, 2, 3]
assert remove_duplicates([3, 2, 1, 2, 3]) == [3, 2, 1]
""",
    ))

    tests.append(ExecTest(
        test_id="e30_chunk_list",
        prompt="Write a Python function called `chunk_list(lst, n)` that splits a list into chunks of size n. The last chunk may be smaller. Example: chunk_list([1,2,3,4,5], 2) returns [[1,2],[3,4],[5]].",
        func_name="chunk_list",
        test_code="""
assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
assert chunk_list([], 3) == []
assert chunk_list([1], 5) == [[1]]
assert chunk_list([1, 2, 3], 1) == [[1], [2], [3]]
assert chunk_list([1, 2, 3], 3) == [[1, 2, 3]]
""",
    ))

    tests.append(ExecTest(
        test_id="e31_deep_merge",
        prompt="Write a Python function called `deep_merge(d1, d2)` that deep merges two dictionaries. Values from d2 override d1. If both values are dicts, merge recursively. Example: deep_merge({'a': 1, 'b': {'c': 2}}, {'b': {'d': 3}}) returns {'a': 1, 'b': {'c': 2, 'd': 3}}.",
        func_name="deep_merge",
        test_code="""
assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}
assert deep_merge({}, {"a": 1}) == {"a": 1}
assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}
assert deep_merge({"a": {"b": 1}}, {"a": {"b": 2}}) == {"a": {"b": 2}}
assert deep_merge({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}}) == {"a": 1, "b": {"c": 2, "d": 3}}
""",
    ))

    # ── Math ────────────────────────────────────────────────

    tests.append(ExecTest(
        test_id="e32_lcm",
        prompt="Write a Python function called `lcm(a, b)` that returns the least common multiple of two non-negative integers. Use the relationship lcm(a,b) = a*b // gcd(a,b). Handle zero: lcm(0, n) = 0.",
        func_name="lcm",
        test_code="""
assert lcm(4, 6) == 12
assert lcm(3, 5) == 15
assert lcm(7, 7) == 7
assert lcm(0, 5) == 0
assert lcm(12, 18) == 36
assert lcm(1, 10) == 10
""",
    ))

    tests.append(ExecTest(
        test_id="e33_nth_prime",
        prompt="Write a Python function called `nth_prime(n)` that returns the nth prime number (1-indexed). nth_prime(1) returns 2, nth_prime(2) returns 3, nth_prime(10) returns 29.",
        func_name="nth_prime",
        test_code="""
assert nth_prime(1) == 2
assert nth_prime(2) == 3
assert nth_prime(3) == 5
assert nth_prime(4) == 7
assert nth_prime(5) == 11
assert nth_prime(10) == 29
assert nth_prime(20) == 71
""",
    ))

    tests.append(ExecTest(
        test_id="e34_count_digits",
        prompt="Write a Python function called `count_digits(n)` that returns the number of digits in an integer. Handle negative numbers (ignore the sign) and zero (which has 1 digit). Example: count_digits(12345) returns 5, count_digits(-42) returns 2.",
        func_name="count_digits",
        test_code="""
assert count_digits(12345) == 5
assert count_digits(0) == 1
assert count_digits(-42) == 2
assert count_digits(9) == 1
assert count_digits(100) == 3
assert count_digits(-1000) == 4
""",
    ))

    tests.append(ExecTest(
        test_id="e35_clamp",
        prompt="Write a Python function called `clamp(value, min_val, max_val)` that clamps a number between min_val and max_val. If value < min_val, return min_val. If value > max_val, return max_val. Otherwise return value.",
        func_name="clamp",
        test_code="""
assert clamp(5, 1, 10) == 5
assert clamp(-5, 0, 100) == 0
assert clamp(150, 0, 100) == 100
assert clamp(0, 0, 0) == 0
assert clamp(1, 1, 1) == 1
assert clamp(50, 50, 100) == 50
assert clamp(100, 50, 100) == 100
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
    mode: str  # "direct", "generic_expert", "tuned_expert"
    total_score: float  # average of test scores
    tests_run: int
    perfect_count: int  # tests with 100% assertions passing
    avg_response_time: float
    avg_tokens_per_second: float
    results: list[ExecTestResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class _ModelShim:
    """Wraps raw llama model to provide .generate() and .count_tokens() for ExpertRouter."""
    def __init__(self, model, max_tokens, temperature):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.total_time = 0
        self.total_tokens = 0

    def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]
        start = time.monotonic()
        # Remove grammar from kwargs if present (not needed for code gen)
        kwargs.pop("grammar", None)
        output = self.model(prompt, max_tokens=mt, temperature=temp, stop=stop, echo=False, **kwargs)
        elapsed = time.monotonic() - start
        self.total_time += elapsed
        text = output["choices"][0]["text"].strip()
        tokens = output.get("usage", {}).get("completion_tokens", max(1, len(text) // 4))
        self.total_tokens += tokens
        return text

    def count_tokens(self, text):
        try:
            return len(self.model.tokenize(text.encode("utf-8")))
        except Exception:
            return len(text) // 4


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

    def run_test(self, model, test: ExecTest, chat_format: str,
                 expert_router=None) -> ExecTestResult:
        """Run a test. If expert_router is provided, use it to build the prompt."""
        if expert_router:
            # Use expert system to build prompt + generate
            from engine.experts import ExpertResult
            shim = _ModelShim(model, self.max_tokens, self.temperature)
            expert_result = expert_router.process(
                query=test.prompt, model=shim, chat_format=chat_format,
                module_hint="code_gen",
                gen_kwargs={"max_tokens": self.max_tokens, "temperature": self.temperature},
            )
            if expert_result:
                response = expert_result.response
                elapsed = shim.total_time
                tokens = shim.total_tokens
            else:
                # Fallback to direct
                system = (
                    "Write a Python function as requested. Return ONLY the code in a ```python block. "
                    "No explanation needed. The function must be complete and runnable."
                )
                prompt = wrap_chat(system, test.prompt, chat_format)
                response, elapsed, tokens = self.generate(model, prompt)
        else:
            system = (
                "Write a Python function as requested. Return ONLY the code in a ```python block. "
                "No explanation needed. The function must be complete and runnable."
            )
            prompt = wrap_chat(system, test.prompt, chat_format)
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

    def run_test_pipeline(self, model, test: ExecTest, chat_format: str) -> ExecTestResult:
        """Run a test through the generate→execute→repair pipeline."""
        from engine.code_pipeline import CodePipeline
        pipeline = CodePipeline(max_retries=2)

        start = time.monotonic()
        result = pipeline.run_single(
            task=test.prompt, model=model, chat_format=chat_format,
            test_code=test.test_code,
            gen_kwargs={"max_tokens": self.max_tokens, "temperature": self.temperature},
        )
        elapsed = time.monotonic() - start

        # Convert to ExecTestResult
        tr = result.test_results or {}
        passed_count = tr.get("passed", 1 if result.passed else 0)
        total_count = tr.get("total", 1)
        test_errors = tr.get("errors", [result.error] if result.error else [])

        return ExecTestResult(
            test_id=test.test_id, prompt=test.prompt,
            response=result.response, extracted_code=result.code,
            exec_error="" if result.passed or tr.get("stage") == "tests" else result.error,
            tests_passed=passed_count, tests_total=total_count,
            test_errors=test_errors, response_time=elapsed,
            score=passed_count / total_count if total_count > 0 else 0.0,
        )

    def run_test_multi(self, generator, debugger, test: ExecTest,
                       gen_format: str, dbg_format: str) -> ExecTestResult:
        """Run a test with multi-model: generate with A, debug with B."""
        from engine.code_pipeline import CodePipeline
        pipeline = CodePipeline(max_retries=2)

        start = time.monotonic()
        result = pipeline.run_multi(
            task=test.prompt,
            generator_model=generator, generator_format=gen_format,
            debugger_model=debugger, debugger_format=dbg_format,
            test_code=test.test_code,
            gen_kwargs={"max_tokens": self.max_tokens, "temperature": self.temperature},
            debug_kwargs={"max_tokens": self.max_tokens, "temperature": self.temperature},
        )
        elapsed = time.monotonic() - start

        tr = result.test_results or {}
        passed_count = tr.get("passed", 1 if result.passed else 0)
        total_count = tr.get("total", 1)
        test_errors = tr.get("errors", [result.error] if result.error else [])

        return ExecTestResult(
            test_id=test.test_id, prompt=test.prompt,
            response=result.response, extracted_code=result.code,
            exec_error="" if result.passed or tr.get("stage") == "tests" else result.error,
            tests_passed=passed_count, tests_total=total_count,
            test_errors=test_errors, response_time=elapsed,
            score=passed_count / total_count if total_count > 0 else 0.0,
        )

    def _run_suite(self, model, model_name, model_path, model_size_mb,
                   chat_format, mode, expert_router=None):
        """Run all tests under one mode."""
        mr = ExecModelResult(
            model_name=model_name, model_path=model_path,
            model_size_mb=model_size_mb, chat_format=chat_format, mode=mode,
            total_score=0, tests_run=0, perfect_count=0,
            avg_response_time=0, avg_tokens_per_second=0,
        )

        all_scores = []
        all_times = []

        for i, test in enumerate(self.tests, 1):
            tag = mode.upper()
            print(f"  [{tag}] ({i}/{len(self.tests)}) {test.test_id}...", end=" ", flush=True)

            if mode == "pipeline":
                tr = self.run_test_pipeline(model, test, chat_format)
            else:
                tr = self.run_test(model, test, chat_format, expert_router=expert_router)
            mr.results.append(tr)
            all_scores.append(tr.score)
            all_times.append(tr.response_time)
            mr.tests_run += 1

            if tr.exec_error:
                print(f"EXEC_ERR ({tr.response_time:.1f}s)")
                print(f"         {tr.exec_error[:100]}")
            elif tr.score == 1.0:
                print(f"PERFECT {tr.tests_passed}/{tr.tests_total} ({tr.response_time:.1f}s)")
                mr.perfect_count += 1
            else:
                print(f"PARTIAL {tr.tests_passed}/{tr.tests_total} ({tr.score:.0%}) ({tr.response_time:.1f}s)")
                for err in tr.test_errors[:2]:
                    print(f"         {err[:100]}")

        mr.total_score = sum(all_scores) / len(all_scores) * 100 if all_scores else 0
        mr.avg_response_time = sum(all_times) / len(all_times) if all_times else 0

        print(f"\n  [{mode.upper()}] RESULTS: {model_name}")
        print(f"    Execution Score: {mr.total_score:.1f}%")
        print(f"    Perfect tests:   {mr.perfect_count}/{mr.tests_run}")
        print(f"    Avg time:        {mr.avg_response_time:.1f}s")

        self.all_results.append(mr)
        return mr

    def run_model(self, model_path: Path, modes: list[str] = None) -> list[ExecModelResult]:
        """Run tests in specified modes. modes: 'direct', 'generic', 'tuned'."""
        if modes is None:
            modes = ["direct"]

        chat_format = detect_chat_format(str(model_path))
        model_name = model_path.stem
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"  Size: {model_size_mb:.0f} MB | Format: {chat_format} | Tests: {len(self.tests)}")
        print(f"  Modes: {', '.join(modes)}")
        print(f"{'='*70}")

        try:
            model = self.load_model(model_path)
        except Exception as e:
            print(f"  FAILED TO LOAD: {e}")
            return []

        results = []
        for mode in modes:
            expert_router = None
            if mode in ("generic", "tuned"):
                from engine.experts import ExpertRouter
                expert_router = ExpertRouter(tuned=(mode == "tuned"))

            mr = self._run_suite(model, model_name, str(model_path),
                                 model_size_mb, chat_format, mode, expert_router)
            results.append(mr)

        del model
        gc.collect()
        print(f"\n  Model unloaded.")

        return results

    def run_multi_model(self, gen_path: Path, dbg_path: Path) -> ExecModelResult:
        """Run multi-model benchmark: generate with gen_path, debug with dbg_path."""
        gen_format = detect_chat_format(str(gen_path))
        dbg_format = detect_chat_format(str(dbg_path))
        gen_name = gen_path.stem
        dbg_name = dbg_path.stem
        combo_name = f"{gen_name}+{dbg_name}"
        total_mb = gen_path.stat().st_size / (1024*1024) + dbg_path.stat().st_size / (1024*1024)

        print(f"\n{'='*70}")
        print(f"MULTI-MODEL: {gen_name} (gen) + {dbg_name} (debug)")
        print(f"  Total size: {total_mb:.0f} MB | Tests: {len(self.tests)}")
        print(f"{'='*70}")

        try:
            generator = self.load_model(gen_path)
            debugger = self.load_model(dbg_path)
        except Exception as e:
            print(f"  FAILED TO LOAD: {e}")
            return None

        mr = ExecModelResult(
            model_name=combo_name, model_path=f"{gen_path}+{dbg_path}",
            model_size_mb=total_mb, chat_format=f"{gen_format}+{dbg_format}",
            mode="multi",
            total_score=0, tests_run=0, perfect_count=0,
            avg_response_time=0, avg_tokens_per_second=0,
        )

        all_scores = []
        all_times = []

        for i, test in enumerate(self.tests, 1):
            print(f"  [MULTI] ({i}/{len(self.tests)}) {test.test_id}...", end=" ", flush=True)

            tr = self.run_test_multi(generator, debugger, test, gen_format, dbg_format)
            mr.results.append(tr)
            all_scores.append(tr.score)
            all_times.append(tr.response_time)
            mr.tests_run += 1

            if tr.exec_error:
                print(f"EXEC_ERR ({tr.response_time:.1f}s)")
                print(f"         {tr.exec_error[:100]}")
            elif tr.score == 1.0:
                print(f"PERFECT {tr.tests_passed}/{tr.tests_total} ({tr.response_time:.1f}s)")
                mr.perfect_count += 1
            else:
                print(f"PARTIAL {tr.tests_passed}/{tr.tests_total} ({tr.score:.0%}) ({tr.response_time:.1f}s)")
                for err in tr.test_errors[:2]:
                    print(f"         {err[:100]}")

        mr.total_score = sum(all_scores) / len(all_scores) * 100 if all_scores else 0
        mr.avg_response_time = sum(all_times) / len(all_times) if all_times else 0

        print(f"\n  [MULTI] RESULTS: {combo_name}")
        print(f"    Execution Score: {mr.total_score:.1f}%")
        print(f"    Perfect tests:   {mr.perfect_count}/{mr.tests_run}")
        print(f"    Avg time:        {mr.avg_response_time:.1f}s")

        del generator, debugger
        gc.collect()
        print(f"\n  Models unloaded.")

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

    print(f"  {'Model':<38s} {'Mode':<8s} {'Score':>6s} {'Perfect':>8s} {'Size':>7s} {'Time':>6s}")
    print(f"  {'-'*38} {'-'*8} {'-'*6} {'-'*8} {'-'*7} {'-'*6}")

    for r in sorted_r:
        name = r.model_name if len(r.model_name) <= 38 else r.model_name[:35] + "..."
        print(f"  {name:<38s} {r.mode:<8s} {r.total_score:>5.1f}% "
              f"{r.perfect_count:>3d}/{r.tests_run:<3d}  "
              f"{r.model_size_mb:>5.0f}MB {r.avg_response_time:>5.1f}s")

    # Expert impact comparison
    model_names = sorted(set(r.model_name for r in results))
    modes_present = sorted(set(r.mode for r in results))
    if len(modes_present) > 1:
        print(f"\n  EXPERT IMPACT:")
        print(f"  {'-'*60}")
        for name in model_names:
            scores = {r.mode: r.total_score for r in results if r.model_name == name}
            if len(scores) > 1:
                parts = [f"{mode}: {score:.1f}%" for mode, score in sorted(scores.items())]
                short = name if len(name) <= 35 else name[:32] + "..."
                print(f"  {short:<38s} {' | '.join(parts)}")

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
    parser.add_argument("--tuned", action="store_true", help="Run with tuned experts")
    parser.add_argument("--expert", action="store_true", help="Run with generic experts")
    parser.add_argument("--pipeline", action="store_true", help="Run with test-execute-retry pipeline")
    parser.add_argument("--compare", action="store_true", help="Run all modes: direct, pipeline")
    parser.add_argument("--compare-all", action="store_true", help="Run ALL modes: direct, generic, tuned, pipeline")
    parser.add_argument("--direct-only", action="store_true", help="Only run direct mode")
    parser.add_argument("--multi", nargs=2, metavar=("GEN_MODEL", "DBG_MODEL"),
                        help="Multi-model: generate with GEN, debug with DBG")
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

    # Determine modes
    if args.compare_all:
        modes = ["direct", "generic", "tuned", "pipeline"]
    elif args.compare:
        modes = ["direct", "pipeline"]
    elif args.direct_only:
        modes = ["direct"]
    else:
        modes = ["direct"]
        if args.expert:
            modes.append("generic")
        if args.tuned:
            modes.append("tuned")
        if args.pipeline:
            modes.append("pipeline")

    if args.model:
        models = [Path(args.model)]
    elif args.all:
        models = discover_models(all_models=True)
    else:
        models = discover_models(all_models=False)

    if not models:
        print("No models found.")
        sys.exit(1)

    print(f"\nExecution Benchmark — {len(tests)} tests, {len(models)} models, modes: {modes}")
    for m in models:
        print(f"  - {m.name} ({m.stat().st_size / (1024*1024):.0f} MB)")

    runner = ExecBenchmarkRunner(
        tests=tests, max_tokens=args.max_tokens, temperature=args.temperature,
        gpu_layers=args.gpu_layers, threads=args.threads,
    )

    # Run multi-model if specified
    if args.multi:
        gen_path = Path(args.multi[0])
        dbg_path = Path(args.multi[1])
        if not gen_path.exists():
            print(f"Generator model not found: {gen_path}")
            sys.exit(1)
        if not dbg_path.exists():
            print(f"Debugger model not found: {dbg_path}")
            sys.exit(1)
        try:
            runner.run_multi_model(gen_path, dbg_path)
        except Exception as e:
            print(f"\n  FATAL: {e}")
            traceback.print_exc()
    else:
        for model_path in models:
            try:
                runner.run_model(model_path, modes=modes)
            except Exception as e:
                print(f"\n  FATAL: {model_path.name}: {e}")
                traceback.print_exc()

    print_summary(runner.all_results)
    save_results(runner.all_results, PROJECT_ROOT / "benchmark_exec_results.json")


if __name__ == "__main__":
    main()
