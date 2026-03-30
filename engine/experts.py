"""
Expert System — Code-Focused Experts

Branched from PIE's expert system. Stripped NPC experts, expanded code expertise.

1. DYNAMIC FEW-SHOT RETRIEVAL — finds similar solved examples
2. OUTPUT SCAFFOLDING — forces structured reasoning
3. VERIFICATION LOOP — checks output, retries with feedback
4. MINIMAL TOKEN FOOTPRINT — examples, not verbose instructions
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SolvedExample:
    """A solved example in the expert's knowledge bank."""
    query: str
    solution: str
    category: str = ""
    embedding: Optional[list[float]] = None

    def format_for_prompt(self) -> str:
        return f"Q: {self.query}\nA: {self.solution}"


@dataclass
class ExpertResult:
    """Result from an expert's processing."""
    response: str
    expert_name: str
    attempts: int = 1
    verified: bool = False
    scaffolding_used: bool = False
    examples_injected: int = 0
    prompt_tokens: int = 0


class Expert:
    """
    A single domain expert with:
    - Embedded example bank for few-shot retrieval
    - Output scaffolding template
    - Verification function
    - Minimal system context
    """

    def __init__(self, name: str, system_context: str,
                 examples: list[SolvedExample],
                 scaffolding: Optional[str] = None,
                 verifier: Optional[Callable] = None,
                 grammar_str: Optional[str] = None,
                 max_examples: int = 3,
                 max_retries: int = 2):
        self.name = name
        self.system_context = system_context
        self.examples = examples
        self.scaffolding = scaffolding
        self.verifier = verifier
        self.grammar_str = grammar_str
        self.max_examples = max_examples
        self.max_retries = max_retries

        self._example_embeddings = None
        self._embedder = None

    def init_embeddings(self, embedder):
        """Pre-embed all examples for fast retrieval."""
        self._embedder = embedder
        if not self.examples:
            return

        texts = [ex.query for ex in self.examples]
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._example_embeddings = embeddings
        for i, ex in enumerate(self.examples):
            ex.embedding = embeddings[i].tolist()

        logger.debug(f"Expert '{self.name}': embedded {len(self.examples)} examples")

    def retrieve_examples(self, query: str, top_k: Optional[int] = None) -> list[SolvedExample]:
        """Find the most relevant solved examples for a query.

        Uses category-aware retrieval: picks the best match, then fills
        remaining slots preferring examples from the same category. Applies
        a minimum similarity threshold to avoid injecting misleading examples.
        """
        if not self._embedder or self._example_embeddings is None or len(self.examples) == 0:
            return self.examples[:self.max_examples]

        k = min(top_k or self.max_examples, len(self.examples))
        query_vec = self._embedder.encode([query], normalize_embeddings=True)

        sims = np.dot(self._example_embeddings, query_vec.T).flatten()

        # Minimum similarity threshold — don't inject irrelevant examples
        min_sim = 0.35
        ranked = sims.argsort()[::-1]

        # Best match sets the target category
        best_idx = ranked[0]
        if sims[best_idx] < min_sim:
            return []

        best_category = self.examples[best_idx].category
        selected = [best_idx]

        # Fill remaining slots: prefer same category, then highest similarity
        same_cat = [i for i in ranked[1:] if sims[i] >= min_sim
                    and self.examples[i].category == best_category
                    and i not in selected]
        other = [i for i in ranked[1:] if sims[i] >= min_sim
                 and self.examples[i].category != best_category
                 and i not in selected]

        for i in same_cat:
            if len(selected) >= k:
                break
            selected.append(i)
        for i in other:
            if len(selected) >= k:
                break
            selected.append(i)

        return [self.examples[i] for i in selected]

    def build_prompt(self, user_input: str, chat_format: str) -> str:
        """Build a minimal, high-signal prompt."""
        parts = [self.system_context.strip()]

        examples = self.retrieve_examples(user_input)
        if examples:
            parts.append("")
            for ex in examples:
                parts.append(ex.format_for_prompt())

        system_block = "\n".join(parts)

        if self.scaffolding:
            user_block = f"{user_input}\n\n{self.scaffolding}"
        else:
            user_block = user_input

        return _wrap_expert_chat(system_block, user_block, chat_format)

    def build_retry_prompt(self, user_input: str, previous_response: str,
                           error_hint: str, chat_format: str) -> str:
        """Build a retry prompt with feedback from the failed attempt."""
        parts = [self.system_context.strip()]

        examples = self.retrieve_examples(user_input, top_k=2)
        if examples:
            parts.append("")
            for ex in examples:
                parts.append(ex.format_for_prompt())

        system_block = "\n".join(parts)

        user_block = (
            f"{user_input}\n\n"
            f"Your previous answer was: {previous_response.strip()}\n"
            f"That answer is wrong. {error_hint}\n"
            f"Try again carefully."
        )
        if self.scaffolding:
            user_block += f"\n\n{self.scaffolding}"

        return _wrap_expert_chat(system_block, user_block, chat_format)

    def verify(self, response: str, query: str) -> tuple[bool, str]:
        """Verify the response using the expert's verifier."""
        if self.verifier is None:
            return True, ""
        try:
            return self.verifier(response, query)
        except Exception as e:
            logger.debug(f"Verifier error: {e}")
            return True, ""


def _wrap_expert_chat(system: str, user: str, chat_format: str) -> str:
    """Minimal chat wrapping."""
    if chat_format == "chatml":
        return (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    elif chat_format == "phi3":
        return (f"<|system|>\n{system}<|end|>\n"
                f"<|user|>\n{user}<|end|>\n"
                f"<|assistant|>\n")
    elif chat_format == "llama3":
        return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    else:
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"


# ── Verifiers ─────────────────────────────────────────────────

def verify_code_gen(response: str, query: str) -> tuple[bool, str]:
    """Verify generated code has proper structure."""
    r = response.strip()

    has_code_block = "```" in r
    has_def = "def " in r or "function " in r or "class " in r or "const " in r or "fn " in r
    has_return = "return" in r.lower() or "print" in r.lower() or "console" in r.lower()

    if not has_code_block and not has_def:
        return False, "Response should contain code in a ```language block or a function/class definition."

    if has_def and not has_return:
        if "def " in r or "function " in r:
            return False, "Function should have a return statement or produce output."

    return True, ""


def verify_code_review(response: str, query: str) -> tuple[bool, str]:
    """Verify code review identifies issues and shows fixes."""
    r = response.lower()

    has_feedback = any(w in r for w in [
        "bug", "issue", "error", "fix", "improve", "change", "should",
        "problem", "refactor", "suggest", "looks good", "clean", "correct",
    ])

    if not has_feedback:
        return False, "Review should identify specific issues or confirm code quality."

    return True, ""


def verify_debug(response: str, query: str) -> tuple[bool, str]:
    """Verify debug response identifies root cause and provides fix."""
    r = response.lower()

    has_diagnosis = any(w in r for w in [
        "because", "cause", "reason", "the issue", "the problem", "the bug",
        "the error", "happens when", "occurs when", "due to",
    ])

    has_fix = "```" in response or "def " in response or "fix" in r or "change" in r

    if not has_diagnosis and not has_fix:
        return False, "Identify the root cause and show the fix."

    return True, ""


def verify_explanation(response: str, query: str) -> tuple[bool, str]:
    """Verify explanation has substance."""
    if len(response.strip()) < 50:
        return False, "Explanation is too brief. Provide more detail with examples."
    return True, ""


# ── Expert Definitions ────────────────────────────────────────

def build_code_gen_expert() -> Expert:
    """Expert for code generation — expanded with diverse examples."""
    examples = [
        SolvedExample(
            "Write a function that reverses a string.",
            '```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```',
            category="basic",
        ),
        SolvedExample(
            "Write a function that finds the maximum in a list.",
            '```python\ndef find_max(lst: list) -> int | None:\n    if not lst:\n        return None\n    result = lst[0]\n    for x in lst[1:]:\n        if x > result:\n            result = x\n    return result\n```',
            category="basic",
        ),
        SolvedExample(
            "Write a function to check if a string is a palindrome.",
            '```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = s.lower().strip()\n    return cleaned == cleaned[::-1]\n```',
            category="string",
        ),
        SolvedExample(
            "Write a function that counts word frequency in a string.",
            '```python\ndef word_frequency(text: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n```',
            category="data",
        ),
        SolvedExample(
            "Write a function to read a JSON file and return the data.",
            '```python\nimport json\nfrom pathlib import Path\n\n\ndef read_json(filepath: str) -> dict:\n    return json.loads(Path(filepath).read_text())\n```',
            category="io",
        ),
        SolvedExample(
            "Write a binary search function.",
            '```python\ndef binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```',
            category="algorithm",
        ),
        SolvedExample(
            "Write a simple stack class.",
            '```python\nclass Stack:\n    def __init__(self) -> None:\n        self._items: list = []\n\n    def push(self, item) -> None:\n        self._items.append(item)\n\n    def pop(self):\n        if not self._items:\n            raise IndexError("pop from empty stack")\n        return self._items.pop()\n\n    def peek(self):\n        if not self._items:\n            raise IndexError("peek at empty stack")\n        return self._items[-1]\n\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n\n    def __len__(self) -> int:\n        return len(self._items)\n```',
            category="data_structure",
        ),
        SolvedExample(
            "Write a function that fetches JSON from a URL.",
            '```python\nimport urllib.request\nimport json\n\n\ndef fetch_json(url: str) -> dict:\n    with urllib.request.urlopen(url) as resp:\n        return json.loads(resp.read().decode())\n```',
            category="api",
        ),
        SolvedExample(
            "Write a retry decorator that retries a function up to 3 times.",
            '```python\nimport functools\nimport time\n\n\ndef retry(max_attempts: int = 3, delay: float = 1.0):\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            for attempt in range(max_attempts):\n                try:\n                    return func(*args, **kwargs)\n                except Exception:\n                    if attempt == max_attempts - 1:\n                        raise\n                    time.sleep(delay)\n        return wrapper\n    return decorator\n```',
            category="pattern",
        ),
        SolvedExample(
            "Write a function that parses command line arguments for a file converter.",
            '```python\nimport argparse\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description="File converter")\n    parser.add_argument("input", help="Input file path")\n    parser.add_argument("-o", "--output", help="Output file path")\n    parser.add_argument("-f", "--format", choices=["json", "csv", "yaml"], default="json")\n    return parser.parse_args()\n```',
            category="cli",
        ),
    ]

    return Expert(
        name="code_gen",
        system_context="Write clean, working code. Use ```language blocks. Include type hints. Handle edge cases.",
        examples=examples,
        verifier=verify_code_gen,
        max_examples=3,
        max_retries=1,
    )


def build_code_review_expert() -> Expert:
    """Expert for code review."""
    examples = [
        SolvedExample(
            "Review this:\ndef add(a, b):\n    return a + b",
            "The function works but could be improved:\n\n```python\ndef add(a: float, b: float) -> float:\n    return a + b\n```\n\nChanges: Added type hints for clarity. The logic is correct and simple.",
            category="simple",
        ),
        SolvedExample(
            "Review this:\ndef read_file(path):\n    f = open(path)\n    data = f.read()\n    return data",
            "Bug: File handle is never closed. This leaks file descriptors.\n\n```python\ndef read_file(path: str) -> str:\n    with open(path) as f:\n        return f.read()\n```\n\nFix: Use `with` statement to ensure the file is always closed.",
            category="bug",
        ),
        SolvedExample(
            "Review this:\nimport os\ndef delete_file(user_input):\n    os.remove('/data/' + user_input)",
            "Critical: Path traversal vulnerability. User can pass `../../etc/passwd` to delete arbitrary files.\n\n```python\nfrom pathlib import Path\n\ndef delete_file(filename: str) -> None:\n    base = Path('/data')\n    target = (base / filename).resolve()\n    if not target.is_relative_to(base):\n        raise ValueError(\"Invalid path\")\n    target.unlink()\n```\n\nFix: Resolve the path and verify it stays within the allowed directory.",
            category="security",
        ),
        SolvedExample(
            "Review this:\ndef find_duplicates(lst):\n    dupes = []\n    for i in range(len(lst)):\n        for j in range(i+1, len(lst)):\n            if lst[i] == lst[j] and lst[i] not in dupes:\n                dupes.append(lst[i])\n    return dupes",
            "Performance: O(n^3) due to nested loops + `in` check on list. Can be O(n):\n\n```python\ndef find_duplicates(lst: list) -> list:\n    seen: set = set()\n    dupes: set = set()\n    for item in lst:\n        if item in seen:\n            dupes.add(item)\n        seen.add(item)\n    return list(dupes)\n```\n\nFix: Use sets for O(1) lookups. Reduces from O(n^3) to O(n).",
            category="performance",
        ),
    ]

    return Expert(
        name="code_review",
        system_context="Review code for bugs, security issues, performance, and quality. Show the fixed version.",
        examples=examples,
        verifier=verify_code_review,
        max_examples=2,
        max_retries=1,
    )


def build_debug_expert() -> Expert:
    """Expert for debugging."""
    examples = [
        SolvedExample(
            "My function returns None instead of the result:\ndef process(data):\n    for item in data:\n        result = item * 2\n    return result",
            "Bug: If `data` is empty, `result` is never assigned, causing `NameError` (or returning the last loop value only).\n\n```python\ndef process(data: list) -> list:\n    return [item * 2 for item in data]\n```\n\nRoot cause: The function returns only the last item's result, not all of them. If you want all results, use a list comprehension.",
            category="logic",
        ),
        SolvedExample(
            "IndexError: list index out of range\ndef get_pairs(lst):\n    pairs = []\n    for i in range(len(lst)):\n        pairs.append((lst[i], lst[i+1]))\n    return pairs",
            "Bug: Off-by-one. When `i` is the last index, `lst[i+1]` is out of bounds.\n\n```python\ndef get_pairs(lst: list) -> list[tuple]:\n    return [(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]\n```\n\nFix: Loop to `len(lst) - 1` to avoid accessing beyond the list end.",
            category="off_by_one",
        ),
        SolvedExample(
            "TypeError: can only concatenate str to str\nage = 25\nprint('Age: ' + age)",
            "Bug: Cannot concatenate `str` and `int` directly.\n\n```python\nage = 25\nprint(f'Age: {age}')\n```\n\nFix: Use f-string (or `str(age)`) to convert the int to string.",
            category="type_error",
        ),
        SolvedExample(
            "My dict comprehension gives wrong results:\ncounts = {}\nfor word in words:\n    counts[word] = counts[word] + 1",
            "Bug: `KeyError` on first occurrence of each word since key doesn't exist yet.\n\n```python\ncounts: dict[str, int] = {}\nfor word in words:\n    counts[word] = counts.get(word, 0) + 1\n```\n\nFix: Use `.get(word, 0)` to default to 0 for missing keys. Or use `collections.Counter(words)`.",
            category="key_error",
        ),
        SolvedExample(
            "My function modifies the original list when I don't want it to:\ndef process(items=[]):\n    items.append('done')\n    return items",
            "Bug: Mutable default argument. The default `[]` is shared across all calls.\n\n```python\ndef process(items: list | None = None) -> list:\n    if items is None:\n        items = []\n    items.append('done')\n    return items\n```\n\nFix: Use `None` as default and create a new list inside the function.",
            category="mutable_default",
        ),
    ]

    return Expert(
        name="debugger",
        system_context="Diagnose the bug. Explain the root cause first, then show the fix.",
        examples=examples,
        verifier=verify_debug,
        max_examples=2,
        max_retries=1,
    )


def build_explainer_expert() -> Expert:
    """Expert for explaining code and concepts."""
    examples = [
        SolvedExample(
            "Explain list comprehensions in Python.",
            "List comprehensions are a concise way to create lists:\n\n```python\n# Traditional loop\nsquares = []\nfor x in range(5):\n    squares.append(x ** 2)\n\n# List comprehension — same result\nsquares = [x ** 2 for x in range(5)]\n\n# With filter\nevens = [x for x in range(10) if x % 2 == 0]\n\n# Nested\npairs = [(x, y) for x in range(3) for y in range(3)]\n```\n\nPattern: `[expression for item in iterable if condition]`",
            category="concept",
        ),
        SolvedExample(
            "What's the difference between a list and a tuple?",
            "```python\n# List — mutable, use for collections that change\nitems = [1, 2, 3]\nitems.append(4)        # OK\nitems[0] = 10          # OK\n\n# Tuple — immutable, use for fixed records\npoint = (3, 4)\npoint[0] = 10          # TypeError!\n```\n\nKey differences:\n- Lists are mutable, tuples are immutable\n- Tuples are hashable (can be dict keys), lists are not\n- Tuples are slightly faster and use less memory\n- Use tuples for fixed data (coordinates, DB rows), lists for collections",
            category="comparison",
        ),
        SolvedExample(
            "Explain what this code does:\nresult = {k: v for k, v in sorted(d.items(), key=lambda x: x[1])}",
            "This sorts a dictionary by its values:\n\n```python\nd = {'b': 3, 'a': 1, 'c': 2}\n\n# Step by step:\nd.items()              # [('b', 3), ('a', 1), ('c', 2)]\nsorted(..., key=lambda x: x[1])  # Sort by value (index 1)\n# Result: [('a', 1), ('c', 2), ('b', 3)]\n\nresult = {k: v for k, v in sorted(d.items(), key=lambda x: x[1])}\n# result = {'a': 1, 'c': 2, 'b': 3}\n```\n\nIt creates a new dict with keys ordered by their values (ascending).",
            category="code_walkthrough",
        ),
    ]

    return Expert(
        name="explainer",
        system_context="Explain code and concepts clearly with concrete examples. Code first, then explain.",
        examples=examples,
        verifier=verify_explanation,
        max_examples=2,
        max_retries=1,
    )


# ── Tuned Expert Definitions ─────────────────────────────────

def build_tuned_code_gen_expert() -> Expert:
    """
    Tuned code_gen expert with algorithm-specific few-shot examples.

    Targets the exact failure modes observed in benchmarks:
    - Fibonacci: wrong base cases / 1-indexed confusion
    - Roman numerals: missing subtractive logic (IV, IX, XL)
    - Kadane's algorithm: broken reset-vs-continue on all-negatives
    - Matrix multiply: syntax errors and wrong dimension checks
    - Stack class: missing IndexError on empty pop
    """
    examples = [
        # ── Benchmark-targeted examples (correct patterns) ───────
        SolvedExample(
            "Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number. fibonacci(0)=0, fibonacci(1)=1.",
            '```python\ndef fibonacci(n: int) -> int:\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```',
            category="algorithm",
        ),
        SolvedExample(
            "Write a Python function called `roman_to_int(s)` that converts a Roman numeral string to an integer. Handle subtractive cases like IV=4, IX=9.",
            '```python\ndef roman_to_int(s: str) -> int:\n    values = {\'I\': 1, \'V\': 5, \'X\': 10, \'L\': 50, \'C\': 100, \'D\': 500, \'M\': 1000}\n    result = 0\n    for i in range(len(s)):\n        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:\n            result -= values[s[i]]\n        else:\n            result += values[s[i]]\n    return result\n```',
            category="algorithm",
        ),
        SolvedExample(
            "Write a Python function called `max_subarray(nums)` that returns the largest sum of a contiguous subarray using Kadane's algorithm.",
            '```python\ndef max_subarray(nums: list[int]) -> int:\n    max_sum = nums[0]\n    current = nums[0]\n    for num in nums[1:]:\n        current = max(num, current + num)\n        max_sum = max(max_sum, current)\n    return max_sum\n```',
            category="algorithm",
        ),
        SolvedExample(
            "Write a Python function called `matrix_multiply(a, b)` that multiplies two matrices. Raise ValueError if dimensions don't match.",
            '```python\ndef matrix_multiply(a: list[list], b: list[list]) -> list[list]:\n    if not a or not b or len(a[0]) != len(b):\n        raise ValueError("Incompatible dimensions")\n    rows_a, cols_b, cols_a = len(a), len(b[0]), len(a[0])\n    result = [[0] * cols_b for _ in range(rows_a)]\n    for i in range(rows_a):\n        for j in range(cols_b):\n            for k in range(cols_a):\n                result[i][j] += a[i][k] * b[k][j]\n    return result\n```',
            category="algorithm",
        ),
        SolvedExample(
            "Write a Python function called `two_sum(nums, target)` that returns indices of two numbers that add up to target.",
            '```python\ndef two_sum(nums: list[int], target: int) -> list[int]:\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []\n```',
            category="algorithm",
        ),
        # ── Stack class (targets empty-pop IndexError miss) ──────
        SolvedExample(
            "Write a simple stack class.",
            '```python\nclass Stack:\n    def __init__(self) -> None:\n        self._items: list = []\n\n    def push(self, item) -> None:\n        self._items.append(item)\n\n    def pop(self):\n        if not self._items:\n            raise IndexError("pop from empty stack")\n        return self._items.pop()\n\n    def peek(self):\n        if not self._items:\n            raise IndexError("peek at empty stack")\n        return self._items[-1]\n\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n\n    def __len__(self) -> int:\n        return len(self._items)\n```',
            category="data_structure",
        ),
        # ── Additional general examples ──────────────────────────
        SolvedExample(
            "Write a function to check if a string has valid parentheses.",
            '```python\ndef valid_parentheses(s: str) -> bool:\n    stack: list[str] = []\n    pairs = {\')\': \'(\', \']\': \'[\', \'}\': \'{\'}\n    for ch in s:\n        if ch in \'([{\':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack[-1] != pairs[ch]:\n                return False\n            stack.pop()\n    return len(stack) == 0\n```',
            category="algorithm",
        ),
        SolvedExample(
            "Write a binary search function.",
            '```python\ndef binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```',
            category="algorithm",
        ),
        SolvedExample(
            "Write a Caesar cipher function that shifts letters by a given amount.",
            '```python\ndef caesar_cipher(text: str, shift: int) -> str:\n    result: list[str] = []\n    for ch in text:\n        if ch.isalpha():\n            base = ord(\'A\') if ch.isupper() else ord(\'a\')\n            result.append(chr((ord(ch) - base + shift) % 26 + base))\n        else:\n            result.append(ch)\n    return \'\'.join(result)\n```',
            category="algorithm",
        ),
    ]

    return Expert(
        name="code_gen",
        system_context=(
            "Write clean, working code. Use ```language blocks. Include type hints. "
            "Handle edge cases. Pay attention to 0-indexed vs 1-indexed conventions. "
            "For algorithms, initialize correctly and handle empty/negative inputs."
        ),
        examples=examples,
        verifier=verify_code_gen,
        max_examples=3,
        max_retries=1,
    )


def build_stress_code_gen_expert() -> Expert:
    """
    Stress-targeted code_gen expert with few-shot examples for patterns
    that sub-3B models universally fail on:

    - Decorator with function attributes (.attempts tracking)
    - Class with mutable history list (not a read-only property)
    - Recursive-descent expression evaluator with operator precedence
    - Event emitter with on/off/once/emit
    - Metaclass-free ORM with classmethods
    - Decorator-based HTTP router with path param extraction
    - Markdown-to-HTML string processing
    - State machine with guards and history
    - Token bucket rate limiter with explicit time
    """
    examples = [
        # ── Decorator with function attributes ──────────────────
        SolvedExample(
            "Write a Python decorator called `count_calls(func)` that counts how many "
            "times the decorated function is called. Store the count as `func.call_count`.",
            '```python\nimport functools\n\n\ndef count_calls(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        wrapper.call_count += 1\n        return func(*args, **kwargs)\n    wrapper.call_count = 0\n    return wrapper\n```\n\nKey: set attributes on `wrapper`, not `func`. The wrapper IS the new function object.',
            category="pattern_decorator",
        ),
        # ── Retry decorator with attempts attribute ─────────────
        SolvedExample(
            "Write a decorator `retry(max_retries=3, exceptions=(Exception,))` that retries a "
            "function and tracks attempts in `func.attempts`.",
            '```python\nimport functools\nimport time\n\n\ndef retry(max_attempts=3, exceptions=(Exception,), backoff=0):\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            last_exc = None\n            for attempt in range(1, max_attempts + 1):\n                try:\n                    result = func(*args, **kwargs)\n                    wrapper.attempts = attempt\n                    return result\n                except exceptions as e:\n                    last_exc = e\n                    wrapper.attempts = attempt\n                    if backoff and attempt < max_attempts:\n                        time.sleep(backoff)\n            raise last_exc\n        wrapper.attempts = 0\n        return wrapper\n    return decorator\n```\n\nKey patterns: triple-nested (factory -> decorator -> wrapper), set `wrapper.attempts` attribute, re-raise last exception after all retries.',
            category="pattern_decorator",
        ),
        # ── State machine with history list ─────────────────────
        SolvedExample(
            "Write a class `TrafficLight` that cycles through states with a `history` list and guard functions.",
            '```python\nclass TrafficLight:\n    def __init__(self, initial):\n        self._state = initial\n        self._history = [initial]\n        self._transitions = {}\n\n    @property\n    def state(self):\n        return self._state\n\n    @property\n    def history(self):\n        return list(self._history)\n\n    def add_transition(self, from_state, event, to_state, guard=None):\n        self._transitions[(from_state, event)] = (to_state, guard)\n\n    def trigger(self, event):\n        key = (self._state, event)\n        if key not in self._transitions:\n            raise ValueError(f"No transition for {event} from {self._state}")\n        to_state, guard = self._transitions[key]\n        if guard is not None and not guard():\n            raise ValueError(f"Guard blocked {event}")\n        self._state = to_state\n        self._history.append(to_state)\n        return self._state\n```\n\nKey: `history` is a property returning a COPY of `_history` list. Use `_history` internally to append. Guards are optional callables.',
            category="pattern_state_machine",
        ),
        # ── Expression evaluator with operator precedence ───────
        SolvedExample(
            "Write a function `calc(expr)` that evaluates math expressions with +, -, *, / "
            "and parentheses, respecting operator precedence.",
            '```python\ndef calc(expr: str) -> float:\n    tokens = _tokenize(expr)\n    pos = [0]\n\n    def parse_expr():\n        result = parse_term()\n        while pos[0] < len(tokens) and tokens[pos[0]] in ("+", "-"):\n            op = tokens[pos[0]]\n            pos[0] += 1\n            right = parse_term()\n            result = result + right if op == "+" else result - right\n        return result\n\n    def parse_term():\n        result = parse_factor()\n        while pos[0] < len(tokens) and tokens[pos[0]] in ("*", "/"):\n            op = tokens[pos[0]]\n            pos[0] += 1\n            right = parse_factor()\n            result = result * right if op == "*" else result / right\n        return result\n\n    def parse_factor():\n        token = tokens[pos[0]]\n        if token == "(":\n            pos[0] += 1\n            result = parse_expr()\n            pos[0] += 1  # skip ")"\n            return result\n        pos[0] += 1\n        return float(token)\n\n    return parse_expr()\n\n\ndef _tokenize(expr: str) -> list[str]:\n    tokens = []\n    i = 0\n    while i < len(expr):\n        if expr[i].isspace():\n            i += 1\n        elif expr[i] in "+-*/()":\n            tokens.append(expr[i])\n            i += 1\n        else:\n            j = i\n            while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):\n                j += 1\n            tokens.append(expr[i:j])\n            i = j\n    return tokens\n```\n\nKey: recursive descent with 3 levels (expr -> term -> factor). expr handles +/-, term handles *// , factor handles numbers and parentheses.',
            category="pattern_parser",
        ),
        # ── Event emitter pattern ───────────────────────────────
        SolvedExample(
            "Write an `EventBus` class with on, off, emit, and once methods.",
            '```python\nclass EventBus:\n    def __init__(self):\n        self._listeners = {}\n\n    def on(self, event, callback):\n        if event not in self._listeners:\n            self._listeners[event] = []\n        self._listeners[event].append(callback)\n\n    def off(self, event, callback):\n        if event in self._listeners:\n            self._listeners[event] = [\n                cb for cb in self._listeners[event] if cb != callback\n            ]\n\n    def emit(self, event, *args, **kwargs):\n        for cb in self._listeners.get(event, []):\n            cb(*args, **kwargs)\n\n    def once(self, event, callback):\n        def wrapper(*args, **kwargs):\n            callback(*args, **kwargs)\n            self.off(event, wrapper)\n        self.on(event, wrapper)\n\n    def listener_count(self, event):\n        return len(self._listeners.get(event, []))\n```\n\nKey: `once` wraps callback, auto-removes via `off`. `emit` calls a COPY or iterates safely. `listener_count` returns 0 for unknown events.',
            category="pattern_event",
        ),
        # ── Mini ORM with classmethods ──────────────────────────
        SolvedExample(
            "Write a `Model` base class and `Field` descriptor for a simple ORM that generates SQL.",
            (
                "```python\n"
                "class Field:\n"
                "    def __init__(self, field_type, primary_key=False, default=None):\n"
                "        self.field_type = field_type\n"
                "        self.primary_key = primary_key\n"
                "        self.default = default\n"
                "        self.name = None\n"
                "\n"
                "\n"
                "class ModelMeta(type):\n"
                "    def __new__(mcs, name, bases, namespace):\n"
                "        fields = {}\n"
                "        for key, value in namespace.items():\n"
                "            if isinstance(value, Field):\n"
                "                value.name = key\n"
                "                fields[key] = value\n"
                '        namespace["_fields"] = fields\n'
                "        return super().__new__(mcs, name, bases, namespace)\n"
                "\n"
                "\n"
                "class Model(metaclass=ModelMeta):\n"
                "    _fields = {}\n"
                "\n"
                "    @classmethod\n"
                "    def create_table_sql(cls):\n"
                "        cols = []\n"
                "        for name, f in cls._fields.items():\n"
                '            col = name + " " + f.field_type\n'
                "            if f.primary_key:\n"
                '                col += " PRIMARY KEY"\n'
                "            cols.append(col)\n"
                '        col_str = ", ".join(cols)\n'
                '        return "CREATE TABLE " + cls.__tablename__ + " (" + col_str + ")"\n'
                "\n"
                "    def insert_sql(self):\n"
                "        names = list(self._fields.keys())\n"
                "        vals = [getattr(self, n, self._fields[n].default) for n in names]\n"
                '        placeholders = ", ".join(["?"] * len(names))\n'
                '        name_str = ", ".join(names)\n'
                '        sql = "INSERT INTO " + self.__tablename__ + " (" + name_str + ") VALUES (" + placeholders + ")"\n'
                "        return sql, vals\n"
                "\n"
                "    @classmethod\n"
                "    def select_sql(cls, where=None, order_by=None, limit=None):\n"
                '        sql = "SELECT * FROM " + cls.__tablename__\n'
                "        if where:\n"
                '            clauses = [k + " = ?" for k in where]\n'
                '            sql += " WHERE " + " AND ".join(clauses)\n'
                "        if order_by:\n"
                '            sql += " ORDER BY " + order_by\n'
                "        if limit:\n"
                '            sql += " LIMIT " + str(limit)\n'
                "        return sql\n"
                "```\n"
                "\n"
                "Key: metaclass collects Field instances into _fields dict. "
                "create_table_sql and select_sql are classmethods. "
                "insert_sql is instance method returning (sql, params)."
            ),
            category="pattern_orm",
        ),
        # ── HTTP Router with path params ────────────────────────
        SolvedExample(
            "Write a `Router` class with decorator-based route registration and path parameter extraction.",
            '```python\nimport re\n\n\nclass Router:\n    def __init__(self):\n        self._routes = []\n\n    def route(self, method, path):\n        pattern = re.sub(r"<(\\w+)>", r"(?P<\\1>[^/]+)", path)\n        pattern = f"^{pattern}$"\n        def decorator(func):\n            self._routes.append({\n                "method": method,\n                "path": path,\n                "pattern": re.compile(pattern),\n                "handler": func,\n            })\n            return func\n        return decorator\n\n    def match(self, method, path):\n        for r in self._routes:\n            if r["method"] != method:\n                continue\n            m = r["pattern"].match(path)\n            if m:\n                return r["handler"], m.groupdict()\n        return None\n\n    def dispatch(self, method, path, **kwargs):\n        result = self.match(method, path)\n        if result is None:\n            raise LookupError(f"No route for {method} {path}")\n        handler, params = result\n        params.update(kwargs)\n        return handler(**params)\n\n    def routes(self):\n        return [\n            {"method": r["method"], "path": r["path"], "handler": r["handler"].__name__}\n            for r in self._routes\n        ]\n```\n\nKey: `route()` returns a decorator. Convert `<param>` to regex named groups. `match` returns (handler, params_dict) or None. `dispatch` calls handler with merged params.',
            category="pattern_router",
        ),
        # ── Token bucket rate limiter ───────────────────────────
        SolvedExample(
            "Write a `TokenBucket` rate limiter with explicit time parameter for testing.",
            '```python\nimport time as _time\n\n\nclass TokenBucket:\n    def __init__(self, rate, capacity):\n        self.rate = rate\n        self.capacity = capacity\n        self._tokens = float(capacity)\n        self._last_time = None\n\n    def _refill(self, now=None):\n        now = now if now is not None else _time.time()\n        if self._last_time is None:\n            self._last_time = now\n            return\n        elapsed = now - self._last_time\n        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)\n        self._last_time = now\n\n    def allow(self, tokens=1, now=None):\n        self._refill(now)\n        if self._tokens >= tokens:\n            self._tokens -= tokens\n            return True\n        return False\n\n    def wait_time(self, tokens=1, now=None):\n        self._refill(now)\n        if self._tokens >= tokens:\n            return 0.0\n        needed = tokens - self._tokens\n        return needed / self.rate\n\n    def available(self, now=None):\n        self._refill(now)\n        return int(self._tokens)\n```\n\nKey: `_refill` computes elapsed time and adds tokens. `allow` consumes tokens. All methods accept optional `now` for deterministic testing.',
            category="pattern_rate_limit",
        ),
        # ── LRU Cache (already works, include for good measure) ─
        SolvedExample(
            "Write an LRU Cache class with get, put, and capacity eviction.",
            '```python\nfrom collections import OrderedDict\n\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self._cache = OrderedDict()\n\n    def get(self, key):\n        if key not in self._cache:\n            return -1\n        self._cache.move_to_end(key)\n        return self._cache[key]\n\n    def put(self, key, value):\n        if key in self._cache:\n            self._cache.move_to_end(key)\n        self._cache[key] = value\n        if len(self._cache) > self.capacity:\n            self._cache.popitem(last=False)\n\n    def __len__(self):\n        return len(self._cache)\n```',
            category="data_structure",
        ),
    ]

    return Expert(
        name="code_gen",
        system_context=(
            "You are a Python expert. Write complete, correct, runnable code. "
            "Use ```python blocks. Include all imports. Handle edge cases. "
            "For classes: implement ALL requested methods. "
            "For decorators: set attributes on the WRAPPER function, not the original. "
            "For properties: use @property for read-only access, store data in _private attributes."
        ),
        examples=examples,
        verifier=verify_code_gen,
        max_examples=3,
        max_retries=1,
    )


# ── Expert Router ─────────────────────────────────────────────

class ExpertRouter:
    """Routes queries to the right expert and runs generate -> verify -> retry."""

    def __init__(self, tuned: bool = False, stress: bool = False):
        self.experts: dict[str, Expert] = {}
        self._embedder = None
        self._tuned = tuned
        self._stress = stress
        # Keep all sets available for runtime switching
        self._generic_experts: dict[str, Expert] = {}
        self._tuned_experts: dict[str, Expert] = {}
        self._stress_experts: dict[str, Expert] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register code-focused experts. Uses tuned code_gen when self._tuned is True."""
        # Always build both sets so we can switch at runtime
        self._generic_experts = {
            "code_gen": build_code_gen_expert(),
            "code_review": build_code_review_expert(),
            "debugger": build_debug_expert(),
            "explainer": build_explainer_expert(),
        }
        self._tuned_experts = {
            "code_gen": build_tuned_code_gen_expert(),
            "code_review": build_code_review_expert(),
            "debugger": build_debug_expert(),
            "explainer": build_explainer_expert(),
        }
        self._stress_experts = {
            "code_gen": build_stress_code_gen_expert(),
            "code_review": build_code_review_expert(),
            "debugger": build_debug_expert(),
            "explainer": build_explainer_expert(),
        }

        if self._stress:
            self.experts = dict(self._stress_experts)
        elif self._tuned:
            self.experts = dict(self._tuned_experts)
        else:
            self.experts = dict(self._generic_experts)

    def init_embeddings(self, embedder):
        """Initialize embeddings for all experts (all sets)."""
        self._embedder = embedder
        all_experts = (set(self._generic_experts.values())
                       | set(self._tuned_experts.values())
                       | set(self._stress_experts.values()))
        for expert in all_experts:
            expert.init_embeddings(embedder)
        logger.info(f"Expert embeddings initialized for {len(self.experts)} active experts")

    def use_tuned_experts(self):
        """Swap in tuned code_gen expert (algorithm-specific few-shot examples)."""
        self._tuned = True
        self._stress = False
        self.experts = dict(self._tuned_experts)
        logger.info("Switched to tuned experts")

    def use_generic_experts(self):
        """Revert to the original generic experts."""
        self._tuned = False
        self._stress = False
        self.experts = dict(self._generic_experts)
        logger.info("Switched to generic experts")

    def use_stress_experts(self):
        """Swap in stress-targeted experts (pattern-specific few-shot examples)."""
        self._tuned = False
        self._stress = True
        self.experts = dict(self._stress_experts)
        logger.info("Switched to stress experts")

    def select_expert(self, query: str, module_hint: Optional[str] = None) -> Optional[Expert]:
        """Select the best expert for a query."""
        if module_hint == "code_gen":
            return self.experts.get("code_gen")
        if module_hint == "code_review":
            return self.experts.get("code_review")
        if module_hint == "debugger":
            return self.experts.get("debugger")
        if module_hint == "explainer":
            return self.experts.get("explainer")

        q = query.lower()
        if any(w in q for w in ["write", "create", "implement", "build", "generate", "function", "class"]):
            return self.experts.get("code_gen")
        if any(w in q for w in ["review", "check", "audit", "refactor", "optimize", "improve"]):
            return self.experts.get("code_review")
        if any(w in q for w in ["bug", "fix", "error", "debug", "crash", "traceback", "broken"]):
            return self.experts.get("debugger")
        if any(w in q for w in ["explain", "how does", "what is", "what's the difference", "teach"]):
            return self.experts.get("explainer")

        if any(w in q for w in ["code", "python", "script", "def ", "import "]):
            return self.experts.get("code_gen")

        return None

    def process(self, query: str, model, chat_format: str,
                module_hint: Optional[str] = None,
                gen_kwargs: Optional[dict] = None) -> Optional[ExpertResult]:
        """Full expert pipeline: select -> build prompt -> generate -> verify -> retry."""
        kwargs = gen_kwargs or {}
        expert = self.select_expert(query, module_hint)

        if expert is None:
            return None

        prompt = expert.build_prompt(query, chat_format)
        prompt_tokens = model.count_tokens(prompt)

        if expert.grammar_str:
            try:
                from llama_cpp import LlamaGrammar
                grammar = LlamaGrammar.from_string(expert.grammar_str)
                kwargs["grammar"] = grammar
            except Exception:
                pass

        response = model.generate(prompt, **kwargs)
        attempts = 1

        for retry in range(expert.max_retries):
            passed, error_hint = expert.verify(response, query)
            if passed:
                break

            logger.debug(
                f"Expert '{expert.name}' verification failed (attempt {attempts}): {error_hint}"
            )

            retry_prompt = expert.build_retry_prompt(
                query, response, error_hint, chat_format,
            )
            response = model.generate(retry_prompt, **kwargs)
            attempts += 1

        verified, _ = expert.verify(response, query)
        examples_used = len(expert.retrieve_examples(query))

        return ExpertResult(
            response=response,
            expert_name=expert.name,
            attempts=attempts,
            verified=verified,
            scaffolding_used=expert.scaffolding is not None,
            examples_injected=examples_used,
            prompt_tokens=prompt_tokens,
        )
