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
        """Find the most relevant solved examples for a query."""
        if not self._embedder or self._example_embeddings is None or len(self.examples) == 0:
            return self.examples[:self.max_examples]

        k = min(top_k or self.max_examples, len(self.examples))
        query_vec = self._embedder.encode([query], normalize_embeddings=True)

        sims = np.dot(self._example_embeddings, query_vec.T).flatten()
        top_indices = sims.argsort()[-k:][::-1]

        return [self.examples[i] for i in top_indices]

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


# ── Expert Router ─────────────────────────────────────────────

class ExpertRouter:
    """Routes queries to the right expert and runs generate -> verify -> retry."""

    def __init__(self):
        self.experts: dict[str, Expert] = {}
        self._embedder = None
        self._register_defaults()

    def _register_defaults(self):
        """Register code-focused experts."""
        self.experts["code_gen"] = build_code_gen_expert()
        self.experts["code_review"] = build_code_review_expert()
        self.experts["debugger"] = build_debug_expert()
        self.experts["explainer"] = build_explainer_expert()

    def init_embeddings(self, embedder):
        """Initialize embeddings for all experts."""
        self._embedder = embedder
        for expert in self.experts.values():
            expert.init_embeddings(embedder)
        logger.info(f"Expert embeddings initialized for {len(self.experts)} experts")

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
