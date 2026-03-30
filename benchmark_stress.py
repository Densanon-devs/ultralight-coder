#!/usr/bin/env python3
"""
Stress Benchmark — Push models beyond simple functions.

Three tiers of difficulty:
  Tier 1 (Medium):     Multi-function classes, real patterns (max_tokens=1024)
  Tier 2 (Heavy):      Mini-app in one prompt (max_tokens=2048)
  Tier 3 (Multi-turn): Step-by-step app building, code accumulates (max_tokens=1024/step)

Reuses infrastructure from benchmark_exec.py (extraction, execution, chat formatting).

Usage:
    python benchmark_stress.py                              # All tiers, top models
    python benchmark_stress.py --tier 1                     # Medium only
    python benchmark_stress.py --tier 2                     # Heavy only
    python benchmark_stress.py --tier 3                     # Multi-turn only
    python benchmark_stress.py --model models/foo.gguf      # Specific model
    python benchmark_stress.py --all                        # All models on disk
    python benchmark_stress.py --quick                      # 1 test per tier
"""

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_exec import (
    detect_chat_format, wrap_chat, extract_code, safe_exec, run_tests,
    discover_models, ExecTestResult, ExecModelResult,
)

logger = logging.getLogger("bench_stress")


# ═══════════════════════════════════════════════════════════════════════
# Test definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StressTest:
    test_id: str
    tier: int           # 1=medium, 2=heavy
    prompt: str
    test_code: str
    func_name: str      # main class/function to check for
    max_tokens: int = 1024


@dataclass
class MultiTurnStep:
    step_id: str
    prompt: str         # instruction for this step (gets prefixed with previous code)
    test_code: str      # assertions to run after this step
    func_name: str      # main symbol to check for after this step
    max_tokens: int = 1024


@dataclass
class MultiTurnTest:
    test_id: str
    description: str
    steps: list[MultiTurnStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TIER 1 — Medium: multi-function, classes with state, real patterns
# ---------------------------------------------------------------------------

def build_tier1_tests() -> list[StressTest]:
    tests = []

    # ── T1.01: LRU Cache ──────────────────────────────────────
    tests.append(StressTest(
        test_id="t1_01_lru_cache",
        tier=1,
        func_name="LRUCache",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `LRUCache` that implements a Least Recently Used cache.\n"
            "- `__init__(self, capacity: int)` — set max capacity\n"
            "- `get(self, key)` — return value if key exists (and mark as recently used), else return -1\n"
            "- `put(self, key, value)` — insert or update. If at capacity, evict the least recently used item first.\n"
            "- `__len__(self)` — return current number of items\n"
            "Use an OrderedDict or implement with a dict + doubly linked list."
        ),
        test_code="""
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)
assert cache.get(2) == -1
cache.put(4, 4)
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
assert len(cache) == 2

cache2 = LRUCache(1)
cache2.put("a", 100)
assert cache2.get("a") == 100
cache2.put("b", 200)
assert cache2.get("a") == -1
assert cache2.get("b") == 200
""",
    ))

    # ── T1.02: Expression Evaluator ───────────────────────────
    tests.append(StressTest(
        test_id="t1_02_expression_eval",
        tier=1,
        func_name="evaluate",
        max_tokens=1024,
        prompt=(
            "Write a Python function called `evaluate(expr: str) -> float` that evaluates a "
            "mathematical expression string containing +, -, *, /, parentheses, and floating point "
            "numbers. Respect operator precedence (* and / before + and -). Parentheses override precedence.\n"
            "Examples: evaluate('2 + 3 * 4') returns 14.0, evaluate('(2 + 3) * 4') returns 20.0, "
            "evaluate('10 / 2 - 1') returns 4.0."
        ),
        test_code="""
assert abs(evaluate("2 + 3") - 5.0) < 1e-9
assert abs(evaluate("2 + 3 * 4") - 14.0) < 1e-9
assert abs(evaluate("(2 + 3) * 4") - 20.0) < 1e-9
assert abs(evaluate("10 / 2 - 1") - 4.0) < 1e-9
assert abs(evaluate("3 + 4 * 2 / (1 - 5)") - 1.0) < 1e-9
assert abs(evaluate("100") - 100.0) < 1e-9
assert abs(evaluate("(((5)))") - 5.0) < 1e-9
assert abs(evaluate("2.5 * 4") - 10.0) < 1e-9
""",
    ))

    # ── T1.03: Trie (Prefix Tree) ────────────────────────────
    tests.append(StressTest(
        test_id="t1_03_trie",
        tier=1,
        func_name="Trie",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `Trie` (prefix tree) with these methods:\n"
            "- `insert(word: str)` — insert a word\n"
            "- `search(word: str) -> bool` — return True if the exact word exists\n"
            "- `starts_with(prefix: str) -> bool` — return True if any word starts with prefix\n"
            "- `count_prefix(prefix: str) -> int` — return how many words start with prefix\n"
            "- `all_words() -> list[str]` — return all words in the trie sorted alphabetically"
        ),
        test_code="""
t = Trie()
t.insert("apple")
t.insert("app")
t.insert("apricot")
t.insert("banana")
assert t.search("apple") == True
assert t.search("app") == True
assert t.search("ap") == False
assert t.starts_with("ap") == True
assert t.starts_with("ban") == True
assert t.starts_with("cat") == False
assert t.count_prefix("ap") == 3
assert t.count_prefix("app") == 2
assert t.count_prefix("banana") == 1
assert t.count_prefix("z") == 0
assert t.all_words() == ["app", "apple", "apricot", "banana"]
""",
    ))

    # ── T1.04: Event Emitter ──────────────────────────────────
    tests.append(StressTest(
        test_id="t1_04_event_emitter",
        tier=1,
        func_name="EventEmitter",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `EventEmitter` with these methods:\n"
            "- `on(event: str, callback)` — register a listener for an event\n"
            "- `off(event: str, callback)` — remove a specific listener\n"
            "- `emit(event: str, *args, **kwargs)` — call all listeners for that event with the given args\n"
            "- `once(event: str, callback)` — register a listener that fires only once then auto-removes\n"
            "- `listener_count(event: str) -> int` — return number of listeners for an event"
        ),
        test_code="""
ee = EventEmitter()
results = []

def on_data(x):
    results.append(x)

ee.on("data", on_data)
ee.emit("data", 1)
ee.emit("data", 2)
assert results == [1, 2]

assert ee.listener_count("data") == 1
ee.off("data", on_data)
ee.emit("data", 3)
assert results == [1, 2]
assert ee.listener_count("data") == 0

once_results = []
def on_once(x):
    once_results.append(x)

ee.once("ping", on_once)
ee.emit("ping", "a")
ee.emit("ping", "b")
assert once_results == ["a"]
assert ee.listener_count("ping") == 0

multi = []
ee.on("multi", lambda x: multi.append(x * 2))
ee.on("multi", lambda x: multi.append(x * 3))
ee.emit("multi", 10)
assert multi == [20, 30]
""",
    ))

    # ── T1.05: CSV Parser ────────────────────────────────────
    tests.append(StressTest(
        test_id="t1_05_csv_parser",
        tier=1,
        func_name="parse_csv",
        max_tokens=1024,
        prompt=(
            "Write a Python function called `parse_csv(text: str, delimiter: str = ',') -> list[dict]` "
            "that parses CSV text into a list of dictionaries. The first line is the header row.\n"
            "Rules:\n"
            "- Fields may be quoted with double quotes\n"
            "- Quoted fields can contain the delimiter, newlines, and escaped quotes (double-double-quote \"\")\n"
            "- Unquoted fields are trimmed of whitespace\n"
            "- Return a list of dicts where keys are the header names"
        ),
        test_code=(
            'simple = parse_csv("name,age\\nAlice,30\\nBob,25")\n'
            'assert len(simple) == 2\n'
            'assert simple[0]["name"] == "Alice"\n'
            'assert simple[0]["age"] == "30"\n'
            'assert simple[1]["name"] == "Bob"\n'
            '\n'
            'quoted = parse_csv(\'name,bio\\nAlice,"likes, commas"\\nBob,"says ""hi"""\' )\n'
            'assert quoted[0]["bio"] == "likes, commas"\n'
            'assert quoted[1]["bio"] == \'says "hi"\'\n'
            '\n'
            'tabbed = parse_csv("a\\tb\\n1\\t2", delimiter="\\t")\n'
            'assert tabbed[0]["a"] == "1"\n'
            'assert tabbed[0]["b"] == "2"\n'
        ),
    ))

    # ── T1.06: JSON Flattener/Unflattener ─────────────────────
    tests.append(StressTest(
        test_id="t1_06_json_flatten",
        tier=1,
        func_name="flatten_json",
        max_tokens=1024,
        prompt=(
            "Write two Python functions:\n"
            "1. `flatten_json(data: dict, sep: str = '.') -> dict` — flatten a nested dict to "
            "dot-separated keys. Lists become key.0, key.1, etc.\n"
            "2. `unflatten_json(data: dict, sep: str = '.') -> dict` — reverse: take flat dict "
            "and rebuild the nested structure. Integer keys become list indices.\n"
            "Example: flatten_json({'a': {'b': 1, 'c': [2, 3]}}) returns "
            "{'a.b': 1, 'a.c.0': 2, 'a.c.1': 3}"
        ),
        test_code="""
assert flatten_json({"a": 1}) == {"a": 1}
assert flatten_json({"a": {"b": 1}}) == {"a.b": 1}
f = flatten_json({"a": {"b": 1, "c": [2, 3]}})
assert f == {"a.b": 1, "a.c.0": 2, "a.c.1": 3}
assert flatten_json({}) == {}
assert flatten_json({"x": {"y": {"z": 42}}}) == {"x.y.z": 42}

assert unflatten_json({"a": 1}) == {"a": 1}
assert unflatten_json({"a.b": 1}) == {"a": {"b": 1}}
u = unflatten_json({"a.b": 1, "a.c.0": 2, "a.c.1": 3})
assert u == {"a": {"b": 1, "c": [2, 3]}}
""",
    ))

    # ── T1.07: Retry Decorator ────────────────────────────────
    tests.append(StressTest(
        test_id="t1_07_retry_decorator",
        tier=1,
        func_name="retry",
        max_tokens=1024,
        prompt=(
            "Write a Python decorator called `retry(max_attempts=3, exceptions=(Exception,), backoff=0)` "
            "that retries a function up to max_attempts times if it raises one of the given exceptions.\n"
            "- If the function succeeds, return its result immediately\n"
            "- If all attempts fail, re-raise the last exception\n"
            "- `backoff` is seconds to wait between attempts (use time.sleep), can be 0\n"
            "- The decorator should work with any function signature (*args, **kwargs)\n"
            "- Add an `attempts` attribute to the decorated function that tracks how many attempts the last call took"
        ),
        test_code="""
import time as _time

call_count = 0

@retry(max_attempts=3, exceptions=(ValueError,), backoff=0)
def fail_twice():
    global call_count
    call_count += 1
    if call_count < 3:
        raise ValueError("not yet")
    return "success"

call_count = 0
result = fail_twice()
assert result == "success"
assert fail_twice.attempts == 3

@retry(max_attempts=2, exceptions=(TypeError,), backoff=0)
def always_fail():
    raise TypeError("bad")

try:
    always_fail()
    assert False, "Should have raised"
except TypeError:
    pass
assert always_fail.attempts == 2

@retry(max_attempts=1, backoff=0)
def instant_success():
    return 42

assert instant_success() == 42
assert instant_success.attempts == 1
""",
    ))

    # ── T1.08: Graph BFS/DFS ─────────────────────────────────
    tests.append(StressTest(
        test_id="t1_08_graph",
        tier=1,
        func_name="Graph",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `Graph` for a directed graph:\n"
            "- `add_edge(u, v)` — add directed edge from u to v\n"
            "- `neighbors(u) -> list` — return list of nodes u points to\n"
            "- `bfs(start) -> list` — breadth-first traversal from start, return list of visited nodes in order\n"
            "- `dfs(start) -> list` — depth-first traversal from start, return list of visited nodes in order\n"
            "- `shortest_path(start, end) -> list` — return shortest path as list of nodes (BFS-based), "
            "or empty list if no path exists\n"
            "- `has_cycle() -> bool` — return True if the graph contains a cycle"
        ),
        test_code="""
g = Graph()
g.add_edge("A", "B")
g.add_edge("A", "C")
g.add_edge("B", "D")
g.add_edge("C", "D")
g.add_edge("D", "E")

bfs_result = g.bfs("A")
assert bfs_result[0] == "A"
assert "E" in bfs_result
assert len(bfs_result) == 5

dfs_result = g.dfs("A")
assert dfs_result[0] == "A"
assert "E" in dfs_result
assert len(dfs_result) == 5

path = g.shortest_path("A", "E")
assert path[0] == "A"
assert path[-1] == "E"
assert len(path) <= 4

assert g.shortest_path("E", "A") == []
assert g.has_cycle() == False

g.add_edge("E", "A")
assert g.has_cycle() == True
""",
    ))

    # ── T1.09: Rate Limiter ──────────────────────────────────
    tests.append(StressTest(
        test_id="t1_09_rate_limiter",
        tier=1,
        func_name="RateLimiter",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `RateLimiter` implementing a token bucket algorithm:\n"
            "- `__init__(self, rate: float, capacity: int)` — rate is tokens added per second, "
            "capacity is max tokens in the bucket. Bucket starts full.\n"
            "- `allow(self, tokens: int = 1, now: float = None) -> bool` — return True if enough "
            "tokens available and consume them. `now` is current time in seconds (pass explicitly "
            "for testing, default to time.time()).\n"
            "- `wait_time(self, tokens: int = 1, now: float = None) -> float` — return seconds "
            "to wait before `tokens` would be available. Return 0.0 if available now.\n"
            "- `available(self, now: float = None) -> int` — return current available tokens (as int)"
        ),
        test_code="""
rl = RateLimiter(rate=10.0, capacity=10)

assert rl.allow(5, now=0.0) == True
assert rl.available(now=0.0) == 5
assert rl.allow(5, now=0.0) == True
assert rl.allow(1, now=0.0) == False
assert rl.available(now=0.0) == 0

assert rl.wait_time(1, now=0.0) > 0
assert abs(rl.wait_time(1, now=0.0) - 0.1) < 0.05

assert rl.allow(1, now=0.5) == True
assert rl.available(now=0.5) == 4

rl2 = RateLimiter(rate=1.0, capacity=5)
assert rl2.allow(5, now=0.0) == True
assert rl2.allow(1, now=0.0) == False
assert rl2.allow(1, now=1.0) == True
assert rl2.allow(1, now=1.5) == False
assert rl2.allow(1, now=2.0) == True
""",
    ))

    # ── T1.10: Mini State Machine ─────────────────────────────
    tests.append(StressTest(
        test_id="t1_10_state_machine",
        tier=1,
        func_name="StateMachine",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `StateMachine`:\n"
            "- `__init__(self, initial_state: str)` — set the starting state\n"
            "- `add_transition(from_state, event, to_state, guard=None)` — register a transition. "
            "`guard` is an optional callable returning bool; transition only fires if guard returns True (or guard is None).\n"
            "- `trigger(event) -> str` — attempt to fire the event from current state. Return new state. "
            "Raise ValueError if no valid transition exists.\n"
            "- `state` property — return current state\n"
            "- `history` property — return list of all states visited (including initial)"
        ),
        test_code="""
sm = StateMachine("idle")
sm.add_transition("idle", "start", "running")
sm.add_transition("running", "pause", "paused")
sm.add_transition("paused", "resume", "running")
sm.add_transition("running", "stop", "idle")

assert sm.state == "idle"
sm.trigger("start")
assert sm.state == "running"
sm.trigger("pause")
assert sm.state == "paused"
sm.trigger("resume")
assert sm.state == "running"
sm.trigger("stop")
assert sm.state == "idle"
assert sm.history == ["idle", "running", "paused", "running", "idle"]

try:
    sm.trigger("pause")
    assert False, "Should raise ValueError"
except ValueError:
    pass

sm2 = StateMachine("locked")
sm2.add_transition("locked", "coin", "unlocked", guard=lambda: True)
sm2.add_transition("locked", "push", "locked", guard=lambda: False)
sm2.trigger("coin")
assert sm2.state == "unlocked"

sm3 = StateMachine("a")
sm3.add_transition("a", "go", "b", guard=lambda: False)
try:
    sm3.trigger("go")
    assert False, "Guard should block"
except ValueError:
    pass
assert sm3.state == "a"
""",
    ))

    return tests


# ---------------------------------------------------------------------------
# TIER 2 — Heavy: mini-app in one prompt
# ---------------------------------------------------------------------------

def build_tier2_tests() -> list[StressTest]:
    tests = []

    # ── T2.01: In-Memory Key-Value Store with TTL ─────────────
    tests.append(StressTest(
        test_id="t2_01_kv_store",
        tier=2,
        func_name="KVStore",
        max_tokens=2048,
        prompt=(
            "Write a Python class called `KVStore` — an in-memory key-value store with TTL support.\n\n"
            "Methods:\n"
            "- `set(key, value, ttl=None)` — store a value. ttl is seconds until expiry (None = never expires). "
            "Use an explicit `now` parameter for testing: `set(key, value, ttl=None, now=None)` where now defaults to time.time().\n"
            "- `get(key, now=None)` — return value if key exists and not expired, else return None. Expired keys should be lazily cleaned up.\n"
            "- `delete(key)` — remove a key. Return True if it existed, False otherwise.\n"
            "- `keys(now=None)` — return list of all non-expired keys.\n"
            "- `clear()` — remove everything.\n"
            "- `__len__(self)` — return count of non-expired keys (use time.time() for now).\n"
            "- `set_many(mapping: dict, ttl=None, now=None)` — bulk set from a dict.\n"
            "- `get_many(keys: list, now=None) -> dict` — bulk get, skip missing/expired.\n"
            "- `incr(key, amount=1, now=None) -> int` — increment a numeric value and return new value. "
            "If key doesn't exist, start from 0. Raise TypeError if value is not numeric.\n\n"
            "The store should handle thousands of keys efficiently."
        ),
        test_code="""
store = KVStore()
store.set("a", 1, now=0.0)
store.set("b", 2, ttl=5, now=0.0)
store.set("c", 3, ttl=1, now=0.0)

assert store.get("a", now=0.0) == 1
assert store.get("b", now=0.0) == 2
assert store.get("c", now=0.0) == 3
assert store.get("missing", now=0.0) is None

assert store.get("c", now=2.0) is None
assert store.get("b", now=4.0) == 2
assert store.get("b", now=6.0) is None
assert store.get("a", now=100.0) == 1

assert store.delete("a") == True
assert store.delete("a") == False
assert store.get("a") is None

store.clear()
store.set_many({"x": 10, "y": 20, "z": 30}, now=0.0)
result = store.get_many(["x", "y", "z", "missing"], now=0.0)
assert result == {"x": 10, "y": 20, "z": 30}

store.set("counter", 0, now=0.0)
assert store.incr("counter", 5, now=0.0) == 5
assert store.incr("counter", 3, now=0.0) == 8
assert store.incr("new_counter", 1, now=0.0) == 1

store.set("text", "hello", now=0.0)
try:
    store.incr("text", 1, now=0.0)
    assert False, "Should raise TypeError"
except TypeError:
    pass

keys = store.keys(now=0.0)
assert "x" in keys
assert "counter" in keys
""",
    ))

    # ── T2.02: Markdown to HTML Converter ─────────────────────
    tests.append(StressTest(
        test_id="t2_02_markdown_html",
        tier=2,
        func_name="markdown_to_html",
        max_tokens=2048,
        prompt=(
            "Write a Python function called `markdown_to_html(text: str) -> str` that converts "
            "a subset of Markdown to HTML.\n\n"
            "Supported syntax:\n"
            "- `# Heading` through `###### Heading` → `<h1>` through `<h6>`\n"
            "- `**bold**` → `<strong>bold</strong>`\n"
            "- `*italic*` → `<em>italic</em>`\n"
            "- `` `code` `` → `<code>code</code>`\n"
            "- `[text](url)` → `<a href=\"url\">text</a>`\n"
            "- Lines starting with `- ` are unordered list items → wrap in `<ul><li>...</li></ul>`\n"
            "- Consecutive list items should be wrapped in a single `<ul>` block\n"
            "- Blank lines between paragraphs → `<p>text</p>`\n"
            "- Code blocks with ``` → `<pre><code>...</code></pre>`\n\n"
            "Return the HTML as a string. Don't worry about edge cases beyond these rules."
        ),
        test_code="""
assert "<h1>Hello</h1>" in markdown_to_html("# Hello")
assert "<h3>Three</h3>" in markdown_to_html("### Three")
assert "<strong>bold</strong>" in markdown_to_html("**bold**")
assert "<em>italic</em>" in markdown_to_html("*italic*")
assert "<code>x</code>" in markdown_to_html("`x`")
assert '<a href="http://example.com">link</a>' in markdown_to_html("[link](http://example.com)")

list_html = markdown_to_html("- one\\n- two\\n- three")
assert "<ul>" in list_html
assert "<li>" in list_html
assert "one" in list_html
assert "three" in list_html

code_block = markdown_to_html("```\\nprint('hi')\\n```")
assert "<pre>" in code_block or "<code>" in code_block
assert "print" in code_block
""",
    ))

    # ── T2.03: Task Scheduler with Dependencies ───────────────
    tests.append(StressTest(
        test_id="t2_03_task_scheduler",
        tier=2,
        func_name="TaskScheduler",
        max_tokens=2048,
        prompt=(
            "Write a Python class called `TaskScheduler` that manages tasks with priorities and dependencies.\n\n"
            "Methods:\n"
            "- `add_task(name: str, priority: int = 0, depends_on: list[str] = None)` — add a task. "
            "Higher priority = more important. depends_on lists task names that must complete first.\n"
            "- `complete(name: str)` — mark a task as completed.\n"
            "- `next_task() -> str | None` — return the name of the highest-priority task that is ready "
            "(all dependencies completed and task not yet completed). Return None if no ready tasks.\n"
            "- `get_order() -> list[str]` — return a valid execution order for ALL uncompleted tasks "
            "(topological sort respecting dependencies, break ties by priority descending, then name alphabetically).\n"
            "- `is_complete(name: str) -> bool` — return whether a task is completed.\n"
            "- `has_cycle() -> bool` — return True if there's a circular dependency.\n"
            "- `ready_tasks() -> list[str]` — return all tasks that are ready to run (deps met, not completed), "
            "sorted by priority desc."
        ),
        test_code="""
s = TaskScheduler()
s.add_task("build", priority=5)
s.add_task("test", priority=3, depends_on=["build"])
s.add_task("deploy", priority=1, depends_on=["test"])
s.add_task("lint", priority=4)

assert s.next_task() == "build"
assert set(s.ready_tasks()) == {"build", "lint"}

s.complete("build")
assert s.is_complete("build") == True
assert "test" in s.ready_tasks()

order = s.get_order()
assert order.index("build") < order.index("test") or s.is_complete("build")
assert order.index("test") < order.index("deploy")

s.complete("lint")
s.complete("test")
assert s.next_task() == "deploy"
s.complete("deploy")
assert s.next_task() is None

cyc = TaskScheduler()
cyc.add_task("a", depends_on=["b"])
cyc.add_task("b", depends_on=["a"])
assert cyc.has_cycle() == True

no_cyc = TaskScheduler()
no_cyc.add_task("x")
no_cyc.add_task("y", depends_on=["x"])
assert no_cyc.has_cycle() == False
""",
    ))

    # ── T2.04: Mini ORM ───────────────────────────────────────
    tests.append(StressTest(
        test_id="t2_04_mini_orm",
        tier=2,
        func_name="Model",
        max_tokens=2048,
        prompt=(
            "Write a mini Python ORM. Provide these classes/functions:\n\n"
            "1. `Field` class — represents a column. `Field(field_type: str, primary_key=False, default=None)`\n"
            "   field_type is one of: 'INTEGER', 'TEXT', 'REAL', 'BOOLEAN'\n\n"
            "2. `Model` base class — subclass it to define models:\n"
            "   - Class attribute `__tablename__` for table name\n"
            "   - Class attributes that are Field instances define columns\n"
            "   - `create_table_sql() -> str` — classmethod returning CREATE TABLE SQL\n"
            "   - `insert_sql(self) -> tuple[str, list]` — return (SQL, params) for INSERT\n"
            "   - `select_sql(cls, where=None, order_by=None, limit=None) -> str` — classmethod, "
            "return SELECT SQL. where is a dict like {'name': 'Alice'}, order_by is a column name string.\n\n"
            "Example usage:\n"
            "```python\n"
            "class User(Model):\n"
            "    __tablename__ = 'users'\n"
            "    id = Field('INTEGER', primary_key=True)\n"
            "    name = Field('TEXT')\n"
            "    age = Field('INTEGER', default=0)\n"
            "```"
        ),
        test_code="""
class User(Model):
    __tablename__ = 'users'
    id = Field('INTEGER', primary_key=True)
    name = Field('TEXT')
    age = Field('INTEGER', default=0)

sql = User.create_table_sql()
assert "CREATE TABLE" in sql
assert "users" in sql
assert "INTEGER" in sql
assert "TEXT" in sql
assert "PRIMARY KEY" in sql.upper() or "primary key" in sql.lower()

u = User()
u.id = 1
u.name = "Alice"
u.age = 30
insert_sql, params = u.insert_sql()
assert "INSERT" in insert_sql.upper()
assert "users" in insert_sql
assert len(params) == 3 or len(params) >= 2

select = User.select_sql()
assert "SELECT" in select.upper()
assert "users" in select

where_sql = User.select_sql(where={"name": "Alice"})
assert "WHERE" in where_sql.upper()
assert "name" in where_sql

order_sql = User.select_sql(order_by="age")
assert "ORDER BY" in order_sql.upper()
assert "age" in order_sql

limit_sql = User.select_sql(limit=10)
assert "LIMIT" in limit_sql.upper()
assert "10" in limit_sql
""",
    ))

    # ── T2.05: HTTP Router ────────────────────────────────────
    tests.append(StressTest(
        test_id="t2_05_http_router",
        tier=2,
        func_name="Router",
        max_tokens=2048,
        prompt=(
            "Write a Python class called `Router` for HTTP-style URL routing.\n\n"
            "Methods:\n"
            "- `route(method: str, path: str)` — decorator to register a handler function. "
            "path can contain params like `/users/<id>` or `/posts/<post_id>/comments/<comment_id>`.\n"
            "- `match(method: str, path: str) -> tuple[callable, dict] | None` — find the handler "
            "for a method+path. Return (handler_fn, params_dict) or None if no match. "
            "Params are extracted from the URL.\n"
            "- `dispatch(method: str, path: str, **kwargs) -> any` — call the matched handler with "
            "extracted params merged with kwargs. Raise a LookupError if no route matches.\n"
            "- `routes() -> list[dict]` — return list of registered routes as "
            "[{'method': 'GET', 'path': '/users/<id>', 'handler': fn_name}, ...]\n\n"
            "Support methods: GET, POST, PUT, DELETE. Path matching should be exact segments."
        ),
        test_code="""
router = Router()

@router.route("GET", "/")
def index():
    return "home"

@router.route("GET", "/users")
def list_users():
    return ["alice", "bob"]

@router.route("GET", "/users/<id>")
def get_user(id):
    return f"user_{id}"

@router.route("POST", "/users")
def create_user(name="anonymous"):
    return f"created_{name}"

@router.route("GET", "/posts/<post_id>/comments/<comment_id>")
def get_comment(post_id, comment_id):
    return f"post{post_id}_comment{comment_id}"

match_result = router.match("GET", "/")
assert match_result is not None
handler, params = match_result
assert params == {} or params == {}

match_result = router.match("GET", "/users/42")
assert match_result is not None
handler, params = match_result
assert params["id"] == "42"

assert router.dispatch("GET", "/") == "home"
assert router.dispatch("GET", "/users/99") == "user_99"
assert router.dispatch("POST", "/users", name="Alice") == "created_Alice"
assert router.dispatch("GET", "/posts/5/comments/3") == "post5_comment3"

assert router.match("DELETE", "/users") is None
assert router.match("GET", "/nonexistent") is None

try:
    router.dispatch("GET", "/nonexistent")
    assert False, "Should raise LookupError"
except LookupError:
    pass

all_routes = router.routes()
assert len(all_routes) == 5
""",
    ))

    return tests


# ---------------------------------------------------------------------------
# TIER 3 — Multi-turn: step-by-step incremental builds
# ---------------------------------------------------------------------------

def build_tier3_tests() -> list[MultiTurnTest]:
    tests = []

    # ── T3.01: Build a Todo App Step by Step ──────────────────
    todo_steps = [
        MultiTurnStep(
            step_id="t3_01_step1",
            func_name="TodoList",
            prompt=(
                "Write a Python Todo application. Start with:\n"
                "1. A `TodoItem` dataclass with fields: id (int), title (str), done (bool, default False)\n"
                "2. A `TodoList` class with:\n"
                "   - `add(title: str) -> TodoItem` — create and add a todo, auto-increment id starting at 1\n"
                "   - `remove(id: int) -> bool` — remove by id, return True if found\n"
                "   - `get(id: int) -> TodoItem | None` — get by id\n"
                "   - `all() -> list[TodoItem]` — return all items\n"
                "   - `complete(id: int) -> bool` — mark as done, return True if found"
            ),
            test_code="""
tl = TodoList()
item = tl.add("Buy milk")
assert item.id == 1
assert item.title == "Buy milk"
assert item.done == False

tl.add("Walk dog")
assert len(tl.all()) == 2

assert tl.complete(1) == True
assert tl.get(1).done == True
assert tl.remove(2) == True
assert len(tl.all()) == 1
assert tl.remove(99) == False
""",
        ),
        MultiTurnStep(
            step_id="t3_01_step2",
            func_name="TodoList",
            prompt=(
                "Extend the TodoItem and TodoList with priority support:\n"
                "- Add a `priority` field to TodoItem: 'low', 'medium', 'high' (default 'medium')\n"
                "- Modify `add()` to accept an optional `priority` parameter\n"
                "- Add `by_priority() -> list[TodoItem]` method to TodoList that returns items "
                "sorted by priority (high first, then medium, then low), then by id ascending within same priority\n"
                "- Add `pending() -> list[TodoItem]` that returns only items where done=False"
            ),
            test_code="""
tl = TodoList()
tl.add("Low task", priority="low")
tl.add("High task", priority="high")
tl.add("Med task")

sorted_items = tl.by_priority()
assert sorted_items[0].title == "High task"
assert sorted_items[-1].title == "Low task"

tl.complete(1)
pending = tl.pending()
assert len(pending) == 2
assert all(not item.done for item in pending)
""",
        ),
        MultiTurnStep(
            step_id="t3_01_step3",
            func_name="TodoList",
            prompt=(
                "Add persistence to the TodoList:\n"
                "- `save(filepath: str)` — save all items to a JSON file\n"
                "- `load(filepath: str)` — classmethod or static method that returns a new TodoList loaded from JSON\n"
                "The JSON should be a list of objects with all TodoItem fields.\n"
                "Make sure id auto-increment continues from the max id in the loaded data."
            ),
            test_code="""
import tempfile, os, json

tl = TodoList()
tl.add("Task A", priority="high")
tl.add("Task B", priority="low")
tl.complete(1)

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    tmppath = f.name

try:
    tl.save(tmppath)
    with open(tmppath) as f:
        data = json.load(f)
    assert len(data) == 2

    loaded = TodoList.load(tmppath)
    assert len(loaded.all()) == 2
    assert loaded.get(1).done == True
    assert loaded.get(2).priority == "low"

    new_item = loaded.add("Task C")
    assert new_item.id == 3
finally:
    os.unlink(tmppath)
""",
        ),
        MultiTurnStep(
            step_id="t3_01_step4",
            func_name="TodoList",
            prompt=(
                "Add search and filtering to TodoList:\n"
                "- `search(keyword: str) -> list[TodoItem]` — return items where title contains keyword (case-insensitive)\n"
                "- `filter(done: bool = None, priority: str = None) -> list[TodoItem]` — "
                "filter by done status and/or priority. None means don't filter on that field.\n"
                "- `stats() -> dict` — return {'total': int, 'done': int, 'pending': int, "
                "'by_priority': {'high': int, 'medium': int, 'low': int}}"
            ),
            test_code="""
tl = TodoList()
tl.add("Buy groceries", priority="high")
tl.add("Buy shoes", priority="low")
tl.add("Walk dog", priority="medium")
tl.add("Walk cat", priority="high")
tl.complete(1)

results = tl.search("buy")
assert len(results) == 2

results = tl.search("walk")
assert len(results) == 2

filtered = tl.filter(done=False)
assert len(filtered) == 3

filtered = tl.filter(priority="high")
assert len(filtered) == 2

filtered = tl.filter(done=False, priority="high")
assert len(filtered) == 1

s = tl.stats()
assert s["total"] == 4
assert s["done"] == 1
assert s["pending"] == 3
assert s["by_priority"]["high"] == 2
assert s["by_priority"]["low"] == 1
""",
        ),
    ]

    tests.append(MultiTurnTest(
        test_id="t3_01_todo_app",
        description="Build a Todo app step by step: dataclass -> priority -> persistence -> search/stats",
        steps=todo_steps,
    ))

    # ── T3.02: Build a Calculator Step by Step ────────────────
    calc_steps = [
        MultiTurnStep(
            step_id="t3_02_step1",
            func_name="Calculator",
            prompt=(
                "Write a Python class called `Calculator` with:\n"
                "- `evaluate(expr: str) -> float` — evaluate a math expression string with +, -, *, /\n"
                "- Must respect operator precedence (* and / before + and -)\n"
                "- Handle spaces between tokens\n"
                "- Handle negative numbers at the start: '-5 + 3' = -2\n"
                "- `history` property — returns list of (expression, result) tuples from past evaluations"
            ),
            test_code="""
calc = Calculator()
assert abs(calc.evaluate("2 + 3") - 5.0) < 1e-9
assert abs(calc.evaluate("2 + 3 * 4") - 14.0) < 1e-9
assert abs(calc.evaluate("10 / 2 - 1") - 4.0) < 1e-9
assert abs(calc.evaluate("10 - 2 * 3") - 4.0) < 1e-9
assert len(calc.history) == 4
assert calc.history[0] == ("2 + 3", 5.0)
""",
        ),
        MultiTurnStep(
            step_id="t3_02_step2",
            func_name="Calculator",
            prompt=(
                "Extend the Calculator to support parentheses:\n"
                "- `evaluate('(2 + 3) * 4')` should return 20.0\n"
                "- Nested parentheses must work: `((2 + 3)) * (4 - 1)` = 15.0\n"
                "- Parentheses override normal operator precedence\n"
                "Keep all existing functionality working."
            ),
            test_code="""
calc = Calculator()
assert abs(calc.evaluate("(2 + 3) * 4") - 20.0) < 1e-9
assert abs(calc.evaluate("((2 + 3)) * (4 - 1)") - 15.0) < 1e-9
assert abs(calc.evaluate("(10 - 2) * (3 + 1)") - 32.0) < 1e-9
assert abs(calc.evaluate("2 + 3 * 4") - 14.0) < 1e-9
assert abs(calc.evaluate("100 / (5 * 4)") - 5.0) < 1e-9
""",
        ),
        MultiTurnStep(
            step_id="t3_02_step3",
            func_name="Calculator",
            prompt=(
                "Add variable support to the Calculator:\n"
                "- `set_var(name: str, value: float)` — store a variable\n"
                "- `get_var(name: str) -> float` — retrieve a variable, raise KeyError if not found\n"
                "- Variables can be used in expressions: if x=5, then `evaluate('x + 3')` returns 8.0\n"
                "- Variable names are alphabetic strings (a-z, A-Z)\n"
                "- Assignment in expressions: `evaluate('x = 5')` should set x to 5 and return 5.0\n"
                "Keep all existing functionality (precedence, parentheses, history) working."
            ),
            test_code="""
calc = Calculator()
calc.set_var("x", 10)
assert abs(calc.evaluate("x + 5") - 15.0) < 1e-9
assert abs(calc.evaluate("x * 2") - 20.0) < 1e-9

assert abs(calc.evaluate("y = 7") - 7.0) < 1e-9
assert abs(calc.get_var("y") - 7.0) < 1e-9
assert abs(calc.evaluate("x + y") - 17.0) < 1e-9

assert abs(calc.evaluate("(x + y) * 2") - 34.0) < 1e-9

try:
    calc.evaluate("unknown_var + 1")
    assert False, "Should raise KeyError"
except (KeyError, NameError, ValueError):
    pass
""",
        ),
        MultiTurnStep(
            step_id="t3_02_step4",
            func_name="Calculator",
            prompt=(
                "Add built-in math functions to the Calculator:\n"
                "- Support: `sqrt(x)`, `abs(x)`, `min(x, y)`, `max(x, y)`, `pow(x, y)`\n"
                "- Functions work inside expressions: `sqrt(16) + 1` = 5.0\n"
                "- Functions can nest: `sqrt(abs(-16))` = 4.0\n"
                "- Functions can use variables: if x=25, `sqrt(x)` = 5.0\n"
                "Use Python's `math` module for sqrt. Keep everything else working."
            ),
            test_code="""
import math

calc = Calculator()
assert abs(calc.evaluate("sqrt(16)") - 4.0) < 1e-9
assert abs(calc.evaluate("abs(-5)") - 5.0) < 1e-9
assert abs(calc.evaluate("sqrt(16) + 1") - 5.0) < 1e-9
assert abs(calc.evaluate("max(3, 7)") - 7.0) < 1e-9
assert abs(calc.evaluate("min(3, 7)") - 3.0) < 1e-9
assert abs(calc.evaluate("pow(2, 10)") - 1024.0) < 1e-9

calc.set_var("x", 25)
assert abs(calc.evaluate("sqrt(x)") - 5.0) < 1e-9
assert abs(calc.evaluate("sqrt(abs(-16))") - 4.0) < 1e-9
""",
        ),
    ]

    tests.append(MultiTurnTest(
        test_id="t3_02_calculator",
        description="Build a Calculator step by step: basic -> parentheses -> variables -> functions",
        steps=calc_steps,
    ))

    return tests


# ═══════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StressTestResult:
    test_id: str
    tier: int
    prompt: str
    response: str
    extracted_code: str
    exec_error: str
    tests_passed: int
    tests_total: int
    test_errors: list[str]
    response_time: float
    score: float


@dataclass
class MultiTurnStepResult:
    step_id: str
    prompt_sent: str
    response: str
    extracted_code: str
    accumulated_code: str
    exec_error: str
    tests_passed: int
    tests_total: int
    test_errors: list[str]
    response_time: float
    score: float


@dataclass
class MultiTurnTestResult:
    test_id: str
    description: str
    steps_completed: int
    steps_total: int
    step_results: list[MultiTurnStepResult]
    overall_score: float   # average of step scores
    total_time: float


@dataclass
class StressModelResult:
    model_name: str
    model_path: str
    model_size_mb: float
    chat_format: str
    tier1_results: list[StressTestResult] = field(default_factory=list)
    tier2_results: list[StressTestResult] = field(default_factory=list)
    tier3_results: list[MultiTurnTestResult] = field(default_factory=list)
    tier1_score: float = 0.0
    tier2_score: float = 0.0
    tier3_score: float = 0.0
    overall_score: float = 0.0
    total_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

class _ModelShim:
    """Wraps raw llama model for AugmentorRouter compatibility."""
    def __init__(self, model, max_tokens, temperature):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.total_time = 0

    def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
        mt = max_tokens or self.max_tokens
        temp = temperature or self.temperature
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]
        kwargs.pop("grammar", None)
        start = time.monotonic()
        output = self.model(prompt, max_tokens=mt, temperature=temp, stop=stop, echo=False, **kwargs)
        self.total_time += time.monotonic() - start
        return output["choices"][0]["text"].strip()

    def count_tokens(self, text):
        try:
            return len(self.model.tokenize(text.encode("utf-8")))
        except Exception:
            return len(text) // 4


class StressBenchmarkRunner:
    def __init__(self, gpu_layers: int = 99, threads: int = 8, context_length: int = 4096,
                 use_augmentors: bool = False):
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.context_length = context_length
        self.use_augmentors = use_augmentors
        self.all_results: list[StressModelResult] = []

    def _get_augmentor_router(self):
        """Create a stress-targeted augmentor router."""
        if not self.use_augmentors:
            return None
        from engine.augmentors import AugmentorRouter
        router = AugmentorRouter(stress=True)
        try:
            from engine.embedder import get_embedder
            embedder = get_embedder()
            if embedder:
                router.init_embeddings(embedder)
        except Exception:
            pass  # Falls back to positional example selection
        return router

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

    def generate(self, model, prompt: str, max_tokens: int = 1024,
                 temperature: float = 0.2) -> tuple[str, float]:
        """Generate a response. Returns (text, elapsed_seconds)."""
        stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>", "\nUser:", "\nHuman:"]
        start = time.monotonic()
        output = model(prompt, max_tokens=max_tokens, temperature=temperature,
                       stop=stop, echo=False)
        elapsed = time.monotonic() - start
        text = output["choices"][0]["text"].strip()
        return text, elapsed

    def run_single_test(self, model, test: StressTest, chat_format: str,
                        augmentor_router=None) -> StressTestResult:
        """Run a single-shot test (tier 1 or 2). Uses augmentor router if provided."""
        if augmentor_router:
            shim = _ModelShim(model, test.max_tokens, 0.2)
            augmentor_result = augmentor_router.process(
                query=test.prompt, model=shim, chat_format=chat_format,
                module_hint="code_gen",
                gen_kwargs={"max_tokens": test.max_tokens, "temperature": 0.2},
            )
            if augmentor_result:
                response = augmentor_result.response
                elapsed = shim.total_time
            else:
                # Fallback to direct
                system = (
                    "You are a Python coding assistant. Write clean, correct, complete Python code. "
                    "Return ONLY the code in a ```python block. No explanation. "
                    "Include all necessary imports. The code must be runnable as-is."
                )
                prompt = wrap_chat(system, test.prompt, chat_format)
                response, elapsed = self.generate(model, prompt, max_tokens=test.max_tokens)
        else:
            system = (
                "You are a Python coding assistant. Write clean, correct, complete Python code. "
                "Return ONLY the code in a ```python block. No explanation. "
                "Include all necessary imports. The code must be runnable as-is."
            )
            prompt = wrap_chat(system, test.prompt, chat_format)
            response, elapsed = self.generate(model, prompt, max_tokens=test.max_tokens)

        code = extract_code(response)
        namespace, exec_error = safe_exec(code)

        if not exec_error and test.func_name not in namespace:
            exec_error = f"'{test.func_name}' not found in generated code"

        if exec_error:
            total = test.test_code.strip().count("assert")
            return StressTestResult(
                test_id=test.test_id, tier=test.tier, prompt=test.prompt,
                response=response, extracted_code=code, exec_error=exec_error,
                tests_passed=0, tests_total=total, test_errors=[exec_error],
                response_time=elapsed, score=0.0,
            )

        passed, total, errors = run_tests(namespace, test.test_code)
        return StressTestResult(
            test_id=test.test_id, tier=test.tier, prompt=test.prompt,
            response=response, extracted_code=code, exec_error="",
            tests_passed=passed, tests_total=total, test_errors=errors,
            response_time=elapsed, score=passed / total if total > 0 else 0.0,
        )

    def run_multi_turn_test(self, model, test: MultiTurnTest,
                            chat_format: str,
                            augmentor_router=None) -> MultiTurnTestResult:
        """Run a multi-turn test — each step builds on accumulated code."""
        step_results = []
        accumulated_code = ""
        total_time = 0.0

        for i, step in enumerate(test.steps):
            # Build prompt with context of previous code
            if accumulated_code:
                user_msg = (
                    f"Here is the current code:\n\n```python\n{accumulated_code}\n```\n\n"
                    f"Now extend it with the following changes. Return the COMPLETE updated code "
                    f"(not just the changes) in a ```python block:\n\n{step.prompt}"
                )
            else:
                user_msg = step.prompt

            # Use augmentor router if available (especially valuable for first step)
            if augmentor_router:
                shim = _ModelShim(model, step.max_tokens, 0.2)
                augmentor_result = augmentor_router.process(
                    query=user_msg, model=shim, chat_format=chat_format,
                    module_hint="code_gen",
                    gen_kwargs={"max_tokens": step.max_tokens, "temperature": 0.2},
                )
                if augmentor_result:
                    response = augmentor_result.response
                    elapsed = shim.total_time
                else:
                    system = (
                        "You are a Python coding assistant. Write clean, correct, complete Python code. "
                        "Return ONLY the full code in a ```python block. No explanation. "
                        "Include all necessary imports. The code must be runnable as-is."
                    )
                    prompt = wrap_chat(system, user_msg, chat_format)
                    response, elapsed = self.generate(model, prompt, max_tokens=step.max_tokens)
            else:
                system = (
                    "You are a Python coding assistant. Write clean, correct, complete Python code. "
                    "Return ONLY the full code in a ```python block. No explanation. "
                    "Include all necessary imports. The code must be runnable as-is."
                )
                prompt = wrap_chat(system, user_msg, chat_format)
                response, elapsed = self.generate(model, prompt, max_tokens=step.max_tokens)

            total_time += elapsed

            code = extract_code(response)
            namespace, exec_error = safe_exec(code)

            if not exec_error and step.func_name not in namespace:
                exec_error = f"'{step.func_name}' not found in generated code"

            if exec_error:
                total = step.test_code.strip().count("assert")
                step_results.append(MultiTurnStepResult(
                    step_id=step.step_id, prompt_sent=user_msg, response=response,
                    extracted_code=code, accumulated_code=accumulated_code,
                    exec_error=exec_error, tests_passed=0, tests_total=total,
                    test_errors=[exec_error], response_time=elapsed, score=0.0,
                ))
                # Don't update accumulated_code on failure — keep previous good code
                # But still try subsequent steps (they might partially work)
                continue

            passed, total, errors = run_tests(namespace, step.test_code)
            score = passed / total if total > 0 else 0.0

            step_results.append(MultiTurnStepResult(
                step_id=step.step_id, prompt_sent=user_msg, response=response,
                extracted_code=code, accumulated_code=accumulated_code,
                exec_error="", tests_passed=passed, tests_total=total,
                test_errors=errors, response_time=elapsed, score=score,
            ))

            # Update accumulated code for next step (even if partial pass)
            if score > 0:
                accumulated_code = code

        steps_completed = sum(1 for sr in step_results if sr.score == 1.0)
        overall_score = (sum(sr.score for sr in step_results) / len(step_results)
                         if step_results else 0.0)

        return MultiTurnTestResult(
            test_id=test.test_id, description=test.description,
            steps_completed=steps_completed, steps_total=len(test.steps),
            step_results=step_results, overall_score=overall_score,
            total_time=total_time,
        )

    def run_model(self, model_path: Path, tiers: list[int] = None,
                  quick: bool = False) -> StressModelResult:
        """Run all requested tiers for a single model."""
        if tiers is None:
            tiers = [1, 2, 3]

        model_name = model_path.stem
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        chat_format = detect_chat_format(str(model_path))

        mode_str = "AUGMENTORS" if self.use_augmentors else "DIRECT"
        print(f"\n{'='*70}")
        print(f"  Model: {model_name} [{mode_str}]")
        print(f"  Size: {model_size_mb:.0f} MB | Format: {chat_format}")
        print(f"  Tiers: {tiers}")
        print(f"{'='*70}")

        model = self.load_model(model_path)

        # Create augmentor router if requested
        augmentor_router = self._get_augmentor_router() if self.use_augmentors else None

        result = StressModelResult(
            model_name=model_name, model_path=str(model_path),
            model_size_mb=model_size_mb, chat_format=chat_format,
        )

        # Tier 1
        if 1 in tiers:
            tier1_tests = build_tier1_tests()
            if quick:
                tier1_tests = tier1_tests[:1]
            print(f"\n  -- Tier 1: Medium ({len(tier1_tests)} tests) --")
            for test in tier1_tests:
                print(f"    {test.test_id}...", end=" ", flush=True)
                tr = self.run_single_test(model, test, chat_format, augmentor_router=augmentor_router)
                result.tier1_results.append(tr)
                status = f"{tr.tests_passed}/{tr.tests_total}"
                icon = "PASS" if tr.score == 1.0 else ("PARTIAL" if tr.score > 0 else "FAIL")
                print(f"{icon} ({status}) [{tr.response_time:.1f}s]")
                if tr.test_errors:
                    for err in tr.test_errors[:2]:
                        print(f"      {err[:100]}")

            if result.tier1_results:
                result.tier1_score = sum(r.score for r in result.tier1_results) / len(result.tier1_results)

        # Tier 2
        if 2 in tiers:
            tier2_tests = build_tier2_tests()
            if quick:
                tier2_tests = tier2_tests[:1]
            print(f"\n  -- Tier 2: Heavy ({len(tier2_tests)} tests) --")
            for test in tier2_tests:
                print(f"    {test.test_id}...", end=" ", flush=True)
                tr = self.run_single_test(model, test, chat_format, augmentor_router=augmentor_router)
                result.tier2_results.append(tr)
                status = f"{tr.tests_passed}/{tr.tests_total}"
                icon = "PASS" if tr.score == 1.0 else ("PARTIAL" if tr.score > 0 else "FAIL")
                print(f"{icon} ({status}) [{tr.response_time:.1f}s]")
                if tr.test_errors:
                    for err in tr.test_errors[:2]:
                        print(f"      {err[:100]}")

            if result.tier2_results:
                result.tier2_score = sum(r.score for r in result.tier2_results) / len(result.tier2_results)

        # Tier 3
        if 3 in tiers:
            tier3_tests = build_tier3_tests()
            if quick:
                tier3_tests = tier3_tests[:1]
            print(f"\n  -- Tier 3: Multi-Turn ({len(tier3_tests)} tests) --")
            for test in tier3_tests:
                print(f"    {test.test_id} ({len(test.steps)} steps):")
                tr = self.run_multi_turn_test(model, test, chat_format,
                                              augmentor_router=augmentor_router)
                result.tier3_results.append(tr)
                for sr in tr.step_results:
                    status = f"{sr.tests_passed}/{sr.tests_total}"
                    icon = "PASS" if sr.score == 1.0 else ("PARTIAL" if sr.score > 0 else "FAIL")
                    print(f"      {sr.step_id}: {icon} ({status}) [{sr.response_time:.1f}s]")
                    if sr.test_errors:
                        for err in sr.test_errors[:2]:
                            print(f"        {err[:100]}")
                print(f"      Overall: {tr.overall_score:.1%} ({tr.steps_completed}/{tr.steps_total} perfect)")

            if result.tier3_results:
                result.tier3_score = sum(r.overall_score for r in result.tier3_results) / len(result.tier3_results)

        # Overall
        scores = []
        if result.tier1_results:
            scores.append(result.tier1_score)
        if result.tier2_results:
            scores.append(result.tier2_score)
        if result.tier3_results:
            scores.append(result.tier3_score)
        result.overall_score = sum(scores) / len(scores) if scores else 0.0
        result.total_time = sum(
            r.response_time for r in result.tier1_results + result.tier2_results
        ) + sum(r.total_time for r in result.tier3_results)

        # Unload
        del model
        gc.collect()

        self.all_results.append(result)
        return result


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def print_summary(results: list[StressModelResult]):
    """Print a summary table of all model results."""
    print(f"\n{'='*80}")
    print(f"  STRESS BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    # Header
    print(f"  {'Model':<40} {'Tier 1':>8} {'Tier 2':>8} {'Tier 3':>8} {'Overall':>8} {'Time':>7}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

    for r in sorted(results, key=lambda x: x.overall_score, reverse=True):
        t1 = f"{r.tier1_score:.0%}" if r.tier1_results else "—"
        t2 = f"{r.tier2_score:.0%}" if r.tier2_results else "—"
        t3 = f"{r.tier3_score:.0%}" if r.tier3_results else "—"
        overall = f"{r.overall_score:.0%}"
        time_str = f"{r.total_time:.0f}s"
        print(f"  {r.model_name:<40} {t1:>8} {t2:>8} {t3:>8} {overall:>8} {time_str:>7}")

    print()

    # Detailed breakdown per tier
    for r in sorted(results, key=lambda x: x.overall_score, reverse=True):
        print(f"\n  {r.model_name}")
        print(f"  {'-'*60}")

        if r.tier1_results:
            print(f"  Tier 1 — Medium ({r.tier1_score:.0%}):")
            for tr in r.tier1_results:
                icon = "+" if tr.score == 1.0 else ("~" if tr.score > 0 else "X")
                print(f"    [{icon}] {tr.test_id}: {tr.tests_passed}/{tr.tests_total} "
                      f"({tr.score:.0%}) [{tr.response_time:.1f}s]")

        if r.tier2_results:
            print(f"  Tier 2 — Heavy ({r.tier2_score:.0%}):")
            for tr in r.tier2_results:
                icon = "+" if tr.score == 1.0 else ("~" if tr.score > 0 else "X")
                print(f"    [{icon}] {tr.test_id}: {tr.tests_passed}/{tr.tests_total} "
                      f"({tr.score:.0%}) [{tr.response_time:.1f}s]")

        if r.tier3_results:
            print(f"  Tier 3 — Multi-Turn ({r.tier3_score:.0%}):")
            for tr in r.tier3_results:
                print(f"    {tr.test_id} ({tr.steps_completed}/{tr.steps_total} steps perfect, "
                      f"{tr.overall_score:.0%}):")
                for sr in tr.step_results:
                    icon = "+" if sr.score == 1.0 else ("~" if sr.score > 0 else "X")
                    print(f"      [{icon}] {sr.step_id}: {sr.tests_passed}/{sr.tests_total} "
                          f"({sr.score:.0%}) [{sr.response_time:.1f}s]")


def save_results(results: list[StressModelResult], filepath: str):
    """Save results to JSON."""
    data = []
    for r in results:
        entry = {
            "model_name": r.model_name,
            "model_path": r.model_path,
            "model_size_mb": r.model_size_mb,
            "chat_format": r.chat_format,
            "tier1_score": r.tier1_score,
            "tier2_score": r.tier2_score,
            "tier3_score": r.tier3_score,
            "overall_score": r.overall_score,
            "total_time": r.total_time,
            "tier1_results": [
                {
                    "test_id": tr.test_id, "score": tr.score,
                    "tests_passed": tr.tests_passed, "tests_total": tr.tests_total,
                    "response_time": tr.response_time,
                    "test_errors": tr.test_errors,
                    "exec_error": tr.exec_error,
                }
                for tr in r.tier1_results
            ],
            "tier2_results": [
                {
                    "test_id": tr.test_id, "score": tr.score,
                    "tests_passed": tr.tests_passed, "tests_total": tr.tests_total,
                    "response_time": tr.response_time,
                    "test_errors": tr.test_errors,
                    "exec_error": tr.exec_error,
                }
                for tr in r.tier2_results
            ],
            "tier3_results": [
                {
                    "test_id": tr.test_id, "description": tr.description,
                    "steps_completed": tr.steps_completed, "steps_total": tr.steps_total,
                    "overall_score": tr.overall_score, "total_time": tr.total_time,
                    "steps": [
                        {
                            "step_id": sr.step_id, "score": sr.score,
                            "tests_passed": sr.tests_passed, "tests_total": sr.tests_total,
                            "response_time": sr.response_time,
                            "test_errors": sr.test_errors,
                            "exec_error": sr.exec_error,
                        }
                        for sr in tr.step_results
                    ],
                }
                for tr in r.tier3_results
            ],
        }
        data.append(entry)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {filepath}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stress Benchmark — Push models to their limits")
    parser.add_argument("--model", type=str, help="Specific model path")
    parser.add_argument("--all", action="store_true", help="Test all models on disk")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Run only one tier")
    parser.add_argument("--quick", action="store_true", help="1 test per tier (fast check)")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--output", type=str, default="benchmark_stress_results.json",
                        help="Output JSON file")
    parser.add_argument("--augmentors", action="store_true",
                        help="Use stress-targeted augmentor system (few-shot examples)")
    parser.add_argument("--list-tests", action="store_true", help="List all tests and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.list_tests:
        print("\nTier 1 — Medium:")
        for t in build_tier1_tests():
            print(f"  {t.test_id}: {t.func_name} (max_tokens={t.max_tokens})")
        print("\nTier 2 — Heavy:")
        for t in build_tier2_tests():
            print(f"  {t.test_id}: {t.func_name} (max_tokens={t.max_tokens})")
        print("\nTier 3 — Multi-Turn:")
        for t in build_tier3_tests():
            print(f"  {t.test_id}: {t.description} ({len(t.steps)} steps)")
            for s in t.steps:
                print(f"    {s.step_id}: {s.func_name}")
        return

    # Discover models
    if args.model:
        models = [Path(args.model)]
        if not models[0].exists():
            print(f"Model not found: {args.model}")
            sys.exit(1)
    else:
        models = discover_models(all_models=args.all)

    if not models:
        print("No models found. Use --model or place .gguf files in models/")
        sys.exit(1)

    tiers = [args.tier] if args.tier else [1, 2, 3]

    # Count tests
    test_counts = {1: len(build_tier1_tests()), 2: len(build_tier2_tests()), 3: len(build_tier3_tests())}
    total_tests = sum(test_counts[t] for t in tiers)
    if args.quick:
        total_tests = len(tiers)

    mode_str = "AUGMENTORS" if args.augmentors else "DIRECT"
    print(f"\n  Stress Benchmark [{mode_str}]")
    print(f"  Models: {len(models)} | Tiers: {tiers} | Tests: {total_tests}/model")
    print(f"  {'Quick mode' if args.quick else 'Full run'}")

    runner = StressBenchmarkRunner(
        gpu_layers=args.gpu_layers,
        threads=args.threads,
        context_length=args.context_length,
        use_augmentors=args.augmentors,
    )

    for model_path in models:
        runner.run_model(model_path, tiers=tiers, quick=args.quick)

    print_summary(runner.all_results)
    save_results(runner.all_results, args.output)


if __name__ == "__main__":
    main()
