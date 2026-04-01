#!/usr/bin/env python3
"""
Programmer Pack Benchmark — Tests across 8 programming domains.

Evaluates model performance on iterator protocol, context managers,
descriptor protocol, thread safety, serialization, binary search trees,
text processing, and middleware chain patterns.

16 tests across 8 domains (2 tests each), tier 1 and tier 2.

Usage:
    python benchmark_programmer.py                              # Top models, all tests
    python benchmark_programmer.py --model models/foo.gguf      # Specific model
    python benchmark_programmer.py --all                        # All models on disk
    python benchmark_programmer.py --quick                      # 4 tests (quick check)
    python benchmark_programmer.py --augmentors                 # With programmer pack augmentors
    python benchmark_programmer.py --yaml                       # With YAML-based augmentors
    python benchmark_programmer.py --graph                      # With graph-based retrieval
    python benchmark_programmer.py --compare                    # YAML vs graph comparison
    python benchmark_programmer.py --list-tests                 # Show all test definitions
"""

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_exec import (
    detect_chat_format, wrap_chat, extract_code, safe_exec, run_tests,
    discover_models,
)

from benchmark_stress import (
    StressTest, StressTestResult, StressModelResult, StressBenchmarkRunner,
)

logger = logging.getLogger("bench_programmer")


# ═══════════════════════════════════════════════════════════════════════
# Test definitions — 8 domains, 16 tests
# ═══════════════════════════════════════════════════════════════════════

def build_programmer_tests() -> list[StressTest]:
    tests = []

    # ─────────────────────────────────────────────────────────────────
    # Domain 1: Iterator Protocol
    # ─────────────────────────────────────────────────────────────────

    tests.append(StressTest(
        test_id="t_iter_01_reusable_range",
        tier=1,
        func_name="ReusableRange",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `ReusableRange` that works like range() but "
            "can be iterated multiple times. Implement `__iter__` returning a fresh "
            "iterator each time, supporting start/stop/step parameters.\n"
            "- `ReusableRange(stop)` — iterate from 0 to stop-1\n"
            "- `ReusableRange(start, stop)` — iterate from start to stop-1\n"
            "- `ReusableRange(start, stop, step)` — iterate with given step\n"
            "Each call to `__iter__` must return a new independent iterator."
        ),
        test_code="""\
assert list(ReusableRange(5)) == [0, 1, 2, 3, 4]
assert list(ReusableRange(2, 5)) == [2, 3, 4]
r = ReusableRange(3)
assert list(r) == [0, 1, 2]
assert list(r) == [0, 1, 2]
assert list(ReusableRange(0, 10, 3)) == [0, 3, 6, 9]
assert list(ReusableRange(0)) == []
assert list(ReusableRange(1, 1)) == []
""",
    ))

    tests.append(StressTest(
        test_id="t_iter_02_pipeline",
        tier=1,
        func_name="Pipeline",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `Pipeline` for lazy data processing. "
            "The constructor takes an iterable. Methods:\n"
            "- `.map(fn)` — apply fn to each element (lazy)\n"
            "- `.filter(fn)` — keep elements where fn returns True (lazy)\n"
            "- `.take(n)` — limit to first n elements (lazy)\n"
            "- `.collect()` — materialize results as a list\n"
            "Each method except collect() returns a new Pipeline (chainable). "
            "Operations should be lazy — only evaluated when collect() is called."
        ),
        test_code="""\
assert Pipeline([1, 2, 3, 4, 5]).map(lambda x: x * 2).collect() == [2, 4, 6, 8, 10]
assert Pipeline(range(100)).filter(lambda x: x % 2 == 0).map(lambda x: x * x).take(5).collect() == [0, 4, 16, 36, 64]
assert Pipeline([]).collect() == []
assert Pipeline(range(10)).take(3).collect() == [0, 1, 2]
assert Pipeline([1, 2, 3]).filter(lambda x: x > 5).collect() == []
assert Pipeline([10, 20, 30]).map(str).collect() == ["10", "20", "30"]
""",
    ))

    # ─────────────────────────────────────────────────────────────────
    # Domain 2: Context Manager
    # ─────────────────────────────────────────────────────────────────

    tests.append(StressTest(
        test_id="t_ctx_01_timer",
        tier=1,
        func_name="Timer",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `Timer` that works as a context manager.\n"
            "- `__enter__` starts timing using `time.monotonic()` and returns self\n"
            "- `__exit__` stops timing\n"
            "- After the block, `elapsed` attribute contains the wall-clock time in seconds\n"
            "Import `time` inside your code. The `elapsed` attribute should only be "
            "available after the context manager block completes."
        ),
        test_code="""\
import time
t = Timer()
with t:
    time.sleep(0.05)
assert t.elapsed >= 0.04
assert t.elapsed < 1.0
t2 = Timer()
with t2:
    pass
assert t2.elapsed >= 0
assert t2.elapsed < 0.1
""",
    ))

    tests.append(StressTest(
        test_id="t_ctx_02_transaction",
        tier=2,
        func_name="Transaction",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `Transaction` that works as a context manager "
            "wrapping a dict.\n"
            "- `__init__(self, data: dict)` — store reference to the dict\n"
            "- `__enter__` — snapshot current state of the dict, return the dict itself\n"
            "- On clean exit (no exception): keep all changes made to the dict\n"
            "- On exception: revert the dict to the snapshot state and RE-RAISE the exception\n"
            "Do NOT suppress exceptions. The revert must restore the dict in-place "
            "(clear + update, not rebind)."
        ),
        test_code="""\
store = {"a": 1, "b": 2}
with Transaction(store) as s:
    s["c"] = 3
assert store == {"a": 1, "b": 2, "c": 3}

store2 = {"x": 10, "y": 20}
try:
    with Transaction(store2) as s:
        s["z"] = 30
        s["x"] = 999
        raise ValueError("rollback!")
except ValueError:
    pass
assert store2 == {"x": 10, "y": 20}
assert "z" not in store2

store3 = {}
with Transaction(store3) as s:
    s["key"] = "value"
assert store3 == {"key": "value"}
""",
    ))

    # ─────────────────────────────────────────────────────────────────
    # Domain 3: Descriptor Protocol
    # ─────────────────────────────────────────────────────────────────

    tests.append(StressTest(
        test_id="t_desc_01_typed_field",
        tier=1,
        func_name="TypedField",
        max_tokens=1024,
        prompt=(
            "Write a Python descriptor class called `TypedField` that validates type on set.\n"
            "- `TypedField(expected_type)` — e.g. `TypedField(int)` only allows int values\n"
            "- Implement `__set_name__(self, owner, name)` to capture the attribute name\n"
            "- Implement `__get__(self, obj, objtype=None)` — return the stored value from the instance, "
            "or the descriptor itself if accessed from the class\n"
            "- Implement `__set__(self, obj, value)` — validate type, raise TypeError with a "
            "descriptive message if wrong type, otherwise store on the instance\n"
            "Store data on the instance dict (not the descriptor) to keep instances independent."
        ),
        test_code="""\
class Config:
    port = TypedField(int)
    host = TypedField(str)

c1 = Config()
c1.port = 8080
c1.host = "localhost"
assert c1.port == 8080
assert c1.host == "localhost"

c2 = Config()
c2.port = 3000
c2.host = "0.0.0.0"
assert c2.port == 3000
assert c1.port == 8080

try:
    c1.port = "not_an_int"
    assert False, "Should raise TypeError"
except TypeError:
    pass

try:
    c1.host = 12345
    assert False, "Should raise TypeError"
except TypeError:
    pass

assert c1.port == 8080
assert c1.host == "localhost"
""",
    ))

    tests.append(StressTest(
        test_id="t_desc_02_cached_property",
        tier=2,
        func_name="cached_property",
        max_tokens=1024,
        prompt=(
            "Write a Python descriptor class called `cached_property` (without using functools).\n"
            "It acts as a decorator for methods. On first access, call the decorated function, "
            "cache the result on the instance, and return it. Subsequent accesses return the "
            "cached value without re-calling the function.\n"
            "Deleting the attribute (via `del obj.attr`) should remove the cached value so the "
            "next access recomputes it.\n"
            "Implement `__init__(self, func)`, `__set_name__`, and `__get__`. "
            "Store the cache on the instance's `__dict__` using the attribute name."
        ),
        test_code="""\
call_count = 0

class Expensive:
    @cached_property
    def value(self):
        global call_count
        call_count += 1
        return 42

call_count = 0
obj = Expensive()
assert obj.value == 42
assert call_count == 1
assert obj.value == 42
assert call_count == 1

del obj.value
assert obj.value == 42
assert call_count == 2

obj2 = Expensive()
call_count = 0
assert obj2.value == 42
assert call_count == 1
""",
    ))

    # ─────────────────────────────────────────────────────────────────
    # Domain 4: Thread Safety
    # ─────────────────────────────────────────────────────────────────

    tests.append(StressTest(
        test_id="t_thread_01_safe_counter",
        tier=1,
        func_name="ThreadSafeCounter",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `ThreadSafeCounter` with:\n"
            "- `__init__(self)` — starting value is 0\n"
            "- `increment(self, n=1)` — atomically add n to the counter\n"
            "- `decrement(self, n=1)` — atomically subtract n from the counter\n"
            "- `value` property — return the current counter value\n"
            "All operations must be thread-safe using `threading.Lock`. "
            "Import threading in your code."
        ),
        test_code="""\
import threading

c = ThreadSafeCounter()
assert c.value == 0
c.increment()
assert c.value == 1
c.increment(5)
assert c.value == 6
c.decrement(2)
assert c.value == 4

c2 = ThreadSafeCounter()
threads = [threading.Thread(target=lambda: [c2.increment() for _ in range(1000)]) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert c2.value == 10000

c3 = ThreadSafeCounter()
inc_threads = [threading.Thread(target=lambda: [c3.increment() for _ in range(500)]) for _ in range(5)]
dec_threads = [threading.Thread(target=lambda: [c3.decrement() for _ in range(500)]) for _ in range(5)]
for t in inc_threads + dec_threads:
    t.start()
for t in inc_threads + dec_threads:
    t.join()
assert c3.value == 0
""",
    ))

    tests.append(StressTest(
        test_id="t_thread_02_future",
        tier=2,
        func_name="Future",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `Future` with:\n"
            "- `set_result(value)` — store the result, wake any waiting threads\n"
            "- `set_exception(exc)` — store an exception, wake any waiting threads\n"
            "- `result(timeout=None)` — block until result/exception is set. "
            "If timeout expires, raise `TimeoutError`. If exception was set, re-raise it.\n"
            "- `done()` — return True if result or exception has been set\n"
            "Use `threading.Condition` for synchronization. "
            "`set_result` and `set_exception` can only be called once — raise "
            "`RuntimeError` on a second call to either. Import threading in your code."
        ),
        test_code="""\
import threading, time

f1 = Future()
assert f1.done() == False

def setter():
    time.sleep(0.05)
    f1.set_result(42)

t = threading.Thread(target=setter)
t.start()
assert f1.result(timeout=2) == 42
assert f1.done() == True
t.join()

try:
    f1.set_result(99)
    assert False, "Should raise RuntimeError"
except RuntimeError:
    pass

try:
    f1.set_exception(ValueError("x"))
    assert False, "Should raise RuntimeError"
except RuntimeError:
    pass

f2 = Future()
try:
    f2.result(timeout=0.05)
    assert False, "Should raise TimeoutError"
except TimeoutError:
    pass

f3 = Future()
f3.set_exception(ValueError("boom"))
assert f3.done() == True
try:
    f3.result()
    assert False, "Should re-raise ValueError"
except ValueError as e:
    assert "boom" in str(e)
""",
    ))

    # ─────────────────────────────────────────────────────────────────
    # Domain 5: Serialization
    # ─────────────────────────────────────────────────────────────────

    tests.append(StressTest(
        test_id="t_serial_01_schema_validate",
        tier=1,
        func_name="validate",
        max_tokens=1024,
        prompt=(
            "Write a Python function called `validate(data: dict, schema: dict) -> list[str]` "
            "that validates a dict against a schema.\n"
            "Schema format: each key maps to a dict with:\n"
            "  - `\"type\"`: one of `int`, `str`, `float`, `bool`, `list`, `dict` (the actual Python types)\n"
            "  - `\"required\"`: True or False\n"
            "Return a list of error strings:\n"
            "  - For missing required fields: `\"Missing required field: <name>\"`\n"
            "  - For wrong types: `\"Field '<name>': expected <type>, got <type>\"`\n"
            "Empty list means the data is valid. Extra fields in data (not in schema) are OK."
        ),
        test_code="""\
schema = {
    "name": {"type": str, "required": True},
    "age": {"type": int, "required": True},
    "email": {"type": str, "required": False},
}

errors = validate({"name": "Alice", "age": 30}, schema)
assert errors == []

errors = validate({"name": "Alice", "age": 30, "extra": True}, schema)
assert errors == []

errors = validate({"name": "Alice"}, schema)
assert len(errors) == 1
assert "age" in errors[0]

errors = validate({"name": "Alice", "age": "thirty"}, schema)
assert len(errors) == 1
assert "age" in errors[0]

errors = validate({}, schema)
assert len(errors) == 2

errors = validate({"name": 123, "age": "bad"}, schema)
assert len(errors) == 2
""",
    ))

    tests.append(StressTest(
        test_id="t_serial_02_roundtrip",
        tier=2,
        func_name="serialize",
        max_tokens=2048,
        prompt=(
            "Write two Python functions:\n"
            "1. `serialize(obj) -> dict` — convert an object to a dict. Use `obj.__dict__` for "
            "attributes. For nested objects, recursively serialize. For lists of objects, serialize "
            "each element. Primitives (str, int, float, bool, None) stay as-is. Add a `\"__class__\"` "
            "key with the class name.\n"
            "2. `deserialize(data: dict, cls) -> object` — create an instance of `cls` and set "
            "attributes from the dict. The class may have a `__serialize_types__` class attribute "
            "that maps field names to types for nested deserialization. For example:\n"
            "  `__serialize_types__ = {\"address\": Address, \"tags\": [Tag]}`\n"
            "  A list type like `[Tag]` means the field is a list of Tag objects.\n"
            "If `__serialize_types__` is missing or a field isn't listed, set the value directly."
        ),
        test_code="""\
class Address:
    def __init__(self, city="", zip_code=""):
        self.city = city
        self.zip_code = zip_code

class Person:
    __serialize_types__ = {"address": Address}
    def __init__(self, name="", age=0, address=None):
        self.name = name
        self.age = age
        self.address = address

class Team:
    __serialize_types__ = {"members": [Person]}
    def __init__(self, name="", members=None):
        self.name = name
        self.members = members or []

p = Person("Alice", 30, Address("NYC", "10001"))
d = serialize(p)
assert d["name"] == "Alice"
assert d["age"] == 30
assert d["address"]["city"] == "NYC"
assert "__class__" in d

p2 = deserialize(d, Person)
assert p2.name == "Alice"
assert p2.age == 30
assert p2.address.city == "NYC"
assert p2.address.zip_code == "10001"

team = Team("Dev", [Person("Bob", 25), Person("Carol", 28)])
td = serialize(team)
assert td["name"] == "Dev"
assert len(td["members"]) == 2
assert td["members"][0]["name"] == "Bob"

team2 = deserialize(td, Team)
assert team2.name == "Dev"
assert len(team2.members) == 2
assert team2.members[0].name == "Bob"
assert team2.members[1].name == "Carol"
""",
    ))

    # ─────────────────────────────────────────────────────────────────
    # Domain 6: Binary Search Tree
    # ─────────────────────────────────────────────────────────────────

    tests.append(StressTest(
        test_id="t_tree_01_bst",
        tier=1,
        func_name="BST",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `BST` (Binary Search Tree) with:\n"
            "- `insert(val)` — insert a value\n"
            "- `search(val) -> bool` — return True if value exists\n"
            "- `delete(val)` — remove a value. Handle all 3 cases: leaf node, one child, "
            "two children (use in-order successor for two-child case)\n"
            "- `inorder() -> list` — return values in sorted order (in-order traversal)\n"
            "- `height() -> int` — return the height of the tree (empty tree = 0, single node = 1)\n"
            "The tree starts empty. Use a helper Node class or nested class internally."
        ),
        test_code="""\
bst = BST()
for v in [5, 3, 7, 1, 4, 6, 8]:
    bst.insert(v)

assert bst.inorder() == [1, 3, 4, 5, 6, 7, 8]
assert bst.search(4) == True
assert bst.search(99) == False
assert bst.height() >= 3

bst.delete(3)
assert 3 not in bst.inorder()
assert bst.inorder() == sorted(bst.inorder())

bst.delete(5)
result = bst.inorder()
assert 5 not in result
assert result == sorted(result)

bst.delete(99)
assert bst.inorder() == sorted(bst.inorder())

empty = BST()
assert empty.inorder() == []
assert empty.height() == 0
assert empty.search(1) == False
""",
    ))

    tests.append(StressTest(
        test_id="t_tree_02_traversals",
        tier=1,
        func_name="BinaryTree",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `BinaryTree` constructed from nested tuples.\n"
            "Format: `(value, left, right)` where left and right are either None or nested tuples.\n"
            "Methods:\n"
            "- `preorder() -> list` — root, left, right (recursive)\n"
            "- `inorder() -> list` — left, root, right (recursive)\n"
            "- `postorder() -> list` — left, right, root (recursive)\n"
            "- `level_order() -> list` — breadth-first, level by level left to right (use a queue)\n"
            "Example: `BinaryTree((1, (2, None, None), (3, None, None)))` builds a tree with "
            "root=1, left=2, right=3."
        ),
        test_code="""\
tree = BinaryTree((1, (2, (4, None, None), (5, None, None)), (3, None, (6, None, None))))
assert tree.preorder() == [1, 2, 4, 5, 3, 6]
assert tree.inorder() == [4, 2, 5, 1, 3, 6]
assert tree.postorder() == [4, 5, 2, 6, 3, 1]
assert tree.level_order() == [1, 2, 3, 4, 5, 6]

single = BinaryTree((42, None, None))
assert single.preorder() == [42]
assert single.inorder() == [42]
assert single.postorder() == [42]
assert single.level_order() == [42]

left_only = BinaryTree((1, (2, (3, None, None), None), None))
assert left_only.preorder() == [1, 2, 3]
assert left_only.inorder() == [3, 2, 1]
assert left_only.postorder() == [3, 2, 1]
assert left_only.level_order() == [1, 2, 3]
""",
    ))

    # ─────────────────────────────────────────────────────────────────
    # Domain 7: Text Processing
    # ─────────────────────────────────────────────────────────────────

    tests.append(StressTest(
        test_id="t_text_01_template",
        tier=2,
        func_name="render",
        max_tokens=2048,
        prompt=(
            "Write a Python function called `render(template: str, context: dict) -> str` "
            "that processes a simple template language.\n"
            "Supported syntax:\n"
            "- `{{var}}` — variable substitution from context. If var not in context, "
            "replace with empty string.\n"
            "- `{% for x in list_var %}...{% endfor %}` — loop block. `list_var` is looked "
            "up in context (must be a list). The body is rendered once per item with `x` "
            "added to the context.\n"
            "- Variables inside loops should work: `{{x}}` refers to the loop variable.\n"
            "- Spaces around `{{` and `}}` or `{%` and `%}` should be flexible.\n"
            "Nested loops are NOT required. Keep it simple but correct."
        ),
        test_code="""\
assert render("Hello {{name}}", {"name": "World"}) == "Hello World"
assert render("{{a}} + {{b}}", {"a": "1", "b": "2"}) == "1 + 2"
assert render("{{missing}}", {}) == ""

result = render("{% for x in items %}{{x}} {% endfor %}", {"items": ["a", "b", "c"]})
assert "a" in result
assert "b" in result
assert "c" in result

result2 = render("Users: {% for u in users %}{{u}}, {% endfor %}done", {"users": ["Alice", "Bob"]})
assert "Alice" in result2
assert "Bob" in result2
assert "done" in result2

assert render("No vars here", {}) == "No vars here"
assert render("{% for x in items %}{{x}}{% endfor %}", {"items": []}) == ""
""",
    ))

    tests.append(StressTest(
        test_id="t_text_02_glob_match",
        tier=1,
        func_name="glob_match",
        max_tokens=1024,
        prompt=(
            "Write a Python function called `glob_match(pattern: str, text: str) -> bool` "
            "that matches a glob pattern against text.\n"
            "Supported wildcards:\n"
            "- `*` matches any sequence of characters (including empty)\n"
            "- `?` matches exactly one character\n"
            "All other characters match literally.\n"
            "Use dynamic programming for correctness. Handle edge cases: empty pattern, "
            "empty text, consecutive wildcards."
        ),
        test_code="""\
assert glob_match("*.py", "test.py") == True
assert glob_match("*.py", "test.js") == False
assert glob_match("t?st", "test") == True
assert glob_match("t?st", "tst") == False
assert glob_match("a*b", "aXXXb") == True
assert glob_match("a*b", "aXXXc") == False
assert glob_match("", "") == True
assert glob_match("", "x") == False
assert glob_match("*", "anything") == True
assert glob_match("*", "") == True
assert glob_match("a*b*c", "aXbYc") == True
assert glob_match("a*b*c", "aXbYd") == False
assert glob_match("?", "a") == True
assert glob_match("?", "") == False
assert glob_match("**", "abc") == True
""",
    ))

    # ─────────────────────────────────────────────────────────────────
    # Domain 8: Middleware Chain
    # ─────────────────────────────────────────────────────────────────

    tests.append(StressTest(
        test_id="t_mw_01_pipeline",
        tier=2,
        func_name="MiddlewarePipeline",
        max_tokens=2048,
        prompt=(
            "Write a Python class called `MiddlewarePipeline` with:\n"
            "- `use(fn)` — add a middleware function to the chain\n"
            "- `execute(request)` — run the middleware chain with the given request\n\n"
            "Each middleware has the signature `fn(request, next_fn)` where:\n"
            "- `request` is the data being processed\n"
            "- `next_fn` is a callable to invoke the next middleware in the chain\n"
            "- Middleware can modify request before calling next_fn\n"
            "- Middleware can modify the response returned by next_fn\n"
            "- Middleware can short-circuit by returning without calling next_fn\n\n"
            "If no middleware is registered or after the last middleware, the request "
            "itself is returned as the response (the terminal handler).\n"
            "Middleware executes in the order added via `use()`."
        ),
        test_code="""\
p = MiddlewarePipeline()
log = []

def mw_upper(request, next_fn):
    request["name"] = request["name"].upper()
    return next_fn(request)

def mw_log(request, next_fn):
    log.append("before")
    result = next_fn(request)
    log.append("after")
    return result

def mw_add_field(request, next_fn):
    request["processed"] = True
    return next_fn(request)

p.use(mw_log)
p.use(mw_upper)
p.use(mw_add_field)

result = p.execute({"name": "alice"})
assert result["name"] == "ALICE"
assert result["processed"] == True
assert log == ["before", "after"]

p2 = MiddlewarePipeline()
order = []

def mw_first(req, nxt):
    order.append(1)
    res = nxt(req)
    order.append(4)
    return res

def mw_second(req, nxt):
    order.append(2)
    res = nxt(req)
    order.append(3)
    return res

p2.use(mw_first)
p2.use(mw_second)
p2.execute({})
assert order == [1, 2, 3, 4]

p3 = MiddlewarePipeline()
def mw_short(req, nxt):
    return {"short": True}

def mw_never(req, nxt):
    raise AssertionError("should not run")

p3.use(mw_short)
p3.use(mw_never)
result3 = p3.execute({"x": 1})
assert result3 == {"short": True}
""",
    ))

    tests.append(StressTest(
        test_id="t_mw_02_pubsub",
        tier=2,
        func_name="PubSub",
        max_tokens=1024,
        prompt=(
            "Write a Python class called `PubSub` with:\n"
            "- `subscribe(pattern: str, callback)` — register a callback for a topic pattern\n"
            "- `publish(topic: str, data)` — deliver data to all matching subscribers\n"
            "- `unsubscribe(pattern: str, callback)` — remove a specific callback from a pattern\n\n"
            "Pattern matching rules (dot-separated segments):\n"
            "- Exact match: `\"user.login\"` matches only `\"user.login\"`\n"
            "- Wildcard segment `*`: `\"user.*\"` matches `\"user.login\"` and `\"user.logout\"` "
            "but NOT `\"order.created\"` or `\"user.profile.update\"`\n"
            "- Single `\"*\"` alone matches ALL topics\n"
            "Multiple callbacks can be subscribed to the same pattern."
        ),
        test_code="""\
ps = PubSub()
results = []

def on_user(data):
    results.append(("user", data))

def on_all(data):
    results.append(("all", data))

ps.subscribe("user.*", on_user)
ps.subscribe("*", on_all)

ps.publish("user.login", {"id": 1})
assert len(results) == 2
assert ("user", {"id": 1}) in results
assert ("all", {"id": 1}) in results

results.clear()
ps.publish("order.created", {"id": 99})
assert len(results) == 1
assert results[0][0] == "all"

results.clear()
ps.unsubscribe("*", on_all)
ps.publish("user.logout", {"id": 2})
assert len(results) == 1
assert results[0][0] == "user"

results.clear()
ps.unsubscribe("user.*", on_user)
ps.publish("user.login", {"id": 3})
assert len(results) == 0

ps2 = PubSub()
exact_results = []
ps2.subscribe("app.start", lambda d: exact_results.append(d))
ps2.publish("app.start", "go")
ps2.publish("app.stop", "no")
assert exact_results == ["go"]
""",
    ))

    return tests


# ═══════════════════════════════════════════════════════════════════════
# Runner subclass — adapts StressBenchmarkRunner for programmer tests
# ═══════════════════════════════════════════════════════════════════════

class ProgrammerBenchmarkRunner(StressBenchmarkRunner):
    """Runs programmer-pack tests using the stress runner infrastructure."""

    def _get_augmentor_router(self):
        """Create a programmer-targeted augmentor router."""
        any_aug = (self.use_augmentors or self.use_yaml or self.use_graph
                   or self.use_adaptive or self.use_hybrid
                   or self.use_rerank or self.use_rerank1 or self.use_plan
                   or self.use_auto)
        if not any_aug:
            return None
        from engine.augmentors import AugmentorRouter
        router = AugmentorRouter(yaml_dir="data/augmentor_examples")
        try:
            from engine.embedder import get_embedder
            embedder = get_embedder()
            if embedder:
                router.init_embeddings(embedder)
        except Exception:
            pass
        if self.use_rerank1:
            router.use_rerank1_augmentors()
        elif self.use_rerank:
            router.use_rerank_augmentors()
        elif self.use_plan:
            router.use_plan_augmentors()
        elif self.use_adaptive:
            router.use_adaptive_augmentors()
        elif self.use_hybrid:
            router.use_hybrid_augmentors()
        elif self.use_graph:
            router.use_graph_augmentors()
        elif self.use_augmentors:
            router = AugmentorRouter(pack=True)
            try:
                from engine.embedder import get_embedder
                embedder = get_embedder()
                if embedder:
                    router.init_embeddings(embedder)
            except Exception:
                pass
        if self.no_failure_routing:
            router.set_skip_failure_routing(True)
        return router

    def run_model(self, model_path: Path, tiers: list[int] = None,
                  quick: bool = False) -> StressModelResult:
        """Run programmer tests for a single model."""
        model_name = model_path.stem
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        chat_format = detect_chat_format(str(model_path))

        if self.use_auto:
            mode_str = f"AUTO ({'rerank' if model_size_mb >= 1500 else 'rerank1'})"
        elif self.use_rerank1:
            mode_str = "RERANK1"
        elif self.use_rerank:
            mode_str = "RERANK"
        elif self.use_plan:
            mode_str = "PLAN"
        elif self.use_adaptive:
            mode_str = "ADAPTIVE"
        elif self.use_hybrid:
            mode_str = "HYBRID"
        elif self.use_graph:
            mode_str = "GRAPH"
        elif self.use_yaml:
            mode_str = "YAML"
        elif self.use_augmentors:
            mode_str = "AUGMENTORS"
        else:
            mode_str = "DIRECT"
        if self.no_failure_routing:
            mode_str += " (pure-retrieval)"
        print(f"\n{'='*70}")
        print(f"  Model: {model_name} [{mode_str}]")
        print(f"  Size: {model_size_mb:.0f} MB | Format: {chat_format}")
        print(f"  Benchmark: Programmer Pack")
        print(f"{'='*70}")

        model = self.load_model(model_path)
        augmentor_router = self._get_augmentor_router()
        if self.use_auto and augmentor_router:
            augmentor_router.use_auto_augmentors(model_size_mb)

        result = StressModelResult(
            model_name=model_name, model_path=str(model_path),
            model_size_mb=model_size_mb, chat_format=chat_format,
        )

        all_tests = build_programmer_tests()

        # Filter by tier if specified
        if tiers:
            all_tests = [t for t in all_tests if t.tier in tiers]

        if quick:
            # Pick 1 test per domain (every other test = first of each pair)
            all_tests = all_tests[::2][:4]

        # Split into tier1 and tier2 for result storage
        tier1_tests = [t for t in all_tests if t.tier == 1]
        tier2_tests = [t for t in all_tests if t.tier == 2]

        if tier1_tests:
            print(f"\n  -- Tier 1 ({len(tier1_tests)} tests) --")
            for test in tier1_tests:
                print(f"    {test.test_id}...", end=" ", flush=True)
                tr = self.run_single_test(model, test, chat_format,
                                          augmentor_router=augmentor_router)
                result.tier1_results.append(tr)
                status = f"{tr.tests_passed}/{tr.tests_total}"
                icon = "PASS" if tr.score == 1.0 else ("PARTIAL" if tr.score > 0 else "FAIL")
                print(f"{icon} ({status}) [{tr.response_time:.1f}s]")
                if tr.test_errors:
                    for err in tr.test_errors[:2]:
                        print(f"      {err[:100]}")

        if tier2_tests:
            print(f"\n  -- Tier 2 ({len(tier2_tests)} tests) --")
            for test in tier2_tests:
                print(f"    {test.test_id}...", end=" ", flush=True)
                tr = self.run_single_test(model, test, chat_format,
                                          augmentor_router=augmentor_router)
                result.tier2_results.append(tr)
                status = f"{tr.tests_passed}/{tr.tests_total}"
                icon = "PASS" if tr.score == 1.0 else ("PARTIAL" if tr.score > 0 else "FAIL")
                print(f"{icon} ({status}) [{tr.response_time:.1f}s]")
                if tr.test_errors:
                    for err in tr.test_errors[:2]:
                        print(f"      {err[:100]}")

        # Compute scores
        if result.tier1_results:
            result.tier1_score = sum(r.score for r in result.tier1_results) / len(result.tier1_results)
        if result.tier2_results:
            result.tier2_score = sum(r.score for r in result.tier2_results) / len(result.tier2_results)

        scores = []
        if result.tier1_results:
            scores.append(result.tier1_score)
        if result.tier2_results:
            scores.append(result.tier2_score)
        result.overall_score = sum(scores) / len(scores) if scores else 0.0
        result.total_time = sum(
            r.response_time for r in result.tier1_results + result.tier2_results
        )

        # Unload
        del model
        gc.collect()

        self.all_results.append(result)
        return result


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def print_programmer_summary(results: list[StressModelResult]):
    """Print a summary table of programmer benchmark results."""
    print(f"\n{'='*80}")
    print(f"  PROGRAMMER PACK BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    print(f"  {'Model':<40} {'Tier 1':>8} {'Tier 2':>8} {'Overall':>8} {'Time':>7}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

    for r in sorted(results, key=lambda x: x.overall_score, reverse=True):
        t1 = f"{r.tier1_score:.0%}" if r.tier1_results else "---"
        t2 = f"{r.tier2_score:.0%}" if r.tier2_results else "---"
        overall = f"{r.overall_score:.0%}"
        time_str = f"{r.total_time:.0f}s"
        print(f"  {r.model_name:<40} {t1:>8} {t2:>8} {overall:>8} {time_str:>7}")

    print()

    # Domain breakdown
    domains = {
        "Iterator Protocol": ["t_iter_01_reusable_range", "t_iter_02_pipeline"],
        "Context Manager":   ["t_ctx_01_timer", "t_ctx_02_transaction"],
        "Descriptor Protocol": ["t_desc_01_typed_field", "t_desc_02_cached_property"],
        "Thread Safety":     ["t_thread_01_safe_counter", "t_thread_02_future"],
        "Serialization":     ["t_serial_01_schema_validate", "t_serial_02_roundtrip"],
        "Binary Search Tree": ["t_tree_01_bst", "t_tree_02_traversals"],
        "Text Processing":   ["t_text_01_template", "t_text_02_glob_match"],
        "Middleware Chain":   ["t_mw_01_pipeline", "t_mw_02_pubsub"],
    }

    for r in sorted(results, key=lambda x: x.overall_score, reverse=True):
        print(f"\n  {r.model_name}")
        print(f"  {'-'*60}")

        all_results_map = {}
        for tr in r.tier1_results + r.tier2_results:
            all_results_map[tr.test_id] = tr

        for domain_name, test_ids in domains.items():
            domain_scores = []
            for tid in test_ids:
                if tid in all_results_map:
                    domain_scores.append(all_results_map[tid])

            if not domain_scores:
                continue

            avg = sum(tr.score for tr in domain_scores) / len(domain_scores)
            print(f"  {domain_name} ({avg:.0%}):")
            for tr in domain_scores:
                icon = "+" if tr.score == 1.0 else ("~" if tr.score > 0 else "X")
                print(f"    [{icon}] {tr.test_id}: {tr.tests_passed}/{tr.tests_total} "
                      f"({tr.score:.0%}) [{tr.response_time:.1f}s]")


def save_programmer_results(results: list[StressModelResult], filepath: str):
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
        }
        data.append(entry)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {filepath}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Programmer Pack Benchmark — 16 tests across 8 programming domains"
    )
    parser.add_argument("--model", type=str, help="Specific model path")
    parser.add_argument("--all", action="store_true", help="Test all models on disk")
    parser.add_argument("--quick", action="store_true", help="4 tests only (fast check)")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--output", type=str, default="benchmark_programmer_results.json",
                        help="Output JSON file")
    parser.add_argument("--augmentors", action="store_true",
                        help="Use programmer-pack augmentor system (few-shot examples)")
    parser.add_argument("--yaml", action="store_true",
                        help="Use YAML-based augmentors from data/augmentor_examples/")
    parser.add_argument("--graph", action="store_true",
                        help="Use graph-based augmentors (pattern dependency traversal)")
    parser.add_argument("--adaptive", action="store_true",
                        help="Adaptive mode: auto flat/graph per-query based on composite signal")
    parser.add_argument("--hybrid", action="store_true",
                        help="Hybrid mode: try graph first, fall back to flat on failure")
    parser.add_argument("--rerank", action="store_true",
                        help="Graph-rerank mode: flat candidates reranked by graph coherence (1-2 injected)")
    parser.add_argument("--rerank1", action="store_true",
                        help="Graph-rerank1 mode: rerank to single best example")
    parser.add_argument("--plan", action="store_true",
                        help="Graph-plan mode: graph identifies subpatterns, injects only 1 example")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select retrieval strategy by model size (rerank1 for <1.5GB, rerank for >=1.5GB)")
    parser.add_argument("--no-failure-routing", action="store_true",
                        help="Disable FAILURE_PATTERNS bypass — pure similarity retrieval only")
    parser.add_argument("--compare", action="store_true",
                        help="Run all retrieval methods, output comparison table")
    parser.add_argument("--list-tests", action="store_true", help="List all tests and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.list_tests:
        tests = build_programmer_tests()
        domains = {}
        for t in tests:
            # Group by domain prefix
            prefix = t.test_id.split("_")[1]
            domain_map = {
                "iter": "Iterator Protocol",
                "ctx": "Context Manager",
                "desc": "Descriptor Protocol",
                "thread": "Thread Safety",
                "serial": "Serialization",
                "tree": "Binary Search Tree",
                "text": "Text Processing",
                "mw": "Middleware Chain",
            }
            domain = domain_map.get(prefix, prefix)
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(t)

        for domain, domain_tests in domains.items():
            print(f"\n{domain}:")
            for t in domain_tests:
                tier_label = "Tier 1" if t.tier == 1 else "Tier 2"
                print(f"  {t.test_id}: {t.func_name} [{tier_label}] (max_tokens={t.max_tokens})")
        print(f"\nTotal: {len(tests)} tests across {len(domains)} domains")
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

    all_tests = build_programmer_tests()
    test_count = 4 if args.quick else len(all_tests)

    no_fr = getattr(args, 'no_failure_routing', False)

    if args.compare:
        # ── Comparison across retrieval methods ──
        methods = [
            ("direct",   {}),
            ("flat",     {"use_yaml": True}),
            ("rerank",   {"use_rerank": True}),
            ("rerank1",  {"use_rerank1": True}),
            ("plan",     {"use_plan": True}),
            ("auto",     {"use_auto": True}),
        ]
        # Apply no-failure-routing to all methods if set
        if no_fr:
            methods = [(n, {**kw, "no_failure_routing": True}) for n, kw in methods]
        method_names = [m[0] for m in methods]
        print(f"\n  Programmer Pack Benchmark [COMPARE: {' vs '.join(m.upper() for m in method_names)}]")
        print(f"  Models: {len(models)} | Tests: {test_count}/model x {len(methods)} methods")
        print(f"  Domains: 8 (Iterator, Context, Descriptor, Thread, Serial, Tree, Text, Middleware)")
        print(f"  {'Quick mode' if args.quick else 'Full run'}")

        comparison: dict[str, dict[str, StressModelResult]] = {}

        for model_path in models:
            model_name = model_path.stem
            comparison[model_name] = {}

            for method_name, method_kwargs in methods:
                print(f"\n  -- {model_name}: {method_name.upper()} --")
                runner = ProgrammerBenchmarkRunner(
                    gpu_layers=args.gpu_layers, threads=args.threads,
                    context_length=args.context_length,
                    **method_kwargs,
                )
                comparison[model_name][method_name] = runner.run_model(model_path, quick=args.quick)

        print_comparison_table(comparison)
        save_comparison_results(comparison, args.output)
    else:
        if args.auto:
            mode_str = "AUTO"
        elif args.rerank1:
            mode_str = "RERANK1"
        elif args.rerank:
            mode_str = "RERANK"
        elif args.plan:
            mode_str = "PLAN"
        elif args.adaptive:
            mode_str = "ADAPTIVE"
        elif args.hybrid:
            mode_str = "HYBRID"
        elif args.graph:
            mode_str = "GRAPH"
        elif args.yaml:
            mode_str = "YAML"
        elif args.augmentors:
            mode_str = "AUGMENTORS"
        else:
            mode_str = "DIRECT"
        if no_fr:
            mode_str += " (pure-retrieval)"
        print(f"\n  Programmer Pack Benchmark [{mode_str}]")
        print(f"  Models: {len(models)} | Tests: {test_count}/model")
        print(f"  Domains: 8 (Iterator, Context, Descriptor, Thread, Serial, Tree, Text, Middleware)")
        print(f"  {'Quick mode' if args.quick else 'Full run'}")

        runner = ProgrammerBenchmarkRunner(
            gpu_layers=args.gpu_layers,
            threads=args.threads,
            context_length=args.context_length,
            use_augmentors=args.augmentors,
            use_yaml=args.yaml,
            use_graph=args.graph,
            use_adaptive=args.adaptive,
            use_hybrid=args.hybrid,
            use_rerank=args.rerank,
            use_rerank1=args.rerank1,
            use_plan=args.plan,
            use_auto=args.auto,
            no_failure_routing=no_fr,
        )

        for model_path in models:
            runner.run_model(model_path, quick=args.quick)

        print_programmer_summary(runner.all_results)
        save_programmer_results(runner.all_results, args.output)


def print_comparison_table(comparison: dict):
    """Print comparison table across all tested retrieval methods."""
    methods = ["direct", "flat", "graph", "rerank", "rerank1", "plan", "adaptive", "hybrid"]
    # Detect which methods are present
    all_methods = set()
    for methods_dict in comparison.values():
        all_methods.update(methods_dict.keys())
    methods = [m for m in methods if m in all_methods]

    print(f"\n{'='*90}")
    print(f"  RETRIEVAL METHOD COMPARISON: {' vs '.join(m.upper() for m in methods)}")
    print(f"{'='*90}\n")

    header = f"  {'Model':<35}"
    for m in methods:
        header += f" {m.upper():>8}"
    header += f" {'Time(g)':>8} {'Tok(g)':>7}"
    print(header)
    print(f"  {'-'*35}" + f" {'-'*8}" * len(methods) + f" {'-'*8} {'-'*7}")

    for model_name in sorted(comparison.keys()):
        row = f"  {model_name:<35}"
        for m in methods:
            result = comparison[model_name].get(m)
            if result:
                row += f" {result.overall_score:>7.0%}"
            else:
                row += f" {'--':>8}"
        # Show time and tokens for graph (or last method)
        graph_result = comparison[model_name].get("graph") or comparison[model_name].get("flat")
        if graph_result:
            all_results = graph_result.tier1_results + graph_result.tier2_results
            total_tokens = sum(getattr(r, "prompt_tokens", 0) for r in all_results)
            row += f" {graph_result.total_time:>7.0f}s {total_tokens:>7}"
        print(row)

    # Per-test breakdown for each model
    print(f"\n  Per-Test Breakdown (all models):")
    header2 = f"  {'Model':<30} {'Test':<30}"
    for m in methods:
        header2 += f" {m[:5]:>6}"
    header2 += f" {'Best':>6}"
    print(header2)
    print(f"  {'-'*30} {'-'*30}" + f" {'-'*6}" * len(methods) + f" {'-'*6}")

    for model_name in sorted(comparison.keys()):
        tests_by_method: dict[str, dict[str, float]] = {}
        for m in methods:
            result = comparison[model_name].get(m)
            if result:
                for r in result.tier1_results + result.tier2_results:
                    if r.test_id not in tests_by_method:
                        tests_by_method[r.test_id] = {}
                    tests_by_method[r.test_id][m] = r.score

        for test_id in sorted(tests_by_method.keys()):
            scores = tests_by_method[test_id]
            row = f"  {model_name:<30} {test_id:<30}"
            best_score = max(scores.values()) if scores else 0
            for m in methods:
                s = scores.get(m, 0)
                row += f" {s:>5.0%}"
            # Mark best method
            best_methods = [m for m in methods if scores.get(m, 0) == best_score]
            if len(best_methods) == len(methods):
                row += f"   all"
            else:
                row += f" {'+'.join(m[:3] for m in best_methods):>6}"
            print(row)
        if tests_by_method:
            print()  # blank line between models


def save_comparison_results(comparison: dict, output_path: str):
    """Save comparison results to JSON."""
    data = {}
    for model_name, methods in comparison.items():
        data[model_name] = {}
        for method, result in methods.items():
            all_results = result.tier1_results + result.tier2_results
            data[model_name][method] = {
                "overall_score": result.overall_score,
                "tier1_score": result.tier1_score,
                "tier2_score": result.tier2_score,
                "total_time": result.total_time,
                "total_prompt_tokens": sum(getattr(r, "prompt_tokens", 0) for r in all_results),
                "tests": {
                    r.test_id: {
                        "score": r.score,
                        "tests_passed": r.tests_passed,
                        "tests_total": r.tests_total,
                        "response_time": r.response_time,
                        "prompt_tokens": getattr(r, "prompt_tokens", 0),
                    }
                    for r in all_results
                },
            }

    out = Path(output_path).with_suffix(".comparison.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Comparison results saved to: {out}")


if __name__ == "__main__":
    main()
