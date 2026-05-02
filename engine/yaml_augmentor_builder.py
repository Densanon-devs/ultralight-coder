"""
YAML augmentor builder — turns FailureRecord entries into augmentor examples.

The existing augmentor system (engine/augmentors.py) retrieves YAML examples
matching the user's query and injects them into the prompt as in-context
demonstrations. Today's verdict from the GPT-5.5 prompt-lift bench was:
  > the 14B responds well to in-context examples that mirror the desired
  > output shape, NOT to abstract behavioral rules.

This builder closes the loop:
  1. failure_flagger detects a failure pattern
  2. yaml_augmentor_builder writes a YAML example demonstrating the FIX
  3. on the next similar query, augmentors retrieves the example
  4. the model sees the correct pattern in-context

Outputs go to `data/augmentor_examples/_auto_generated/<category>/<sha>.yaml`
where they live in a SEPARATE directory until promoted by a human review.
The retrieval system can be configured to include or exclude this dir.

Schema is the same as the rest of the augmentor library
(domain / category / examples[] with query + solution).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

try:
    from engine.failure_flagger import FailureRecord
except ImportError:  # pragma: no cover
    from failure_flagger import FailureRecord


# ── Templates per category ──────────────────────────────────────────


# Each entry: (domain, category, query_template, solution_template, tags).
# Solutions are CONCRETE working code — the model sees these in-context
# and copies the shape rather than re-deriving it.

_TEMPLATES = {
    "json_quote_leak": dict(
        domain="agentic",
        category="json_array_form_for_multiline_writes",
        query_template=(
            "Write a multi-line file via write_file that contains Python "
            "string literals (single-quoted), f-strings, or embedded \"."
        ),
        solution_template="""\
When writing multi-line Python files via the agent's write_file tool,
ALWAYS use the array form for `content`. One element per line. Inner
single-quoted Python strings stay natural — no JSON escape gymnastics.

WRONG (single string with embedded `'r'` — JSON parser rejects):

    <tool_call>{"name": "write_file", "arguments": {"path": "x.py",
      "content": "with open('x', 'r') as f:\\n    pass\\n"}}</tool_call>

RIGHT (array form, one line per element):

    <tool_call>{"name": "write_file", "arguments": {"path": "x.py",
      "content": ["with open('x', 'r') as f:", "    pass"]}}</tool_call>

The array form sidesteps the entire JSON-escape pain by treating each
element as a complete line. Joins with \\n at write time. Use it for
ANY file with 5+ lines OR any embedded `"` characters.
""",
        tags=["write_file", "json", "escaping", "multi-line", "agentic"],
    ),
    "fstring_nested_quote": dict(
        domain="agentic",
        category="fstring_nested_quotes_python310",
        query_template=(
            "Print or format a value with an f-string that contains a "
            "string literal inside its `{ }` expression."
        ),
        solution_template="""\
Python 3.10/3.11 cannot parse a single-quoted f-string containing another
single-quoted string literal inside its `{ }` block. Same rule for
double-quoted f-strings with embedded double-quoted literals.

WRONG (Python 3.10 SyntaxError — inner `'` closes outer `'`):

    print(f'{todo.id}: {'Done' if todo.done else 'Pending'}')

RIGHT — flip the outer quote style:

    print(f"{todo.id}: {'Done' if todo.done else 'Pending'}")

Or extract the value before the f-string:

    label = 'Done' if todo.done else 'Pending'
    print(f'{todo.id}: {label}')

Default: when an f-string contains string literals inside `{...}`, use
double-quoted f-string + single-quoted inner literals.
""",
        tags=["f-string", "python3.10", "syntax", "nested-quotes"],
    ),
    "missing_import": dict(
        domain="agentic",
        category="prepend_missing_import_after_write",
        query_template=(
            "After writing a Python file, auto_verify reports "
            "'references undefined names: <name>' — add the missing import."
        ),
        solution_template="""\
When auto_verify reports `references undefined names: X`, prepend the
import to the SAME file via edit_file with empty `old_string`. Empty
old_string is the canonical "prepend at top of file" pattern.

WRONG (adding the import to __init__.py or another file):

    <tool_call>{"name": "edit_file", "arguments": {"path": "__init__.py",
      "old_string": "", "new_string": "from module import X\\n"}}</tool_call>

Python imports are per-file. The undefined name is in X.py, so X.py
needs the import — adding it elsewhere doesn't help.

RIGHT (prepend to the file the auto_verify message named):

    <tool_call>{"name": "edit_file", "arguments": {"path": "X.py",
      "old_string": "", "new_string": "from module import X\\n"}}</tool_call>

If MULTIPLE names are undefined, prepend ALL needed imports in ONE
edit_file call:

    <tool_call>{"name": "edit_file", "arguments": {"path": "X.py",
      "old_string": "",
      "new_string": "import json\\nfrom Y import Z\\n"}}</tool_call>
""",
        tags=["edit_file", "import", "auto_verify", "prepend"],
    ),
    "premature_bail": dict(
        domain="agentic",
        category="check_stderr_before_declaring_done",
        query_template=(
            "After running a CLI or test, decide whether the task is done."
        ),
        solution_template="""\
Before emitting a final plain-text answer, scan the LAST run_bash or
run_tests result for failure signatures: `Traceback`, `SyntaxError`,
`AssertionError`, non-zero exit codes, or empty stdout when output was
expected. If ANY are present, the task is not done.

WRONG — bailing despite stderr Traceback:

    [run_bash result] stderr: Traceback (most recent call last): ...
    [final answer]    "Task complete. CLI works as expected."

RIGHT — read the traceback, identify the broken line, fix and re-run:

    [run_bash result] stderr: Traceback ... NameError: name 'X' is not
                              defined
    [next tool_call]  edit_file to fix the import or definition
    [next tool_call]  re-run the same command
    [final answer]    only after the re-run shows no error

A final answer is a CLAIM that you've verified the deliverable. Never
make that claim while the most recent observation contains an error.
""",
        tags=["stderr", "verification", "completion-check", "agentic"],
    ),
    "stuck_repeat_loop": dict(
        domain="agentic",
        category="recover_from_stuck_repeat",
        query_template=(
            "An edit_file or write_file keeps failing with the same error — "
            "the agent loop will fire `stuck_repeat`."
        ),
        solution_template="""\
When the agent emits `stuck_repeat`, it means you tried the same call
3+ times. The task is still doable — switch tactics:

1. read_file the target — its actual current state probably differs
   from your assumption.
2. Switch tool: write_file -> edit_file with a SHORT unique old_string,
   or edit_file -> write_file (full rewrite) for a small file, or
   edit_file -> insert_at_line when no unique anchor exists.
3. For HTML/JSON/YAML, prefer edit_html / structured tools over
   text edit_file.

Do NOT respond to `stuck_repeat` with "I cannot complete this task".
The harness only fires it AFTER 3 identical failures — that's a signal
the approach is wrong, not the task.
""",
        tags=["stuck_repeat", "recovery", "agentic", "edit_file"],
    ),
    "stale_anchor_edit": dict(
        domain="agentic",
        category="reread_before_edit_when_anchor_stale",
        query_template=(
            "edit_file returns 'old_string not found' — the file has "
            "changed since you last read it."
        ),
        solution_template="""\
When edit_file fails with `old_string not found`, the file's current
content no longer matches your old_string. The fix is to re-read first.

WRONG (relying on stale knowledge of file contents):

    <tool_call>{"name": "edit_file", "arguments": {"path": "x.py",
      "old_string": "def old_form():", "new_string": "..."}}</tool_call>
    -> err: old_string not found in x.py
    <tool_call>{"name": "edit_file", "arguments": {"path": "x.py",
      "old_string": "def old_form():", "new_string": "..."}}</tool_call>
    -> err: stuck_repeat

RIGHT (re-read first, then build old_string from current content):

    <tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>
    -> [shows current contents]
    <tool_call>{"name": "edit_file", "arguments": {"path": "x.py",
      "old_string": "<short unique substring from CURRENT content>",
      "new_string": "..."}}</tool_call>

Pick a SHORT unique anchor — usually 1 line of code is enough. Strip
leading/trailing whitespace from old_string; whitespace must match
EXACTLY (tabs vs spaces).
""",
        tags=["edit_file", "stale_anchor", "read_file", "agentic"],
    ),
    "cwd_assumption": dict(
        domain="agentic",
        category="confirm_path_before_run_bash",
        query_template=(
            "run_bash failed with FileNotFoundError or 'No such file or "
            "directory' — the command assumed a path that doesn't exist."
        ),
        solution_template="""\
When run_bash fails because of a wrong path, the model assumed a cwd
or file location that isn't true in the workspace. Verify the path
BEFORE running the command.

WRONG (assumes tests/ exists without checking):

    <tool_call>{"name": "run_bash",
      "arguments": {"command": "python tests/test_x.py"}}</tool_call>
    -> stderr: python: can't open file 'tests/test_x.py': No such file

RIGHT — verify with list_dir or glob first:

    <tool_call>{"name": "list_dir", "arguments": {"path": "."}}</tool_call>
    -> [shows test_x.py is at workspace root, not tests/]
    <tool_call>{"name": "run_bash",
      "arguments": {"command": "python test_x.py"}}</tool_call>

Or, even better, prefer `run_tests` over `run_bash` for test runs —
it auto-detects the framework and the right invocation.
""",
        tags=["run_bash", "cwd", "filesystem", "list_dir", "agentic"],
    ),
    "incomplete_deliverable": dict(
        domain="agentic",
        category="address_every_numbered_requirement",
        query_template=(
            "Goal lists multiple numbered requirements (1. ... 2. ... 3. ...) "
            "and you must address every one before declaring complete."
        ),
        solution_template="""\
When the goal lists numbered requirements, address EVERY one before
giving a final answer. Common miss: a goal of the form
"1. write code; 2. write tests; 3. run tests" gets steps 1-2 done and
the agent stops. Run-tests is required when the goal says "run tests"
or "verify they pass".

WRONG (declaring done after partial coverage):

    Goal: 1. bookmark.py 2. storage.py 3. cli.py 4. test_bookmarks.py 5. run tests
    write_file(bookmark.py)
    write_file(storage.py)
    write_file(cli.py)
    [final answer] "Files created."
    -> 2 of 5 requirements skipped (test file, test run)

RIGHT (every numbered item addressed):

    write_file(bookmark.py)
    write_file(storage.py)
    write_file(cli.py)
    write_file(test_bookmarks.py)
    run_tests(path=".", runner="pytest")
    [final answer] "All 5 requirements addressed; tests pass."

Build an internal checklist from the goal's numbered items. Tick each
off before the final answer. If you skipped one, go back and do it
before declaring done.
""",
        tags=["completeness", "numbered_requirements", "agentic"],
    ),
    "unaddressed_file_in_goal": dict(
        domain="agentic",
        category="touch_every_file_named_in_goal",
        query_template=(
            "Goal names multiple files; ensure every named file is "
            "actually written or edited before declaring complete."
        ),
        solution_template="""\
When the goal names specific files, EVERY named file must appear as a
mutating tool_call path before you give a final answer. The 14B's
common failure: fixate on one file and silently skip the others.

WRONG (goal said both files; only edited one):

    Goal: "Fix bug in storage.py and bug in cli.py"
    edit_file(path="cli.py", ...)   # one fix
    edit_file(path="cli.py", ...)   # second fix in cli.py
    edit_file(path="cli.py", ...)   # third fix in cli.py
    [final answer] "Bugs fixed."
    -> storage.py never touched. Goal half-done.

RIGHT (every named file gets a write or edit):

    edit_file(path="storage.py", ...)   # storage.py addressed
    edit_file(path="cli.py", ...)       # cli.py addressed
    [final answer] "Both bugs fixed."

Build a mental list at the start of every multi-file goal:
  files_named = ["storage.py", "cli.py"]
  files_touched = []
After each tool_call, append to files_touched. Don't give a final
answer until files_named is a subset of files_touched.

Note: files mentioned in "DO NOT modify X" / "do not edit X" don't
count — those are the files you should NOT touch.
""",
        tags=["multi_file", "coverage", "agentic", "completion"],
    ),
    "narration_without_action": dict(
        domain="agentic",
        category="emit_tool_call_immediately_after_narration",
        query_template=(
            "Don't emit narration prose ('I will now do X') and stop — "
            "actually emit the tool_call block in the same turn."
        ),
        solution_template="""\
When you describe what you're about to do, emit the actual tool_call
block IMMEDIATELY in the same turn. Stopping after the narration
sentence makes the harness treat the prose as your final answer and
no work happens.

WRONG (narration without action — turn ends here):

    "I have read the file. I will now emit the necessary tool calls
    to add the new subcommands."
    [no tool_call block follows]
    [harness accepts as final answer; no edits made]

RIGHT (narration + tool call in same turn):

    "I have read the file. Adding the tag subcommand now."
    <tool_call>{"name": "edit_file", "arguments": {...}}</tool_call>

Or skip the narration entirely — go straight to the tool call:

    <tool_call>{"name": "edit_file", "arguments": {...}}</tool_call>

Future-tense intent prose ("I will now", "I'll proceed", "Next, I
will") is the danger signal. If you write it, follow up with the
tool_call in the SAME turn — never end a turn on intent.
""",
        tags=["narration", "tool_call", "completion", "agentic"],
    ),
    "loop_limit_exhausted": dict(
        domain="agentic",
        category="escalate_after_failed_edit_streak",
        query_template=(
            "Stuck in an edit_file retry loop on the same file — three "
            "consecutive edit attempts have failed. Escalate strategy."
        ),
        solution_template="""\
When edit_file fails 3+ times in a row on the same file (old_string
not found, repeated stale_anchor_edit, etc.), STOP escalating with
more tiny edits and switch to a whole-file rewrite. Tiny edits compound
state errors — each failed edit leaves the file in a weirder state
than the last one.

WRONG (death-spiral pattern that exhausts max_iterations):

    edit_file(path="config.yml", old_string="...", new_string="...")
        err: old_string not found
    edit_file(path="config.yml", old_string="...", new_string="...")
        err: old_string not found
    edit_file(path="config.yml", old_string="...", new_string="...")
        err: ambiguous match
    edit_file(path="config.yml", ...)   # again
    edit_file(path="config.yml", ...)   # again
    [out of iterations; final answer never emitted]

RIGHT (escalate to whole-file rewrite at attempt 4):

    edit_file(path="config.yml", ...)
        err: old_string not found
    edit_file(path="config.yml", ...)
        err: old_string not found
    edit_file(path="config.yml", ...)
        err: ambiguous match
    # Three edit failures in a row -> escalate.
    read_file(path="config.yml")          # see CURRENT state
        ok: <full file contents>
    write_file(path="config.yml", content=[
        "service: webapp",
        "database:",
        "  host: localhost",
        "  port: 5432",
    ])
    [final answer with the corrected file in one shot]

The trigger is "3+ failed edits in a row on the same path." When you
notice that pattern, escalate. The file might be tiny (4 lines is
plenty stuck-worthy on a YAML or JSON) — small files are EASIER to
rewrite, not harder. Don't wait for max_iterations.

For multi-line content in write_file, use the array form (one element
per line) so JSON quote escaping doesn't compound the original
problem.
""",
        tags=["edit_file", "write_file", "loop_limit", "escalation",
              "stuck", "agentic"],
    ),
    "test_import_path": dict(
        domain="agentic",
        category="test_must_import_target_module",
        query_template=(
            "Test failed with ModuleNotFoundError or ImportError when "
            "trying to import the module under test."
        ),
        solution_template="""\
When pytest can't find your module, the issue is usually the test
file's location relative to the target. Three reliable fixes:

1. Put the test file in the SAME directory as the target — pytest
   adds the directory to sys.path automatically when it discovers
   tests there:
       project/
         my_module.py
         test_my_module.py   <- pytest finds my_module via cwd

2. Use `run_tests` (the agent's wrapper), NOT `run_bash python -m pytest`.
   run_tests handles the cwd + framework detection for you.

3. If the project has a src/ layout, add an empty conftest.py at the
   project root — pytest treats it as the rootdir marker:
       project/
         conftest.py        <- empty file
         src/
           my_module.py
         tests/
           test_my_module.py

Do NOT chase the import error by adding sys.path.append() to the test
file — it's brittle and breaks when the test is moved.
""",
        tags=["pytest", "imports", "test_path", "run_tests", "agentic"],
    ),
}


# ── Builder ─────────────────────────────────────────────────────────


def _solution_signature(payload: dict) -> str:
    """Stable short identifier derived from the BUILT solution body —
    the lesson the YAML teaches. Two YAMLs whose solutions normalize to
    the same shape collide on disk (filename match), so writing the
    second is idempotent — the first occurrence wins.

    This is the harvest-time half of the dedup story; retrieval-time
    dedup in engine.augmentors lives in `_dedupe_examples`. Both use
    the same lowercased + whitespace-collapsed first-200-chars
    signature so identical lessons hash to identical IDs across
    layers.

    Earlier behavior used `time.time_ns()` to keep each occurrence
    distinct on disk — but that's the opposite of what we want. We
    WANT duplicate lessons to collide; surfacing every flagged
    occurrence as a separate file just bloats the review queue (a
    single overnight bench produced 35 missing_import YAMLs all
    teaching the same canonical fix).
    """
    examples = payload.get("examples") or []
    if not examples:
        # Defensive — should never happen, but if it does, fall back
        # to category-only so duplicates still collide.
        return hashlib.sha256(payload.get("category", "?").encode()).hexdigest()[:10]
    solution = examples[0].get("solution") or ""
    normalized = " ".join(solution.lower().split())[:200]
    h = hashlib.sha256()
    h.update(payload.get("category", "?").encode("utf-8"))
    h.update(normalized.encode("utf-8"))
    return h.hexdigest()[:10]


def build_yaml(record: FailureRecord, goal: str) -> dict | None:
    """Convert one FailureRecord into a YAML-shaped dict matching the
    augmentor library schema. Returns None if the category has no
    template (silent skip — caller can iterate without filtering)."""
    tmpl = _TEMPLATES.get(record.category)
    if tmpl is None:
        return None
    query = goal.strip() if goal.strip() else tmpl["query_template"]
    return {
        "domain": tmpl["domain"],
        "category": tmpl["category"],
        "examples": [{
            "query": query + "\n\n# Triggered failure: " + record.category,
            "solution": tmpl["solution_template"],
            "tags": tmpl["tags"],
            "difficulty": "medium",
            "source": "auto-generated from failure_flagger",
            "captured_at": time.time(),
            "triggering_file": record.file_path or "",
            "suggested_fix_summary": record.suggested_fix,
        }],
    }


def write_yaml(record: FailureRecord, goal: str, repo_root: Path | str) -> Path | None:
    """Persist a single FailureRecord as a YAML augmentor file under
    `data/auto_generated_review/<category>/`. Returns the file path
    written, or None if the category has no template.

    IMPORTANT: this directory is INTENTIONALLY OUTSIDE
    `data/augmentor_examples/`. The retrieval system in
    densanon.core.example_loader.load_all_examples uses
    `base.rglob("*.yaml")` (recursive), so anything under
    `data/augmentor_examples/` — including subdirectories starting with
    underscore — would be auto-loaded into the retrieval index on next
    boot. Putting auto-generated YAMLs in a separate top-level directory
    physically prevents that. Promotion to the retrieval index requires
    a human moving (or symlinking) the file into a real
    `data/augmentor_examples/<domain>/` subdirectory.
    """
    payload = build_yaml(record, goal)
    if payload is None:
        return None
    repo_root = Path(repo_root)
    target_dir = repo_root / "data" / "auto_generated_review" / record.category
    target_dir.mkdir(parents=True, exist_ok=True)
    # Filename is a hash of the SOLUTION body, not the failure record —
    # two failures that produce identical lessons collide on disk and
    # writing the second is idempotent. Confirmed by overnight bench:
    # 35 missing_import failures had collapsed into 35 distinct files
    # via time.time_ns()-based naming, all teaching the SAME lesson.
    fname = f"{_solution_signature(payload)}.yaml"
    path = target_dir / fname
    if path.exists():
        # Same lesson already harvested — no-op rewrite would just
        # bump mtime. Return the existing path so callers can still
        # account for the occurrence.
        return path
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def write_all(records: Iterable[FailureRecord], goal: str, repo_root: Path | str) -> list[Path]:
    """Persist every flag-able record. Returns the list of paths written.
    Records whose category has no template are silently skipped."""
    out: list[Path] = []
    for r in records:
        p = write_yaml(r, goal, repo_root)
        if p is not None:
            out.append(p)
    return out


# ── Smoke test ──────────────────────────────────────────────────────


if __name__ == "__main__":
    import tempfile
    try:
        from engine.failure_flagger import FailureRecord as FR
    except ImportError:
        from failure_flagger import FailureRecord as FR

    rec = FR(
        category="json_quote_leak",
        iteration=2,
        tool_name="write_file",
        error_excerpt="Expecting ',' delimiter: line 4 column 158",
        triggering_args={"path": "storage.py", "content": ["..."]},
        file_path="storage.py",
        suggested_fix="Use array form...",
    )
    tmp = Path(tempfile.mkdtemp())
    p = write_yaml(rec, "Build a todo CLI", tmp)
    print(f"wrote {p}")
    print("--- file contents ---")
    print(p.read_text(encoding="utf-8"))
