"""Self-test: apply a correct manual solution to each new multi-lang task's
workspace, then call the task's check() — must return (True, ...). This
catches check-logic bugs before we waste 14B inference cycles debugging
them."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_agentic import (
    setup_add_json_field, check_add_json_field,
    setup_fix_yaml_indent, check_fix_yaml_indent,
    setup_js_reducer, check_js_reducer_fixed,
    setup_write_bash_lister, check_write_bash_lister,
    setup_add_ts_interface, check_add_ts_interface,
)

import json


def case(name, setup, fix, check):
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        setup(ws)
        fix(ws)
        ok, msg = check(ws, None)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}: {msg}")
        return ok


def fix_add_json_field(ws):
    p = ws / "package.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["license"] = "MIT"
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def fix_yaml_indent(ws):
    (ws / "config.yml").write_text(
        "service: webapp\n"
        "database:\n"
        "  port: 5432\n"
        "  host: localhost\n"
        "  name: appdb\n"
        "features:\n"
        "  - auth\n"
        "  - billing\n",
        encoding="utf-8",
    )


def fix_js_reducer(ws):
    (ws / "sum.js").write_text(
        "function sumRange(arr, from, to) {\n"
        "  let total = 0;\n"
        "  for (let i = from; i <= to; i++) { total += arr[i]; }\n"
        "  return total;\n"
        "}\nmodule.exports = { sumRange };\n",
        encoding="utf-8",
    )


def fix_write_bash_lister(ws):
    (ws / "list_py.sh").write_text(
        "#!/bin/bash\nfind . -type f -name '*.py' | sed 's|^\\./||'\n",
        encoding="utf-8",
    )


def fix_add_ts_interface(ws):
    (ws / "user.ts").write_text(
        "export interface User { id: number; name: string; email?: string; }\n"
        "export function greet(u: User): string { return `Hello, ${u.name}`; }\n",
        encoding="utf-8",
    )


cases = [
    ("add_json_field",    setup_add_json_field,    fix_add_json_field,    check_add_json_field),
    ("fix_yaml_indent",   setup_fix_yaml_indent,   fix_yaml_indent,       check_fix_yaml_indent),
    ("fix_js_reducer",    setup_js_reducer,        fix_js_reducer,        check_js_reducer_fixed),
    ("write_bash_lister", setup_write_bash_lister, fix_write_bash_lister, check_write_bash_lister),
    ("add_ts_interface",  setup_add_ts_interface,  fix_add_ts_interface,  check_add_ts_interface),
]

results = [case(n, s, f, c) for n, s, f, c in cases]
print(f"\n{sum(results)}/{len(results)} self-tests pass")
sys.exit(0 if all(results) else 1)
