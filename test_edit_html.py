"""
Tests for the edit_html extended agent tool.

Exercises:
- append-child / prepend-child insert at the right position
- replace swaps the element
- remove deletes the element
- set-attr sets and (with null payload) deletes attributes
- set-text changes the text content
- selector with multiple matches uses index
- index out of range raises
- selector matching nothing raises
- non-HTML extensions are rejected
- unknown action is rejected
- file is actually written to disk
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_CORE_ROOT = ROOT.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from engine.agent_builtins import build_default_registry


def _setup(html: str, name: str = "page.html"):
    tmp = Path(tempfile.mkdtemp())
    p = tmp / name
    p.write_text(html, encoding="utf-8")
    reg = build_default_registry(tmp, extended_tools=True)
    return tmp, p, reg


def _exec(reg, **kw):
    fn = reg.get("edit_html").function
    return fn(**kw)


def test_append_child():
    tmp, p, reg = _setup('<html><body><ul id="x"><li>a</li></ul></body></html>')
    _exec(reg, path=str(p), selector="#x", action="append-child", payload="<li>b</li>")
    out = p.read_text(encoding="utf-8")
    assert "<li>a</li><li>b</li>" in out


def test_prepend_child():
    tmp, p, reg = _setup('<html><body><ul id="x"><li>a</li></ul></body></html>')
    _exec(reg, path=str(p), selector="#x", action="prepend-child", payload="<li>z</li>")
    out = p.read_text(encoding="utf-8")
    assert "<li>z</li><li>a</li>" in out


def test_replace():
    tmp, p, reg = _setup('<html><body><h1 class="t">old</h1></body></html>')
    _exec(reg, path=str(p), selector="h1.t", action="replace", payload="<h2>new</h2>")
    out = p.read_text(encoding="utf-8")
    assert "<h2>new</h2>" in out
    assert "<h1" not in out


def test_remove():
    tmp, p, reg = _setup('<html><body><div>keep</div><div class="x">drop</div></body></html>')
    _exec(reg, path=str(p), selector="div.x", action="remove")
    out = p.read_text(encoding="utf-8")
    assert "drop" not in out
    assert "keep" in out


def test_set_attr_set():
    tmp, p, reg = _setup('<html><body><a href="/old">link</a></body></html>')
    _exec(reg, path=str(p), selector="a", action="set-attr", attr="href", payload="/new")
    out = p.read_text(encoding="utf-8")
    assert 'href="/new"' in out


def test_set_attr_delete():
    tmp, p, reg = _setup('<html><body><a href="/old" class="c">link</a></body></html>')
    _exec(reg, path=str(p), selector="a", action="set-attr", attr="class", payload=None)
    out = p.read_text(encoding="utf-8")
    assert "class=" not in out
    assert 'href="/old"' in out


def test_set_text():
    tmp, p, reg = _setup('<html><body><h1>old</h1></body></html>')
    _exec(reg, path=str(p), selector="h1", action="set-text", payload="new title")
    out = p.read_text(encoding="utf-8")
    assert "new title" in out
    assert ">old<" not in out


def test_index_disambiguation():
    tmp, p, reg = _setup('<html><body><p>0</p><p>1</p><p>2</p></body></html>')
    _exec(reg, path=str(p), selector="p", action="set-text", payload="X", index=1)
    out = p.read_text(encoding="utf-8")
    assert ">0<" in out
    assert ">X<" in out
    assert ">2<" in out


def test_index_out_of_range_raises():
    tmp, p, reg = _setup('<html><body><p>0</p></body></html>')
    try:
        _exec(reg, path=str(p), selector="p", action="set-text", payload="X", index=5)
    except ValueError as e:
        assert "out of range" in str(e)
        return
    assert False, "expected ValueError"


def test_no_match_raises():
    tmp, p, reg = _setup('<html><body><p>0</p></body></html>')
    try:
        _exec(reg, path=str(p), selector="#nonexistent", action="remove")
    except ValueError as e:
        assert "0 elements" in str(e)
        return
    assert False, "expected ValueError"


def test_non_html_rejected():
    tmp, p, reg = _setup("def f(): pass", name="code.py")
    try:
        _exec(reg, path=str(p), selector="p", action="remove")
    except ValueError as e:
        assert "only supports HTML" in str(e)
        return
    assert False, "expected ValueError"


def test_unknown_action_rejected():
    tmp, p, reg = _setup('<html><body><p>0</p></body></html>')
    try:
        _exec(reg, path=str(p), selector="p", action="frobnicate")
    except ValueError as e:
        assert "action must be" in str(e)
        return
    assert False, "expected ValueError"


def test_append_to_real_gallery_pattern():
    """Reproduces the extend_real_gallery failure shape — multiple
    similar gallery-item divs, model needs to insert one more."""
    html = """<html><body>
<div class="gallery">
  <div class="gallery-item" data-cat="a">A</div>
  <div class="gallery-item" data-cat="b">B</div>
  <div class="gallery-item" data-cat="c">C</div>
</div>
</body></html>"""
    tmp, p, reg = _setup(html)
    _exec(reg, path=str(p),
          selector=".gallery", action="append-child",
          payload='<div class="gallery-item" data-cat="d">D</div>')
    out = p.read_text(encoding="utf-8")
    assert 'data-cat="d"' in out
    assert out.count("gallery-item") == 4


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
