"""Unit tests for engine.tool_validator — pre-dispatch JSON Schema validator."""

from __future__ import annotations

import pytest

from engine.tool_validator import format_rejection, validate_arguments


# Realistic schema mirroring how engine/agent_builtins.py registers tools.
_READ_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "..."},
        "offset": {"type": "integer", "description": "..."},
        "limit": {"type": "integer", "description": "..."},
    },
    "required": ["path"],
}

_RUN_TESTS_SCHEMA = {
    "type": "object",
    "properties": {
        "framework": {"type": "string"},
        "verbose": {"type": "boolean"},
    },
    "required": [],
    "additionalProperties": False,
}


# ── happy path ───────────────────────────────────────────────────


def test_valid_call_returns_empty_errors():
    assert validate_arguments({"path": "x.py"}, _READ_FILE_SCHEMA) == []


def test_valid_with_optional_fields():
    args = {"path": "x.py", "offset": 0, "limit": 100}
    assert validate_arguments(args, _READ_FILE_SCHEMA) == []


def test_empty_schema_accepts_anything():
    assert validate_arguments({"whatever": 1}, {}) == []


# ── required fields ──────────────────────────────────────────────


def test_missing_required_field_reported():
    errors = validate_arguments({"offset": 0}, _READ_FILE_SCHEMA)
    assert len(errors) == 1
    assert "missing required field" in errors[0]
    assert "`path`" in errors[0]


def test_multiple_missing_required_fields():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"},
        },
        "required": ["a", "b", "c"],
    }
    errors = validate_arguments({}, schema)
    assert len(errors) == 3


# ── type mismatches ──────────────────────────────────────────────


def test_wrong_type_for_required_field():
    errors = validate_arguments({"path": 42}, _READ_FILE_SCHEMA)
    assert len(errors) == 1
    assert "`path`" in errors[0]
    assert "string" in errors[0]
    assert "int" in errors[0]


def test_wrong_type_for_optional_field():
    errors = validate_arguments({"path": "x.py", "limit": "100"}, _READ_FILE_SCHEMA)
    assert len(errors) == 1
    assert "`limit`" in errors[0]
    assert "integer" in errors[0]


def test_boolean_not_accepted_as_integer():
    """JSON Schema integer type should reject Python bool (which is a
    subclass of int). Otherwise schemas can't distinguish 0/False."""
    errors = validate_arguments({"path": "x.py", "offset": True}, _READ_FILE_SCHEMA)
    assert len(errors) == 1
    assert "integer" in errors[0]


def test_arguments_must_be_dict():
    errors = validate_arguments("not a dict", _READ_FILE_SCHEMA)
    assert len(errors) == 1
    assert "JSON object" in errors[0]


def test_arguments_list_rejected():
    errors = validate_arguments(["a", "b"], _READ_FILE_SCHEMA)
    assert len(errors) == 1
    assert "JSON object" in errors[0]


# ── additionalProperties handling ────────────────────────────────


def test_extra_field_allowed_when_additional_true():
    # default additionalProperties=True → unknown fields pass through.
    args = {"path": "x.py", "extra_thing": "ok"}
    assert validate_arguments(args, _READ_FILE_SCHEMA) == []


def test_extra_field_rejected_when_additional_false():
    args = {"framework": "pytest", "rogue": 1}
    errors = validate_arguments(args, _RUN_TESTS_SCHEMA)
    assert len(errors) == 1
    assert "`rogue`" in errors[0]
    assert "not declared" in errors[0]


# ── union types ──────────────────────────────────────────────────


def test_union_type_accepts_either():
    schema = {
        "type": "object",
        "properties": {
            "content": {"type": ["string", "array"]},
        },
        "required": ["content"],
    }
    assert validate_arguments({"content": "x"}, schema) == []
    assert validate_arguments({"content": ["x", "y"]}, schema) == []
    errors = validate_arguments({"content": 42}, schema)
    assert len(errors) == 1
    assert "string | array" in errors[0]


# ── format_rejection ─────────────────────────────────────────────


def test_format_rejection_mentions_tool_and_errors():
    msg = format_rejection("read_file", ["missing required field `path`"])
    assert "read_file" in msg
    assert "REJECTED" in msg
    assert "missing required field" in msg
    # Make it explicit the function did not run.
    assert "did NOT run" in msg


def test_format_rejection_joins_multiple_errors():
    msg = format_rejection("foo", ["err one", "err two", "err three"])
    assert "err one; err two; err three" in msg


# ── unknown type keyword is permissive ───────────────────────────


def test_unknown_type_keyword_does_not_reject():
    """If a schema uses a JSON Schema keyword we don't recognize, the
    validator should not block — better to let the call through and
    rely on the function's own arg checks than to over-reject."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "weird-custom-type"}},
        "required": [],
    }
    assert validate_arguments({"x": "anything"}, schema) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
