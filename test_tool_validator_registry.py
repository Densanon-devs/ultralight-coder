"""Integration test: tool_validator wired into ToolRegistry.execute.

Verifies the pre-dispatch JSON Schema check rejects bad calls *before*
the registered function runs (so any state changes the function would
have made don't happen on rejected calls).
"""

from __future__ import annotations

import pytest

from engine.agent_tools import ToolCall, ToolRegistry, ToolSchema


def _make_registry(*, additional_properties: bool = True):
    """Registry with a single 'set_value' tool that mutates a side dict.

    The side dict lets us prove the function did NOT run when the call
    is rejected.
    """
    state: dict[str, object] = {}

    def set_value(key: str, value: int) -> str:
        state[key] = value
        return f"set {key}={value}"

    schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "integer"},
        },
        "required": ["key", "value"],
    }
    if not additional_properties:
        schema["additionalProperties"] = False

    reg = ToolRegistry()
    reg.register(
        ToolSchema(
            name="set_value",
            description="Set a key in the side dict",
            parameters=schema,
            function=set_value,
        )
    )
    return reg, state


def test_valid_call_runs_function():
    reg, state = _make_registry()
    result = reg.execute(ToolCall(name="set_value", arguments={"key": "a", "value": 1}, raw=""))
    assert result.success is True
    assert state == {"a": 1}


def test_missing_required_field_rejects_before_run():
    reg, state = _make_registry()
    result = reg.execute(ToolCall(name="set_value", arguments={"key": "a"}, raw=""))
    assert result.success is False
    assert "REJECTED" in result.error
    assert "missing required field" in result.error
    assert "`value`" in result.error
    # The function never ran — state is untouched.
    assert state == {}


def test_wrong_type_rejects_before_run():
    reg, state = _make_registry()
    result = reg.execute(
        ToolCall(name="set_value", arguments={"key": "a", "value": "not-an-int"}, raw="")
    )
    assert result.success is False
    assert "REJECTED" in result.error
    assert "expected `integer`" in result.error
    assert state == {}


def test_extra_field_passes_when_additional_true():
    """Default additionalProperties=True: extra fields pass the validator
    and the function gets them — Python TypeError handling kicks in."""
    reg, state = _make_registry(additional_properties=True)
    result = reg.execute(
        ToolCall(name="set_value", arguments={"key": "a", "value": 1, "extra": "x"}, raw="")
    )
    # Validator passes the call → function runs → TypeError on unknown kwarg.
    # Either branch produces success=False; what we want to assert is that
    # the rejection path was NOT used — i.e. the error is "Bad arguments"
    # not "REJECTED".
    assert result.success is False
    assert "REJECTED" not in result.error
    assert "Bad arguments" in result.error
    # state still untouched because Python's TypeError fired before the
    # function body ran.
    assert state == {}


def test_extra_field_rejected_when_additional_false():
    reg, state = _make_registry(additional_properties=False)
    result = reg.execute(
        ToolCall(name="set_value", arguments={"key": "a", "value": 1, "extra": "x"}, raw="")
    )
    assert result.success is False
    assert "REJECTED" in result.error
    assert "`extra`" in result.error
    assert state == {}


def test_arguments_not_object_rejected():
    reg, state = _make_registry()
    # Simulate the model emitting a JSON array instead of an object as arguments.
    result = reg.execute(ToolCall(name="set_value", arguments=["wrong"], raw=""))
    assert result.success is False
    assert "REJECTED" in result.error
    assert "JSON object" in result.error
    assert state == {}


def test_unknown_tool_unaffected():
    """Unknown-tool branch is checked BEFORE schema validation and
    shouldn't be perturbed by the validator addition."""
    reg, _ = _make_registry()
    result = reg.execute(ToolCall(name="nonexistent", arguments={}, raw=""))
    assert result.success is False
    assert "Unknown tool" in result.error
    assert "REJECTED" not in result.error


def test_disabled_tool_unaffected():
    reg, _ = _make_registry()
    reg.unregister("set_value")
    reg.register(
        ToolSchema(
            name="set_value",
            description="x",
            parameters={"type": "object", "required": ["key"]},
            function=lambda **kw: "x",
            enabled=False,
        )
    )
    result = reg.execute(ToolCall(name="set_value", arguments={}, raw=""))
    assert result.success is False
    assert "disabled" in result.error
    assert "REJECTED" not in result.error


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
