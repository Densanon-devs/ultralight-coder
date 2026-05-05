"""Pre-dispatch JSON-Schema validator for tool calls.

Adapted from arXiv 2604.26091 ("Operating-Layer Controls for Onchain
LLM Agents"). 21-day live deployment, 3,505 user-funded agents, 7.5M
invocations: pre-intervention fabricated-rule failures sat at 57%;
adding typed controls + policy validation + prompt compilation
dropped them to 3%. The blunt conclusion: "reliability did not come
from the base model alone" — system-level harness controls dominate
at the margin.

This module is the typed-controls half: validate every parsed tool
call's arguments against the tool's JSON-Schema *before* dispatching.
On failure, produce a structured rejection message that surfaces the
specific field problem (missing required, wrong type, unknown extra).
The function itself never runs.

Composes with the existing tool-call parser (which catches
JSON-syntax failures) and the live self-heal classifier (which fires
on consecutive same-class failures). Together they form a three-layer
defense: parse → schema → self-heal.

Validator scope is intentionally narrow — JSON Schema's `type`,
`required`, `properties`, and `additionalProperties` are all the
existing ToolSchema definitions actually use. We don't pull in
`jsonschema` or any other dependency.
"""

from __future__ import annotations

from typing import Any


# JSON-Schema "type" → Python isinstance check. "number" matches both
# int and float (per JSON Schema spec); "integer" matches int but not
# bool (Python's bool is a subclass of int — explicit guard below).
_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _format_type(t: Any) -> str:
    """JSON Schema `type` may be a string or a list (union). Format for messages."""
    if isinstance(t, list):
        return " | ".join(t)
    return str(t)


def _matches_type(value: Any, expected: Any) -> bool:
    """JSON Schema `type` may be a string or a list (union of allowed types)."""
    if isinstance(expected, list):
        return any(_matches_type(value, t) for t in expected)
    check = _TYPE_CHECKS.get(expected)
    if check is None:
        # Unknown type keyword — be permissive rather than over-reject.
        return True
    return check(value)


def validate_arguments(arguments: Any, schema: dict) -> list[str]:
    """Validate `arguments` against the tool's JSON Schema.

    Parameters
    ----------
    arguments
        The model-supplied argument object — should be a dict per
        Hermes tool-call format. Anything else is itself an error.
    schema
        The tool's JSON Schema dict (i.e. ToolSchema.parameters).

    Returns
    -------
    list[str]
        Human-readable error messages, one per problem found. Empty
        list = valid call.
    """
    errors: list[str] = []

    if not isinstance(arguments, dict):
        return [
            f"`arguments` must be a JSON object, got {type(arguments).__name__}"
        ]

    # Defensive: a missing/empty schema is a no-op (early-stage tool defs).
    if not schema:
        return []

    # Top-level type check (JSON Schema lets the schema declare its own type).
    declared = schema.get("type")
    if declared and not _matches_type(arguments, declared):
        errors.append(
            f"arguments expected type `{_format_type(declared)}`, got `{type(arguments).__name__}`"
        )
        return errors  # No point checking properties if the shape is wrong.

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    additional = schema.get("additionalProperties", True)

    # Required fields must be present.
    for key in required:
        if key not in arguments:
            errors.append(f"missing required field `{key}`")

    # Per-property type checks.
    for key, value in arguments.items():
        if key in properties:
            prop_schema = properties[key]
            prop_type = prop_schema.get("type") if isinstance(prop_schema, dict) else None
            if prop_type and not _matches_type(value, prop_type):
                got = type(value).__name__
                errors.append(
                    f"field `{key}` expected `{_format_type(prop_type)}`, got `{got}`"
                )
        else:
            # Field not declared in the schema. Reject ONLY when the
            # schema explicitly says additionalProperties=False.
            if additional is False:
                errors.append(f"unknown field `{key}` (not declared in schema)")
            # Otherwise let it through — many real tools accept **kwargs.

    return errors


def format_rejection(tool_name: str, errors: list[str]) -> str:
    """Format the rejection message the model will see in its next turn.

    Front-loaded with the tool name and a clear "rejected before
    execution" framing so the model treats it as an input shape
    problem rather than as a runtime failure to debug.
    """
    body = "; ".join(errors)
    return (
        f"Tool call `{tool_name}` REJECTED before execution — schema check failed: "
        f"{body}. Fix the arguments and retry. The function did NOT run, so "
        f"any state you expected it to change is unchanged."
    )
