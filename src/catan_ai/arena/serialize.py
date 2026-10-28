"""JSON round-tripping for canonical actions and engine outcomes."""

from typing import Any

from catan_ai.engine import ActionKind, CanonicalAction


def to_json_value(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [to_json_value(v) for v in value]
    return value


def from_json_value(value: Any) -> Any:
    """JSON arrays come back as tuples, matching the canonical representation."""
    if isinstance(value, list):
        return tuple(from_json_value(v) for v in value)
    return value


def action_to_json(action: CanonicalAction) -> dict[str, Any]:
    return {"kind": str(action.kind), "seat": action.seat, "params": to_json_value(action.params)}


def action_from_json(data: dict[str, Any]) -> CanonicalAction:
    return CanonicalAction(ActionKind(data["kind"]), data["seat"], from_json_value(data["params"]))
