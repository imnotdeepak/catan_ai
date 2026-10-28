"""Engine boundary: canonical actions and the Catanatron adapter."""

from catan_ai.engine.actions import (
    RESOURCES,
    ActionKind,
    CanonicalAction,
    Resource,
    ResourceVector,
    Seat,
    offer_trade,
)
from catan_ai.engine.adapter import SEAT_COLORS, EngineAdapter, IllegalActionError

__all__ = [
    "RESOURCES",
    "SEAT_COLORS",
    "ActionKind",
    "CanonicalAction",
    "EngineAdapter",
    "IllegalActionError",
    "Resource",
    "ResourceVector",
    "Seat",
    "offer_trade",
]
