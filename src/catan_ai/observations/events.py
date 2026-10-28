"""Referee event log with per-seat visibility (ENG-005).

Every executed action becomes one event (plus BONUS_TRANSFER / GAME_END events
when they apply). Payloads are derived from the referee state before and after
the action, so they stay consistent with what actually happened.

Visibility (see docs/observation-contract.md):
- `public` is seen by every seat.
- `private[seat]` is seen only by that seat. Only two facts are private:
  the card drawn on BUY_DEV_CARD (buyer) and the resource taken by a robber
  steal (thief and victim).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from catan_ai.engine.actions import RESOURCES, ActionKind, CanonicalAction, Seat

DEV_CARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
NUM_SEATS = 4
PLAY_KINDS = (
    ActionKind.PLAY_KNIGHT,
    ActionKind.PLAY_YEAR_OF_PLENTY,
    ActionKind.PLAY_MONOPOLY,
    ActionKind.PLAY_ROAD_BUILDING,
)


class EventType:
    """Event types beyond the canonical action kinds (which are used verbatim)."""

    BONUS_TRANSFER = "BONUS_TRANSFER"
    GAME_END = "GAME_END"


@dataclass(frozen=True)
class Event:
    """Referee-side event: carries every seat's private payload."""

    sequence_id: int
    turn_index: int
    actor: Seat | None
    event_type: str
    public: Mapping[str, Any]
    private: Mapping[Seat, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Payloads are shared with every seat's history, so make them read-only.
        object.__setattr__(self, "public", MappingProxyType(dict(self.public)))
        private = {seat: MappingProxyType(dict(p)) for seat, p in self.private.items()}
        object.__setattr__(self, "private", MappingProxyType(private))

    def visible_to(self, seat: Seat) -> "VisibleEvent":
        return VisibleEvent(
            sequence_id=self.sequence_id,
            turn_index=self.turn_index,
            actor=self.actor,
            event_type=self.event_type,
            public=self.public,
            private=self.private.get(seat),
        )


@dataclass(frozen=True)
class VisibleEvent:
    """An event as one seat sees it."""

    sequence_id: int
    turn_index: int
    actor: Seat | None
    event_type: str
    public: Mapping[str, Any]
    private: Mapping[str, Any] | None


@dataclass(frozen=True)
class Snapshot:
    """The referee facts needed to derive an event's deltas. Never shown to agents."""

    hands: tuple[tuple[int, ...], ...]
    dev_hands: tuple[tuple[int, ...], ...]
    longest_road: Seat | None
    largest_army: Seat | None


def snapshot(state: Any) -> Snapshot:
    ps = state.player_state

    def holder(flag: str) -> Seat | None:
        return next((s for s in range(NUM_SEATS) if ps[f"P{s}_{flag}"]), None)

    return Snapshot(
        hands=tuple(tuple(ps[f"P{s}_{r}_IN_HAND"] for r in RESOURCES) for s in range(NUM_SEATS)),
        dev_hands=tuple(
            tuple(ps[f"P{s}_{d}_IN_HAND"] for d in DEV_CARDS) for s in range(NUM_SEATS)
        ),
        longest_road=holder("HAS_ROAD"),
        largest_army=holder("HAS_ARMY"),
    )


def action_event(
    sequence_id: int,
    turn_index: int,
    action: CanonicalAction,
    outcome: Any,
    before: Snapshot,
    after: Snapshot,
) -> Event:
    """Build the event for one executed action."""
    resource_delta = tuple(
        tuple(a - b for a, b in zip(after.hands[s], before.hands[s], strict=True))
        for s in range(NUM_SEATS)
    )
    public: dict[str, Any] = {
        "action": _action_payload(action),
        "card_count_delta": tuple(sum(d) for d in resource_delta),
        "dev_card_count_delta": tuple(
            sum(after.dev_hands[s]) - sum(before.dev_hands[s]) for s in range(NUM_SEATS)
        ),
    }
    private: dict[Seat, dict[str, Any]] = {}

    if action.kind == ActionKind.ROLL:
        public["dice"] = tuple(outcome)
    if action.kind == ActionKind.BUY_DEV_CARD:
        private[action.seat] = {"dev_card": outcome}

    victim = action.params[1] if action.kind == ActionKind.MOVE_ROBBER else None
    if victim is not None and outcome is not None:
        # Steal: everyone sees one card move; only thief and victim know which.
        private[action.seat] = {"stolen_resource": outcome}
        private[victim] = {"stolen_resource": outcome}
    else:
        public["resource_delta"] = resource_delta

    if action.kind in PLAY_KINDS:
        # Played cards are revealed: PLAY_KNIGHT -> KNIGHT, etc.
        public["dev_card_played"] = action.kind.removeprefix("PLAY_")

    return Event(sequence_id, turn_index, action.seat, str(action.kind), public, private)


def bonus_events(
    sequence_id: int, turn_index: int, before: Snapshot, after: Snapshot
) -> list[Event]:
    events: list[Event] = []
    for award, old, new in (
        ("LONGEST_ROAD", before.longest_road, after.longest_road),
        ("LARGEST_ARMY", before.largest_army, after.largest_army),
    ):
        if old != new:
            payload = {"award": award, "from_seat": old, "to_seat": new}
            events.append(
                Event(
                    sequence_id + len(events), turn_index, None, EventType.BONUS_TRANSFER, payload
                )
            )
    return events


def _action_payload(action: CanonicalAction) -> dict[str, Any]:
    return {"kind": str(action.kind), "seat": action.seat, "params": action.params}
