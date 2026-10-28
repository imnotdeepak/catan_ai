"""Canonical, seat-based action representation and translation to/from Catanatron.

Canonical actions identify players by seat (0..3, turn order), never by engine
color, and never carry hidden outcomes: a development-card purchase does not
name the card drawn, and a robber move does not name the card stolen.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypeAlias

from catanatron import Action, ActionType

Resource: TypeAlias = Literal["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
RESOURCES: tuple[Resource, ...] = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
ResourceVector: TypeAlias = tuple[int, int, int, int, int]
Seat: TypeAlias = int


class ActionKind(StrEnum):
    ROLL = "ROLL"
    END_TURN = "END_TURN"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"
    BUILD_ROAD = "BUILD_ROAD"
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"
    BUILD_CITY = "BUILD_CITY"
    BUY_DEV_CARD = "BUY_DEV_CARD"
    PLAY_KNIGHT = "PLAY_KNIGHT"
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"
    PLAY_MONOPOLY = "PLAY_MONOPOLY"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"
    MARITIME_TRADE = "MARITIME_TRADE"
    OFFER_TRADE = "OFFER_TRADE"
    ACCEPT_TRADE = "ACCEPT_TRADE"
    REJECT_TRADE = "REJECT_TRADE"
    CONFIRM_TRADE = "CONFIRM_TRADE"
    CANCEL_TRADE = "CANCEL_TRADE"


@dataclass(frozen=True)
class CanonicalAction:
    """A player decision. `params` by kind:

    - BUILD_ROAD: (node_a, node_b) with node_a < node_b
    - BUILD_SETTLEMENT, BUILD_CITY: (node_id,)
    - DISCARD, PLAY_MONOPOLY: (resource,)
    - PLAY_YEAR_OF_PLENTY: one or two resources, in RESOURCES order
    - MOVE_ROBBER: (tile_coordinate, victim_seat or None)
    - MARITIME_TRADE: (give_resource, give_count, receive_resource)
    - OFFER_TRADE: (give_vector, receive_vector)
    - CONFIRM_TRADE: (partner_seat,)
    - every other kind: ()
    """

    kind: ActionKind
    seat: Seat
    params: tuple[Any, ...] = ()

    @property
    def key(self) -> str:
        """Stable string id, e.g. for logs and policy candidate indexing."""
        return f"{self.kind}:{self.seat}:{self.params!r}"


_SIMPLE_KINDS: dict[Any, ActionKind] = {
    ActionType.ROLL: ActionKind.ROLL,
    ActionType.END_TURN: ActionKind.END_TURN,
    ActionType.BUY_DEVELOPMENT_CARD: ActionKind.BUY_DEV_CARD,
    ActionType.PLAY_KNIGHT_CARD: ActionKind.PLAY_KNIGHT,
    ActionType.PLAY_ROAD_BUILDING: ActionKind.PLAY_ROAD_BUILDING,
    ActionType.ACCEPT_TRADE: ActionKind.ACCEPT_TRADE,
    ActionType.REJECT_TRADE: ActionKind.REJECT_TRADE,
    ActionType.CANCEL_TRADE: ActionKind.CANCEL_TRADE,
}
_SIMPLE_TYPES = {kind: action_type for action_type, kind in _SIMPLE_KINDS.items()}


def to_canonical(action: Any, colors: Sequence[Any]) -> CanonicalAction:
    """Translate an engine `Action` requested by a player into canonical form."""
    seat = colors.index(action.color)
    action_type, value = action.action_type, action.value

    if action_type in _SIMPLE_KINDS:
        return CanonicalAction(_SIMPLE_KINDS[action_type], seat)
    if action_type == ActionType.BUILD_ROAD:
        return CanonicalAction(ActionKind.BUILD_ROAD, seat, (tuple(sorted(value)),))
    if action_type == ActionType.BUILD_SETTLEMENT:
        return CanonicalAction(ActionKind.BUILD_SETTLEMENT, seat, (value,))
    if action_type == ActionType.BUILD_CITY:
        return CanonicalAction(ActionKind.BUILD_CITY, seat, (value,))
    if action_type == ActionType.DISCARD_RESOURCE:
        return CanonicalAction(ActionKind.DISCARD, seat, (value,))
    if action_type == ActionType.PLAY_MONOPOLY:
        return CanonicalAction(ActionKind.PLAY_MONOPOLY, seat, (value,))
    if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        return CanonicalAction(
            ActionKind.PLAY_YEAR_OF_PLENTY, seat, tuple(sorted(value, key=RESOURCES.index))
        )
    if action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = value
        victim_seat = None if victim is None else colors.index(victim)
        return CanonicalAction(ActionKind.MOVE_ROBBER, seat, (tuple(coordinate), victim_seat))
    if action_type == ActionType.MARITIME_TRADE:
        given = [r for r in value[:4] if r is not None]
        return CanonicalAction(ActionKind.MARITIME_TRADE, seat, (given[0], len(given), value[4]))
    if action_type == ActionType.OFFER_TRADE:
        return CanonicalAction(ActionKind.OFFER_TRADE, seat, (tuple(value[:5]), tuple(value[5:10])))
    if action_type == ActionType.CONFIRM_TRADE:
        return CanonicalAction(ActionKind.CONFIRM_TRADE, seat, (colors.index(value[10]),))
    raise ValueError(f"Unsupported engine action type: {action_type}")


def to_engine(action: CanonicalAction, colors: Sequence[Any], current_trade: Any) -> Any:
    """Translate a canonical action into the engine `Action` it denotes.

    `current_trade` is the engine's pending trade tuple, needed for trade
    responses and confirmations.
    """
    color = colors[action.seat]
    kind, params = action.kind, action.params

    if kind in (ActionKind.ACCEPT_TRADE, ActionKind.REJECT_TRADE):
        return Action(color, _SIMPLE_TYPES[kind], current_trade)
    if kind in _SIMPLE_TYPES:
        return Action(color, _SIMPLE_TYPES[kind], None)
    if kind == ActionKind.BUILD_ROAD:
        return Action(color, ActionType.BUILD_ROAD, params[0])
    if kind == ActionKind.BUILD_SETTLEMENT:
        return Action(color, ActionType.BUILD_SETTLEMENT, params[0])
    if kind == ActionKind.BUILD_CITY:
        return Action(color, ActionType.BUILD_CITY, params[0])
    if kind == ActionKind.DISCARD:
        return Action(color, ActionType.DISCARD_RESOURCE, params[0])
    if kind == ActionKind.PLAY_MONOPOLY:
        return Action(color, ActionType.PLAY_MONOPOLY, params[0])
    if kind == ActionKind.PLAY_YEAR_OF_PLENTY:
        return Action(color, ActionType.PLAY_YEAR_OF_PLENTY, params)
    if kind == ActionKind.MOVE_ROBBER:
        coordinate, victim_seat = params
        victim = None if victim_seat is None else colors[victim_seat]
        return Action(color, ActionType.MOVE_ROBBER, (coordinate, victim))
    if kind == ActionKind.MARITIME_TRADE:
        give, count, receive = params
        return Action(
            color,
            ActionType.MARITIME_TRADE,
            tuple([give] * count + [None] * (4 - count) + [receive]),
        )
    if kind == ActionKind.OFFER_TRADE:
        give, receive = params
        return Action(color, ActionType.OFFER_TRADE, (*give, *receive))
    if kind == ActionKind.CONFIRM_TRADE:
        return Action(color, ActionType.CONFIRM_TRADE, (*current_trade[:10], colors[params[0]]))
    raise ValueError(f"Unsupported canonical action kind: {kind}")


def offer_trade(seat: Seat, give: ResourceVector, receive: ResourceVector) -> CanonicalAction:
    """Build a domestic trade proposal. Legality is checked by the adapter."""
    return CanonicalAction(ActionKind.OFFER_TRADE, seat, (tuple(give), tuple(receive)))
