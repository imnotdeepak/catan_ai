"""Per-seat observation firewall and referee (ENG-005).

`Referee` owns the privileged engine game and the full event log. Agents only
ever receive `PlayerObservation`s, built field by field from an explicit
whitelist of public facts plus the viewing seat's own hand and private events.
Never pass `Referee`, `EngineAdapter` or engine objects to an agent.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from catan_ai.engine.actions import RESOURCES, CanonicalAction, Seat
from catan_ai.engine.adapter import EngineAdapter
from catan_ai.observations.events import (
    DEV_CARDS,
    NUM_SEATS,
    Event,
    EventType,
    VisibleEvent,
    action_event,
    bonus_events,
    snapshot,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BoardView:
    """Public board. Node ids, edges and tile coordinates are Catanatron's static topology."""

    tiles: tuple[
        tuple[tuple[int, int, int], str | None, int | None], ...
    ]  # coord, resource, number
    robber: tuple[int, int, int]
    ports: tuple[tuple[str | None, tuple[int, ...]], ...]  # resource (None = 3:1), node ids
    buildings: tuple[tuple[int, Seat, str], ...]  # node, owner seat, SETTLEMENT/CITY
    roads: tuple[tuple[int, int, Seat], ...]  # node_a < node_b, owner seat


@dataclass(frozen=True)
class PublicPlayer:
    seat: Seat
    visible_victory_points: int  # excludes hidden Victory Point cards
    resource_card_count: int
    dev_card_count: int  # unplayed, face down
    played_dev_cards: tuple[int, ...]  # by DEV_CARDS order (VP cards are never "played")
    roads_available: int
    settlements_available: int
    cities_available: int
    longest_road_length: int
    has_longest_road: bool
    has_largest_army: bool
    has_rolled: bool
    has_played_dev_card_this_turn: bool


@dataclass(frozen=True)
class PendingTrade:
    offerer: Seat
    give: tuple[int, ...]
    receive: tuple[int, ...]
    accepted_by: tuple[bool, ...]


@dataclass(frozen=True)
class PlayerObservation:
    schema_version: int
    seat: Seat
    current_seat: Seat  # who must decide now
    turn_seat: Seat  # whose turn it is
    prompt: str
    is_initial_build_phase: bool
    num_turns: int
    board: BoardView
    players: tuple[PublicPlayer, ...]
    bank_resources: tuple[int, ...]  # supply stacks are face up (rulebook p.4)
    dev_deck_remaining: int
    discards_pending: tuple[int, ...]
    pending_trade: PendingTrade | None
    own_resources: tuple[int, ...]  # RESOURCES order
    own_dev_cards: tuple[int, ...]  # DEV_CARDS order, unplayed
    history: tuple[VisibleEvent, ...]
    legal_actions: tuple[CanonicalAction, ...]  # empty unless seat == current_seat


class Referee:
    def __init__(self, adapter: EngineAdapter) -> None:
        self._adapter = adapter
        self._events: list[Event] = []
        self._visible: list[list[VisibleEvent]] = [[] for _ in range(NUM_SEATS)]
        self._action_log: list[tuple[CanonicalAction, Any]] = []

    @classmethod
    def new_game(cls, seed: int, engine_players: Sequence[Any] | None = None) -> "Referee":
        return cls(EngineAdapter.new_game(seed, engine_players))

    # ----- Privileged (referee, arena, offline evaluation)

    @property
    def adapter(self) -> EngineAdapter:
        return self._adapter

    @property
    def events(self) -> tuple[Event, ...]:
        """Full-truth event log including every seat's private payloads."""
        return tuple(self._events)

    @property
    def action_log(self) -> tuple[tuple[CanonicalAction, Any], ...]:
        """Executed actions with their random outcomes; with the seed, enough to replay."""
        return tuple(self._action_log)

    # ----- Play

    def execute(self, action: CanonicalAction) -> None:
        state = self._adapter.referee_game.state
        turn_index = self._adapter.num_turns
        before = snapshot(state)
        outcome = self._adapter.execute(action)
        after = snapshot(state)

        self._record(action_event(len(self._events), turn_index, action, outcome, before, after))
        for event in bonus_events(len(self._events), turn_index, before, after):
            self._record(event)
        if self._adapter.is_over:
            self._record(self._game_end_event(turn_index))
        self._action_log.append((action, outcome))

    def visible_history(self, seat: Seat) -> tuple[VisibleEvent, ...]:
        return tuple(self._visible[seat])

    def observe(self, seat: Seat) -> PlayerObservation:
        adapter = self._adapter
        state = adapter.referee_game.state
        ps = state.player_state
        own = f"P{seat}"
        trade = None
        if state.is_resolving_trade:
            trade = PendingTrade(
                offerer=int(state.current_turn_index),
                give=tuple(state.current_trade[:5]),
                receive=tuple(state.current_trade[5:10]),
                accepted_by=tuple(bool(a) for a in state.acceptees),
            )
        return PlayerObservation(
            schema_version=SCHEMA_VERSION,
            seat=seat,
            current_seat=adapter.current_seat,
            turn_seat=adapter.turn_seat,
            prompt=adapter.prompt,
            is_initial_build_phase=adapter.is_initial_build_phase,
            num_turns=adapter.num_turns,
            board=_board_view(state),
            players=tuple(_public_player(state, s) for s in range(NUM_SEATS)),
            bank_resources=tuple(int(n) for n in state.resource_freqdeck),
            dev_deck_remaining=len(state.development_listdeck),
            discards_pending=tuple(int(n) for n in state.discard_counts),
            pending_trade=trade,
            own_resources=tuple(int(ps[f"{own}_{r}_IN_HAND"]) for r in RESOURCES),
            own_dev_cards=tuple(int(ps[f"{own}_{d}_IN_HAND"]) for d in DEV_CARDS),
            history=self.visible_history(seat),
            legal_actions=adapter.legal_actions() if seat == adapter.current_seat else (),
        )

    def _record(self, event: Event) -> None:
        self._events.append(event)
        for seat in range(NUM_SEATS):
            self._visible[seat].append(event.visible_to(seat))

    def _game_end_event(self, turn_index: int) -> Event:
        winner = self._adapter.winner_seat
        ps = self._adapter.referee_game.state.player_state
        revealed = None if winner is None else int(ps[f"P{winner}_VICTORY_POINT_IN_HAND"])
        payload = {
            "winner_seat": winner,
            "truncated": self._adapter.is_truncated,
            "revealed_victory_point_cards": revealed,  # winner reveals VP cards to show 10
        }
        return Event(len(self._events), turn_index, None, EventType.GAME_END, payload)


def _board_view(state: Any) -> BoardView:
    board = state.board
    seat_of = {color: seat for seat, color in enumerate(state.colors)}
    tiles = tuple(
        sorted(
            (tuple(coord), tile.resource, tile.number)
            for coord, tile in board.map.land_tiles.items()
        )
    )
    ports = tuple(
        sorted(
            ((resource, tuple(sorted(nodes))) for resource, nodes in board.map.port_nodes.items()),
            key=lambda port: str(port[0]),
        )
    )
    buildings = tuple(
        sorted((node, seat_of[color], kind) for node, (color, kind) in board.buildings.items())
    )
    roads = tuple(sorted((a, b, seat_of[color]) for (a, b), color in board.roads.items() if a < b))
    return BoardView(tiles, tuple(board.robber_coordinate), ports, buildings, roads)


def _public_player(state: Any, seat: Seat) -> PublicPlayer:
    ps = state.player_state
    key = f"P{seat}"
    return PublicPlayer(
        seat=seat,
        visible_victory_points=int(ps[f"{key}_VICTORY_POINTS"]),
        resource_card_count=sum(int(ps[f"{key}_{r}_IN_HAND"]) for r in RESOURCES),
        dev_card_count=sum(int(ps[f"{key}_{d}_IN_HAND"]) for d in DEV_CARDS),
        played_dev_cards=tuple(int(ps[f"{key}_PLAYED_{d}"]) for d in DEV_CARDS),
        roads_available=int(ps[f"{key}_ROADS_AVAILABLE"]),
        settlements_available=int(ps[f"{key}_SETTLEMENTS_AVAILABLE"]),
        cities_available=int(ps[f"{key}_CITIES_AVAILABLE"]),
        longest_road_length=int(ps[f"{key}_LONGEST_ROAD_LENGTH"]),
        has_longest_road=bool(ps[f"{key}_HAS_ROAD"]),
        has_largest_army=bool(ps[f"{key}_HAS_ARMY"]),
        has_rolled=bool(ps[f"{key}_HAS_ROLLED"]),
        has_played_dev_card_this_turn=bool(ps[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]),
    )
