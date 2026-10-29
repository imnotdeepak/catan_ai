"""Player-view feature encoding for the learned baseline (ML-012, ML-013).

The only input is a `PlayerObservation`, so encoded features cannot contain
anything the viewing seat could not know (plan §2.1). Seats are encoded
relative to the viewer (0 = me, 1 = next to act after me, ...), so one model
plays every seat.

This is feature version 1, the Release 0.2 baseline (plan §9.2). Later versions
append blocks to it; see `encoding.py`.

Board topology (tile coordinates, node ids, edges) is Catanatron's static base
map (`catan_ai.engine.topology`), re-exported here.
"""

from collections import defaultdict
from typing import Final

import numpy as np

from catan_ai.engine.actions import RESOURCES, ActionKind, CanonicalAction
from catan_ai.engine.topology import (
    EDGE_INDEX,
    EDGES,
    NUM_EDGES,
    NUM_NODES,
    NUM_SEATS,
    NUM_TILES,
    PORT_KINDS,
    TILE_COORDS,
    TILE_INDEX,
    TILE_NODES,
    TILE_RESOURCES,
    pips,
)
from catan_ai.observations import DEV_CARDS, PlayerObservation

__all__ = [
    "ACTION_DIM",
    "ACTION_KINDS",
    "EDGES",
    "EDGE_INDEX",
    "NUM_EDGES",
    "NUM_NODES",
    "NUM_SEATS",
    "NUM_TILES",
    "PORT_KINDS",
    "PROMPTS",
    "PROMPT_SLICE",
    "STATE_DIM",
    "TILE_COORDS",
    "TILE_INDEX",
    "TILE_NODES",
    "TILE_RESOURCES",
    "BoardContext",
    "encode_actions",
    "encode_state",
    "pips",
]

PROMPTS: Final = (
    "BUILD_INITIAL_SETTLEMENT",
    "BUILD_INITIAL_ROAD",
    "PLAY_TURN",
    "DISCARD",
    "MOVE_ROBBER",
    "DECIDE_TRADE",
    "DECIDE_ACCEPTEES",
)
ACTION_KINDS: Final = tuple(ActionKind)

# ----- State layout

TILE_WIDTH: Final = len(TILE_RESOURCES) + 2  # resource, pips, robber
NODE_WIDTH: Final = 2 * NUM_SEATS + len(RESOURCES) + len(PORT_KINDS)  # building, production, port
EDGE_WIDTH: Final = NUM_SEATS  # road owner
PLAYER_WIDTH: Final = 3 + len(DEV_CARDS) + 3 + 1 + 4
GLOBAL_WIDTH: Final = (
    len(PROMPTS)
    + 2  # initial phase, turn count
    + NUM_SEATS  # whose turn
    + len(RESOURCES)  # bank
    + 1  # dev deck
    + NUM_SEATS  # discards pending
    + 1  # trade pending
    + NUM_SEATS  # offerer
    + 2 * len(RESOURCES)  # give, receive
    + NUM_SEATS  # accepted by
    + len(RESOURCES)  # own resources
    + 1  # own resource total
    + len(DEV_CARDS)  # own dev cards
)
_TILES_AT: Final = 0
_NODES_AT: Final = _TILES_AT + NUM_TILES * TILE_WIDTH
_EDGES_AT: Final = _NODES_AT + NUM_NODES * NODE_WIDTH
_PLAYERS_AT: Final = _EDGES_AT + NUM_EDGES * EDGE_WIDTH
_GLOBAL_AT: Final = _PLAYERS_AT + NUM_SEATS * PLAYER_WIDTH
STATE_DIM: Final = _GLOBAL_AT + GLOBAL_WIDTH
PROMPT_SLICE: Final = slice(_GLOBAL_AT, _GLOBAL_AT + len(PROMPTS))  # prompt one-hot comes first

# ----- Action layout

ACTION_DIM: Final = (
    len(ACTION_KINDS)
    + NUM_NODES  # settlement / city node
    + NUM_EDGES  # road
    + NUM_TILES  # robber destination
    + 1
    + NUM_SEATS  # robber victim: none, or relative seat
    + 2 * len(RESOURCES)  # resource vectors a, b
    + NUM_SEATS  # trade partner
    + len(RESOURCES)
    + len(PORT_KINDS)  # target node(s): production, port
    + len(TILE_RESOURCES)
    + 1
    + NUM_SEATS  # robber tile: resource, pips, buildings exposed by seat
)


class BoardContext:
    """Per-observation lookups shared by the state and action encoders."""

    def __init__(self, observation: PlayerObservation) -> None:
        self.viewer = observation.seat
        self.production = np.zeros((NUM_NODES, len(RESOURCES)), dtype=np.float32)
        self.tile_resource: dict[tuple[int, int, int], str | None] = {}
        self.tile_pips: dict[tuple[int, int, int], int] = {}
        for coord, resource, number in observation.board.tiles:
            self.tile_resource[coord] = resource
            self.tile_pips[coord] = pips(number)
            if resource is not None:
                for node in TILE_NODES[coord]:
                    self.production[node, RESOURCES.index(resource)] += pips(number)
        self.port = np.zeros((NUM_NODES, len(PORT_KINDS)), dtype=np.float32)
        for resource, nodes in observation.board.ports:
            for node in nodes:
                self.port[node, PORT_KINDS.index(resource)] = 1.0
        self.buildings: dict[int, tuple[int, str]] = defaultdict(lambda: (-1, ""))
        for node, owner, kind in observation.board.buildings:
            self.buildings[node] = (self.relative(owner), kind)

    def relative(self, seat: int) -> int:
        return (seat - self.viewer) % NUM_SEATS


def encode_state(observation: PlayerObservation, context: BoardContext | None = None) -> np.ndarray:
    ctx = context or BoardContext(observation)
    rel = ctx.relative
    x = np.zeros(STATE_DIM, dtype=np.float32)

    tiles = x[_TILES_AT:_NODES_AT].reshape(NUM_TILES, TILE_WIDTH)
    for coord, resource, _ in observation.board.tiles:
        row = tiles[TILE_INDEX[coord]]
        row[TILE_RESOURCES.index(resource)] = 1.0
        row[-2] = ctx.tile_pips[coord] / 5
        row[-1] = coord == observation.board.robber

    nodes = x[_NODES_AT:_EDGES_AT].reshape(NUM_NODES, NODE_WIDTH)
    for node, (owner, kind) in ctx.buildings.items():
        nodes[node, owner + (NUM_SEATS if kind == "CITY" else 0)] = 1.0
    nodes[:, 2 * NUM_SEATS : 2 * NUM_SEATS + len(RESOURCES)] = ctx.production / 5
    nodes[:, 2 * NUM_SEATS + len(RESOURCES) :] = ctx.port

    edges = x[_EDGES_AT:_PLAYERS_AT].reshape(NUM_EDGES, EDGE_WIDTH)
    for a, b, owner in observation.board.roads:
        edges[EDGE_INDEX[(a, b)], rel(owner)] = 1.0

    players = x[_PLAYERS_AT:_GLOBAL_AT].reshape(NUM_SEATS, PLAYER_WIDTH)
    for p in observation.players:
        players[rel(p.seat)] = (
            p.visible_victory_points / 10,
            p.resource_card_count / 10,
            p.dev_card_count / 5,
            *(n / 5 for n in p.played_dev_cards),
            p.roads_available / 15,
            p.settlements_available / 5,
            p.cities_available / 4,
            p.longest_road_length / 15,
            p.has_longest_road,
            p.has_largest_army,
            p.has_rolled,
            p.has_played_dev_card_this_turn,
        )

    g = x[_GLOBAL_AT:]
    i = 0

    def put(values: list[float]) -> None:
        nonlocal i
        g[i : i + len(values)] = values
        i += len(values)

    put([float(observation.prompt == p) for p in PROMPTS])
    put([float(observation.is_initial_build_phase), observation.num_turns / 100])
    put([float(rel(observation.turn_seat) == s) for s in range(NUM_SEATS)])
    put([n / 19 for n in observation.bank_resources])
    put([observation.dev_deck_remaining / 25])
    discards = [0.0] * NUM_SEATS
    for seat, n in enumerate(observation.discards_pending):
        discards[rel(seat)] = n / 10
    put(discards)
    trade = observation.pending_trade
    put([float(trade is not None)])
    put([float(trade is not None and rel(trade.offerer) == s) for s in range(NUM_SEATS)])
    put([n / 4 for n in trade.give] if trade else [0.0] * len(RESOURCES))
    put([n / 4 for n in trade.receive] if trade else [0.0] * len(RESOURCES))
    accepted = [0.0] * NUM_SEATS
    if trade is not None:
        for seat, yes in enumerate(trade.accepted_by):
            accepted[rel(seat)] = float(yes)
    put(accepted)
    put([n / 10 for n in observation.own_resources])
    put([sum(observation.own_resources) / 10])
    put([n / 5 for n in observation.own_dev_cards])
    assert i == GLOBAL_WIDTH
    return x


def encode_actions(
    observation: PlayerObservation,
    actions: tuple[CanonicalAction, ...] | None = None,
    context: BoardContext | None = None,
) -> np.ndarray:
    """One row per candidate action, defaulting to the observation's legal actions."""
    ctx = context or BoardContext(observation)
    candidates = observation.legal_actions if actions is None else actions
    out = np.zeros((len(candidates), ACTION_DIM), dtype=np.float32)
    for row, action in zip(out, candidates, strict=True):
        _encode_action(row, action, ctx)
    return out


_NODE_AT: Final = len(ACTION_KINDS)
_EDGE_AT: Final = _NODE_AT + NUM_NODES
_TILE_AT: Final = _EDGE_AT + NUM_EDGES
_VICTIM_AT: Final = _TILE_AT + NUM_TILES
_RES_A_AT: Final = _VICTIM_AT + 1 + NUM_SEATS
_RES_B_AT: Final = _RES_A_AT + len(RESOURCES)
_PARTNER_AT: Final = _RES_B_AT + len(RESOURCES)
_LOCAL_AT: Final = _PARTNER_AT + NUM_SEATS
_ROBBER_LOCAL_AT: Final = _LOCAL_AT + len(RESOURCES) + len(PORT_KINDS)


def _encode_action(row: np.ndarray, action: CanonicalAction, ctx: BoardContext) -> None:
    kind, params = action.kind, action.params
    row[ACTION_KINDS.index(kind)] = 1.0

    def target_nodes(nodes: tuple[int, ...]) -> None:
        local = row[_LOCAL_AT:_ROBBER_LOCAL_AT]
        local[: len(RESOURCES)] = ctx.production[list(nodes)].mean(axis=0) / 5
        local[len(RESOURCES) :] = ctx.port[list(nodes)].max(axis=0)

    def resources(at: int, vector: tuple[int, ...]) -> None:
        row[at : at + len(RESOURCES)] = [n / 4 for n in vector]

    def one_resource(at: int, resource: str, amount: int = 1) -> None:
        row[at + RESOURCES.index(resource)] += amount / 4

    if kind in (ActionKind.BUILD_SETTLEMENT, ActionKind.BUILD_CITY):
        row[_NODE_AT + params[0]] = 1.0
        target_nodes((params[0],))
    elif kind == ActionKind.BUILD_ROAD:
        row[_EDGE_AT + EDGE_INDEX[params[0]]] = 1.0
        target_nodes(params[0])
    elif kind == ActionKind.MOVE_ROBBER:
        coord, victim = params
        row[_TILE_AT + TILE_INDEX[coord]] = 1.0
        row[_VICTIM_AT + (0 if victim is None else 1 + ctx.relative(victim))] = 1.0
        local = row[_ROBBER_LOCAL_AT:]
        local[TILE_RESOURCES.index(ctx.tile_resource[coord])] = 1.0
        local[len(TILE_RESOURCES)] = ctx.tile_pips[coord] / 5
        for node in TILE_NODES[coord]:
            owner, building = ctx.buildings[node]
            if owner >= 0:
                local[len(TILE_RESOURCES) + 1 + owner] += (2 if building == "CITY" else 1) / 6
    elif kind in (ActionKind.DISCARD, ActionKind.PLAY_MONOPOLY):
        one_resource(_RES_A_AT, params[0])
    elif kind == ActionKind.PLAY_YEAR_OF_PLENTY:
        for resource in params:
            one_resource(_RES_A_AT, resource)
    elif kind == ActionKind.MARITIME_TRADE:
        give, count, receive = params
        one_resource(_RES_A_AT, give, count)
        one_resource(_RES_B_AT, receive)
    elif kind == ActionKind.OFFER_TRADE:
        resources(_RES_A_AT, params[0])
        resources(_RES_B_AT, params[1])
    elif kind == ActionKind.CONFIRM_TRADE:
        row[_PARTNER_AT + ctx.relative(params[0])] = 1.0
