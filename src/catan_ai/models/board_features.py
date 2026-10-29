"""Board-analysis features (features v2, STR-016), appended to the base encoding.

All values come from `catan_ai.opening.board.analyze`, so they are public.
Seats are relative to the viewer, as in the base encoding.
"""

from typing import Final

import numpy as np

from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.models.features import NUM_NODES, NUM_SEATS
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import NODE_NEIGHBORS, BoardAnalysis

NODE_EXTRA: Final = 5  # buildable, my reach, nearest opponent reach, numbers, robber-blocked
PLAYER_EXTRA: Final = 5  # longest road, sites within 0/1/2 roads, best site within 2
BOARD_STATE_DIM: Final = NUM_NODES * NODE_EXTRA + NUM_SEATS * PLAYER_EXTRA
SETTLEMENT_EXTRA: Final = 5  # pips, resources, numbers, contest, denial
ROAD_EXTRA: Final = 6  # sites and best site within 0 and 1 roads after, road gain, takes award
BOARD_ACTION_DIM: Final = SETTLEMENT_EXTRA + ROAD_EXTRA
MAX_REACH: Final = 4


def _reach(distance: np.ndarray) -> np.ndarray:
    """New roads needed, scaled to [0, 1]; 1 also means unreachable."""
    return np.asarray(np.minimum(distance, MAX_REACH) / MAX_REACH, dtype=np.float32)


def _sites(analysis: BoardAnalysis, distance: np.ndarray, within: int) -> tuple[float, float]:
    """(count / 5, best total pips / 15) of buildable sites within `within` roads."""
    sites = np.flatnonzero(analysis.buildable & (distance <= within))
    best = float(analysis.pips[sites].max()) if len(sites) else 0.0
    return len(sites) / 5, best / 15


def board_state(observation: PlayerObservation, analysis: BoardAnalysis) -> np.ndarray:
    viewer = observation.seat
    opponents = [s for s in range(NUM_SEATS) if s != viewer]
    nodes = np.zeros((NUM_NODES, NODE_EXTRA), dtype=np.float32)
    nodes[:, 0] = analysis.buildable
    nodes[:, 1] = _reach(analysis.distance[viewer])
    nodes[:, 2] = _reach(analysis.distance[opponents].min(axis=0))
    nodes[:, 3] = [len(n) / 3 for n in analysis.numbers]
    nodes[:, 4] = analysis.blocked_production.sum(axis=1) / 15

    players = np.zeros((NUM_SEATS, PLAYER_EXTRA), dtype=np.float32)
    for r in range(NUM_SEATS):
        seat = (viewer + r) % NUM_SEATS
        distance = analysis.distance[seat]
        players[r] = (
            analysis.longest_road[seat] / 15,
            _sites(analysis, distance, 0)[0],
            _sites(analysis, distance, 1)[0],
            *_sites(analysis, distance, 2),
        )
    return np.concatenate([nodes.ravel(), players.ravel()])


def board_actions(
    observation: PlayerObservation,
    analysis: BoardAnalysis,
    actions: tuple[CanonicalAction, ...],
) -> np.ndarray:
    out = np.zeros((len(actions), BOARD_ACTION_DIM), dtype=np.float32)
    viewer = observation.seat
    opponents = [s for s in range(NUM_SEATS) if s != viewer]
    rival_road = max(analysis.longest_road[s] for s in opponents)
    for row, action in zip(out, actions, strict=True):
        if action.kind in (ActionKind.BUILD_SETTLEMENT, ActionKind.BUILD_CITY):
            node = action.params[0]
            removed = [n for n in (node, *NODE_NEIGHBORS[node]) if analysis.buildable[n]]
            row[:SETTLEMENT_EXTRA] = (
                analysis.pips[node] / 15,
                np.count_nonzero(analysis.production[node]) / 5,
                len(analysis.numbers[node]) / 3,
                _reach(analysis.distance[opponents, node].min()),
                analysis.pips[removed].sum() / 30
                if action.kind == ActionKind.BUILD_SETTLEMENT
                else 0,
            )
        elif action.kind == ActionKind.BUILD_ROAD:
            edge = action.params[0]
            after = analysis.distance_after_road(viewer, edge)
            longest = analysis.longest_road_after(viewer, edge)
            row[SETTLEMENT_EXTRA:] = (
                *_sites(analysis, after, 0),
                *_sites(analysis, after, 1),
                min(longest - analysis.longest_road[viewer], 2) / 2,
                float(longest >= 5 and longest > rival_road),
            )
    return out
