"""Board analysis agrees with the engine (STR-016, plan §4.5)."""

import numpy as np
import pytest

from catan_ai.engine import ActionKind
from catan_ai.opening.board import NODE_NEIGHBORS, analyze
from tests.observations.test_firewall import play

SEEDS = [3, 14, 15]


def test_node_graph_is_the_standard_board() -> None:
    degrees = [len(n) for n in NODE_NEIGHBORS.values()]
    assert len(degrees) == 54 and sum(degrees) == 2 * 72
    assert set(degrees) == {2, 3}


@pytest.mark.parametrize("seed", SEEDS)
def test_production_matches_engine(seed: int) -> None:
    referee = next(play(seed))
    engine_map = referee.adapter.referee_game.state.board.map
    analysis = analyze(referee.observe(0))
    for node, counter in engine_map.node_production.items():
        expected = [36 * counter.get(r, 0.0) for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")]
        assert analysis.production[node] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_legality_and_longest_road_match_engine(seed: int) -> None:
    checked = {"settlement": 0, "road": 0, "longest": 0}
    for referee in play(seed):
        if referee.adapter.is_over:
            break
        seat = referee.adapter.current_seat
        obs = referee.observe(seat)
        a = analyze(obs)
        legal = obs.legal_actions
        settlements = {x.params[0] for x in legal if x.kind == ActionKind.BUILD_SETTLEMENT}
        roads = {x.params[0] for x in legal if x.kind == ActionKind.BUILD_ROAD}

        if obs.prompt == "BUILD_INITIAL_SETTLEMENT":
            assert settlements == set(np.flatnonzero(a.buildable))
            checked["settlement"] += 1
        elif settlements:
            assert settlements == set(a.reachable_sites(seat, within=0))
            checked["settlement"] += 1

        if roads and obs.prompt == "PLAY_TURN":
            expected = {
                edge
                for edge in _all_edges()
                if edge not in a.road_owner
                and any(a.distance[seat][n] == 0 and a.owner[n] in (-1, seat) for n in edge)
            }
            assert roads == expected
            checked["road"] += 1

        # Compare with the engine's own recount where conventions agree: no road of
        # the seat touches another player's building (rules matrix Q-05).
        board = referee.adapter.referee_game.state.board
        for s, color in enumerate(referee.adapter.referee_game.state.colors):
            touches = any(
                a.owner[n] not in (-1, s) for e, o in a.road_owner.items() if o == s for n in e
            )
            if not touches:
                fresh = max((len(p) for p in board.continuous_roads_by_player(color)), default=0)
                assert a.longest_road[s] == fresh
                checked["longest"] += 1
    assert min(checked.values()) > 0, checked


def test_longest_road_is_interrupted_but_not_shortened_by_other_buildings() -> None:
    from catan_ai.opening.board import _longest_road

    owner = np.full(54, -1, dtype=np.int8)
    chain = {(0, 1): 0, (1, 2): 0, (2, 3): 0, (3, 4): 0, (4, 5): 0}  # a ring segment
    assert all(b in NODE_NEIGHBORS[a] for a, b in chain)
    assert _longest_road(0, chain, owner) == 5
    owner[2] = 1  # another player settles in the middle: 2 + 3
    assert _longest_road(0, chain, owner) == 3
    owner[2], owner[5] = -1, 1  # at the end: every piece still counts
    assert _longest_road(0, chain, owner) == 5
    owner[5] = 0  # own buildings never interrupt
    assert _longest_road(0, chain, owner) == 5


def _all_edges() -> list[tuple[int, int]]:
    from catan_ai.models.features import EDGES

    return list(EDGES)


def test_new_road_effects() -> None:
    for referee in play(8):
        obs = referee.observe(referee.adapter.current_seat)
        roads = [x.params[0] for x in obs.legal_actions if x.kind == ActionKind.BUILD_ROAD]
        if obs.prompt != "PLAY_TURN" or not roads:
            continue
        a = analyze(obs)
        seat = obs.seat
        for edge in roads:
            after = a.distance_after_road(seat, edge)
            assert (after <= a.distance[seat]).all()
            assert after[edge[0]] == 0 and after[edge[1]] == 0
            assert a.longest_road_after(seat, edge) >= a.longest_road[seat]
        return
    pytest.fail("no road decision found")
