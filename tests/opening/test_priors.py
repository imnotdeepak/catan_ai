"""Opening priors (STR-017)."""

import numpy as np
import pytest

from catan_ai.engine import ActionKind
from catan_ai.opening.board import analyze
from catan_ai.opening.priors import (
    PRIORS,
    RULES,
    balanced,
    cartel,
    ore_wheat_sheep,
    prior_scores,
    production_after,
    road_builder,
    score,
)
from tests.observations.test_firewall import play

ORE_HEAVY = np.array([0, 0, 2, 5, 5], dtype=np.float32)
WOOD_HEAVY = np.array([5, 5, 2, 0, 0], dtype=np.float32)


def test_priors_prefer_their_own_economies() -> None:
    assert ore_wheat_sheep(ORE_HEAVY) > ore_wheat_sheep(WOOD_HEAVY)
    assert road_builder(WOOD_HEAVY) > road_builder(ORE_HEAVY)
    even = np.array([2, 2, 2, 2, 2], dtype=np.float32)
    assert balanced(even) > balanced(ORE_HEAVY)
    board = np.full(5, 20.0, dtype=np.float32)
    assert cartel(np.array([0, 0, 0, 0, 10.0], dtype=np.float32), board_total=board) > cartel(
        even, board_total=board
    )


def test_scores_on_a_real_opening() -> None:
    referee = next(play(21))
    obs = referee.observe(referee.adapter.current_seat)
    assert obs.prompt == "BUILD_INITIAL_SETTLEMENT"
    analysis = analyze(obs)
    nodes = [a.params[0] for a in obs.legal_actions if a.kind == ActionKind.BUILD_SETTLEMENT]
    for node in nodes:
        assert np.array_equal(production_after(analysis, obs.seat, node), analysis.production[node])
        scores = prior_scores(analysis, obs.seat, node)
        assert scores.shape == (len(PRIORS),) and np.isfinite(scores).all()
        assert score("pips", analysis, obs.seat, node) == pytest.approx(analysis.pips[node])
    best = max(nodes, key=lambda n: score("pips", analysis, obs.seat, n))
    assert analysis.pips[best] == analysis.pips[nodes].max()
    assert set(RULES) == {"pips", *PRIORS}
