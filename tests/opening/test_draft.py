"""Opening pair search and snake-draft prediction (STR-018)."""

from pathlib import Path

import numpy as np
import pytest

from catan_ai.arena import Contestant, run_tournament
from catan_ai.arena.agents import RandomAgent
from catan_ai.engine import ActionKind
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import NODE_NEIGHBORS, analyze
from catan_ai.opening.draft import (
    SNAKE,
    DraftSearchAgent,
    place,
    placed,
    plan_settlement,
    predict_draft,
)
from catan_ai.opening.priors import score
from tests.observations.test_firewall import play


def first_observation(seed: int = 21) -> PlayerObservation:
    referee = next(play(seed))
    return referee.observe(referee.adapter.current_seat)


def test_snake_order() -> None:
    assert SNAKE == (0, 1, 2, 3, 3, 2, 1, 0)


def test_predicted_draft_obeys_the_distance_rule() -> None:
    analysis = analyze(first_observation())
    assert placed(analysis) == 0
    final, picks = predict_draft(analysis, 0, len(SNAKE), "pips")
    assert len(set(picks)) == len(SNAKE)
    for i, node in enumerate(picks):
        assert final.owner[node] == SNAKE[i]
        assert not any(n in picks for n in NODE_NEIGHBORS[node])
    assert not final.buildable[picks].any()
    # Greedy: the first pick is a highest-pip site on the empty board.
    assert analysis.pips[picks[0]] == analysis.pips[analysis.buildable].max()


def test_place_blocks_the_site_and_its_neighbors() -> None:
    analysis = analyze(first_observation())
    after = place(analysis, 2, 10)
    assert after.owner[10] == 2 and not after.buildable[[10, *NODE_NEIGHBORS[10]]].any()
    assert analysis.owner[10] == -1 and analysis.buildable[10]  # the original is untouched


def test_first_pick_plan_is_at_least_as_good_as_the_greedy_pair() -> None:
    obs = first_observation()
    analysis = analyze(obs)
    plan = plan_settlement(analysis, obs.seat, "balanced")
    assert plan.second is not None and analysis.buildable[plan.settlement]
    assert plan.final.owner[plan.second] == obs.seat

    # The greedy alternative: best single site now, then the predicted draft.
    second_at = len(SNAKE) - 1 - obs.seat
    greedy = int(
        max(
            np.flatnonzero(analysis.buildable),
            key=lambda n: score("balanced", analysis, obs.seat, int(n)),
        )
    )
    board, _ = predict_draft(place(analysis, obs.seat, greedy), 1, second_at, "pips")
    greedy_pair = max(
        score("balanced", board, obs.seat, int(n)) for n in np.flatnonzero(board.buildable)
    )
    planned, _ = predict_draft(place(analysis, obs.seat, plan.settlement), 1, second_at, "pips")
    assert score("balanced", planned, obs.seat, plan.second) >= greedy_pair


def test_plan_rejects_a_seat_that_is_not_on_the_clock() -> None:
    obs = first_observation()
    with pytest.raises(ValueError, match="not on the clock"):
        plan_settlement(analyze(obs), (obs.seat + 1) % 4, "balanced")


def test_agent_roads_lead_to_open_sites_and_games_finish(tmp_path: Path) -> None:
    lineup = [Contestant("draft:balanced:random"), Contestant("draft-settle:pips:random")]
    results = run_tournament(
        [*lineup, Contestant("random"), Contestant("random")], 1, 5, out_dir=tmp_path
    )
    assert all(r.error is None for r in results)

    agent = DraftSearchAgent("balanced", RandomAgent("x"))
    for referee in play(21):
        obs = referee.observe(referee.adapter.current_seat)
        if obs.prompt == "BUILD_INITIAL_SETTLEMENT":
            choice = agent.decide(obs)
            assert choice in obs.legal_actions and choice.kind == ActionKind.BUILD_SETTLEMENT
        elif obs.prompt == "BUILD_INITIAL_ROAD":
            assert agent.decide(obs) in obs.legal_actions
        else:
            break


def test_unknown_rules_fail_fast() -> None:
    with pytest.raises(ValueError, match="rule"):
        DraftSearchAgent("nope", RandomAgent("x"))
    with pytest.raises(ValueError, match="rule"):
        Contestant("draft:nope:random")
