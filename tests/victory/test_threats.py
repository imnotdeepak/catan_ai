"""Turn-order threats and defensive robber moves (STR-021, plan §5.5)."""

import math
from pathlib import Path

import pytest
from catanatron.models.actions import generate_playable_actions
from catanatron.state_functions import build_city

from catan_ai.arena.contestants import Contestant
from catan_ai.arena.tournament import run_tournament
from catan_ai.engine import ActionKind
from catan_ai.engine.topology import NUM_SEATS, TILE_NODES
from catan_ai.observations import Referee
from catan_ai.opening.board import analyze
from catan_ai.victory.threats import (
    DefendAgent,
    Threat,
    finish_turn,
    first_winner,
    robber_scores,
    threats,
    turns_until,
    urgency,
)
from tests.observations.test_firewall import play
from tests.rules.helpers import set_hand


def test_a_route_wins_only_on_its_own_turn() -> None:
    assert finish_turn(0, 0.0) == 0  # can win now, on its own turn
    assert finish_turn(2, 0.0) == 2  # can win at its next turn
    assert finish_turn(2, 0.4) == 2 + NUM_SEATS  # needs another round of income
    assert finish_turn(1, 2.0) == 1 + 2 * NUM_SEATS
    assert finish_turn(0, 60.0) == math.inf


def test_equal_routes_go_to_the_seat_that_acts_first() -> None:
    def threat(seat: int, turns: int, rounds: float) -> Threat:
        return Threat(seat, (8, 8), rounds, rounds, turns)

    # Turn seat 1: seat 3 acts two turns from now, seat 0 three.
    race = (threat(0, 3, 1.5), threat(1, 0, 9.0), threat(2, 1, 9.0), threat(3, 2, 1.5))
    assert first_winner(race) == 3
    # A slightly faster route that still needs another round does not overtake.
    race = (threat(0, 3, 1.0), threat(1, 0, 9.0), threat(2, 1, 9.0), threat(3, 2, 1.01))
    assert first_winner(race) == 0
    weights = urgency(race, viewer=1)
    assert weights[1] == 0 and weights[0] == 1 and 0 < weights[3] < 1


def robber_decision(seed: int) -> Referee:
    for referee in play(seed):
        obs = referee.observe(referee.adapter.current_seat)
        if obs.prompt == "MOVE_ROBBER" and not obs.is_initial_build_phase:
            return referee
    raise AssertionError("no robber move")


def test_turns_until_counts_from_the_current_turn() -> None:
    obs = (referee := robber_decision(2)).observe(referee.adapter.current_seat)
    assert turns_until(obs, obs.turn_seat) == 0
    assert sorted(turns_until(obs, s) for s in range(NUM_SEATS)) == [0, 1, 2, 3]
    assert turns_until(obs, (obs.turn_seat + 1) % NUM_SEATS) == 1


def near_win(seed: int) -> tuple[Referee, int]:
    """A robber decision where an opponent is at 9: 7 visible (a settlement, a city and
    both awards) and 2 hidden VP cards, one ore short of a winning city."""
    referee = robber_decision(seed)
    game = referee.adapter.referee_game
    state = game.state
    viewer = referee.adapter.current_seat
    rival = (viewer + 1) % NUM_SEATS
    color = state.colors[rival]
    own = [n for n, s, _ in referee.observe(viewer).board.buildings if s == rival]
    build_city(state, color, own[0])
    state.board.build_city(color, own[0])
    for seat in range(NUM_SEATS):
        state.player_state[f"P{seat}_HAS_ROAD"] = seat == rival
        state.player_state[f"P{seat}_HAS_ARMY"] = seat == rival
    state.player_state[f"P{rival}_VICTORY_POINT_IN_HAND"] = 2
    set_hand(game, rival, (0, 0, 0, 2, 2))
    game.playable_actions = generate_playable_actions(state)
    return referee, rival


def test_hidden_vp_threat_is_counted_in_the_worst_case() -> None:
    referee, rival = near_win(2)
    obs = referee.observe(referee.adapter.current_seat)
    race = threats(obs)
    assert race[rival].points == (7, 9)
    assert race[rival].worst_rounds_to_win < race[rival].rounds_to_win
    # Its hand is hidden: the planner spreads its 4 cards by income (exact ledger: BEL-022),
    # so the deficit is estimated, not known; it still threatens first.
    assert race[rival].worst_rounds_to_win == min(t.worst_rounds_to_win for t in race)
    assert first_winner(race, worst=True) == rival


def test_defense_blocks_and_robs_the_near_winner() -> None:
    """Tactical fixture (plan §5.5): a 9 VP opponent with a winning city resource deficit."""
    for seed in (2, 13):
        referee, rival = near_win(seed)
        obs = referee.observe(referee.adapter.current_seat)
        choice = DefendAgent(inner=None).decide(obs)  # type: ignore[arg-type]
        coordinate, victim = choice.params
        assert victim == rival
        assert rival in {int(analyze(obs).owner[n]) for n in TILE_NODES[tuple(coordinate)]}


def test_blocking_only_yourself_never_scores_positive() -> None:
    for seed in (2, 13):
        obs = (referee := robber_decision(seed)).observe(referee.adapter.current_seat)
        analysis = analyze(obs)
        actions = [a for a in obs.legal_actions if a.kind == ActionKind.MOVE_ROBBER]
        for action, value in zip(actions, robber_scores(obs, actions), strict=True):
            owners = {int(analysis.owner[n]) for n in TILE_NODES[tuple(action.params[0])]}
            if owners <= {obs.seat, -1} and action.params[1] is None:
                assert value <= 0


def test_defend_contestant(tmp_path: Path) -> None:
    assert isinstance(Contestant("defend:random").make_agent("s"), DefendAgent)
    with pytest.raises(ValueError, match="cannot be wrapped"):
        Contestant("defend:catanatron:F")
    with pytest.raises(ValueError, match="Unknown contestant"):
        Contestant("defend:nonsense")
    lineup = [Contestant("defend:random"), *[Contestant("random")] * 3]
    results = run_tournament(lineup, 1, 7, out_dir=tmp_path)
    assert all(r.error is None for r in results)
