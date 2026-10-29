"""Victory route planner (STR-020)."""

import numpy as np
import pytest

from catan_ai.observations import PlayerObservation, Referee
from catan_ai.opening.board import analyze
from catan_ai.victory.planner import (
    CITY,
    MAX_ROADS_TO_SITE,
    MAX_ROUNDS,
    SETTLEMENT,
    objectives,
    pay,
    plan_for,
    plans,
    rounds_to_afford,
)
from catan_ai.victory.score import POINTS_TO_WIN, pieces_left
from tests.observations.test_firewall import play
from tests.rules.helpers import set_hand

FOUR = np.full(5, 4.0, dtype=np.float32)


def test_rounds_to_afford() -> None:
    zero = np.zeros(5, dtype=np.float32)
    assert rounds_to_afford(CITY, np.array([0, 0, 0, 2, 3.0]), zero, FOUR) == 0.0
    assert rounds_to_afford(CITY, zero, zero, FOUR) == MAX_ROUNDS
    rate = np.array([0, 0, 0, 0.5, 0.5], dtype=np.float32)
    assert rounds_to_afford(CITY, zero, rate, FOUR) == pytest.approx(6.0, abs=1e-3)
    # Only wood income: 4:1 trades for everything, faster with a 2:1 wood port.
    wood = np.array([1.0, 0, 0, 0, 0], dtype=np.float32)
    port = np.array([2.0, 4, 4, 4, 4], dtype=np.float32)
    slow = rounds_to_afford(SETTLEMENT, zero, wood, FOUR)
    fast = rounds_to_afford(SETTLEMENT, zero, wood, port)
    assert 0 < fast < slow < MAX_ROUNDS
    assert slow == pytest.approx(13.0, abs=1e-3)  # 1 wood + 3 x 4 for brick, sheep, wheat


def test_pay_trades_surpluses() -> None:
    hand = np.array([9.0, 0, 0, 2, 3], dtype=np.float32)
    left = pay(CITY, hand, FOUR)
    np.testing.assert_allclose(left, [9, 0, 0, 0, 0])
    left = pay(SETTLEMENT, hand, FOUR)  # wood pays itself; 8 wood buy brick and sheep
    np.testing.assert_allclose(left, [0, 0, 0, 1, 3])


def sample_observations(seed: int, every: int = 25) -> list[PlayerObservation]:
    out = []
    for step, referee in enumerate(play(seed)):
        if step % every == 0 and not referee.adapter.is_over:
            obs = referee.observe(referee.adapter.current_seat)
            if not obs.is_initial_build_phase:
                out.append(obs)
    return out


@pytest.mark.parametrize("seed", [2, 13])
def test_plans_respect_pieces_locations_and_time(seed: int) -> None:
    for obs in sample_observations(seed):
        analysis = analyze(obs)
        for plan in plans(obs):
            left = pieces_left(obs, plan.seat)
            settlements = [s.objective for s in plan.steps if s.objective.kind == "settlement"]
            cities = [s.objective for s in plan.steps if s.objective.kind == "city"]
            assert len(cities) <= left["cities"]
            assert len(settlements) <= left["settlements"] + len(cities)
            assert sum(s.objective.roads for s in plan.steps) <= left["roads"]
            targets = [o.target for o in settlements]
            assert len(set(targets)) == len(targets)
            for o in settlements:
                assert o.target is not None and analysis.buildable[o.target]
                assert analysis.distance[plan.seat][o.target] == o.roads <= MAX_ROADS_TO_SITE
            own = set(np.flatnonzero((analysis.owner == plan.seat) & ~analysis.is_city))
            for o in cities:
                assert o.target in own or o.target in targets
            rounds = [s.rounds for s in plan.steps]
            assert rounds == sorted(rounds)
            gained = sum(s.objective.points for s in plan.steps)
            if plan.rounds_to_win < MAX_ROUNDS:
                assert plan.points + gained >= POINTS_TO_WIN
                assert plan.rounds_to_win == pytest.approx(rounds[-1])


def test_contested_sites_carry_an_uncontested_fallback() -> None:
    seen = 0
    for obs in sample_observations(2):
        analysis = analyze(obs)
        others = [s for s in range(4) if s != obs.seat]
        rival = analysis.distance[others].min(axis=0)
        for o in objectives(obs, analysis, obs.seat):
            if o.kind != "settlement" or o.target is None:
                continue
            assert o.contested == (rival[o.target] <= analysis.distance[obs.seat][o.target])
            if o.contested and o.fallback is not None:
                assert rival[o.fallback] > analysis.distance[obs.seat][o.fallback]
                seen += 1
    assert seen > 0


def first_turn(seed: int) -> Referee:
    for referee in play(seed):
        if not referee.observe(referee.adapter.current_seat).is_initial_build_phase:
            return referee
    raise AssertionError("game ended during setup")


def test_robot_can_win_now_with_a_city() -> None:
    """Tactical fixture (plan §5.5): at 9 points with city resources in hand."""
    referee = first_turn(5)
    seat = referee.adapter.current_seat
    state = referee.adapter.referee_game.state
    state.player_state[f"P{seat}_VICTORY_POINT_IN_HAND"] = 7  # 2 settlements + 7 = 9
    state.player_state[f"P{seat}_ACTUAL_VICTORY_POINTS"] += 7
    set_hand(referee.adapter.referee_game, seat, (0, 0, 0, 2, 3))
    plan = plan_for(referee.observe(seat), seat)
    assert plan.points == 9
    assert plan.first is not None and plan.first.kind == "city"
    assert plan.rounds_to_win == 0.0


def test_dev_points_and_awards_are_not_secure() -> None:
    for obs in sample_observations(13)[:5]:
        for o in objectives(obs, analyze(obs), obs.seat):
            if o.kind in ("dev", "army", "road_award"):
                assert not o.secure
            if o.kind == "dev":
                assert 0 <= o.points < 1
