"""Domestic trading: proposals, judgement, negotiators and the trading agent (TRD-026..028)."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from catan_ai.arena import Contestant, play_match
from catan_ai.arena.agents import RandomAgent
from catan_ai.arena.tournament import run_tournament
from catan_ai.engine import ActionKind
from catan_ai.engine.actions import RESOURCES
from catan_ai.observations import Referee
from catan_ai.trading.negotiators import STYLES, NegotiatorAgent, judge
from catan_ai.trading.proposals import bundles, candidate_offers, is_lawful
from catan_ai.trading.response import FEATURES, ResponseModel
from tests.observations.test_firewall import play
from tests.rules.helpers import force_turn, set_hand

LINEUPS = [
    [
        "trader:rules:heuristic",
        "negotiator:cooperative:heuristic",
        "negotiator:opportunistic:heuristic",
        "negotiator:self-interested:heuristic",
    ],
    [
        "trader:model:heuristic",
        "negotiator:risk-averse:heuristic",
        "negotiator:trade-resistant:heuristic",
        "negotiator:cooperative:heuristic",
    ],
]


def test_bundles_and_lawful_offers() -> None:
    assert bundles(np.array([2, 0, 1, 0, 0]), 2) == [
        (0, 0, 1, 0, 0), (1, 0, 0, 0, 0), (1, 0, 1, 0, 0), (2, 0, 0, 0, 0),
    ]  # fmt: skip
    hand = (1, 1, 0, 0, 0)
    assert is_lawful((1, 0, 0, 0, 0), (0, 0, 1, 0, 0), hand)
    assert not is_lawful((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), hand)  # a gift
    assert not is_lawful((1, 0, 0, 0, 0), (1, 0, 1, 0, 0), hand)  # like for like
    assert not is_lawful((2, 0, 0, 0, 0), (0, 0, 1, 0, 0), hand)  # cards not held


def test_candidate_offers_are_lawful_and_help_the_proposer() -> None:
    seen = 0
    for step, referee in enumerate(play(7)):
        obs = referee.observe(referee.adapter.current_seat)
        if step % 15 or obs.is_initial_build_phase:
            continue
        for offer in candidate_offers(obs):
            assert is_lawful(offer.give, offer.receive, obs.own_resources)
            assert offer.benefit > 0
            seen += 1
    assert seen > 10


def test_a_trade_that_lets_the_other_side_win_is_refused() -> None:
    """Tactical fixture (plan §8.5): the proposer is at 9 points and one wheat short of
    a city; a responder never supplies the wheat, however good the deal."""
    referee = Referee.new_game(5)
    for r in play(5):
        if not r.observe(r.adapter.current_seat).is_initial_build_phase:
            referee = r
            break
    game = referee.adapter.referee_game
    proposer = referee.adapter.current_seat
    responder = (proposer + 1) % 4
    state = game.state
    # 2 settlements + both awards = 6 visible, + 3 hidden VP cards = 9.
    for seat in range(4):
        state.player_state[f"P{seat}_HAS_ROAD"] = state.player_state[f"P{seat}_HAS_ARMY"] = (
            seat == proposer
        )
    state.player_state[f"P{proposer}_VICTORY_POINT_IN_HAND"] = 3
    state.player_state[f"P{proposer}_KNIGHT_IN_HAND"] = 0
    set_hand(game, proposer, (3, 0, 0, 1, 3))
    set_hand(game, responder, (0, 0, 0, 2, 0))
    force_turn(game, proposer)
    obs = referee.observe(responder)
    # The proposer offers 3 wood for 1 wheat: a great deal for the responder. The hand
    # is set outside the event log, so it is passed as known (as the ledger would know it).
    known = np.array([3, 0, 0, 1, 3], dtype=np.float32)
    verdict = judge(obs, proposer, (3, 0, 0, 0, 0), (0, 0, 0, 1, 0), other_hand=known)
    assert verdict.mine > 0 and verdict.enables_win
    fair = judge(
        obs, proposer, (1, 0, 0, 0, 0), (0, 0, 0, 1, 0), other_hand=known - [0, 0, 0, 1, 0]
    )
    assert not fair.enables_win  # without the wheat it cannot build the city


@pytest.fixture(scope="module")
def trading_games(tmp_path_factory: pytest.TempPathFactory) -> list[Referee]:
    games = []
    for i, lineup in enumerate(LINEUPS):
        result, referee = play_match(f"trade-{i}", 20 + i, [Contestant(n) for n in lineup])
        assert result.error is None
        games.append(referee)
    return games


def test_trading_games_complete_with_lawful_transfers(trading_games: list[Referee]) -> None:
    """Zero invalid transfers: every game ends without error, resources are conserved,
    trades actually happen, and counteroffers are made."""
    kinds: Counter[str] = Counter()
    for referee in trading_games:
        assert referee.adapter.is_over
        state = referee.adapter.referee_game.state
        for r, resource in enumerate(RESOURCES):
            held = sum(state.player_state[f"P{s}_{resource}_IN_HAND"] for s in range(4))
            assert held + state.resource_freqdeck[r] == 19
        kinds.update(e.event_type for e in referee.events)
        kinds["counters"] += sum(
            1
            for e in referee.events
            if e.event_type == "REJECT_TRADE" and e.public["action"]["params"]
        )
    assert kinds[ActionKind.CONFIRM_TRADE] > 5
    assert kinds["counters"] > 0


def test_loop_guard_limits_offers_per_turn(trading_games: list[Referee]) -> None:
    for referee in trading_games:
        offers: Counter[tuple[int, int]] = Counter()
        repeated: Counter[tuple[int, int, str]] = Counter()
        for e in referee.events:
            if e.event_type == ActionKind.OFFER_TRADE:
                offers[(e.turn_index, e.actor)] += 1  # type: ignore[index]
                repeated[(e.turn_index, e.actor, repr(e.public["action"]["params"]))] += 1  # type: ignore[index]
        assert max(offers.values()) <= 2
        assert max(repeated.values()) == 1


def test_negotiator_styles_and_contestants(tmp_path: Path) -> None:
    for style in STYLES:
        agent = Contestant(f"negotiator:{style}:random").make_agent("s")
        assert isinstance(agent, NegotiatorAgent) and agent.name == style
    with pytest.raises(ValueError, match="Unknown option"):
        Contestant("negotiator:greedy:random")
    with pytest.raises(ValueError, match="cannot be wrapped"):
        Contestant("trader:rules:catanatron:F")
    lineup = [Contestant(n) for n in ("trader:rules:random", "negotiator:cooperative:random",
                                      "random", "random")]  # fmt: skip
    results = run_tournament(lineup, 1, 3, out_dir=tmp_path)
    assert all(r.error is None for r in results)
    assert NegotiatorAgent("cooperative", RandomAgent("x")).style.offers_per_turn == 2


def test_response_model_recovers_known_weights(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    x = np.column_stack([np.ones(n), rng.normal(size=(n, len(FEATURES) - 1))])
    truth = rng.normal(scale=0.8, size=len(FEATURES))
    pay = rng.uniform(0.3, 1.0, size=n)
    accepted = rng.random(n) < pay / (1 + np.exp(-(x @ truth)))
    model = ResponseModel.fit(x, pay, accepted)
    assert np.abs(model.weights - truth).max() < 0.15
    model.save(tmp_path / "m.json")
    assert np.allclose(ResponseModel.load(tmp_path / "m.json").weights, model.weights)
    assert 0 < model.yes(x[0]) < 1


def test_traders_log_features_with_their_responses() -> None:
    lineup = ["trader:rules:heuristic", "negotiator:cooperative:heuristic",
              "catanatron:F", "negotiator:trade-resistant:heuristic"]  # fmt: skip
    agents: dict[int, object] = {}
    result, _ = play_match("log", 31, [Contestant(n) for n in lineup], agent_sink=agents)
    assert result.error is None
    trader = agents[0]
    samples = trader.samples  # type: ignore[attr-defined]
    assert len(samples) == len(trader.calibration) > 0  # type: ignore[attr-defined]
    for seat, x, pay, _ in samples:
        assert seat != 0 and x.shape == (len(FEATURES),) and 0 <= pay <= 1
    # The bot never accepts.
    assert not any(accepted for seat, _, _, accepted in samples if seat == 2)
