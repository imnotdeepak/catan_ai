"""Root search correctness (SRCH-030, plan §10.1-10.2)."""

import random
from collections.abc import Iterator
from dataclasses import replace

from catan_ai.engine import ActionKind
from catan_ai.models import HeuristicPolicyValue, PolicyAgent
from catan_ai.observations import Referee
from catan_ai.search import SearchAgent, SearchConfig, should_search
from catan_ai.search.root import rollout
from tests.observations.test_firewall import scramble_hidden_state

POLICY = HeuristicPolicyValue()
SMALL = SearchConfig(worlds=3, turns=2)


def play(seed: int, steps: int) -> Iterator[Referee]:
    """A heuristic game, yielding the referee at each decision up to `steps`."""
    referee = Referee.new_game(seed)
    agents = [PolicyAgent(POLICY, seed=f"{seed}:{s}") for s in range(4)]
    for _ in range(steps):
        if referee.adapter.is_over:
            return
        yield referee
        seat = referee.adapter.current_seat
        referee.execute(agents[seat].decide(referee.observe(seat)))


def searched_positions(seed: int, steps: int = 400, every: int = 25) -> Iterator[Referee]:
    for step, referee in enumerate(play(seed, steps)):
        obs = referee.observe(referee.adapter.current_seat)
        priors = POLICY.evaluate(obs).priors
        if step % every == 0 and should_search(obs, priors, SMALL):
            yield referee


def _fingerprint(referee: Referee) -> tuple[object, ...]:
    state = referee.adapter.referee_game.state
    return (
        referee.adapter.referee_game.random.getstate(),
        dict(state.player_state),
        list(state.development_listdeck),
        len(referee.events),
    )


def test_searching_does_not_touch_the_real_game() -> None:
    checked = 0
    for referee in searched_positions(3):
        seat = referee.adapter.current_seat
        before = _fingerprint(referee)
        agent = SearchAgent(POLICY, SMALL, seed="a")
        agent.decide(referee.observe(seat))
        assert agent.stats["searches"] == 1 and agent.stats["rollouts"] > 0
        assert _fingerprint(referee) == before  # same next dice, hands, deck, log
        checked += 1
    assert checked >= 3


def test_root_choice_depends_only_on_the_observation() -> None:
    """Indistinguishability (plan §2.1): changing only what the viewer cannot see
    leaves the scores and the choice unchanged."""
    rng = random.Random(0)
    checked = 0
    for referee in searched_positions(5):
        seat = referee.adapter.current_seat
        twin = scramble_hidden_state(referee, seat, rng)
        runs = []
        for r in (referee, twin):
            agent = SearchAgent(POLICY, SMALL, seed="b")
            obs = r.observe(seat)
            picks = should_search(obs, POLICY.evaluate(obs).priors, SMALL)
            runs.append((agent.scores(obs, picks), agent.decide(obs)))
        assert runs[0] == runs[1]
        checked += 1
    assert checked >= 3


def test_rollouts_are_reproducible_and_leave_the_world_alone() -> None:
    referee = next(searched_positions(7))
    seat = referee.adapter.current_seat
    world = referee.adapter.branch(1)  # any engine game will do here
    before = dict(world.referee_game.state.player_state)
    action = referee.observe(seat).legal_actions[0]
    first = rollout(world, action, seat, POLICY, SMALL, seed=42)
    assert rollout(world, action, seat, POLICY, SMALL, seed=42) == first
    assert dict(world.referee_game.state.player_state) == before
    value, played = first
    assert 0.0 <= value <= 1.0 and played > 0


def test_rollouts_that_finish_the_game_score_the_winner() -> None:
    referee = next(searched_positions(7))
    seat = referee.adapter.current_seat
    world = referee.adapter.branch(2)
    action = referee.observe(seat).legal_actions[0]
    long = replace(SMALL, turns=10_000, max_decisions=100_000)
    value, _ = rollout(world, action, seat, POLICY, long, seed=3)
    assert value in (0.0, 1.0)


def test_forced_and_clear_decisions_are_not_searched() -> None:
    referee = next(iter(play(9, 1)))
    obs = referee.observe(referee.adapter.current_seat)
    one = replace(obs, legal_actions=obs.legal_actions[:1], is_initial_build_phase=False)
    assert should_search(one, (1.0,), SMALL) == []
    two = replace(obs, legal_actions=obs.legal_actions[:2], is_initial_build_phase=False)
    assert should_search(two, (0.9, 0.1), SMALL) == []  # clear-cut, nothing at stake
    assert should_search(two, (0.6, 0.4), SMALL) == [0, 1]  # close
    setup = replace(two, is_initial_build_phase=True)
    assert should_search(setup, (0.9, 0.1), SMALL) == [0, 1]


def test_search_agents_are_deterministic_and_legal() -> None:
    choices = []
    for _ in range(2):
        agent = SearchAgent(POLICY, SMALL, seed="c")
        picks = []
        for referee in searched_positions(11, steps=300, every=40):
            obs = referee.observe(referee.adapter.current_seat)
            action = agent.decide(obs)
            assert action in obs.legal_actions
            picks.append(action)
        choices.append(picks)
        assert agent.stats["failures"] == 0
    assert choices[0] == choices[1] and choices[0]
    assert all(a.kind != ActionKind.OFFER_TRADE for a in choices[0])
