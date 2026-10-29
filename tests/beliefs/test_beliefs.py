"""Resource and development-card beliefs (BEL-022, BEL-023, BEL-024)."""

import numpy as np
import pytest

from catan_ai.beliefs.dev_cards import DECK_COUNTS, dev_belief
from catan_ai.beliefs.particles import ENUMERATED, EXACT, PARTICLES, BeliefTracker, resample
from catan_ai.beliefs.resources import (
    BeliefError,
    ResourceBelief,
    apply_delta,
    apply_steal,
    initial_belief,
)
from catan_ai.engine.actions import RESOURCES
from catan_ai.observations import DEV_CARDS, Referee
from tests.observations.test_firewall import play

SEEDS = (1, 2, 3, 4, 5, 6)


def truth(referee: Referee) -> np.ndarray:
    ps = referee.adapter.referee_game.state.player_state
    return np.array([[ps[f"P{s}_{r}_IN_HAND"] for r in RESOURCES] for s in range(4)])


def true_dev(referee: Referee, seat: int) -> np.ndarray:
    ps = referee.adapter.referee_game.state.player_state
    return np.array([ps[f"P{seat}_{d}_IN_HAND"] for d in DEV_CARDS])


def hand(*rows: tuple[int, ...]) -> ResourceBelief:
    return ResourceBelief(np.array([rows], dtype=np.int16), np.ones(1))


ZERO = (0, 0, 0, 0, 0)


def test_unseen_steal_branches_by_the_victims_cards() -> None:
    belief = apply_steal(hand(ZERO, (2, 1, 0, 0, 0), ZERO, ZERO), thief=0, victim=1, resource=None)
    assert belief.size == 2
    assert belief.prob_of(0, (1, 0, 0, 0, 0)) == pytest.approx(2 / 3)
    assert belief.prob_of(0, (0, 1, 0, 0, 0)) == pytest.approx(1 / 3)
    np.testing.assert_allclose(belief.expected()[1], [4 / 3, 2 / 3, 0, 0, 0])


def test_later_public_deltas_prune_hypotheses() -> None:
    belief = apply_steal(hand(ZERO, (2, 1, 0, 0, 0), ZERO, ZERO), thief=0, victim=1, resource=None)
    spend = np.zeros((4, 5), dtype=np.int16)
    spend[1, 1] = -1  # the victim then spends a brick: it still had one
    belief = apply_delta(belief, spend)
    assert belief.exact and belief.prob_of(0, (1, 0, 0, 0, 0)) == 1.0


def test_seen_steals_and_merging_keep_the_ledger_exact() -> None:
    belief = apply_steal(hand(ZERO, (2, 1, 0, 0, 0), ZERO, ZERO), thief=0, victim=1, resource=1)
    assert belief.exact and belief.prob_of(1, (2, 0, 0, 0, 0)) == 1.0
    # Two unseen steals of wood or brick give 3 distinct joint hands, not 4.
    start = hand(ZERO, (2, 2, 0, 0, 0), ZERO, ZERO)
    twice = apply_steal(apply_steal(start, 0, 1, None), 0, 1, None)
    assert twice.size == 3 and twice.weights.sum() == pytest.approx(1)


def test_impossible_public_change_is_a_conservation_failure() -> None:
    spend = np.zeros((4, 5), dtype=np.int16)
    spend[2, 4] = -1
    with pytest.raises(BeliefError):
        apply_delta(initial_belief(), spend)


@pytest.fixture(scope="module")
def tracked() -> dict[str, int]:
    """Track every seat through random games; check truth and conservation at every decision."""
    stats = {"decisions": 0, "enumerated": 0}
    for seed in SEEDS:
        trackers = [BeliefTracker(s) for s in range(4)]
        for referee in play(seed):
            real = truth(referee)
            for seat, tracker in enumerate(trackers):
                obs = referee.observe(seat)
                belief = tracker.update(obs)  # asserts conservation
                assert (belief.hands == real[None]).all(axis=(1, 2)).any(), "truth ruled out"
                assert tracker.tier in (EXACT, ENUMERATED)
                check_dev(referee, obs)
            stats["decisions"] += 1
            stats["enumerated"] += trackers[0].tier == ENUMERATED
    return stats


def check_dev(referee: Referee, obs: object) -> None:
    belief = dev_belief(obs)  # type: ignore[arg-type]
    for seat in range(4):
        if seat == belief.viewer:
            continue
        real = true_dev(referee, seat)
        assert (real <= belief.unseen).all()
        assert belief.expected(seat).sum() == pytest.approx(real.sum())


def test_tracking_keeps_the_truth_and_conserves_cards(tracked: dict[str, int]) -> None:
    assert tracked["decisions"] > 1000
    assert tracked["enumerated"] > 0  # unseen steals were exercised


def test_dev_belief_is_a_proper_distribution() -> None:
    assert DECK_COUNTS.sum() == 25
    for referee in play(3):
        obs = referee.observe(referee.adapter.current_seat)
        belief = dev_belief(obs)
        for seat in range(4):
            dist = belief.hidden_vp_distribution(seat)
            assert dist.sum() == pytest.approx(1)
            assert 1 - dist[0] == pytest.approx(belief.prob_holds(seat, "VICTORY_POINT"))


def test_cards_held_unplayed_are_more_likely_vp_cards() -> None:
    from catan_ai.beliefs.dev_cards import DevBelief

    unseen = DECK_COUNTS.copy()
    fresh, held = (
        DevBelief(unseen, np.array([0, 2, 0, 0]), np.array([0, exposure, 0, 0]), 0)
        for exposure in (0.0, 6.0)
    )
    plain = DevBelief(unseen, np.array([0, 2, 0, 0]), np.zeros(4), 0, play_rate=0, hoard=0)
    assert fresh.prob_holds(1, "VICTORY_POINT") == pytest.approx(
        plain.prob_holds(1, "VICTORY_POINT")
    )
    assert held.prob_holds(1, "VICTORY_POINT") > 2 * fresh.prob_holds(1, "VICTORY_POINT")
    assert held.prob_holds(1, "KNIGHT") < fresh.prob_holds(1, "KNIGHT")
    assert held.expected(1).sum() == pytest.approx(2)


def test_resampling_is_unbiased_and_bounded() -> None:
    weights = np.array([0.5, 0.3, 0.2])
    belief = ResourceBelief(np.zeros((3, 4, 5), dtype=np.int16), weights)
    picked = resample(belief, 10, np.random.default_rng(0))
    assert picked.size <= 3 and picked.weights.sum() == pytest.approx(1)
    np.testing.assert_allclose(picked.weights, weights, atol=0.1)


def test_tiny_budget_falls_back_to_particles_and_recovers() -> None:
    """Tier 3: with a budget of 1 the truth is sometimes resampled away; collapse is
    detected and recovered by replaying with a larger budget."""
    tiers, recoveries = set(), 0
    for seed in SEEDS:
        trackers = [BeliefTracker(s, budget=1) for s in range(4)]
        for referee in play(seed):
            for seat, tracker in enumerate(trackers):
                tracker.update(referee.observe(seat))
                tiers.add(tracker.tier)
        recoveries += sum(t.recoveries for t in trackers)
    assert PARTICLES in tiers and recoveries > 0


def test_tracker_restarts_on_a_new_history() -> None:
    tracker = BeliefTracker(0)
    first = list(play(1))[-1]
    tracker.update(first.observe(0))
    fresh = Referee.new_game(2)
    belief = tracker.update(fresh.observe(0))
    assert belief.exact and not belief.hands.any()
