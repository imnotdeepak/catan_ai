"""Opponent model: components, identification, predictions and leakage (OPP-025)."""

import random

import pytest

from catan_ai.arena import Contestant, play_match
from catan_ai.engine import CanonicalAction
from catan_ai.observations import Referee
from catan_ai.opponents.components import standard_components
from catan_ai.opponents.model import OpponentModel
from tests.observations.test_firewall import play, scramble_hidden_state


def watch(seating: list[str], seed: int, viewer: int = 0) -> OpponentModel:
    model = OpponentModel(viewer, worlds=2, likelihood_worlds=1, seed=seed)

    def observe(seat: int, referee: Referee, action: CanonicalAction) -> None:
        model.update(referee.observe(viewer))

    play_match(f"opp-{seed}", seed, [Contestant(n) for n in seating], observer=observe)
    return model


@pytest.fixture(scope="module")
def mixed() -> OpponentModel:
    return watch(["random", "random", "catanatron:W", "catanatron:F"], seed=11)


def test_components_give_distributions_over_legal_actions() -> None:
    components = standard_components()
    model = OpponentModel(0, components, worlds=1)
    for step, referee in enumerate(play(5)):
        if step % 60:
            continue
        model.update(referee.observe(0))
        world = model.sampler.worlds[0]
        if world.is_over:
            continue
        n = len(world.legal_actions())
        for component in components:
            dist = component.distribution(world, model._observe)
            assert dist.shape == (n,) and dist.min() > 0
            assert dist.sum() == pytest.approx(1)


def test_known_policies_are_identified(mixed: OpponentModel) -> None:
    """A uniformly random seat goes to `uniform` (or `victory-point`, which plays
    uniformly whenever no action gains a point); ValueFunction's weight goes to the
    components that play like it, not to the random styles."""
    assert all(mixed.scored[1:] > 10)
    random_seat = mixed.policy_posterior(1)
    assert random_seat["uniform"] + random_seat["victory-point"] > 0.9
    weighted = mixed.policy_posterior(2)
    assert weighted["weighted"] > weighted["value-function"]
    posterior = mixed.policy_posterior(3)
    assert posterior["uniform"] + posterior["weighted"] < 0.1


def test_predictions_are_well_formed(mixed: OpponentModel) -> None:
    for seat in (1, 2, 3):
        prediction = mixed.predict(seat)
        assert sum(prediction.policy_posterior.values()) == pytest.approx(1)
        assert sum(prediction.style_posterior.values()) == pytest.approx(1)
        assert sum(prediction.objective_probabilities.values()) == pytest.approx(1)
        assert list(prediction.win_by_horizon) == sorted(prediction.win_by_horizon)
        if prediction.next_action_distribution:
            assert sum(prediction.next_action_distribution.values()) == pytest.approx(1)


def test_predictions_do_not_depend_on_hidden_state() -> None:
    """Leakage: games the viewer cannot tell apart give identical predictions."""
    *_, referee = play(6)
    viewer = 2
    twin = scramble_hidden_state(referee, viewer, random.Random(1))
    predictions = []
    for r in (referee, twin):
        model = OpponentModel(viewer, worlds=2, likelihood_worlds=1, seed=3)
        model.update(r.observe(viewer))
        states = [w.referee_game.state.player_state for w in model.sampler.worlds]
        predictions.append(([model.predict(s) for s in range(4) if s != viewer], states))
    assert predictions[0] == predictions[1]
