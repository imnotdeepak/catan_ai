"""Candidate policies for the opponent mixture (OPP-025, plan §7.3).

Each component gives a distribution over the deciding seat's legal actions
in a world (`catan_ai.beliefs.worlds`), i.e. from that seat's information
state with a sampled hand. Deterministic policies are smoothed so one
unexpected move never rules a component out:
P = (1 - SMOOTHING) * pi + SMOOTHING / n.

- `uniform`: every legal action equally likely (the "irrational" component).
- `weighted`: Catanatron's WeightedRandom (cities x10000, settlements x1000,
  development cards x100, everything else x1).
- `victory-point`: Catanatron's VictoryPoint (uniform over the actions that
  reach the most points this step).
- `value-function`: Catanatron's ValueFunction (argmax of its value).
- `heuristic`: our deterministic HeuristicPolicyValue priors.
- `model:<checkpoint>`: a trained policy network: our learned next-action
  predictor, trained on ValueFunction and AlphaBeta play.
"""

from collections.abc import Callable
from typing import Final, Protocol

import numpy as np
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WEIGHTS_BY_ACTION_TYPE

from catan_ai.engine import EngineAdapter
from catan_ai.engine.actions import to_canonical
from catan_ai.models import HeuristicPolicyValue, PolicyValue
from catan_ai.observations import PlayerObservation

SMOOTHING: Final = 0.05

ObserveFn = Callable[[EngineAdapter, int], PlayerObservation]


class Component(Protocol):
    name: str

    def distribution(self, world: EngineAdapter, observe: ObserveFn) -> np.ndarray:
        """Probabilities aligned with `world.legal_actions()`."""
        ...


def smooth(pi: np.ndarray) -> np.ndarray:
    return np.asarray((1 - SMOOTHING) * pi / pi.sum() + SMOOTHING / len(pi))


def _engine_actions(world: EngineAdapter) -> list[object]:
    """The engine's playable actions in `world.legal_actions()` order."""
    game = world.referee_game
    colors = game.state.colors
    by_key = {to_canonical(a, colors).key: a for a in game.playable_actions}
    return [by_key[a.key] for a in world.legal_actions()]


class Uniform:
    name = "uniform"

    def distribution(self, world: EngineAdapter, observe: ObserveFn) -> np.ndarray:
        n = len(world.legal_actions())
        return np.full(n, 1.0 / n)


class WeightedRandom:
    name = "weighted"

    def distribution(self, world: EngineAdapter, observe: ObserveFn) -> np.ndarray:
        weights = [
            WEIGHTS_BY_ACTION_TYPE.get(a.action_type, 1)  # type: ignore[attr-defined]
            for a in _engine_actions(world)
        ]
        pi = np.array(weights, dtype=np.float64)
        return smooth(pi)


class VictoryPoints:
    name = "victory-point"

    def distribution(self, world: EngineAdapter, observe: ObserveFn) -> np.ndarray:
        seat = world.current_seat
        values = []
        for action in _engine_actions(world):
            copy = world.branch(0).referee_game  # own RNG: never perturbs the world
            copy.execute(action)
            values.append(copy.state.player_state[f"P{seat}_ACTUAL_VICTORY_POINTS"])
        best = np.array(values) == max(values)
        return smooth(best.astype(np.float64))


class ValueFunction:
    name = "value-function"

    def distribution(self, world: EngineAdapter, observe: ObserveFn) -> np.ndarray:
        game = world.branch(0).referee_game
        color = game.state.colors[world.current_seat]
        player = ValueFunctionPlayer(color)
        actions = _engine_actions(world)
        choice = player.decide(game, actions)
        pi = np.array([a == choice for a in actions], dtype=np.float64)
        return smooth(pi)


class Policy:
    """A `PolicyValue` (our heuristic or a trained network) seen from the deciding seat."""

    def __init__(self, name: str, policy: PolicyValue) -> None:
        self.name = name
        self.policy = policy
        self._last: tuple[tuple[object, ...], np.ndarray] | None = None

    def distribution(self, world: EngineAdapter, observe: ObserveFn) -> np.ndarray:
        # Several components can share this one as a base: remember the last state.
        state = world.referee_game.state
        records = state.action_records
        key = (id(state), len(records), id(records[-1]) if records else None,
               state.current_player_index, state.current_prompt, world.legal_actions())  # fmt: skip
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        observation = observe(world, world.current_seat)
        priors = np.array(self.policy.evaluate(observation).priors, dtype=np.float64)
        result = smooth(np.maximum(priors, 0) + 1e-12)
        self._last = (key, result)
        return result


def standard_components(checkpoint: str | None = None) -> list[Component]:
    """The baseline mixture: Catanatron's styles, our heuristic and, optionally, a model."""
    components: list[Component] = [
        Uniform(),
        WeightedRandom(),
        VictoryPoints(),
        ValueFunction(),
        Policy("heuristic", HeuristicPolicyValue()),
    ]
    if checkpoint is not None:
        from catan_ai.arena.contestants import load_policy

        components.append(Policy(f"model:{checkpoint}", load_policy(checkpoint)))
    return components
