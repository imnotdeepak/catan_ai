"""Common policy/value interface, a deterministic heuristic, and the agent (ML-013).

A `PolicyValue` maps one observation to priors over its legal actions and a
four-seat winner distribution. Search, evaluation and agents depend only on
this interface, so the heuristic and the learned network are interchangeable.
"""

import math
import random
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.models.features import TILE_NODES, BoardContext
from catan_ai.observations import DEV_CARDS, PlayerObservation


@dataclass(frozen=True)
class Evaluation:
    priors: tuple[float, ...]  # aligned with observation.legal_actions, sums to 1
    win_probabilities: tuple[float, ...]  # by absolute seat, sums to 1


class PolicyValue(Protocol):
    def evaluate(self, observation: PlayerObservation) -> Evaluation: ...


def softmax(scores: list[float]) -> tuple[float, ...]:
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    return tuple(w / total for w in weights)


_KIND_SCORE: Final = {
    ActionKind.BUILD_CITY: 6.0,
    ActionKind.BUILD_SETTLEMENT: 5.0,
    ActionKind.PLAY_KNIGHT: 2.0,
    ActionKind.PLAY_YEAR_OF_PLENTY: 3.0,
    ActionKind.PLAY_MONOPOLY: 2.0,
    ActionKind.PLAY_ROAD_BUILDING: 2.0,
    ActionKind.BUY_DEV_CARD: 3.0,
    ActionKind.BUILD_ROAD: 1.0,
    ActionKind.ROLL: 1.0,
    ActionKind.END_TURN: 0.0,
    ActionKind.MARITIME_TRADE: -1.0,
    ActionKind.REJECT_TRADE: 0.5,
    ActionKind.CANCEL_TRADE: 0.0,
}


class HeuristicPolicyValue:
    """Deterministic reference implementation: prefer building on productive nodes,
    and estimate winning chances from known victory points."""

    def evaluate(self, observation: PlayerObservation) -> Evaluation:
        ctx = BoardContext(observation)
        scores = []
        for action in observation.legal_actions:
            score = _KIND_SCORE.get(action.kind, 0.0)
            if action.kind in (ActionKind.BUILD_SETTLEMENT, ActionKind.BUILD_CITY):
                score += 0.2 * float(ctx.production[action.params[0]].sum())
            elif action.kind == ActionKind.BUILD_ROAD:
                score += 0.05 * float(ctx.production[list(action.params[0])].sum())
            elif action.kind == ActionKind.MOVE_ROBBER:
                coord, victim = action.params
                score += 0.1 * ctx.tile_pips[coord] * (victim is not None)
                score -= any(ctx.buildings[n][0] == 0 for n in TILE_NODES[coord])
            scores.append(score)
        points = [float(p.visible_victory_points) for p in observation.players]
        points[observation.seat] += observation.own_dev_cards[DEV_CARDS.index("VICTORY_POINT")]
        return Evaluation(softmax(scores), softmax([p / 2 for p in points]))


class PolicyAgent:
    """Plays from any `PolicyValue`. If it fails or returns unusable priors, the
    heuristic decides instead, so the agent can always act legally (plan §9.1)."""

    def __init__(
        self,
        policy: PolicyValue,
        mode: Literal["greedy", "sample"] = "greedy",
        seed: str = "",
    ) -> None:
        self.policy = policy
        self.mode = mode
        self.fallbacks = 0
        self._rng = random.Random(seed)
        self._fallback = HeuristicPolicyValue()

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        legal = observation.legal_actions
        if len(legal) == 1:
            return legal[0]
        try:
            priors = self.policy.evaluate(observation).priors
            if len(priors) != len(legal) or not all(math.isfinite(p) for p in priors):
                raise ValueError("unusable priors")
        except Exception:
            self.fallbacks += 1
            priors = self._fallback.evaluate(observation).priors
        if self.mode == "sample":
            return self._rng.choices(legal, weights=priors)[0]
        return legal[max(range(len(legal)), key=priors.__getitem__)]
