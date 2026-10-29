"""Opponent model: a Bayesian policy mixture over sampled worlds (OPP-025, plan §7).

For every opponent the model keeps a posterior over candidate policies
(`catan_ai.opponents.components`). Each opponent decision with a real choice
updates it by the likelihood of the action taken under each policy, averaged
over sampled worlds (so over the opponent's possible hidden hands):

    P(policy | actions) ∝ P(policy) · Π_t mean_w P(a_t | world_w, policy)

Opening style is tracked separately: each initial settlement updates a
posterior over the six opening priors (`catan_ai.opening.priors`) by a
softmax of their site scores. The posterior is never frozen: every later
action keeps updating it.

Predictions (`OpponentPrediction`, plan §7.2) are computed from the
opponent's information state in each world, never from referee truth.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from catan_ai.beliefs.worlds import WorldSampler
from catan_ai.engine import EngineAdapter
from catan_ai.engine.topology import NUM_SEATS
from catan_ai.observations import PlayerObservation, Referee
from catan_ai.observations.events import VisibleEvent
from catan_ai.opening.board import analyze
from catan_ai.opening.priors import RULES, score
from catan_ai.opponents.components import Component, standard_components
from catan_ai.victory.planner import plan_for

STYLE_TEMPERATURE: Final = 2.0  # prior-score units per e-fold of preference
HORIZON: Final = 15  # rounds for win_by_horizon
LIKELIHOOD_WORLDS: Final = 2


@dataclass(frozen=True)
class OpponentPrediction:
    player_id: int
    next_action_distribution: dict[str, float]  # action key -> probability; empty if not deciding
    style_posterior: dict[str, float]  # opening style -> weight
    policy_posterior: dict[str, float]  # mixture component -> weight
    objective_probabilities: dict[str, float]  # first objective kind -> P(pursuing)
    win_by_horizon: tuple[float, ...]  # P(this player's route completes by round h), h = 1..H
    uncertainty: dict[str, float] = field(default_factory=dict)


def observe_world(
    world: EngineAdapter, seat: int, history: tuple[VisibleEvent, ...]
) -> PlayerObservation:
    """`seat`'s observation of a world. The history is the viewer's: public events are
    shared; `seat`'s own private payloads are not known to the viewer."""
    referee = Referee(world)
    referee._visible[seat] = list(history)
    return referee.observe(seat)


class OpponentModel:
    def __init__(
        self,
        viewer: int,
        components: list[Component] | None = None,
        worlds: int = 4,
        likelihood_worlds: int = LIKELIHOOD_WORLDS,
        seed: int = 0,
        diagnostics: bool = False,
    ) -> None:
        self.viewer = viewer
        self.components = components if components is not None else standard_components()
        self.sampler = WorldSampler(viewer, count=worlds, seed=seed)
        self.likelihood_worlds = likelihood_worlds
        k = len(self.components)
        self.log_policy = np.full((NUM_SEATS, k), -math.log(k))
        self.log_style = np.full((NUM_SEATS, len(RULES)), -math.log(len(RULES)))
        self.scored = np.zeros(NUM_SEATS, dtype=int)
        self.log_loss: list[float] = []  # mixture's predictive loss, before each update
        self.diagnostics: list[dict[str, Any]] | None = [] if diagnostics else None
        self._history: tuple[VisibleEvent, ...] = ()

    # ----- Updates

    def update(self, observation: PlayerObservation) -> None:
        self._history = observation.history
        self.sampler.update(observation, before=self._score)

    def _score(self, event: VisibleEvent, worlds: list[EngineAdapter]) -> None:
        seat = event.actor
        if seat is None or seat == self.viewer or not worlds:
            return
        payload = event.public["action"]
        params = () if payload["kind"] == "REJECT_TRADE" else tuple(payload["params"])
        key = f"{payload['kind']}:{seat}:{params!r}"  # a rejection's counteroffer is talk
        likelihoods = np.zeros(len(self.components))
        used = 0
        rank_view: tuple[np.ndarray, int] | None = None  # first world's [components, n], index
        live = [w for w in worlds if not w.is_over]  # a world that ended early is dropped next
        for world in live[: self.likelihood_worlds]:
            if world.current_seat != seat:
                return
            legal = world.legal_actions()
            if len(legal) < 2:
                return  # no choice, no evidence
            keys = [a.key for a in legal]
            if key not in keys:
                continue  # offers are not enumerated; contradicted worlds are skipped
            index = keys.index(key)
            dists = np.array([c.distribution(world, self._observe) for c in self.components])
            likelihoods += dists[:, index]
            if rank_view is None:
                rank_view = (dists, index)
                if world.prompt == "BUILD_INITIAL_SETTLEMENT":
                    self._score_style(world, seat, legal[index].params[0])
            used += 1
        if used == 0 or rank_view is None:
            return
        likelihoods /= used
        posterior = np.exp(self.log_policy[seat] - self.log_policy[seat].max())
        posterior /= posterior.sum()
        self.log_loss.append(-math.log(max(float(posterior @ likelihoods), 1e-300)))
        if self.diagnostics is not None:
            dists, index = rank_view
            mixture = posterior @ dists
            rank = int((mixture > mixture[index]).sum())
            self.diagnostics.append(
                {
                    "seat": seat,
                    "mixture": float(posterior @ likelihoods),
                    "components": likelihoods.copy(),
                    "rank": rank,
                    "choices": len(mixture),
                }
            )
        self.log_policy[seat] += np.log(np.maximum(likelihoods, 1e-300))
        self.log_policy[seat] -= _logsumexp(self.log_policy[seat])
        self.scored[seat] += 1

    def _score_style(self, world: EngineAdapter, seat: int, node: int) -> None:
        observation = self._observe(world, seat)
        analysis = analyze(observation)
        sites = [a.params[0] for a in observation.legal_actions]
        for r, rule in enumerate(RULES):
            scores = np.array([score(rule, analysis, seat, n) for n in sites]) / STYLE_TEMPERATURE
            log_p = scores - _logsumexp(scores)
            self.log_style[seat, r] += log_p[sites.index(node)]
        self.log_style[seat] -= _logsumexp(self.log_style[seat])

    def _observe(self, world: EngineAdapter, seat: int) -> PlayerObservation:
        return observe_world(world, seat, self._history)

    # ----- Predictions

    def policy_posterior(self, seat: int) -> dict[str, float]:
        weights = np.exp(self.log_policy[seat])
        return {c.name: float(w) for c, w in zip(self.components, weights, strict=True)}

    def style_posterior(self, seat: int) -> dict[str, float]:
        weights = np.exp(self.log_style[seat])
        return {rule: float(w) for rule, w in zip(RULES, weights, strict=True)}

    def predict(self, seat: int) -> OpponentPrediction:
        worlds = self.sampler.worlds
        weights = np.exp(self.log_policy[seat])
        next_action: dict[str, float] = {}
        deciding = [w for w in worlds if w.current_seat == seat and not w.is_over]
        for world in deciding:
            legal = world.legal_actions()
            mix = sum(
                weight * component.distribution(world, self._observe)
                for weight, component in zip(weights, self.components, strict=True)
            )
            for action, p in zip(legal, mix, strict=True):
                next_action[action.key] = next_action.get(action.key, 0.0) + p / len(deciding)

        objectives: dict[str, float] = {}
        rounds = []
        for world in worlds:
            plan = plan_for(self._observe(world, seat), seat)
            kind = plan.first.kind if plan.first is not None else "none"
            objectives[kind] = objectives.get(kind, 0.0) + 1 / len(worlds)
            rounds.append(plan.rounds_to_win)
        win_by = tuple(float(np.mean([r <= h for r in rounds])) for h in range(1, HORIZON + 1))
        entropy = -sum(p * math.log(p) for p in weights if p > 0)
        return OpponentPrediction(
            player_id=seat,
            next_action_distribution=next_action,
            style_posterior=self.style_posterior(seat),
            policy_posterior=self.policy_posterior(seat),
            objective_probabilities=objectives,
            win_by_horizon=win_by,
            uncertainty={
                "policy_entropy": entropy,
                "decisions_scored": float(self.scored[seat]),
                "worlds": float(len(worlds)),
            },
        )


def _logsumexp(x: np.ndarray) -> float:
    top = float(np.max(x))
    return top + math.log(float(np.exp(x - top).sum()))
