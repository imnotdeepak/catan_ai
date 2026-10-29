"""A trading agent: offers, responses, counteroffers and confirmation (TRD-027, TRD-028).

Every non-trade decision goes to the wrapped agent. On its own turn, after
rolling, the agent scores each candidate offer (`catan_ai.trading.proposals`,
plus counteroffers it received this turn) by its expected value over every
set of acceptors, under independent responses:

- value of trading with responder r: the rounds the agent saves, minus r's
  saving weighted by how soon r threatens to win (`victory.threats.urgency`);
  a trade after which r could win at once is never made;
- P(r accepts) = P(r can pay) * P(r says yes | can pay). P(can pay) comes from
  the resource belief; P(yes) from either a fixed rule (`mode="rules"`: a
  logistic in r's estimated saving) or the opponent model (`mode="model"`:
  each policy in the mixture asked in sampled worlds where the offer is on
  the table, weighted by the posterior over r's policy);
- the agent then confirms the best acceptor, so the offer is worth
  sum_i P(best acceptor is i) * max(value_i, 0).

It offers the best candidate if that beats a minimum and the best bank/port
trade; at most MAX_OFFERS_PER_TURN offers a turn, never the same one twice
(agent-level loop guard, plan §8.4). As a responder it accepts when its
value is positive and otherwise rejects, attaching its best counteroffer.

Predicted acceptance probabilities are logged with the observed responses
for calibration (`calibration`).
"""

import math
from typing import Final, Literal

import numpy as np

from catan_ai.arena.agents import Agent
from catan_ai.beliefs.particles import BeliefTracker
from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.engine.actions import offer_trade
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import BoardAnalysis, analyze
from catan_ai.opponents.model import OpponentModel
from catan_ai.trading.negotiators import (
    MAX_OFFERS_PER_TURN,
    can_offer,
    counters_this_turn,
    judge,
    offers_this_turn,
)
from catan_ai.trading.proposals import Vector, candidate_offers, is_lawful, maritime_benefits
from catan_ai.victory.threats import threats, urgency

OFFER_MIN: Final = 0.15  # rounds of expected value an offer must bring
ACCEPT_MIN: Final = 0.05
RULE_SLOPE: Final = 2.0
RULE_OFFSET: Final = 0.25
ACCEPTANCE_WORLDS: Final = 1
MODEL_SHORTLIST: Final = 2  # offers whose acceptance the opponent model scores (rules rank them)

Mode = Literal["rules", "model"]


def expected_value(values: dict[int, float], accepts: dict[int, float]) -> float:
    """Value of an offer when the best acceptor is confirmed: sum over seats, best
    first, of P(it is the best acceptor) * its value (never below no trade)."""
    ev, none_yet = 0.0, 1.0
    for seat in sorted(values, key=lambda s: -values[s]):
        ev += none_yet * accepts[seat] * max(values[seat], 0.0)
        none_yet *= 1 - accepts[seat]
    return ev


class TradingAgent:
    def __init__(self, inner: Agent, mode: Mode = "rules", seed: int = 0) -> None:
        self.inner = inner
        self.mode = mode
        self.seed = seed
        self._tracker: BeliefTracker | None = None
        self._model: OpponentModel | None = None
        self.calibration: list[tuple[float, bool]] = []  # (P(accept), accepted) per responder
        self._pending: list[tuple[int, dict[int, float]]] = []  # (offer event index, p by seat)
        self._urgency: tuple[tuple[int, int], np.ndarray] | None = None  # per observation

    # ----- Plumbing

    def _sync(self, observation: PlayerObservation) -> None:
        if self._tracker is None or self._tracker.seat != observation.seat:
            self._tracker = BeliefTracker(observation.seat, seed=self.seed)
            if self.mode == "model":
                from catan_ai.trading.components import trading_components

                self._model = OpponentModel(
                    observation.seat, trading_components(), worlds=2, likelihood_worlds=1,
                    seed=self.seed,
                )  # fmt: skip
        self._tracker.update(observation)
        if self._model is not None:
            self._model.update(observation)
        self._settle(observation)

    def _settle(self, observation: PlayerObservation) -> None:
        """Match logged predictions with the responses that followed."""
        still = []
        for index, predicted in self._pending:
            responses = {
                e.actor: e.event_type == ActionKind.ACCEPT_TRADE
                for e in observation.history[index + 1 :]
                if e.event_type in (ActionKind.ACCEPT_TRADE, ActionKind.REJECT_TRADE)
                and e.actor in predicted
            }
            ended = any(
                e.event_type
                in (ActionKind.CONFIRM_TRADE, ActionKind.CANCEL_TRADE, ActionKind.END_TURN)
                or e.event_type == ActionKind.OFFER_TRADE
                for e in observation.history[index + 1 :]
            ) or len(responses) == len(predicted)
            if not ended:
                still.append((index, predicted))
                continue
            for seat, p in predicted.items():
                if seat in responses:
                    self.calibration.append((p, responses[seat]))
        self._pending = still

    # ----- Decisions

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        if len(observation.legal_actions) == 1 and observation.prompt != "PLAY_TURN":
            return observation.legal_actions[0]
        if observation.prompt in ("DECIDE_TRADE", "DECIDE_ACCEPTEES") or can_offer(observation):
            self._sync(observation)
        if observation.prompt == "DECIDE_TRADE":
            return self.respond(observation)
        if observation.prompt == "DECIDE_ACCEPTEES":
            return self.confirm(observation)
        if can_offer(observation):
            offer = self.propose(observation)
            if offer is not None:
                return offer
        return self.inner.decide(observation)

    def _values(
        self, observation: PlayerObservation, gets: Vector, gives: Vector, analysis: BoardAnalysis
    ) -> dict[int, float]:
        """Value of making this trade with each other seat (rounds saved, threat-adjusted)."""
        owner = (id(observation), len(observation.history))
        if self._urgency is None or self._urgency[0] != owner:
            self._urgency = (owner, urgency(threats(observation, analysis), observation.seat))
        weights = self._urgency[1]
        expected = self._tracker.belief.expected() if self._tracker is not None else None
        values = {}
        for seat in range(4):
            if seat == observation.seat:
                continue
            hand = None if expected is None else expected[seat]
            verdict = judge(observation, seat, gets, gives, analysis, other_hand=hand)
            if verdict.enables_win:
                values[seat] = -math.inf
            else:
                values[seat] = verdict.mine - weights[seat] * max(verdict.theirs, 0.0)
        return values

    def acceptance(
        self,
        observation: PlayerObservation,
        give: Vector,
        receive: Vector,
        analysis: BoardAnalysis,
        use_model: bool = False,
    ) -> dict[int, float]:
        """P(each other seat accepts our offer of `give` for `receive`)."""
        assert self._tracker is not None
        belief = self._tracker.belief
        probs = {}
        expected = belief.expected()
        model_yes = self._model_yes(observation, give, receive) if use_model else {}
        for seat in range(4):
            if seat == observation.seat:
                continue
            pay = belief.prob_covers(seat, np.array(receive))
            if seat in model_yes:
                yes = model_yes[seat]
            else:
                theirs = judge(
                    observation, seat, receive, give, analysis, other_hand=expected[seat]
                )
                yes = 1 / (1 + math.exp(-RULE_SLOPE * (theirs.theirs - RULE_OFFSET)))
            probs[seat] = pay * yes
        return probs

    def _model_yes(
        self, observation: PlayerObservation, give: Vector, receive: Vector
    ) -> dict[int, float]:
        """P(yes | can pay) per responder from the opponent model, in sampled worlds."""
        assert self._model is not None
        model = self._model
        action = offer_trade(observation.seat, give, receive)  # type: ignore[arg-type]
        totals: dict[int, list[float]] = {}
        for index, world in enumerate(model.sampler.worlds[:ACCEPTANCE_WORLDS]):
            if world.is_over:
                continue
            branch = world.branch(self.seed + index)
            if not branch.is_legal(action):
                continue
            branch.execute(action)
            while branch.prompt == "DECIDE_TRADE":
                seat = branch.current_seat
                legal = branch.legal_actions()
                accept = next(
                    (i for i, a in enumerate(legal) if a.kind == ActionKind.ACCEPT_TRADE), None
                )
                if accept is not None:
                    weights = np.exp(model.log_policy[seat])
                    p = sum(
                        w * c.distribution(branch, model._observe)[accept]
                        for w, c in zip(weights, model.components, strict=True)
                    )
                    totals.setdefault(seat, []).append(float(p))
                reject = next(a for a in legal if a.kind == ActionKind.REJECT_TRADE)
                branch.execute(reject)
        return {seat: float(np.mean(ps)) for seat, ps in totals.items()}

    def propose(self, observation: PlayerObservation) -> CanonicalAction | None:
        made = offers_this_turn(observation)
        if len(made) >= MAX_OFFERS_PER_TURN:
            return None
        tried = {a.params for a in made}
        analysis = analyze(observation)
        options = [(o.give, o.receive) for o in candidate_offers(observation, analysis)]
        options += [
            (give, receive)
            for _, give, receive in counters_this_turn(observation)
            if is_lawful(give, receive, observation.own_resources)
        ]
        scored = []
        for give, receive in dict.fromkeys(options):
            if (give, receive) in tried:
                continue
            values = self._values(observation, receive, give, analysis)
            accepts = self.acceptance(observation, give, receive, analysis, use_model=False)
            scored.append((expected_value(values, accepts), give, receive, values, accepts))
        scored.sort(key=lambda x: -x[0])
        if self.mode == "model":  # re-score the rules' shortlist with the opponent model
            for i, (_, give, receive, values, _) in enumerate(scored[:MODEL_SHORTLIST]):
                accepts = self.acceptance(observation, give, receive, analysis, use_model=True)
                scored[i] = (expected_value(values, accepts), give, receive, values, accepts)
            scored.sort(key=lambda x: -x[0])
        best = (scored[0][0], scored[0][1], scored[0][2], scored[0][4]) if scored else None
        if best is None or best[0] < OFFER_MIN:
            return None
        bank = maritime_benefits(observation, analysis)
        if bank and bank[0][1] >= best[0]:
            return None  # the bank or a port does at least as well, with certainty
        self._pending.append((len(observation.history), best[3]))
        return offer_trade(observation.seat, best[1], best[2])  # type: ignore[arg-type]

    def respond(self, observation: PlayerObservation) -> CanonicalAction:
        trade = observation.pending_trade
        assert trade is not None
        analysis = analyze(observation)
        legal = {a.kind: a for a in observation.legal_actions}
        value = self._values(observation, trade.give, trade.receive, analysis)[trade.offerer]
        if value > ACCEPT_MIN and ActionKind.ACCEPT_TRADE in legal:
            return legal[ActionKind.ACCEPT_TRADE]
        reject = legal[ActionKind.REJECT_TRADE]
        for offer in candidate_offers(observation, analysis):
            counter_value = self._values(observation, offer.receive, offer.give, analysis)
            if counter_value[trade.offerer] > ACCEPT_MIN:
                return CanonicalAction(reject.kind, reject.seat, (offer.receive, offer.give))
        return reject

    def confirm(self, observation: PlayerObservation) -> CanonicalAction:
        trade = observation.pending_trade
        assert trade is not None
        analysis = analyze(observation)
        values = self._values(observation, trade.receive, trade.give, analysis)
        confirms = [a for a in observation.legal_actions if a.kind == ActionKind.CONFIRM_TRADE]
        cancel = next(a for a in observation.legal_actions if a.kind == ActionKind.CANCEL_TRADE)
        best = max(confirms, key=lambda a: (values[a.params[0]], -a.params[0]), default=None)
        if best is None or values[best.params[0]] <= 0:
            return cancel
        return best
