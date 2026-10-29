"""Five negotiator baselines (plan §8.5) and the shared trade judgement.

Built-in Catanatron bots never propose and (ValueFunction, AlphaBeta) never
accept, so trading is measured against these agents, which do both. Every
non-trade decision goes to the wrapped agent; they differ only in how they
trade:

- cooperative: accepts at a loss of up to 0.25 rounds; proposes fair offers
  (gives at least as many cards), up to 2 a turn; no counteroffers.
- self-interested: accepts savings above 0.25 rounds; proposes its best
  offers, 1 a turn; counteroffers.
- risk-averse: accepts savings above 0.5 rounds, only from a proposer that is
  not the leader and gains less; proposes its best offer if it saves a round.
- opportunistic: accepts any saving, even from the leader; proposes greedy
  asks first (more cards in than out), up to 2 a turn; counteroffers.
- trade-resistant: accepts savings above 3 rounds; never proposes.

Every style refuses a trade after which the other side could win at once
(on its turn, with the cards), and, as a proposer, confirms the acceptor that
threatens it least. Opponents' hands are
estimated from their card counts and income (the planner's estimate).

Counteroffers ride on REJECT_TRADE as params (the proposer's give and receive
vectors). The engine ignores them; they are public in the event log, and the
proposer may issue one as an ordinary offer (trade audit, D-08).
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from catan_ai.arena.agents import Agent
from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.engine.actions import offer_trade
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import BoardAnalysis, analyze
from catan_ai.trading.proposals import Offer, Vector, benefit, candidate_offers, is_lawful
from catan_ai.victory.planner import estimated_hand, plan_for
from catan_ai.victory.score import points_range
from catan_ai.victory.threats import first_winner, threats

MAX_OFFERS_PER_TURN: Final = 2


@dataclass(frozen=True)
class Style:
    accept_above: float
    offers_per_turn: int
    propose_above: float = 0.0
    fair_only: bool = False
    greedy: bool = False
    counters: bool = False
    avoid_leader: bool = False


STYLES: Final = {
    "cooperative": Style(-0.25, 2, fair_only=True),
    "self-interested": Style(0.25, 1, counters=True),
    "risk-averse": Style(0.5, 1, propose_above=1.0, avoid_leader=True),
    "opportunistic": Style(0.0, 2, greedy=True, counters=True),
    "trade-resistant": Style(3.0, 0),
}


def hand_of(observation: PlayerObservation, seat: int, analysis: BoardAnalysis) -> np.ndarray:
    if seat == observation.seat:
        return np.array(observation.own_resources, dtype=np.float32)
    return estimated_hand(observation, seat, analysis)


def wins_now(
    observation: PlayerObservation, seat: int, hand: np.ndarray, analysis: BoardAnalysis
) -> bool:
    """Whether `seat`, on its turn with `hand`, could reach 10 at once (worst case for
    its hidden VP cards)."""
    _, high = points_range(observation, seat)
    return plan_for(observation, seat, analysis, points=high, hand=hand).rounds_to_win == 0


@dataclass(frozen=True)
class Judgement:
    mine: float  # rounds the viewer saves
    theirs: float  # rounds the other side saves (estimated)
    enables_win: bool  # the other side could win at once with its hand afterwards


def judge(
    observation: PlayerObservation,
    other: int,
    viewer_gets: Vector,
    viewer_gives: Vector,
    analysis: BoardAnalysis | None = None,
    other_hand: np.ndarray | None = None,
) -> Judgement:
    """A trade between the viewer and `other`, from the viewer's view."""
    analysis = analysis if analysis is not None else analyze(observation)
    change = np.array(viewer_gets, dtype=np.float32) - np.array(viewer_gives, dtype=np.float32)
    mine = benefit(observation, observation.seat, hand_of(observation, observation.seat, analysis),
                   change, analysis)  # fmt: skip
    theirs_hand = other_hand if other_hand is not None else hand_of(observation, other, analysis)
    theirs_hand = np.maximum(theirs_hand, 0)
    theirs = benefit(observation, other, theirs_hand, -change, analysis)
    enables = wins_now(observation, other, np.maximum(theirs_hand - change, 0), analysis)
    return Judgement(mine, theirs, enables)


def leader(observation: PlayerObservation, analysis: BoardAnalysis) -> int | None:
    return first_winner(threats(observation, analysis), worst=True)


def offers_this_turn(observation: PlayerObservation) -> list[CanonicalAction]:
    return [
        CanonicalAction(
            ActionKind.OFFER_TRADE, observation.seat, tuple(e.public["action"]["params"])
        )
        for e in observation.history
        if e.event_type == ActionKind.OFFER_TRADE
        and e.actor == observation.seat
        and e.turn_index == observation.num_turns
    ]


def counters_this_turn(observation: PlayerObservation) -> list[tuple[int, Vector, Vector]]:
    """(responder, proposer give, proposer receive) counteroffers made to the viewer this turn."""
    found = []
    for e in observation.history:
        params = e.public.get("action", {}).get("params") if e.actor is not None else None
        if (
            e.event_type == ActionKind.REJECT_TRADE
            and e.turn_index == observation.num_turns
            and params
            and len(params) == 2
        ):
            found.append((e.actor, tuple(params[0]), tuple(params[1])))
    return found  # type: ignore[return-value]


def can_offer(observation: PlayerObservation) -> bool:
    kinds = {a.kind for a in observation.legal_actions}
    return observation.prompt == "PLAY_TURN" and ActionKind.END_TURN in kinds


class NegotiatorAgent:
    def __init__(self, style: str, inner: Agent) -> None:
        if style not in STYLES:
            raise ValueError(f"Unknown negotiator style {style!r}; known: {', '.join(STYLES)}")
        self.name = style
        self.style = STYLES[style]
        self.inner = inner

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        if observation.prompt == "DECIDE_TRADE":
            return self.respond(observation)
        if observation.prompt == "DECIDE_ACCEPTEES":
            return self.confirm(observation)
        if can_offer(observation):
            offer = self.propose(observation)
            if offer is not None:
                return offer
        return self.inner.decide(observation)

    # ----- Responding

    def respond(self, observation: PlayerObservation) -> CanonicalAction:
        trade = observation.pending_trade
        assert trade is not None
        analysis = analyze(observation)
        verdict = judge(observation, trade.offerer, trade.give, trade.receive, analysis)
        accept = self.accepts(observation, trade.offerer, verdict, analysis)
        legal = {a.kind: a for a in observation.legal_actions}
        if accept and ActionKind.ACCEPT_TRADE in legal:
            return legal[ActionKind.ACCEPT_TRADE]
        reject = legal[ActionKind.REJECT_TRADE]
        if self.style.counters:
            counter = self.counter(observation, trade.offerer, analysis)
            if counter is not None:
                return CanonicalAction(reject.kind, reject.seat, counter)
        return reject

    def accepts(
        self,
        observation: PlayerObservation,
        other: int,
        verdict: Judgement,
        analysis: BoardAnalysis,
    ) -> bool:
        if verdict.enables_win or verdict.mine <= self.style.accept_above:
            return False
        if self.style.avoid_leader:
            return leader(observation, analysis) != other and verdict.theirs < verdict.mine
        return True

    def counter(
        self, observation: PlayerObservation, proposer: int, analysis: BoardAnalysis
    ) -> tuple[Vector, Vector] | None:
        """The responder's best own offer the proposer also gains from, as the proposer's
        (give, receive)."""
        for offer in candidate_offers(observation, analysis):
            verdict = judge(observation, proposer, offer.receive, offer.give, analysis)
            if verdict.theirs > 0 and not verdict.enables_win:
                return offer.receive, offer.give
        return None

    # ----- Proposing

    def propose(self, observation: PlayerObservation) -> CanonicalAction | None:
        made = offers_this_turn(observation)
        if len(made) >= min(self.style.offers_per_turn, MAX_OFFERS_PER_TURN):
            return None
        analysis = analyze(observation)
        tried = {a.params for a in made}
        options: list[tuple[Vector, Vector, float]] = [
            (give, receive, 0.5)
            for _, give, receive in counters_this_turn(observation)
            if is_lawful(give, receive, observation.own_resources)
        ]
        for offer in self._ranked(candidate_offers(observation, analysis)):
            options.append((offer.give, offer.receive, offer.benefit))
        for give, receive, gain in options:
            if (give, receive) in tried or gain <= self.style.propose_above:
                continue
            action = offer_trade(observation.seat, give, receive)  # type: ignore[arg-type]
            return action
        return None

    def _ranked(self, offers: list[Offer]) -> list[Offer]:
        if self.style.fair_only:
            return [o for o in offers if sum(o.give) >= sum(o.receive)]
        if self.style.greedy:
            return sorted(offers, key=lambda o: (sum(o.give) - sum(o.receive), -o.benefit))
        return offers

    # ----- Confirming

    def confirm(self, observation: PlayerObservation) -> CanonicalAction:
        analysis = analyze(observation)
        race = threats(observation, analysis)
        confirms = [a for a in observation.legal_actions if a.kind == ActionKind.CONFIRM_TRADE]
        cancel = next(a for a in observation.legal_actions if a.kind == ActionKind.CANCEL_TRADE)
        trade = observation.pending_trade
        assert trade is not None
        safe = [
            a
            for a in confirms
            if not judge(observation, a.params[0], trade.receive, trade.give, analysis).enables_win
        ]
        if not safe:
            return cancel
        return max(safe, key=lambda a: (race[a.params[0]].worst_win_turn, -a.params[0]))
