"""Incremental belief tracking with a hypothesis budget (BEL-024, plan §6).

`BeliefTracker` follows one seat's observations. It applies only events it
has not seen, and rebuilds from the start of the history when the history is
not the one it has been following (a new game, or a replay).

Tiers, reported per update:
1. exact: one joint hand;
2. enumerated: every hypothesis kept, the posterior is exact;
3. particles: the hypothesis count exceeded the budget, so it was resampled
   to `budget` particles (systematic resampling, unbiased). The posterior is
   now approximate.

Collapse: in tier 3 the true hand can be resampled away, and a later public
delta then eliminates every particle. The tracker detects this (BeliefError)
and recovers by replaying the whole history with four times the budget and a
fresh resampling seed, up to MAX_RECOVERIES times. Tiers 1 and 2 are exact, so
a BeliefError there is a real conservation failure and is raised.
"""

from collections import Counter
from typing import Final

import numpy as np

from catan_ai.beliefs.resources import (
    BeliefError,
    ResourceBelief,
    apply_event,
    check,
    initial_belief,
)
from catan_ai.engine import ActionKind
from catan_ai.observations import PlayerObservation
from catan_ai.observations.events import VisibleEvent

DEFAULT_BUDGET: Final = 4096
# Events after which no offer is on the table.
TRADE_ENDS: Final = {ActionKind.CONFIRM_TRADE, ActionKind.CANCEL_TRADE, ActionKind.END_TURN}
MAX_RECOVERIES: Final = 3
EXACT, ENUMERATED, PARTICLES = 1, 2, 3


def resample(belief: ResourceBelief, budget: int, rng: np.random.Generator) -> ResourceBelief:
    """Systematic resampling to at most `budget` distinct particles, weighted by multiplicity."""
    cumulative = np.cumsum(belief.weights)
    cumulative[-1] = 1.0
    points = (rng.random() + np.arange(budget)) / budget
    picks = np.searchsorted(cumulative, points)
    counts = np.bincount(picks, minlength=belief.size)
    kept = np.flatnonzero(counts)
    return ResourceBelief(belief.hands[kept], counts[kept] / budget, belief.steals[kept])


class BeliefTracker:
    def __init__(self, seat: int, budget: int = DEFAULT_BUDGET, seed: int = 0) -> None:
        self.seat = seat
        self.budget = budget
        self.seed = seed
        self.recoveries = 0
        self.tier_counts: Counter[int] = Counter()
        self._reset(budget)

    def _reset(self, budget: int) -> None:
        self._belief = initial_belief()
        self._seen = 0
        self._last: VisibleEvent | None = None
        self._approximate = False
        self._active_budget = budget
        self._trade: tuple[int, ...] | None = None  # receive vector of the offer on the table

    @property
    def belief(self) -> ResourceBelief:
        return self._belief

    @property
    def tier(self) -> int:
        """Once resampled, approximate for the rest of the game: a lone surviving
        particle is consistent with the record but not proven true."""
        if self._approximate:
            return PARTICLES
        return EXACT if self._belief.exact else ENUMERATED

    def update(self, observation: PlayerObservation) -> ResourceBelief:
        if observation.seat != self.seat:
            raise ValueError(f"tracker for seat {self.seat} got seat {observation.seat}")
        history = observation.history
        if self._seen > len(history) or (self._seen and history[self._seen - 1] is not self._last):
            self._reset(self.budget)  # not the history this tracker has been following
        try:
            self._advance(history)
        except BeliefError:
            if not self._approximate:
                raise
            self._recover(history)
        check(self._belief, observation)
        self.tier_counts[self.tier] += 1
        return self._belief

    def _advance(self, history: tuple[VisibleEvent, ...]) -> None:
        for event in history[self._seen :]:
            self._belief = apply_event(self._belief, event, self._trade)
            if event.event_type == ActionKind.OFFER_TRADE:
                self._trade = tuple(event.public["action"]["params"][1])
            elif event.event_type in TRADE_ENDS:
                self._trade = None
            if self._belief.size > self._active_budget:
                rng = np.random.default_rng((self.seed, self.recoveries, event.sequence_id))
                self._belief = resample(self._belief, self._active_budget, rng)
                self._approximate = True
            self._seen += 1
            self._last = event

    def _recover(self, history: tuple[VisibleEvent, ...]) -> None:
        budget = self._active_budget
        while self.recoveries < MAX_RECOVERIES:
            self.recoveries += 1
            budget *= 4
            self._reset(budget)
            try:
                self._advance(history)
                return
            except BeliefError:
                continue
        raise BeliefError(f"particle collapse persisted after {MAX_RECOVERIES} recoveries")
