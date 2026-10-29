"""Opponent hands from the public event log (BEL-022, BEL-023, plan §6).

Every resource movement in base Catan is public except one: a robber steal
between two other players. Payouts, builds, purchases, bank and domestic
trades, discards, Monopoly and Year of Plenty all carry a public
`resource_delta`. So a seat's belief is:

- Tier 1, exact ledger: a single joint hand for all four seats, while every
  steal the viewer did not take part in has been resolved.
- Tier 2, enumerated hypotheses: each unseen steal branches every hypothesis
  by the stolen resource, weighted by the victim's share of that resource
  (the card is drawn at random). Later public deltas prune hypotheses that
  would make a hand negative, and identical joint hands are merged.

Conservation is asserted at every step: a public delta that no hypothesis
can absorb, or a hypothesis whose card counts or own hand disagree with the
observation, raises `BeliefError`.

Tier 3, the budgeted fallback (BEL-024), lives in `catan_ai.beliefs.particles`.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from catan_ai.engine.actions import RESOURCES, ActionKind
from catan_ai.observations import PlayerObservation
from catan_ai.observations.events import NUM_SEATS, VisibleEvent

NUM_RESOURCES: Final = len(RESOURCES)


class BeliefError(AssertionError):
    """The public record contradicts every hypothesis (a conservation failure)."""


@dataclass(frozen=True)
class ResourceBelief:
    """A weighted set of joint hands, hands[k, seat, resource], weights summing to 1."""

    hands: np.ndarray  # [K, seats, resources] int
    weights: np.ndarray  # [K] float64

    @property
    def size(self) -> int:
        return len(self.weights)

    @property
    def exact(self) -> bool:
        return self.size == 1

    def expected(self) -> np.ndarray:
        """Expected hand per seat and resource [seats, resources]."""
        return np.asarray(np.tensordot(self.weights, self.hands, axes=1), dtype=np.float64)

    def marginal(self, seat: int, resource: int) -> np.ndarray:
        """P(seat holds exactly n of `resource`), n = 0..max."""
        counts = self.hands[:, seat, resource]
        return np.bincount(counts, weights=self.weights)

    def prob_covers(self, seat: int, cost: np.ndarray) -> float:
        """P(seat's hand covers `cost` without trading)."""
        ok = (self.hands[:, seat, :] >= np.asarray(cost)[None, :]).all(axis=1)
        return float(self.weights[ok].sum())

    def prob_of(self, seat: int, hand: tuple[int, ...]) -> float:
        """P(seat's hand is exactly `hand`)."""
        match = (self.hands[:, seat, :] == np.asarray(hand)[None, :]).all(axis=1)
        return float(self.weights[match].sum())


def initial_belief() -> ResourceBelief:
    return ResourceBelief(
        np.zeros((1, NUM_SEATS, NUM_RESOURCES), dtype=np.int16), np.ones(1, dtype=np.float64)
    )


def apply_delta(belief: ResourceBelief, delta: np.ndarray) -> ResourceBelief:
    """A public change to every seat's hand [seats, resources]."""
    hands = belief.hands + delta[None, :, :].astype(np.int16)
    keep = (hands >= 0).all(axis=(1, 2))
    if not keep.any():
        raise BeliefError("a public resource change is impossible under every hypothesis")
    return _normalized(hands[keep], belief.weights[keep])


def apply_steal(
    belief: ResourceBelief, thief: int, victim: int, resource: int | None
) -> ResourceBelief:
    """One card moves from victim to thief; `resource` is None if the viewer did not see it."""
    if resource is not None:
        delta = np.zeros((NUM_SEATS, NUM_RESOURCES), dtype=np.int16)
        delta[victim, resource] -= 1
        delta[thief, resource] += 1
        return apply_delta(belief, delta)
    held = belief.hands[:, victim, :].astype(np.float64)  # [K, resources]
    total = held.sum(axis=1)
    if (total <= 0).any():
        # The victim had cards (a steal happened), so hypotheses where it had none are out.
        keep = total > 0
        if not keep.any():
            raise BeliefError("a steal from a seat that holds no cards under every hypothesis")
        belief = _normalized(belief.hands[keep], belief.weights[keep])
        held, total = held[keep], total[keep]
    branches, weights = [], []
    for r in range(NUM_RESOURCES):
        possible = held[:, r] > 0
        if not possible.any():
            continue
        hands = belief.hands[possible].copy()
        hands[:, victim, r] -= 1
        hands[:, thief, r] += 1
        branches.append(hands)
        weights.append(belief.weights[possible] * held[possible, r] / total[possible])
    return _merged(np.concatenate(branches), np.concatenate(weights))


def apply_event(belief: ResourceBelief, event: VisibleEvent) -> ResourceBelief:
    if "resource_delta" in event.public:
        delta = np.asarray(event.public["resource_delta"], dtype=np.int16)
        return belief if not delta.any() else apply_delta(belief, delta)
    if event.event_type == ActionKind.MOVE_ROBBER:
        params = event.public["action"]["params"]
        victim = params[1]
        stolen = None if event.private is None else event.private.get("stolen_resource")
        resource = None if stolen is None else RESOURCES.index(stolen)
        return apply_steal(belief, event.actor, victim, resource)  # type: ignore[arg-type]
    return belief


def check(belief: ResourceBelief, observation: PlayerObservation) -> None:
    """Conservation: every hypothesis matches public card counts and the viewer's own hand."""
    counts = belief.hands.sum(axis=2)
    public = np.array([p.resource_card_count for p in observation.players])
    if not (counts == public[None, :]).all():
        raise BeliefError("a hypothesis disagrees with public card counts")
    own = np.asarray(observation.own_resources)
    if not (belief.hands[:, observation.seat, :] == own[None, :]).all():
        raise BeliefError("a hypothesis disagrees with the viewer's own hand")


def _normalized(hands: np.ndarray, weights: np.ndarray) -> ResourceBelief:
    return ResourceBelief(hands, weights / weights.sum())


def _merged(hands: np.ndarray, weights: np.ndarray) -> ResourceBelief:
    flat = hands.reshape(len(hands), -1)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    summed = np.bincount(inverse.reshape(-1), weights=weights, minlength=len(unique))
    return _normalized(unique.reshape(-1, NUM_SEATS, NUM_RESOURCES), summed)
