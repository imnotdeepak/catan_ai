"""Trade value and candidate proposals (TRD-026, plan §8.2).

A hand is scored by the victory planner: `progress` is the expected rounds
to win from that hand, plus a quarter of the rounds to the route's next
objective (so a trade that brings the next build closer counts even when the
whole route is long). A trade's benefit to a seat is how many rounds it
saves: progress before minus progress after.

Candidate offers come from the proposer's deficits and surpluses for its
best routes: what the next objectives need beyond the hand, and what the
hand holds beyond them. Bundles give up to `MAX_CARDS` surplus cards for up
to `MAX_CARDS` needed cards, never a gift and never the same resource on
both sides (rulebook p.7); they are ranked by the proposer's own benefit.
Maritime trades from the legal actions are scored on the same scale, as the
bank/port fallback.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Final

import numpy as np

from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.engine.actions import RESOURCES
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import BoardAnalysis, analyze
from catan_ai.victory.planner import MAX_ROUNDS, plan_for

MAX_CARDS: Final = 2
SHORTLIST: Final = 6
PREFILTER: Final = 12  # bundles scored by the planner, after a cheap need-coverage ranking
NEXT_STEP_WEIGHT: Final = 0.25
OBJECTIVES_FOR_NEEDS: Final = 3

Vector = tuple[int, ...]


@dataclass(frozen=True)
class Offer:
    give: Vector  # what the proposer gives, RESOURCES order
    receive: Vector  # what the proposer receives
    benefit: float  # rounds the proposer saves


_memo: dict[tuple[int, bytes], float] = {}
_memo_owner: list[tuple[int, int, int]] = [(-1, -1, -1)]


def progress(
    observation: PlayerObservation,
    seat: int,
    hand: np.ndarray,
    analysis: BoardAnalysis | None = None,
) -> float:
    """Expected rounds to win from `hand`, plus a share of the rounds to the next step.

    Memoized per observation: one decision asks about the same hands many times."""
    owner = (id(observation), len(observation.history), observation.seat)
    if _memo_owner[0] != owner:
        _memo.clear()
        _memo_owner[0] = owner
    key = (seat, np.asarray(hand, dtype=np.float32).tobytes())
    if key not in _memo:
        plan = plan_for(observation, seat, analysis, hand=np.asarray(hand, dtype=np.float32))
        first = plan.steps[0].rounds if plan.steps else MAX_ROUNDS
        _memo[key] = min(plan.rounds_to_win, MAX_ROUNDS) + NEXT_STEP_WEIGHT * min(first, MAX_ROUNDS)
    return _memo[key]


def benefit(
    observation: PlayerObservation,
    seat: int,
    hand: np.ndarray,
    change: np.ndarray,
    analysis: BoardAnalysis | None = None,
) -> float:
    """Rounds `seat` saves when `hand` changes by `change`."""
    after = hand + change
    if (after < 0).any():
        return -MAX_ROUNDS
    return progress(observation, seat, hand, analysis) - progress(
        observation, seat, after, analysis
    )


def needs_and_surplus(
    observation: PlayerObservation, analysis: BoardAnalysis
) -> tuple[np.ndarray, np.ndarray]:
    """What the next objectives need beyond the hand, and what the hand holds beyond them."""
    hand = np.array(observation.own_resources, dtype=np.float32)
    plan = plan_for(observation, observation.seat, analysis)
    cost = np.zeros(len(RESOURCES), dtype=np.float32)
    for step in plan.steps[:OBJECTIVES_FOR_NEEDS]:
        cost += np.array(step.objective.cost, dtype=np.float32)
    first = np.array(plan.steps[0].objective.cost, dtype=np.float32) if plan.steps else cost
    needs = np.maximum(np.ceil(cost) - hand, 0)
    surplus = np.maximum(hand - np.ceil(first), 0)
    return needs, surplus


def bundles(counts: np.ndarray, size: int) -> list[Vector]:
    """Resource vectors of 1..size cards drawn from `counts`."""
    found: set[Vector] = set()
    pool = [r for r in range(len(RESOURCES)) if counts[r] > 0]
    for n in range(1, size + 1):
        for combo in combinations_with_replacement(pool, n):
            vector = np.bincount(combo, minlength=len(RESOURCES))
            if (vector <= counts).all():
                found.add(tuple(int(v) for v in vector))
    return sorted(found)


def is_lawful(give: Vector, receive: Vector, hand: Vector) -> bool:
    """No gifts, no like-for-like, and the proposer holds what it gives."""
    if sum(give) == 0 or sum(receive) == 0:
        return False
    if any(g > 0 and r > 0 for g, r in zip(give, receive, strict=True)):
        return False
    return all(g <= h for g, h in zip(give, hand, strict=True))


def candidate_offers(
    observation: PlayerObservation,
    analysis: BoardAnalysis | None = None,
    max_cards: int = MAX_CARDS,
    shortlist: int = SHORTLIST,
) -> list[Offer]:
    """The proposer's best lawful offers by its own benefit, best first."""
    analysis = analysis if analysis is not None else analyze(observation)
    hand = np.array(observation.own_resources, dtype=np.float32)
    needs, surplus = needs_and_surplus(observation, analysis)
    if not needs.any():
        return []
    gives = bundles(surplus if surplus.any() else hand, max_cards)
    receives = bundles(needs, max_cards)
    before = progress(observation, observation.seat, hand, analysis)
    pairs = [
        (give, receive)
        for give in gives
        for receive in receives
        if is_lawful(give, receive, observation.own_resources)
    ]

    def coverage(pair: tuple[Vector, Vector]) -> tuple[float, int]:
        give, receive = pair
        got = float(np.minimum(np.array(receive), needs).sum())
        lost = float(np.minimum(np.array(give), needs).sum())
        return (-(got - lost - 0.5 * sum(give)), sum(give))

    pairs.sort(key=coverage)
    offers = []
    for give, receive in pairs[:PREFILTER]:
        after = hand - np.array(give) + np.array(receive)
        gain = before - progress(observation, observation.seat, after, analysis)
        if gain > 0:
            offers.append(Offer(give, receive, gain))
    offers.sort(key=lambda o: (-o.benefit, sum(o.give), o.give, o.receive))
    return offers[:shortlist]


def maritime_benefits(
    observation: PlayerObservation, analysis: BoardAnalysis | None = None
) -> list[tuple[CanonicalAction, float]]:
    """Each legal bank/port trade with the rounds it saves, best first."""
    analysis = analysis if analysis is not None else analyze(observation)
    hand = np.array(observation.own_resources, dtype=np.float32)
    before = progress(observation, observation.seat, hand, analysis)
    scored = []
    for action in observation.legal_actions:
        if action.kind != ActionKind.MARITIME_TRADE:
            continue
        give, count, receive = action.params
        after = hand.copy()
        after[RESOURCES.index(give)] -= count
        after[RESOURCES.index(receive)] += 1
        scored.append((action, before - progress(observation, observation.seat, after, analysis)))
    scored.sort(key=lambda x: -x[1])
    return scored
