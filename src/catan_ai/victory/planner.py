"""Victory route planner (STR-020, plan §5.1-5.3).

A route to 10 points is a sequence of objectives, each an AND of
prerequisites: a legal location, the pieces, the roads to get there and the
resources. Resources come by an OR of production and maritime trade. The
planner builds a route greedily: it repeatedly takes the objective with the
most points per expected round, given the income the route has built so far.

- Objectives: a city on an own settlement; a settlement on a reachable site
  (with the roads it needs); a development card (expected VP); Largest Army
  and Longest Road (transferable, so their points are marked insecure).
- Time: expected rounds until the hand plus income covers the cost, where
  surpluses can be traded at the seat's best port rates. One round is four
  dice rolls.
- Contingencies: a settlement site an opponent can reach at least as fast is
  contested; each contested site carries the next-best uncontested site as
  its fallback.

It sees only public information plus the viewer's own hand. For opponents,
hidden hands are estimated from their card count and income.
"""

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from catan_ai.engine.actions import RESOURCES
from catan_ai.engine.topology import NUM_SEATS, PORT_KINDS
from catan_ai.observations import DEV_CARDS, PlayerObservation
from catan_ai.opening.board import BoardAnalysis, analyze
from catan_ai.victory.score import (
    ARMY_MIN_KNIGHTS,
    KNIGHT,
    POINTS_TO_WIN,
    ROAD_MIN_LENGTH,
    VP_CARDS_IN_DECK,
    pieces_left,
    points_range,
    seat_scores,
)

ROAD: Final = np.array([1, 1, 0, 0, 0], dtype=np.float32)
SETTLEMENT: Final = np.array([1, 1, 1, 1, 0], dtype=np.float32)
CITY: Final = np.array([0, 0, 0, 2, 3], dtype=np.float32)
DEV_CARD: Final = np.array([0, 0, 1, 1, 1], dtype=np.float32)
ROLLS_PER_ROUND: Final = NUM_SEATS
MAX_ROADS_TO_SITE: Final = 3
MAX_ROUNDS: Final = 60.0  # "not within the horizon"
KNIGHTS_IN_DECK: Final = 14
DECK_SIZE: Final = 25


@dataclass(frozen=True)
class Objective:
    kind: str  # city, settlement, dev, army, road_award
    target: int | None  # node for city/settlement
    points: float  # expected points
    secure: bool  # False for transferable awards and uncertain dev VP
    cost: tuple[float, ...]  # resources, RESOURCES order
    roads: int = 0  # new roads needed (settlement, road award)
    contested: bool = False
    fallback: int | None = None  # next-best uncontested site for a contested settlement


@dataclass(frozen=True)
class Step:
    objective: Objective
    rounds: float  # expected rounds from now until this step completes


@dataclass(frozen=True)
class Plan:
    seat: int
    points: int  # current points (the viewer's exact; others' visible)
    steps: tuple[Step, ...]
    rounds_to_win: float  # MAX_ROUNDS if the route does not reach 10
    contested_targets: tuple[int, ...] = field(default=())

    @property
    def first(self) -> Objective | None:
        return self.steps[0].objective if self.steps else None


def income(analysis: BoardAnalysis, seat: int, extra: tuple[int, ...] = ()) -> np.ndarray:
    """Expected cards per round by resource, from the seat's buildings (robber tile excluded)."""
    weights = np.where(analysis.is_city, 2.0, 1.0) * (analysis.owner == seat)
    yields = analysis.production - analysis.blocked_production
    total = (yields * weights[:, None]).sum(axis=0)
    for node in extra:
        total = total + yields[node]
    return np.asarray(total * ROLLS_PER_ROUND / 36, dtype=np.float32)


def trade_rates(analysis: BoardAnalysis, seat: int, extra: tuple[int, ...] = ()) -> np.ndarray:
    """Cards given per card received, by resource given: 4, 3 with a 3:1 port, 2 with its port."""
    rates = np.full(len(RESOURCES), 4.0, dtype=np.float32)
    nodes = [*np.flatnonzero(analysis.owner == seat), *extra]
    for node in nodes:
        kind = int(analysis.port[node])
        if kind < 0:
            continue
        if PORT_KINDS[kind] is None:
            rates = np.minimum(rates, 3.0)
        else:
            rates[RESOURCES.index(PORT_KINDS[kind])] = 2.0
    return rates


def rounds_to_afford(
    cost: np.ndarray, hand: np.ndarray, rate: np.ndarray, rates: np.ndarray
) -> float:
    """Expected rounds until `hand` plus `rate` per round covers `cost`, trading surpluses."""
    return float(rounds_to_afford_all(cost[None, :], hand, rate, rates)[0])


def rounds_to_afford_all(
    costs: np.ndarray, hand: np.ndarray, rate: np.ndarray, rates: np.ndarray
) -> np.ndarray:
    """`rounds_to_afford` for each row of `costs` [K, resources], by bisection."""

    def affordable(t: np.ndarray) -> np.ndarray:
        have = hand[None, :] + rate[None, :] * t[:, None]
        deficit = np.maximum(costs - have, 0).sum(axis=1)
        tradable = np.floor(np.maximum(have - costs, 0) / rates[None, :]).sum(axis=1)
        return np.asarray(tradable >= deficit)

    k = len(costs)
    now = affordable(np.zeros(k))
    ever = affordable(np.full(k, MAX_ROUNDS))
    low, high = np.zeros(k), np.full(k, MAX_ROUNDS)
    for _ in range(20):
        mid = (low + high) / 2
        ok = affordable(mid)
        high = np.where(ok, mid, high)
        low = np.where(ok, low, mid)
    return np.where(now, 0.0, np.where(ever, high, MAX_ROUNDS))


def pay(cost: np.ndarray, have: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """The hand left after paying `cost`, covering shortfalls by trading surpluses
    (from the resource with the most trades to spare first)."""
    left = have - cost
    deficit = float(np.maximum(-left, 0).sum())
    left = np.maximum(left, 0)
    while deficit > 1e-6:
        spare = np.floor(left / rates)
        if spare.max() < 1:
            break
        r = int(np.argmax(spare))
        units = min(deficit, float(spare[r]))
        left[r] -= units * rates[r]
        deficit -= units
    return np.asarray(left, dtype=np.float32)


def estimated_hand(
    observation: PlayerObservation, seat: int, analysis: BoardAnalysis
) -> np.ndarray:
    """The viewer's own hand, or an opponent's card count spread by its income."""
    if seat == observation.seat:
        return np.array(observation.own_resources, dtype=np.float32)
    count = observation.players[seat].resource_card_count
    share = income(analysis, seat)
    share = share / share.sum() if share.sum() > 0 else np.full(len(RESOURCES), 0.2)
    return np.asarray(count * share, dtype=np.float32)


def objectives(
    observation: PlayerObservation, analysis: BoardAnalysis, seat: int
) -> list[Objective]:
    """Every objective `seat` could pursue now, with its prerequisites checked."""
    left = pieces_left(observation, seat)
    scores = seat_scores(observation)
    me = scores[seat]
    found: list[Objective] = []

    if left["cities"] > 0:
        for city in np.flatnonzero((analysis.owner == seat) & ~analysis.is_city):
            found.append(Objective("city", int(city), 1.0, True, tuple(CITY)))

    if left["settlements"] > 0:
        opponents = [s for s in range(NUM_SEATS) if s != seat]
        distance = analysis.distance[seat]
        sites = [
            int(n)
            for n in np.flatnonzero(analysis.buildable & (distance <= MAX_ROADS_TO_SITE))
            if distance[n] <= left["roads"]
        ]
        rival = analysis.distance[opponents].min(axis=0)
        by_value = sorted(sites, key=lambda n: (-float(analysis.pips[n]), distance[n], n))
        safe = [n for n in by_value if rival[n] > distance[n]]
        for node in by_value:
            contested = bool(rival[node] <= distance[node])
            fallback = next((n for n in safe if n != node), None) if contested else None
            cost = SETTLEMENT + ROAD * float(distance[node])
            found.append(
                Objective(
                    "settlement",
                    node,
                    1.0,
                    True,
                    tuple(cost),
                    roads=int(distance[node]),
                    contested=contested,
                    fallback=fallback,
                )
            )

    if observation.dev_deck_remaining > 0:
        own_vp = observation.own_dev_cards[DEV_CARDS.index("VICTORY_POINT")]
        unseen_vp = VP_CARDS_IN_DECK - (own_vp if seat == observation.seat else 0)
        unseen_cards = observation.dev_deck_remaining + sum(
            p.dev_card_count for p in observation.players if p.seat != observation.seat
        )
        p_vp = min(1.0, unseen_vp / max(unseen_cards, 1))
        found.append(Objective("dev", None, p_vp, False, tuple(DEV_CARD)))

        holder = next((s for s in scores if s.largest_army), None)
        if holder is None or holder.seat != seat:
            target = max(ARMY_MIN_KNIGHTS, (holder.knights_played + 1) if holder else 0)
            in_hand = observation.own_dev_cards[KNIGHT] if seat == observation.seat else 0
            needed = max(target - me.knights_played - in_hand, 0)
            cards = needed / (KNIGHTS_IN_DECK / DECK_SIZE)
            found.append(Objective("army", None, 2.0, False, tuple(DEV_CARD * cards)))

    road_holder = next((s for s in scores if s.longest_road), None)
    if road_holder is None or road_holder.seat != seat:
        target = max(ROAD_MIN_LENGTH, (road_holder.road_length + 1) if road_holder else 0)
        needed = max(target - analysis.longest_road[seat], 0)
        if 0 < needed <= left["roads"]:
            found.append(
                Objective("road_award", None, 2.0, False, tuple(ROAD * needed), roads=needed)
            )
    return found


def plan_for(
    observation: PlayerObservation, seat: int, analysis: BoardAnalysis | None = None
) -> Plan:
    analysis = analysis if analysis is not None else analyze(observation)
    low, high = points_range(observation, seat)
    points = low if seat != observation.seat else high
    hand = estimated_hand(observation, seat, analysis)
    built: list[int] = []
    taken: set[int] = set()
    remaining = objectives(observation, analysis, seat)
    left = pieces_left(observation, seat)
    steps: list[Step] = []
    elapsed, total = 0.0, float(points)

    while total < POINTS_TO_WIN and remaining:
        rate = income(analysis, seat, tuple(built))
        rates = trade_rates(analysis, seat, tuple(built))
        candidates = [o for o in remaining if _available(o, left, taken)]
        if not candidates:
            break
        costs = np.array([o.cost for o in candidates], dtype=np.float32)
        waits = rounds_to_afford_all(costs, hand, rate, rates)
        options = [
            (o.points / max(float(w), 0.5), -float(w), -(o.target or 0), o)
            for o, w in zip(candidates, waits, strict=True)
        ]
        if not options:
            break
        _, neg_wait, _, best = max(options, key=lambda x: x[:3])
        wait = -neg_wait
        if elapsed + wait >= MAX_ROUNDS:
            break
        elapsed += wait
        hand = pay(np.array(best.cost, dtype=np.float32), hand + rate * wait, rates)
        total += best.points
        steps.append(Step(best, elapsed))
        left = _spend(best, left)
        if best.kind != "dev":  # development cards can be bought again
            remaining = [o for o in remaining if o is not best]
        if best.target is not None:
            taken.add(best.target)
            built.append(best.target)  # a settlement adds the node; a city doubles it
            if best.kind == "settlement":
                remaining.append(Objective("city", best.target, 1.0, True, tuple(CITY)))

    rounds = elapsed if total >= POINTS_TO_WIN else MAX_ROUNDS
    contested = tuple(s.objective.target for s in steps if s.objective.contested)
    return Plan(seat, points, tuple(steps), rounds, tuple(t for t in contested if t is not None))


def _available(objective: Objective, left: dict[str, int], taken: set[int]) -> bool:
    if objective.kind == "city":
        return left["cities"] > 0
    if objective.kind == "settlement":
        return (
            left["settlements"] > 0
            and left["roads"] >= objective.roads
            and objective.target not in taken
        )
    if objective.kind == "road_award":
        return left["roads"] >= objective.roads
    return True


def _spend(objective: Objective, left: dict[str, int]) -> dict[str, int]:
    left = dict(left)
    if objective.kind == "city":
        left["cities"] -= 1
        left["settlements"] += 1
    elif objective.kind == "settlement":
        left["settlements"] -= 1
    left["roads"] -= objective.roads
    return left


def plans(
    observation: PlayerObservation, analysis: BoardAnalysis | None = None
) -> tuple[Plan, ...]:
    """A plan for every seat, sharing one board analysis."""
    analysis = analysis if analysis is not None else analyze(observation)
    return tuple(plan_for(observation, seat, analysis) for seat in range(NUM_SEATS))
