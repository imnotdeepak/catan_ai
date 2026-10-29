"""Opening settlement+road pair search with snake-draft prediction (STR-018, plan §4.4).

Initial settlements go in snake order: seats 0, 1, 2, 3, 3, 2, 1, 0. Before its
first settlement, a seat can predict the picks other seats will make before its
second, and choose the first settlement whose best follow-up pair is strongest.

- Prediction: each other seat greedily takes the free site with the highest
  score under `predictor` (default: total pips), given its own buildings.
- Pair search: for each of the top `width` first sites, play out the predicted
  draft, take the best second site, and score the pair under `rule`.
- Road: the initial road points to the best site that one more road would
  reach once the predicted draft is over.

Everything uses only public information (`BoardAnalysis`).
"""

from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from catan_ai.arena.agents import Agent
from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.engine.topology import NUM_SEATS
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import NODE_NEIGHBORS, BoardAnalysis, Edge, analyze
from catan_ai.opening.priors import RULES, score

SNAKE: Final = (*range(NUM_SEATS), *reversed(range(NUM_SEATS)))
SECOND_ROAD_WEIGHT: Final = 0.5  # sites two roads away count for less than sites one away


def place(analysis: BoardAnalysis, seat: int, node: int) -> BoardAnalysis:
    """The analysis after `seat` builds a settlement at `node` (no roads or distances)."""
    owner = analysis.owner.copy()
    owner[node] = seat
    buildable = analysis.buildable.copy()
    buildable[[node, *NODE_NEIGHBORS[node]]] = False
    return replace(analysis, owner=owner, buildable=buildable)


def best_site(analysis: BoardAnalysis, seat: int, rule: str) -> int:
    """The free site `seat` scores highest under `rule`; ties go to pips, then lower id."""
    sites = np.flatnonzero(analysis.buildable)
    return int(
        max(
            sites,
            key=lambda n: (score(rule, analysis, seat, int(n)), float(analysis.pips[n]), -n),
        )
    )


def placed(analysis: BoardAnalysis) -> int:
    """How many initial settlements are already on the board."""
    return int(np.count_nonzero(analysis.owner >= 0))


def predict_draft(
    analysis: BoardAnalysis, start: int, stop: int, predictor: str
) -> tuple[BoardAnalysis, list[int]]:
    """Play snake positions [start, stop) with every seat following `predictor`."""
    picks = []
    for position in range(start, stop):
        seat = SNAKE[position]
        node = best_site(analysis, seat, predictor)
        analysis = place(analysis, seat, node)
        picks.append(node)
    return analysis, picks


@dataclass(frozen=True)
class Plan:
    settlement: int
    second: int | None  # the planned second settlement, for a first pick
    final: BoardAnalysis  # the board after the predicted draft


def plan_settlement(
    analysis: BoardAnalysis, seat: int, rule: str, predictor: str = "pips", width: int = 12
) -> Plan:
    position = placed(analysis)
    if SNAKE[position] != seat:
        raise ValueError(f"seat {seat} is not on the clock at draft position {position}")
    sites = np.flatnonzero(analysis.buildable)
    if position >= NUM_SEATS:  # second settlement: nothing of ours follows
        node = best_site(analysis, seat, rule)
        final, _ = predict_draft(place(analysis, seat, node), position + 1, len(SNAKE), predictor)
        return Plan(node, None, final)

    second_at = len(SNAKE) - 1 - seat
    ranked = sorted(
        sites,
        key=lambda n: (score(rule, analysis, seat, int(n)), float(analysis.pips[n])),
        reverse=True,
    )
    best: tuple[tuple[float, float, int], Plan] | None = None
    for node in (int(n) for n in ranked[:width]):
        board, _ = predict_draft(place(analysis, seat, node), position + 1, second_at, predictor)
        second = best_site(board, seat, rule)
        value = score(rule, board, seat, second)  # counts both settlements
        final, _ = predict_draft(place(board, seat, second), second_at + 1, len(SNAKE), predictor)
        key = (value, float(analysis.pips[node] + analysis.pips[second]), -node)
        if best is None or key > best[0]:
            best = (key, Plan(node, second, final))
    assert best is not None
    return best[1]


def road_value(final: BoardAnalysis, seat: int, rule: str, edge: Edge, home: int) -> float:
    """Best site reachable from the far end of `edge` (away from `home`), after the draft."""
    far = edge[1] if edge[0] == home else edge[0]
    one = [n for n in NODE_NEIGHBORS[far] if n != home and final.buildable[n]]
    two = {
        m
        for n in NODE_NEIGHBORS[far]
        if n != home and final.owner[n] < 0
        for m in NODE_NEIGHBORS[n]
        if m != far and final.buildable[m]
    } - set(one)
    values = [score(rule, final, seat, n) for n in one]
    values += [SECOND_ROAD_WEIGHT * score(rule, final, seat, n) for n in two]
    return max(values, default=0.0)


class DraftSearchAgent:
    """Initial settlements (and, with `roads`, initial roads) by pair search.

    Every other decision goes to the wrapped agent.
    """

    def __init__(
        self, rule: str, inner: Agent, roads: bool = True, predictor: str = "pips"
    ) -> None:
        for name in (rule, predictor):
            if name not in RULES:
                raise ValueError(f"Unknown opening rule {name!r}; known: {', '.join(RULES)}")
        self.rule, self.inner, self.roads, self.predictor = rule, inner, roads, predictor
        self._final: BoardAnalysis | None = None

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        if observation.prompt == "BUILD_INITIAL_SETTLEMENT":
            plan = plan_settlement(
                analyze(observation), observation.seat, self.rule, self.predictor
            )
            self._final = plan.final
            return _settlement(observation, plan.settlement)
        if observation.prompt == "BUILD_INITIAL_ROAD" and self.roads and self._final is not None:
            return self._road(observation, self._final)
        return self.inner.decide(observation)

    def _road(self, observation: PlayerObservation, final: BoardAnalysis) -> CanonicalAction:
        roads = [a for a in observation.legal_actions if a.kind == ActionKind.BUILD_ROAD]
        own = {n for n, seat, _ in observation.board.buildings if seat == observation.seat}
        built = {
            n for a, b, seat in observation.board.roads if seat == observation.seat for n in (a, b)
        }
        home = next(n for a in roads for n in a.params[0] if n in own - built)

        def key(action: CanonicalAction) -> tuple[float, int]:
            edge = tuple(sorted(action.params[0]))
            return road_value(final, observation.seat, self.rule, edge, home), -min(edge)

        return max(roads, key=key)


def _settlement(observation: PlayerObservation, node: int) -> CanonicalAction:
    return next(
        a
        for a in observation.legal_actions
        if a.kind == ActionKind.BUILD_SETTLEMENT and a.params[0] == node
    )
