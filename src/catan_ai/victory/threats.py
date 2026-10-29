"""Win threats in turn order, and defensive robber placement (STR-021, plan §5.1, §5.4).

- Turn order: a seat wins only on its own turn, so a route that completes after
  `r` rounds wins at that seat's first turn after it. With equal routes, the
  seat that acts sooner after the current turn wins the race.
- Hidden VP cards: an opponent's likely route starts from its visible points;
  its worst case counts every face-down card it could hold as a VP card.
- Defense: without domestic trades (Release 0.4), the robber is the main
  lever. A robber move is scored by the rounds it adds to each opponent's
  route, weighted by how soon that opponent threatens to win, minus the
  rounds it adds to the viewer's own route. Stealing delays the victim by
  about one card's worth of its income.
"""

import math
from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from catan_ai.arena.agents import Agent
from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.engine.actions import RESOURCES
from catan_ai.engine.topology import NUM_SEATS, TILE_NODES, pips
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import BoardAnalysis, analyze
from catan_ai.victory.planner import MAX_ROUNDS, income, plan_for
from catan_ai.victory.score import points_range

DELAY_CAP: Final = 20.0  # rounds; a blocked route is replanned long before "never"
URGENCY_HALF_LIFE: Final = 2.0  # rounds later a threat finishes, half the weight


@dataclass(frozen=True)
class Threat:
    seat: int
    points: tuple[int, int]  # (least, most) given what the viewer knows
    rounds_to_win: float  # likely route, from visible points
    worst_rounds_to_win: float  # every possible hidden VP card counted
    turns_until: int  # turns before the seat next acts on its own turn; 0 if now

    @property
    def win_turn(self) -> float:
        return finish_turn(self.turns_until, self.rounds_to_win)

    @property
    def worst_win_turn(self) -> float:
        return finish_turn(self.turns_until, self.worst_rounds_to_win)


def turns_until(observation: PlayerObservation, seat: int) -> int:
    """Turns before `seat`'s own turn: 0 for the seat whose turn it is."""
    return (seat - observation.turn_seat) % NUM_SEATS


def finish_turn(turns: int, rounds: float) -> float:
    """Turns from now until a route of `rounds` rounds wins: it can only win on its own turn."""
    if rounds >= MAX_ROUNDS:
        return math.inf
    return float(turns + NUM_SEATS * math.ceil(rounds - 1e-6))


def threats(
    observation: PlayerObservation, analysis: BoardAnalysis | None = None
) -> tuple[Threat, ...]:
    """Every seat's race to 10, the viewer's included."""
    analysis = analysis if analysis is not None else analyze(observation)
    found = []
    for seat in range(NUM_SEATS):
        low, high = points_range(observation, seat)
        likely = plan_for(observation, seat, analysis, low).rounds_to_win
        worst = likely if high == low else plan_for(observation, seat, analysis, high).rounds_to_win
        found.append(Threat(seat, (low, high), likely, worst, turns_until(observation, seat)))
    return tuple(found)


def first_winner(race: tuple[Threat, ...], worst: bool = False) -> int | None:
    """The seat whose route wins first, if any route finishes in the horizon."""
    turns = [t.worst_win_turn if worst else t.win_turn for t in race]
    best = min(range(len(race)), key=lambda i: turns[i])
    return race[best].seat if math.isfinite(turns[best]) else None


def urgency(race: tuple[Threat, ...], viewer: int) -> np.ndarray:
    """Weight per seat for delaying it: 1 for the opponent that threatens to win
    first (worst case), halving every URGENCY_HALF_LIFE rounds later; 0 for the viewer."""
    turns = np.array([t.worst_win_turn for t in race])
    weights = np.zeros(NUM_SEATS)
    opponents = [t.seat for t in race if t.seat != viewer and math.isfinite(turns[t.seat])]
    if not opponents:
        return weights
    soonest = min(turns[s] for s in opponents)
    for s in opponents:
        weights[s] = 0.5 ** ((turns[s] - soonest) / (NUM_SEATS * URGENCY_HALF_LIFE))
    return weights


def with_robber(
    observation: PlayerObservation, analysis: BoardAnalysis, coordinate: tuple[int, ...]
) -> BoardAnalysis:
    """The analysis with the robber moved to `coordinate`."""
    blocked = np.zeros_like(analysis.production)
    for coord, resource, number in observation.board.tiles:
        if tuple(coord) == tuple(coordinate) and resource is not None and number is not None:
            for node in TILE_NODES[tuple(coord)]:
                blocked[node, RESOURCES.index(resource)] += pips(number)
    return replace(analysis, blocked_production=blocked)


def steal_delay(analysis: BoardAnalysis, seat: int) -> float:
    """Rounds of `seat`'s income one stolen card is worth, at most one round."""
    per_round = float(income(analysis, seat).sum())
    return min(1.0, 1.0 / per_round) if per_round > 0 else 1.0


def robber_scores(
    observation: PlayerObservation,
    actions: list[CanonicalAction],
    analysis: BoardAnalysis | None = None,
) -> list[float]:
    """Defensive value of each MOVE_ROBBER action, in urgency-weighted rounds of delay."""
    analysis = analysis if analysis is not None else analyze(observation)
    race = threats(observation, analysis)
    weights = urgency(race, observation.seat)
    weights[observation.seat] = -1.0  # delaying ourselves counts against
    now = np.array([min(t.rounds_to_win, DELAY_CAP) for t in race])
    counted = [s for s in range(NUM_SEATS) if weights[s] != 0]
    delays: dict[tuple[int, ...], float] = {}
    scores = []
    for action in actions:
        if action.kind != ActionKind.MOVE_ROBBER:
            raise ValueError(f"not a robber move: {action}")
        coordinate, victim = tuple(action.params[0]), action.params[1]
        if coordinate not in delays:
            delays[coordinate] = 0.0
            owners = {int(o) for o in analysis.owner[list(TILE_NODES[coordinate])]}
            touched = [s for s in counted if s in owners]
            if touched:  # a tile without buildings blocks no income
                moved = with_robber(observation, analysis, coordinate)
                for s in touched:
                    after = plan_for(observation, s, moved, race[s].points[0]).rounds_to_win
                    delays[coordinate] += weights[s] * (min(after, DELAY_CAP) - now[s])
        score = delays[coordinate]
        if victim is not None:
            score += weights[victim] * steal_delay(analysis, victim)
        scores.append(score)
    return scores


class DefendAgent:
    """Moves the robber by `robber_scores`; every other decision by `inner`."""

    def __init__(self, inner: Agent) -> None:
        self.inner = inner

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        if observation.prompt != "MOVE_ROBBER":
            return self.inner.decide(observation)
        actions = [a for a in observation.legal_actions if a.kind == ActionKind.MOVE_ROBBER]
        scores = robber_scores(observation, actions)
        best = max(range(len(actions)), key=lambda i: (scores[i], -i))
        return actions[best]
