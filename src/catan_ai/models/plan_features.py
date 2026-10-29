"""Victory-plan features (features v4, STR-020), appended to the v3 encoding.

From `catan_ai.victory.planner`, for the viewer and every opponent:

- state: each seat's expected rounds to win and point range (relative seats),
  and the kind of the viewer's next objective;
- actions: whether the action advances the viewer's plan (builds a planned
  city or settlement, a road toward a planned site, a development card for
  a dev or army objective), whether it is the plan's next objective, how soon
  that objective falls due, and, for robber moves, the threat of the victim.
"""

from typing import Final

import numpy as np

from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.engine.topology import NUM_SEATS
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import BoardAnalysis
from catan_ai.victory.planner import Plan, plans
from catan_ai.victory.score import POINTS_TO_WIN, points_range

KINDS: Final = ("city", "settlement", "dev", "army", "road_award")
HORIZON: Final = 30.0  # rounds; longer counts as "not soon"
SEAT_EXTRA: Final = 3  # rounds to win, least points, most points
PLAN_STATE_DIM: Final = NUM_SEATS * SEAT_EXTRA + len(KINDS) + 1  # + "no objective"
PLAN_ACTION_DIM: Final = 4  # advances plan, next objective, due soon, robber victim threat


def _soon(rounds: float) -> float:
    return 1.0 - min(rounds, HORIZON) / HORIZON


def plan_state(
    observation: PlayerObservation, analysis: BoardAnalysis
) -> tuple[np.ndarray, tuple[Plan, ...]]:
    all_plans = plans(observation, analysis)
    seats = np.zeros((NUM_SEATS, SEAT_EXTRA), dtype=np.float32)
    for r in range(NUM_SEATS):
        seat = (observation.seat + r) % NUM_SEATS
        low, high = points_range(observation, seat)
        seats[r] = (_soon(all_plans[seat].rounds_to_win), low / POINTS_TO_WIN, high / POINTS_TO_WIN)
    kind = np.zeros(len(KINDS) + 1, dtype=np.float32)
    first = all_plans[observation.seat].first
    kind[KINDS.index(first.kind) if first is not None else len(KINDS)] = 1
    return np.concatenate([seats.ravel(), kind]), all_plans


def plan_actions(
    observation: PlayerObservation,
    analysis: BoardAnalysis,
    actions: tuple[CanonicalAction, ...],
    all_plans: tuple[Plan, ...],
) -> np.ndarray:
    out = np.zeros((len(actions), PLAN_ACTION_DIM), dtype=np.float32)
    viewer = observation.seat
    plan = all_plans[viewer]
    steps = list(plan.steps)
    sites = [
        s for s in steps if s.objective.kind == "settlement" and s.objective.target is not None
    ]
    for row, action in zip(out, actions, strict=True):
        match = None
        if action.kind in (ActionKind.BUILD_CITY, ActionKind.BUILD_SETTLEMENT):
            kind = "city" if action.kind == ActionKind.BUILD_CITY else "settlement"
            match = next(
                (
                    s
                    for s in steps
                    if s.objective.kind == kind and s.objective.target == action.params[0]
                ),
                None,
            )
        elif action.kind == ActionKind.BUILD_ROAD and sites:
            after = analysis.distance_after_road(viewer, action.params[0])
            match = next(
                (
                    s
                    for s in sites
                    if after[s.objective.target] < analysis.distance[viewer][s.objective.target]
                ),
                None,
            )
        elif action.kind in (ActionKind.BUY_DEV_CARD, ActionKind.PLAY_KNIGHT):
            wanted = ("dev", "army") if action.kind == ActionKind.BUY_DEV_CARD else ("army",)
            match = next((s for s in steps if s.objective.kind in wanted), None)
        elif action.kind == ActionKind.MOVE_ROBBER and action.params[1] is not None:
            row[3] = _soon(all_plans[action.params[1]].rounds_to_win)
        if match is not None:
            row[0] = 1.0
            row[1] = float(match is steps[0])
            row[2] = _soon(match.rounds)
    return out
