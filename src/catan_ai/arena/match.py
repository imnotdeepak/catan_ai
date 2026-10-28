"""One four-player match, its result record, and exact replay (ENG-008)."""

import contextlib
import io
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from catan_ai.arena.contestants import Contestant
from catan_ai.arena.serialize import (
    action_from_json,
    action_to_json,
    from_json_value,
    to_json_value,
)
from catan_ai.engine import SEAT_COLORS, CanonicalAction, IllegalActionError
from catan_ai.observations import Referee
from catan_ai.provenance import provenance

REPLAY_SCHEMA_VERSION = 1
MAX_DECISIONS = 30_000  # safety cap; the engine also truncates at 1000 turns
_BOT_CHATTER = io.StringIO()

# Called with (seat, referee, chosen action) before each action is executed.
DecisionObserver = Callable[[int, Referee, CanonicalAction], None]


@dataclass(frozen=True)
class MatchResult:
    match_id: str
    seed: int
    seating: tuple[str, ...]  # contestant name by seat
    privileged_seats: tuple[bool, ...]
    winner_seat: int | None
    winner: str | None
    truncated: bool
    truncation_reason: str | None
    error: str | None
    num_turns: int
    decisions: tuple[int, ...]  # by seat
    decision_seconds: tuple[float, ...]  # by seat, including observation building
    victory_points: tuple[int, ...]  # actual, including hidden VP cards
    visible_victory_points: tuple[int, ...]
    wall_seconds: float
    engine_commit: str
    catan_ai_commit: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def play_match(
    match_id: str,
    seed: int,
    seating: Sequence[Contestant],
    max_decisions: int = MAX_DECISIONS,
    observer: DecisionObserver | None = None,
) -> tuple[MatchResult, Referee]:
    """Play one game. Our agents see only their observations; engine bots see the game.

    `observer` (e.g. a dataset recorder) is not counted in decision time.
    """
    started = time.perf_counter()
    referee = Referee.new_game(
        seed, [c.engine_player(SEAT_COLORS[s]) for s, c in enumerate(seating)]
    )
    adapter = referee.adapter
    agents = {
        seat: c.make_agent(f"{seed}:{seat}:{c.name}")
        for seat, c in enumerate(seating)
        if not c.is_engine_bot
    }
    decisions, seconds = [0] * 4, [0.0] * 4
    error = None

    while not adapter.is_over and sum(decisions) < max_decisions:
        seat = adapter.current_seat
        tick = time.perf_counter()
        if seat in agents:
            action = agents[seat].decide(referee.observe(seat))
        else:
            with contextlib.redirect_stdout(_BOT_CHATTER):  # some bots print per decision
                action = adapter.engine_player_decision()
            _BOT_CHATTER.seek(0)
            _BOT_CHATTER.truncate()
        seconds[seat] += time.perf_counter() - tick
        decisions[seat] += 1
        if observer is not None:
            observer(seat, referee, action)
        try:
            referee.execute(action)
        except IllegalActionError as exc:
            error = f"seat {seat} ({seating[seat].name}): {exc}"
            break

    state = adapter.referee_game.state
    winner_seat = adapter.winner_seat
    reason = None
    if winner_seat is None and error is None:
        reason = "engine_turn_limit" if adapter.is_truncated else "decision_cap"
    info = provenance()
    result = MatchResult(
        match_id=match_id,
        seed=seed,
        seating=tuple(c.name for c in seating),
        privileged_seats=tuple(c.privileged for c in seating),
        winner_seat=winner_seat,
        winner=None if winner_seat is None else seating[winner_seat].name,
        truncated=reason is not None,
        truncation_reason=reason,
        error=error,
        num_turns=adapter.num_turns,
        decisions=tuple(decisions),
        decision_seconds=tuple(round(s, 6) for s in seconds),
        victory_points=tuple(
            int(state.player_state[f"P{s}_ACTUAL_VICTORY_POINTS"]) for s in range(4)
        ),
        visible_victory_points=tuple(
            int(state.player_state[f"P{s}_VICTORY_POINTS"]) for s in range(4)
        ),
        wall_seconds=round(time.perf_counter() - started, 6),
        engine_commit=str(info["engine_commit"]),
        catan_ai_commit=str(info["catan_ai_commit"]),
    )
    return result, referee


def replay_record(result: MatchResult, referee: Referee) -> dict[str, Any]:
    """Everything needed to rebuild the match exactly, independent of the players."""
    actions, outcomes = zip(*referee.action_log, strict=True) if referee.action_log else ((), ())
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "match_id": result.match_id,
        "seed": result.seed,
        "seating": list(result.seating),
        "engine_commit": result.engine_commit,
        "actions": [action_to_json(a) for a in actions],
        "outcomes": [to_json_value(o) for o in outcomes],
    }


def replay(record: dict[str, Any]) -> Referee:
    """Rebuild a match from its replay record, forcing every recorded random outcome.

    Built-in search bots draw from the game RNG while thinking, so the dice a
    match saw depend on their internal search; forcing outcomes makes replay
    exact regardless.
    """
    if record["schema_version"] != REPLAY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported replay schema {record['schema_version']}")
    referee = Referee.new_game(record["seed"])
    for action, outcome in zip(record["actions"], record["outcomes"], strict=True):
        referee.execute(action_from_json(action), from_json_value(outcome))
    return referee
