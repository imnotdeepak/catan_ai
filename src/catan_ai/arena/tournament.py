"""Seat-rotated tournaments with JSON results and replays (ENG-008).

Each board seed is played four times, rotating the lineup so every contestant
sits in every seat on the same board (the adapter keeps the board fixed for a
seed regardless of seating).
"""

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from catan_ai.arena.contestants import Contestant
from catan_ai.arena.match import MAX_DECISIONS, MatchResult, play_match, replay_record
from catan_ai.provenance import provenance


def rotations(lineup: Sequence[Contestant]) -> list[list[Contestant]]:
    return [[lineup[(seat + r) % 4] for seat in range(4)] for r in range(4)]


def _play(spec: tuple[str, int, list[Contestant], int]) -> tuple[MatchResult, dict[str, Any]]:
    """Worker entry point: play one match, return its result and replay record."""
    match_id, seed, seating, max_decisions = spec
    result, referee = play_match(match_id, seed, seating, max_decisions)
    return result, replay_record(result, referee)


def run_tournament(
    lineup: Sequence[Contestant],
    boards: int,
    base_seed: int,
    out_dir: Path | None = None,
    max_decisions: int = MAX_DECISIONS,
    on_result: Callable[[MatchResult], None] | None = None,
    workers: int = 1,
) -> list[MatchResult]:
    """Play every board in all four seat rotations.

    With `workers > 1`, matches run in parallel processes. Each match depends
    only on its seed and seating, so results are identical to a serial run;
    they are returned and written in match order.
    """
    if len(lineup) != 4:
        raise ValueError("A lineup is exactly four contestants")
    specs = [
        (f"s{base_seed}-b{board}-r{r}", base_seed + board, seating, max_decisions)
        for board in range(boards)
        for r, seating in enumerate(rotations(lineup))
    ]
    replays_dir = None
    if out_dir is not None:
        replays_dir = out_dir / "replays"
        replays_dir.mkdir(parents=True, exist_ok=True)

    def finished(result: MatchResult, record: dict[str, Any]) -> None:
        if replays_dir is not None:
            path = replays_dir / f"{result.match_id}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
        if on_result is not None:
            on_result(result)

    by_id: dict[str, MatchResult] = {}
    if workers <= 1:
        for spec in specs:
            result, record = _play(spec)
            by_id[result.match_id] = result
            finished(result, record)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed(pool.submit(_play, spec) for spec in specs):
                result, record = future.result()
                by_id[result.match_id] = result
                finished(result, record)
    results = [by_id[spec[0]] for spec in specs]

    if out_dir is not None:
        with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.to_json()) + "\n")
        summary = summarize(results)
        summary["config"] = {
            "lineup": [c.name for c in lineup],
            "boards": boards,
            "base_seed": base_seed,
            "max_decisions": max_decisions,
            **provenance(),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results


def summarize(results: Sequence[MatchResult]) -> dict[str, Any]:
    """Per-contestant counts. Win rates here are raw proportions; the statistical
    report with clustered intervals belongs to the evaluation package (EVAL-032)."""
    per: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"seats_played": 0, "wins": 0, "wins_by_seat": [0, 0, 0, 0], "privileged": False}
    )
    for result in results:
        for seat, name in enumerate(result.seating):
            entry = per[name]
            entry["seats_played"] += 1
            entry["privileged"] = result.privileged_seats[seat]
            if result.winner_seat == seat:
                entry["wins"] += 1
                entry["wins_by_seat"][seat] += 1
    for entry in per.values():
        entry["win_rate"] = round(entry["wins"] / entry["seats_played"], 4)
    return {
        "matches": len(results),
        "decided": sum(r.winner_seat is not None for r in results),
        "truncated": sum(r.truncated for r in results),
        "errors": sum(r.error is not None for r in results),
        "contestants": dict(per),
    }
