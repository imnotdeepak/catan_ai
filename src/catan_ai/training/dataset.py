"""Frozen imitation dataset from reference-ladder games (ML-012, plan §9.2 phase A).

Teacher bots (built-in Catanatron bots) play seat-rotated four-player games.
At teacher decisions with more than one legal action (each kept with the
group's `keep` probability, seeded by match) we store:

- inputs (`inputs.npz`): the player-view state and candidate-action encodings,
  built from the teacher seat's `PlayerObservation` only;
- labels (`labels.npz`): the teacher's choice and the game outcome;
- privileged labels (`privileged.npz`): hidden facts at decision time
  (opponents' hands, true victory points). Training may use them as targets,
  never as inputs, and they live in their own file so that stays visible;
- `meta.npz`: game, seat, turn and teacher of every sample.

Teachers read the full game, so some labels depend on hidden cards (plan §9.2,
"Imitating privileged bots"); model inputs stay player-view regardless.

Keeping a slice of many games rather than every decision of a few matters for
the value head: all samples of a game share one board and one outcome, so with
few games the network memorizes winners by board (reports/imitation_training.md).

Splits are disjoint sets of board seeds. Every array file's sha256 is in
`manifest.json`, so a training run can name exactly the data it used.
"""

import gzip
import hashlib
import json
import random
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from catan_ai.arena import Contestant, rotations
from catan_ai.arena.match import play_match, replay, replay_record
from catan_ai.engine import RESOURCES, CanonicalAction
from catan_ai.models.encoding import LATEST_FEATURE_VERSION, check, dims, encode
from catan_ai.models.features import NUM_SEATS
from catan_ai.observations import Referee
from catan_ai.provenance import provenance

DATASET_VERSION = 1
FILES = ("inputs", "labels", "privileged", "meta")


@dataclass
class _Recorder:
    """Decision observer: records teacher decisions as samples."""

    teacher_seats: dict[int, int]  # seat -> teacher index
    keep: float
    rng: random.Random
    feature_version: int
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    seats: list[int] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    hands: list[np.ndarray] = field(default_factory=list)
    points: list[np.ndarray] = field(default_factory=list)

    def __call__(self, seat: int, referee: Referee, action: CanonicalAction) -> None:
        if seat not in self.teacher_seats:
            return
        observation = referee.observe(seat)
        if len(observation.legal_actions) < 2 or self.rng.random() >= self.keep:
            return
        state, rows = encode(observation, self.feature_version)
        self.states.append(state.astype(np.float16))
        self.actions.append(rows.astype(np.float16))
        self.labels.append(observation.legal_actions.index(action))
        self.seats.append(seat)
        self.turns.append(observation.num_turns)

        ps = referee.adapter.referee_game.state.player_state
        order = [(seat + r) % NUM_SEATS for r in range(NUM_SEATS)]  # relative -> absolute
        self.hands.append(
            np.array([[ps[f"P{s}_{r}_IN_HAND"] for r in RESOURCES] for s in order[1:]], np.int8)
        )
        self.points.append(np.array([ps[f"P{s}_ACTUAL_VICTORY_POINTS"] for s in order], np.int8))


# id, seed, seating, teachers, keep probability, feature version
GameSpec = tuple[str, int, list[str], list[str], float, int]


def _recorder(spec: GameSpec) -> _Recorder:
    match_id, _, seating, teachers, keep, version = spec
    teacher_seats = {s: teachers.index(n) for s, n in enumerate(seating) if n in teachers}
    return _Recorder(teacher_seats, keep, random.Random(f"keep:{match_id}"), version)


def _record_game(spec: GameSpec) -> dict[str, Any]:
    """Worker entry point: play one game and return its samples and replay."""
    match_id, seed, seating, *_ = spec
    recorder = _recorder(spec)
    result, referee = play_match(
        match_id, seed, [Contestant(n) for n in seating], observer=recorder
    )
    if result.error is not None:
        raise RuntimeError(f"{match_id}: {result.error}")
    return {**_package(recorder, result.to_json()), "replay": replay_record(result, referee)}


def _reencode_game(job: tuple[GameSpec, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Worker entry point: rebuild a recorded game from its replay with the current encoder.

    The keep draws are seeded by match id and made at the same decisions, so the
    same decisions are sampled as when the game was first played.
    """
    spec, result, record = job
    if record["match_id"] != spec[0] or result["match_id"] != spec[0]:
        raise ValueError(f"replay {record['match_id']} does not match spec {spec[0]}")
    recorder = _recorder(spec)
    replay(record, observer=recorder)
    return {**_package(recorder, result), "replay": record}


def _package(recorder: _Recorder, result: dict[str, Any]) -> dict[str, Any]:
    n = len(recorder.labels)
    state_dim, action_dim = dims(recorder.feature_version)
    seats = np.array(recorder.seats, np.int8)
    winner = np.full(n, -1, np.int8)
    if result["winner_seat"] is not None:
        winner = ((result["winner_seat"] - seats) % NUM_SEATS).astype(np.int8)
    final = np.array(result["victory_points"], np.int8)
    return {
        "result": result,
        "states": np.stack(recorder.states) if n else np.zeros((0, state_dim), np.float16),
        "actions": (
            np.concatenate(recorder.actions) if n else np.zeros((0, action_dim), np.float16)
        ),
        "counts": np.array([len(a) for a in recorder.actions], np.int32),
        "label": np.array(recorder.labels, np.int16),
        "winner": winner,
        "final_points": np.stack([np.roll(final, -int(s)) for s in seats])
        if n
        else np.zeros((0, NUM_SEATS), np.int8),
        "opponent_hands": (
            np.stack(recorder.hands) if n else np.zeros((0, NUM_SEATS - 1, 5), np.int8)
        ),
        "points_now": np.stack(recorder.points) if n else np.zeros((0, NUM_SEATS), np.int8),
        "seat": seats,
        "turn": np.array(recorder.turns, np.int16),
        "teacher": np.array([recorder.teacher_seats[s] for s in recorder.seats], np.int8),
    }


def game_specs(
    split: dict[str, Any], teachers: Sequence[str], feature_version: int = LATEST_FEATURE_VERSION
) -> list[GameSpec]:
    """Every distinct seat rotation of every lineup on every board of a split."""
    specs: list[GameSpec] = []
    for group in split["groups"]:
        lineup = [Contestant(n) for n in group["lineup"]]
        seatings = list(dict.fromkeys(tuple(c.name for c in r) for r in rotations(lineup)))
        for board in range(group["boards"]):
            seed = group["seed"] + board
            for r, seating in enumerate(seatings):
                keep = float(group.get("keep", 1.0))
                specs.append(
                    (f"s{seed}-r{r}", seed, list(seating), list(teachers), keep, feature_version)
                )
    return specs


def _games(
    worker: Callable[[Any], dict[str, Any]], jobs: list[Any], workers: int
) -> Iterator[dict[str, Any]]:
    """Run `worker` over `jobs`, yielding results in job order."""
    if workers <= 1:
        yield from map(worker, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, job): i for i, job in enumerate(jobs)}
        done = {}
        for future in as_completed(futures):
            done[futures[future]] = future.result()
        for i in range(len(jobs)):  # deterministic order, independent of completion order
            yield done[i]


def build_split(
    name: str,
    split: dict[str, Any],
    teachers: Sequence[str],
    out_dir: Path,
    workers: int = 1,
    source: Path | None = None,
    feature_version: int = LATEST_FEATURE_VERSION,
) -> dict[str, Any]:
    """Record one split and write its files. Returns its manifest entry.

    With `source` (a dataset directory holding this split's replays), games are
    re-encoded from their replays instead of played again.
    """
    started = time.perf_counter()
    specs = game_specs(split, teachers, feature_version)
    if source is None:
        games = list(_games(_record_game, specs, workers))
    else:
        results = read_jsonl(source / name / "games.jsonl")
        records = read_jsonl(source / name / "replays.jsonl.gz")
        if not len(specs) == len(results) == len(records):
            raise ValueError(f"{source / name} does not hold the games of this split")
        jobs = list(zip(specs, results, records, strict=True))
        games = list(_games(_reencode_game, jobs, workers))
    for index, game in enumerate(games):
        game["game"] = np.full(len(game["label"]), index, np.int32)

    def cat(key: str) -> np.ndarray:
        return np.concatenate([g[key] for g in games])

    counts = cat("counts")
    arrays = {
        "inputs": {
            "states": cat("states"),
            "actions": cat("actions"),
            "offsets": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        },
        "labels": {
            "label": cat("label"),
            "winner": cat("winner"),
            "final_points": cat("final_points"),
        },
        "privileged": {"opponent_hands": cat("opponent_hands"), "points_now": cat("points_now")},
        "meta": {
            "game": cat("game"),
            "seat": cat("seat"),
            "turn": cat("turn"),
            "teacher": cat("teacher"),
        },
    }
    split_dir = out_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for file, content in arrays.items():
        path = split_dir / f"{file}.npz"
        np.savez_compressed(path, **content)  # type: ignore[arg-type]  # numpy stub: kwds typed bool
        digests[file] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (split_dir / "games.jsonl").open("w", encoding="utf-8") as f:
        for game in games:
            f.write(json.dumps(game["result"]) + "\n")
    with gzip.open(split_dir / "replays.jsonl.gz", "wt", encoding="utf-8") as f:
        for game in games:
            f.write(json.dumps(game["replay"]) + "\n")

    results = [g["result"] for g in games]
    return {
        "games": len(games),
        "decided": sum(r["winner_seat"] is not None for r in results),
        "samples": len(counts),
        "candidates": int(counts.sum()),
        "max_candidates": int(counts.max()) if len(counts) else 0,
        "board_seeds": sorted({spec[1] for spec in specs}),
        "groups": split["groups"],
        "sha256": digests,
        "minutes": round((time.perf_counter() - started) / 60, 2),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_dataset(
    config: dict[str, Any], out_dir: Path, workers: int = 1, source: Path | None = None
) -> dict[str, Any]:
    """Play (or, with `source`, re-encode from replays) every split of `config`."""
    teachers = config["teachers"]
    version = config.get("feature_version", LATEST_FEATURE_VERSION)
    check(version)
    state_dim, action_dim = dims(version)
    seeds: dict[int, str] = {}
    for name, split in config["splits"].items():
        for spec in game_specs(split, teachers):
            if seeds.setdefault(spec[1], name) != name:
                raise ValueError(f"board seed {spec[1]} is in splits {seeds[spec[1]]} and {name}")
    manifest: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "feature_version": version,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "teachers": teachers,
        "config": config,
        "provenance": provenance(),
        "reencoded_from": None if source is None else str(source),
        "splits": {},
    }
    for name, split in config["splits"].items():
        manifest["splits"][name] = build_split(
            name, split, teachers, out_dir, workers, source, version
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass
class Split:
    """One loaded split. Candidates of sample i are actions[offsets[i]:offsets[i+1]]."""

    states: np.ndarray
    actions: np.ndarray
    offsets: np.ndarray
    label: np.ndarray
    winner: np.ndarray
    final_points: np.ndarray
    opponent_hands: np.ndarray
    points_now: np.ndarray
    game: np.ndarray
    seat: np.ndarray
    turn: np.ndarray
    teacher: np.ndarray

    feature_version: int = 1

    def __len__(self) -> int:
        return len(self.label)


def load_split(data_dir: Path, name: str, verify: bool = True) -> Split:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["feature_version"])
    entry = manifest["splits"][name]
    arrays: dict[str, np.ndarray] = {}
    for file in FILES:
        path = data_dir / name / f"{file}.npz"
        if verify and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"][file]:
            raise ValueError(f"{path} does not match the manifest")
        with np.load(path) as content:
            arrays.update({key: content[key] for key in content.files})
    return Split(**arrays, feature_version=manifest["feature_version"])
