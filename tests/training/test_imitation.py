"""Imitation dataset (ML-012) and training loop (ML-014) on a tiny dataset."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from catan_ai.models.features import ACTION_KINDS
from catan_ai.models.network import load_checkpoint
from catan_ai.training.dataset import build_dataset, game_specs, load_split
from catan_ai.training.imitation import evaluate, train

TEACHERS = ["catanatron:F"]
LINEUP = ["catanatron:F", "catanatron:F", "catanatron:W", "catanatron:R"]


def config(train_seed: int = 10, dev_seed: int = 20) -> dict[str, Any]:
    return {
        "teachers": TEACHERS,
        "splits": {
            "train": {"groups": [{"lineup": LINEUP, "boards": 1, "seed": train_seed}]},
            "dev": {"groups": [{"lineup": LINEUP, "boards": 1, "seed": dev_seed}]},
        },
    }


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("imitation")
    build_dataset(config(), out)
    return out


def test_specs_cover_distinct_rotations_only() -> None:
    same = {"groups": [{"lineup": ["catanatron:F"] * 4, "boards": 2, "seed": 0}]}
    assert len(game_specs(same, TEACHERS)) == 2  # identical rotations are one game
    assert len(game_specs(config()["splits"]["train"], TEACHERS)) == 4


def test_overlapping_split_seeds_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="board seed"):
        build_dataset(config(train_seed=10, dev_seed=10), tmp_path)


def test_samples_are_teacher_decisions_with_valid_labels(data_dir: Path) -> None:
    split = load_split(data_dir, "train")
    assert len(split) > 100
    counts = np.diff(split.offsets)
    assert (counts >= 2).all()  # forced moves are not stored
    assert ((split.label >= 0) & (split.label < counts)).all()
    assert split.offsets[-1] == len(split.actions)
    games = [json.loads(line) for line in (data_dir / "train" / "games.jsonl").open()]
    for g, seat in zip(split.game, split.seat, strict=True):
        assert games[g]["seating"][seat] in TEACHERS
    # the label row is a real candidate encoding: exactly one action kind set
    kinds = split.actions[split.offsets[:-1] + split.label, : len(ACTION_KINDS)]
    assert (kinds.sum(1) == 1).all()


def test_outcome_and_privileged_labels_are_relative(data_dir: Path) -> None:
    split = load_split(data_dir, "train")
    games = [json.loads(line) for line in (data_dir / "train" / "games.jsonl").open()]
    for i in range(0, len(split), 25):
        game = games[split.game[i]]
        seat = int(split.seat[i])
        assert split.winner[i] == (game["winner_seat"] - seat) % 4
        assert split.final_points[i][0] == game["victory_points"][seat]
    assert split.opponent_hands.shape == (len(split), 3, 5)
    assert not {"opponent_hands", "points_now"} & set(
        np.load(data_dir / "train" / "inputs.npz").files
    )


def test_tampered_files_fail_verification(data_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    for path in data_dir.rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(data_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    labels = copy / "train" / "labels.npz"
    labels.write_bytes(labels.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="manifest"):
        load_split(copy, "train")


def test_training_runs_and_checkpoints(data_dir: Path, tmp_path: Path) -> None:
    settings = {
        "seed": 0,
        "epochs": 2,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "value_weight": 1.0,
        "aux_weight": 0.1,
        "net": {"hidden": 32, "layers": 2, "action_hidden": 16},
    }
    train_split, dev_split = load_split(data_dir, "train"), load_split(data_dir, "dev")
    _, history = train(
        settings, train_split, dev_split, tmp_path, {"data": "test"}, log=lambda _: None
    )
    assert len(history) == 2
    assert history[-1]["train_policy"] < history[0]["train_policy"] * 1.5
    net, metadata = load_checkpoint(tmp_path / "best.pt")
    assert metadata["data"] == "test"
    assert 1 <= metadata["policy_epoch"] <= 2 and 1 <= metadata["value_epoch"] <= 2
    metrics = evaluate(net, dev_split)
    assert 0 <= metrics["top1"] <= 1 and metrics["uniform_top1"] < 1
    assert (tmp_path / "metrics.csv").exists()


def test_keep_rate_thins_samples_reproducibly() -> None:
    from catan_ai.training.dataset import _record_game

    spec = ("k", 10, LINEUP, TEACHERS)
    full = _record_game((*spec, 1.0))
    thin = _record_game((*spec, 0.3))
    again = _record_game((*spec, 0.3))
    assert 0 < len(thin["label"]) < 0.5 * len(full["label"])
    assert np.array_equal(thin["states"], again["states"])
    assert np.array_equal(thin["label"], again["label"])
