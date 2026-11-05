"""Imitation dataset (ML-012) and training loop (ML-014) on a tiny dataset."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from catan_ai.models.encoding import dims
from catan_ai.models.features import ACTION_KINDS
from catan_ai.models.network import (
    NetConfig,
    PolicyValueNet,
    is_outcome_parameter,
    load_checkpoint,
    save_checkpoint,
)
from catan_ai.training.dataset import build_dataset, game_specs, load_split
from catan_ai.training.imitation import batch, evaluate, train

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
    full = _record_game((*spec, 1.0, 2))
    thin = _record_game((*spec, 0.3, 2))
    again = _record_game((*spec, 0.3, 2))
    assert 0 < len(thin["label"]) < 0.5 * len(full["label"])
    assert np.array_equal(thin["states"], again["states"])
    assert np.array_equal(thin["label"], again["label"])


def test_reencoding_from_replays_reproduces_the_dataset(data_dir: Path, tmp_path: Path) -> None:
    manifest = build_dataset(config(), tmp_path, source=data_dir)
    assert manifest["reencoded_from"] == str(data_dir)
    for name in ("train", "dev"):
        original, rebuilt = load_split(data_dir, name), load_split(tmp_path, name)
        for field in ("states", "actions", "offsets", "label", "winner", "opponent_hands", "seat"):
            assert np.array_equal(getattr(original, field), getattr(rebuilt, field)), field


def test_search_tags_and_weights(data_dir: Path) -> None:
    split = load_split(data_dir, "train")
    assert split.searched is not None and (split.searched == 0).all()  # no search teacher
    split.searched = (np.arange(len(split)) % 3).astype(np.int8)
    index = np.arange(12)
    b = batch(split, index, np.array([1.0, 2.0, 0.0], np.float32))
    assert b["weight"].tolist() == [1.0, 2.0, 0.0] * 4
    assert batch(split, index)["weight"].tolist() == [1.0] * 12


def test_distillation_fine_tunes_the_policy_tower_only(data_dir: Path, tmp_path: Path) -> None:
    split = load_split(data_dir, "train")
    start = PolicyValueNet(
        NetConfig(split.feature_version, *dims(split.feature_version), hidden=32, layers=2)
    )
    save_checkpoint(tmp_path / "init.pt", start, {})
    split.searched = (np.arange(len(split)) % 3).astype(np.int8)
    settings = {
        "seed": 0,
        "init": str(tmp_path / "init.pt"),
        "epochs": 1,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "value_weight": 0.0,
        "aux_weight": 0.0,
        "search_weights": [1.0, 1.0, 4.0],
        "net": {},
    }
    net, _ = train(settings, split, load_split(data_dir, "dev"), tmp_path, {}, log=lambda _: None)
    before, after = start.state_dict(), net.state_dict()
    for name in before:
        same = torch.equal(before[name], after[name])
        assert same == (is_outcome_parameter(name) or name.startswith("aux_hands")), name


def test_value_recalibration_trains_the_outcome_tower_only(data_dir: Path, tmp_path: Path) -> None:
    split = load_split(data_dir, "train")
    start = PolicyValueNet(
        NetConfig(split.feature_version, *dims(split.feature_version), hidden=32, layers=2)
    )
    save_checkpoint(tmp_path / "init.pt", start, {})
    settings = {
        "seed": 0,
        "init": str(tmp_path / "init.pt"),
        "epochs": 1,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.01,
        "policy_weight": 0.0,
        "value_weight": 1.0,
        "aux_weight": 0.1,
        "net": {},
    }
    net, _ = train(settings, split, load_split(data_dir, "dev"), tmp_path, {}, log=lambda _: None)
    before, after = start.state_dict(), net.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert changed and all(is_outcome_parameter(name) for name in changed)
