"""Behavioral cloning with value and auxiliary heads (ML-014, plan §9.2 phases B-C).

Losses, all on player-view inputs:
- policy: cross-entropy of the teacher's choice over the legal candidates;
- value: cross-entropy of the eventual winner (truncated games excluded);
- auxiliary: opponents' hidden hands and final victory points (privileged
  labels, used only as targets), weighted by `aux_weight`.

The saved checkpoint combines the policy tower from the epoch with the lowest
dev policy loss and the outcome tower from the epoch with the lowest dev value
loss (the towers share no weights; see models/network.py). Metrics are
written per epoch to `metrics.csv`.
"""

import csv
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from catan_ai.models.encoding import dims
from catan_ai.models.features import ACTION_KINDS, PROMPT_SLICE, PROMPTS
from catan_ai.models.network import (
    NetConfig,
    PolicyValueNet,
    is_outcome_parameter,
    save_checkpoint,
)
from catan_ai.training.dataset import Split

Batch = dict[str, torch.Tensor]


def batch(split: Split, index: np.ndarray) -> Batch:
    """Gather samples, padding candidate sets to the largest in the batch."""
    start = split.offsets[index]
    counts = split.offsets[index + 1] - start
    width = int(counts.max())
    slots = np.arange(width)
    mask = slots[None, :] < counts[:, None]
    rows = np.where(mask, start[:, None] + slots[None, :], 0)
    actions = split.actions[rows].astype(np.float32)
    actions[~mask] = 0.0
    return {
        "states": torch.from_numpy(split.states[index].astype(np.float32)),
        "actions": torch.from_numpy(actions),
        "mask": torch.from_numpy(mask),
        "label": torch.from_numpy(split.label[index].astype(np.int64)),
        "winner": torch.from_numpy(split.winner[index].astype(np.int64)),
        "final_points": torch.from_numpy(split.final_points[index].astype(np.float32)),
        "opponent_hands": torch.from_numpy(split.opponent_hands[index].astype(np.float32)),
    }


def batches(split: Split, size: int, rng: np.random.Generator | None = None) -> Iterator[Batch]:
    order = rng.permutation(len(split)) if rng is not None else np.arange(len(split))
    for at in range(0, len(order), size):
        yield batch(split, order[at : at + size])


def losses(out: dict[str, torch.Tensor], b: Batch) -> dict[str, torch.Tensor]:
    decided = b["winner"] >= 0
    value = (
        F.cross_entropy(out["value_logits"][decided], b["winner"][decided])
        if decided.any()
        else out["value_logits"].sum() * 0
    )
    return {
        "policy": F.cross_entropy(out["policy_logits"], b["label"]),
        "value": value,
        "aux_hands": F.mse_loss(out["aux_hands"], b["opponent_hands"] / 4),
        "aux_points": F.mse_loss(out["aux_points"], b["final_points"] / 10),
    }


@torch.inference_mode()
def evaluate(net: PolicyValueNet, split: Split, batch_size: int = 1024) -> dict[str, Any]:
    """Held-out metrics: imitation accuracy and loss, value calibration, aux error."""
    net.eval()
    totals: dict[str, float] = {"policy": 0.0, "value": 0.0, "aux_hands": 0.0, "aux_points": 0.0}
    correct = np.zeros(len(split), bool)
    my_win = np.zeros(len(split), np.float32)
    hand_error = points_error = 0.0
    seen = decided_seen = 0
    for at in range(0, len(split), batch_size):
        index = np.arange(at, min(at + batch_size, len(split)))
        b = batch(split, index)
        out = net(b["states"], b["actions"], b["mask"])
        parts = losses(out, b)
        n, decided = len(index), int((b["winner"] >= 0).sum())
        for key in ("policy", "aux_hands", "aux_points"):
            totals[key] += float(parts[key]) * n
        totals["value"] += float(parts["value"]) * decided
        seen, decided_seen = seen + n, decided_seen + decided
        correct[index] = (out["policy_logits"].argmax(1) == b["label"]).numpy()
        my_win[index] = torch.softmax(out["value_logits"], 1)[:, 0].numpy()
        hand_error += float((out["aux_hands"] * 4 - b["opponent_hands"]).abs().sum())
        points_error += float((out["aux_points"] * 10 - b["final_points"]).abs().sum())

    decided_mask = split.winner >= 0
    won = (split.winner == 0)[decided_mask]
    p = my_win[decided_mask]
    prompts = split.states[:, PROMPT_SLICE].argmax(1)
    label_kind = split.actions[split.offsets[:-1] + split.label, : len(ACTION_KINDS)].argmax(1)
    return {
        "samples": len(split),
        "policy_loss": totals["policy"] / seen,
        "top1": float(correct.mean()),
        "uniform_top1": float((1 / np.diff(split.offsets)).mean()),
        "value_loss": totals["value"] / max(decided_seen, 1),
        "brier": float(((p - won) ** 2).mean()),
        "ece": expected_calibration_error(p, won),
        "reliability": reliability(p, won),
        "hand_mae": hand_error / (seen * 15),
        "points_mae": points_error / (seen * 4),
        "top1_by_prompt": {
            PROMPTS[k]: (float(correct[prompts == k].mean()), int((prompts == k).sum()))
            for k in np.unique(prompts)
        },
        "top1_by_label_kind": {
            str(ACTION_KINDS[k]): (
                float(correct[label_kind == k].mean()),
                int((label_kind == k).sum()),
            )
            for k in np.unique(label_kind)
        },
        "top1_by_teacher": {
            int(t): (float(correct[split.teacher == t].mean()), int((split.teacher == t).sum()))
            for t in np.unique(split.teacher)
        },
    }


def reliability(p: np.ndarray, won: np.ndarray, bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean predicted, observed win rate, count) per equal-width probability bin."""
    which = np.minimum((p * bins).astype(int), bins - 1)
    return [
        (float(p[which == k].mean()), float(won[which == k].mean()), int((which == k).sum()))
        for k in range(bins)
        if (which == k).any()
    ]


def expected_calibration_error(p: np.ndarray, won: np.ndarray, bins: int = 10) -> float:
    return sum(n * abs(mean_p - rate) for mean_p, rate, n in reliability(p, won, bins)) / max(
        len(p), 1
    )


def train(
    config: dict[str, Any],
    train_split: Split,
    dev_split: Split,
    out_dir: Path,
    metadata: dict[str, Any],
    log: Any = print,
) -> tuple[PolicyValueNet, list[dict[str, Any]]]:
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    version = train_split.feature_version
    state_dim, action_dim = dims(version)
    net = PolicyValueNet(NetConfig(version, state_dim, action_dim, **config["net"]))
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    steps = config["epochs"] * math.ceil(len(train_split) / config["batch_size"])
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, config["lr"], total_steps=steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best = {"policy": (math.inf, 0), "outcome": (math.inf, 0)}  # dev loss, epoch
    kept: dict[str, dict[str, Tensor]] = {}

    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        net.train()
        sums: dict[str, float] = {}
        count = 0
        for b in batches(train_split, config["batch_size"], rng):
            parts = losses(net(b["states"], b["actions"], b["mask"]), b)
            total = (
                parts["policy"]
                + config["value_weight"] * parts["value"]
                + config["aux_weight"] * (parts["aux_hands"] + parts["aux_points"])
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            schedule.step()
            n = len(b["label"])
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * n
            count += n
        dev = evaluate(net, dev_split)
        row = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 1),
            **{f"train_{k}": round(v / count, 5) for k, v in sums.items()},
            **{
                f"dev_{k}": round(dev[k], 5)
                for k in (
                    "policy_loss",
                    "top1",
                    "value_loss",
                    "brier",
                    "ece",
                    "hand_mae",
                    "points_mae",
                )
            },
        }
        history.append(row)
        log(", ".join(f"{k}={v}" for k, v in row.items()))
        improved = False
        for tower, loss, outcome in (
            ("policy", dev["policy_loss"], False),
            ("outcome", dev["value_loss"], True),
        ):
            if loss < best[tower][0]:
                best[tower], improved = (loss, epoch), True
                kept[tower] = {
                    name: tensor.detach().clone()
                    for name, tensor in net.state_dict().items()
                    if is_outcome_parameter(name) == outcome
                }
        if improved:
            merged = PolicyValueNet(net.config)
            merged.load_state_dict({**kept["policy"], **kept["outcome"]})
            epochs = {"policy_epoch": best["policy"][1], "value_epoch": best["outcome"][1]}
            save_checkpoint(out_dir / "best.pt", merged, {**metadata, **epochs, "training": config})
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    return net, history
