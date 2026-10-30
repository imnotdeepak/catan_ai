"""Fit the choice-evidence miss rates (plan §6.2, `catan_ai.beliefs.evidence`) on recorded games.

Replays a split's games once, tracking every seat under every setting in a
grid of miss rates side by side, and scores each setting by the mean log
loss of the true count per opponent and resource (the beliefs report's
accuracy measure) and by the calibration of P(holds at least one), where the
belief is uncertain. Prints the best settings; the chosen one goes into
`DEFAULT_EVIDENCE`.

Example:
    python -m catan_ai.cli.evidence_fit --split dev --games 150 --workers 11
"""

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from catan_ai.arena import replay
from catan_ai.beliefs.evidence import ChoiceEvidence
from catan_ai.beliefs.particles import BeliefTracker
from catan_ai.engine import CanonicalAction
from catan_ai.engine.actions import RESOURCES
from catan_ai.observations import Referee
from catan_ai.provenance import REPO_ROOT
from catan_ai.training.dataset import read_jsonl

# Refined around the coarse fit (city 0.15 or 0.05, settlement 0.4, dev 1, maritime 0.5).
GRIDS = (
    (0.15, 0.08, 0.04, 0.02),  # city
    (0.6, 0.4, 0.25),  # settlement
    (1.0, 0.7),  # development card
    (0.7, 0.5, 0.35),  # maritime
)
BINS = np.linspace(0, 1, 11)
EVERY = 4  # score every EVERY-th decision (the trackers update at every one)


def settings() -> list[ChoiceEvidence]:
    return [ChoiceEvidence(*rates) for rates in itertools.product(*GRIDS)]


def score_game(record: dict[str, Any]) -> dict[str, np.ndarray]:
    grid = settings()
    trackers = [[BeliefTracker(s, evidence=e) for s in range(4)] for e in grid]
    loss = np.zeros(len(grid))
    cases = np.zeros(len(grid))
    # Per setting and bin: count, sum of p, sum of hits (uncertain P(holds >= 1) only).
    bins = np.zeros((len(grid), len(BINS) - 1, 3))
    step = [0]

    def observe(seat: int, referee: Referee, action: CanonicalAction) -> None:
        step[0] += 1
        score = step[0] % EVERY == 0
        ps = referee.adapter.referee_game.state.player_state
        truth = np.array([[ps[f"P{s}_{r}_IN_HAND"] for r in RESOURCES] for s in range(4)])
        for viewer in range(4):
            obs = referee.observe(viewer)
            for g, row in enumerate(trackers):
                belief = row[viewer].update(obs)
                if not score or belief.exact:
                    continue
                for other in range(4):
                    if other == viewer:
                        continue
                    for r in range(len(RESOURCES)):
                        dist = belief.marginal(other, r)
                        held = truth[other, r]
                        prob = dist[held] if held < len(dist) else 0.0
                        loss[g] += -np.log(max(prob, 1e-12))
                        cases[g] += 1
                        p = 1.0 - dist[0]
                        if 1e-9 < p < 1 - 1e-9:
                            b = min(int(p * (len(BINS) - 1)), len(BINS) - 2)
                            bins[g, b] += (1, p, float(held > 0))

    replay(record, observe)
    return {"loss": loss, "cases": cases, "bins": bins}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "imitation-v3")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    records = read_jsonl(args.data / args.split / "replays.jsonl.gz")[: args.games]
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(score_game, records, chunksize=1))
    loss = np.sum([p["loss"] for p in parts], axis=0) / np.sum([p["cases"] for p in parts], axis=0)
    bins = np.sum([p["bins"] for p in parts], axis=0)
    count = bins[:, :, 0]
    gap = np.abs(bins[:, :, 1] - bins[:, :, 2]).sum(axis=1) / count.sum(axis=1)
    bias = (bins[:, :, 2].sum(axis=1) - bins[:, :, 1].sum(axis=1)) / count.sum(axis=1)
    order = np.argsort(loss)
    grid = settings()
    print(f"{len(records)} games of {args.split}; {(time.perf_counter() - started) / 60:.1f} min")
    print("city settlement dev maritime | log loss | ECE | observed - predicted")
    for i in order[:10]:
        e = grid[i]
        print(
            f"{e.city:.2f} {e.settlement:.2f} {e.dev_card:.2f} {e.maritime:.2f} | "
            f"{loss[i]:.4f} | {gap[i]:.4f} | {bias[i]:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
