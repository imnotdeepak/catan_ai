"""Masked PPO training against a league (ML-029) -> runs/ppo/<name>/.

Each iteration plays `--games` rollout games in worker processes with the
current learner, then runs one PPO update. Every `--snapshot-every`
iterations the learner is saved as a snapshot and joins the league. Writes
`latest.pt`, `snap-<iteration>.pt`, `metrics.csv` and `config.json`.

Example:
    python -m catan_ai.cli.train_ppo --name ppo-1 --iterations 40 --games 220 --workers 11
"""

import argparse
import csv
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from catan_ai.models.network import load_checkpoint, save_checkpoint
from catan_ai.provenance import REPO_ROOT, provenance
from catan_ai.training.ppo import (
    LEARNER,
    Job,
    League,
    PPOConfig,
    build_buffer,
    elapsed,
    rollout,
    update,
)

CHAMPION = "champions/imitation-features-v3.pt"
LEAGUE = {
    f"model:{CHAMPION}": 0.35,
    "catanatron:F": 0.35,
    "catanatron:W": 0.05,
    "catanatron:VP": 0.05,
}
SNAPSHOT_WEIGHT = 0.20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--init", default=CHAMPION)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--games", type=int, default=220, help="rollout games per iteration")
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--resume", action="store_true")
    for name, value in asdict(PPOConfig()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args(argv)
    config = PPOConfig(**{k: getattr(args, k) for k in asdict(PPOConfig())})

    out = REPO_ROOT / "runs" / "ppo" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(12)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    init = REPO_ROOT / args.init
    net, metadata = load_checkpoint(out / "latest.pt" if args.resume else init)
    reference, _ = load_checkpoint(init)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    league = League(dict(LEAGUE), SNAPSHOT_WEIGHT)
    start = 0
    if args.resume:
        start = int(metadata["iteration"])
        league.snapshots = list(metadata["snapshots"])
    else:
        (out / "config.json").write_text(
            json.dumps({**vars(args), "ppo": asdict(config), "league": LEAGUE,
                        "provenance": provenance()}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )  # fmt: skip

    metrics = out / "metrics.csv"
    fields = [
        "iteration", "games", "kept", "samples", "learner_win", "vs_champion", "vs_F",
        "truncated", "errors", "policy", "value", "entropy", "kl", "anchor", "clipped",
        "rollout_min", "update_min",
    ]  # fmt: skip
    if not args.resume or not metrics.exists():
        with metrics.open("w", newline="") as f:
            csv.writer(f).writerow(fields)

    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for iteration in range(start + 1, args.iterations + 1):
            weights = out / "latest.pt"
            state = {"iteration": iteration - 1, "snapshots": league.snapshots, "init": args.init}
            save_checkpoint(weights, net, state)
            jobs = [
                Job(
                    args.seed + iteration * 10_000 + g, league.seating(rng), str(weights), iteration
                )
                for g in range(args.games)
            ]
            tick = time.perf_counter()
            games = list(pool.map(rollout, jobs, chunksize=1))
            rollout_min = (time.perf_counter() - tick) / 60

            trajectories = [t for g in games for t in g["trajectories"]]
            buffer = build_buffer(trajectories, config)
            tick = time.perf_counter()
            stats = update(net, optimizer, buffer, config, reference, np_rng)
            update_min = (time.perf_counter() - tick) / 60

            decided = [g for g in games if g["winner_seat"] is not None]
            learner_wins = sum(g["seating"][g["winner_seat"]] == LEARNER for g in decided)
            seats = sum(g["seating"].count(LEARNER) for g in decided)

            def share(opponent: str, decided: list[dict[str, Any]] = decided) -> float:
                rows = [g for g in decided if opponent in g["seating"]]
                wins = sum(g["seating"][g["winner_seat"]] == LEARNER for g in rows)
                return float(wins / max(sum(g["seating"].count(LEARNER) for g in rows), 1))

            row = {
                "iteration": iteration,
                "games": len(games),
                "kept": len(decided),
                "samples": len(buffer),
                "learner_win": learner_wins / max(seats, 1),
                "vs_champion": share(f"model:{CHAMPION}"),
                "vs_F": share("catanatron:F"),
                "truncated": sum(g["truncated"] for g in games),
                "errors": sum(g["error"] is not None for g in games),
                **stats,
                "rollout_min": rollout_min,
                "update_min": update_min,
            }
            with metrics.open("a", newline="") as f:
                csv.writer(f).writerow([f"{row[k]:.4f}" if isinstance(row[k], float) else row[k]
                                        for k in fields])  # fmt: skip
            print(
                f"[{elapsed(started)}] it {iteration}: win/seat {row['learner_win']:.3f} "
                f"(vs champion {row['vs_champion']:.3f}, vs F {row['vs_F']:.3f}); "
                f"{len(buffer):,} samples; entropy {stats['entropy']:.3f}, kl {stats['kl']:.4f}, "
                f"clipped {stats['clipped']:.3f}; rollout {rollout_min:.1f} + update "
                f"{update_min:.1f} min",
                flush=True,
            )
            if iteration % args.snapshot_every == 0:
                snap = out / f"snap-{iteration}.pt"
                save_checkpoint(snap, net, {"iteration": iteration, "init": args.init})
                league.add_snapshot(f"model:{snap.relative_to(REPO_ROOT).as_posix()}")
    state = {"iteration": args.iterations, "snapshots": league.snapshots, "init": args.init}
    save_checkpoint(out / "latest.pt", net, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
