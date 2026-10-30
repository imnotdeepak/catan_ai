"""Belief tier usage and calibration on recorded games (BEL-023, BEL-024) -> reports/.

Replays a dataset split's games and tracks every seat's beliefs at every
decision, comparing them with the true hands:

- tier usage: how often the ledger is exact, enumerated or resampled, and how
  many hypotheses it holds; decides whether the particle fallback is needed;
- impossible hypotheses: the true joint hand must never be ruled out;
- calibration: P(an opponent holds at least one of a resource) and P(it holds
  a face-down VP card), binned against how often that was true.

Example:
    python -m catan_ai.cli.beliefs_report --data data/imitation-v3 --split holdout
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from catan_ai.arena import replay
from catan_ai.beliefs.dev_cards import DevBelief, dev_belief
from catan_ai.beliefs.evidence import DEFAULT_EVIDENCE, NO_EVIDENCE
from catan_ai.beliefs.particles import DEFAULT_BUDGET, BeliefTracker
from catan_ai.engine import CanonicalAction
from catan_ai.engine.actions import RESOURCES
from catan_ai.observations import DEV_CARDS, Referee
from catan_ai.provenance import REPO_ROOT, provenance
from catan_ai.training.dataset import read_jsonl

BINS = np.linspace(0, 1, 11)
VP = DEV_CARDS.index("VICTORY_POINT")
PLAY_RATE_GRID = np.round(np.arange(0.0, 1.0, 0.05), 2)
HOARD_GRID = np.round(np.arange(0.0, 0.91, 0.05), 2)


def track_game(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
    record, budget = job
    trackers = [BeliefTracker(s, budget=budget) for s in range(4)]
    plain = [BeliefTracker(s, budget=budget, evidence=NO_EVIDENCE) for s in range(4)]
    out: dict[str, list[Any]] = {
        "tier": [], "size": [], "seconds": [], "ruled_out": [], "resource": [], "vp": [],
        "abs_error": [], "naive_error": [], "log_loss": [], "resource_plain": [],
        "abs_error_plain": [], "log_loss_plain": [],
    }  # fmt: skip

    def observe(seat: int, referee: Referee, action: CanonicalAction) -> None:
        ps = referee.adapter.referee_game.state.player_state
        truth = np.array([[ps[f"P{s}_{r}_IN_HAND"] for r in RESOURCES] for s in range(4)])
        for viewer, tracker in enumerate(trackers):
            obs = referee.observe(viewer)
            tick = time.perf_counter()
            belief = tracker.update(obs)
            out["seconds"].append(time.perf_counter() - tick)
            out["tier"].append(tracker.tier)
            out["size"].append(belief.size)
            out["ruled_out"].append(not (belief.hands == truth[None]).all(axis=(1, 2)).any())
            expected = belief.expected()
            devs = dev_belief(obs)
            if DEFAULT_EVIDENCE.active:  # the ledger without choice evidence, for comparison
                base = plain[viewer].update(obs)
                base_expected = base.expected()
                for other in range(4):
                    if other == viewer:
                        continue
                    out["abs_error_plain"].append(np.abs(base_expected[other] - truth[other]).sum())
                    for r in range(len(RESOURCES)):
                        dist = base.marginal(other, r)
                        held = truth[other, r]
                        out["resource_plain"].append((1.0 - dist[0], held > 0))
                        prob = dist[held] if held < len(dist) else 0.0
                        out["log_loss_plain"].append(-np.log(max(prob, 1e-12)))
            for other in range(4):
                if other == viewer:
                    continue
                count = truth[other].sum()
                out["abs_error"].append(np.abs(expected[other] - truth[other]).sum())
                out["naive_error"].append(np.abs(count / len(RESOURCES) - truth[other]).sum())
                for r in range(len(RESOURCES)):
                    p = 1.0 - belief.marginal(other, r)[0]
                    out["resource"].append((p, truth[other, r] > 0))
                    dist = belief.marginal(other, r)
                    held = truth[other, r]
                    prob = dist[held] if held < len(dist) else 0.0
                    out["log_loss"].append(-np.log(max(prob, 1e-12)))
                if devs.hidden[other] > 0:
                    held_vp = ps[f"P{other}_{DEV_CARDS[VP]}_IN_HAND"] > 0
                    case = (int(devs.hidden[other]), tuple(devs.unseen), devs.exposure[other])
                    out["vp"].append((*case, held_vp))

    replay(record, observe)
    out["recoveries"] = [sum(t.recoveries for t in trackers)]
    return out


def vp_probabilities(
    cases: list[tuple[Any, ...]], play_rate: float, hoard: float = 0.0
) -> np.ndarray:
    """P(holds a VP card) for recorded (hidden, unseen, exposure, held) cases."""
    out = []
    for hidden, unseen, exposure, _ in cases:
        belief = DevBelief(
            np.array(unseen), np.array([hidden]), np.array([exposure]), -1, play_rate, hoard
        )
        out.append(belief.prob_holds(0, "VICTORY_POINT"))
    return np.array(out)


def fit_play_rate(cases: list[tuple[Any, ...]]) -> tuple[float, float, float]:
    """(play rate, hoard share, log loss) minimizing log loss of P(holds a VP card),
    by grid search."""
    held = np.array([c[3] for c in cases], dtype=np.float64)
    best = (0.0, 0.0, np.inf)
    for rate in PLAY_RATE_GRID:
        for hoard in HOARD_GRID:
            p = np.clip(vp_probabilities(cases, float(rate), float(hoard)), 1e-9, 1 - 1e-9)
            loss = float(-np.mean(held * np.log(p) + (1 - held) * np.log(1 - p)))
            if loss < best[2]:
                best = (float(rate), float(hoard), loss)
    return best


def run(records: list[dict[str, Any]], budget: int, workers: int) -> dict[str, list[Any]]:
    jobs = [(r, budget) for r in records]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(track_game, jobs, chunksize=4))
    return {k: [x for part in parts for x in part[k]] for k in parts[0]}


def calibration_rows(pairs: np.ndarray, uncertain_only: bool) -> list[str]:
    p, hit = pairs[:, 0], pairs[:, 1]
    if uncertain_only:
        keep = (p > 1e-9) & (p < 1 - 1e-9)
        p, hit = p[keep], hit[keep]
    rows = []
    index = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    for b in range(len(BINS) - 1):
        mask = index == b
        if mask.sum() == 0:
            continue
        rows.append(
            f"| {BINS[b]:.1f} to {BINS[b + 1]:.1f} | {mask.sum():,} | {p[mask].mean():.3f} "
            f"| {hit[mask].mean():.3f} |"
        )
    ece = sum(
        (index == b).sum() * abs(p[index == b].mean() - hit[index == b].mean())
        for b in range(len(BINS) - 1)
        if (index == b).any()
    ) / max(len(p), 1)
    rows.append(f"| **all** | {len(p):,} | {p.mean():.3f} | {hit.mean():.3f} (ECE {ece:.4f}) |")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "imitation-v3")
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--fit-split", default="dev", help="split to fit the VP play rate on")
    parser.add_argument("--games", type=int, default=0, help="0 = all")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "reports" / "beliefs.md")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    records = read_jsonl(args.data / args.split / "replays.jsonl.gz")
    if args.games:
        records = records[: args.games]
    merged = run(records, args.budget, args.workers)
    fit_records = read_jsonl(args.data / args.fit_split / "replays.jsonl.gz")
    if args.games:
        fit_records = fit_records[: args.games]
    play_rate, hoard, fit_loss = fit_play_rate(run(fit_records, args.budget, args.workers)["vp"])

    tier = np.array(merged["tier"])
    size = np.array(merged["size"])
    seconds = np.array(merged["seconds"])
    ruled_out = int(np.sum(merged["ruled_out"]))
    resource = np.array(merged["resource"], dtype=np.float64)
    resource_plain = np.array(merged["resource_plain"], dtype=np.float64).reshape(-1, 2)
    e = DEFAULT_EVIDENCE
    evidence_note = (
        f"Choice evidence (`catan_ai.beliefs.evidence`): at an opponent's END_TURN, hypotheses "
        f"under which it could have built a city, a settlement or a development card, or "
        f"traded four of a kind with the bank, are weighted by miss rates {e.city:.2f}, "
        f"{e.settlement:.2f}, {e.dev_card:.2f} and {e.maritime:.2f} (fitted on `dev` by "
        f"`catan_ai.cli.evidence_fit`)."
    )
    vp_held = np.array([c[3] for c in merged["vp"]], dtype=np.float64)
    vp_plain = np.column_stack([vp_probabilities(merged["vp"], 0.0), vp_held])
    vp_fitted = np.column_stack([vp_probabilities(merged["vp"], play_rate, hoard), vp_held])
    info = provenance()
    lineups = sorted({", ".join(r["seating"]) for r in records}) if "seating" in records[0] else []

    lines = [
        "# Belief Tier Usage and Calibration (BEL-022 to BEL-024)",
        "",
        f"Generated by `python -m catan_ai.cli.beliefs_report` on {time.strftime('%Y-%m-%d')}. "
        f"catan_ai `{str(info['catan_ai_commit'])[:7]}`, "
        f"engine `{str(info['engine_commit'])[:7]}`. "
        f"{len(records)} games of `{args.data.name}/{args.split}`, every seat tracked at every "
        f"decision ({len(tier):,} updates); hypothesis budget {args.budget:,}; "
        f"{(time.perf_counter() - started) / 60:.1f} min.",
        "",
        *(["Lineups: " + "; ".join(f"`{x}`" for x in lineups), ""] if lineups else []),
        "## Tiers",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Exact ledger (tier 1) | {np.mean(tier == 1):.1%} of updates |",
        f"| Enumerated hypotheses (tier 2) | {np.mean(tier == 2):.1%} |",
        f"| Resampled particles (tier 3) | {np.mean(tier == 3):.2%} |",
        f"| Hypotheses: mean / p99 / max | {size.mean():.2f} / {np.percentile(size, 99):.0f} / "
        f"{size.max()} |",
        f"| Collapse recoveries | {sum(merged['recoveries'])} |",
        f"| Updates where the true hands were ruled out | **{ruled_out}** |",
        f"| Update time: mean / p99 | {seconds.mean() * 1e3:.3f} ms / "
        f"{np.percentile(seconds, 99) * 1e3:.3f} ms |",
        "",
        "## Accuracy",
        "",
        "Error is the summed absolute difference between an opponent's expected and true hand.",
        "The naive baseline spreads its card count evenly over the five resources.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Mean error per opponent hand, belief | {np.mean(merged['abs_error']):.3f} cards |",
        f"| Mean error per opponent hand, naive | {np.mean(merged['naive_error']):.3f} cards |",
        f"| Mean log loss of the true count, per opponent and resource | "
        f"{np.mean(merged['log_loss']):.4f} nats |",
        *(
            [
                f"| Without choice evidence: error / log loss | "
                f"{np.mean(merged['abs_error_plain']):.3f} cards / "
                f"{np.mean(merged['log_loss_plain']):.4f} nats |",
                "",
                evidence_note,
            ]
            if len(resource_plain)
            else []
        ),
        "",
        "## Calibration",
        "",
        "P(an opponent holds at least one of a resource), only where the belief is uncertain "
        "(0 < p < 1):",
        "",
        "| Predicted | Cases | Mean predicted | Observed |",
        "|---|---|---|---|",
        *calibration_rows(resource, uncertain_only=True),
        "",
        *(
            [
                "Without choice evidence:",
                "",
                "| Predicted | Cases | Mean predicted | Observed |",
                "|---|---|---|---|",
                *calibration_rows(resource_plain, uncertain_only=True),
                "",
            ]
            if len(resource_plain)
            else []
        ),
        "P(an opponent with face-down development cards holds a VP card). Deck odds alone "
        "(hypergeometric, play rate 0):",
        "",
        "| Predicted | Cases | Mean predicted | Observed |",
        "|---|---|---|---|",
        *calibration_rows(vp_plain, uncertain_only=False),
        "",
        f"With hold-time evidence, play rate {play_rate:.2f} per playable card-turn and "
        f"hoarded share {hoard:.2f}, fitted on "
        f"`{args.fit_split}` (log loss {fit_loss:.4f}):",
        "",
        "| Predicted | Cases | Mean predicted | Observed |",
        "|---|---|---|---|",
        *calibration_rows(vp_fitted, uncertain_only=False),
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(
        f"wrote {args.report}; ruled out {ruled_out}; tier 3 {np.mean(tier == 3):.2%}; "
        f"play rate {play_rate:.2f}, hoard {hoard:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
