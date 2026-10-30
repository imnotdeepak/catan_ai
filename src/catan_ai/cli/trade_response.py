"""Fit the learned trade-response predictor (OPP-025) -> champions/, reports/.

Plays rules traders against a mix of responders (the five negotiator
styles, the plain champion and `catanatron:F`, which never accept), in every
seat rotation. Each response to a trader's offer is a sample: the response
features at the time of the offer, P(responder can pay) from the belief, and
whether it accepted. Boards are split: the first `--train` share fits the
model, the rest is held out. The report compares log loss, Brier score and
calibration on the held-out boards against the rule the rules trader uses.

Example:
    python -m catan_ai.cli.trade_response --boards 40 --workers 11
"""

import argparse
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from catan_ai.arena import Contestant, play_match
from catan_ai.provenance import REPO_ROOT, provenance
from catan_ai.trading.response import DEFAULT_PATH, FEATURES, ResponseModel, brier, log_loss

M = "model:champions/imitation-features-v3.pt"
LINEUPS = [
    [f"trader:rules:{M}", f"negotiator:cooperative:{M}", f"negotiator:risk-averse:{M}",
     "catanatron:F"],
    [f"trader:rules:{M}", f"negotiator:self-interested:{M}", f"negotiator:opportunistic:{M}",
     f"negotiator:trade-resistant:{M}"],
    [f"trader:rules:{M}", f"trader:rules:{M}", M, f"negotiator:cooperative:{M}"],
    [f"trader:rules:{M}", "catanatron:F", f"negotiator:opportunistic:{M}", M],
]  # fmt: skip
BINS = np.linspace(0, 1, 6)


def short(name: str) -> str:
    return name.replace(f":{M}", "").replace(M, "champion")


def run_game(job: tuple[int, int, list[str]]) -> list[dict[str, Any]]:
    board, seed, seating = job
    agents: dict[int, Any] = {}
    result, _ = play_match(
        f"response-{seed}", seed, [Contestant(n) for n in seating], agent_sink=agents
    )
    if result.error is not None:
        raise RuntimeError(f"game {seed}: {result.error}")
    rows = []
    for agent in agents.values():
        if not hasattr(agent, "samples"):
            continue
        for (seat, x, pay, accepted), (rule, _) in zip(
            agent.samples, agent.calibration, strict=True
        ):
            rows.append(
                {"board": board, "x": x, "pay": pay, "accepted": accepted, "rule": rule,
                 "responder": short(seating[seat])}
            )  # fmt: skip
    return rows


def calibration_rows(p: np.ndarray, y: np.ndarray) -> list[str]:
    lines = ["| Predicted | Responses | Mean predicted | Accepted |", "|---|---|---|---|"]
    index = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    for b in range(len(BINS) - 1):
        mask = index == b
        if mask.any():
            lines.append(
                f"| {BINS[b]:.1f} to {BINS[b + 1]:.1f} | {mask.sum():,} | "
                f"{p[mask].mean():.3f} | {y[mask].mean():.3f} |"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--boards", type=int, default=40)
    parser.add_argument("--seed", type=int, default=800000)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "runs" / "trade_response")
    parser.add_argument("--reuse", action="store_true", help="refit on saved samples")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEFAULT_PATH)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "reports" / "trade_response.md")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    data = args.data / "samples.npz"
    if args.reuse:
        saved = np.load(data, allow_pickle=False)
        board, x, pay = saved["board"], saved["x"], saved["pay"]
        y, rule, responder = saved["accepted"], saved["rule"], saved["responder"]
    else:
        jobs = [
            (b, args.seed + b * 16 + i * 4 + r, lineup[r:] + lineup[:r])
            for b in range(args.boards)
            for i, lineup in enumerate(LINEUPS)
            for r in range(4)
        ]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = [row for game in pool.map(run_game, jobs, chunksize=1) for row in game]
        board = np.array([r["board"] for r in rows])
        x = np.stack([r["x"] for r in rows])
        pay = np.array([r["pay"] for r in rows])
        y = np.array([r["accepted"] for r in rows])
        rule = np.array([r["rule"] for r in rows])
        responder = np.array([r["responder"] for r in rows])
        args.data.mkdir(parents=True, exist_ok=True)
        np.savez(data, board=board, x=x, pay=pay, accepted=y, rule=rule, responder=responder)

    cut = int(args.boards * args.train)
    train, hold = board < cut, board >= cut
    model = ResponseModel.fit(x[train], pay[train], y[train])
    model.save(args.out)
    learned = model.accept(x, pay)
    base = float(y[train].mean())

    lines = [
        "# Learned Trade-Response Predictor (OPP-025)",
        "",
        f"Generated by `python -m catan_ai.cli.trade_response` on {time.strftime('%Y-%m-%d')}. "
        f"catan_ai `{str(provenance()['catan_ai_commit'])[:7]}`. {args.boards} boards from seed "
        f"{args.seed}, {len(LINEUPS)} lineups, all seat rotations; {len(y):,} responses to rules "
        f"traders' offers ({train.sum():,} on the first {cut} boards for fitting, "
        f"{hold.sum():,} held out); {(time.perf_counter() - started) / 60:.1f} min.",
        "",
        "P(accept) = P(responder can pay, from the belief) * sigmoid(w . x). The rule is the "
        "rules trader's: P(can pay) * sigmoid(2 * (responder saving - 0.25)). Lineups:",
        "",
        *(f"- {', '.join(f'`{short(n)}`' for n in lineup)}" for lineup in LINEUPS),
        "",
        "## Held-out accuracy",
        "",
        "| Predictor | Log loss | Brier |",
        "|---|---|---|",
        f"| Base rate ({base:.3f}) | {log_loss(np.full(hold.sum(), base), y[hold]):.4f} | "
        f"{brier(np.full(hold.sum(), base), y[hold]):.4f} |",
        f"| Rule | {log_loss(rule[hold], y[hold]):.4f} | {brier(rule[hold], y[hold]):.4f} |",
        f"| **Learned** | **{log_loss(learned[hold], y[hold]):.4f}** | "
        f"**{brier(learned[hold], y[hold]):.4f}** |",
        "",
        "By responder (held out):",
        "",
        "| Responder | Responses | Accepted | Rule mean / Brier | Learned mean / Brier |",
        "|---|---|---|---|---|",
    ]
    groups: dict[str, list[int]] = defaultdict(list)
    for i in np.flatnonzero(hold):
        groups[str(responder[i])].append(int(i))
    for name in sorted(groups):
        g = np.array(groups[name])
        lines.append(
            f"| `{name}` | {len(g):,} | {y[g].mean():.3f} | {rule[g].mean():.3f} / "
            f"{brier(rule[g], y[g]):.4f} | {learned[g].mean():.3f} / "
            f"{brier(learned[g], y[g]):.4f} |"
        )
    lines += ["", "Learned calibration (held out):", ""]
    lines += calibration_rows(learned[hold], y[hold].astype(np.float64))
    lines += ["", "Rule calibration (held out):", ""]
    lines += calibration_rows(rule[hold], y[hold].astype(np.float64))
    lines += [
        "",
        "## Weights",
        "",
        "| Feature | Weight |",
        "|---|---|",
        *(f"| {n} | {w:+.3f} |" for n, w in zip(FEATURES, model.weights, strict=True)),
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(
        f"wrote {args.out} and {args.report}: {len(y)} responses; held-out Brier rule "
        f"{brier(rule[hold], y[hold]):.4f}, learned {brier(learned[hold], y[hold]):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
