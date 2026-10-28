"""Simulator throughput benchmark (ENG-009) -> reports/throughput.md, configs/budgets.json.

Example:
    python -m catan_ai.cli.throughput --games 200 --seed 42
"""

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from catanatron import Color, Game, RandomPlayer

from catan_ai.arena import Contestant, play_match
from catan_ai.arena.match import MAX_DECISIONS
from catan_ai.engine import EngineAdapter
from catan_ai.observations import Referee
from catan_ai.provenance import REPO_ROOT, provenance

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
# Each bot is seated with three random agents. Search bots simulate many games per
# decision, so they are timed over a capped number of decisions, not full games.
FULL_GAME_BOTS = ("catanatron:R", "catanatron:W", "catanatron:VP", "catanatron:F")
CAPPED_BOTS = ("catanatron:AB:2", "catanatron:G:10", "catanatron:M:50")
CAPPED_MATCHES = 2
CAPPED_DECISIONS = 60  # total across seats, so the bot makes about a quarter
DECISION_BUDGETS_S = (0.1, 1.0, 5.0)


def timed(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def bench_raw_engine(games: int, seed: int) -> dict[str, float]:
    """Catanatron alone: four RandomPlayers via Game.play()."""
    ticks = 0
    elapsed = 0.0
    for i in range(games):
        game = Game([RandomPlayer(c) for c in COLORS], seed=seed + i)
        elapsed += timed(game.play)
        ticks += len(game.state.action_records)
    return {"games": games, "games_per_s": games / elapsed, "decisions_per_s": ticks / elapsed}


def bench_firewalled(games: int, seed: int) -> dict[str, float]:
    """Our full path: Referee events + observation per decision + random agents."""
    decisions = 0
    elapsed = 0.0
    turns = []
    for i in range(games):
        result, _ = play_match(f"t{i}", seed + i, [Contestant("random")] * 4)
        elapsed += result.wall_seconds
        decisions += sum(result.decisions)
        turns.append(result.num_turns)
    return {
        "games": games,
        "games_per_s": games / elapsed,
        "decisions_per_s": decisions / elapsed,
        "mean_decisions_per_game": decisions / games,
        "mean_turns_per_game": statistics.mean(turns),
    }


def midgame_adapter(seed: int) -> EngineAdapter:
    adapter = EngineAdapter.new_game(seed)
    policy = random.Random(seed)
    for _ in range(400):
        adapter.execute(policy.choice(adapter.legal_actions()))
    return adapter


def bench_operations(samples: int, seed: int) -> dict[str, float]:
    """Per-call cost of copying, branching, observing and listing legal actions."""
    adapter = midgame_adapter(seed)
    referee = Referee(adapter)
    game = adapter.referee_game

    def per_call(fn: Callable[[], Any]) -> float:
        return timed(lambda: [fn() for _ in range(samples)]) / samples * 1e6

    return {
        "engine_game_copy_us": per_call(game.copy),
        "adapter_branch_us": per_call(lambda: adapter.branch(1)),
        "legal_actions_us": per_call(adapter.legal_actions),
        "observe_us": per_call(lambda: referee.observe(adapter.current_seat)),
    }


def bench_rollouts(rollouts: int, seed: int) -> dict[str, float]:
    """Random playouts to the end from a branched midgame position (a search primitive)."""
    adapter = midgame_adapter(seed)
    policy = random.Random(seed)
    lengths = []

    def run() -> None:
        for i in range(rollouts):
            branch = adapter.branch(i)
            steps = 0
            while not branch.is_over:
                branch.execute(policy.choice(branch.legal_actions()))
                steps += 1
            lengths.append(steps)

    elapsed = timed(run)
    return {
        "rollouts": rollouts,
        "rollouts_per_s": rollouts / elapsed,
        "mean_rollout_decisions": statistics.mean(lengths),
    }


def bench_bots(games: int, seed: int) -> dict[str, dict[str, float]]:
    """Mean seconds per decision for each built-in bot, seated with three random agents."""
    out = {}
    plans: list[tuple[str, int, int | None]] = [(b, games, None) for b in FULL_GAME_BOTS]
    plans += [(b, CAPPED_MATCHES, CAPPED_DECISIONS) for b in CAPPED_BOTS]
    for name, n, cap in plans:
        print(f"  {name}", file=sys.stderr)
        seconds = decisions = 0.0
        for i in range(n):
            seating = [Contestant(name)] + [Contestant("random")] * 3
            result, _ = play_match(f"b{i}", seed + i, seating, cap or MAX_DECISIONS)
            seconds += result.decision_seconds[0]
            decisions += result.decisions[0]
        out[name] = {
            "matches": n,
            "full_games": cap is None,
            "bot_decisions": int(decisions),
            "ms_per_decision": seconds / decisions * 1000,
        }
    return out


def derive_budgets(rollouts: dict[str, float]) -> dict[str, Any]:
    rate = rollouts["rollouts_per_s"]
    return {
        "source": "reports/throughput.md (ENG-009)",
        "random_rollouts_per_s": round(rate, 1),
        "rollouts_per_decision": {f"{b}s": int(rate * b) for b in DECISION_BUDGETS_S},
        "note": (
            "Upper bounds for uniformly random, single-process playouts from midgame. "
            "Learned or heuristic rollout policies cost more per step; re-measure."
        ),
    }


def render_report(data: dict[str, Any]) -> str:
    raw, fw, ops, roll, bots, env = (
        data[k] for k in ("raw_engine", "firewalled", "operations", "rollouts", "bots", "env")
    )
    overhead = raw["decisions_per_s"] / fw["decisions_per_s"]
    lines = [
        "# Throughput Report (ENG-009)",
        "",
        f"Generated by `python -m catan_ai.cli.throughput` on {env['date']}. "
        "Single process; numbers are machine-specific, so re-run on new hardware.",
        "",
        "| Environment | Value |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in env.items() if k != "date"),
        "",
        "## Full games",
        "",
        "| Path | Games | Games/s | Decisions/s |",
        "|---|---|---|---|",
        f"| Catanatron alone (`Game.play`, 4 RandomPlayers) | {raw['games']} "
        f"| {raw['games_per_s']:.1f} | {raw['decisions_per_s']:,.0f} |",
        f"| Our path (Referee events + observation per decision, 4 random agents) "
        f"| {fw['games']} | {fw['games_per_s']:.1f} | {fw['decisions_per_s']:,.0f} |",
        "",
        f"Firewall and event logging cost **{overhead:.1f}x** per decision over the bare "
        f"engine. A random game averages {fw['mean_decisions_per_game']:.0f} decisions "
        f"and {fw['mean_turns_per_game']:.0f} turns.",
        "",
        "## Per-call costs (midgame position)",
        "",
        "| Operation | Microseconds |",
        "|---|---|",
        f"| Engine `Game.copy()` (unsafe: shares RNG) | {ops['engine_game_copy_us']:.0f} |",
        f"| `EngineAdapter.branch()` (safe copy) | {ops['adapter_branch_us']:.0f} |",
        f"| `legal_actions()` | {ops['legal_actions_us']:.0f} |",
        f"| `Referee.observe()` | {ops['observe_us']:.0f} |",
        "",
        "## Random rollouts from midgame (search primitive)",
        "",
        f"{roll['rollouts_per_s']:.1f} rollouts/s; a rollout averages "
        f"{roll['mean_rollout_decisions']:.0f} decisions to the end of the game.",
        "",
        "## Built-in bots (each seated with three random agents)",
        "",
        "| Bot | Matches | Measured over | Bot decisions | ms per decision |",
        "|---|---|---|---|---|",
        *(
            f"| `{b}` | {v['matches']} | {'full games' if v['full_games'] else 'opening only'} "
            f"| {v['bot_decisions']:,} | {v['ms_per_decision']:,.1f} |"
            for b, v in bots.items()
        ),
        "",
        f"Search bots are timed over the first {CAPPED_DECISIONS} decisions of "
        f"{CAPPED_MATCHES} matches (setup and early turns), so treat their numbers as "
        "order-of-magnitude.",
        "",
        "## Derived budgets",
        "",
        "Written to `configs/budgets.json`. Random-playout counts affordable per decision "
        "at the given time budget:",
        "",
        "| Time per decision | Random rollouts |",
        "|---|---|",
        *(f"| {k} | {v:,} |" for k, v in data["budgets"]["rollouts_per_decision"].items()),
        "",
        "Implications for the plan's placeholder budgets (§4.4, §8.3, §10.3) are discussed "
        "in `docs/architecture-decisions.md` (ADR-006).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "reports" / "throughput.md")
    parser.add_argument("--budgets", type=Path, default=REPO_ROOT / "configs" / "budgets.json")
    args = parser.parse_args(argv)

    data: dict[str, Any] = {
        "env": {
            "date": time.strftime("%Y-%m-%d"),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "engine_commit": provenance()["engine_commit"],
            "games": args.games,
            "seed": args.seed,
        }
    }
    for key, fn in (
        ("raw_engine", lambda: bench_raw_engine(args.games, args.seed)),
        ("firewalled", lambda: bench_firewalled(args.games, args.seed)),
        ("operations", lambda: bench_operations(500, args.seed)),
        ("rollouts", lambda: bench_rollouts(max(20, args.games // 4), args.seed)),
        ("bots", lambda: bench_bots(max(8, args.games // 10), args.seed)),
    ):
        print(f"benchmarking {key}...", file=sys.stderr)
        data[key] = fn()
    data["budgets"] = derive_budgets(data["rollouts"])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(data), encoding="utf-8")
    args.budgets.parent.mkdir(parents=True, exist_ok=True)
    args.budgets.write_text(json.dumps(data["budgets"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
