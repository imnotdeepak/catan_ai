"""Run a seat-rotated four-player tournament.

Example:
    python -m catan_ai.cli.arena --lineup random,random,catanatron:W,catanatron:F \
        --boards 25 --seed 42 --out runs/arena-demo
"""

import argparse
import json
import sys
from pathlib import Path

from catan_ai.arena import Contestant, MatchResult, run_tournament, summarize
from catan_ai.arena.match import MAX_DECISIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lineup", required=True, help="four comma-separated contestants")
    parser.add_argument("--boards", type=int, default=10, help="board seeds; 4 matches each")
    parser.add_argument("--seed", type=int, default=0, help="first board seed")
    parser.add_argument("--out", type=Path, default=None, help="directory for JSON output")
    parser.add_argument("--max-decisions", type=int, default=MAX_DECISIONS)
    args = parser.parse_args(argv)

    lineup = [Contestant(name.strip()) for name in args.lineup.split(",")]

    def progress(result: MatchResult) -> None:
        outcome = result.winner or f"no winner ({result.truncation_reason or result.error})"
        print(f"{result.match_id}: {outcome} in {result.num_turns} turns", file=sys.stderr)

    results = run_tournament(lineup, args.boards, args.seed, args.out, args.max_decisions, progress)
    print(json.dumps(summarize(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
