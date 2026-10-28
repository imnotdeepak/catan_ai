"""Probe how built-in bots respond to domestic trade offers (ENG-010 evidence).

Seats one `random-trader` against three copies of each bot, rotating seats, and
counts offers, responses and completed trades from the referee event log.

Example:
    python -m catan_ai.cli.trade_probe --boards 5 --seed 7
"""

import argparse
import json
from collections import Counter
from typing import Any

from catan_ai.arena import Contestant, play_match, rotations
from catan_ai.engine import ActionKind

DEFAULT_BOTS = ("catanatron:R", "catanatron:W", "catanatron:VP", "catanatron:F")


def probe(bot: str, boards: int, seed: int) -> dict[str, Any]:
    lineup = [Contestant("random-trader")] + [Contestant(bot)] * 3
    counts: Counter[str] = Counter()
    for board in range(boards):
        for r, seating in enumerate(rotations(lineup)):
            result, referee = play_match(f"p{board}-{r}", seed + board, seating)
            counts["matches"] += 1
            counts["errors"] += result.error is not None
            trader = next(s for s, c in enumerate(seating) if c.name == "random-trader")
            for event in referee.events:
                kind = event.event_type
                if kind == ActionKind.OFFER_TRADE:
                    counts["offers" if event.actor == trader else "bot_offers"] += 1
                elif kind == ActionKind.ACCEPT_TRADE and event.actor != trader:
                    counts["bot_accepts"] += 1
                elif kind == ActionKind.REJECT_TRADE and event.actor != trader:
                    counts["bot_rejects"] += 1
                elif kind == ActionKind.CONFIRM_TRADE:
                    counts["completed_trades"] += 1
    responses = counts["bot_accepts"] + counts["bot_rejects"]
    return {
        **{k: counts[k] for k in ("matches", "errors", "offers", "bot_offers")},
        **{k: counts[k] for k in ("bot_accepts", "bot_rejects", "completed_trades")},
        "accept_rate": round(counts["bot_accepts"] / responses, 3) if responses else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bots", default=",".join(DEFAULT_BOTS))
    parser.add_argument("--boards", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    report = {bot: probe(bot, args.boards, args.seed) for bot in args.bots.split(",")}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
