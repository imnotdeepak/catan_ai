"""Time search budgets: ms per searched decision and per decision, [search, plain, F, F]."""
import sys
import time
from multiprocessing import Pool

from catan_ai.arena.contestants import Contestant
from catan_ai.arena.match import play_match

C = "champions/robber-ppo3-r60-imitation-value.pt"
M = "champions/robber-ppo3-r60.pt"
GAMES = 4


def run(budget: str) -> str:
    seating = [Contestant(f"search:{budget}:model:{C}"), Contestant(f"model:{M}"),
               Contestant("catanatron:F"), Contestant("catanatron:F")]
    dec = srch = 0
    sec = wall = 0.0
    for g in range(GAMES):
        sink: dict = {}
        t = time.perf_counter()
        play_match(f"t{budget}-{g}", 700000 + g, seating, agent_sink=sink)
        wall += time.perf_counter() - t
        a = sink[0]
        dec += a.stats["decisions"]
        srch += a.stats["searches"]
        sec += a.search_seconds
    return (f"{budget}: {srch}/{dec} searched ({srch / dec:.0%}), "
            f"{1000 * sec / srch:.0f} ms/searched, {1000 * sec / dec:.0f} ms/decision avg, "
            f"{wall / GAMES:.0f} s/game")


if __name__ == "__main__":
    budgets = sys.argv[1:]
    with Pool(len(budgets)) as pool:
        for line in pool.map(run, budgets):
            print(line, flush=True)
