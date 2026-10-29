# Victory: Accounting, Route Planner and Defense (STR-019, STR-020, STR-021)

Release 0.3, plan §5. Three layers, each built on the one before:

1. **Accounting** (`catan_ai.victory.score`, STR-019): points, pieces, award thresholds and development-card timing from a player's observation. `tests/victory/test_score.py` checks it against the engine's private state at every decision of random games.
2. **Route planner** (`catan_ai.victory.planner`, STR-020): for every seat, a route to 10 points.
3. **Threats and defense** (`catan_ai.victory.threats`, STR-021): who wins first in turn order, and robber moves that delay them.

## Route planner (STR-020)

A route is a sequence of objectives. Each objective is an AND of prerequisites: a legal location, the pieces, the roads to get there, and the resources. Resources come by an OR of production and maritime trade at the seat's best rates.

- **Objectives:**
  - a city on an own settlement;
  - a settlement on a buildable site within three roads, with the roads' cost;
  - a development card, worth the chance it is a VP card;
  - Largest Army and Longest Road.
- **Insecure points:** award points and development-card points are marked insecure, since awards transfer and draws are uncertain.
- **Building the route:** the planner repeatedly takes the objective with the most points per expected round, given the income the route has built so far. Paying uses `pay`, which covers shortfalls by trading surpluses.
- **Contingencies:** a site an opponent can reach at least as fast is *contested*, and carries the best uncontested site as its fallback.
- **Information:** the planner sees only the viewer's observation. An opponent's hand is its card count spread by its income. An exact ledger is BEL-022, in Release 0.4.

`tests/victory/test_planner.py` checks that every route respects pieces, locations, road budgets and time order, and that routes reaching 10 add up. It also covers the "robot can win now" fixture: at 9 points with city resources in hand, the first step is the city and the route takes 0 rounds. A full plan for four seats takes about 19 ms.

### Planner outputs as model features (features v4): no reliable gain

Features v4 adds, per seat, the planner's time to win and point range, plus per action whether it advances the viewer's route. The `imitation-v3` replays were re-encoded with these features, the network was trained with two seeds, and each model played 4,000 games (500 boards) head to head. Each comparison pits the v4 model against a v3 model trained with the same seed, which is the cleanest test of the features alone.

| Training seed | v4 share of the pair's wins vs v3, same seed | 95% interval (boards resampled) | Report |
|---|---|---|---|
| 0 (v3 = the champion) | 45.9% | 44.6% to 47.2% | `reports/ablations/features-v4.md` |
| 1 | 52.9% | 51.6% to 54.2% | `reports/ablations/features-v4-seed1-matched.md` |

The v4 seed 1 model also beats the champion, 54.5% (`reports/ablations/features-v4-seed1.md`).

The two seeds point in opposite directions, so the features do not reliably help; the training seed matters more than they do. They do improve winner prediction: the dev-set value loss is 0.983 and 0.991, against 1.017 and 1.025 for v3. They do not improve move prediction: policy top-1 is 69.6% and 74.1%, against 69.7% and 72.6%. The champion stays `champions/imitation-features-v3.pt`. The planner is used directly instead, by the defense below.

## Threats and defense (STR-021)

- **Turn order:** a seat wins only on its own turn. A route of *r* rounds therefore wins `turns_until(seat) + 4·ceil(r)` turns from now. With equal routes, the seat that acts sooner wins. A slightly faster route that still needs another round does not overtake a seat whose turn comes first.
- **Hidden VP cards:** each opponent has a likely route, which starts from its visible points, and a worst case, which counts every face-down card it could hold as a VP card.
- **Urgency:** delaying an opponent is weighted 1 for the one that threatens to win first in the worst case. The weight halves for every two rounds later an opponent would finish.
- **Robber moves:** without domestic trades (Release 0.4), the robber is the main defensive lever. A robber move scores the weighted rounds it adds to each opponent's route, minus the rounds it adds to the viewer's own route. Stealing adds about one card's worth of the victim's income. Only tiles with buildings are re-planned, since a tile without buildings blocks no income. A decision takes 0.1–0.3 s.
- **Contestant:** `defend:<agent>` moves the robber this way and leaves every other decision to `<agent>`.

`tests/victory/test_threats.py` covers:
- turn-order arithmetic and races;
- the hidden-VP threat: an opponent at 7 visible points with 2 hidden VP cards threatens first in the worst case only;
- the 9-VP-opponent fixture: the defender robs the near-winner, on a tile where it has buildings;
- blocking only the defender's own tiles never scores positive.

### Measured: no detectable gain

The champion with `defend:` robber moves played the plain champion, 4,000 games (500 boards) per comparison:

| Other two seats | Defender's share of the pair's wins | 95% interval (boards resampled) | p | Report |
|---|---|---|---|---|
| WeightedRandom (`catanatron:W`), VictoryPoint (`catanatron:VP`) | 51.3% | 50.0% to 52.5% | 0.11 | `reports/ablations/defend-robber.md` |
| ValueFunction (`catanatron:F`) ×2 | 50.3% | 47.8% to 52.9% | 0.85 | `reports/ablations/defend-robber-vs-F.md` |

Neither result is significant. Two reasons are likely:

- The robber moves a few times a game, so even a better choice changes few outcomes.
- The champion's own robber choices, imitated from F, already tend to hit the leader.

The defense stays available as a contestant. The threat model is ready for Release 0.4, where trade refusal ("do not enable a near-winner") is the next defensive lever.

## Release 0.3 exit gate

| Criterion | Evidence |
|---|---|
| End-to-end legal openings | Opening rules, pair search and the champion played tens of thousands of arena games with 0 errors (`reports/ablations/`) |
| Valid scoring routes | `tests/victory/test_planner.py`, `tests/victory/test_score.py` |
| Statistically supported improvement over the 0.2 learned baseline | Features v3 beat the 0.2 champion with 63.1% of the pair's wins, under two training seeds (`reports/openings.md`); ladder in `reports/evaluation_0.3.md` |
