# Release 0.6: Search + Distillation

Release 0.6, plan §10 and §9.2 phase F (SRCH-030, SRCH-031). Exit gate (plan §12): a measured improvement at fixed compute, with no information-set violations. **It is met.**

- **Gain:** at every search budget tried, search beats the same network playing without search, on held-out boards.
- **Correctness:** the root's choice depends only on the searcher's own observation, and searching never touches the real game. Both properties are tested.

With a calibrated value tower, the search champion takes 63.8% of the head-to-head wins against the plain champion at a table with two `catanatron:F`. In the strong ladder tier it wins 51.5% of games, against AlphaBeta's 28.0%.

Distilling search back into the plain policy gave no significant gain. The plain champion stays `champions/robber-ppo3-r60.pt`.

## Method

`catan_ai.search.root` (`SearchAgent`, `SearchConfig`, `should_search`); arena spec `search:<worlds>[t<turns>]:model:<checkpoint>`.

- **Which decisions are searched:** setup placements, robber moves, any decision once a seat shows 8 or more points, and decisions where the policy's top two priors are within 0.5.
  - Forced moves and trade-acceptee choices are left to the policy.
  - About 20–35% of the searcher's decisions are searched.
- **Candidates:** the policy's top 3 legal actions with prior at least 0.02.
- **Belief worlds:** each world is a full engine game sampled by `catan_ai.beliefs.worlds.WorldSampler` from the searcher's observation only: opponents' hands, development cards and the deck.
  - Root sampling with fixed root actions: every candidate is played in the same worlds, and the choice averages over them.
  - Each world's rollouts share one branch seed across candidates (common random numbers), so candidates are compared on the same dice.
- **Rollouts:** each runs on `EngineAdapter.branch`, which gives it its own RNG and a reshuffled deck. It goes through its own `Referee`, so every seat acts from its own observation of that world.
  - All seats play the greedy policy for `turns` turns.
  - The rollout stops at the searcher's next decision, which is scored by the value head's P(searcher wins). A finished game scores 1 or 0.
- **Choice:** the search leaves the policy's pick only when a candidate's mean value is more than 0.02 higher.

### Information-set safety (`tests/search/test_root.py`)

- **Searching does not touch the real game:** after a search, the next dice, every hand, the development deck and the event log are unchanged.
- **The root choice depends only on the observation:** scrambling everything the searcher cannot see, using the firewall tests' `scramble_hidden_state`, leaves the scores and the choice identical.
- **Rollouts are reproducible** and leave their world unchanged; a finished rollout scores exactly 0 or 1.
- **Search agents are deterministic and always legal**, with no failures over hundreds of searched decisions.
- **Evaluation:** 0 errors and 0 truncations in every evaluation game.

## Results

Head-to-head comparisons use 100 boards from seed 500000, eight seatings each, 800 games. Share is the candidate's share of the games either it or the baseline won. "W/VP" means opponents `catanatron:W` and `catanatron:VP`; "F" means two `catanatron:F`.

### Search against plain play (SRCH-030)

Policy and value from `robber-ppo3-r60`, against the same checkpoint without search.

| Budget | Opponents | Share | 95% interval (boards resampled) | p | Report |
|---|---|---|---|---|---|
| 8 worlds × 4 turns | W/VP | 54.5% | 51.3% to 57.5% | 0.013 | `ablations/search-8-vs-plain.md` |
| 8 worlds × 4 turns | F | 54.2% | 49.8% to 58.7% | 0.096 | `ablations/search-8-vs-plain-F.md` |
| 4 worlds × 8 turns | W/VP | 55.2% | 52.1% to 58.4% | 0.003 | `ablations/search-4t8-vs-plain.md` |
| 16 worlds × 2 turns | W/VP | 55.3% | 51.9% to 58.6% | 0.003 | `ablations/search-16t2-vs-plain.md` |
| 16 worlds × 4 turns (double) | W/VP | 53.8% | 50.6% to 57.1% | 0.033 | `ablations/search-16-vs-plain.md` |

- **Doubling the budget did not help.** Among the three budgets that imagine the same number of turns, 4 × 8 was the fastest to run.
- **Overconfident value head:** the PPO value head is badly overconfident, with an expected calibration error (ECE) of 0.30 on held-out ppo-3 play, against 0.07 for the imitation champion's head. Rollout leaves are scored by this head, so its errors become search's errors.

### Calibrated leaf values

`champions/robber-ppo3-r60-imitation-value.pt` joins `robber-ppo3-r60`'s policy tower with the outcome tower of the imitation champion `imitation-features-v3`. The two towers share no weights, so plain play is unchanged.

| Candidate | Baseline | Opponents | Share | 95% interval | p | Report |
|---|---|---|---|---|---|---|
| search 4×8, imitation values | search 4×8, ppo-3 values | W/VP | **55.4%** | 52.4% to 58.5% | 0.003 | `ablations/search-4t8-imitation-value-vs-ppo-value.md` |
| search 4×8, imitation values | plain `robber-ppo3-r60` | W/VP | **58.8%** | 55.4% to 62.1% | < 0.0001 | `ablations/search-4t8-imitation-value-vs-plain.md` |
| search 4×8, imitation values | plain `robber-ppo3-r60` | F | **63.8%** | 59.5% to 68.1% | < 0.0001 | `ablations/search-4t8-imitation-value-vs-plain-F.md` |

Against F, search with calibrated values won 37.6% of all games; plain Robber won 21.4% and each F bot about 20.5%.

### Ladder

Same tiers and board seeds as `evaluation_0.5_ppo3.md`; win rate per seat. `catanatron:` bots read the full game state (privileged); search Robber sees only its own observation. Report: `evaluation_0.6_search.md`.

| Tier | Plain `robber-ppo3-r60` | **Search 4×8, imitation values** | Rank |
|---|---|---|---|
| weak (W, VP, R) | 97.8% | **98.2%** | 1st |
| value-function (F, W, VP) | 56.8% (F 43.0%) | **64.5%** (F 35.5%) | 1st |
| strong (AB:2, F, W) | 40.5% (AB:2 35.5%, F 23.5%) | **51.5%** (AB:2 28.0%, F 20.0%) | 1st |

In the strong tier, search Robber spends 268 ms per decision on average, and AlphaBeta 118 ms. The strong tier is 200 games, so its interval is about ±7 points.

### 1,000 games against AlphaBeta

`evaluation_ab_search.md`: search 4×8 with the value-ppo3 tower in the strong tier (AB:2, F, W) on the 250 boards (seed 440000, four rotations) of the Release 0.5 AlphaBeta evaluation.

| Bot | Search Robber's table | Plain `robber-ppo2-r60`'s table (`evaluation_ab.md`) |
|---|---|---|
| Robber | **48.3%** (45.2% to 51.4%) | 38.5% |
| `catanatron:AB:2` | 27.0% | 33.4% |
| `catanatron:F` | 24.7% | 28.0% |

Against AlphaBeta directly, search Robber wins **64.1%** of the games either of them won (483 to 270; p < 1e-12), against plain Robber's 53.5% (p of about 0.06).

### Budget against strength (SRCH-031)

The table below uses calibrated values and F opponents, with each budget playing the plain champion. Times are rough: 4 games per budget, timed in parallel processes (`runs/logs/time_budgets.py`). Report: `ablations/search-budget-cost.md`.

| Budget | Share | ms per searched decision |
|---|---|---|
| 4 × 8 | **63.8%** | 617 |
| 2 × 8 | 59.5% | 355 |
| 2 × 4 | 58.4% | 208 |
| 4 × 2 | 57.1% | 347 |
| 1 × 8 | 56.6% | 186 |

- **Every budget is significantly better than plain play** (p ≤ 0.006). Plain play costs about 2 ms per decision.
- **Cost:** each searched decision costs about worlds × (30 ms + 18 ms per turn).
- **Strength still rises at 4 × 8.** 2 × 4 is the cheap choice: about a third of the cost, keeping a 58% share.

### Value tower retrained on Robber's own play

`configs/value_ppo3.json` and `reports/value_ppo3_training.md`.

- **Data:** 4,000 games of greedy ppo-3 self-play, and of ppo-3 with two F, 102,018 samples. Only the ppo-3 seats are recorded.
- **Training:** the outcome tower only (`policy_weight` 0), started from the imitation tower. It overfit after epoch 2, which is the checkpoint kept.
- **Result:** `champions/robber-ppo3-r60-value-ppo3.pt`.

Calibration on 360 held-out games:

| Outcome tower | Value loss | Brier | ECE |
|---|---|---|---|
| ppo-3 | 2.563 | 0.314 | 0.299 |
| imitation-features-v3 | 1.198 | 0.176 | 0.074 |
| **value-ppo3** | **1.086** | **0.154** | **0.017** |

In play, search 4×8 with the value-ppo3 tower against search 4×8 with the imitation tower, F opponents, won **52.8%** (274/519; 48.9% to 56.7%; p = 0.22; `ablations/search-4t8-value-ppo3-vs-imitation-value-F.md`). The better calibration gives at most a small gain in play.

### Distillation (SRCH-031)

Search played 630 training games, 270 boards from `configs/distill.json`: 4-seat self-play, and two search seats with two F. Every decision with a choice was kept and tagged as plain, searched-and-kept or searched-and-switched; about 14% of decisions were switched.

The policy tower was fine-tuned on the search's choices, with switched decisions weighted 4×. The value and auxiliary losses were off.

| Teacher | Holdout top-1 when search switched | vs champion, W/VP | vs champion, F | Reports |
|---|---|---|---|---|
| search 4×8, ppo-3 values | 8.7% | 51.1% (p = 0.57) | 51.7% (p = 0.52) | `distillation_training.md`, `ablations/distill-4t8-vs-champion*.md` |
| search 4×8, value-ppo3 | 13.8% | 53.4% (p = 0.06) | 52.3% (p = 0.39) | `distillation_training_vppo3.md`, `ablations/distill-vppo3-4t8-vs-champion*.md` |

Both students played without search, against plain `robber-ppo3-r60`.

- **The second student is slightly ahead,** but neither match is significant. Pooled, it is 53.0% of 1,188 decided games (normal-approximation p ≈ 0.04): suggestive, not a release claim.
- **The student rarely reproduces a switch.** A switch comes from 4 sampled worlds at that one position, so as a training label it is mostly noise.

## Champions

- **Plain play:** `champions/robber-ppo3-r60.pt`, unchanged from 0.5.
- **Search:**
  - `search:4t8:model:champions/robber-ppo3-r60-value-ppo3.pt` uses the best-calibrated tower. It is level with or slightly ahead of the laddered imitation-value version.
  - `search:2t4:...` for a third of the cost.

## Caveats

- **Search is slow.** At 4 × 8 it spends about 250 ms per decision on average, against about 2 ms for plain play and about 118 ms for AlphaBeta. "Fixed compute" here means a fixed search budget compared with no search, plus budgets compared at equal cost. It is not a comparison against AlphaBeta at equal time.
- **AlphaBeta at depth 2 only.** The 1,000-game lead is over `catanatron:AB:2`; deeper AlphaBeta and the MCTS bots (`M`, `G`) are not yet tested.
- **Distillation did not transfer search's strength.** Better targets would be the search's value estimates for each candidate rather than its single choice, or many more worlds per labelled position.
- **Rollout opponents are the greedy policy.** Rollouts model every seat as Robber's own policy, never as the opponent model from Release 0.4.
