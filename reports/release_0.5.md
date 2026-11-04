# Release 0.5: League RL

Release 0.5, plan §9.2 phase D and §9.3 (ML-029). Exit gate (plan §12): a statistically credible improvement over imitation-only play and over the previous champion, across held-out evaluations. It is met: each PPO champion beats the one before it head to head, and the final champion ranks first in every ladder tier on boards disjoint from training.

## Method

`catan_ai.training.ppo` and `catan_ai.cli.train_ppo`.

- **Rollouts:**
  - Worker processes play full four-player games.
  - The learner holds one or two random seats and samples from its masked policy over the legal candidates.
  - The other seats come from a league: the imitation champion (35%), `catanatron:F` (35%), `catanatron:W` (5%), `catanatron:VP` (5%) and up to eight snapshots of the learner (20% between them).
- **Reward:**
  - 1 for the winner, 0 for everyone else.
  - Truncated games are dropped, never scored as losses.
- **Value and advantages:**
  - The value is the value head's P(this seat wins).
  - Advantages are GAE over the seat's own decisions, with gamma 0.999 and lambda 0.95.
- **Update:**
  - Clipped surrogate with clip 0.2, 4 epochs and minibatches of 256.
  - Value cross-entropy on the actual winner.
  - Entropy bonus.
  - A KL anchor (0.02) to the starting policy.
- **Scale:** 220 games per iteration, about 1 minute per iteration on 10 workers.

| Run | Start | Learning rate | Entropy bonus | Iterations | Champion |
|---|---|---|---|---|---|
| `ppo-1` | imitation champion (`imitation-features-v3`) | 1e-4 | 0.01 | 50 (stopped; entropy was rising) | round 30, `champions/robber-ppo1-r30.pt` |
| `ppo-2` | `robber-ppo1-r30` | 1e-4 | 0.005 | 60 | round 60, `champions/robber-ppo2-r60.pt` |

With the entropy bonus at 0.01, `ppo-1`'s policy entropy rose from 0.79 to 0.89 and its training win rate fell after round 30. Halving the bonus in `ppo-2` kept entropy at 0.78 to 0.83.

## Head to head

Each comparison: candidate and baseline with two `catanatron:F`, 100 boards from seed 500000 (disjoint from the training seeds 900000+), eight seatings each, 800 games. Share is the candidate's share of the games either of them won.

| Candidate | Baseline | Candidate share | 95% interval (boards resampled) | Report |
|---|---|---|---|---|
| `ppo-1` round 20 | imitation champion | 57.1% | 51.1% to 62.7% | `ablations/ppo-1-snap-20-vs-F.md` |
| **`ppo-1` round 30** | imitation champion | **63.7%** | 58.5% to 68.9% | `ablations/ppo-1-snap-30-vs-F.md` |
| `ppo-1` round 50 | imitation champion | 61.0% | 55.3% to 66.7% | `ablations/ppo-1-snap-50-vs-F.md` |
| `ppo-2` round 30 | `robber-ppo1-r30` | 62.0% | 57.1% to 66.8% | `ablations/ppo-2-snap-30-vs-robber.md` |
| **`ppo-2` round 60** | `robber-ppo1-r30` | **61.6%** | 56.8% to 66.6% | `ablations/ppo-2-snap-60-vs-robber.md` |

## Ladder

The same tiers and board seeds as the Release 0.3 evaluation (`reports/evaluation_0.3.md`). Win rate per seat; `catanatron:` bots read the full game state (privileged).

| Tier | 0.3 imitation champion | `robber-ppo1-r30` | **`robber-ppo2-r60`** | Rank |
|---|---|---|---|---|
| weak (W, VP, R) | 91.5% | 95.0% | **96.0%** | 1st |
| value-function (F, W, VP) | 35.5% | 40.5% | **56.5%** | 1st (F 43.2%) |
| strong (AB:2, F, W) | 16.5% | 27.0% | **36.5%** | 1st (AB:2 34.0%, F 28.5%) |

Reports: `evaluation_0.5.md` (`robber-ppo1-r30`) and `evaluation_0.5_ppo2.md` (`robber-ppo2-r60`). 0 errors and 0 truncations in all evaluation games.

## Caveats

- **Margins over AlphaBeta:** in the strong tier, Robber's 36.5% against AlphaBeta's 34.0% is within the intervals (200 seat-games each). The plan's 1.0 gate, beating AlphaBeta on the fair four-player protocol, needs a larger, dedicated evaluation (EVAL-032).
- **League scope:** the league has no trading negotiators or opening archetypes yet (plan §9.3), and the champion is the base policy without the trading wrapper. Training with trading partners is the next step.
- **Snapshot choice:** snapshots are compared against `catanatron:F` line-ups, the same opponents as much of the league. The ladder tiers use other seeds and line-ups, and they agree.
