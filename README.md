# Robber

**Robber** (repository and package `catan_ai`) is a research AI for **standard four-player base Catan**, built on a pinned [Catanatron](https://github.com/bcollazo/catanatron) engine. The agent may only use information a real player could legally know.

The design, release plan and backlog live in [CATAN_AI_MASTER_PLAN.md](CATAN_AI_MASTER_PLAN.md). Decisions are logged in [docs/architecture-decisions.md](docs/architecture-decisions.md).

**Status:** Release 0.6 (search) is complete. With belief-world root search and a calibrated value tower, Robber takes 63.8% of the head-to-head wins against its own plain policy at a table with two ValueFunction bots, and wins the strong ladder tier with 51.5% of games against AlphaBeta's 28.0%, still seeing only what a player could legally know (measured with the imitation champion's value tower, `champions/robber-ppo3-r60-imitation-value.pt`; the retrained `champions/robber-ppo3-r60-value-ppo3.pt` is level or slightly ahead; `reports/release_0.6.md`, `reports/evaluation_0.6_search.md`). In 1,000 strong-tier games on fresh boards it wins 48.3% against AlphaBeta's 27.0%, 64.1% of the games either won (`reports/evaluation_ab_search.md`). At a table of three strong bots (AlphaBeta depths 3 and 2, ValueFunction) it is level with AlphaBeta depth 2, 28.2% to 28.7% (`reports/evaluation_deep_search.md`). Search costs about 250 ms per decision; a cheaper 2-world, 4-turn search keeps a 58% share at a third of the cost. Distilling search back into the plain policy gave no significant gain.

Release 0.5 (league reinforcement learning) is complete. The champion is `champions/robber-ppo3-r60.pt`: the imitation network improved by three masked PPO runs, the last against a full league with trading on (55.7% of the head-to-head wins against the previous champion when both trade). Its predecessor `champions/robber-ppo2-r60.pt` was built by two masked PPO runs against a league of built-in bots, the imitation champion and its own snapshots. Each run's champion beat the previous one head to head (63.7%, then 61.6% of the pair's wins). On the ladder `robber-ppo2-r60` ranked first in every tier: 96.0% against the weak bots, 56.5% with ValueFunction (which wins 43.2%) and 36.5% in the strong tier (AlphaBeta 34.0%, ValueFunction 28.5%), seeing only what a player could legally know. The current champion is level or ahead in every tier: 97.8%, 56.8% and 40.5% (AlphaBeta 35.5%; `reports/evaluation_0.5_ppo3.md`). In 1,000 strong-tier games it wins 38.5% against AlphaBeta's 33.4% (`reports/release_0.5.md`, `reports/evaluation_ab.md`).

Release 0.4 added beliefs (opponent hands tracked from the public record, never ruling out the true hands), a Bayesian opponent model and trading with no invalid transfers. With trading, the champion takes 90.5% of the head-to-head wins against the plain 0.3 champion at a table where others trade (`reports/evaluation_0.4.md`).

Earlier: Release 0.3 (openings and victory) is complete. The champion is `champions/imitation-features-v3.pt`: the imitation network given board-analysis and opening-prior features. It beats the 0.2 champion in 63.1% of their head-to-head decided games (two training seeds, `reports/openings.md`). On the 0.2 ladder it wins 91.5% against the weak bots (0.2: 86.8%), 35.5% head to head with ValueFunction (0.2: 23.2%) and 16.5% in the strong tier (0.2: 14.0%) (`reports/evaluation_0.3.md`). Release 0.3 also added exact victory accounting, a route-to-10 planner and a turn-order threat model with defensive robber play (`reports/victory.md`); neither the planner's outputs as features nor the defense measurably improved play.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```powershell
git clone --recurse-submodules https://github.com/imnotdeepak/catan_ai.git
cd catan_ai
uv sync
```

If you cloned without `--recurse-submodules`, run `git submodule update --init` before `uv sync`.

## Checks

```powershell
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest
```

## Learned baseline

```powershell
uv run python -m catan_ai.cli.imitation_data --workers 11   # dataset -> data/<name>/ (~70 min)
uv run python -m catan_ai.cli.train_imitation               # training -> checkpoints/<name>/ (~13 min, CPU)
uv run python -m catan_ai.cli.evaluate --workers 11         # ladder evaluation -> reports/evaluation_0.2.md
```

Any arena lineup can seat `model:champions/imitation-v1.pt` or `heuristic`.

## Layout

| Path | Contents |
|---|---|
| `src/catan_ai/` | Project package |
| `tests/` | Test suite |
| `vendor/catanatron/` | Catanatron engine: our fork with rule fixes, pinned as a git submodule (GPL-3.0-or-later; see ADR-005) |
| `champions/` | Promoted model checkpoints (small, committed) |
| `configs/` | Ladder, imitation and evaluation configs |
| `docs/` | Rules matrix, audits, decision log |
| `reports/` | Generated reports (ladder, baseline, training, evaluation) |
