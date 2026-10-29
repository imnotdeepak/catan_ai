# catan_ai

Research AI for **standard four-player base Catan**, built on a pinned [Catanatron](https://github.com/bcollazo/catanatron) engine. The agent may only use information a real player could legally know.

The design, release plan and backlog live in [CATAN_AI_MASTER_PLAN.md](CATAN_AI_MASTER_PLAN.md). Decisions are logged in [docs/architecture-decisions.md](docs/architecture-decisions.md).

**Status:** Release 0.3 (openings and victory) in progress. The champion is now `champions/imitation-features-v3.pt`: the same imitation network given board-analysis and opening-prior features. It beats the 0.2 champion in 63.1% of their head-to-head decided games (two training seeds, `reports/openings.md`). On the 0.2 ladder it wins 91.5% against the weak bots (0.2: 86.8%), 35.5% head to head with ValueFunction (0.2: 23.2%) and 16.5% in the strong tier (0.2: 14.0%) (`reports/evaluation_0.3.md`). STR-016 to STR-018 are done; next is exact victory accounting (STR-019).

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
