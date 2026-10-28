# catan_ai

Research AI for **standard four-player base Catan**, built on a pinned [Catanatron](https://github.com/bcollazo/catanatron) engine. The agent may only use information a real player could legally know.

The design, release plan and backlog live in [CATAN_AI_MASTER_PLAN.md](CATAN_AI_MASTER_PLAN.md). Decisions are logged in [docs/architecture-decisions.md](docs/architecture-decisions.md).

**Status:** Release 0.1 (Rules + Arena) complete: rules-audited engine fork, observation firewall, seat-rotated arena with exact replay, throughput benchmark, trade audit and reference ladder. Next: Release 0.2 (learning baseline). No learned agent yet.

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

## Layout

| Path | Contents |
|---|---|
| `src/catan_ai/` | Project package |
| `tests/` | Test suite |
| `vendor/catanatron/` | Catanatron engine: our fork with rule fixes, pinned as a git submodule (GPL-3.0-or-later; see ADR-005) |
| `docs/` | Rules matrix, audits, decision log |
