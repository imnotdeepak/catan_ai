# Architecture Decision Records

Each entry records date, context, decision, alternatives and evidence (master plan §17). Entries are append-only; a reversed decision gets a new entry that supersedes the old one.

---

## ADR-001: Standalone repository with Catanatron as a pinned submodule

**Date:** 2026-10-28  
**Status:** Accepted

**Context:** The project folder sat inside an unrelated parent repository (`Documents/Coding`). The engine must be pinned to an exact commit (§1.3), and any engine patches must be isolated (§1.3 go/no-go).

**Decision:** `catan/` is its own GitHub repository, https://github.com/imnotdeepak/catan_ai. Catanatron is a git submodule at `vendor/catanatron`, installed as an editable path dependency through uv. (Amended by ADR-005: the submodule now tracks the fork, with the upstream commit below as its base.)

**Pinned upstream base:**

| Field | Value |
|---|---|
| Repository | https://github.com/bcollazo/catanatron |
| Commit | `ecf931181b9a65bb4116a2153fb78c16f1438e00` |
| Commit date | 2026-09-08 |
| Commit subject | Rework custom bots: shared registry, one lifecycle, bots in any language (#386) |
| Package version | 3.3.0 |
| License | GPL-3.0-or-later |
| Python requirement | >=3.11 |
| Runtime dependencies | networkx, click, rich |

Only the core package is installed. The `gym` extra (gymnasium<=0.29.1, numpy, pandas, fastparquet, pygame) is deferred until it is needed.

**Alternatives:** a plain git-ignored clone (commit not recorded in history); a pinned VCS dependency in `pyproject.toml` (harder to patch and inspect locally).

**Consequences:** fresh clones need `--recurse-submodules`. Engine patches, if required, go on a fork referenced by the submodule, behind the adapter (ENG-004).

---

## ADR-002: Tooling

**Date:** 2026-10-28  
**Status:** Accepted

**Decision:**

- **Environment and lockfile:** uv, with `uv.lock` committed (§12 "choose one, pin").
- **Python:** minimum 3.11 (Catanatron's floor); development on 3.12; CI tests both.
- **Format and lint:** ruff (`ruff format`, `ruff check`), line length 100.
- **Types:** mypy in strict mode over `src/` and `tests/`. Catanatron has no type information, so its imports are untyped.
- **Tests:** pytest.
- **CI:** GitHub Actions on push to `main` and on pull requests: format, lint, type check, tests.

**Alternatives:** requirements.lock with pip-tools; black + flake8 (ruff covers both, faster); pyright (mypy chosen for its strictness and CLI use in CI).

---

## ADR-003: Project license (open)

**Date:** 2026-10-28  
**Status:** Open, owner decision

**Context:** The project imports and may patch Catanatron, which is GPL-3.0-or-later (§1.3 item 7). Distributing a derivative work carries GPL obligations; Catan trademarks are a separate question.

**Decision:** No license file yet. The repository owner chooses one before any public release or distribution. A GPL-compatible license is the expected default if Catanatron code is distributed with the project.

---

## ADR-004: Baseline rules edition is the 2025 6th edition

**Date:** 2026-10-28  
**Status:** Accepted (supersedes the plan's provisional "expect pre-2025 rules")

**Context:** The plan (§1.1) said to pin whichever official edition Catanatron most closely implements, to be identified in ENG-003.

**Evidence:** Catanatron allows trading and building in any order within one phase after rolling, and allows dev cards before rolling. That matches the 2025 6th-edition rulebook (CN3081 v6.250401, p.6–7), which defines a single Action phase. The plan notes earlier editions documented separate trade and build phases. Every other mechanic checked matches the 6th edition apart from the deviations D-01 to D-08 in `docs/rules-matrix.md`. No earlier rulebook was audited clause by clause.

**Decision:** The baseline is the 2025 6th edition. There is no separate migration target. Deviations are patched behind the engine adapter or in a minimal fork, each guarded by a strict-xfail test that flips when fixed.

**Consequences:** Terminology follows the 6th edition where it differs (Longest Route, Invention), while code keeps Catanatron's names (`ROAD`, `YEAR_OF_PLENTY`).

---

## ADR-005: Patch engine rule deviations in a fork

**Date:** 2026-10-28  
**Status:** Accepted (amends ADR-001: the submodule now tracks the fork)

**Context:** ENG-003 found rule deviations inside engine internals (`yield_resources`, `Board` Longest Road logic, the trade response loop) that an outside adapter cannot fix cleanly.

**Options considered:**

- **Fork (chosen):** fixes are ordinary, reviewable commits; the engine is correct for every entry point (built-in bots, `catanatron-play`, our adapter); rebasing onto upstream surfaces conflicts explicitly; fixes can be offered upstream.
- **Runtime monkeypatching:** no second repository, but fixes are invisible in the engine source, entry points that skip our package get the bugs back, and upstream refactors can silently stop a patch applying.
- **Upstream only, wait for merge:** no maintenance, but benchmarks would run on wrong rules until merged, which §1.3 forbids.

**Decision:** The submodule points at https://github.com/imnotdeepak/catanatron, branch `catan-ai-fixes`. The `upstream` remote in the submodule is https://github.com/bcollazo/catanatron. Each fix is one commit with tests, and the upstream test suite must pass at every commit.

| Fork commit | Fix | Matrix row |
|---|---|---|
| `a36ac1d` | Single affected player takes the remaining supply on shortage | D-03 |
| `adbb0ed` | Longest Road after a cut: ≥5, unique leader or holder, else supply; VP removed on return | D-04, D-05 |
| `8e44f47` | Offerer not asked about its own offer; affordability checked at offer and confirm | D-06, D-07 |
| `d252ad8` | Only the active player can win | D-02 |
| `52213b3` | AlphaBeta treats domestic trade steps as deterministic instead of crashing | H-11 (ENG-010) |
| `25f3599` | Sort sets of colors and resources in move generation and AlphaBeta, so seeded games replay identically in any process | H-12 (ENG-011) |

D-01 (shared RNG on copy) stays in the adapter (ENG-007): upstream shares the RNG deliberately, and changing it would alter its search bots' behavior.

**Upstream test baseline (Windows, core suite):** 248 pass. `tests/test_stdio_bot.py` (12 tests) fails at the unmodified upstream commit too, because shell-style splitting strips Windows path backslashes; gym, web and ML suites need extras not installed. Run with `--basetemp` inside the scratch area to avoid a temp-dir permission error.

**Consequences:** upgrading Catanatron means rebasing `catan-ai-fixes` onto a newer upstream commit and re-running both test suites. The fixes are offered upstream as bcollazo/catanatron #395 (bank shortage), #396 (Longest Road after a cut), #397 (trade responses), #398 (only the active player wins), #399 (AlphaBeta with trade actions) and #400 (reproducible seeded games), opened 2026-10-28. If all are merged, the submodule can return to upstream.

---

## ADR-006: Search and rollout budgets come from measured throughput

**Date:** 2026-10-28  
**Status:** Accepted

**Context:** The plan's budgets (§4.4 opening rollouts 32–256 per finalist, §8.3 trade simulations 64–1024, §10.3 search 64/256/1024) were placeholders until ENG-009 measured the simulator.

**Evidence** (`reports/throughput.md`, one core of a 12-core AMD desktop):

- Bare Catanatron: ~35,600 decisions/s. Our firewalled path (events + observation per decision): ~5,600 decisions/s, a 6.4x overhead.
- Random playouts to the end of the game from a branched midgame: ~34/s, about 680 decisions each (random play is slow to finish; games last ~330 turns).
- `EngineAdapter.branch()`: ~0.1 ms. `Referee.observe()`: ~0.1 ms.
- Built-in bots per decision: F ~2 ms, AB:2 ~45 ms, M:50 ~0.6 s, G:10 ~2.3 s.

**Implications:** at one core, 256 random playouts cost ~7.5 s and 1,024 cost ~30 s, so full-length random rollouts cannot meet the plan's budgets at interactive speed.

**Decision:** budgets are derived from `configs/budgets.json`, regenerated by `python -m catan_ai.cli.throughput`. Search work (SRCH-030/031) must use, in order of expected payoff:

1. **Truncated rollouts** that stop after a fixed horizon and score the position with the value model (plan §3) instead of playing to the end.
2. **Parallel workers**, up to one per core, using the existing seeded branches.
3. **An engine-direct rollout path** that skips per-step observation building and legality re-checks inside simulations (the bare engine is ~1.5x faster per decision than the adapter path, and 6.4x faster than the firewalled path).
4. **Stronger rollout policies**, which finish games in far fewer decisions than random play.

Training and arena runs use the firewalled path; only search internals may use the faster paths, and only on branched or belief-sampled worlds.

---

## ADR-007: Learning stack for the Release 0.2 baseline

**Date:** 2026-10-28  
**Status:** Accepted

**Context:** Release 0.2 needs a trainable policy/value model, a frozen imitation dataset, and a checkpoint the arena can play. The development machine has no GPU (12-core CPU, 16 GB RAM).

**Decision:**

- **PyTorch, CPU wheels**, from the `pytorch-cpu` index pinned in `pyproject.toml`. CI on Linux would otherwise download the CUDA build. A ~850k-parameter MLP trains in about 13 minutes on CPU and decides in under 1 ms per move, so a GPU is not needed at this scale.
- **Features** (`src/catan_ai/models/features.py`, version 1) take only a `PlayerObservation`. Seats are relative to the viewer, so one network plays every seat. Board topology comes from Catanatron's static base map; tests check it matches every board.
- **Action-conditioned policy:** a shared scorer f(state, action) over the engine's legal candidates, with a masked softmax, never a fixed action index. `PolicyAgent` falls back to the deterministic `HeuristicPolicyValue` if the model fails, so every phase always has a legal move.
- **Dataset storage: compressed NumPy `.npz`, not Parquet** (plan §9.4). Samples are dense fixed-width arrays plus variable-length candidate sets in CSR form (`offsets`), which `.npz` holds without another dependency. Privileged labels are in their own file (`privileged.npz`). `manifest.json` records the config, provenance and sha256 of every file. Loading verifies the hashes, and a dataset directory is never overwritten.
- **Two towers.** The policy tower and the outcome tower (value, final-points auxiliary) share no weights, and each is early-stopped on its own dev loss. Outcome labels are constant within a game, so a shared trunk learns to recognize training boards, which damages both heads (`reports/imitation_training.md`, design notes).
- **Many games, few samples each.** The dataset keeps each teacher decision with probability 0.15 across 7,400 games instead of every decision of a few hundred games.
- **Champions are committed** under `champions/` (about 3.4 MB each), so anyone can play or evaluate them without retraining. Datasets and working checkpoints stay out of git (`data/`, `checkpoints/`).
- **Splits are disjoint board-seed ranges** (train 100000+, dev 200000+, holdout 300000+, evaluation 400000+), which the builder checks.

**Alternatives:** Parquet via pyarrow (columnar, but awkward for nested candidate sets and an extra 100 MB dependency); storing raw observations and encoding during training (smaller files, but re-encoding every epoch costs more than the training itself); a fixed flat action space (cannot represent trade offers or grow with later candidate generators).

**Consequences:** changing the encoder bumps `FEATURE_VERSION`. Old datasets and checkpoints then refuse to load instead of silently mismatching. Dataset size is bounded by RAM: about 7 KB per sample in float16.

---

## Release timeboxes (§13.1)

| Release | Started | Timebox | Status |
|---|---|---|---|
| 0.1 Rules + Arena | 2026-10-28 | 6 weeks, target 2026-12-09; cut to exit gate if unfinished by 2026-12-30 (+50%) | **Complete 2026-10-28**: exit gate passed (`reports/baseline_report.md`), ladder in `reports/ladder.md` |
| 0.2 Learning baseline | 2026-10-28 | 4 weeks, target 2026-11-25; cut to exit gate if unfinished by 2026-12-09 (+50%) | **Complete 2026-10-28**: champion `champions/imitation-v1.pt`, `reports/imitation_training.md`, `reports/evaluation_0.2.md` |
| 0.3 Openings + Victory | 2026-10-28 | 6 weeks, target 2026-12-09; cut to exit gate if unfinished by 2026-12-30 (+50%) | In progress: STR-016 done (`catan_ai.opening.board`, features v2); STR-017 done (`reports/openings.md`: the `balanced` prior beats the pip baseline and the champion's own opening); prior scores as features v3 beat the 0.2 champion (63.1%, two training seeds), promoted to `champions/imitation-features-v3.pt`; STR-018 done (pair search measured, no gain over greedy `balanced` against near-random placers) |

Release 0.1 timebox assumes roughly 10–15 hours/week. Estimate by task: ENG-003 3–4 days; ENG-004 ~1 week; ENG-005/006 1.5–2 weeks; ENG-007 1–2 days; ENG-008 3–5 days; ENG-009 1–2 days; ENG-010 2–3 days; ENG-011 1–2 days.

Release 0.2 estimate by task: ML-012 3–5 days; ML-013 3–5 days; ML-014 1 week; ML-015 2–3 days.

Release 0.3 estimate by task: STR-016 3–5 days; STR-017 1 week; STR-018 1–2 weeks; STR-019 3–5 days; STR-020 1.5–2 weeks; STR-021 1 week. Every component is measured as an ablation against the current champion (plan §11.2); the exit gate needs a statistically supported gain over the 0.2 baseline.
