# Architecture Decision Records

Each entry records date, context, decision, alternatives and evidence (master plan §17). Entries are append-only; a reversed decision gets a new entry that supersedes the old one.

---

## ADR-001: Standalone repository with Catanatron as a pinned submodule

**Date:** 2026-10-28  
**Status:** Accepted

**Context:** The project folder sat inside an unrelated parent repository (`Documents/Coding`). The engine must be pinned to an exact commit (§1.3), and any engine patches must be isolated (§1.3 go/no-go).

**Decision:** `catan/` is its own GitHub repository, https://github.com/imnotdeepak/catan_ai. Catanatron is a git submodule at `vendor/catanatron`, installed as an editable path dependency through uv.

**Pinned engine:**

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

## Release timeboxes (§13.1)

| Release | Started | Timebox | Status |
|---|---|---|---|
| 0.1 Rules + Arena | 2026-10-28 | 6 weeks, target 2026-12-09; cut to exit gate if unfinished by 2026-12-30 (+50%) | In progress |

Release 0.1 timebox assumes roughly 10–15 hours/week. Estimate by task: ENG-003 3–4 days; ENG-004 ~1 week; ENG-005/006 1.5–2 weeks; ENG-007 1–2 days; ENG-008 3–5 days; ENG-009 1–2 days; ENG-010 2–3 days; ENG-011 1–2 days.
