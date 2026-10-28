# Domestic Trade Audit (ENG-010)

Audits Catanatron's player-to-player trading against the 2025 rulebook (p.7) and decides how Release 0.4 builds trading on it.

| Field | Value |
|---|---|
| Engine | Fork `imnotdeepak/catanatron` at `52213b3` (upstream `ecf9311` + rule fixes + AlphaBeta trade fix, ADR-005) |
| Method | Code inspection of `apply_action.py`, `game.py`, `models/actions.py`, built-in bots and the Gymnasium env; empirical probe below |
| Probe | `python -m catan_ai.cli.trade_probe --boards 5 --seed 7`: one `random-trader` (proposes one random 1-for-1 trade per turn) against three copies of a bot, all seats rotated |
| Date | 2026-10-28 |

## Lifecycle as implemented

1. **Offer.** In the Action phase after rolling, the active player submits `OFFER_TRADE` (give vector, receive vector). Offers are never listed in `playable_actions`; `is_valid_action` accepts them if the offerer holds the cards, gives and asks for something, and doesn't swap a resource for itself.
2. **Responses.** Every other player is asked in seat order: `ACCEPT_TRADE` (offered only if they can pay) or `REJECT_TRADE`.
3. **Confirmation.** If anyone accepted, the offerer chooses `CONFIRM_TRADE` with one acceptor or `CANCEL_TRADE`. Cards move only on confirmation, and both sides are re-checked. If nobody accepted, play returns to the Action phase.

Through the adapter, offers are proposed with `offer_trade(...)` and checked with `EngineAdapter.is_legal`.

## Rule comparison

| Rule (p.7) | Status | Notes |
|---|---|---|
| Only the active player trades; others trade only with them | Supported | |
| No gifts; no trading matching resources | Supported | `is_valid_trade` |
| Players can only trade cards they hold | Supported | Fixed in fork `8e44f47` (D-07) |
| Offerer is not a responder | Supported | Fixed in fork `8e44f47` (D-06) |
| Others may **make counteroffers** | **Missing** | D-08 |
| Others may **make their own proposals** (to the active player) | **Missing** | D-08 |
| Several proposals on the table at once | One at a time | Equivalent outcome via sequential offers |
| Trading only after rolling | Supported | |

Behaviors that affect agents but break no rule:

| Behavior | Consequence |
|---|---|
| Responses are collected one seat at a time, so later responders see earlier answers | Information ordering; recorded in the event log |
| No limit on offers per turn (H-06) | Agent-level loop guard needed (plan §8.4) |
| Offers are accepted between free roads of Road Building (H-06) | Unusual but not forbidden; accepted |
| The Gymnasium env's action space has maritime trades only | Catanatron's RL example cannot learn domestic trading |

## Built-in bots and trading

| Bot | Offers made | Responses | Accept rate | Trades completed | Errors |
|---|---|---|---|---|---|
| `R` RandomPlayer | 0 | 4,143 | 25.5% | 793 | 0 |
| `W` WeightedRandom | 0 | 3,186 | 23.5% | 582 | 0 |
| `VP` VictoryPoint | 0 | 2,811 | 26.8% | 554 | 0 |
| `F` ValueFunction | 0 | 1,176 | **0%** | 0 | 0 |
| `AB:2` AlphaBeta | 0 | 435 | **0.2%** | 1 | 0 (after fix; crashed before) |

(20 matches per row, 8 for `AB:2` with `--bots catanatron:AB:2 --boards 2`; offers made by the `random-trader`: 1,381, 1,062, 937, 392 and 145.)

Findings:

1. **No built-in bot ever proposes a trade** (H-02). `OFFER_TRADE` never appears in their legal action lists.
2. **The random-style bots accept about a quarter of offers they can pay for**, close to a random choice between accept and reject.
3. **ValueFunction never accepts.** It looks one action ahead and keeps the first action with the strictly best value. Accepting moves no cards until the offerer confirms, so `ACCEPT` scores the same as `REJECT`, and `REJECT` is listed first.
4. **AlphaBeta crashed on any trade offer** (`RuntimeError: Unknown ActionType REJECT_TRADE`): its search code had no handling for trade steps, so it could not sit in any game where someone trades. Fixed in fork `52213b3` by treating trade steps as deterministic; afterwards it accepted 1 of 435 offers, so it is effectively trade-blind like ValueFunction.
5. **The trade lifecycle is robust after the fork fixes:** about 3,800 offers and about 1,900 completed trades across 80 matches with zero errors or illegal states.

## Patch scope

| Need (plan §8) | Where | Estimate |
|---|---|---|
| Offer, respond, confirm, cancel | Done (engine + fork fixes) | — |
| Counteroffers and proposals from other players (D-08) | **Adapter-level negotiation layer**, no engine change: a responder may attach a public counter-proposal to its rejection; the active player may then issue it as a normal `OFFER_TRADE`. Cards still move only through the engine's offer, accept and confirm path, so every completed trade stays lawful | 2–3 days (TRD-027) |
| Loop guard (offers per turn, no repeated identical offer) | Adapter or agent level | 0.5 day (TRD-027) |
| Trade-capable opponents | Built-in bots are trade-blind (F, AB) or random (R, W, VP), so they cannot evaluate a trading agent. Build the plan's five negotiator baselines (§8.5) as our own agents | 2–3 days (TRD-026 / OPP-025) |

## Decision

**GO** for trading in Release 0.4 on the engine path, with no further engine patches. Counteroffers and other players' proposals are implemented as an adapter-level negotiation layer over lawful engine trades, so benchmarks remain standard Catan. No bank/port-only benchmark substitution is needed.

**Consequence for the reference ladder (ENG-011):** built-in bots never trade with each other, so ladder games contain no domestic trades unless one of our trading agents is seated. Strength measured against the ladder says little about play against trading humans; the trading release must be evaluated against trade-capable baselines.
