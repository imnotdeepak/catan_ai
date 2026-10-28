# Observation Contract (ENG-005, ENG-006)

Defines what an agent may know at a decision: the public game plus its own hand and its own private events. This is the only information path to policies, candidate generators and search (master plan §2.1).

Implementation: `src/catan_ai/observations/` (`Referee`, `PlayerObservation`, `Event`). Tests: `tests/observations/test_firewall.py`.

## Data products

| Product | Who may read it | Contents |
|---|---|---|
| `Referee` (adapter, `events`, `action_log`) | Referee, arena, offline evaluation, supervised labels | Full engine state, every private payload, random outcomes. With the seed, `action_log` replays the game |
| `PlayerObservation` for seat *s* | Seat *s*'s agent | The whitelisted fields below |

Never pass a `Referee`, `EngineAdapter`, engine `Game` or other engine object to an agent. Catanatron's built-in bots receive the full `Game` through `Player.decide` and are therefore **privileged** (rules matrix H-01).

## `PlayerObservation` fields (schema version 1)

| Field | Content | Why it is legal knowledge |
|---|---|---|
| `seat`, `current_seat`, `turn_seat`, `prompt`, `is_initial_build_phase`, `num_turns` | Whose decision and what kind | Table state |
| `board` | Tiles (resource, number), robber, ports, buildings and roads with owner seats | Visible on the table |
| `players[*]` | Visible VP (excludes hidden VP cards), resource card count, unplayed dev card count, played dev cards by type, pieces left, Longest Road length and holder, Largest Army holder, has rolled, has played a dev card this turn | Visible or derivable from the table |
| `bank_resources` | Supply count per resource | Supply stacks are face up (rulebook p.4) |
| `dev_deck_remaining` | Cards left in the deck | Visible stack height |
| `discards_pending` | Cards each seat must still discard on a 7 | Announced by rule (half, rounded down) |
| `pending_trade` | Offerer, offered and requested cards, who has accepted | Spoken aloud at the table |
| `own_resources`, `own_dev_cards` | The viewer's own hand by type | Own hand |
| `history` | Every event as this seat sees it (below) | Everything observed so far |
| `legal_actions` | Canonical legal actions, only when `seat == current_seat` | Rules applied to public state plus own hand |

Never included: the seed, RNG state, dev deck order, other seats' resource composition, other seats' unplayed dev card types, hidden VP totals.

## Events

One event per executed action, with `event_type` equal to the canonical action kind, plus `BONUS_TRANSFER` (Longest Road / Largest Army changes holder) and `GAME_END`. Payloads are read-only and derived from the referee state before and after the action.

| Payload field | Visibility | Notes |
|---|---|---|
| `action` (kind, seat, params) | Public | Canonical action; never contains drawn cards or stolen resources |
| `card_count_delta`, `dev_card_count_delta` | Public | Per seat |
| `resource_delta` | Public, **except robber steals** | Per seat and resource. Covers production, setup resources, builds, trades, discards, Monopoly, Year of Plenty |
| `dice` | Public | On `ROLL` |
| `dev_card_played` | Public | On `PLAY_*` |
| `dev_card` | Private to the buyer | On `BUY_DEV_CARD` |
| `stolen_resource` | Private to thief and victim | On `MOVE_ROBBER` with a steal; the public payload omits `resource_delta` and shows only the one-card count change |
| `GAME_END`: winner, truncated, winner's revealed VP cards | Public | The winner reveals VP cards to show 10 (rulebook p.10) |

**Mapping to the plan's minimum event list (§2.3):** setup placements and setup resources are `BUILD_SETTLEMENT`/`BUILD_ROAD` events during the initial phase with a `resource_delta`. `ROLL_DICE` plus `PRODUCTION` is the `ROLL` event. `DISCARD_COUNT`/`DISCARD_PRIVATE_TYPES` is the `DISCARD` event (types public, below). `STEAL_COUNT`/`STEAL_PRIVATE_TYPE` is the `MOVE_ROBBER` event. Trade, build, dev card, `END_TURN`, `BONUS_TRANSFER` and `GAME_END` map one to one.

## Decisions

| ID | Decision | Rationale |
|---|---|---|
| OC-01 | **Discarded card types are public** (resolves rules-matrix Q-02) | Discards return to face-up supply stacks (p.4, p.6), so the table sees them. Changing this means moving `resource_delta` for `DISCARD` events into private payloads |
| OC-02 | Supply counts are public | Face-up stacks (p.4) |
| OC-03 | Engine node ids, edge pairs and tile coordinates are used as public identifiers | They name the fixed board topology, not hidden state |
| OC-04 | Later discarders see earlier discards (engine resolves discards in seat order; H-05) | Accepted for now; simultaneous discard would need engine changes |

## Guarantees and their tests

| Guarantee | Test |
|---|---|
| A seat's observation is unchanged when everything it cannot observe is altered: other seats' resource mix and dev card types (counts kept), dev-card ownership age, dev deck order, RNG | `test_observations_do_not_depend_on_hidden_state` (every 25th decision of 6 games, all 4 seats) |
| Private payloads reach only the buyer, or the thief and victim, and never appear in public payloads | `test_private_payloads_reach_only_entitled_seats` |
| A seat's visible history alone reconstructs every seat's card and dev card counts and the seat's exact own hand | `test_visible_history_reconstructs_counts_and_own_hand` |
| Observations contain only plain data (ints, strings, bools, tuples, read-only mappings, project dataclasses) | `test_observations_hold_only_plain_data` |
| Legal actions are given only to the deciding seat | `test_legal_actions_only_for_the_deciding_seat` |

The tests were checked against five deliberately planted leaks (an opponent's wood count in a public field, the dev deck's top card, a stolen resource in the public payload, a drawn card shown to all seats, a steal missing from the history). Each was caught.

## Known limits

- Policy-level indistinguishability (identical action probabilities under scrambled hidden state) needs agents and is tested once agents exist (Release 0.2). The observation-level test is its foundation: a policy reading only observations cannot do better than its input.
- `history` grows with the game; observations share event objects, which are immutable.
