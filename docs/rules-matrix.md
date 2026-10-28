# Rules Matrix (ENG-003)

Maps each base-game rule to the Catanatron code that implements it, the test that checks it, and any patch needed. The official rulebook overrides engine behavior (master plan §1.3).

| Field | Value |
|---|---|
| Rule set | **CATAN – The Game, 6th edition (2025)**, rulebook CN3081 v6.250401 (12 pages) |
| Rulebook | https://www.catan.com/sites/default/files/2025-03/CN3081%20CATAN%E2%80%93The%20Game%20Rulebook%20secure%20%281%29.pdf |
| Engine | Fork `imnotdeepak/catanatron`, branch `catan-ai-fixes` at `d252ad8`: upstream `ecf9311` (v3.3.0) plus four rule fixes (ADR-001, ADR-005). Audit performed on upstream `ecf9311` |
| Audited | 2026-10-28 |
| Tests | `tests/rules/test_conforming_rules.py` (R rows), `tests/rules/test_known_deviations.py` (D rows: regression tests once fixed, strict xfail while open) |

**Status key:** **Pass** = test passes. **Inspected** = checked by reading code; test still to write. **Fixed** = upstream deviation patched in the fork; its test is now a regression test. **Open** = still deviates; strict-xfail test asserts the correct rule.

## Edition choice

The plan expected Catanatron to follow pre-2025 rules. Inspection shows its turn structure matches the 2025 6th edition: one Action phase in which trading and building happen in any order, and development cards playable before rolling (rulebook p.6–7). The plan notes that earlier editions used separate trade and build phases. No 5th-edition rulebook was audited clause by clause, so the 6th edition is adopted as the baseline (ADR-004). Every deviation below is measured against it.

## Conforming rules

| ID | Mechanic | Rule (page) | Engine location | Test | Status |
|---|---|---|---|---|---|
| R-01 | Supply: 19 of each resource; dev deck 14 Knight, 5 VP, 2 Road Building, 2 Invention (Year of Plenty), 2 Monopoly; 15 roads, 5 settlements, 4 cities per player | p.3, p.8–9 | `models/decks.py` `starting_resource_bank`, `starting_devcard_bank`; `state.py` `PLAYER_INITIAL_STATE` | `test_r01_component_supply` | Pass |
| R-02 | Costs: road WB, settlement WBSWh, city 2Wh 3O, dev card S Wh O | player aid, p.8–9 | `models/decks.py` `*_COST_FREQDECK` | `test_r02_building_costs` | Pass |
| R-03 | Board: 4 forest, 3 hills, 4 pasture, 4 fields, 3 mountains, 1 desert; 18 number discs | p.3, p.11 | `models/map.py` `BASE_MAP_TEMPLATE` | `test_r03_base_map_composition` | Pass |
| R-04 | Setup snake order P1→P4 then P4→P1; first player starts turn 1 | p.12 steps 6–7 | `apply_action.py` `apply_build_road` (initial branch) | `test_r04_setup_snake_order_and_first_player` | Pass |
| R-05 | Starting resources: one card per producing hex adjacent to the **second** settlement | p.12 step 8 | `apply_action.py` `apply_build_settlement` | `test_r05_starting_resources_come_from_second_settlement` (also checks 95-card conservation) | Pass |
| R-06 | Production: 1 per settlement, 2 per city; robber hex does not produce | p.6 | `apply_action.py` `yield_resources` | — | Inspected |
| R-07 | Rolling 7: no production; each player with more than 7 cards discards half, rounded down | p.6 | `apply_action.py` `apply_roll` (`discard_limit=7`) | `test_r07_seven_discards_half_rounded_down_above_seven` | Pass |
| R-08 | Robber must move to a new hex; steal 1 random card from a chosen player with a building there (mandatory when possible) | p.6 | `models/actions.py` `robber_possibilities`; `apply_move_robber` | — | Inspected |
| R-09 | Dev cards don't count toward the 7-card limit and can't be stolen | p.9 | `state_functions.py` `player_num_resource_cards` (resources only) | — | Inspected |
| R-10 | At most one dev card per turn; not one built this turn | p.9 | `state_functions.py` `player_can_play_dev`, `player_clean_turn` (`*_OWNED_AT_START`) | `test_r10_dev_card_not_playable_on_turn_bought` | Pass |
| R-11 | A dev card may be played before rolling or during the Action phase | p.6, p.9 | `models/actions.py` `generate_playable_actions` | `test_r11_dev_card_playable_before_rolling` | Pass |
| R-12 | Dev cards never return to the supply; none can be built once it runs out | p.9 | `generate_playable_actions` (`len(development_listdeck) > 0`) | — | Inspected |
| R-13 | Monopoly: all cards of one type from each player. Invention: any 2 cards from supply. Road Building: 2 free roads | p.9 | `apply_play_monopoly`, `apply_play_year_of_plenty`, `apply_play_road_building` | — | Inspected (see Q-03) |
| R-14 | VP cards stay hidden; they count toward 10, including ones built this turn | p.9–10 | `state_functions.py` `buy_dev_card` (`ACTUAL_VICTORY_POINTS`) | — | Inspected |
| R-15 | Turn = Production phase, then Action phase with trades and builds in any order; trading only after rolling; other players trade only with the active player | p.6–7 | `generate_playable_actions`; `game.py` `is_valid_action` | `test_r15_trades_require_roll_and_no_inactive_player_trades` | Pass |
| R-16 | Maritime trade: 4:1, 3:1 port, 2:1 matching port, for a different resource from the supply | p.7 | `models/actions.py` `inner_maritime_trade_possibilities` | — | Inspected |
| R-17 | Domestic trade: no giving cards away, no trading matching resources | p.7 | `game.py` `is_valid_trade` | — | Inspected |
| R-18 | Placement: roads connect to own road/building and not past an opponent's building; settlements obey the distance rule and connect to own road; city replaces own settlement and the settlement piece returns | p.8–9 | `models/board.py` `buildable_edges`, `buildable_node_ids`, `build_city`; `state_functions.py` `build_city` | — | Inspected |
| R-19 | Largest Army: first to 3 knights; another player must have **more** to take it | p.9 | `state_functions.py` `maintain_largest_army` | — | Inspected |
| R-20 | Longest Route when building: first to 5 continuous roads; another player must have **more** to take it | p.8 | `models/board.py` `build_road` | — | Inspected |
| R-21 | Number discs placed in A-B-C order counterclockwise from a corner, skipping the desert; robber starts on the desert | p.11 | `models/map.py` `initialize_tiles` (`official_spiral`); `Board.__init__` | — | Inspected |

## Deviations

| ID | Rule (page) | Upstream behavior | Engine location | Test | Impact | Status / patch |
|---|---|---|---|---|---|---|
| D-01 | *Plan §10.2, not a rulebook rule:* search branches need independent randomness | `Game.copy()` and `State.copy()` share the RNG object with the original, so simulating a copy changes the real game's future dice and dev draws | `state.py:166-167`, `game.py:218` | `test_d01_copy_does_not_perturb_original_rng` | **Critical for search.** Any rollout perturbs the real game and breaks reproducibility | **Fixed in the adapter** (ENG-007): `EngineAdapter.branch(seed)` gives each copy its own RNG and a reshuffled dev deck; tests in `tests/engine/test_branching.py`. Engine `Game.copy()` still shares the RNG (upstream design for its search bots), so never call it directly |
| D-02 | Win only when reaching 10 VP **during your own turn** (p.10) | `winning_color()` returns any player with ≥10 VP after any action | `game.py` `winning_color` | `test_d02_only_active_player_can_win` | Off-turn wins, e.g. a third player handed Longest Route when another player's settlement cuts a road | **Fixed** in fork `d252ad8`: only the active player can win; a player who reached 10 off-turn wins when their turn begins |
| D-03 | Bank shortage: nobody receives that resource, **unless only one player is affected**, who then takes what remains (p.6) | Nobody receives it, even when only one player is affected | `apply_action.py` `yield_resources` | `test_d03_bank_shortage_single_player_gets_remainder` | Rare; late-game production | **Fixed** in fork `a36ac1d` |
| D-04 | If a broken route no longer meets the requirements, the tile returns to the supply until a **single** player has the longest route of **at least 5** (p.8) | After a cut, the tile goes to whoever has the maximum length, **even below 5** | `models/board.py:143-145` | `test_d04_longest_route_returns_to_supply_when_no_route_of_five` | 2 VP wrongly kept or awarded | **Fixed** in fork `adbb0ed` (`Board._holder_after_cut`; `maintain_longest_road` now removes the holder's VP when the tile returns to the supply) |
| D-05 | As D-04: tile stays in the supply while the longest route is **tied** among non-holders (p.8) | `max()` over the length dictionary picks a tied player by insertion order; a tied former holder may also lose it | `models/board.py:143-145` | `test_d05_longest_route_ties_after_cut` | 2 VP wrongly awarded or removed | **Fixed** in fork `adbb0ed`; holder keeps the tile on a tie per provisional Q-01 |
| D-06 | Domestic trade lifecycle (p.7) | The response loop doesn't skip the offering player, which gets asked to accept or reject its own offer | `apply_action.py` `apply_accept_trade`, `apply_reject_trade` | `test_d06_offerer_is_not_asked_to_respond_to_own_offer` | Corrupts the trade flow and lets the offerer "accept" itself | **Fixed** in fork `8e44f47` |
| D-07 | Players trade cards they actually hold; nothing may be given away (p.7) | Offerer's ability to pay is checked neither at offer nor at confirmation; hands can go negative | `game.py` `is_valid_action`; `apply_confirm_trade` | `test_d07_offerer_cannot_trade_cards_it_does_not_hold` | Illegal resource creation | **Fixed** in fork `8e44f47` (offer rejected unless offerer holds the cards; confirm re-checks both sides) |
| D-08 | Other players "may accept your proposal, make counteroffers, or make their own proposals" (p.7) | Only the active player can propose; responders can only accept or reject; no counteroffers | `models/enums.py` `ActionType` (no such actions) | — | Trading is narrower than the rule | **Open.** Scoped by the trade audit (ENG-010 → TRD-027) |

## Engine behaviors that affect the harness (not rule deviations)

| ID | Behavior | Location | Consequence |
|---|---|---|---|
| H-01 | `Player.decide(game, playable_actions)` receives the full `Game`, including every hand and the shuffled dev deck order | `game.py` `play_tick`; `state.py` `development_listdeck` | Built-in bots are privileged; our agents must go through the observation firewall (ENG-005) |
| H-02 | Built-in bots never initiate domestic trades: `OFFER_TRADE` never appears in `playable_actions` and is accepted only through `is_valid_action` | `models/actions.py`, `game.py` | Ladder games contain no player-to-player trades (input to ENG-010) |
| H-03 | Seating order is shuffled by the game RNG (`random.sample(players)`) | `state.py:100` | Handled: `EngineAdapter.new_game` seats players in the given order; the seed still fixes the board (ENG-004) |
| H-04 | `TURNS_LIMIT = 1000` counts `num_turns`, which also advances during setup placements | `game.py`, `apply_action.py` `advance_turn` | Truncation threshold is approximate; arena reports truncation separately |
| H-05 | Discards are resolved one card at a time in seat order starting from seat 0, so later discarders see earlier discards | `apply_action.py` `apply_roll`, `apply_discard` | Information ordering for the observation contract (Q-02) |
| H-06 | No limit on repeated trade offers in a turn; an offer can be made during Road Building placement | `game.py` `is_valid_action` | Agent-level loop guard needed (plan §8.4) |
| H-07 | Port positions are fixed in the template and only the port types are shuffled; the rulebook shuffles the frame pieces that carry the ports. The number spiral always starts from the same corner | `models/map.py` | Board distribution differs slightly from physical variable setup; not a rule violation. Accepted and documented |
| H-08 | Invention (Year of Plenty) offers a single card when the supply cannot supply two | `models/actions.py` `year_of_plenty_possibilities` | See Q-03 |

## Open interpretation questions

| ID | Question | Current position | Resolve via |
|---|---|---|---|
| Q-01 | If the Longest Route holder's route is broken and they now **tie** with another player, do they keep the tile? | Rule text is "no longer meet the requirements"; holding requires having the longest route, and a tie doesn't take it away elsewhere. Provisional: holder keeps it on a tie | Official FAQ (https://www.catan.com/faq/basegame) before patching D-05 |
| Q-02 | Are discarded card types public? | **Resolved:** public. Supply stacks are face up (p.4, p.11) and discards return to them | `docs/observation-contract.md` OC-01 |
| Q-03 | Invention when the supply has fewer than 2 suitable cards | Rule is silent; keep engine behavior (take what is available) | FAQ; low impact |
| Q-04 | A player who reaches 10 VP on another player's turn | Under p.10 they win once their own turn begins, if they still have 10 | Implemented in fork with D-02 |
