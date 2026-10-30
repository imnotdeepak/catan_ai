# CATAN AI — MASTER IMPLEMENTATION PLAN

**Project:** Superhuman Catan AI (research and engineering)  
**Document version:** 1.1 — consolidated architecture 3.1  
**Prepared:** 2026-10-28 · **Revised:** 2026-10-28  
**Status:** Approved design / implementation not yet started  
**Primary engine:** [Catanatron](https://github.com/bcollazo/catanatron) (exact commit to be pinned in Release 0.1)

> **North star:** Build an AI that maximizes the probability of winning **standard four-player base Catan** against diverse, capable opponents, using only information an actual player could legally know. The long-term goal is credible superiority to expert human play. That outcome is **not yet established**.

### Revision 1.1 changes

1. **Release order:** a learned policy/value baseline now ships in Release 0.2, directly after Rules + Arena. Specialist systems (opening, victory, beliefs, trade, opponents) are added afterwards and kept only when ablations against that learned baseline show a gain.
2. **Reference ladder:** Catanatron's built-in bots (up to `AlphaBetaPlayer`) are now mandatory reference opponents. Bots that read the full game state are labeled **privileged** in every report.
3. **Throughput gate:** simulator speed (games/sec, decisions/sec, clone cost) is measured in Release 0.1. All search and rollout budgets in this plan are placeholders until then.
4. **Trade audit:** Catanatron's domestic-trade lifecycle is audited in Release 0.1, with an explicit patch-scope estimate and go/no-go for the trading release.
5. **Rules edition:** the baseline rule set is the official edition Catanatron's pinned commit most closely implements. ENG-003 found that to be the 2025 6th edition (ADR-004).
6. **Beliefs:** exact accounting plus enumerated hypotheses comes first. Particle filtering is a fallback for when enumeration exceeds its budget.
7. **Opponent Intelligence:** expanded with an interface, model progression, consumers and tests (§7).
8. **Process:** tiered definition of done, release timeboxes, renumbered backlog, Windows/PowerShell bootstrap commands.
9. **Repository:** `catan/` is its own standalone GitHub repository; Catanatron is a pinned git submodule (§12).

---

## 0. Executive decisions

These decisions carry forward the project discussions and are **locked unless a failed test or measured result justifies changing them**.

| Area | Decision |
|---|---|
| Rules | Exactly four players; standard base Catan; target **10 VP**; no house rules, expansions, friendly robber, or other optional rules. Pinned to the **2025 6th edition**, which Catanatron's turn structure matches; every engine behavior is validated against it (§1.1, `docs/rules-matrix.md`). |
| Engine | Catanatron, wrapped rather than trusted as an infallible rules/visibility reference. |
| Opening | **Opening Strategy Engine 2.0**: all six named archetypes as *priors*, legal settlement/road pair evaluation, opponent-aware snake-draft search, full-game outcome testing. |
| Victory planning | **Victory Path Planner 2.0**: AND/OR dependencies, multiple contingent routes to 10 VP, opponent win threats, and defensive actions. |
| Resources | **Resource Intelligence 2.0**: append-only player-visible event history, exact accounting where possible, enumerated joint hypotheses for ambiguous events, particle sampling only when enumeration exceeds budget. |
| Development cards | Separate hidden deck/hand inference with deck composition, purchase-age constraints, public reveals, and probability of hidden VP. |
| Trading | **Trade Intelligence 2.0**: proposals, acceptance/rejection/counteroffers, opponent-specific behavior, joint counterfactual win estimates, bank/port and no-trade alternatives. |
| Opponents | **Opponent Intelligence 2.0**: separate estimates of capability, intention, and likely behavior from their own information. |
| Learning | Shared **PyTorch action-conditioned policy/value model**, imitation → masked PPO → diversified league self-play; alternatives tested rather than assumed. A learned baseline exists **before** specialist systems are built, so each specialist is measured against it. |
| Baselines | Catanatron built-in bots form a **reference ladder** (Random → … → AlphaBeta). Full-state bots are reported as **privileged** opponents, never as fair-information peers. |
| Search | Adaptive **information-set-aware** planning; no perfect-information oracle in deployed decisions. |
| Authority | Specialized systems advise; **Final Decision Controller** selects a legal action using shared value estimates and optional search. |
| Success | Full four-player evaluations, seat rotation, held-out opponents/boards, uncertainty intervals, ablations, and expert-human testing before any “superhuman” claim. |

**Interpretation:** “Specified” means a design exists; it does **not** mean implemented, benchmarked, or validated.

---

## 1. Product boundaries, rules, and invariants

### 1.1 Scope

**In:** Four-player base game; randomized *standard* board configurations; both initial settlements and adjoining roads; production; domestic and maritime trade; roads/settlements/cities; robber; development cards; Longest Road; Largest Army; exact turn order; first-to-10-points win condition on the winner's own turn.

**Out:** Seafarers, Cities & Knights, 5–6-player extensions, beginner protection/friendly robber, house variants, cheating, privileged opponent-hand access, automated play on third-party platforms without authorization, unrestricted conversational negotiation or partnership promises that cannot be completed under standard rules.

**Rules-version decision:** The **baseline rule set** is the official base-game edition whose rules Catanatron's pinned commit most closely implements.

**Resolved in ENG-003 (2026-10-28, ADR-004):** the baseline is the **2025 6th edition** (rulebook CN3081 v6.250401). Catanatron's turn structure matches it: one Action phase with trades and builds in any order, and dev cards playable before rolling. Earlier editions documented separate trade/build phases. Engine deviations from the 6th edition are listed and tested in `docs/rules-matrix.md`; patch the ones that affect play. Never mix editions within a benchmark. “No optional rules” means rules from the selected edition, not a blend of old defaults and newer tournament customs.

### 1.2 Canonical mechanics to test

- Opening settlement/road placement in snake order: **P1, P2, P3, P4, P4, P3, P2, P1**. The second settlement grants starting resources from adjacent producing terrain hexes.
- Standard piece supplies: **15 roads, 5 settlements, 4 cities** per player; a city upgrades an existing settlement and returns the settlement piece to supply.
- Building costs, resources in order **[wood, brick, sheep, wheat, ore]**:
  - Road `[1,1,0,0,0]`
  - Settlement `[1,1,1,1,0]`
  - City `[0,0,0,2,3]`
  - Development card `[0,0,1,1,1]`
- Standard resource supply: **19 cards of each of the five resources**. Respect bank shortages and resource conservation.
- Base development-card deck: **14 Knight, 5 VP, 2 Road Building, 2 Year of Plenty, 2 Monopoly** = 25.
- Largest Army: at least **3 played knights** and transfer/tie rules; Longest Road: at least **5 continuous segments** with exact opponent interruption, loops and tie behavior. Each award is 2 VP and can change hands.
- Development-card timing: maximum one non-VP development card played per turn, none purchased that same turn; VP card reveal/win exceptions according to selected rules.
- Domestic trading: active player's participation required; both sides exchange actual resources, never development cards, gifts, IOUs, or non-resource services. Counteroffer **discussion** can be represented, but actual transfer must be lawful. Disallow inactive-player-to-inactive-player completed trades.
- Maritime rates: 4:1 bank, 3:1 general port, 2:1 matching resource port; verify bank availability and selected-edition build/trade order.
- When 7 is rolled: discard if more than 7 cards, then robber movement and possible theft; no pre-resolution trade. Track what the acting player personally learns from theft versus what bystanders can observe.
- Terminal: win on own turn at **≥10 VP**; avoid awarding a win to an opponent merely because their hidden VP estimate reaches 10 outside their own turn.

### 1.3 Source-of-truth and open audit items

The official rulebook overrides undocumented simulator behavior. The engine's **exact Git commit** is to be recorded, not inferred from the branch name. Maintain `docs/rules-matrix.md` with `mechanic | official rule/page | Catanatron function | reproduction test | pass/fail | required patch`.

Mandatory audits at Release 0.1:

1. Complete four-player phase transitions and player setup order.
2. Trade offer / response / confirmation lifecycle; *actual* affordability at offer and completion; multiple acceptors and counteroffers. Output: `docs/trade-audit.md` listing which trade actions exist, whether built-in bots ever initiate or accept domestic trades, what is missing, an estimate of patch scope, and a go/no-go for the trading release.
3. Player-specific event visibility, especially robbery, discarded cards, secret dev draws, deck order, and private hands.
4. Deep-copy isolation of RNG and hidden state for search. **Do not assume `Game.copy()` guarantees branch RNG independence without a tested assertion.**
5. Standard edition configuration, robber, bank shortage, Longest Road and Largest Army edge cases.
6. Reproducible gameplay from seed + engine version + player-policy versions.
7. Check Catanatron's **GPL-3.0** license and Catan trademarks before distributing a derivative engine or public app; review distribution terms with a qualified professional if commercializing.
8. **Simulation throughput:** games/sec for random and each built-in bot, decisions/sec, `Game.copy()` cost, and observation/firewall overhead on the development machine. Output: `reports/throughput.md`. Every search/rollout budget in this plan is a placeholder until it is derived from these measurements.
9. **Reference ladder:** four-player, seat-rotated matches among the built-in bots. Mark which bots read the full game state (privileged). The upstream leaderboard is 1v1 with 25-game samples, so re-measure in our four-player protocol.

**Go/no-go:** If a rules-critical behavior is missing, patch behind an adapter or maintain a minimal tested fork before training. Never substitute a convenient simplified trading mode in final benchmarks while calling it standard Catan.

---

## 2. Architecture and information boundaries

```text
                    Authoritative Catanatron state (privileged; engine only)
                                          |
                               Rules-tested event recorder
                                          |
                 +------------------------+-----------------------+
                 |                                                |
       Full truth / referee log                         Per-player observation firewall
       [offline evaluation only]                                    |
                                                       Public + own-private event history
                                                                    |
                                                 Exact ledger + joint hidden-state beliefs
                                                                    |
                    +------------------+---------------------------+------------------+
                    |                  |                           |                  |
                Opening 2.0       Victory Path 2.0            Opponent 2.0        Trade 2.0
                    |                  |                           |                  |
                    +------------------+---------------------------+------------------+
                                                                    |
                                             Shared state/action encoder + policy/value
                                                                    |
                                            Legal action filter + optional belief search
                                                                    |
                                                   Final Decision Controller
                                                                    |
                                                       Chosen legal game action
                                                                    |
                                                    Engine executes and records

 Training side: self-play workers -> immutable experience -> learner -> candidate checkpoints
                           -> independent arena -> champion promotion
```

### 2.1 Invariant: no illegal hidden-information path

There are **two separate data products**:

- **Referee log:** privileged complete state and RNG outcomes, inaccessible to policies/search at a real decision. Used for debug, supervised labels, and evaluation only.
- **Player view:** public board/history plus the acting player's own hand and private observations. This and derived beliefs are the **only inputs** to policy, candidate generation, and decision search.

A training label can be privileged ground truth while the **model features** must remain player-visible. No hidden label may appear in inference inputs through caches, auxiliary embeddings, action generators, replay metadata, normalization statistics, or search selection.

**Indistinguishability test:** Replace all unobservable opponents' resources/development cards/deck ordering with a different legal configuration while preserving the same player-visible history. With identical model state and policy RNG, the action probabilities and root choice must be identical. Search may sample hidden worlds but must **not** branch its real decision on which hidden world was secretly true.

### 2.2 Source interfaces (proposed contracts, not existing library APIs)

```python
ResourceVector = tuple[int, int, int, int, int]  # wood, brick, sheep, wheat, ore
PlayerId = int  # stable seat, 0..3

@dataclass(frozen=True)
class PlayerObservation:
    schema_version: int
    match_id: str
    player_id: PlayerId
    phase: str
    turn_index: int
    board_public: dict                # tiles, numbers, robber, nodes, roads, ports
    public_players: tuple[dict, ...] # public VP, hand count, dev-card count, played knights
    own_resources: ResourceVector
    own_development_cards: tuple[str, ...]
    new_visible_events: tuple[dict, ...]
    legal_actions: tuple[dict, ...]  # validated + canonical encoded actions

@dataclass(frozen=True)
class VisibleEvent:
    sequence_id: int
    phase: str
    actor: PlayerId | None
    event_type: str
    public_payload: dict
    viewer_private_payload: dict | None

@dataclass(frozen=True)
class DecisionContext:
    observation: PlayerObservation
    belief_summary: dict
    opponent_predictions: dict
    candidate_plans: tuple[dict, ...]

@dataclass(frozen=True)
class DecisionResult:
    action: dict  # canonical legal action; translator maps back to engine action
    predicted_our_win_probability: float | None
    policy_version: str
    search_simulations: int
    audit_reason: str
```

`DecisionResult.audit_reason` is compact debug metadata, **not** a promise of human-language explanation fidelity.

### 2.3 Event schema and visibility policy

Each event has: `match_id, sequence_id, turn_index, actor_id, type, public_payload, private_payload_by_player, rule_version, engine_commit`. Include `observed_before_action` and `observed_after_action` timestamps in *logical sequence numbers*, not wall-clock-dependent game rules.

Minimum event types: `SETUP_PLACE_SETTLEMENT`, `SETUP_PLACE_ROAD`, `SETUP_RESOURCE_RECEIPT`, `ROLL_DICE`, `PRODUCTION`, `DISCARD_COUNT`, `DISCARD_PRIVATE_TYPES`, `MOVE_ROBBER`, `STEAL_COUNT`, `STEAL_PRIVATE_TYPE`, `BUILD`, `BUY_DEV_CARD`, `PLAY_DEV_CARD`, `DOMESTIC_TRADE_OFFER`, `DOMESTIC_TRADE_RESPONSE`, `DOMESTIC_TRADE_COMPLETE`, `MARITIME_TRADE`, `END_TURN`, `BONUS_TRANSFER`, `GAME_END`.

**Event-specific visibility tests** are required. Examples:

- `PRODUCTION`: public hex and known recipients; exact publicly knowable resource counts.
- `STEAL_PRIVATE_TYPE`: only participating player(s) as specified by chosen rule/information semantics; never all spectators.
- `BUY_DEV_CARD`: public purchase, concealed type; purchaser learns its own type.
- `DISCARD_COUNT`: publicly known count; discard types only visible where actually permitted.
- `DOMESTIC_TRADE_COMPLETE`: amounts/types transferred publicly observable, no secret deal.

Use versioned serializers with a whitelist of exposed fields; never serialize a raw `Game`/`State` object into agent inputs.

---

## 3. Final Decision Controller and shared evaluation

A single controller is the **only entry point** for move selection. Specialized components never execute actions directly.

1. Obtain vetted `PlayerObservation` and canonical legal action set.
2. Detect legal terminal-winning moves and other forced rule actions before expensive search.
3. Ask opening/victory/belief/opponent/trade specialists for **features and candidate actions** (not mandatory orders).
4. Score all viable legal actions with the shared action-conditioned policy and four-outcome value model.
5. Allocate deeper belief-aware search to close, decisive, or uncertain choices.
6. Validate selected action against current authoritative legality **without passing authoritative hidden values to the agent**.
7. Execute and log.

Optimization target:

\[
a^* = \arg\max_{a\in\mathcal A_{\mathrm{legal}}(I_t)}
\mathbb E_{\pi_{-i},\,\text{dice},\,\text{hidden states}}[\mathbf 1\{\mathrm{winner}=i\}\mid I_t,a].
\]

`I_t` denotes the acting player's information state. The final policy may disagree with any opening archetype or plan if the legal alternative has greater estimated winning value.

**Value model:** output a calibrated joint distribution `[P(red wins), P(blue wins), P(white wins), P(orange wins)]`; sum to 1 when modeling eventual decisive outcomes under a specified continuation model. For finite-horizon forecasts use **five classes** including `no winner by horizon`; do not assign unfinished rollouts arbitrary wins. The value is conditional on a defined opponent population and compute budget.

---

## 4. Opening Strategy Engine 2.0

### 4.1 Objective and outputs

At every setup decision, choose a **legal next action**, optimizing expected game win probability over subsequent draft choices and later play. Plan settlement+attached-road *as a linked pair*, but submit them as separate actions if Catanatron's phase machine requires that sequence.

Return `OpeningRecommendation`:

```text
next_legal_action; shortlist_of_complete_setups; predicted_second_sites;
production_by_resource; early_build_probabilities; port_paths;
expansion_risks; opponent_draft_assumptions; estimated_win_value;
confidence_interval; six_archetype_scores; rationale_features
```

### 4.2 Six archetypes are *priors*, not mandatory policies

1. **Ore–Wheat–Sheep (OWS):** cities, development cards, Largest Army; check expansion bottlenecks and knight timing.
2. **Road Builder (wood–brick):** rapid expansion and junction denial; check wheat/sheep access and eventual ore/wheat transition.
3. **City + Road Hybrid:** complementary settlement economies; Longest Road is contestable, do not prohibit dev cards.
4. **Specialized Port:** high output of a convertible resource and feasible access to the appropriate 2:1 port; evaluate true surplus and cost.
5. **Balanced Five-Resource:** production diversity and flexibility; no magical universal pip threshold.
6. **Resource Monopoly / Cartel:** resource-scarcity leverage and trade incentives; opponents retain bank/port and alternative-production routes, so control is not absolute.

Do **not** train six isolated permanent end-to-end agents. Implement six configurable feature/prior modules, plus optional baseline agents for tournaments. The policy can mix them or discover novel openings.

### 4.3 Opening feature library

- **Expected yield:** per resource, sum `dice_combinations(number)/36 × building production`, excluding desert and robber-blocked output; explicitly track number correlation (same roll can trigger multiple sites), variance, and exposure to 7-induced discards.
- **Useful resource bundles:** expected time-to-road/settlement/city/dev-card, conditional shortages, port conversion, trade dependence.
- **Expansion graph:** shortest legal road paths to potential future sites, contention, intersection value, blocking, future settlement piece availability.
- **Position denial:** cost to competing players of our placement (not automatically a positive if it sacrifices too much self-value).
- **Early tempo:** resources from second settlement, probability of early useful builds given initial hand, turn order.
- **Flexibility:** number and quality of distinct feasible economic/expansion/victory continuations.
- **Ports:** current and reachable port rates, actual surplus, distance and exposure to competition.

### 4.4 Draft search algorithm

```text
for each legal settlement candidate S:
    for each legal adjacent initial-road action R:
        evaluate cheap opening features for (S, R)
keep top K pairs by provisional value + diversity
for each retained pair:
    sample plausible remaining snake-draft decisions from opponent-policy mixture
    on our later setup decision, re-optimize using observed placements
    compute complete starting layout, starting hand and production
    evaluate fast learned value + Victory Path Planner proposals
for top M finalists:
    run multiple full-game rollouts under diverse opponent policies
    pair simulation randomness where meaningful and legal
choose next *currently legal* action with highest calibrated expected win probability
```

Suggested **experiment knobs**, not requirements: initial `K=20` candidate pairs, `M=3–5` finalists, draft continuations `32–128`, game rollouts `32–256`; benchmark compute and strength and adapt budgets. For P4 consecutive placements, the planner should treat the two choices jointly while still respecting action order.

### 4.5 Acceptance tests

- Every output settlement and road is legal; no distance-rule violation.
- All four draft seats and intermediate setup phases covered.
- Initial resource receipt is attached to the second settlement, not the first.
- Board graph, pip expectations, ports and edge IDs match engine state.
- Opponent placements can invalidate top choices; planner replans correctly.
- Against same downstream policy, measure six-prior vs pip baseline; add draft search and rollout search in ablations.
- No strategy prior can bypass legal-action masking or force the final controller's choice.

---

## 5. Victory Path Planner 2.0

### 5.1 Concept

A route to 10 VP is **not simply a list of point sources**. It is a contingent **AND/OR dependency graph** with real resources, legal positions, timing, uncertainty, counterplay and fallback actions.

**AND nodes:** settlement requires legal vertex, reachable road connection, settlement piece, and wood+brick+sheep+wheat. City requires existing settlement, city piece, 3 ore+2 wheat.  
**OR nodes:** acquire a missing resource by production, legal bank/port exchange, accepted domestic trade, resource theft, or favorable development-card effects.

Candidate scoring objectives include settlements, cities, Longest Road, Largest Army, and uncertain VP development cards. The graph includes **defensive/denial objectives**: robber relocation, block a road/settlement site, refuse enabling trades, challenge transferable awards, and race to win first.

### 5.2 Planner representation

```python
@dataclass(frozen=True)
class PlanObjective:
    kind: str                    # city, settlement, army, road_award, dev_vp, defense
    target: str | int | None
    expected_vp_delta: int
    prerequisite_ids: tuple[str, ...]

@dataclass(frozen=True)
class ContingentPlan:
    player_id: int
    objectives: tuple[PlanObjective, ...]
    first_legal_action_ids: tuple[str, ...]
    branching_conditions: tuple[str, ...]
    contested_targets: tuple[str, ...]
    completion_probability: float
    p_win_before_opponents: float
    completion_round_quantiles: tuple[float, float, float]  # P25, P50, P75
    uncertainty: dict
```

`expected_vp_delta` must not count transferable awards as permanently secured or dev VP draws as guaranteed. A path is discarded if it violates remaining roads/cities/settlements, connectivity, purchase/card timing, board occupancy, or plausible available dev-card supply.

### 5.3 Route generation and ranking

1. Compute public and belief-adjusted effective scores for all players.
2. Enumerate plausible **scoring bundles** that can reach 10, respecting existing pieces and bonuses.
3. Expand only promising bundles into concrete board locations, road edges, building dependencies, trade/resource options, and defensive subgoals.
4. Prune dominated, impossible, or excessively slow plans; retain diverse alternatives (e.g., top city, expansion, army, road and disruption variants).
5. Shortlist using production, resource cost, feasibility, competition, flexibility and a cheap time-to-completion heuristic.
6. Evaluate first legal actions using shared value prediction and, selectively, belief-aware rollouts.
7. **Replan after every material event** (or at action boundaries with a stable cache key); don't preserve loyalty to sunk-cost strategies.

Do not simply sum independent plan probabilities; plans overlap. The final comparison is over **mutually exclusive game outcomes**, not a sum of separately scored goals.

### 5.4 Forecasting jointly competing win timelines

For horizon `H` full rounds, return `P(player i wins by H)` for all four players **plus `P(no winner by H)`**, summing to 1. Rollouts terminate at the actual first winner on their turn; all player policies obey what they can observe. Report separately (i) probable resource affordability and (ii) probability the action is legally available: those are not equivalent.

For exact opponent threats, handle hidden VP probabilities, purchase age and knight counts, turn order, city-ready settlement locations, port access, Longest Road ties/interruptions, and possible robber disruption. Maintain intervals calibrated by held-out games and distinguish Monte Carlo variance from opponent-model error.

### 5.5 Evaluation

- Unit tests: legal score sources, 10-point win timing, city upgrade, piece inventory, connectivity, longest-road tie/interruption, army threshold, dev purchase/play timing.
- Tactical fixtures: 9 VP opponent with a winning city resource deficit; robot can win now; loss of expansion route; contested longest road; hidden VP threat.
- Ablation: greedy VP → legal graph planner → opponent-aware planner → full search, holding other policies constant.

---

## 6. Resource & Development Card Intelligence 2.0

### 6.1 Exact event-sourced memory

Preserve **all observable** dice rolls, production, known transfers, building expenses, dev purchases, robber actions, discards and trade responses. Maintain a per-player public history and the agent's own private observations. Do not assert that all opponents' resources are always knowable: hidden steals and chosen discards create ambiguity.

### 6.2 Joint hidden-state posterior: exact first, sampling only when needed

Most of a Catan hand can be worked out from public events: production, builds, maritime trades and completed domestic trades are all observable. Real ambiguity comes from a small set of events: hidden steal types, discard types the observer cannot see, and unknown development-card draws. Inference therefore runs in three tiers, each used only when the previous one is insufficient.

\[
B_{t+1}(s) \propto P(o_{t+1}\mid s,\pi_{\text{opponents}})\,B_t(s)
\]

**Tier 1: exact ledger.** Apply deterministic transitions for every observed event. Propagate public domestic trade details exactly. Where no ambiguous event has occurred, the hand is known and no inference runs.

**Tier 2: enumerated hypotheses (default).** Each ambiguous event branches the joint state into its small set of legal outcomes. A hidden steal from a victim branches into at most five resource types, weighted by the victim's hand composition. Keep a weighted set of joint hypotheses over **all players' resource hands + remaining bank resources + unknown dev-card assignments**. Each hypothesis must obey known hand totals, deck counts, transaction conservation, public production and individual card knowledge. Prune hypotheses that later public events contradict; for example, an opponent building a road proves they held wood and brick. Weight discards with opponent behavior likelihoods (§7), not naïvely uniform. Treat rejected trade offers as weak, strategic evidence, not proof of missing cards.

**Tier 3: particle fallback.** If the hypothesis count exceeds a configured cap (set from ENG-009 latency measurements), switch that game to a weighted particle population drawn from the hypothesis set:

- Prototype `N=2048` particles; compare `N=512, 2048, 8192` on calibration and latency.
- Track effective sample size; resample only when useful and **rejuvenate** legally consistent states afterward.
- If all weights collapse, diagnose missing/incorrect public events; recover with constrained inference conditioned on complete event ledger, not ad hoc unsupported hand guesses.
- Validate particles against Tier 2 enumeration on benchmark fixtures small enough to enumerate.

**Across all tiers:**

- Keep hidden development-card allocations **without replacement**, based on initial 25-card composition, public reveals, purchases, own private draws, remaining deck count, and card age. Never reveal hidden deck *order* to the policy.
- Distinguish bank inventory known from public transactions from any inventory not visible under the implemented observation contract; assert full conservation in referee tests.
- Record how often each tier is used in real games; that frequency decides whether Tier 3 is worth maintaining.

### 6.3 Required outputs

`P(can_afford_road/settlement/city/devcard)`, `P(can_legally_build_next_turn)`, `P(owns_playable_knight)`, `P(hidden VP count = k)`, expected shortages for each candidate win path, resource marginal distributions, joint hand samples, posterior diagnostics and calibrated confidence.

### 6.4 Key tests

- Exact public events replay to known states; every hypothesis (and particle, if Tier 3 is active) matches all hand counts.
- Secret steal is jointly consistent for thief/victim; visible identity only for entitled observers.
- Monopoly and Year of Plenty correctly change counts and knowledge.
- Purchased and played dev cards obey deck conservation and card-age restrictions.
- Identical observable histories yield identical inference/prior when initialized with the same RNG seed.
- Posterior calibration: log-loss, Brier score and reliability curves for affordable-build and hidden-VP events.
- Resource model improves downstream play in an ablation, not just likelihood metrics.

---

## 7. Opponent Intelligence 2.0

### 7.1 Role

For each rival, maintain **capabilities** (what legal moves they plausibly can make), **intentions** (which objectives they may pursue), and **behavior** (what they will actually choose given their *own* information).

Several other systems depend on this one, so it is specified as a shared service:

| Consumer | Uses |
|---|---|
| Trade Intelligence (§8) | `P(accept \| can pay, offer, history)` per responder, counteroffer likelihoods |
| Victory Path Planner (§5) | Opponent objective probabilities and win-by-horizon threats |
| Resource beliefs (§6) | Discard-choice and robber-target likelihoods for weighting hypotheses |
| Search (§10) | Simulated opponent policies inside rollouts |

### 7.2 Interface (proposed contract)

```python
@dataclass(frozen=True)
class OpponentPrediction:
    player_id: PlayerId
    next_action_distribution: dict[str, float]   # canonical action id -> probability
    style_posterior: dict[str, float]            # prior/model name -> posterior weight
    objective_probabilities: dict[str, float]    # PlanObjective id -> P(pursuing)
    win_by_horizon: tuple[float, ...]            # P(this player wins by round h)
    trade_acceptance: dict[str, float]           # offer id -> P(accept | can pay)
    uncertainty: dict
```

Predictions are computed from the opponent's **own information state**, which we approximate by sampling their hands from our belief state (§6). Predictions are never computed from referee truth.

### 7.3 Model progression

1. **Policy mixture (Release 0.4 baseline).** Candidate models are the Catanatron built-in bots, our heuristic agents, and interpretable style priors: OWS, expansion, port, hybrid, resource control, flexible play, and the five trading personalities from §8.5. Maintain a Bayesian posterior over mixture components, using the likelihood of each observed action under each model. Retain uncertainty rather than permanently labeling a player after opening moves.
2. **Learned next-action predictor.** Reuse the shared encoder (§9.1) with a head that predicts the opponent's next action from their perspective. Train on held-out **player-visible** trajectories.
3. **Trade-response model.** Start rule-based from estimated deficit/surplus. Replace it with a learned model once enough trade interactions exist in self-play data.

Keep the simpler stage whenever the learned stage fails to improve downstream play.

### 7.4 Tests

- Next-action log-loss and top-k accuracy on held-out games from policies **not** in the training mixture.
- Style posterior identifies a known bot in synthetic games within a measured number of actions.
- Leakage: identical predictions under indistinguishable hidden swaps (§2.1).
- Calibration of trade-acceptance probabilities against observed responses.
- Downstream ablation: opponent model on/off for trading, threat detection and search.
- Stress tests against holdout policies, adversarial exploiters, humans where permitted, and intentionally irrational opponents. Simulated rivals must not see hidden cards they could not observe.

---

## 8. Trade Intelligence 2.0

### 8.1 Objective

For every legal trading-related action, compare **complete expected continuations** (proposal, response, selection/confirmation, cancellation, different offer, bank/port action, build, or no trade). Evaluate turn order: the active player may act immediately after trade; the off-turn acceptor generally must wait.

A trade that helps an opponent more in raw VP or resources can still be right if it maximizes our **absolute probability of winning** versus *all available alternatives*. Do not hard-code “never trade with the leader” or “never help rivals”; assess immediate-win risks carefully.

### 8.2 Candidate and acceptance model

1. Compute own resource deficits for the best **several** contingent victory plans.
2. Identify actual expendable surplus after accounting for alternative builds.
3. Generate candidate exact trade bundles; prototype with up to two cards each side and broaden later to remain rules-complete.
4. Check proposer ability to pay **at creation and final execution**; filter invalid exchanges, non-resource swaps, gifts and same-resource-only swaps.
5. Predict for every eligible responder `P(can pay)`, `P(accept | can pay, offer, history)`, and strategic consequence if completed.
6. Evaluate **all possible subsets of acceptors**, legal confirmation/cancellation, and counteroffers. Do not assume a specific responder accepts first or that a proposal guarantees a trade.
7. Compare expected value to bank/port exchanges, alternative trades, building and stopping negotiation.

### 8.3 Counterfactual estimator

```text
for each shortlisted offer o:
    for each plausible joint hidden world:
        sample opponents' available resources + response policy
        simulate all legal responses to o, possible counteroffers and follow-on decisions
        for each legally completable deal j:
            estimate full four-player winner distribution from post-trade state
        include no-deal, cancel, and alternative action branches
    aggregate response-weighted win probabilities and uncertainty
choose maximum expected our-win probability subject to legal action constraints
```

Use common-random-number designs for candidate comparisons where valid, but never expose hidden values to the policy. Start with heuristic/neural filters and reserve `64–1024` simulation trials for high-stakes decisions; tune by empirical value and latency.

### 8.4 Negotiation state machine

`IDLE -> OFFERED -> RESPONSES_PENDING -> [CONFIRM_PARTNER | CANCEL | NEW_LEGAL_OFFER] -> IDLE`. The amount of engine work here is set by the Release 0.1 trade audit (`docs/trade-audit.md`). If that audit finds the engine path impractical, any interim bank/port-only benchmark must be labeled as such and never reported as standard Catan. Represent counteroffers explicitly if engine support is incomplete, but do **not** invent an off-turn right to execute a trade. Enforce active player participation and normal trade/build sequencing. Track repeated offers, responder-specific history and search costs; apply **agent-level** loop prevention without changing the game's official rules.

### 8.5 Tests and ablations

Zero invalid transfers and infinite loops; trading observations reveal only real public/private information; immediate opponent-win enablement recognized; response probabilities calibrated; agent wins more often with learned trading than heuristic trading, and improve further with search only if benchmarks show it. Evaluate against five negotiator baselines: cooperative, self-interested, risk-averse, opportunistic, and trade-resistant.

---

## 9. Learning architecture 2.0

### 9.1 Network

**Initial baseline:** PyTorch MLP, e.g. shared trunk of three 256-unit layers, not a fixed production requirement. Replace or augment with board graph neural network if controlled tests justify it.

- **Encoder:** board graph (hexes, nodes, edges, ownership, pips, ports, robber), phase, own resources/dev cards, public history-derived summary, posterior resource features, opponent beliefs and plan features. No private enemy cards.
- **Action encoder:** variable-length candidates with type-specific parameters; legal actions scored by shared `f(state, action)`. For proposals, Trade Intelligence supplies a bounded legal *candidate* set, rather than enumerating impossible enormous combinations.
- **Policy head:** masked softmax over legal candidate actions; training log-probs must reflect the **same** candidate-generation process used at inference. Preserve a complete fallback for every phase so the agent can always act legally.
- **Value head:** four-player eventual winner distribution. Optional `no winner by H` for finite-horizon or truncated tasks.
- **Auxiliary predictions:** opponent next action, build affordability, development-card distribution, strategic objectives. Train only with valid observation-time inputs; privileged labels allowed offline.
- **Memory:** event ledger and belief state are the explicit state baseline. Consider GRU/transformer history encoder only if it adds benchmarked value.

### 9.2 Training progression

| Phase | Method | Deliverable | Promotion evidence |
|---|---|---|---|
| A | Reference-ladder dataset (built-in bots + heuristics) | Legal examples with player-view inputs, privileged labels stored separately, reproducible game results | Schema, leakage and legal-action tests pass |
| B | Behavioral cloning | Strong initial action-conditioned policy | Measured placement on the reference ladder; beats WeightedRandom |
| C | Auxiliary supervised training | Calibrated resource/opponent/value predictions | Held-out reliability and accuracy |
| D | Masked PPO | Self-improving policy | Stronger than imitation-only in held-out four-player tournament |
| E | Population league | Robust self-play against varied strategies and checkpoints | Cross-population gains, no major regressions |
| F | Search-guided data + distillation | Improved policy targeting stronger search decisions | Search helps at fixed compute; distilled policy retains gain |

Phases A–C form **Release 0.2** and use only simple board features. Specialist features from §4–§8 are added in later releases as ablations against this baseline.

**Imitating privileged bots:** `AlphaBetaPlayer` and `ValueFunctionPlayer` receive the full `Game`, so some of their moves may depend on hidden cards. Their actions may be used as **labels** (privileged labels are allowed offline, §2.1), but model **inputs** stay player-view only. Expect a ceiling from moves that cannot be justified from public information; RL in Phase D is what corrects it.

PPO **initial experimental config**, tune rather than assume: `lr=3e-4`, `gamma=0.995`, `GAE lambda=0.95`, `clip=0.2`, `entropy_coef=0.01`, minibatch `256`, 4 epochs per rollout, mandatory legal-action masks. Track decision count and full-game outcomes (not only gradient steps). Terminal reward `1` for winner and `0` for others; treat truncated games separately, never label an unfinished game as a win. Evaluate any shaping carefully against actual win rate.

### 9.3 League design

Sample opponent lineups from: Catanatron built-in bots (Random, WeightedRandom, VictoryPoint, ValueFunction, GreedyPlayouts, MCTS, AlphaBeta), engineered heuristic, six opening archetypes, five trading personalities, historical champions, current champion, and targeted exploitation agents. Randomize **all seats** and board seeds. Avoid training only against copies of the latest model; maintain population diversity and historical snapshots.

Privileged opponents (bots reading the full game state) are acceptable training and evaluation opponents. They do not leak information to *our* agent; they only make the opposition stronger than a fair player. Results against them are reported separately and labeled privileged. An optional **firewalled variant** runs the same bot on a world sampled from its own belief state rather than the true state. That gives a fair-information comparison.

Promotion requires independent held-out evaluation; maintain immutable champion checkpoints and ability to roll back. Separate train/dev/holdout board seeds and opponent lineups. An experiment change is accepted only if it improves full-game outcomes at acceptable compute cost and does not fail legality, leakage or calibration tests.

### 9.4 Compute

Begin CPU simulation benchmarks and modest GPU training; only scale worker count / cloud spending after measuring simulator steps/sec, inference latency, memory footprint and training samples required. These measurements come from ENG-009 in Release 0.1 (`reports/throughput.md`). Catanatron is pure Python, so rollout-heavy designs (§4.4, §8.3, §10.3) must have their budgets derived from measured throughput rather than taken from this plan. Respect hardware differences and CPU/RAM limits in experiment configs. Store experiences in Parquet, models as checkpoint artifacts, configurations/seeds as JSON/YAML, metrics in TensorBoard/CSV. Save enough to reproduce any promoted model.

---

## 10. Search Engine 2.0

### 10.1 Search correctness

Implement information-set-consistent root action selection. Root sampling of hidden states alone is **not a proof** against strategy fusion: after sampling an imagined world, simulated future choices must depend only on what each acting player would observe at that simulated history. Do not allow the root policy or its value estimates to condition on secretly selected worlds as though the root knew them.

Prototype: shallow belief rollouts with fixed root legal actions, multiple determinization samples, observation-respecting simulated policies, and aggregation into robust root action estimates. Compare later against information-set MCTS/POMCP-inspired methods, measuring failures due to strategy fusion, non-locality, multiplayer response complexity, and trading branching.

### 10.2 Safe simulator branching

A rollout branch must have an **independent RNG stream**, independent mutable state, and no future die/dev-deck knowledge available to action policy. Assert:

```text
clone(A) == clone(B) initially;
advancing A does not change B or original;
searching a position does not perturb next real dice outcome;
changing only unobserved truth does not change root choice for fixed belief/history;
identical seed+version+policy produces identical referee replay.
```

Prefer explicit copied RNG state or a verified branch-seed scheme. Do not depend on documentation's `Game.copy()` promises without the above tests.

### 10.3 Adaptive search

Priority: setup placements, critical trades, opponent near win, contested Longest Road/Largest Army, robber decisions, close strategic alternatives. Routine forced or obvious wins use no expensive search. Tune budgets e.g. `64, 256, 1024` simulations, with early stopping only if uncertainty supports it. Report search win-rate improvement **at equal time/compute**, not just raw improvement over a cheaper model.

---

## 11. Tournament and scientific evaluation

### 11.1 Primary protocol

- Standard **4-player** base game, same rules/engine commit for every contender.
- Seat rotation across four positions and varied legal standard boards.
- Fixed independent held-out seed sets and **held-out opponents**. For paired comparisons, coordinate board randomness while avoiding RNG leakage or unrealistic identical future states after divergent moves.
- **Winner = first legal game winner**; log `TRUNCATED` separately. Record opponent pool, matchups, seat, board hash, match seed, engine commit, policy commit, model checkpoint, search budget and complete legal event trace.
- Start with approximately 1,000 four-player games for broad directional evaluations. Increase to 5,000–10,000+ when differences are small or uncertainty demands it; sample size ultimately driven by power analysis, not a magic threshold.
- Report win rate with cluster-aware/bootstrap uncertainty grouped by board/seed structure, seat-stratified performance, finishing VP, turn count, truncations, compute and per-action failure rates.
- When entrants are balanced in a symmetric four-player lineup, 25% is the expected seat-level rate; **do not** treat 25% as the universal baseline for fixed asymmetric pools.
- **Reference ladder:** every release reports results against the Catanatron built-in bots, from Random through AlphaBeta. The upstream claim that `AlphaBetaPlayer` (depth 2) is strongest rests on 25-game 1v1 matches, so ladder order is re-established under this four-player protocol in Release 0.1. Bots that read the full game state are marked **privileged** in every table.

### 11.2 Mandatory ablation matrix

Unless stated otherwise, each row is measured as an addition to the Release 0.2 learned baseline, with everything else held constant.

| Component on/off | Test question |
|---|---|
| Learned baseline vs reference ladder | Where does a plain policy/value model sit before any specialist is added? |
| Six opening archetypes vs pip heuristic | Do engineered priors improve opening value? |
| + opponent-aware draft search | Do remaining-placement forecasts help? |
| + full-game opening rollouts | Is search worth compute? |
| + Victory Path Planner | Does conditional planning improve outcomes? |
| + joint hidden-resource beliefs | Do correct probabilistic estimates help decisions? |
| + opponent model | Does opponent adaptation generalize? |
| + player trading | Is negotiation better than heuristic or no-player-trade policy? |
| + information-set search | Does lookahead outperform policy-only at fixed compute? |
| Full model vs combinations disabled | Are two individually useful modules redundant or harmful together? |

### 11.3 Superhuman claim gate

“Superhuman” requires evaluation against credible **expert humans** in fair, consented, same-rules games with proper seat/board balancing, comparable turn time budgets, independent result verification and meaningful uncertainty intervals. Be explicit about opponents' expertise, trade interfaces and rules. Beating random bots, older 1v1 search players, or one's own league does **not** by itself justify the claim.

---

## 12. Engineering structure

The repository root is this folder (`catan/`); the Python package is `catan_ai`.

**Decision (2026-10-28):** `catan/` is its **own standalone GitHub repository**, independent of the parent `Documents/Coding` repository it currently sits inside. The parent repository must not track this folder. Catanatron is added as a pinned **git submodule** at `vendor/catanatron`, so the exact upstream commit is recorded in this repository's history. Before publishing anything that bundles or modifies Catanatron code, check its GPL-3.0 obligations (§1.3 item 7).

```text
catan/
├── README.md
├── CATAN_AI_MASTER_PLAN.md
├── pyproject.toml
├── uv.lock or requirements.lock (choose one, pin)
├── vendor/
│   └── catanatron/   # pinned upstream git submodule
├── configs/
│   ├── rules_standard_4p.yaml
│   ├── baseline.yaml
│   ├── opening.yaml
│   ├── beliefs.yaml
│   ├── trade.yaml
│   ├── imitation.yaml
│   ├── ppo.yaml
│   ├── search.yaml
│   └── tournament.yaml
├── docs/
│   ├── rules-matrix.md
│   ├── trade-audit.md
│   ├── observation-contract.md
│   ├── experiment-protocol.md
│   ├── architecture-decisions.md
│   └── known-engine-deviations.md
├── src/catan_ai/
│   ├── engine/       # pinned Catanatron adapter, rules audit, deterministic clones
│   ├── observations/ # public/private filter, event schema, ledger replay
│   ├── opening/      # six priors, board analyzer, draft search
│   ├── planning/     # victory dependency graph, timing, defense
│   ├── beliefs/      # resource and development-card posteriors
│   ├── opponents/    # next-action, strategy and acceptance models
│   ├── trading/      # candidate generation, negotiation and counterfactuals
│   ├── models/       # policy/value, action encoder, auxiliary heads
│   ├── training/     # imitation, masked PPO, league, distillation
│   ├── search/       # information-set root search, rollout wrappers
│   ├── decisions/    # final controller and action validation
│   ├── evaluation/   # arena, statistics, ablations, reports
│   └── cli/          # smoke_test, throughput, ladder, evaluate, replay
├── tests/
│   ├── rules/
│   ├── observations/
│   ├── opening/
│   ├── planning/
│   ├── beliefs/
│   ├── trades/
│   ├── search/
│   ├── training/
│   └── integration/
├── experiments/      # committed configs + result manifests, not huge raw data
├── data/             # generated, ignored by git
├── checkpoints/      # generated, ignored by git
└── reports/          # generated, selected summaries versioned (throughput.md, ladder.md, baseline_report.md)
```

Build the dashboard **after** trustworthy referee replays and benchmarks exist. Proposed front end: Next.js with a board replay viewer, action/value comparisons, belief distributions (debug views clearly marked), training progress and tournament reports. Never show private opponent cards in a live human-facing player view.

---

## 13. Versioned delivery plan and go/no-go gates

| Release | Must ship | Exit gate |
|---|---|---|
| **0.1 Rules + Arena** | Pinned engine; four-player referee; exact rule matrix; visibility firewall; action translation; deterministic replay; seat-rotating tournament; throughput benchmark; domestic-trade audit; reference ladder of built-in bots | ≥1,000 reproducible completed or explicitly truncated games; **zero observed** illegal actions/leaks; targeted exhaustive rules tests and branch RNG tests pass; `reports/throughput.md`, `docs/trade-audit.md` (with trading go/no-go) and `reports/ladder.md` published |
| **0.2 Learning baseline** | Player-view encoder with simple board features; action-conditioned policy/value network; imitation from reference-ladder games; auxiliary training; checkpointing | Legal action coverage, calibrated value on holdout, beats WeightedRandom, measured placement on the reference ladder |
| **0.3 Openings + Victory** | Production graph; six priors; pair/draft search; AND/OR victory objectives; basic defense; integrated as features/candidates into the 0.2 model | End-to-end legal openings, valid scoring routes, and statistically supported improvement **over the 0.2 learned baseline** in four-player ablations |
| **0.4 Beliefs + Trade + Opponents** | Public ledger; enumerated hidden-state hypotheses (particle fallback only if needed); dev belief; opponent policy mixture; acceptance and outcome counterfactuals; negotiation scoped by the trade audit | Probability calibration, no impossible hypotheses/particles, no invalid transfers, and game-level ablation gains over 0.3 |
| **0.5 League RL** | Masked PPO, multi-style opponent league including the reference ladder, frozen champion evaluation | Statistically credible improvement over imitation-only and previous champion across holdouts |
| **0.6 Search + Distillation** | Belief-aware branch-safe rollouts, adaptive budget derived from measured throughput, search-target data | Measured improvement at fixed compute; no information-set violations |
| **1.0 Competitive research release** | Robust agent, documented results, dashboard, human trials where available | Beats `AlphaBetaPlayer` and other strong public baselines on the fair four-player protocol (privileged opponents reported separately); expert-level claims only if validated |

**Dependency order:** 0.1 → learned baseline → exact opening board analysis / exact scoring → observation-ledger beliefs → trade + opponent inference → league RL → search. Strong estimates rely on a trustworthy simulator and player-view contract. Every later system needs a learned baseline to be measured against.

### 13.1 Scope control

This plan is sized as a long research program, not a single project, so each release is managed to avoid open-ended work:

- **Timebox each release** when it starts, and record the estimate in `docs/architecture-decisions.md`. If a release runs 50% past its timebox, cut scope back to the minimum that satisfies its exit gate and move the rest to the backlog.
- **Specialists earn their place.** A specialist system (§4–§8) stays in the main agent only if its ablation shows a gain over the current champion at acceptable compute. Otherwise it is archived with its negative result recorded.
- **Ship something playable early.** After 0.2, the agent should be able to complete full four-player games against the ladder, so every later change has an end-to-end measurement.

---

## 14. Prioritized engineering backlog

Each task should become a GitHub issue with an owner, acceptance tests and a reproducible command.

### P0 — Release 0.1: foundation (before any neural training)

- [x] **ENG-001:** Initialize `catan/` as a standalone GitHub repository (§12), Python environment (3.11+), project config, formatter, type checking, pytest and CI.
- [x] **ENG-002:** Add Catanatron as a git submodule at `vendor/catanatron` and **pin** its commit; record Python/dependency versions and license.
- [x] **ENG-003:** Create `docs/rules-matrix.md`; identify the edition Catanatron implements and verify all mechanics in §1. Patches for its deviations land in ENG-004 (adapter) and ENG-007 (RNG).
- [x] **ENG-004:** Implement action adapter with exhaustive legal-mask/offer-affordability tests (`src/catan_ai/engine/`).
- [x] **ENG-005:** Implement full truth referee event log + strictly filtered per-player observations (`src/catan_ai/observations/`, `docs/observation-contract.md`).
- [x] **ENG-006:** Write hidden-state indistinguishability and private-card leak tests (`tests/observations/`).
- [x] **ENG-007:** Verify independent branch RNG and non-perturbation of real-game future rolls (`EngineAdapter.branch`).
- [x] **ENG-008:** Build deterministic four-player arena with seat rotation, JSON results and replay (`src/catan_ai/arena/`, `python -m catan_ai.cli.arena`).
- [x] **ENG-009:** Throughput benchmark: games/sec for each built-in bot, decisions/sec, `Game.copy()` cost, firewall overhead → `reports/throughput.md`; derive search/rollout budget defaults in `configs/`.
- [x] **ENG-010:** Domestic-trade audit: available trade actions, whether built-in bots trade, missing lifecycle pieces, patch-scope estimate and go/no-go for Release 0.4 → `docs/trade-audit.md`.
- [x] **ENG-011:** Reference ladder: four-player seat-rotated matches across Random, WeightedRandom, VictoryPoint, ValueFunction, GreedyPlayouts, MCTS and AlphaBeta; mark privileged bots; report truncation separately → `reports/ladder.md`.

### P1 — Release 0.2: learning baseline

- [x] **ML-012:** Frozen imitation dataset from ladder games, with player-view-only inputs and privileged labels isolated.
- [x] **ML-013:** Action-conditioned PyTorch model with legal masking and fallback, behind a common policy/value evaluation interface (with a deterministic heuristic implementation of the same interface).
- [x] **ML-014:** Policy/value and auxiliary training, loss/evaluation dashboards.
- [x] **ML-015:** Evaluate the learned baseline on the reference ladder; it becomes the first champion.

### P2 — Release 0.3: strategic foundations

- [x] **STR-016:** Board graph/pip-production/port/road-network feature extractor.
- [x] **STR-017:** Six opening strategy priors, comparison against pip heuristic.
- [x] **STR-018:** Opening settlement+road pair candidate search and remaining snake-draft prediction.
- [x] **STR-019:** Exact VP, city, road, army and dev timing tests.
- [x] **STR-020:** Victory AND/OR dependency graph with branch contingencies.
- [x] **STR-021:** Defensive action and turn-order threat tests.

### P3 — Release 0.4: intelligence and trading

- [x] **BEL-022:** Event-sourced exact resource/development-card ledger (Tier 1).
- [x] **BEL-023:** Enumerated joint hypotheses for ambiguous events, with conservation assertions (Tier 2).
- [x] **BEL-024:** Particle fallback, collapse detection/recovery and calibration harness (Tier 3). Build only if tier-usage data from BEL-023 shows enumeration exceeds budget.
- [ ] **OPP-025:** Opponent interface (§7.2) and policy-mixture baseline; then learned next-action and trade-response predictors. Interface, mixture and simulated trade responses done (`reports/opponents.md`); learned trade-response predictor open.
- [x] **TRD-026:** Candidate proposal generator and bank/port fallback.
- [x] **TRD-027:** Domestic trade state machine, rule-lawful counteroffers and loop guards, scoped by ENG-010.
- [x] **TRD-028:** Joint response/counterfactual trade evaluation, immediate-win threat tests.

### P4 — Releases 0.5–1.0: league, search and evaluation

- [ ] **ML-029:** Masked PPO worker/learner, checkpoint league, champion promotion.
- [ ] **SRCH-030:** Observation-consistent root sampling and independent search branches.
- [ ] **SRCH-031:** Adaptive search budgets, search-vs-no-search ablations and distillation.
- [ ] **EVAL-032:** Held-out tournaments, statistical reports, human-expert protocol and replay UI.

### Definition of done (tiered)

A change is not “done” because it runs once. Requirements scale with risk:

| Tier | Applies to | Required |
|---|---|---|
| **1: Correctness-critical** | Engine adapter, rules, observations/firewall, beliefs, search, decision controller, anything touching legality or hidden information | Unit/integration tests, proof it cannot leak hidden state, deterministic seed/config artifact, docs |
| **2: Strength claims** | Features, models, specialist modules, training changes | Tests, deterministic seed/config artifact, and a game-level comparison to the simplest viable alternative before it is kept |
| **3: Tooling** | Scripts, reports, CLI, dashboards, docs | Works, has a reproducible command, reviewed |

Regressions must be diagnosed before champion promotion.

---

## 15. First implementation sprint (concrete sequence)

The following commands are a **bootstrap reference**, not evidence they were executed or that current upstream dependencies are compatible until run locally.

Primary development machine is Windows, so PowerShell comes first, run from the repository root (this folder). Catanatron requires **Python 3.11+**.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Engine is a pinned submodule (see §12).
git submodule add https://github.com/bcollazo/catanatron.git vendor/catanatron
git -C vendor/catanatron rev-parse HEAD     # save output in docs/architecture-decisions.md
python -m pip install -e "vendor/catanatron[gym,dev]"
python -m pip install torch numpy pandas pyarrow pytest
catanatron-play --players=R,R,R,R --num=10  # upstream sanity check
catanatron-play --help-players              # confirm bot codes for the ladder
```

Bash equivalent: `python3 -m venv .venv`, `source .venv/bin/activate`; the remaining commands are unchanged. A fresh clone of this repository needs `git submodule update --init` to fetch the engine. Upstream documents the extras `[web,gym,dev]`. Re-check the package layout and extras at the pinned commit, because the repository has been restructured over time.

**Sprint 1 deliverables, in order:**

1. `smoke_engine.py`: instantiate four `RandomPlayer`s, run at least 10 games and capture winner/turn count/truncation/seed; use known documented Catanatron APIs.
2. `rules_standard_4p.yaml`: four seats, VP target 10, friendly robber disabled, verified standard map/number placement, pinned rules edition, and no optional variants.
3. `throughput.py`: ENG-009 measurements → `reports/throughput.md`. Run it early, because the numbers shape every later budget.
4. `docs/trade-audit.md`: ENG-010 code inspection of the trade lifecycle at the pinned commit, with patch-scope estimate. This is a document; no engine changes yet.
5. `engine_adapter.py`: canonical action representation, legal moves and safe execution; isolate any engine patches behind one interface.
6. `event_recorder.py` and `observation_firewall.py`: actual test fixtures for hidden cards/steals. **This precedes training.**
7. `arena.py`: N matches with fixed seeds, 4-seat rotation, player-pool configs, structured CSV/JSON output, deterministic replay.
8. `test_clone_rng.py`: prove search branch and real-state RNG independence.
9. `reports/ladder.md` and `baseline_report.md`: four-player reference ladder plus 1,000-game baseline with truncation and rule-test pass/fail matrix. Privileged bots are labeled, and nothing here is claimed as our agent's strength.

**Suggested CLI contract (our future app, not upstream Catanatron's CLI):**

```powershell
python -m catan_ai.cli.smoke_test --games 10 --seed 42
python -m catan_ai.cli.throughput --games 200 --seed 42
python -m catan_ai.cli.audit_rules --config configs/rules_standard_4p.yaml
python -m catan_ai.cli.ladder --config configs/tournament.yaml --games 1000
python -m catan_ai.cli.arena --config configs/tournament.yaml --games 1000
python -m pytest tests/rules tests/observations tests/integration -q
```

Don't start PPO until the rules audit, deterministic replays, and observation-contract tests pass.

---

## 16. Risks and research decisions

| Risk | Impact | Mitigation |
|---|---|---|
| Engine deviates from chosen rules edition | Entire training distribution wrong | Baseline edition (2025 6th) chosen to match the engine; rule matrix with strict-xfail deviation tests; minimal fork/adapter |
| Built-in reference bots read full game state | Ladder results overstate fair-play opponent strength or get misreported | Label privileged bots in every report; optional firewalled variants on belief-sampled worlds |
| Domestic trading needs substantial engine work | Release 0.4 slips or trading is quietly simplified | ENG-010 audit with patch-scope estimate and go/no-go in 0.1; any bank/port-only benchmark explicitly labeled non-standard |
| Scope exceeds available capacity | Program never reaches a playable agent | Timeboxed releases (§13.1), learned baseline in 0.2, specialists kept only on ablation evidence |
| Imitating privileged bots | Policy learns moves unjustifiable from public information | Player-view inputs only; treat imitation as a starting point; RL and leakage tests correct it |
| Raw game state leaks private cards | False “superhuman” performance | Per-player firewall and indistinguishability tests |
| RNG copies share state or search sees future draws | Nonreproducible/cheating search | Explicit RNG cloning and nonperturbation assertions |
| Hidden-state inference wrong (hypothesis blow-up or particle collapse) | Confidently wrong trade/robber choices | Exact-first tiers (§6.2), hypothesis caps from measured latency, ESS diagnostics, rejuvenation, constrained recovery |
| Trading implementation incomplete | Unrepresentative full-game strategy | Patch negotiation and validate complete standard rules |
| Opening overfits human heuristics | Learns conventional rather than optimal play | Priors only; sampled search; neural policy can override |
| Value model assumes independent player win events | Probabilities mathematically inconsistent | Joint 4-player outcome distribution / 5 finite-horizon classes |
| Search strategy fusion | Uses inaccessible hidden world at decision | Information-set-consistent policies and leakage tests |
| Self-play cycles / exploits weak styles | Poor human generalization | Diverse league, historical champions, exploitation adversaries |
| PPO stalls or shaping distorts goals | Weak end-to-end agent | Compare algorithms, terminal reward focus, ablations |
| Huge simulation compute costs (pure-Python engine) | Slow/expensive progress; search budgets unaffordable | Measure in 0.1 (ENG-009) and derive budgets from it; adaptive search; distillation |
| Apparent benchmark gains from lucky boards | Unsupported strength claims | Held-out tests, seat rotation, clustered CIs, preregistered tests |
| Catanatron GPL/trademark considerations | Distribution constraints | License review before any commercial/released derivative |

### Experiment decision rules

Do not tune hyperparameters against final holdout. Register each experiment with seed set, engine/rule version, opponent lineup, network/feature ablations and compute budget. Prefer the **simplest** algorithm that clears a promotion gate. Record negative results, not just the champion.

---

## 17. Deliverables to create during implementation

1. `docs/rules-matrix.md` — canonical mechanic tests, edition choice, engine deviations and open interpretation questions.
2. `docs/trade-audit.md` — domestic-trade lifecycle findings, patch scope and go/no-go.
3. `docs/observation-contract.md` — exact public/own-private field semantics and redaction tests.
4. `docs/experiment-protocol.md` — seed partitioning, opponent pools, metrics and confidence intervals.
5. `docs/architecture-decisions.md` — ADRs with date, context, choice, alternatives, evidence; release timeboxes.
6. `reports/throughput.md` — measured simulator and clone throughput, derived budgets.
7. Reproducible reference matches + four-player reference ladder (`reports/ladder.md`) with privileged bots labeled.
8. Opening Engine 2.0 and Victory Path Planner 2.0 interface tests.
9. Belief / trade / opponent inference accuracy and calibration reports.
10. Versioned learned policy checkpoints and reproducible RL experiment manifests.
11. Information-set-safe search experiments and compute-quality Pareto curves.
12. Optional replay/research dashboard once trustworthy data exists.

---

## 18. Primary references and verification notes

- **Official game rules portal:** https://www.catan.com/understand-catan/game-rules
- **Official base game FAQ:** https://www.catan.com/faq/basegame
- **Official 2025 base-game rulebook (linked from rules portal):** https://www.catan.com/sites/default/files/2025-03/CN3081%20CATAN%E2%80%93The%20Game%20Rulebook%20secure%20%281%29.pdf
- **Catanatron repository:** https://github.com/bcollazo/catanatron
- **Catanatron API documentation:** https://catanatron.readthedocs.io/en/latest/catanatron.html
- **Catanatron model/action documentation:** https://catanatron.readthedocs.io/en/latest/catanatron.models.html
- **Catanatron Gymnasium interface documentation:** https://docs.catanatron.com/advanced/openapi (the URL slug is misleading; the page is the Gymnasium guide)
- **Catanatron CLI documentation (bot codes):** https://docs.catanatron.com/using-catanatron/interactive-blocks
- **Catanatron bot leaderboard (“Making Catanatron Stronger”):** https://docs.catanatron.com/advanced/making-catanatron-stronger
- **Catanatron license:** https://github.com/bcollazo/catanatron/blob/main/LICENSE

**Verified from public documentation at planning time:** Catanatron supports four-player `Game`, custom `Player.decide(game, playable_actions)`, `play_tick`, a `copy` method, an `is_valid_trade` helper, and a Gymnasium interface. The custom player receives the complete game object; the docs alone do **not** prove correct player-specific observation filtering. Trading semantics and RNG branch behavior require exact-commit code inspection and tests, even if earlier reviews suggested known issues. The Gymnasium example should **not** be treated as proof of a full four-player multiplayer RL environment with complete domestic negotiation.

**Additionally checked 2026-10-28 (upstream README and docs):**

- Requires Python 3.11+; install via `pip install -e .` with optional extras `[web,gym,dev]`; CLI entry point `catanatron-play`.
- Built-in bots documented: Random (`R`), VictoryPoint (`VP`), ValueFunction (`F`), GreedyPlayouts (`G:n`), AlphaBeta (`AB:depth:prune`); WeightedRandom and MCTS appear on the leaderboard. Codes `W` and `H` appear in examples without description; confirm with `catanatron-play --help-players`.
- The docs state “The best bot right now is `AlphaBetaPlayer` with n = 2.” Leaderboard entries are 1v1 with 15–1000 games per pairing (25 for the top pairings). That is too few to fix four-player ordering, so the ladder is re-measured (ENG-011).
- Gymnasium environment id `catanatron/Catanatron-v0`; example uses MaskablePPO with an action mask and configurable opponents. It is single-agent against configured bots.
- Neither the README nor the docs pages checked mention player-to-player trading. Trade support is unverified until ENG-010.

**No results claimed:** Nothing in this plan represents a completed Catanatron integration, trained model, validated tournament, or measured win rate.

---

## 19. First release decision

**Build Release 0.1 now.** Its purpose is to produce a trustworthy four-player simulation and evaluation foundation, not to write a sophisticated neural network immediately. The success criteria are:

- a pinned, rules-correct engine;
- lawful player observations;
- isolated randomness;
- replayable games and fair tournaments;
- measured throughput;
- a trade audit;
- a four-player reference ladder.

Next comes a plain learned policy/value baseline (Release 0.2). Then the Opening Strategy Engine and Victory Path Planner, followed by resource/opponent/trade intelligence, league RL and search. Each is kept only if it beats the current champion.

**Guiding principle:** Learn what makes a Catan move win, rather than force the AI to follow our existing definitions of “good” strategy.
