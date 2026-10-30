# Release 0.4 Evaluation: Beliefs, Trade and Opponents

Release 0.4, plan §6 to §8. Exit gate (plan §12): probability calibration, no impossible hypotheses or particles, no invalid transfers, and game-level ablation gains over 0.3. All four are met; details and caveats below.

| Exit criterion | Result | Evidence |
|---|---|---|
| Probability calibration | **Met**, with a small known bias | Resource beliefs ECE 0.049; development-card beliefs ECE 0.034; trade acceptance Brier 0.062 (model) and 0.078 (rules) |
| No impossible hypotheses or particles | **Met** | 950,484 belief updates over 740 holdout games: the true hands were never ruled out; tier 3 used 0.00% of the time; 0 collapse recoveries |
| No invalid transfers | **Met** | 3,400 trading games across the reports and ablations below, 0 errors, 0 truncations; card conservation asserted in `tests/trading` |
| Game-level gain over 0.3 | **Met** at tables that trade; neutral against the trade-blind ladder | Trading champion takes 90.5% of the pair's wins against the 0.3 champion with negotiating opponents; 50.0% with two `catanatron:F` |

## Beliefs (BEL-022 to BEL-024)

`reports/beliefs.md`. Every seat tracked at every decision of the `imitation-v3/holdout` games.

- **Tiers:**
  - The exact ledger holds 41.5% of the time. Otherwise unseen steals are enumerated: 4.8 hypotheses on average, 46 at p99, 367 at most, against a budget of 4,096.
  - The particle fallback is built and tested but was never needed. An update takes 0.07 ms on average.
- **Accuracy:** the expected opponent hand is off by 0.57 cards on average, against 4.44 cards for spreading the card count evenly.
- **Calibration:**
  - *Resources (ECE 0.049).* Uncertain resource probabilities are slightly underconfident in every band: opponents hold the resource 3 to 9 points more often than predicted. The likely cause is that hypotheses are weighted only by the random steal, not by what the player then chose to do (a player that does not build a city probably cannot). Weighting hypotheses by the opponent model's action likelihoods (plan §6.2) is the planned fix.
  - *Development cards (ECE 0.034).* The deck odds alone are badly calibrated for VP cards (ECE 0.38), because unplayed cards are disproportionately VP cards. A hold-time likelihood fixes this. It has a play rate of 0.90 per playable card-turn and a hoarded share of 0.10, fitted on `dev` and evaluated on `holdout`.

## Opponent model (OPP-025)

`reports/opponents.md`.

- **Mixture:**
  - The opponent model is a Bayesian mixture of six policies: uniform, weighted random, victory-point, value function, our heuristic and the 0.3 champion network.
  - Each policy is scored in worlds sampled consistently with the viewer's beliefs (`catan_ai.beliefs.worlds`).
  - There is also a posterior over opening styles.
- **Prediction against `catanatron:F`:** the mixture predicts its next action with a log loss of 0.79 nats over about 9 legal choices. It ranks the actual choice first 74% of the time.
- **Held-out `catanatron:AB:2`:** AlphaBeta is not a mixture component, but the mixture still predicts it at 1.19 nats.
- **Identification:**
  - The mixture identifies random-like policies well.
  - It splits weight between the value-function component and the champion network for `catanatron:F`, which is expected, since the network imitates F.
- **Still open in OPP-025:** a trade-response predictor learned from data. The model trader simulates responders in sampled worlds instead.

## Trading (TRD-026 to TRD-028)

`reports/trading.md`, `reports/trading_2.md`.

- **Proposals:**
  - The victory planner scores what a trade is worth: the expected rounds it saves.
  - Offers trade up to two cards for up to two cards.
  - They are compared against the best bank or port trade.
- **Negotiation:**
  - Responders may attach a public counteroffer to a rejection (D-08), at the adapter level, so every completed trade still goes through the engine's offer, accept and confirm path.
  - A loop guard allows at most two offers per turn and no repeated offer.
- **Win refusal:** a trader never supplies a card that lets the other side win now. This is covered by a tactical test.
- **Trading agents:**
  - The rules trader predicts acceptance from the responder's judged benefit and its ability to pay.
  - The model trader re-scores its shortlist by simulating each responder's answer under the opponent model.
- **Negotiator baselines:** five styles, as trading partners and opponents: cooperative, self-interested, risk-averse, opportunistic and trade-resistant.
- **Activity:** 75 to 86 offers, 20 to 27 completed trades and 48 counteroffers per game.

## Game-level ablations

Each comparison plays the candidate and the baseline against the same two opponents, over every board with eight seatings. The pair's win share is the candidate's share of the games either of them won.

| Candidate | Baseline | Opponents | Games | Candidate share | 95% interval (boards resampled) | Verdict | Report |
|---|---|---|---|---|---|---|---|
| 0.3 champion + rules trading | 0.3 champion | cooperative + self-interested negotiators | 1,200 | **90.5%** | 88.5% to 92.5% | better | `ablations/trader-rules.md` |
| model trader | rules trader | opportunistic + risk-averse negotiators | 1,200 | **54.2%** | 51.5% to 57.0% | better (p = 0.012) | `ablations/trader-model-vs-rules.md` |
| 0.3 champion + rules trading | 0.3 champion | `catanatron:F` × 2 | 800 | 50.0% | 48.3% to 51.7% | no difference | `ablations/trader-rules-vs-F.md` |

### What the ablations show

- **Trading wins at a trading table.** The plain 0.3 champion learned from F, which never accepts a trade. So it refuses almost every offer and falls behind three players that trade with each other: it won 6% of the games, against 61% for the trading champion.
- **Trading costs nothing against the trade-blind ladder.** No one accepts, the trader falls back to normal play, and the result is an exact tie (99 wins each).
- **Better acceptance predictions win more games.** The model trader beats the rules trader overall, but seat order matters a lot:
  - Its share is 60.8% when it moves just before the rules trader and 47.5% when it moves just after.
  - The trader moving first gets the first offers in at partners whose hands are not yet spent.
  - This also explains the small 4-player reports, where the rules trader always sat just before the model trader and won more.
  - The ablation averages both orders, so its 54.2% is the fair comparison.
- **Speed:** the model trader takes about 45 s of decision time per game, against 12 s for the rules trader.

### Caveats

- **Partners are our own agents.** Trading partners are our own negotiator baselines, which trade far more readily than strong humans. The size of the gain against human traders is unknown.
- **Champion unchanged.** The ladder champion stays `champions/imitation-features-v3.pt`. Trading is a wrapper (`trader:<rules|model>:<spec>`) and makes no difference against bots that do not trade.
