# Openings: Priors, Learned Priors and Pair Search (STR-017, STR-018)

Generated on 2026-10-28 from the head-to-head ablations in `reports/ablations/`
(`python -m catan_ai.cli.compare`). catan_ai `83e4bf4`, engine `25f3599`.

Each candidate is `opening:<rule>:model:champions/imitation-v1.pt`: it places
both initial settlements by the rule and makes every other decision, initial
roads included, with the 0.2 champion. The candidate and the baseline play in
the same games against `catanatron:W` and `catanatron:VP`. Each comparison
covers 500 boards from seed 500000, with 8 seatings per board, for 4,000 games.
Every comparison uses the same boards. The share is the candidate's share of
the games won by either of the two. The interval resamples whole boards.
p-values are exact two-sided binomial tests. No games had errors.

## Results

| Rule | vs pip baseline | 95% interval | vs champion's own opening | 95% interval |
|---|---|---|---|---|
| `pips` (baseline) | — | — | 44.2% (worse) | 42.2–46.1% |
| `balanced` | **64.2% (better)** | 62.3–66.0% | **59.1% (better)** | 57.5–60.8% |
| `hybrid` | 50.2% (p = 0.83) | 48.0–52.4% | 44.7% (worse) | 42.6–47.0% |
| `road` | 48.3% (p = 0.048) | 45.9–50.7% | 44.3% (worse) | 42.1–46.3% |
| `cartel` | 36.6% (worse) | 34.6–38.7% | 33.1% (worse) | 31.0–35.1% |
| `port` | 26.8% (worse) | 24.7–29.0% | 22.1% (worse) | 20.1–24.1% |
| `ows` | 20.9% (worse) | 19.0–22.8% | 17.0% (worse) | 15.3–18.7% |

Every p-value is below 0.0001 unless shown otherwise.

`balanced` also beats the champion against a stronger field. With opponents
`catanatron:F` and `catanatron:F`, over 300 boards (2,400 games), it took 60.5%
of the pair's wins, with a 95% interval of 56.5–64.5%. See
`reports/ablations/opening-balanced-vs-F.md`.

## Reading

- **Resource coverage matters more than raw pips.** `balanced` rewards
  production of every resource, and it is the only rule that beats both pure
  pips and the champion's learned opening. Pure pips loses to the champion,
  which means the network already learned more than pip counting from its
  teachers.
- **Specialized openings lose by large margins under this downstream policy.**
  `ows`, `port` and `cartel` concentrate production. These results measure
  each rule paired with the champion's mid-game play. The champion learned that
  play from teachers' games, so it may not know how to exploit a specialized
  opening. The losses are evidence against these rules as fixed openings for
  this policy, not against the strategies in general. The priors stay useful
  as features: the model can learn when they apply.
- **Move order is a large effect.** For every rule, the candidate's share is
  higher when it moves before the baseline (`balanced`: 69.3% first, 59.0%
  second). The 8-seating design balances this. Any opening search (STR-018)
  needs to model the remaining snake draft.

## Consequences

- `balanced` is the strongest opening signal so far. A wrapped champion
  already shows a statistically supported gain over the 0.2 baseline, which
  is the Release 0.3 exit criterion. It is a candidate for the 0.3 agent and a
  benchmark for STR-018's pair search.
- Next: give the model the six prior scores as features (a later feature
  version) instead of fixing any one rule. Done below.

## Learning from the priors (features v3)

Feature version 3 adds the six prior scores to each settlement candidate.
Version 2 adds the board analysis (STR-016). Both models were trained on the
same imitation games as the champion: `data/imitation-v3`, re-encoded from its
replays. Each training setup was run with two seeds. Every comparison is
4,000 games, played against the 0.2 champion with `W` and `VP` as opponents.

| Model | Seed | Holdout top-1 | Share vs champion | 95% interval |
|---|---|---|---|---|
| features v2 | 0 | 71.4% | 49.1% (p = 0.28) | — |
| features v2 | 1 | 67.8% | 44.7% (worse) | 43.0–46.4% |
| **features v3** | 0 | 69.7% | **63.1% (better)** | 61.4–64.8% |
| **features v3** | 1 | 72.6% | **63.1% (better)** | 61.3–64.9% |

The features-v3 model (seed 0) passed two further checks:

- Against `catanatron:F` and `catanatron:F` as opponents, over 300 boards,
  it took 56.9% of the pair's wins (p = 0.001).
- Against the champion wrapped in the `balanced` rule, it took 54.6% of the
  pair's wins (p < 0.0001). The model learned more than that one rule.

Teacher agreement does not predict playing strength: the two v2 seeds
differ by 4 points of top-1, and neither one beats the champion. The board
analysis alone does not help. The prior scores do, by the same margin under
both seeds. **The features-v3 model (seed 0) is promoted to champion as
`champions/imitation-features-v3.pt`.**

## Pair search with draft prediction (STR-018)

`draft:<rule>:<agent>` searches the first settlement over the predicted
snake draft (`catan_ai.opening.draft`). It takes the top 12 sites by rule. For
each, it predicts the other seats' picks up to its second turn, assuming each
takes the highest-pip free site. It keeps the site whose best pair scores
highest. Initial roads point to the best site left after the predicted draft.
`draft-settle:` searches settlements only. All runs below use the 0.2
champion as the inner agent and `balanced` as the rule, over 4,000 games.

| Candidate | Baseline | Share | p |
|---|---|---|---|
| `draft:balanced` | champion | 58.9% (better) | < 0.0001 |
| `draft:balanced` | `opening:balanced` | 50.0% | 0.99 |
| `draft-settle:balanced` | `opening:balanced` | 50.7% | 0.41 |
| `draft:balanced` (search roads) | `draft-settle:balanced` (champion's roads) | 50.7% | 0.38 |

The search adds nothing over the greedy `balanced` rule in this field. The
prediction is why (`reports/draft_prediction.md`). `W` and `VP` place close to
at random: every rule's top site matches 2–3% of their picks. The draft the
search plans for rarely happens. The champion is predictable. `balanced`
names its first-round pick 91% of the time, but its second-round pick only
27%, which is where the rule changes its play. The search remains available
for fields of deliberate placers. Its value there is unmeasured.
