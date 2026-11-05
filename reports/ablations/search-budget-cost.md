# Ablation: search budget against strength (Release 0.6)

Search with the calibrated value tower (`champions/robber-ppo3-r60-imitation-value.pt`) at
several budgets, each against the plain champion `model:champions/robber-ppo3-r60.pt` with
`catanatron:F`, `catanatron:F` opponents: 800 games each (100 boards from seed 500000, 8 seatings).
Budget `<worlds>t<turns>`: sampled belief worlds, and rollout horizon in turns; 3 candidates.

| Budget | Share of the pair's wins | 95% interval (boards) | ms per searched decision | ms per decision (avg) | Report |
|---|---|---|---|---|---|
| 4t8 | **63.8%** (301/472) | 59.5% to 68.1% | 617 | 135 | `search-4t8-imitation-value-vs-plain-F.md` |
| 2t8 | 59.5% (267/449) | 54.8% to 64.0% | 355 | 115 | `search-2t8-imitation-value-vs-plain-F.md` |
| 2t4 | 58.4% (265/454) | 54.3% to 62.5% | 208 | 67 | `search-2t4-imitation-value-vs-plain-F.md` |
| 4t2 | 57.1% (260/455) | 52.5% to 61.8% | 347 | 121 | `search-4t2-imitation-value-vs-plain-F.md` |
| 1t8 | 56.6% (248/438) | 51.9% to 61.2% | 186 | 58 | `search-1t8-imitation-value-vs-plain-F.md` |

All five are significantly better than plain play (p <= 0.006). Each share has a margin of
about +-4.5 points, so neighbouring rows are not separated. Plain play costs about 2 ms per
decision.

Timings come from `runs/logs/time_budgets.py`: 4 games per budget (seeds 700000-700003,
lineup [search, plain, F, F]), five budgets timed in parallel processes, so they are rough.
Fit per searched decision: about worlds x (30 ms + 18 ms per turn). Each world pays a fixed
cost (sampling, branching, leaf evaluation), so 4t2 costs about as much as 2t8 and is weaker.

**Reading.** Strength rises with budget, with no plateau yet at 4t8. 2t4 is the cheap
choice: about a third of 4t8's cost per searched decision, keeping about 58%. For a fixed
cost, fewer worlds with a longer horizon (2t8) did better than more worlds with a short one
(4t2), and 2t4 did a little better than 1t8 at the same cost; both gaps are within the noise.
