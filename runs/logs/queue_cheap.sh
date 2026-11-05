#!/bin/bash
# Release 0.6 speed/strength: cheaper search budgets (calibrated values) vs plain, F opponents.
C=champions/robber-ppo3-r60-imitation-value.pt
M=champions/robber-ppo3-r60.pt
for b in 2t8 1t8 2t4 4t2; do
  uv run python -m catan_ai.cli.compare --candidate search:$b:model:$C --baseline model:$M \
    --opponents catanatron:F,catanatron:F --label search-$b-imitation-value-vs-plain-F \
    --workers 10 > runs/logs/search-$b-imv-F.log 2>&1
  echo "done $b: $(grep 'candidate share' runs/logs/search-$b-imv-F.log) $(date +%H:%M)"
done
