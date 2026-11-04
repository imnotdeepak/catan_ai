#!/bin/bash
# SRCH-031 budget allocation: same cost (4t8, 16t2) and double cost (16) vs the plain policy.
M=champions/robber-ppo3-r60.pt
for b in 4t8 16t2 16; do
  uv run python -m catan_ai.cli.compare --candidate search:$b:model:$M --baseline model:$M \
    --label search-$b-vs-plain --workers 10 > runs/logs/search-$b.log 2>&1
  echo "done $b: $(grep 'candidate share' runs/logs/search-$b.log)"
done
