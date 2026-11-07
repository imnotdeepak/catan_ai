#!/bin/bash
# ppo-4 snapshots (plain play) against the plain champion at an AB:2 + F table.
M=champions/robber-ppo3-r60.pt
for s in 60 30; do
  uv run python -m catan_ai.cli.compare --candidate model:runs/ppo/ppo-4/snap-$s.pt --baseline model:$M \
    --opponents catanatron:AB:2,catanatron:F --label ppo-4-snap-$s-vs-robber-AB-F --workers 10 \
    > runs/logs/ppo4-snap-$s-AB-F.log 2>&1
  echo "snap $s: $(grep 'candidate share' runs/logs/ppo4-snap-$s-AB-F.log) $(date +%H:%M)"
done
