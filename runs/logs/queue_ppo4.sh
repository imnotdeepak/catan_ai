#!/bin/bash
# After the search-vs-plain diagnostic, PPO ppo-4 from the search champion's checkpoint
# (ppo-3 policy, calibrated value-ppo3 outcome tower) against a strong-opponent league.
until grep -q "wrote\|Traceback" runs/logs/search-vs-plain-AB-F.log 2>/dev/null; do sleep 60; done
echo "diagnostic: $(grep 'candidate share' runs/logs/search-vs-plain-AB-F.log) $(date +%H:%M)"
uv run python -m catan_ai.cli.train_ppo --name ppo-4 --init champions/robber-ppo3-r60-value-ppo3.pt \
  --league configs/league_strong.json --iterations 60 --games 220 --workers 10 \
  --snapshot-every 10 --seed 980000 --lr 0.0001 --entropy 0.005 --anchor 0.02 \
  > runs/logs/ppo4.log 2>&1
echo "ppo-4 exited $? $(date +%H:%M)"
