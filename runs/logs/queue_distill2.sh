#!/bin/bash
# Release 0.6 distillation, second attempt: teacher = 4t8 search with the recalibrated value tower.
M=champions/robber-ppo3-r60.pt
D=checkpoints/distill-vppo3-4t8/best.pt
uv run python -m catan_ai.cli.imitation_data --config configs/distill_vppo3.json --workers 10 \
  --feature-version 3 > runs/logs/distill2_data.log 2>&1 || { echo "data failed"; exit 1; }
echo "data done $(date +%H:%M)"
uv run python -m catan_ai.cli.train_imitation --config configs/distill_vppo3.json \
  --report reports/distillation_training_vppo3.md > runs/logs/distill2_train.log 2>&1 || { echo "train failed"; exit 1; }
echo "train done $(date +%H:%M)"
uv run python -m catan_ai.cli.compare --candidate model:$D --baseline model:$M \
  --label distill-vppo3-4t8-vs-champion --workers 10 > runs/logs/distill2_cmp.log 2>&1
echo "W/VP: $(grep 'candidate share' runs/logs/distill2_cmp.log) $(date +%H:%M)"
uv run python -m catan_ai.cli.compare --candidate model:$D --baseline model:$M \
  --label distill-vppo3-4t8-vs-champion-F --opponents catanatron:F,catanatron:F --workers 10 > runs/logs/distill2_cmp_F.log 2>&1
echo "F: $(grep 'candidate share' runs/logs/distill2_cmp_F.log) $(date +%H:%M)"
