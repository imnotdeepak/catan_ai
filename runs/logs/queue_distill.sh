#!/bin/bash
# SRCH-031 distillation: fine-tune the champion on search decisions, then test it without search.
M=champions/robber-ppo3-r60.pt
D=checkpoints/distill-ppo3-4t8/best.pt
uv run python -m catan_ai.cli.train_imitation --config configs/distill.json \
  --report reports/distillation_training.md > runs/logs/distill_train.log 2>&1 || exit 1
uv run python -m catan_ai.cli.compare --candidate model:$D --baseline model:$M \
  --label distill-4t8-vs-champion --workers 10 > runs/logs/distill_cmp.log 2>&1
echo "W/VP: $(grep 'candidate share' runs/logs/distill_cmp.log)"
uv run python -m catan_ai.cli.compare --candidate model:$D --baseline model:$M \
  --label distill-4t8-vs-champion-F --opponents catanatron:F,catanatron:F --workers 10 > runs/logs/distill_cmp_F.log 2>&1
echo "F: $(grep 'candidate share' runs/logs/distill_cmp_F.log)"
