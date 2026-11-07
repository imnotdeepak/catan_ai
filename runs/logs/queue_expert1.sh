#!/bin/bash
# Expert iteration cycle 1: search-target data, policy training, then the student without search
# against the plain champion at an AB:2 + F table and a W/VP table.
M=champions/robber-ppo3-r60.pt
E=checkpoints/expert-1/best.pt
uv run python -m catan_ai.cli.imitation_data --config configs/expert_1.json --workers 10 \
  --feature-version 3 > runs/logs/expert1_data.log 2>&1 || { echo "data failed"; exit 1; }
echo "data done $(date +%H:%M)"
uv run python -m catan_ai.cli.train_imitation --config configs/expert_1.json \
  --report reports/expert_1_training.md > runs/logs/expert1_train.log 2>&1 || { echo "train failed"; exit 1; }
echo "train done $(date +%H:%M)"
uv run python -m catan_ai.cli.compare --candidate model:$E --baseline model:$M \
  --opponents catanatron:AB:2,catanatron:F --label expert-1-vs-robber-AB-F --workers 10 > runs/logs/expert1_cmp_ABF.log 2>&1
echo "AB+F: $(grep 'candidate share' runs/logs/expert1_cmp_ABF.log) $(date +%H:%M)"
uv run python -m catan_ai.cli.compare --candidate model:$E --baseline model:$M \
  --label expert-1-vs-robber --workers 10 > runs/logs/expert1_cmp.log 2>&1
echo "W/VP: $(grep 'candidate share' runs/logs/expert1_cmp.log) $(date +%H:%M)"
