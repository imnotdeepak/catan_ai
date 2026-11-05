#!/bin/bash
# Release 0.6 value recalibration: data, value-only training, then search with the new tower.
uv run python -m catan_ai.cli.imitation_data --config configs/value_ppo3.json --workers 10 \
  --feature-version 3 > runs/logs/value_data.log 2>&1 || { echo "data failed"; exit 1; }
echo "data done $(date +%H:%M)"
uv run python -m catan_ai.cli.train_imitation --config configs/value_ppo3.json \
  --report reports/value_ppo3_training.md > runs/logs/value_train.log 2>&1 || { echo "train failed"; exit 1; }
echo "train done $(date +%H:%M)"; tail -3 runs/logs/value_train.log
