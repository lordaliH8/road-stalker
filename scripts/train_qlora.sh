#!/bin/bash
# Launch training with QLoRA

CONFIG="configs/train_small.yaml"

echo ">>> Starting training with config: $CONFIG"
python train/trainer.py --config $CONFIG
