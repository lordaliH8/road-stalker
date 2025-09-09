#!/bin/bash
# Run evaluation on val split

MODEL_DIR="outputs/checkpoints/latest"
VAL_FILE="data/processed/val.jsonl"

echo ">>> Evaluating model from $MODEL_DIR"
python eval/vqa_eval.py --model $MODEL_DIR --data $VAL_FILE
python eval/grounding_eval.py --model $MODEL_DIR --data $VAL_FILE
