#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root no matter where you run this from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY="${ROOT_DIR}/train/build_dataset.py"

# Adjust these if your paths differ
DATA_DIR="${ROOT_DIR}/data/raw/bdd100k"
PROC_DIR="${ROOT_DIR}/data/processed"

mkdir -p "${DATA_DIR}" "${PROC_DIR}"

echo ">>> Using ROOT_DIR=${ROOT_DIR}"
echo ">>> DATA_DIR=${DATA_DIR}"
echo ">>> PROC_DIR=${PROC_DIR}"

# If you already downloaded BDD100K, ensure it looks like:
# ${DATA_DIR}/images/100k/train/*.jpg
# ${DATA_DIR}/images/100k/val/*.jpg
# ${DATA_DIR}/labels/bdd100k_labels_images_train.json
# ${DATA_DIR}/labels/bdd100k_labels_images_val.json

echo ">>> Generating Q&A pairs..."
python "${PY}" \
  --images "${DATA_DIR}/images/100k/train" \
  --labels "${DATA_DIR}/labels/bdd100k_labels_images_train.json" \
  --output "${PROC_DIR}/train.jsonl" \
  --max_images 8000

python "${PY}" \
  --images "${DATA_DIR}/images/100k/val" \
  --labels "${DATA_DIR}/labels/bdd100k_labels_images_val.json" \
  --output "${PROC_DIR}/val.jsonl" \
  --max_images 1000

echo ">>> Done. Processed files are in ${PROC_DIR}"
