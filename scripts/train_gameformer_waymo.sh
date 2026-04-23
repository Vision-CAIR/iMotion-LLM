#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
python gameformer/interaction_prediction/train.py \
  --train_set "${IMOTION_GF_TRAIN_SET:-data/processed/waymo/gameformer/train}" \
  --valid_set "${IMOTION_GF_VALID_SET:-data/processed/waymo/gameformer/val}" \
  --save_path "${IMOTION_GF_SAVE_PATH:-outputs/gameformer/waymo}" \
  --name "${IMOTION_GF_RUN_NAME:-gameformer_waymo_release}" \
  "$@"
