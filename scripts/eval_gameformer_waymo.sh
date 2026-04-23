#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_PATH="${IMOTION_GF_CKPT:-checkpoints/gameformer/waymo/cgf_l1/epochs_29.pth}"
SAVE_PATH="${IMOTION_GF_SAVE_PATH:-outputs/gameformer/waymo/eval}"
TRAIN_SET="${IMOTION_GF_TRAIN_SET:-data/processed/waymo/gameformer/train}"
VALID_SET="${IMOTION_GF_VALID_SET:-data/processed/waymo/gameformer/val}"
RUN_NAME="${IMOTION_GF_RUN_NAME:-gameformer_waymo_eval}"
EVAL_MODE="${IMOTION_GF_EVAL_MODE:-gt1}"

cd "$ROOT_DIR"
python gameformer/interaction_prediction/train.py \
  --eval_only \
  --load_dir "$CKPT_PATH" \
  --save_path "$SAVE_PATH" \
  --train_set "$TRAIN_SET" \
  --valid_set "$VALID_SET" \
  --name "$RUN_NAME" \
  --new_eval \
  --new_eval_mode "$EVAL_MODE" \
  "$@"
