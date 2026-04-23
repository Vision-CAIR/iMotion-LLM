#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_PATH="${IMOTION_GF_NUPLAN_CKPT:-checkpoints/gameformer/nuplan/cgf_l1/epochs_29.pth}"
TRAIN_SET="${IMOTION_GF_NUPLAN_TRAIN_SET:-data/processed/nuplan/gpt_prompt_14types}"
VALID_SET="${IMOTION_GF_NUPLAN_VALID_SET:-data/processed/nuplan/test_gpt_prompt_14types}"
SAVE_PATH="${IMOTION_GF_NUPLAN_SAVE_PATH:-outputs/gameformer/nuplan/eval}"
RUN_NAME="${IMOTION_GF_NUPLAN_RUN_NAME:-gameformer_nuplan_eval}"
EVAL_MODE="${IMOTION_GF_NUPLAN_EVAL_MODE:-safe_with_context}"

cd "$ROOT_DIR"
python gameformer/interaction_prediction/train_nuplan.py \
  --eval_only \
  --load_dir "$CKPT_PATH" \
  --train_set "$TRAIN_SET" \
  --valid_set "$VALID_SET" \
  --save_path "$SAVE_PATH" \
  --name "$RUN_NAME" \
  --new_eval \
  --new_eval_mode "$EVAL_MODE" \
  "$@"
