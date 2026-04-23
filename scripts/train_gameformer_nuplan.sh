#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SET="${IMOTION_GF_NUPLAN_TRAIN_SET:-data/processed/nuplan/gpt_prompt_14types}"
VALID_SET="${IMOTION_GF_NUPLAN_VALID_SET:-data/processed/nuplan/test_gpt_prompt_14types}"
SAVE_PATH="${IMOTION_GF_NUPLAN_SAVE_PATH:-outputs/gameformer/nuplan}"
RUN_NAME="${IMOTION_GF_NUPLAN_RUN_NAME:-gameformer_nuplan_release}"
LOAD_DIR="${IMOTION_GF_NUPLAN_LOAD_DIR:-}"

cd "$ROOT_DIR"
python gameformer/interaction_prediction/train_nuplan.py \
  --train_set "$TRAIN_SET" \
  --valid_set "$VALID_SET" \
  --save_path "$SAVE_PATH" \
  --name "$RUN_NAME" \
  --load_dir "$LOAD_DIR" \
  "$@"
