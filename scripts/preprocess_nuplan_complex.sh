#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
python gameformer/nuplan_preprocess/data_preprocess_complex.py \
  --data_path "${IMOTION_NUPLAN_CACHE_DIR:-data/raw/nuplan/cache/train_combined}" \
  --map_path "${IMOTION_NUPLAN_MAP_DIR:-data/raw/nuplan/maps}" \
  --save_path "${IMOTION_NUPLAN_OUTPUT_DIR:-data/processed/nuplan/gpt_prompt_14types}" \
  "$@"
