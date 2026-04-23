#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
python gameformer/data_preprocess_v09.py \
  --load_path "${IMOTION_WAYMO_RAW_DIR:-data/raw/waymo/validation_interactive}" \
  --save_path "${IMOTION_WAYMO_PROCESSED_DIR:-data/processed/waymo/gameformer/val}" \
  "$@"
