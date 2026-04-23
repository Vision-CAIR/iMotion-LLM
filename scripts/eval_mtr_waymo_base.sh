#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_PATH="${IMOTION_MTR_CONFIG:-$ROOT_DIR/configs/release/mtr_waymo.yaml}"
EVAL_MODE="${IMOTION_MTR_EVAL_MODE:-gt1}"

cd "$ROOT_DIR"
python mtr/tools/test.py --cfg_file "$CFG_PATH" --eval_mode "$EVAL_MODE" "$@"
