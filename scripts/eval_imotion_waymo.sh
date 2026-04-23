#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${IMOTION_LLM_CONFIG:-$ROOT_DIR/configs/release/imotion_waymo_eval.yaml}"

cd "$ROOT_DIR"
python trajgpt/eval.py --cfg-path "$CONFIG_PATH" "$@"
