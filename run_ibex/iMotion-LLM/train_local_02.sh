#!/usr/bin/env bash

# Pick a free port
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

CFG_PATHS=(
    "/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/02_nuplan_d_llama_7b_finetuneC.yaml"
    "/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/02_nuplan_d_llama_7b_NuplanCGF_l1_b64_finetuneC.yaml"
    "/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/02_nuplan_d_llama_7b_NuplanCGF_l1_finetuneC.yaml"
    "/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/02_nuplan_d_llama_7b_womdPretrained_finetuneC.yaml"
)

for CFG in "${CFG_PATHS[@]}"; do
    echo "========================================="
    echo " Starting training with config: $CFG"
    echo " Using PORT=$PORT"
    echo "========================================="

    torchrun --nproc-per-node=3 --master_port=$PORT \
        /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
        --cfg-path "$CFG"

    echo "✅ Finished config: $CFG"
    echo
done

echo "🎉 All trainings completed!"
