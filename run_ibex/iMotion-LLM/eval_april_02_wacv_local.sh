#!/usr/bin/env bash

# CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025"
# MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/3_wacv_rebuttal_15_sep_2025"

# CHECKPOINT="checkpoint-11963"
# # CHECKPOINT="checkpoint-7194" # complex
# BATCH_SIZE=32

# CONFIGS=(
#     01_llama_7b_2a_i2_i1of2_iNone_l1
#     01_llama_7b_2a_i2_i1of2_iNone
#     01_llama_7b_2a_i2_i1of2_l1
#     01_llama_7b_2a_i2_i1of2
#     01_llama_7b_2a_i2_l1
#     01_llama_7b_2a_i2
# )

# MODES=("pos1" "neg1")
# # new_eval_mode: safe_with_context # 'safe_with_context', 'safe_no_context', 'unsafe_with_context', 'unsafe_no_context'

# # Activate environment once
# source ~/miniconda3/bin/activate /ibex/project/c2253/felembaa/envs/imotion_mtr03

# for CONFIG in "${CONFIGS[@]}"; do
#     for MODE in "${MODES[@]}"; do
#         MODEL_DIR="${MODELS_DIR}/${CONFIG}"
#         CONFIG_PATH="${CONFIG_DIR}/${CONFIG}.yaml"
#         EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"

#         echo "========================================="
#         echo " Running Evaluation"
#         echo " Config: ${CONFIG}"
#         echo " Mode:   ${MODE}"
#         echo "========================================="

#         python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
#             --cfg-path ${CONFIG_PATH} \
#             --options run.eval_dir=${EVAL_DIR} run.distributed=False run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}

#         echo "✅ Completed evaluation for ${CONFIG} (${MODE})"
#     done
# done

# echo "🎉 All evaluations finished!"



#!/usr/bin/env bash

BATCH_SIZE=32
# MODES=('safe_with_context', 'safe_no_context', 'unsafe_with_context', 'unsafe_no_context')
MODES=('safe_with_context', 'safe_no_context')
# new_eval_mode: safe_with_context # 
# Activate environment once
source ~/miniconda3/bin/activate /ibex/project/c2253/felembaa/envs/imotion_mtr03

# CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025"
# MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/3_wacv_rebuttal_15_sep_2025"

# # CHECKPOINT="checkpoint-11963"
# CONFIGS=(
#     03_nuplan_c_llama_7b_NuplanCGF_l1
#     03_nuplan_c_llama_7b_NuplanCGF_l1_b64
# )
# CHECKPOINT="checkpoint-7194" # complex

# for CONFIG in "${CONFIGS[@]}"; do
#     for MODE in "${MODES[@]}"; do
#         MODEL_DIR="${MODELS_DIR}/${CONFIG}"
#         CONFIG_PATH="${CONFIG_DIR}/${CONFIG}.yaml"
#         EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"

#         echo "========================================="
#         echo " Running Evaluation"
#         echo " Config: ${CONFIG}"
#         echo " Mode:   ${MODE}"
#         echo "========================================="

#         python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
#             --cfg-path ${CONFIG_PATH} \
#             --options run.eval_dir=${EVAL_DIR} run.distributed=False run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}

#         echo "✅ Completed evaluation for ${CONFIG} (${MODE})"
#     done
# done

# ###################
# CONFIGS=(
# 02_nuplan_d_llama_7b_NuplanCGF_l1_b64_finetuneC
# 02_nuplan_d_llama_7b_NuplanCGF_l1_finetuneC
# 02_nuplan_d_llama_7b_womdPretrained_finetuneC
# 02_nuplan_d_llama_7b_finetuneC
# )
# CHECKPOINT="checkpoint-2398" 

# for CONFIG in "${CONFIGS[@]}"; do
#     for MODE in "${MODES[@]}"; do
#         MODEL_DIR="${MODELS_DIR}/${CONFIG}"
#         CONFIG_PATH="${CONFIG_DIR}/${CONFIG}.yaml"
#         EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"

#         echo "========================================="
#         echo " Running Evaluation"
#         echo " Config: ${CONFIG}"
#         echo " Mode:   ${MODE}"
#         echo "========================================="

#         python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
#             --cfg-path ${CONFIG_PATH} \
#             --options run.eval_dir=${EVAL_DIR} run.distributed=False run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}

#         echo "✅ Completed evaluation for ${CONFIG} (${MODE})"
#     done
# done

# ################

# CONFIGS=(
#     02_nuplan_d_llama_7b
#     02_nuplan_d_llama_7b_NuplanCGF_l1
#     02_nuplan_d_llama_7b_NuplanCGF_l1_b64
#     02_nuplan_d_llama_7b_womdPretrained
# )
# CHECKPOINT="checkpoint-1780"

# for CONFIG in "${CONFIGS[@]}"; do
#     for MODE in "${MODES[@]}"; do
#         MODEL_DIR="${MODELS_DIR}/${CONFIG}"
#         CONFIG_PATH="${CONFIG_DIR}/${CONFIG}.yaml"
#         EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"

#         echo "========================================="
#         echo " Running Evaluation"
#         echo " Config: ${CONFIG}"
#         echo " Mode:   ${MODE}"
#         echo "========================================="

#         python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
#             --cfg-path ${CONFIG_PATH} \
#             --options run.eval_dir=${EVAL_DIR} run.distributed=False run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}

#         echo "✅ Completed evaluation for ${CONFIG} (${MODE})"
#     done
# done

# echo "🎉 All evaluations finished!"

CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/2_iccv_rebuttal_may_2025"
MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/2_iccv_rebuttal_may_2025"


CONFIGS=(
    01_llama_7b
)
CHECKPOINT="checkpoint-6726"

for CONFIG in "${CONFIGS[@]}"; do
    for MODE in "${MODES[@]}"; do
        MODEL_DIR="${MODELS_DIR}/${CONFIG}"
        CONFIG_PATH="${CONFIG_DIR}/${CONFIG}.yaml"
        EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"

        echo "========================================="
        echo " Running Evaluation"
        echo " Config: ${CONFIG}"
        echo " Mode:   ${MODE}"
        echo "========================================="

        python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
            --cfg-path ${CONFIG_PATH} \
            --options run.eval_dir=${EVAL_DIR} run.distributed=False run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}

        echo "✅ Completed evaluation for ${CONFIG} (${MODE})"
    done
done

echo "🎉 All evaluations finished!"

# /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/2_iccv_rebuttal_may_2025/01_llama_7b.yaml

# /ibex/project/c2278/felembaa/models/imotion/2_iccv_rebuttal_may_2025/01_llama_7b/checkpoint-6726