#!/usr/bin/env bash
# nohup bash /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/2_iccv_rebuttal_may_2025/local_eval.sh > /home/felembaa/projects/iMotion-LLM-ICLR/nohup/eval2.log 2>&1 &


CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/2_iccv_rebuttal_may_2025/complex"
MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/2_iccv_rebuttal_may_2025/complex"

CHECKPOINT="checkpoint-7194"
NUM_GPUS=1
BATCH_SIZE=50

CONFIGS=(
    # Done
    # 01_c_llama_7b
    # 02_c_mistral_7b
    # 03_c_llama_1b
    # 07_c_llama_7b_e2e
    # 08_c_mistral_7b_e2e
    # 09_c_llama_1b_e2e
    # 10_c_mistral_7b_e2e_1e4_1e3_1e5
    # 11_c_llama_7b_e2e_1e4_1e3_1e5
    # 12_c_llama_7b_womd
    # 13_c_llama_7b_womd_1e5
    # 14_c_llama_7b_womd_1e5_e2e

    #Done
    # 11_c_llama_7b_e2e_1e4_1e3_1e5
    # 04_c_llama_8b
    # 12_c_llama_7b_womd_1e4
    # 15_c_llama_7b_womd_1e5_e2e_1e4_1e3_1e5

    # Running
    # 05_c_qwen2_7b
    # 06_c_vicuna_7b
    # 16_c_llama_7b_womd_1e5_none
)

# MODES=("gt1" "pos1" "neg1")
MODES=("safe_with_context" "safe_no_context" "unsafe_with_context" "unsafe_no_context")

# Activate conda environment
source ~/miniconda3/bin/activate /ibex/project/c2253/felembaa/envs/imotion_mtr03

for CONFIG in "${CONFIGS[@]}"; do
    for MODE in "${MODES[@]}"; do
        MODEL_DIR="${MODELS_DIR}/${CONFIG}"
        CONFIG_PATH="${CONFIG_DIR}/${CONFIG}.yaml"
        EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"

        echo "Running Evaluation Mode: ${MODE} for ${CONFIG}"

        if [ ${NUM_GPUS} -eq 1 ]; then
            python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
                --cfg-path ${CONFIG_PATH} \
                --options run.eval_dir=${EVAL_DIR} run.distributed=False run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}
        else
            # Find an available port for distributed training
            read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
            while :
            do
                PORT="$(shuf -i $LOWERPORT-$UPPERPORT -n 1)"
                ss -lpn | grep -q ":$PORT " || break
            done
            
            # Alternative method for finding port
            # PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
            
            CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --master_port=${PORT} \
                /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
                --cfg-path ${CONFIG_PATH} \
                --options run.eval_dir=${EVAL_DIR} run.distributed=True run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}
        fi

        echo "Completed Evaluation for Mode: ${MODE} on ${CONFIG}"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"





# ########################################################################################################

CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/2_iccv_rebuttal_may_2025"
MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/2_iccv_rebuttal_may_2025"

CHECKPOINT="checkpoint-6726"
NUM_GPUS=1
BATCH_SIZE=50

CONFIGS=(
    # Done
    # 01_llama_7b
    # 02_llama_7b_synth
    # 03_llama_7b_linearIn_linearOut
    # 04_llama_7b_2mlpIn_2mlpOut
    # 05_llama_7b_4mlpIn_4mlpOut
    # 06_llama_7b_2mlpIn_2mlpOut_inShared
    # 07_llama_7b_freezeGf
    # 08_llama_7b_freezeDec
    # 09_llama_7b_noCB

    #Done
    # 12_llama_7b_e2e

    # Running
    # 10_llama_7b_largeGradNorm
    # 13_llama_7b_4mlpIn_4mlpOut_inShared
)

MODES=("gt1" "pos1" "neg1")
# MODES=("safe_with_context" "safe_no_context" "unsafe_with_context" "unsafe_no_context")

# Activate conda environment
source ~/miniconda3/bin/activate /ibex/project/c2253/felembaa/envs/imotion_mtr03

for CONFIG in "${CONFIGS[@]}"; do
    for MODE in "${MODES[@]}"; do
        MODEL_DIR="${MODELS_DIR}/${CONFIG}"
        CONFIG_PATH="${CONFIG_DIR}/${CONFIG}.yaml"
        EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"

        echo "Running Evaluation Mode: ${MODE} for ${CONFIG}"

        if [ ${NUM_GPUS} -eq 1 ]; then
            python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
                --cfg-path ${CONFIG_PATH} \
                --options run.eval_dir=${EVAL_DIR} run.distributed=False run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}
        else
            # Find an available port for distributed training
            read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
            while :
            do
                PORT="$(shuf -i $LOWERPORT-$UPPERPORT -n 1)"
                ss -lpn | grep -q ":$PORT " || break
            done
            
            # Alternative method for finding port
            # PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
            
            CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --master_port=${PORT} \
                /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
                --cfg-path ${CONFIG_PATH} \
                --options run.eval_dir=${EVAL_DIR} run.distributed=True run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}
        fi

        echo "Completed Evaluation for Mode: ${MODE} on ${CONFIG}"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"





# ########################################################################################################

CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/2_iccv_rebuttal_may_2025/llms"
MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/2_iccv_rebuttal_may_2025/llms"

CHECKPOINT="checkpoint-6726"
NUM_GPUS=1
BATCH_SIZE=50

CONFIGS=(
    # Running
    # 09_llama_8b
    # 10_llama_8b_both
    
    # 01_mistral_7b
    # 02_vicuna_7b
    # 03_llama_1b
    # 04_qwen_7b
    # 05_mistral_7b_both
    06_vicuna_7b_both
    07_llama_1b_both
    08_qwen_7b_both
    09_llama_8b
    10_llama_8b_both
)

MODES=("gt1" "pos1" "neg1")
# MODES=("safe_with_context" "safe_no_context" "unsafe_with_context" "unsafe_no_context")

# Activate conda environment
source ~/miniconda3/bin/activate /ibex/project/c2253/felembaa/envs/imotion_mtr03

for CONFIG in "${CONFIGS[@]}"; do
    for MODE in "${MODES[@]}"; do
        MODEL_DIR="${MODELS_DIR}/${CONFIG}"
        CONFIG_PATH="${CONFIG_DIR}/${CONFIG}.yaml"
        EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"

        echo "Running Evaluation Mode: ${MODE} for ${CONFIG}"

        if [ ${NUM_GPUS} -eq 1 ]; then
            python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
                --cfg-path ${CONFIG_PATH} \
                --options run.eval_dir=${EVAL_DIR} run.distributed=False run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}
        else
            # Find an available port for distributed training
            read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
            while :
            do
                PORT="$(shuf -i $LOWERPORT-$UPPERPORT -n 1)"
                ss -lpn | grep -q ":$PORT " || break
            done
            
            # Alternative method for finding port
            # PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
            
            CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --master_port=${PORT} \
                /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \
                --cfg-path ${CONFIG_PATH} \
                --options run.eval_dir=${EVAL_DIR} run.distributed=True run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}
        fi

        echo "Completed Evaluation for Mode: ${MODE} on ${CONFIG}"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"