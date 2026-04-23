#!/usr/bin/env bash

CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025"
MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/3_wacv_rebuttal_15_sep_2025"


# mtr
# CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_april_2025"
# MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/apr_2025/mtr/imotion_mtr"


# CHECKPOINT="checkpoint-6726"
# CHECKPOINT="checkpoint-13451"
CHECKPOINT="checkpoint-11963"
NUM_GPUS=1
BATCH_SIZE=32

CONFIGS=(
    # "imotion_mtr"
    # "imotion_mtr_dec"
    # "imotion_mtr_q"
    # "imotion_mtr_q_dec"
    # "imotion_mtr_1epoch"
    
    # imotion_mtr_alignment_2tokens
    # imotion_mtr_alignment_1token_lr5r32a16
    01_llama_7b_2a_i2_i1of2_iNone_l1
    01_llama_7b_2a_i2_i1of2_iNone
    01_llama_7b_2a_i2_i1of2_l1
    01_llama_7b_2a_i2_i1of2
    01_llama_7b_2a_i2_l1
    01_llama_7b_2a_i2
    # imotion_mtr_alignment_1token
    # imotion_mtr_alignment
)

/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/01_llama_7b_2a_i2_i1of2_iNone_l1.yaml
/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/01_llama_7b_2a_i2_i1of2_iNone.yaml
/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/01_llama_7b_2a_i2_i1of2_l1.yaml
/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/01_llama_7b_2a_i2_i1of2.yaml
/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/01_llama_7b_2a_i2_l1.yaml
/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/01_llama_7b_2a_i2.yaml
# MODES=("gt1" "neg1" "pos1")
MODES=("gt1")

for CONFIG in "${CONFIGS[@]}"; do
    for MODE in "${MODES[@]}"; do
        MODEL_DIR="${MODELS_DIR}/${CONFIG}"
        CONFIG_PATH="${CONFIG_DIR}/${CONFIG}.yaml"
        EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"

        JOB_SCRIPT="run_eval_${CONFIG}_${MODE}.sh"

        cat > ${JOB_SCRIPT} << EOF
#!/usr/bin/env bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=${NUM_GPUS}
#SBATCH --constraint=a100
#SBATCH --mem=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --job-name=eval_${CONFIG}_${MODE}
#SBATCH --output=/home/felembaa/logs/%j-%x.out
#SBATCH --error=/home/felembaa/errs/%j-%x.err
##SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
##SBATCH --mail-type=ALL
##SBATCH --account conf-neurips-2025.05.22-elhosemh

# source ~/miniconda3/bin/activate /ibex/project/c2278/felembaa/envs/imotion
source ~/miniconda3/bin/activate /ibex/project/c2253/felembaa/envs/imotion_mtr03

if [ ${NUM_GPUS} -gt 1 ]; then
    read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
    while :
    do
        PORT="\$(shuf -i \$LOWERPORT-\$UPPERPORT -n 1)"
        ss -lpn | grep -q ":\$PORT " || break
    done
    PORT=\$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
fi

echo "Running Evaluation Mode: ${MODE}"

if [ ${NUM_GPUS} -eq 1 ]; then
    python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \\
        --cfg-path ${CONFIG_PATH} \\
        --options run.eval_dir=${EVAL_DIR} run.distributed=False run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}
else
    CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --master_port=\${PORT} \\
        /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \\
        --cfg-path ${CONFIG_PATH} \\
        --options run.eval_dir=${EVAL_DIR} run.distributed=True run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=${MODE}
fi

echo "Completed Evaluation for Mode: ${MODE} on ${CONFIG}"
EOF

        chmod +x ${JOB_SCRIPT}
        echo "Submitting evaluation job for ${CONFIG} with mode ${MODE}..."
        sbatch ${JOB_SCRIPT}
    done
done

echo "All evaluation jobs submitted!"
