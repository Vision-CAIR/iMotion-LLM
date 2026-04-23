#!/usr/bin/env bash



# CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_mar_2025/ablation_modeling"
# CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_april_2025"
# MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/mar_2025/ablation_modeling"cd /
# MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/mar_2025/ablation_modeling"
# MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/apr_2025/mtr"


# Felemba
# CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_mar_2025/ablation_llms"
# MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/mar_2025/ablation_llms"


# non complex
CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/2_iccv_rebuttal_may_2025"
MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/2_iccv_rebuttal_may_2025"

# complex
# CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/may_experiments"
# MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/apr_2025/ablation_llms"
# mtr
# CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_april_2025"
# MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/apr_2025/mtr/imotion_mtr"


CHECKPOINT="checkpoint-6726"
# CHECKPOINT="checkpoint-13452"
# CHECKPOINT="checkpoint-20178"
# CHECKPOINT="checkpoint-26904" # 2 or 4?
# CHECKPOINT="checkpoint-33630"
# CHECKPOINT="checkpoint-40356"
# CHECKPOINT="checkpoint-47082"
# CHECKPOINT="checkpoint-53808" # 4 or 8?

# CHECKPOINT="checkpoint-67260"
# CHECKPOINT="checkpoint-80712"


# CHECKPOINT="checkpoint-7194"
NUM_GPUS=1
BATCH_SIZE=1

CONFIGS=(

 "01_llama_7b"
# "12_Llama_3.2_1B_1e4_1e3_1e4_both_r32_a16_2r_25"
#    "02_llama_7b_synth"
#    "03_llama_7b_linearIn_linearOut"
#    "04_llama_7b_2mlpIn_2mlpOut"
#    "05_llama_7b_4mlpIn_4mlpOut"
#    "06_llama_7b_2mlpIn_2mlpOut_inShared"
#    "07_llama_7b_freezeGf"
#    "08_llama_7b_freezeDec"
#    "09_llama_7b_noCB"
# "01_c_llama_7b"
# "02_c_mistral_7b"
# "07_c_llama_7b_e2e"
# "08_c_mistral_7b_e2eyaml"
# "12_c_llama_7b_womd_1e4"
# "13_c_llama_7b_womd_1e5"

)

# Nussair 
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
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=${NUM_GPUS}
#SBATCH --constraint=a100
#SBATCH --mem=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --job-name=a100_eval_${CONFIG}_${MODE}
#SBATCH --output=/home/felembaa/logs/%j-%x.out
#SBATCH --error=/home/felembaa/errs/%j-%x.err
##SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
##SBATCH --mail-type=ALL

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
