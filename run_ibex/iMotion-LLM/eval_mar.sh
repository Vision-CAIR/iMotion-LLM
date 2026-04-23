I see the issue. Your working example uses a different environment path and the port selection method is slightly different. Let me fix the script:
bashCopy#!/usr/bin/env bash

# Base directories
CONFIG_DIR="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_mar_2025"
MODELS_DIR="/ibex/project/c2278/felembaa/models/imotion/mar_2025"
CHECKPOINT="checkpoint-6726"
NUM_GPUS=2
BATCH_SIZE=50

# List of model configurations
CONFIGS=(
    "imotion_e2e"
    "imotion_e2e_simpleTxt"
    "imotion_e2e_lr02"
    "imotion_e2e_lr01"
    "imotion_e2e_lora02"
    "imotion_e2e_lora01"
    "imotion_e2e_llmOnly"
    "imotion_e2e_decOnly"
    "imotion_e2e_cb"
)

# Evaluation modes
MODES=("gt1" "neg1" "pos1")

# Create and submit a job for each configuration
for CONFIG in "${CONFIGS[@]}"; do
    MODEL_DIR="${MODELS_DIR}/${CONFIG}"
    CONFIG_PATH="${CONFIG_DIR}/train_${CONFIG}.yaml"
    EVAL_DIR="${MODEL_DIR}/${CHECKPOINT}"
    
    # Create a job script for this configuration
    JOB_SCRIPT="run_eval_${CONFIG}.sh"
    
    cat > ${JOB_SCRIPT} << EOF
#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=${NUM_GPUS}
#SBATCH --constraint=a100
#SBATCH --mem=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --job-name=eval_${CONFIG}
#SBATCH --output=/home/felembaa/logs/%j-%x.out
#SBATCH --error=/home/felembaa/errs/%j-%x.err
#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL

source ~/miniconda3/bin/activate /ibex/project/c2278/felembaa/envs/imotion

# Set up port selection
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
    PORT="\$(shuf -i \$LOWERPORT-\$UPPERPORT -n 1)"
    ss -lpn | grep -q ":\$PORT " || break
done
PORT=\$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Run the evaluation for each mode
for MODE in "${MODES[@]}"; do
    echo "Running Evaluation Mode: \${MODE}"
    
    CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --master_port=\${PORT} \\
        /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/eval.py \\
        --cfg-path ${CONFIG_PATH} \\
        --options run.eval_dir=${EVAL_DIR} run.distributed=True run.batch_size_eval=${BATCH_SIZE} datasets.traj_align_valid.processor.valid.new_eval_mode=\${MODE}

    echo "Completed Evaluation for Mode: \${MODE}"
    echo "----------------------------------------"
done

echo "All evaluations completed for ${CONFIG}."
EOF

    # Make the job script executable
    chmod +x ${JOB_SCRIPT}
    
    # Submit the job
    echo "Submitting evaluation job for ${CONFIG}..."
    sbatch ${JOB_SCRIPT}
done

echo "All evaluation jobs submitted!"