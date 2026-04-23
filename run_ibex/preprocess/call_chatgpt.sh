#!/usr/bin/env bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err

#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --job-name=openai  # Set an initial job name here

# run_num=$1

# Dynamically rename the job using scontrol
echo "Renaming job to: openai_${run_num}"
scontrol update JobID=$SLURM_JOB_ID JobName=openai_${run_num}

source ~/miniconda3/bin/activate gameformer

export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/call_chatgpt.py --type_select $run_num --processes 20
python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/call_chatgpt_test.py --processes 20
