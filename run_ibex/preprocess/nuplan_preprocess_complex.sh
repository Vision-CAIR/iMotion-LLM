#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err

##SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
##SBATCH --mail-type=ALL
#SBATCH --job-name=nuplan_train  # Set an initial job name here

# Get the run_num argument passed to the script
run_num=$1

# Dynamically rename the job using scontrol
echo "Renaming job to: nuplan_train_${run_num}"
scontrol update JobID=$SLURM_JOB_ID JobName=nuplan_${run_num}

# Set the job name to include run_num
#SBATCH --job-name=${run_num}_nuplan_train

source ~/miniconda3/bin/activate gameformer

export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/data_preprocess.py \
#   --data_path /ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/cache/train_pittsburgh\
#   --save_path /ibex/project/c2278/felembaa/datasets/nuplan/train/train_pittsburgh_processed\
#   --processes 10 --run_num $run_num

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/data_preprocess.py \
#   --data_path /ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/cache/train_boston\
#   --save_path /ibex/project/c2278/felembaa/datasets/nuplan/train/train_boston_processed\
#   --processes 10 --run_num $run_num

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/data_preprocess.py \
#   --data_path /ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/cache/train_singapore\
#   --save_path /ibex/project/c2278/felembaa/datasets/nuplan/train/train_singapore_processed\
#   --processes 10 --run_num $run_num

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/data_preprocess.py \
#   --data_path /ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/cache/train_boston\
#   --save_path /ibex/project/c2278/felembaa/datasets/nuplan/train/train_boston_processed\
#   --processes 10 --run_num $run_num



# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/data_preprocess.py \
#   --data_path /ibex/project/c2278/felembaa/datasets/nuplan/test/data/cache/test\
#   --save_path /ibex/project/c2278/felembaa/datasets/nuplan/test/test_pittsburgh_processed\
#   --processes 10 --run_num $run_num --map_names us-pa-pittsburgh-hazelwood

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/data_preprocess.py \
#   --data_path /ibex/project/c2278/felembaa/datasets/nuplan/test/data/cache/test\
#   --save_path /ibex/project/c2278/felembaa/datasets/nuplan/test/test_boston_processed\
#   --processes 10 --run_num $run_num --map_names us-ma-boston

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/data_preprocess_complex.py \
#   --processes 10 --run_num $run_num

python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/data_preprocess_complex.py \
  --processes 10 --run_num $run_num



