#!/usr/bin/env bash
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=64
##SBATCH --ntasks=64
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err

#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --job-name=03trainset

# Get the run_num argument passed to the script
run_num=$1

source ~/miniconda3/bin/activate gameformer

export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

python ~/projects/iMotion-LLM-ICLR/gameformer/data_preprocess_v06.py \
    --load_path /ibex/project/c2278/felembaa/datasets/waymo/training/ \
    --run_num $run_num \
    --save_path /ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_11may_fullmap_split2 \
    --not_debug --use_multiprocessing --processes 32 --small_data