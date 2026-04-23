#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err

##SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
##SBATCH --mail-type=ALL
#SBATCH --job-name=validset

# Get the run_num argument passed to the script
run_num=$1

source ~/miniconda3/bin/activate gameformer

export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

python ~/projects/iMotion-LLM-ICLR/gameformer/data_preprocess_v06.py \
  --load_path /ibex/project/c2278/felembaa/datasets/waymo/validation_interactive \
  --save_path /ibex/project/c2278/felembaa/datasets/waymo/gameformer/validation_23aug \
  --run_num $run_num \
  --not_debug --use_multiprocessing --processes 10 --small_data