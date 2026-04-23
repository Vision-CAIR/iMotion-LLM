#!/usr/bin/env bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err
#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --job-name=mtr_data
#SBATCH --mem=256G

source ~/miniconda3/bin/activate gameformer

export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

python /home/felembaa/projects/iMotion-LLM-ICLR/mtr/mtr/datasets/waymo/data_preprocess.py /ibex/project/c2278/felembaa/datasets/waymo /ibex/project/c2253/felembaa/waymo_dataset/mtr2 64