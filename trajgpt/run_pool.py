#!/usr/bin/env bash
##!/bin/bash --login
#SBATCH --time=09:00:00

#SBATCH --partition=batch
#SBATCH --nodes=1
##SBATCH --constraint=cascadelake|skylake
#SBATCH --constraint=rome|amd

#SBATCH --mem=200G
##SBATCH --mem-per-cpu=8G
#SBATCH --job-name=tpool
#SBATCH --output=./logs/%j.out
#SBATCH --error=./errs/%j.err

source ~/miniconda3/bin/activate minigpt4

python /home/felembaa/projects/trajgpt/extract_incstuct_future.py --mp True --input_folder '/ibex/project/c2253/felembaa/waymo_dataset/training_interactive_original_20'
