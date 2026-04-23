#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --constraint=v100
#SBATCH --mem=150GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err
##SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --exclude=gpu101-02-l
##SBATCH  --account conf-neurips-2024.05.22-elhosemh
#SBATCH --job-name=noq

source ~/miniconda3/bin/activate gameformer

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done

# Get the first node in the node list
MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

torchrun --nnodes 2 --nproc-per-node 2 --master_addr $MASTER_NODE --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --full_map --two_agents --distributed --workers 4 --batch_size 128 --level 6 --modalities 6 --future_len 80 --learning_rate 5e-5 --lr_steps '15,18,21,24,27,30' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_small_1jul --save_model /ibex/project/c2278/felembaa/models/gameformer/temp/ --name temp --load_dir '' --wandb --act_dec