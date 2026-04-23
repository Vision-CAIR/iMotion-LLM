#!/usr/bin/env bash
#SBATCH --time=03:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --constraint=a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --job-name=no_caption
#SBATCH --output=/home/felembaa/logs/%j-%x.out
#SBATCH --error=/home/felembaa/errs/%j-%x.err
#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL

#module load cuda/12.2

source ~/miniconda3/bin/activate /ibex/project/c2278/felembaa/envs/imotion

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')


## 26 dec

timeout 10800s torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
    --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv_no_caption_27dec.yaml

# sbatch $0