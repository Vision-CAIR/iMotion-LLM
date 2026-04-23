#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100
##SBATCH --constraint=v100
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --output=./logs/%j.out
#SBATCH --error=./errs/%j.err
#SBATCH --job-name=evalonly
#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --exclude=gpu101-02-l
#SBATCH --account conf-eccv-2024.03.14-elhosemh

source ~/miniconda3/bin/activate minigpt4

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done

# PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

## python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/hyperparameter_search/e/e${yaml_number}.yaml

## PARAMETER SEARCH
# gameformer
# yaml_number=$1
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/train_subset_search/gf_train_subset${yaml_number}.yaml
# no gameformer
# yaml_number=$1
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/train_subset_search/train_subset${yaml_number}.yaml

# torchrun --nproc-per-node 4 /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/train_subset_search/train_subset${yaml_number}.yaml
# torchrun --nproc-per-node 4 /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/train_subset_search/train_subset11.yaml

# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/hyperparameter_search/l/l${yaml_number}.yaml

## RESUME
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/train_subset.yaml
## torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/resume1.yaml


## TRAIN
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/r128.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/l9_28nov_01_r1028.yaml

# Eval 
# yaml_number=$1
# python /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/gf_align_04feb_01/exp0${yaml_number}.yaml

# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/dec14/train2.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/dec14/train2_8bit.yaml

# yaml_number=$1
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/gf_align_04feb_01/exp0${yaml_number}.yaml
torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/instruct_gf_14feb/exp01_eval.yaml