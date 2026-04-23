#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100
##SBATCH --constraint=v100
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --job-name=sep22_imotion_3tokens_smallLLM
#SBATCH --output=/home/felembaa/logs/%j-%x.out
#SBATCH --error=/home/felembaa/errs/%j-%x.err
#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
##SBATCH --exclude=gpu101-02-l
#SBATCH --account conf-iclr-2025.10.01-elhosemh

module load cuda/12.2

source ~/miniconda3/bin/activate gameformer

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

### 25 aug
## 34946805 gf_4tokens_smallLLM
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_aug/train_configs_24aug/train_gf_predOnly_fewTokens4_smallLLM.yaml
## 34946807 gf_2tokens_smallLLM
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_aug/train_configs_24aug/train_gf_predOnly_fewTokens2_smallLLM.yaml
## 34947204 cgf_ci_5tokens_smallLLM
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_aug/train_configs_24aug/train_cgf_predOnly_fewTokens4_cimotion_smallLLM.yaml

# aug24_cgf_original_smallLLM
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_aug/train_configs_24aug/train_cgf_cimotion_smallLLM.yaml

# 34947428 aug24_gf_e2e_4tokens_smallLLM
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_aug/train_configs_24aug/train_gf_predOnly_e2e_fewTokens4_smallLLM.yaml
torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_sep/train_22sep/imotion_2tokens_smallLLM.yaml