#!/usr/bin/env bash
#SBATCH --time=44:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100
##SBATCH --constraint=v100
##SBATCH --mem=48GB
#SBATCH --mem=96GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err
##SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
##SBATCH --exclude=gpu101-02-l
#SBATCH --job-name=mtr
#SBATCH --account=conf-neurips-2025.05.22-elhosemh

# module load cuda/12.1
cd /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools
source ~/miniconda3/bin/activate /ibex/project/c2253/felembaa/envs/imotion_mtr03
# python setup.py develop
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# torchrun --nproc_per_node=3 --nnodes=1 /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data_act.yaml --act --batch_size 120 --launcher pytorch
# torchrun --nproc_per_node=3 --nnodes=1 /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data.yaml --batch_size 120 --launcher pytorch
torchrun --nproc_per_node=4 --nnodes=1 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+100_percent_data.yaml --batch_size 160 --launcher pytorch
# torchrun --nproc_per_node=4 --nnodes=1 /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+100_percent_data_act.yaml --batch_size 160 --launcher pytorch --act


# NUM_GPUS=4
# nohup bash -c "bash /home/felembaa/projects/iMotion-LLM-ICLR/zzz_launch.sh" > nohup/train_imotion06_dec_lowlr_highnorm.log 2>&1 &
# torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data_act.yaml --act --batch_size 80 --launcher pytorch

# torchrun --nproc_per_node=4 --nnodes=1 /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data_act.yaml --act --batch_size 80 --launcher pytorch

# torchrun --nproc_per_node=4 train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data.yaml --batch_size 80 --launcher pytorch
# torchrun --nproc_per_node=4 train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data_act.yaml --act --batch_size 80 --launcher pytorch

# torchrun --nproc_per_node=4 /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data.yaml--batch_size 80


# nohup bash -c "bash /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/launch.sh" > /home/felembaa/projects/iMotion-LLM-ICLR/nohup/mtr.log 2>&1 &