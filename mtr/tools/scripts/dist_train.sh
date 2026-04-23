#!/usr/bin/env bash

# set -x
# NGPUS=$1
# PY_ARGS=${@:2}

source ~/miniconda3/bin/activate /ibex/project/c2278/felembaa/envs/mtr
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py
# python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} /home/felembaa/projects/mtr_p/tools/train.py --launcher pytorch ${PY_ARGS}

# torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

# python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/waymo/mtr+20_percent_data.yaml --batch_size 20 --epochs 30 --extra_tag temp

# bash /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/scripts/dist_train.sh 4 --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data_act.yaml --batch_size 80 --epochs 30 --extra_tag act_mtr
# bash scripts/dist_train.sh 4 --cfg_file cfgs/waymo/mtr+20_percent_data_act.yaml --batch_size 80 --epochs 30 --extra_tag act_mtr

CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=3 --nnodes=1 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py