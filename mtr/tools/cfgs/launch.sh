# module load cuda/12.1
cd /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools
source ~/miniconda3/bin/activate /ibex/project/c2278/felembaa/envs/mtr
# python setup.py develop
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc_per_node=4 --nnodes=1 /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data_act.yaml --act --batch_size 80 --launcher pytorch


# NUM_GPUS=4
# nohup bash -c "bash /home/felembaa/projects/iMotion-LLM-ICLR/zzz_launch.sh" > nohup/train_imotion06_dec_lowlr_highnorm.log 2>&1 &
# torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data_act.yaml --act --batch_size 80 --launcher pytorch

# torchrun --nproc_per_node=4 --nnodes=1 /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data_act.yaml --act --batch_size 80 --launcher pytorch

# torchrun --nproc_per_node=4 train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data.yaml --batch_size 80 --launcher pytorch
# torchrun --nproc_per_node=4 train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data_act.yaml --act --batch_size 80 --launcher pytorch

# torchrun --nproc_per_node=4 /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/train.py --cfg_file /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/waymo/mtr+20_percent_data.yaml--batch_size 80


# nohup bash -c "bash /home/felembaa/projects/iMotion-LLM-ICLR/mtr/tools/cfgs/launch.sh" > /home/felembaa/projects/iMotion-LLM-ICLR/nohup/mtr.log 2>&1 &