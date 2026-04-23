#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --mem=30GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err
#SBATCH --job-name=eval
#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
##SBATCH --exclude=gpu101-02-l

module load cuda/12.2

source ~/miniconda3/bin/activate gameformer

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')


checkpoints=("checkpoint_4.pth" "checkpoint_3.pth" "checkpoint_2.pth" "checkpoint_1.pth" "checkpoint_0.pth")
# checkpoints=("checkpoint_2.pth")
for checkpoint in "${checkpoints[@]}"; do
    echo "*********************************************************"
    echo "*********************************************************"
    echo "*********************************************************"
    echo "*********************************************************"
    echo "*********************************************************"
    echo "*********************************************************"
    echo "*********************************************************"
    echo "*********************************************************"
    python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
            --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
            --options model.freeze_gf=True \
            model.freeze_gf_env=True \
            model.freeze_gf_dec=True \
            model.lora_dropout=0 \
            run.distributed=False \
            model.img_token_num=1 \
            model.gf_act_dec=True \
            model.neighbors_to_predict=0 \
            model.gf_token_num=114 \
            model.late_fusion=True \
            model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData/epochs_29.pth' \
            run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/${checkpoint}" \
            run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/eval
done