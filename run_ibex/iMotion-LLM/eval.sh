#!/usr/bin/env bash
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
##SBATCH --constraint=v100
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err
#SBATCH --job-name=eval
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

# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval07/c.yaml

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e03gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e03pos1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e03neg1.yaml

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e02gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e02pos1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e02neg1.yaml

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e04gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e04pos1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e04neg1.yaml

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e01gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e01gt2.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e01pos1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e01pos2.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e01neg1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e01neg2.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e01pos12.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e01neg12.yaml

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e05/e05_5gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e05/e05_30gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e05/e05_10gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e05/e05_20gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e05/e05_15gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e05/e05_25gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e05/e05_0gt1.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/eval_jul/e05/e05_25gt1yaml

# rebuttal, 4 aug, randomdrop inference only, not trained with random drop
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/rebuttal/eval_m07_gt_fulldata_noInstruct.yaml

# random drop trained model
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/rebuttal/eval_may11_07_act_contrastive_short_randomDrop_noIstruct.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/rebuttal/eval_may11_07_act_contrastive_short_randomDrop_gt.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/rebuttal/eval_may11_07_act_contrastive_short_randomDrop_neg.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/rebuttal/eval_may11_07_act_contrastive_short_randomDrop_pos.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/rebuttal/eval_m07_noInstruct.yaml

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/rebuttal/eval_m07_nuplan.yaml

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/rebuttal/eval_finetune_m07_nuplan_e2e_gt.yaml
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_new/rebuttal/eval_finetune_m07_waymo_e2e_p.yaml

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_sep/eval_22sep/imotion_2tokens/imotion_2tokens_gt.yaml
python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
    --options model.lisa_like=True \
    model.img_token_num=1000 \
    model.freeze_gf=True \
    model.freeze_gf_env=True \
    model.freeze_gf_dec=True \
    model.lora_dropout=0 \
    run.resume_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_6qtokens_gf_21nov_01/checkpoint_last.pth" \
    run.gf_encoder_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_6qtokens_gf_21nov_01/checkpoint_last.pth" \
    run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_6qtokens_gf_21nov_01 \
    run.distributed=False