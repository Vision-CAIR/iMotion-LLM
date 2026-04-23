#!/usr/bin/env bash
#SBATCH --time=03:40:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --constraint=a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --job-name=complex
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

# timeout 10800s torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv_caption_26dec.yaml
#     --options #run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs/checkpoint_last.pth"

timeout 12600s torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
    --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_nuplan_complex_26dec.yaml

sbatch $0
# Nuplan complex
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_nuplan_complex.yaml \
#     --options run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/nuplan_complex \
#     run.job_name=nuplan_complex


# timeout 13200s torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv_caption.yaml \
#     --options run.max_epoch=30 run.warmup_steps=500 \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs \
#     run.job_name=kvtoken_caption_30epochs run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs/checkpoint_last.pth"


# timeout 13200s 
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_nuplan_complex.yaml \
#     --options run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/nuplan_complex_pretrained \
#     run.job_name=nuplan_complex_pretrained \
#     model.new_optimizer=True run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs/checkpoint_last.pth"

# timeout 13200s 
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_nuplan_complex.yaml \
#     --options run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/nuplan_complex \
#     run.job_name=nuplan_complex

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_nuplan_complex.yaml \
#     --options run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/nuplan_complex \
#     run.job_name=nuplan_complex run.distributed=False

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_nuplan_complex.yaml \
#     --options run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/nuplan_complex_pretrained \
#     run.distributed=False run.job_name=nuplan_complex_pretrained \
#     run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs/checkpoint_last.pth" model.new_optimizer=True

# sbatch $0

### 25 aug

## 34946609 cgf_ci
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_aug/train_configs_24aug/train_cgf_predOnly_fewTokens4_cimotion.yaml
## 34947202 cgf_3tok
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_aug/train_configs_24aug/train_cgf_predOnly_fewTokens2_cimotion.yaml

## 34954545 aug24_cgf_original
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_aug/train_configs_24aug/train_cgf_cimotion.yaml


# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_sep/train/sep09_gf_clsHeadFinetune_4tokens_smallLLM.yaml

# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_sep/train/sep09_gf_encFinetune_4tokens_smallLLM.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_sep/train_22sep/imotion_2tokens.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_sep/train_22sep/imotion_2tokens.yaml
# 30 sep
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_sep/sep30/imotion_2tokens_waymo/train/imotion_2tokens.yaml
# torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_sep/sep30/imotion_2tokens_waymo/train/imotion_2tokens.yaml

# torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_sep/sep30/imotion_2tokens_waymo/eval_new_model/imotion_2tokens_neg.yaml

### Oct

# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_oct/oct12/train01_3t_waymo.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_oct/oct12/train01_3t_waymo_ablation.yaml
## nuplan
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_oct/oct12/train01_3t_nuplan_boston.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_oct/oct12/train01_3t_nuplan_pittsburgh.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_oct/oct12/train01_3t_nuplan_singapore.yaml



#### Nov

# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_3t_waymo.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_3t_waymo.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_lisa_like.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_lisa_like_1token.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_lisa_like_gf_1token.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_lisa_like_1token_e2e.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_lisa_like_1token_encOnly.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_lisa_like_1token_LoRAOnly.yaml

## in terminal launch:
# torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_3t_waymo.yaml
# torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_lisa_like.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_lisa_like_gf.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/train/train_3t_waymo_e2e.yaml

####### Dec
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_lisa_like_1token_LoRAOnly.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_lisa_like_1token.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv.yaml
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_lisa_like_1token_latefusion.yaml


####### Dec, 2 GPU
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv.yaml \
#     --options run.iters_per_epoch=125000 \
#     run.warmup_steps=244 \
#     run.max_epoch=12

# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_lisa_like_1token_latefusion.yaml \
#     --options run.iters_per_epoch=125000 \
#     run.warmup_steps=244 \
#     run.max_epoch=12

# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_lisa_like_1token.yaml\
#     --options run.iters_per_epoch=125000 \
#     run.warmup_steps=244 \
#     run.max_epoch=12

# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_lisa_like_1token_latefusion_e2e.yaml \
#     --options run.iters_per_epoch=125000 \
#     run.warmup_steps=244 \
#     run.max_epoch=12

# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_lisa_like_1token_LoRAOnly.yaml\
#     --options run.iters_per_epoch=125000 \
#     run.warmup_steps=244 \
#     run.max_epoch=12




# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv.yaml \
#     --options run.iters_per_epoch=125000 \
#     run.warmup_steps=244 \
#     run.max_epoch=12 \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_6dec_4hrs \
#     run.job_name=kvtoken_6dec


### 12 Dec

# i1
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv_pos_caption.yaml \
#     --options run.max_epoch=30 run.warmup_steps=977

# i2
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv_pos_caption.yaml \
#     --options model.caption=False \
#     run.max_epoch=30 run.warmup_steps=977 \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_30epochs \
#     run.job_name=kvtoken_30epochs
# #     run.max_epoch=30 run.warmup_steps=488 \

# i3
# timeout 13200s torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv_caption.yaml \
#     --options run.max_epoch=30 run.warmup_steps=500 \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs \
#     run.job_name=kvtoken_caption_30epochs run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs/checkpoint_last.pth"

# sbatch $0
#python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#    --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv_caption.yaml \
#    --options run.max_epoch=30 run.warmup_steps=1000 \
#    run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs \
#    run.job_name=kvtoken_caption_30epochs run.distributed=False

# i4
# torchrun --nproc-per-node 2 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_kv_caption.yaml \
#     --options run.max_epoch=30 \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_fulldata \
#     run.job_name=kvtoken_caption_fulldata
