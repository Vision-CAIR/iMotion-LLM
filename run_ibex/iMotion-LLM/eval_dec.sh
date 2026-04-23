#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --mem=32GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --job-name=eval
#SBATCH --output=/home/felembaa/logs/%j-%x.out
#SBATCH --error=/home/felembaa/errs/%j-%x.err
#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL


# # module load cuda/12.2
# # source ~/miniconda3/bin/activate gameformer
source ~/miniconda3/bin/activate /ibex/project/c2278/felembaa/envs/imotion


python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
    --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/eval/eval_base_27dec.yaml \
    --options model.freeze_gf=True \
    model.freeze_gf_env=True \
    model.freeze_gf_dec=True \
    model.lora_dropout=0 \
    run.distributed=False \
    model.img_token_num=1 \
    model.neighbors_to_predict=0 \
    model.gf_token_num=114 \
    model.late_fusion=False \
    model.act_kv=True \
    model.gf_act_dec=False \
    model.caption=True \
    model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
    model.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/template_31dec/checkpoint_1.pth" \
    run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/template_31dec/checkpoint_1.pth" \
    run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/template_31dec/eval
    
    
## 22 dec
# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/eval/eval_base_captiongt.yaml \
#     --options model.freeze_gf=True \
#     model.freeze_gf_env=True \
#     model.freeze_gf_dec=True \
#     model.lora_dropout=0 \
#     run.distributed=False \
#     model.img_token_num=1 \
#     model.neighbors_to_predict=0 \
#     model.gf_token_num=114 \
#     model.late_fusion=False \
#     model.act_kv=True \
#     model.gf_act_dec=False \
#     model.caption=True \
#     model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
#     run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/template_31dec/checkpoint_10.pth" \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/template_31dec/eval

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/eval/eval_base_captionneg.yaml \
#     --options model.freeze_gf=True \
#     model.freeze_gf_env=True \
#     model.freeze_gf_dec=True \
#     model.lora_dropout=0 \
#     run.distributed=False \
#     model.img_token_num=1 \
#     model.neighbors_to_predict=0 \
#     model.gf_token_num=114 \
#     model.late_fusion=False \
#     model.act_kv=True \
#     model.gf_act_dec=False \
#     model.caption=True \
#     model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
#     run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs/checkpoint_4.pth" \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs/eval

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/eval/eval_base_captionpos.yaml \
#     --options model.freeze_gf=True \
#     model.freeze_gf_env=True \
#     model.freeze_gf_dec=True \
#     model.lora_dropout=0 \
#     run.distributed=False \
#     model.img_token_num=1 \
#     model.neighbors_to_predict=0 \
#     model.gf_token_num=114 \
#     model.late_fusion=False \
#     model.act_kv=True \
#     model.gf_act_dec=False \
#     model.caption=True \
#     model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
#     run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs/checkpoint_4.pth" \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption_30epochs/eval



# checkpoints=("checkpoint_4.pth" "checkpoint_3.pth" "checkpoint_2.pth" "checkpoint_1.pth")
# for checkpoint in "${checkpoints[@]}"; do
#     echo "*********************************************************"
#     python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=False \
#         model.img_token_num=1 \
#         model.gf_act_dec=True \
#         model.neighbors_to_predict=0 \
#         model.gf_token_num=114 \
#         model.late_fusion=False \
#         model.act_kv=True \
#         model.gf_act_dec=False \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_6dec/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_6dec/eval
# done

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#     --options model.freeze_gf=True \
#     model.freeze_gf_env=True \
#     model.freeze_gf_dec=True \
#     model.lora_dropout=0 \
#     run.distributed=False \
#     model.img_token_num=1 \
#     model.gf_act_dec=True \
#     model.neighbors_to_predict=0 \
#     model.gf_token_num=114 \
#     model.late_fusion=False \
#     model.act_kv=True \
#     model.gf_act_dec=False \
#     model.caption=True \
#     model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
#     run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_captionPos/checkpoint_4.pth" \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_captionPos/eval

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#     --options model.freeze_gf=True \
#     model.freeze_gf_env=True \
#     model.freeze_gf_dec=True \
#     model.lora_dropout=0 \
#     run.distributed=False \
#     model.img_token_num=1 \
#     model.gf_act_dec=True \
#     model.neighbors_to_predict=0 \
#     model.gf_token_num=114 \
#     model.late_fusion=False \
#     model.act_kv=True \
#     model.gf_act_dec=False \
#     model.caption=True \
#     model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
#     run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption/checkpoint_4.pth" \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption/eval

# python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/eval/eval_base_captionneg.yaml \
#     --options model.freeze_gf=True \
#     model.freeze_gf_env=True \
#     model.freeze_gf_dec=True \
#     model.lora_dropout=0 \
#     run.distributed=False \
#     model.img_token_num=1 \
#     model.gf_act_dec=True \
#     model.neighbors_to_predict=0 \
#     model.gf_token_num=114 \
#     model.late_fusion=False \
#     model.act_kv=True \
#     model.gf_act_dec=False \
#     model.caption=True \
#     model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
#     run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption/checkpoint_4.pth" \
#     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_caption/eval



#!/usr/bin/env bash
# #SBATCH --time=03:00:00
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --constraint=a100
# #SBATCH --mem=32GB
# #SBATCH --cpus-per-gpu=4
# #SBATCH --partition=batch
# #SBATCH --job-name=kvtoken_6dec
# #SBATCH --output=/home/felembaa/logs/%j-%x.out
# #SBATCH --error=/home/felembaa/errs/%j-%x.err
# #SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
# #SBATCH --mail-type=ALL

# module load cuda/12.2
# source ~/miniconda3/bin/activate gameformer

# checkpoints=("checkpoint_4.pth" "checkpoint_3.pth" "checkpoint_2.pth" "checkpoint_1.pth")
# for checkpoint in "${checkpoints[@]}"; do
#     echo "*********************************************************"
#     python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=False \
#         model.img_token_num=1 \
#         model.gf_act_dec=True \
#         model.neighbors_to_predict=0 \
#         model.gf_token_num=114 \
#         model.late_fusion=False \
#         model.act_kv=True \
#         model.gf_act_dec=False \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_6dec/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_6dec/eval
# done



##SBATCH --time=03:00:00
##SBATCH --nodes=1
##SBATCH --gpus-per-node=1
##SBATCH --constraint=a100
##SBATCH --mem=32GB
##SBATCH --cpus-per-gpu=4
##SBATCH --partition=batch
##SBATCH --job-name=qtoken_LoRAonly_6dec
##SBATCH --output=/home/felembaa/logs/%j-%x.out
##SBATCH --error=/home/felembaa/errs/%j-%x.err
##SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
##SBATCH --mail-type=ALL

# module load cuda/12.2
# source ~/miniconda3/bin/activate gameformer

# checkpoints=("checkpoint_4.pth" "checkpoint_3.pth" "checkpoint_2.pth" "checkpoint_1.pth")
# for checkpoint in "${checkpoints[@]}"; do
#     echo "*********************************************************"
#     python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=False \
#         model.img_token_num=1 \
#         model.gf_act_dec=True \
#         model.neighbors_to_predict=0 \
#         model.gf_token_num=114 \
#         model.late_fusion=False \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_LoRAonly_6dec/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_LoRAonly_6dec/eval
# done



# #SBATCH --time=03:00:00
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --constraint=a100
# #SBATCH --mem=32GB
# #SBATCH --cpus-per-gpu=4
# #SBATCH --partition=batch
# #SBATCH --job-name=qtoken_6dec
# #SBATCH --output=/home/felembaa/logs/%j-%x.out
# #SBATCH --error=/home/felembaa/errs/%j-%x.err
# #SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
# #SBATCH --mail-type=ALL

# module load cuda/12.2
# source ~/miniconda3/bin/activate gameformer

# checkpoints=("checkpoint_4.pth" "checkpoint_3.pth" "checkpoint_2.pth" "checkpoint_1.pth")
# for checkpoint in "${checkpoints[@]}"; do
#     echo "*********************************************************"
#     python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=False \
#         model.img_token_num=1 \
#         model.gf_act_dec=True \
#         model.neighbors_to_predict=0 \
#         model.gf_token_num=114 \
#         model.late_fusion=False \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_6dec/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_6dec/eval
# done



# #SBATCH --time=03:00:00
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --constraint=a100
# #SBATCH --mem=32GB
# #SBATCH --cpus-per-gpu=4
# #SBATCH --partition=batch
# #SBATCH --job-name=qtoken_latefusion_6dec
# #SBATCH --output=/home/felembaa/logs/%j-%x.out
# #SBATCH --error=/home/felembaa/errs/%j-%x.err
# #SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
# #SBATCH --mail-type=ALL

# module load cuda/12.2
# source ~/miniconda3/bin/activate gameformer

# checkpoints=("checkpoint_4.pth" "checkpoint_3.pth" "checkpoint_2.pth" "checkpoint_1.pth")
# for checkpoint in "${checkpoints[@]}"; do
#     echo "*********************************************************"
#     python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=False \
#         model.img_token_num=1 \
#         model.gf_act_dec=True \
#         model.neighbors_to_predict=0 \
#         model.gf_token_num=114 \
#         model.late_fusion=True \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion_6dec/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion_6dec/eval
# done
