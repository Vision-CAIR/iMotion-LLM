#!/bin/bash

conda activate gameformer
## DEC
checkpoints=("checkpoint_4.pth" "checkpoint_3.pth" "checkpoint_2.pth" "checkpoint_1.pth")
# checkpoints=("checkpoint_2.pth")




for checkpoint in "${checkpoints[@]}"; do
    # /ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion_6dec
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
            run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion_6dec/${checkpoint}" \
            run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/eval
    
    # /ibex/project/c2278/felembaa/models/imotion/dec/qtoken_6dec
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
            model.late_fusion=False \
            model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData/epochs_29.pth' \
            run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_6dec/${checkpoint}" \
            run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/eval

    
    # /ibex/project/c2278/felembaa/models/imotion/dec/qtoken_LoRAonly_6dec
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
            model.late_fusion=False \
            model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData/epochs_29.pth' \
            run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_LoRAonly_6dec/${checkpoint}" \
            run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/eval
    
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
            model.late_fusion=False \
            model.act_kv=True \
            model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth' \
            run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_6dec/${checkpoint}" \
            run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/eval

    # /ibex/project/c2278/felembaa/models/imotion/dec/kvtoken_6dec

    
done

# for checkpoint in "${checkpoints[@]}"; do
#     # /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_dec/train/train_lisa_like_1token.yaml
#     echo "*********************************************************"
#     echo "*********************************************************"
#     echo "*********************************************************"
#     echo "*********************************************************"
#     echo "*********************************************************"
#     echo "*********************************************************"
#     echo "*********************************************************"
#     echo "*********************************************************"
#     python /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#             --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#             --options model.freeze_gf=True \
#             model.freeze_gf_env=True \
#             model.freeze_gf_dec=True \
#             model.lora_dropout=0 \
#             run.distributed=False \
#             model.img_token_num=1 \
#             model.gf_act_dec=True \
#             model.neighbors_to_predict=0 \
#             model.gf_token_num=114 \
#             model.late_fusion=True \
#             model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData/epochs_29.pth' \
#             run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/${checkpoint}" \
#             run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/eval

#     # torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     #         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#     #         --options model.freeze_gf=True \
#     #         model.freeze_gf_env=True \
#     #         model.freeze_gf_dec=True \
#     #         model.lora_dropout=0 \
#     #         run.distributed=True \
#     #         model.img_token_num=1 \
#     #         model.gf_act_dec=True \
#     #         model.neighbors_to_predict=0 \
#     #         model.gf_token_num=114 \
#     #         model.late_fusion=True \
#     #         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData/epochs_29.pth' \
#     #         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/${checkpoint}" \
#     #         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/dec/qtoken_latefusion/eval
# done

## NOV
# gf_encoder_path: '/ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth'
# gf_encoder_path: '/ibex/project/c2278/felembaa/models/gameformer/gf_23aug/epochs_29.pth'

# checkpoints=("checkpoint_last.pth" "checkpoint_0.pth" "checkpoint_5.pth" "checkpoint_10.pth" "checkpoint_15.pth")
# checkpoints=("checkpoint_last.pth" "checkpoint_5.pth")
# checkpoints=("checkpoint_5.pth")
# checkpoints=("checkpoint_15.pth" "checkpoint_10.pth" "checkpoint_5.pth" "checkpoint_0.pth")

# for checkpoint in "${checkpoints[@]}"; do    
#     echo "/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_24nov_onlyEnc"
#     torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=True \
#         model.img_token_num=1 \
#         model.gf_act_dec=True \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_24nov_onlyEnc/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_24nov_onlyEnc/eval
    
#     echo "/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_24nov_OnlyLora"
#     torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=True \
#         model.img_token_num=1 \
#         model.gf_act_dec=True \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_24nov_OnlyLora/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_24nov_OnlyLora/eval
    
#     echo "/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_24nov_OnlyLora"
#     torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=True \
#         model.img_token_num=1 \
#         model.gf_act_dec=True \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_24nov_e2e/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_24nov_e2e/eval
    
# done

# for checkpoint in "${checkpoints[@]}"; do
    # echo "/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtoken_gf_21nov_01"
    # torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
    #     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
    #     --options model.freeze_gf=True \
    #     model.freeze_gf_env=True \
    #     model.freeze_gf_dec=True \
    #     model.lora_dropout=0 \
    #     run.distributed=True \
    #     model.img_token_num=1 \
    #     model.gf_act_dec=False \
    #     model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/gf_23aug/epochs_29.pth' \
    #     run.resume_ckpt_path=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtoken_gf_21nov_01/checkpoint_last.pth \
    #     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtoken_gf_21nov_01/eval
    
    # echo "/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_21nov_01"
    # torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
    #     --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
    #     --options model.freeze_gf=True \
    #     model.freeze_gf_env=True \
    #     model.freeze_gf_dec=True \
    #     model.lora_dropout=0 \
    #     run.distributed=True \
    #     model.img_token_num=1 \
    #     model.gf_act_dec=True \
    #     model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth' \
    #     run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_21nov_01/${checkpoint}" \
    #     run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_1qtokens_21nov_01/eval

#     echo "/ibex/project/c2278/felembaa/models/imotion/nov/imotion_6qtokens_21nov_01"
#     torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=True \
#         model.img_token_num=6 \
#         model.gf_act_dec=True \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_6qtokens_21nov_01/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_6qtokens_21nov_01/eval

#     echo "/ibex/project/c2278/felembaa/models/imotion/nov/imotion_6qtokens_gf_21nov_01"
#     torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=True \
#         model.img_token_num=6 \
#         model.gf_act_dec=False \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/gf_23aug/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_6qtokens_gf_21nov_01/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_6qtokens_gf_21nov_01/eval

#     echo "/ibex/project/c2278/felembaa/models/imotion/nov/imotion_3tokens_21nov_01"
#     torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=True \
#         model.img_token_num=3 \
#         model.gf_act_dec=True \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_3tokens_21nov_01/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_3tokens_21nov_01/eval

#     echo "/ibex/project/c2278/felembaa/models/imotion/nov/imotion_3tokens_e2e_21nov_01"
#     torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#         --cfg-path /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_nov/eval/eval_base.yaml \
#         --options model.freeze_gf=True \
#         model.freeze_gf_env=True \
#         model.freeze_gf_dec=True \
#         model.lora_dropout=0 \
#         run.distributed=True \
#         model.img_token_num=3 \
#         model.gf_act_dec=True \
#         model.gf_encoder_path='/ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth' \
#         run.resume_ckpt_path="/ibex/project/c2278/felembaa/models/imotion/nov/imotion_3tokens_e2e_21nov_01/${checkpoint}" \
#         run.output_dir=/ibex/project/c2278/felembaa/models/imotion/nov/imotion_3tokens_e2e_21nov_01/eval
# done
# echo "Done"


