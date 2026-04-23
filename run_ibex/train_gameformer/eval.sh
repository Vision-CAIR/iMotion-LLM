#!/usr/bin/env bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
##SBATCH --constraint=v100
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err
#SBATCH --job-name=ev_gf
#SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --exclude=gpu101-02-l
##SBATCH  --account conf-neurips-2024.05.22-elhosemh

source ~/miniconda3/bin/activate gameformer

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done

python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/gf_l1/epochs_29.pth --workers 4 --batch_size 1 --modalities 6 --future_len 80 --neighbors_to_predict 0 --num_act_classes 5 --level 1 --new_eval_mode gt1

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train_Forecasting.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/train_forecasting/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode gt1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train_Forecasting.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/train_forecasting/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode neg1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train_Forecasting.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/train_forecasting/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode pos1


# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/C_train_Forecasting.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_forecasting/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode gt1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/C_train_Forecasting.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_forecasting/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode neg1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/C_train_Forecasting.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_forecasting/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode pos1


# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/C_train_Forecasting.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_forecasting_without/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode gt1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/C_train_Forecasting.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_forecasting_without/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode neg1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/C_train_Forecasting.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_forecasting_without/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode pos1


# 21 Feb 2025
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_l1/epochs_29.pth --workers 4 --batch_size 1 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode gt1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_l1/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode neg1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_l1/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 1 --new_eval_mode pos1

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 3 --new_eval_mode gt1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 3 --new_eval_mode neg1
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf/epochs_29.pth --workers 4 --batch_size 64 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec --num_act_classes 5 --level 3 --new_eval_mode pos1

## 19 Feb 2025
# GF
# # torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --distributed --workers 8 --batch_size 64 --level 3 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov --save_model /ibex/project/c2278/felembaa/models/gameformer/gf_1a_29nov_newData --name gf_1a_29nov_newData --load_dir '' --wandb --neighbors_to_predict 0
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/gf/epochs_29.pth --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --neighbors_to_predict 0

## Synthetic data generaton
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/generate_of_synth_examples.py --batch_size 64

# # GF
# # torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --distributed --workers 8 --batch_size 64 --level 3 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov --save_model /ibex/project/c2278/felembaa/models/gameformer/gf_1a_29nov_newData --name gf_1a_29nov_newData --load_dir '' --wandb --neighbors_to_predict 0
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_1a_29nov_newData/epochs_29.pth --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --neighbors_to_predict 0

# # CGF - Act files only
# # torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --distributed --workers 8 --batch_size 64 --level 3 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov --save_model /ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData --name cgf_1a_29nov_newData --act_dec --load_dir '' --wandb --neighbors_to_predict 0 --files_with_act_only --no_random_drop_act
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData/epochs_29.pth --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec
# # CGF - Act files only - kv acts
# # torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --distributed --workers 8 --batch_size 64 --level 3 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov --save_model /ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact --name cgf_1a_29nov_newData_kvact --act_kv --load_dir '' --wandb --neighbors_to_predict 0 --files_with_act_only --no_random_drop_act
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_kvact/epochs_29.pth --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_kv
# # CGF - All files
# # torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --distributed --workers 8 --batch_size 64 --level 3 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov --save_model /ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_alldata_randDrop --name cgf_1a_29nov_newData_alldata_randDrop --act_dec --load_dir '' --wandb --neighbors_to_predict 0
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_alldata_randDrop/epochs_29.pth --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec
# # CGF - Act file only & random drip
# # torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --distributed --workers 8 --batch_size 64 --level 3 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov --save_model /ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_randDrop --name cgf_1a_29nov_newData_randDrop --act_dec --load_dir '' --wandb --neighbors_to_predict 0 --files_with_act_only
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_1a_29nov_newData_randDrop/epochs_29.pth --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --neighbors_to_predict 0 --act_dec


# torchrun --nproc-per-node 4 --master_port ${PORT} \
# --nproc-per-node 4 \
# /home/felembaa/projects/gameformer_p/interaction_prediction/train.py \
# --wandb \
# --distributed \
# --workers 8 \
# --batch_size 64 \
# --level 6 \
# --modalities 6 \
# --encoder_layers 6 \
# --future_len 80 \
# --learning_rate 1e-4 \
# --subsample False \
# --lr_steps '18,21,24,27,30' \
# --training_epochs 30 \
# --name gf_15feb_01 \
# --train_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20 \
# --valid_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20 \
# --save_model /ibex/user/felembaa/gameformer_models/gf_4mar_lvl0/ \
# --load_dir ''
# --load_dir /ibex/user/felembaa/gameformer_models/gf_15feb_02/epochs_last.pth \
# --gmm \
#/ibex/project/c2253/felembaa/waymo_dataset/training_interactive_original_20


# torchrun --nproc-per-node 4 /home/felembaa/projects/gameformer_p/interaction_prediction/train.py --save_model /ibex/user/felembaa/gameformer_models/gf_4mar_lvl0/ --name gf_4mar_lvl0 --load_dir '' --wandb --distributed --workers 8 --batch_size 64 --level 0 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20 --valid_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/gameformer_p/interaction_prediction/train.py --act --act_dec --save_model /ibex/user/felembaa/gameformer_models/gf_4mar_lvl6_act/ --name gf_4mar_lvl6_act --load_dir '' --wandb --distributed --workers 8 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20 --valid_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20

# act decoder only
# torchrun --nproc-per-node 4 /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --act_dec --save_model /ibex/user/felembaa/gameformer_models/gf_5may_fullmap/ --name gf_5may_fullmap --load_dir '' --wandb --distributed --workers 4 --batch_size 150 --level 3 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '10,15,21,24,27,30' --training_epochs 40 --train_set /ibex/project/c2253/felembaa/waymo/gameformer/training_5may_fullmap_small --valid_set /ibex/project/c2253/felembaa/waymo/gameformer/validation_5may_fullmap_small

# torchrun --nproc-per-node 4 /home/felembaa/projects/gameformer_p/interaction_prediction/train.py --act_dec --save_model /ibex/user/felembaa/gameformer_models/gf_4mar_lvl6_actdec_valid/ --name gf_4mar_lvl6_actdec_valid --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '10,15,21,24,27,30' --training_epochs 30 --train_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20 --valid_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20
# torchrun --nproc-per-node 4 /home/felembaa/projects/gameformer_p/interaction_prediction/train.py --act_dec --save_model /ibex/user/felembaa/gameformer_models/gf_4mar_lvl6_actdec_valid/ --name gf_4mar_lvl6_actdec_valid --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20 --valid_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20

# torchrun --nproc-per-node 4 /home/felembaa/projects/gameformer_p/interaction_prediction/train.py --act --act_dec --save_model /ibex/user/felembaa/gameformer_models/gf_4mar_lvl6_actdec_act_valid/ --name gf_4mar_lvl6_actdec_act_valid --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '10,15,21,24,27,30' --training_epochs 30 --train_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20 --valid_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20

# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/gameformer_p/interaction_prediction/train.py --save_model /ibex/user/felembaa/gameformer_models/gf_4mar_lvl0_3m/ --name gf_4mar_lvl0_3m --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 0 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set /ibex/project/c2253/felembaa/waymo_dataset/training_interactive_original_20 --valid_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20


# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/gameformer_p/interaction_prediction/train.py --act --act_dec --save_model /ibex/user/felembaa/gameformer_models/gf_4mar_lvl6_3m_act/ --name gf_4mar_lvl6_3m_act --load_dir '' --wandb --distributed --workers 8 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set /ibex/project/c2253/felembaa/waymo_dataset/training_interactive_original_20 --valid_set /ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_original_20

### 7 May
## temp
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --save_model /ibex/user/felembaa/gameformer_models/temp/ --name temp --load_dir '' --wandb --distributed --workers 4 --batch_size 128 --level 3 --modalities 6 --future_len 80 --learning_rate 2.8e-4 --lr_steps '20,22,24,26,28' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap_val --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap_val
## base
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --save_model /ibex/user/felembaa/gameformer_models/gf_7may_base_smalldata/ --name gf_7may_base_smalldata --load_dir '' --wandb --distributed --workers 4 --batch_size 128 --level 3 --modalities 6 --future_len 80 --learning_rate 2.8e-4 --lr_steps '20,22,24,26,28' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_training_5may_fullmap --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap_val
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --eval_only --load_dir /ibex/user/felembaa/gameformer_models/gf_7may_base_smalldata/epochs_last.pth --name eval_gf_7may_base_smalldata --wandb --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap
## full_map
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --full_map --save_model /ibex/user/felembaa/gameformer_models/gf_7may_fullmap_smalldata/ --name gf_7may_fullmap_smalldata --load_dir '' --wandb --distributed --workers 4 --batch_size 128 --level 3 --modalities 6 --future_len 80 --learning_rate 2.8e-4 --lr_steps '20,22,24,26,28' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_training_5may_fullmap --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap_val
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --full_map --eval_only --load_dir /ibex/user/felembaa/gameformer_models/gf_7may_fullmap_smalldata/epochs_last.pth --name eval_gf_7may_fullmap_smalldata --wandb --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap
## act_dec + full_map
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --act_dec --full_map --save_model /ibex/user/felembaa/gameformer_models/gf_7may_fullmap_act_smalldata/ --name gf_7may_fullmap_act_smalldata --load_dir '' --wandb --distributed --workers 4 --batch_size 128 --level 3 --modalities 6 --future_len 80 --learning_rate 2.8e-4 --lr_steps '20,22,24,26,28' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_training_5may_fullmap --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap_val
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --act_dec --full_map --eval_only --load_dir /ibex/user/felembaa/gameformer_models/gf_7may_fullmap_act_smalldata/epochs_last.pth --name eval_gf_7may_fullmap_act_smalldata --wandb --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap
## act_dec
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --act_dec --save_model /ibex/user/felembaa/gameformer_models/gf_7may_act_smalldata/ --name gf_7may_act_smalldata --load_dir '' --wandb --distributed --workers 4 --batch_size 128 --level 3 --modalities 6 --future_len 80 --learning_rate 2.8e-4 --lr_steps '20,22,24,26,28' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_training_5may_fullmap --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap_val
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --act_dec --eval_only --load_dir /ibex/user/felembaa/gameformer_models/gf_7may_act_smalldata/epochs_last.pth --name eval_gf_7may_act_smalldata --wandb --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap --valid_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/small_validation_5may_fullmap


## Jul
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --act_dec --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_2jul_smalldata/epochs_29.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode pos1 --act_dec --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_2jul_smalldata/epochs_29.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode neg1 --act_dec --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_2jul_smalldata/epochs_29.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --act_dec --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_7jul_fulldata/epochs_26.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode pos1 --act_dec --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_7jul_fulldata/epochs_26.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode neg1 --act_dec --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_7jul_fulldata/epochs_26.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_2jul_smalldata/epochs_29.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode pos1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_2jul_smalldata/epochs_29.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode neg1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_2jul_smalldata/epochs_29.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --full_map --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_2jul_smalldata_fullmap/epochs_29.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode pos1 --new_eval --eval_only --full_map --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_2jul_smalldata_fullmap/epochs_29.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode neg1 --new_eval --eval_only --full_map --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_2jul_smalldata_fullmap/epochs_29.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_7jul_fulldata/epochs_20.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode pos1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_7jul_fulldata/epochs_20.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80
# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode neg1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/gf_7jul_fulldata/epochs_20.pth --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80

# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80


# python /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train.py --new_eval_mode gt1 --new_eval --eval_only --load_dir /ibex/project/c2278/felembaa/models/gameformer/cgf_23aug/epochs_29.pth --workers 4 --batch_size 64 --level 3 --modalities 6 --future_len 80