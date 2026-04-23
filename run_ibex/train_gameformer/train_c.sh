#!/usr/bin/env bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
##SBATCH --constraint=a100
#SBATCH --constraint=v100
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --output=/home/felembaa/logs/%j.out
#SBATCH --error=/home/felembaa/errs/%j.err
## SBATCH --mail-user=nussair.hroub@kaust.edu.sa
#SBATCH --mail-type=ALL
##SBATCH --exclude=gpu101-02-l
#SBATCH --job-name=train_mlp
##SBATCH --account conf-iccv-2025.03.08-elhosemh

source ~/miniconda3/bin/activate gameformer
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done


# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/C_train_Forecasting.py --distributed --weighted_sampling --workers 8 --batch_size 64 --level 5 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/training --save_model /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_forecasting --name cgf_forecasting_july_1 --load_dir '' --wandb --neighbors_to_predict 0 --act_dec --num_act_classes 5

# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/C_train_Forecasting.py --distributed --workers 8 --batch_size 64 --level 1 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/training --save_model /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_forecasting_without --name cgf_forecasting_without_july_1 --load_dir '' --wandb --neighbors_to_predict 0 --act_dec --num_act_classes 5
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train_mlp.py --distributed --weighted_sampling --workers 8 --batch_size 64 --level 1 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/training --save_model /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/mlp_classifier --name mlp_classifier_15_july --load_dir '' --wandb --neighbors_to_predict 0 --act_dec --num_act_classes 5

# train mlp classifier only cgf
torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train_mlp.py --distributed --weighted_sampling --workers 8 --batch_size 64 --level 1 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/training --save_model /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/mlp_classifier_cgf --name mlp_classifier_15_july --load_dir '/ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/cgf_l1/epochs_29.pth' --wandb --neighbors_to_predict 0 --num_act_classes 5

# MLP Classifier based on Query content
# torchrun --nproc-per-node 4 --master_port ${PORT} /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/interaction_prediction/train_Forecasting.py --distributed --weighted_sampling --workers 8 --batch_size 64 --level 1 --modalities 6 --future_len 80 --learning_rate 1e-4 --lr_steps '15,18,21,24,27' --training_epochs 30 --train_set /ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/training --save_model /ibex/project/c2278/felembaa/models/gameformer/feb_18_2025/train_forecasting --name train_forecasting_17_july --load_dir '' --wandb --neighbors_to_predict 0 --num_act_classes 5
