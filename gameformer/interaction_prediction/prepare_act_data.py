# torchrun --nproc-per-node 4 ~/project/gameformer_p/interaction_prediction/train.py --wandb --distributed --name gameformer_temp1
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --wandb --distributed --name gameformer_temp1 --batch_size 8
# torchrun --nproc-per-node 2 train.py --distributed --workers 8 --batch_size 64 --train_set <internal_user_root>/waymo_dataset/training_interactive_gameformer_10hz_agentorder --valid_set <internal_user_root>/waymo_dataset/validation_interactive_gameformer_10hz --subsample False --future_len 80
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --act --act_dec --save_model <internal_user_root>/gameformer_models/gf_4mar_lvl0_act/ --name gf_4mar_lvl0_act --load_dir '' --wandb --distributed --workers 8 --batch_size 64 --level 0 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20nvidia-smi
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --save_model <internal_user_root>/gameformer_models/gf_4mar_lvl6_valid/ --name gf_4mar_lvl6_valid --load_dir '' --wandb --distributed --workers 8 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20nvidia-smi
import torch
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('<legacy_repo_root>')
sys.path.append('...')
import csv
import argparse
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from gameformer.interaction_prediction.eval import validation_epoch, validation_epoch_pkl

import time
from gameformer.model.GameFormer import GameFormer, GameFormer_
# from model.GameFormer import GameFormer_2
from gameformer.utils.inter_pred_utils import *
import wandb

from gameformer.interaction_prediction.dist_utils import get_rank, init_distributed_mode
from gameformer.interaction_prediction.logger import setup_logger
from gameformer.interaction_prediction.logger import MetricLogger, SmoothedValue
from tqdm import tqdm
# from multimodal_viz import *

# from exctract_instruct import *

data_dir = '<internal_dataset_root>/waymo/gameformer/training_21aug/*'
# data_list = glob.glob(data_dir)
# agentJsons_dir = f"{data_dir[:-2]}_agentJsons"
# agentJsons_list = glob.glob(f"{agentJsons_dir}/*")
# out_data = {file_name.split('/')[-1].split('.')[0]:{'act0':'-1', 'act1':'-1'} for file_name in data_list}
# agentJsons_dir = f"{data_dir[:-2]}_agentJsons"
template_dir = '<internal_dataset_root>/waymo/gameformer/training_21aug_templateLLM/*'
data_list = glob.glob(template_dir)
for file_name in tqdm(data_list):
    with open(file_name) as f:
        lines = f.readlines()
    templates = [json.loads(line) for line in lines]
    continue
    # acts = np.array([-1,-1])
    file_name = file_name.split('/')[-1].split('.')[0]
    agent_json_dir = f"{agentJsons_dir}/{file_name}.json"
    
    if agent_json_dir in agentJsons_list:
        with open(agent_json_dir, 'r') as file:
            agent_json_data = file.read()
        # agent_data_dict = json.loads(agent_json_data)
        # acts[0] = agent_data_dict['Agent-1']['direction 0.1to8_cls'] if ('Agent-1' in agent_data_dict.keys())  else -1
        # acts[1] = agent_data_dict['Agent-2']['direction 0.1to8_cls'] if ('Agent-2' in agent_data_dict.keys()) else -1
        # out_data[file_name]['act0'] = str(acts[0])
        # out_data[file_name]['act1'] = str(acts[1])

print('')