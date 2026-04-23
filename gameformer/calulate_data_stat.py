import torch
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('/home/felembaa/projects/iMotion-LLM-ICLR')
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

print("************* ******** ****************")
print("************* OLD DATA ****************")

# train_set = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_23aug'
# train_dataset = DrivingData(train_set+'/*', act=True, full_map=False, ego_act_only=True, nuplan=False, random_drop_act=False, files_with_act_only=True)
# train_dataloader = DataLoader(train_dataset, batch_size=1000, num_workers=10)
# directions_stat_overall = {i:0 for i in range(8)}
# for batch in tqdm(train_dataloader):
#     acts = batch[7][:,0]
#     acts_frequency = [sum(acts==i).item() for i in range(8)]
#     for i, count in enumerate(acts_frequency):
#         directions_stat_overall[i] += count

# print("overall directions stat")
# print(directions_stat_overall)

# directions_stat_overall = {0: 43927, 1: 1095470, 2: 110125, 3: 117542, 4: 285034, 5: 343776, 6: 1618, 7: 13773}
# print(f"Total examples: {sum(list(directions_stat_overall.values()))}")
# print(directions_stat_overall)
# for i, cls in enumerate(train_dataset.direction_classes):
#     print(f"> {cls}: {np.array(list(directions_stat_overall.values()))[i] / sum(list(directions_stat_overall.values()))*100:.2f}%")


print()
print("************* ******** ****************")
print("************* NEW DATA ****************")
train_set = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov'
train_dataset = DrivingData(train_set+'/*', act=True, full_map=False, ego_act_only=True, nuplan=False, random_drop_act=False, files_with_act_only=True)
train_dataloader = DataLoader(train_dataset, batch_size=1000, num_workers=10)
directions_stat_overall = {i:0 for i in range(8)}
for batch in tqdm(train_dataloader):
    acts = batch[7][:,0]
    acts_frequency = [sum(acts==i).item() for i in range(8)]
    for i, count in enumerate(acts_frequency):
        directions_stat_overall[i] += count

# directions_stat_overall = 
print("************* ******** ****************")
print("************* NEW DATA ****************")
print(f"Total examples: {sum(list(directions_stat_overall.values()))}")
print(directions_stat_overall)
for i, cls in enumerate(train_dataset.direction_classes):
    print(f"> {cls}: {np.array(list(directions_stat_overall.values()))[i] / sum(list(directions_stat_overall.values()))*100:.2f}%")

print("overall directions stat")
print(directions_stat_overall)