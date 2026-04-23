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
print("************* NEW DATA ****************")
train_set = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/training'
# train_set = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation'
train_dataset = DrivingData(train_set+'/*', act=True, full_map=False, ego_act_only=True, nuplan=False, random_drop_act=False, files_with_act_only=True, num_classes=5)
train_dataloader = DataLoader(train_dataset, batch_size=1000, num_workers=10)
# directions_stat_overall = {i:0 for i in range(8)}
directions_stat_overall = {i:0 for i in range(5)}
for batch in tqdm(train_dataloader):
    acts = batch[7][:,0]
    acts_frequency = [sum(acts==i).item() for i in range(5)]
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

# Compute percentages
total_examples = sum(directions_stat_overall.values())
direction_percentages = {
    train_dataset.direction_classes[i]: (directions_stat_overall[i] / total_examples * 100) if total_examples > 0 else 0
    for i in range(5)
}


# Save statistics to CSV
csv_path = train_set + '_stat.csv'

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Direction Class", "Count", "Percentage"])
    
    for i, cls in enumerate(train_dataset.direction_classes):
        writer.writerow([cls, directions_stat_overall[i], direction_percentages[cls]])

    # Write total count
    writer.writerow(["Total", total_examples, "100%"])

print(f"Statistics saved to {csv_path}")