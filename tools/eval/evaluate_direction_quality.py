### Code to load trajectory dataset, and visualize each example with the direction 
### The code should give 100 examples, per direction, per instruction type (gt, other feasible, infeasible)
### The could prompt for correct or wrong score, and save the example name, the instruction type, and the direction in a csv along with the given score
### Eventually it calculate the score for each direction per each instruction type
from gameformer.utils.inter_pred_utils import *
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

train_dataset = DrivingData('/ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov/*', act=True, full_map=True, ego_act_only=True, nuplan=False, random_drop_act=False, files_with_act_only=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
for sample in train_loader:
    break
print('')