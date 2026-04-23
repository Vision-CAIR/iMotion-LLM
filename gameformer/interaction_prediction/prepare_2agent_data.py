## This code takes an already preprocessed data, and search for agent pairs in order to augment the data
# The goal, is to have a template description for both the ego and the interactive agents
# This can be searched using the file names in /ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_full_3jul
# Since we are using the small data for finetuning the LLM, the small data should be also augmented not the training_full_3jul
## Dirs to augment
# > training_full_3jul
# > validation_3jul
# > training_small_1jul
# > 
# >

import os
import torch
import logging
import glob
import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from google.protobuf import text_format

from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
# from instructions.direction_instructions import DirectionClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import json
import matplotlib.pyplot as plt

from tqdm import tqdm
import shutil

root_dir = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/'

train_dir_full = 'training_full_3jul'
train_dir_small = 'training_small_1jul'
valid_dir = 'validation_3jul'


train_full_data_list = glob.glob(root_dir+train_dir_full+'/*')
print(len(train_full_data_list))
filenames = train_full_data_list
output_dir = root_dir+train_dir_full+'_filenames_T/filenames_T.json'
output_root_dir = root_dir+train_dir_full

# train_small_data_list = glob.glob(root_dir+train_dir_small+'/*')
# print(len(train_small_data_list))
# filenames = train_small_data_list
# output_dir = root_dir+train_dir_small+'_filenames_T/filenames_T.json'
# output_root_dir = root_dir+train_dir_small

# valid_data_list = glob.glob(root_dir+valid_dir+'/*')
# print(len(valid_data_list))
# filenames = valid_data_list
# output_dir = root_dir+valid_dir+'_filenames_T/filenames_T.json'
# output_root_dir = root_dir+valid_dir

filename_T_dict = {}
filenames_str = ';'.join(filenames)
long_string = filenames_str
for filename in tqdm(filenames):
    start_substring = '_'.join([filename.split('/')[-1].split('_')[0], filename.split('/')[-1].split('_')[2]])
    start_index = long_string.find(start_substring)
    end_substring = '.npz'
    if start_index != -1:
        end_index = long_string.find(end_substring, start_index) + len(end_substring)
        if end_index != -1:
            filename_T_dict[filename] = output_root_dir+'/'+long_string[start_index:end_index]

ff = '/'.join(output_dir.split('/')[:-1])
if os.path.exists(ff):
    shutil.rmtree(ff)
os.makedirs(ff, exist_ok=True)
with open(output_dir, 'w') as file:
    json.dump(filename_T_dict, file, indent=4)



