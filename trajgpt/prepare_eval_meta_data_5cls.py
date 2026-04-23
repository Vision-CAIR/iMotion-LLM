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
import sys
sys.path.append(".")
sys.path.append("/home/felembaa/projects/iMotion-LLM-ICLR/")
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

from minigpt4.datasets.datasets.traj_dataset import TrajAlignDataset
from torch.utils.data import DataLoader
from collections import Counter
# /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/minigpt4/datasets/datasets/traj_dataset.py

def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

# root_dir = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/'
# root_dir = '/ibex/project/c2278/felembaa/datasets/nuplan/test/'

# root_dir = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/'
root_dir = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation_more_data/'
valid_dir = 'validation'

# train_dir_full = 'training_full_3jul'
# train_dir_small = 'training_small_1jul'
# valid_dir = 'validation_30sep'
# valid_dir = 'test_boston_processed'
# valid_dir = 'test_pittsburgh_processed'
# valid_dir = 'test_singapore_processed'


# train_full_data_list = glob.glob(root_dir+train_dir_full+'/*')
# print(len(train_full_data_list))
# filenames = train_full_data_list
# output_dir = root_dir+train_dir_full+'_filenames_T/filenames_T.json'
# output_root_dir = root_dir+train_dir_full

# train_small_data_list = glob.glob(root_dir+train_dir_small+'/*')
# print(len(train_small_data_list))
# filenames = train_small_data_list
# output_dir = root_dir+train_dir_small+'_filenames_T/filenames_T.json'
# output_root_dir = root_dir+train_dir_small

filenames = glob.glob(root_dir+valid_dir+'/*')
two_agent=False
if not two_agent:
        # if two_agent:
    dataset_ = TrajAlignDataset(root_dir+valid_dir+'/*', act=True, template_select=0, contrastive=False, train=False, two_agent=False, act_json=False, return_meta_data=True, num_classes=5)
    loader = DataLoader(dataset_, batch_size=1,num_workers=0)
    # dataset_contrastive = TrajAlignDataset(root_dir+valid_dir+'/*', act=True, template_select=3, contrastive=True, train=True, two_agent=True, act_json=False)
    # loader_c = DataLoader(dataset_contrastive, batch_size=1,num_workers=0)
    print(len(dataset_))
    output_dir = root_dir+valid_dir+'_eval_meta/eval_meta.json'
    output_root_dir = root_dir+valid_dir
    # meta_data = {'filename': {}, 'filename_T': {}, 'gt': {}, 'feasible': {}, 'infeasible':{}}
    meta_data = {}
    # filenames_str = ';'.join(filenames)
    # long_string = filenames_str
    ff = '/'.join(output_dir.split('/')[:-1])
    if not os.path.exists(ff):
        for data in tqdm(loader):
            # meta_data['filename'] = {'gt': , 'pos': [], 'neg':[]}
            meta_data[data['file_name'][0]] = {
                                    'gt1': data['act'][0].item(), 'pos1': list(data['positive_acts'][0].numpy()), 'neg1': list(data['contrastive_acts'][0].numpy()),
                                    # 'gt2': data['act'][0][1].item(), 'pos2': list(data['positive_acts2'][0].numpy()), 'neg2': list(data['contrastive_acts2'][0].numpy()),
                                    # 'filename_T': data['file_name_2'][0],
                                    }
        
        meta_data2 = convert_to_serializable(meta_data)

    # ff = '/'.join(output_dir.split('/')[:-1])
    # if os.path.exists(ff):
    #     shutil.rmtree(ff)
        os.makedirs(ff, exist_ok=True)
        with open(output_dir, 'w') as file:
            json.dump(meta_data2, file, indent=4)


    with open(output_dir, "r") as f:
        meta_data = f.read()
    # Convert JSON string back to Python dictionary
    meta_data = json.loads(meta_data)

    # if two_agent:
    #     # checking files of valid instruction detected
    valid_act_files = []
    for k,v in tqdm(meta_data.items()):
        if v['gt1']!=-1:
            if k not in valid_act_files:
                valid_act_files.append(k)
    print(f"Files with two valid ground truth instructions: {len(valid_act_files)}/{len(meta_data)}")

    



    print('***** processing data splits *****')

    meta_data_ = {k:v for k,v in meta_data.items() if k in valid_act_files}
    # desired_per_label_count = [800, 800, 800, 800, 64, 800]
    meta_gt1, meta_pos1, meta_neg1 = {}, {}, {}

    if 'nuplan' in root_dir:
        desired_per_label_count = [250, 250, 250, 250, 250, 250, 0, 0]
    else:
        # desired_per_label_count = [400, 400, 400, 400, 400, 400, 37, 400]
        # desired_per_label_count = [200, 200, 200, 200, 200]
        desired_per_label_count = [400, 400, 400, 400, 400]
    
    # desired_per_label_count = [400, 400, 400, 400, 400, 400, 37, 400]

    for desired_label, desired_count in enumerate(desired_per_label_count):
        c_gt1, c_pos1, c_neg1 = 0, 0, 0
        for k,v in meta_data_.items():
            save_name = k+f"_act{desired_label}"
            if v['gt1']==desired_label and c_gt1<desired_count and save_name not in meta_gt1.keys():
                meta_gt1[save_name] = {'gt1': desired_label}
                c_gt1+=1
            if desired_label in v['pos1'] and c_pos1<desired_count and save_name not in meta_pos1.keys():
                meta_pos1[save_name] = {'pos1': desired_label, 'gt1': v['gt1']}
                c_pos1+=1
            if desired_label in v['neg1'] and c_neg1<desired_count and save_name not in meta_neg1.keys():
                meta_neg1[save_name] = {'neg1': desired_label, 'gt1': v['gt1']}
                c_neg1+=1
        print(f"Label {desired_label}: AS ({c_gt1}), OF ({c_pos1}), INF ({c_neg1})")
    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_gt1.json", 'w') as file:
        json.dump(meta_gt1, file, indent=4)
    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_pos1.json", 'w') as file:
        json.dump(meta_pos1, file, indent=4)
    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_neg1.json", 'w') as file:
        json.dump(meta_neg1, file, indent=4)
else:
    # if two_agent:
    dataset_ = TrajAlignDataset(root_dir+valid_dir+'/*', act=True, template_select=3, contrastive=False, train=True, two_agent=True, act_json=False)
    loader = DataLoader(dataset_, batch_size=1,num_workers=0)
    # dataset_contrastive = TrajAlignDataset(root_dir+valid_dir+'/*', act=True, template_select=3, contrastive=True, train=True, two_agent=True, act_json=False)
    # loader_c = DataLoader(dataset_contrastive, batch_size=1,num_workers=0)
    print(len(dataset_))
    output_dir = root_dir+valid_dir+'_eval_meta/eval_meta.json'
    output_root_dir = root_dir+valid_dir
    # meta_data = {'filename': {}, 'filename_T': {}, 'gt': {}, 'feasible': {}, 'infeasible':{}}
    meta_data = {}
    filenames_str = ';'.join(filenames)
    long_string = filenames_str
    ff = '/'.join(output_dir.split('/')[:-1])
    if not os.path.exists(ff):
        for data in tqdm(loader):
            # meta_data['filename'] = {'gt': , 'pos': [], 'neg':[]}
            meta_data[data['file_name'][0]] = {
                                    'gt1': data['act'][0][0].item(), 'pos1': list(data['positive_acts'][0].numpy()), 'neg1': list(data['contrastive_acts'][0].numpy()),
                                    'gt2': data['act'][0][1].item(), 'pos2': list(data['positive_acts2'][0].numpy()), 'neg2': list(data['contrastive_acts2'][0].numpy()),
                                    'filename_T': data['file_name_2'][0],
                                    }
        
        meta_data2 = convert_to_serializable(meta_data)

    # ff = '/'.join(output_dir.split('/')[:-1])
    # if os.path.exists(ff):
    #     shutil.rmtree(ff)
        os.makedirs(ff, exist_ok=True)
        with open(output_dir, 'w') as file:
            json.dump(meta_data2, file, indent=4)


    with open(output_dir, "r") as f:
        meta_data = f.read()
    # Convert JSON string back to Python dictionary
    meta_data = json.loads(meta_data)

    # checking files of valid instruction detected for both interactive agents
    files_with_2_agents = []
    for k,v in tqdm(meta_data.items()):
        if len(v['filename_T'])>1 and v['gt1']!=-1 and v['gt2']!=-1:
            if k not in files_with_2_agents:
                files_with_2_agents.append(k)
    print(f"Files with two valid agents ground truth instructions: {len(files_with_2_agents)}/{len(meta_data)}")



    print('***** processing data splits *****')
    meta_data_2agent = {k:v for k,v in meta_data.items() if k in files_with_2_agents}
    # desired_per_label_count = [800, 800, 800, 800, 64, 800]
    meta_gt1, meta_pos1, meta_neg1 = {}, {}, {}
    meta_gt2, meta_pos2, meta_neg2 = {}, {}, {}
    meta_pos12, meta_neg12 = {}, {}

    desired_per_label_count = [400, 400, 400, 400, 400, 400, 64, 400]
    for desired_label, desired_count in tqdm(enumerate(desired_per_label_count)):
        c_gt1, c_pos1, c_neg1 = 0, 0, 0
        c_gt2, c_pos2, c_neg2 = 0, 0, 0
        c_pos12, c_neg12 = 0, 0
        for k,v in tqdm(meta_data_2agent.items()):
            save_name = k+f"_act{desired_label}"
            if v['gt1']==desired_label and c_gt1<desired_count and save_name not in meta_gt1.keys():
                meta_gt1[save_name] = {'filename_T': v['filename_T'], 'gt1': desired_label, 'gt2': v['gt2']}
                c_gt1+=1
            if desired_label in v['pos1'] and c_pos1<desired_count and save_name not in meta_pos1.keys():
                meta_pos1[save_name] = {'filename_T': v['filename_T'], 'pos1': desired_label, 'gt2': v['gt2']}
                c_pos1+=1
            if desired_label in v['neg1'] and c_neg1<desired_count and save_name not in meta_neg1.keys():
                meta_neg1[save_name] = {'filename_T': v['filename_T'], 'neg1': desired_label, 'gt2': v['gt2']}
                c_neg1+=1
            if v['gt2']==desired_label and c_gt2<desired_count and save_name not in meta_gt2.keys():
                meta_gt2[save_name] = {'filename_T': v['filename_T'], 'gt1': v['gt1'], 'gt2': desired_label}
                c_gt2+=1
            if desired_label in v['pos2'] and c_pos2<desired_count and save_name not in meta_pos2.keys():
                meta_pos2[save_name] = {'filename_T': v['filename_T'], 'pos2': desired_label, 'gt1': v['gt1']}
                c_pos2+=1
            if desired_label in v['neg2'] and c_neg2<desired_count and save_name not in meta_neg2.keys():
                meta_neg2[save_name] = {'filename_T': v['filename_T'], 'neg2': desired_label, 'gt1': v['gt1']}
                c_neg2+=1
            
            # two agent, ++, --
            if desired_label in v['pos1'] and len(v['pos2'])>0 and c_pos12<desired_count:
                if len(v['pos2'])==1:
                    desired_label2 = v['pos2'][0]
                else:
                    desired_label2 = v['pos2'][1]
                meta_pos12[save_name] = {'filename_T': v['filename_T'], 'pos1': desired_label, 'pos2': desired_label2}
                c_pos12+=1
            if desired_label in v['neg1'] and len(v['neg2'])>0 and c_neg12<desired_count:
                if len(v['neg2'])==1:
                    desired_label2 = v['neg2'][0]
                else:
                    desired_label2 = v['neg2'][1]
                meta_neg12[save_name] = {'filename_T': v['filename_T'], 'neg1': desired_label, 'neg2': desired_label2}
                c_neg12+=1


            



    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_gt1.json", 'w') as file:
        json.dump(meta_gt1, file, indent=4)
    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_gt2.json", 'w') as file:
        json.dump(meta_gt2, file, indent=4)
    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_pos1.json", 'w') as file:
        json.dump(meta_pos1, file, indent=4)
    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_pos2.json", 'w') as file:
        json.dump(meta_pos2, file, indent=4)
    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_neg1.json", 'w') as file:
        json.dump(meta_neg1, file, indent=4)
    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_neg2.json", 'w') as file:
        json.dump(meta_neg2, file, indent=4)

    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_pos12.json", 'w') as file:
        json.dump(meta_pos12, file, indent=4)
    with open('/'.join(output_dir.split('/')[:-1])+f"/meta_neg12.json", 'w') as file:
        json.dump(meta_neg12, file, indent=4)