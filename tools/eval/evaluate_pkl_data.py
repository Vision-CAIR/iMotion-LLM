import pickle
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "mtr") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "mtr"))
import csv
import argparse
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from gameformer.utils.inter_pred_utils import *
from gameformer.utils.data_utils import *
import time
from scipy import stats
import csv
from tqdm import tqdm
import os
from instructions.extract_instructions import futureNavigation
from instructions.direction_instructions import DirectionClassifier

from chatgpt_instruct_v02 import generate_template_json
from evaluate_pkl_data_utils import get_act_util




# Function to append or replace an entry in the CSV
def append_or_replace_entry(file_path, new_entry, key_column):
    # Read the existing data if the file exists
    if len(new_entry)==1:
        new_entry = new_entry[0]
    rows = []
    fieldnames = new_entry.keys()  # Dynamically determine fieldnames from the new entry
    if os.path.exists(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            if rows:
                fieldnames = rows[0].keys()  # Ensure fieldnames match existing CSV

    # Check if the entry already exists (based on the key_column)
    entry_found = False
    for i, row in enumerate(rows):
        if row[key_column] == new_entry[key_column]:  # Replace entry with the same key
            rows[i] = new_entry
            entry_found = True
            break

    if not entry_found:
        rows.append(new_entry)  # Append new entry if not found

    # Sort rows alphabetically by the key_column (Directory)
    rows.sort(key=lambda x: x[key_column])

    # Write the updated data back to the file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def get_agent_caption(history, future):
        navigation_extractor = futureNavigation()
        # normalizing the view with respect to the agent, to cacluate directional feature. Irrespective of other agents, so using the location information with the normalized version is not correct
        history_normalized_view = history.copy()
        # history_ = history[history[:,0]!=0]
        # history[history[:,0]!=0].copy()[0,:2], history[history[:,0]!=0].copy()[0,2]
        # history.copy()[0,:2], history.copy()[0,2]
        # agent_center, agent_angle = history.copy()[0,:2], history.copy()[0,2]
        # valid_mask = history[:,0]!=0
        
        # print(history[:,:2])
        # history_normalized_view[valid_mask,:5] = agent_norm(history.copy()[valid_mask,:], agent_center, agent_angle, impute=True) # with respect to itself
        history_instructs = navigation_extractor.get_navigation_dict_history(torch.tensor(history_normalized_view[:,:5]))

        if future is not None:
            future_normalized_view = future.copy()
            # agent_center, agent_angle = future.copy()[0,:2], future.copy()[0,2]
            # valid_mask = future[:,0]!=0
            # future_normalized_view[valid_mask,:] = agent_norm(future.copy(), agent_center, agent_angle, impute=False) # with respect to itself
            # future_normalized_view[valid_mask,:5] = agent_norm(future.copy()[valid_mask,:], agent_center, agent_angle, impute=True) # with respect to itself
            future_instructs = navigation_extractor.get_navigation_dict(torch.tensor(future_normalized_view))
        else:
            return history_instructs
        
        return {
            **history_instructs,
            **future_instructs
            }

def gen_instruct_caption(history_ego, future_ego, navigation_extractor=None):
    ## TODO: This function act as a tempelate to implement any rule based captioning, in this code I am only using the ego information
    traj = history_ego.copy()
    # future_traj = future_ego
    future_traj_ = future_ego.copy() # correction
    if len(traj)==0 or len(future_traj_)==0:
        return {}
    
    alphabets = [i+1 for i in range(50)]
    alphabets_idx = [i for i in range(len(alphabets))]

    subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int) # subsample to sample/second
    future_traj_subsampled = future_traj_[subsample_indices]
    
    agent_type = 1 # agent_type = [1:car, 2:pedestrian, 3:cylcist], if 0 then undefined
    # if agent_type == 1: # Only if car
    agent_type_str_map = ['Unknown', 'Vehicle', 'Pedestrian', 'Cyclist']
    if agent_type != 0:
        state_features = {
            'Agent-1 speed': str(np.linalg.norm(traj[-1,3:5]).round(2)),
            'Agent-1 heading angle in radians': str(traj[-1,2].round(2)),
            'Agent-1 location': "(0.00, 0.00)",
            # 'Agent-1 type': agent_type_str_map[agent_type],
            }
        agent_name = '1'
        ego_num = 0
        inter_num = 1
        instructs_ego = navigation_extractor.get_navigation_dict(torch.tensor(future_traj_))
        
        dict2 = {f"Agent-1 {k}":v for k,v in instructs_ego.items()}
        
        
        agents_types_dict = {
            'Agent-1 type': agent_type_str_map[agent_type],
        }
        caption_json = {
            **state_features,
            **dict2,
            **agents_types_dict,
        }
        return caption_json

def mode_diversity_metric(acts):
    ## mode class diversity
    num_modalities = len(acts)
    mode_, count_ = stats.mode(acts, keepdims=False)
    diversity = (num_modalities - count_)/(num_modalities-1)
    return diversity, count_

def categorical_diversity_metric(acts):
    # calculates how many modalities are in the mode direction

    num_modalities = len(acts)
    per_categ_count = {i+1:0 for i in range(num_modalities)}
    mode_, count_ = stats.mode(acts, keepdims=False)
    for i in per_categ_count.keys():
        per_categ_count[i] = 1 if count_==i else 0 # count_ ==  num_modalities, means all modalities in the same direction
    return per_categ_count 


print_all_waymo_metrics = False

# models_names_and_description = [
#     # 'GameFormer',
#     # 'GameFormer + FullMap',
#     # 'C-GameFormer',
#     # 'C-GameFormer + FullMap',
#     # 'iMotionLLM + C-GameFormer',
#     # 'iMotionLLM + C-GameFormer + FullMap',
#     'iMotionLLM + C-GameFormer + FullMap + Short Instruct only (overall direction only)',
# ]
# data_dirs = [
#     # '<internal_model_root>/gameformer/gf_7may_base_smalldata/validation_11may_fullmap_val_splits/results',
#     # '<internal_model_root>/gameformer/gf_7may_fullmap_smalldata/validation_11may_fullmap_val_splits/results',
#     # '<internal_model_root>/gameformer/gf_7may_act_smalldata/validation_11may_fullmap_val_splits/results',
#     # '<internal_model_root>/gameformer/gf_7may_fullmap_act_smalldata/validation_11may_fullmap_val_splits/results',
#     # '<internal_model_root>/imotion/gf_13may_act_imotion/validation_11may_fullmap_val_splits/eval90/data/',
#     # '<internal_model_root>/imotion/gf_13may_fmact_imotion/validation_11may_fullmap_val_splits/eval90/data/',
#     '<internal_model_root>/imotion/gf_15may_fmact_03imotion_shortInstruct/validation_11may_fullmap_val_splits/eval90/data/',
#     # No data: '<internal_model_root>/imotion/gf_15may_act_04imotion_shortInstruct/validation_11may_fullmap_val_splits/eval90/data/',
# ]
# root_data_dir = '<internal_dataset_root>/waymo/gameformer/validation_11may_fullmap_val'

# models_names_and_description = [
#     # 'GameFormer',
#     # 'GameFormer_FullMap',
    
#     # 'CGameFormer_GT',
    
#     # 'CGameFormer_C',
    

#     # 'CGameFormer_FullMap_GT',
#     # 'CGameFormer_FullMap_P',
#     # 'CGameFormer_FullMap_C',
    

#     'iMotionLLM_t3_GT',
#     # 'iMotionLLM1_t3_P',
#     # 'iMotionLLM_t3_C',
    

#     'iMotionLLM1(t3) + C-GameFormer + FullMap + template3 || GT',
#     # 'iMotionLLM1(t3) + C-GameFormer + FullMap + template3 || Contrastive',
#     # 'iMotionLLM1(t3) + C-GameFormer + FullMap + template3 || Positive not GT',

#     'iMotionLLM(t4) + C-GameFormer + FullMap + template4 || GT',
#     # 'iMotionLLM(t4) + C-GameFormer + FullMap + template3 || Contrastive',
#     # 'iMotionLLM(t4) + C-GameFormer + FullMap + template3 || Positive not GT',

#     # 'iMotionLLM* (negative & random examples) + t3 + fm (epoch87)|| GT', #18
#     # 'iMotionLLM* (negative & random examples) + t3 + fm (epoch87)|| p', #19
#     # 'iMotionLLM* (negative & random examples) + t3 + fm (epoch87)|| c', #20
    
#     # 'iMotionLLM* (negative & random examples) + t3 + fm || GT', #21
#     # 'iMotionLLM* (negative & random examples) + t3 + fm || c', #22
#     # 'iMotionLLM* (negative & random examples) + t3 + fm || p', #23

#     # 'iMotionLLM (negative & random examples) + t3 + fm || GT', #21
#     # 'iMotionLLM (negative & random examples) + t3 + fm || c', #22
#     # 'iMotionLLM (negative & random examples) + t3 + fm || p', #23

#     # 'iMotionLLM(negative & random examples) + t3 + fm || GT', #
#     # 'iMotionLLM(negative & random examples) + t3 + fm || c', #
#     # 'iMotionLLM(negative & random examples) + t3 + fm || p', #

#     # 'iMotionLLM(negative & random examples) + t3 || GT', #
#     # 'iMotionLLM(negative & random examples) + t3 || c', #
#     # 'iMotionLLM(negative & random examples) + t3 || p', #
# ]



data_dirs = [
    # '<internal_model_root>/gameformer/gf_7may_base_smalldata/wPlausbility_val_gt_s1/results', #1
    # '<internal_model_root>/gameformer/gf_7may_fullmap_smalldata/wPlausbility_val_gt_s1/results', #2

    # '<internal_model_root>/gameformer/gf_7may_act_smalldata/wPlausbility_val_gt_s1/results', #3
    # '<internal_model_root>/gameformer/gf_7may_act_smalldata/wPlausbility_val_c_s1/results', #4
    # '<internal_model_root>/gameformer/gf_7may_act_smalldata/wPlausbility_val_p_s1/results', #5

    # '<internal_model_root>/gameformer/gf_7may_fullmap_act_smalldata/wPlausbility_val_gt_s1/results', #6
    # '<internal_model_root>/gameformer/gf_7may_fullmap_act_smalldata/wPlausbility_val_c_s1/results', #7
    # '<internal_model_root>/gameformer/gf_7may_fullmap_act_smalldata/wPlausbility_val_p_s1/results', #8

    # '<internal_model_root>/imotion/gf_15may_fmact_03imotion_shortInstruct/wPlausbility_val_gt_s1/eval90/data', #9
    # '<internal_model_root>/imotion/gf_15may_fmact_03imotion_shortInstruct/wPlausbility_val_contrastive_s1/eval90/data', #10
    # '<internal_model_root>/imotion/gf_15may_fmact_03imotion_shortInstruct/wPlausbility_val_pos_s1/eval90/data', #11

    # '<internal_model_root>/imotion/gf_15may_act_04imotion_shortInstruct/wPlausbility_val_gt_s1/eval90/data', #12
    # '<internal_model_root>/imotion/gf_15may_act_04imotion_shortInstruct/wPlausbility_val_contrastive_s1/eval90/data', #13
    # '<internal_model_root>/imotion/gf_15may_act_04imotion_shortInstruct/wPlausbility_val_pos_s1/eval90/data', #14

    # '<internal_model_root>/imotion/m01_gf_13may_act_imotion/wPlausbility_val_gt_shortInstruct_s1/eval90/data',
    # '<internal_model_root>/imotion/m01_gf_13may_act_imotion/wPlausbility_val_gt_template4_s1/eval90/data',
    # '<internal_model_root>/imotion/m01_gf_13may_act_imotion/wPlausbility_val_p_shortInstruct_s1/eval90/data',
    # '<internal_model_root>/imotion/m01_gf_13may_act_imotion/wPlausbility_val_c_shortInstruct_s1/eval90/data',

    # '<internal_model_root>/imotion/m02_gf_13may_fmact_imotion/wPlausbility_val_gt_shortInstruct_s1/eval90/data',
    # '<internal_model_root>/imotion/m02_gf_13may_fmact_imotion/wPlausbility_val_gt_template4_s1/eval90/data',
    # '<internal_model_root>/imotion/m02_gf_13may_fmact_imotion/wPlausbility_val_p_shortInstruct_s1/eval90/data',
    # '<internal_model_root>/imotion/m02_gf_13may_fmact_imotion/wPlausbility_val_c_shortInstruct_s1/eval90/data',

    # '<internal_model_root>/imotion/m05_imotion_fmact_contrastive_randomtemplate/wPlausbility_val_gt_template3_s1e87/eval90/data/', #18
    # '<internal_model_root>/imotion/m05_imotion_fmact_contrastive_randomtemplate/wPlausbility_val_p_template3_s1e87/eval90/data/', #19
    # '<internal_model_root>/imotion/m05_imotion_fmact_contrastive_randomtemplate/wPlausbility_val_c_template3_s1e87/eval90/data/', #20

    # '<internal_model_root>/imotion/m05_imotion_fmact_contrastive_randomtemplate/wPlausbility_val_gt_template3_s1/eval90/data/', #21
    # '<internal_model_root>/imotion/m05_imotion_fmact_contrastive_randomtemplate/wPlausbility_val_p_template3_s1/eval90/data/', #22
    # '<internal_model_root>/imotion/m05_imotion_fmact_contrastive_randomtemplate/wPlausbility_val_c_template3_s1/eval90/data/', #23

    # '<internal_model_root>/imotion/m06_imotion_act_contrastive_randomtemplate/wPlausbility_val_gt_template3_s1/eval90/data/', #23
    # '<internal_model_root>/imotion/m06_imotion_act_contrastive_randomtemplate/wPlausbility_val_p_template3_s1/eval90/data/', #23
    # '<internal_model_root>/imotion/m06_imotion_act_contrastive_randomtemplate/wPlausbility_val_c_template3_s1/eval90/data/', #23

    # '<internal_model_root>/imotion/m05_imotion_fmact_contrastive_randomtemplate/wPlausbility_val_gt_template3_s1e41/eval90/data/'

    # '<internal_model_root>/imotion/m08_base_imotion/wPlausbility_val_gt_s1/eval90/data/',

    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/wPlausbility_val_gt_template3_s1/eval90/data/',
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/wPlausbility_val_p_template3_s1/eval90/data/',
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/wPlausbility_val_c_template3_s1/eval90/data/',
    
    # '<internal_model_root>/gameformer/cgf_2jul_smalldata/validation_3jul/gt1results/',
    # '<internal_model_root>/gameformer/cgf_2jul_smalldata/validation_3jul/pos1results/',
    # '<internal_model_root>/gameformer/cgf_2jul_smalldata/validation_3jul/neg1results/',

    # '<internal_model_root>/gameformer/cgf_7jul_fulldata/validation_3jul/gt1results/',
    # '<internal_model_root>/gameformer/cgf_7jul_fulldata/validation_3jul/pos1results/',
    # '<internal_model_root>/gameformer/cgf_7jul_fulldata/validation_3jul/neg1results/',

    ## Jul
    # '<internal_model_root>/imotion/jul_m03_imotion_act_contrastive_small_smallPretrain/gt1_eval0/data',
    # '<internal_model_root>/imotion/jul_m03_imotion_act_contrastive_small_smallPretrain/pos1_eval0/data',
    # '<internal_model_root>/imotion/jul_m03_imotion_act_contrastive_small_smallPretrain/neg1_eval0/data',

    # '<internal_model_root>/imotion/jul_m04_imotion_act_contrastive_small_smallPretrain_fullmap/gt1_eval0/data',
    # '<internal_model_root>/imotion/jul_m04_imotion_act_contrastive_small_smallPretrain_fullmap/pos1_eval0/data',
    # '<internal_model_root>/imotion/jul_m04_imotion_act_contrastive_small_smallPretrain_fullmap/neg1_eval0/data',

    # '<internal_model_root>/imotion/jul_m02_imotion_act_contrastive_small_largePretrain_e14/gt1_eval0/data',
    # '<internal_model_root>/imotion/jul_m02_imotion_act_contrastive_small_largePretrain_e14/pos1_eval0/data',
    # '<internal_model_root>/imotion/jul_m02_imotion_act_contrastive_small_largePretrain_e14/neg1_eval0/data',

    # '<internal_model_root>/gameformer/cgf_2jul_smalldata/validation_3jul/gt1results',
    # '<internal_model_root>/gameformer/cgf_2jul_smalldata/validation_3jul/pos1results',
    # '<internal_model_root>/gameformer/cgf_2jul_smalldata/validation_3jul/neg1results',

    # '<internal_model_root>/gameformer/gf_2jul_smalldata/validation_3jul/gt1results',
    # '<internal_model_root>/gameformer/gf_2jul_smalldata/validation_3jul/pos1results',
    # '<internal_model_root>/gameformer/gf_2jul_smalldata/validation_3jul/neg1results',

    # '<internal_model_root>/gameformer/gf_2jul_smalldata_fullmap/validation_3jul/gt1results',
    # '<internal_model_root>/gameformer/gf_2jul_smalldata_fullmap/validation_3jul/pos1results',
    # '<internal_model_root>/gameformer/gf_2jul_smalldata_fullmap/validation_3jul/neg1results',

    # '<internal_model_root>/gameformer/gf_7jul_fulldata/validation_3jul/gt1results',
    # '<internal_model_root>/gameformer/gf_7jul_fulldata/validation_3jul/pos1results',
    # '<internal_model_root>/gameformer/gf_7jul_fulldata/validation_3jul/neg1results',

    # '<internal_model_root>/gameformer/cgf_7jul_fulldata/validation_3jul/gt1results',
    # '<internal_model_root>/gameformer/cgf_7jul_fulldata/validation_3jul/pos1results',
    # '<internal_model_root>/gameformer/cgf_7jul_fulldata/validation_3jul/neg1results',
    
    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/gt1_eval0/data',
    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/pos1_eval0/data',
    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/neg1_eval0/data',
    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/pos12_eval0/data',
    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/neg12_eval0/data',

    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/gt2_eval0/data',
    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/pos2_eval0/data',
    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/neg2_eval0/data',
    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/neg12_eval0/data',
    # '<internal_model_root>/imotion/jul17_m01_imotion_act_contrastive_shortInstruct_2agent/pos12_eval0/data',

    # scale of training

    # rebuttal
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/wPlausbility_val_gt_template3_4aug_noInstruct_s1/_eval90/data',
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct_5aug_randomDrop/wPlausbility_val_gt_template3_4aug_noInstruct_s1/_eval90/data',
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct_5aug_randomDrop/wPlausbility_val_template3_4aug_gt_s1/_eval90/data',
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct_5aug_randomDrop/wPlausbility_val_template3_4aug_neg_s1/_eval90/data',

    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/nuplan/_eval90/data',
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/nuplan_finetune/boston_gt/_eval90/data',
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/nuplan_finetune_e2e/boston_gt/_eval90/data',

    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/waymo_finetune_e2e/gt/_eval90/data',
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/waymo_finetune_e2e/p/_eval90/data',
    # '<internal_model_root>/imotion/m07_imotion_act_contrastive_shortInstruct/waymo_finetune_e2e/c/_eval90/data',

    # '<internal_model_root>/imotion/smallmodel01/pos/_eval90/data',
    # '<internal_model_root>/imotion/smallmodel01/gt/_eval90/data',
    # '<internal_model_root>/imotion/smallmodel01/neg/_eval90/data',

    # '<internal_model_root>/gameformer/gf_23aug/validation_23aug/results',
    # '<internal_model_root>/gameformer/cgf_23aug/results_validation_23aug_gt/results',
    
    # '<internal_model_root>/imotion/aug24_gf_2tokens_smallLLM/results_90e_gt1/pred_only_eval90/data',
    # '<internal_model_root>/gameformer/gf_23aug/results_validation_23aug_gt1/gt1results',
    # '<internal_model_root>/imotion/aug24_gf_4tokens_smallLLM/results_90e_gt1/pred_only_eval90/data',
    # '<internal_model_root>/imotion/aug24_gf_e2e_4tokens_smallLLM/checkpoint_80_gt1/pred_only_eval90/data',

    # sep
    # '<internal_model_root>/imotion/sep09_gf_clsHeadFinetune_4tokens_smallLLM/checkpoint_70_gt1/pred_only_eval90/data',

    # '<internal_model_root>/imotion/sep22_imotion_2tokens/checkpoint_last_gt1/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep22_imotion_2tokens/checkpoint_last_pos1/pos1_eval90/data',
    # '<internal_model_root>/imotion/sep22_imotion_2tokens/checkpoint_last_neg1/neg1_eval90/data',
    # '<internal_model_root>/imotion/sep22_imotion_2tokens/checkpoint_last_pred/pred_only_eval90/data',
    # '<internal_model_root>/imotion/imotion_2tokens_smallLLM/gt1/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_5_gt/gt1_eval90/data',

    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_5_gt/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_5_neg/neg1_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_last_gt/gt1_eval90/data',

    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_70_gt/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_70_neg/neg1_eval90/data',

    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_last_pred/pred_only_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_last_pos/pos1_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_last_gt/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_last_neg/neg1_eval90/data',

    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_last_gt_30sep/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_last_pos_30sep/pos1_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_last_neg_30sep/neg1_eval90/data',
    # '<internal_model_root>/imotion/sep26_imotion_3tokens/checkpoint_last_pred_30sep/pred_only_eval90/data',

    # '<internal_model_root>/imotion/sep30_imotion_3tokens/e45_gt/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep30_imotion_3tokens/e45_neg/neg1_eval90/data',
    # '<internal_model_root>/imotion/sep30_imotion_3tokens/e45_pos/pos1_eval90/data',

    # '<internal_model_root>/gameformer/cgf_23aug/results30sep/pos1results',
    # '<internal_model_root>/gameformer/cgf_23aug/results30sep/gt1results',
    # '<internal_model_root>/gameformer/cgf_23aug/results30sep/neg1results',

    # '<internal_model_root>/gameformer/gf_23aug/results30sep/gt1results',
    # '<internal_model_root>/gameformer/gf_23aug/results30sep/neg1results',
    # '<internal_model_root>/gameformer/gf_23aug/results30sep/pos1results',
    # '<internal_model_root>/gameformer/cgf_23aug/results/gt1results',
    # '<internal_model_root>/gameformer/cgf_23aug/results/pos1results',
    # '<internal_model_root>/gameformer/cgf_23aug/results/neg1results',

    # '<internal_model_root>/imotion/sep30_imotion_3tokens/e70_gt/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep30_imotion_3tokens/e70_pos/pos1_eval90/data',
    # '<internal_model_root>/imotion/sep30_imotion_3tokens/e70_neg/neg1_eval90/data',
    # '<internal_model_root>/imotion/sep30_imotion_3tokens/e70_pred/pred_only_eval90/data'

    # '<internal_model_root>/imotion/sep30_imotion_3tokens/train_pos_augmentation/pos1_eval90/data'

    # '<internal_model_root>/imotion/sep30_imotion_3tokens_stage2/gt/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep30_imotion_3tokens_stage2/pos/pos1_eval90/data',
    # '<internal_model_root>/imotion/sep30_imotion_3tokens_stage2/neg/neg1_eval90/data',
    # '<internal_model_root>/imotion/sep30_imotion_3tokens_stage2/pred/pred_only_eval90/data'

    # '<internal_model_root>/imotion/sep30_imotion_3tokens_stage2_02/gt/gt1_eval90/data',
    # '<internal_model_root>/imotion/sep30_imotion_3tokens_stage2_02/pos/pos1_eval90/data'
    # '<internal_model_root>/imotion/sep30_imotion_3tokens_stage2_02/neg/neg1_eval90/data',
    # '<internal_model_root>/imotion/oct12_imotion_3tokens_nuplan_boston_02/eval_boston_gt/gt1_eval90/data',

    # '<internal_model_root>/imotion/oct12_imotion_3tokens_nuplan_pittsburgh_02/eval_boston_gt/gt1_eval90/data',
    # '<internal_model_root>/imotion/oct12_imotion_3tokens_nuplan_pittsburgh_02/eval_boston_neg/neg1_eval90/data',
    # '<internal_model_root>/imotion/oct12_imotion_3tokens_nuplan_boston_02/eval_boston_gt/gt1_eval90/data',
    # '<internal_model_root>/imotion/oct12_imotion_3tokens_nuplan_boston_02/eval_boston_neg/neg1_eval90/data',

    ## 22 Nov
    # '<internal_model_root>/imotion/nov/imotion_3tokens_e2e_21nov_01/e20_gt/gt1_eval20/data',
    # '<internal_model_root>/imotion/nov/imotion_6qtokens_gf_21nov_01/gt1_eval20/data',

    # '<internal_model_root>/imotion/nov/imotion_1qtoken_gf_21nov_01/eval/gt1_eval20/data',
    # '<internal_model_root>/imotion/nov/imotion_3tokens_21nov_01/eval/gt1_eval20/data',
    # '<internal_model_root>/imotion/nov/imotion_6qtokens_21nov_01/eval/gt1_eval20/data',
    # '<internal_model_root>/imotion/nov/imotion_1qtokens_21nov_01/eval/gt1_eval20/data',
    # '<internal_model_root>/imotion/nov/imotion_3tokens_e2e_21nov_01/eval/gt1_eval20/data',
    # '<internal_model_root>/imotion/nov/imotion_6qtokens_gf_21nov_01/eval/gt1_eval20/data',
    # '<internal_model_root>/imotion/nov/imotion_1qtokens_21nov_01/eval/gt1_eval16/data',
    # '<internal_model_root>/imotion/nov/imotion_1qtokens_21nov_01/eval/gt1_eval11/data',
    # '<internal_model_root>/imotion/nov/imotion_1qtokens_21nov_01/eval/gt1_eval6/data',
    # '<internal_model_root>/imotion/nov/imotion_1qtokens_21nov_01/eval/gt1_eval1/data',

    # '<internal_model_root>/gameformer/gf_23aug/results_fullData_gt/gt1results',
    # '<internal_model_root>/gameformer/cgf_23aug/results_fullData_gt/gt1results',
    
    # '<internal_model_root>/imotion/nov/imotion_1qtokens_24nov_e2e/eval/gt1_eval6/data',
    # '<internal_model_root>/imotion/nov/imotion_1qtokens_24nov_OnlyLora/eval/gt1_eval6/data',
    # '<internal_model_root>/imotion/nov/imotion_1qtokens_24nov_onlyEnc/eval/gt1_eval6/data',
    # '<internal_model_root>/imotion/nov/imotion_1qtokens_21nov_01/eval/gt1_eval6/data',

    # '<internal_model_root>/imotion/nov/imotion_6qtokens_21nov_01/eval/gt1_eval6/data',
    # '<internal_model_root>/imotion/nov/imotion_6qtokens_gf_21nov_01/eval/gt1_eval6/data',

    ## DEC
    # '<internal_model_root>/gameformer/gf_1a_29nov_newData/validation_30sep/gt1results',
    # '<internal_model_root>/gameformer/cgf_1a_29nov_newData_kvact/validation_30sep/gt1results',
    # '<internal_model_root>/gameformer/cgf_1a_29nov_newData/validation_30sep/gt1results',
    # '<internal_model_root>/gameformer/cgf_1a_29nov_newData_alldata_randDrop/validation_30sep/gt1results',
    # '<internal_model_root>/gameformer/cgf_1a_29nov_newData_randDrop/validation_30sep/gt1results',

    # '<internal_model_root>/imotion/dec/qtoken_latefusion/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/qtoken_latefusion/eval/gt1_eval4/data',
    # '<internal_model_root>/imotion/dec/qtoken_latefusion/eval/gt1_eval3/data',
    # '<internal_model_root>/imotion/dec/qtoken_latefusion/eval/gt1_eval2/data',
    # '<internal_model_root>/imotion/dec/qtoken_latefusion/eval/gt1_eval1/data',
    # '<internal_model_root>/imotion/dec/qtoken_latefusion/eval/gt1_eval0/data'

    # '<internal_model_root>/imotion/dec/kvtoken_6dec/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/kvtoken_6dec/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/kvtoken_6dec/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/kvtoken_6dec/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/kvtoken_6dec/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/qtoken_LoRAonly_6dec/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/qtoken_latefusion_6dec/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/qtoken_6dec/eval/gt1_eval5/data',

    # '<internal_model_root>/imotion/dec/kvtoken_6dec/eval/gt1_eval4/data',
    # '<internal_model_root>/imotion/dec/kvtoken_6dec/eval/gt1_eval3/data',
    # '<internal_model_root>/imotion/dec/kvtoken_6dec/eval/gt1_eval2/data',
    # '<internal_model_root>/imotion/dec/kvtoken_6dec/eval/gt1_eval1/data',

    # '<internal_model_root>/imotion/dec/kvtoken_captionPos/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/kvtoken_caption/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/kvtoken_caption/eval/neg1_eval5/data',
    # '<internal_model_root>/imotion/dec/kvtoken_caption/eval/pos1_eval5/data',

    # '<internal_model_root>/imotion/dec/kvtoken_caption_30epochs/eval/gt1_eval30/data',
    # '<internal_model_root>/imotion/dec/kvtoken_caption_30epochs/eval/neg1_eval30/data',
    # '<internal_model_root>/imotion/dec/kvtoken_caption_30epochs/eval/pos1_eval30/data',

    # '<internal_model_root>/imotion/dec/kvtoken_caption_30epochs/eval/gt1_eval5/data',
    # '<internal_model_root>/imotion/dec/kvtoken_caption_30epochs/eval/neg1_eval5/data',
    # '<internal_model_root>/imotion/dec/kvtoken_caption_30epochs/eval/pos1_eval5/data',

    # '<internal_model_root>/imotion/dec/template_26dec/gt1_eval17/data',
    # '<internal_model_root>/imotion/dec/template_26dec/gt1_eval2/data',
    # '<internal_model_root>/imotion/dec/template_26dec/gt1_eval1/data',
    # '<internal_model_root>/imotion/dec/template_27dec_alwyasGT/gt1_eval2/data',
    # '<internal_model_root>/imotion/dec/template_31dec/eval/gt1_eval80/data',
    # '<internal_model_root>/imotion/dec/template_31dec/eval/gt1_eval2/data',
    # '<internal_model_root>/imotion/dec/template_31dec/eval/gt1_eval11/data',
    
    # '<internal_experiment_root>/imotion/checkpoint-5000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion/checkpoint-10000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion/checkpoint-5000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_gt/checkpoint-10000/result/gt1_eval0/data'
    # '<internal_experiment_root>/imotion_03_temp/checkpoint-302/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_03/checkpoint-1000_old_/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_03/checkpoint-1000_old/result/gt1_eval0/data'

    # '<internal_model_root>/gameformer/cgf_1a_29nov_newData/validation_30sep/pos1results',
    # '<internal_model_root>/gameformer/cgf_1a_29nov_newData/validation_30sep/neg1results',
    # '<internal_model_root>/gameformer/cgf_1a_29nov_newData/validation_30sep/gt1results',

    # '<internal_experiment_root>/imotion_03/checkpoint-2000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_03/checkpoint-2000/result/neg1_eval0/data',
    # '<internal_experiment_root>/imotion_03/checkpoint-2000/result/pos1_eval0/data',

    # '<internal_experiment_root>/imotion_03_e2e/checkpoint-2000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_03_e2e/checkpoint-2000/result/pos1_eval0/data',
    # '<internal_experiment_root>/imotion_03_e2e/checkpoint-2000/result/neg1_eval0/data',
    
    # '<internal_experiment_root>/imotion_03/checkpoint-4000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_03_e2e/checkpoint-4000/result/gt1_eval0/data',

    # '<internal_experiment_root>/imotion_03/checkpoint-1000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_03_e2e/checkpoint-1000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_03/checkpoint-1000/result/pos1_eval0/data',
    # '<internal_experiment_root>/imotion_03_e2e/checkpoint-1000/result/pos1_eval0/data',
    # '<internal_experiment_root>/imotion_03_e2e/checkpoint-1000/result/neg1_eval0/data',
    
    # '<internal_experiment_root>/imotion_04/checkpoint-1000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_04/checkpoint-1000/result/neg1_eval0/data',
    # '<internal_experiment_root>/imotion_04/checkpoint-1000/result/pos1_eval0/data',

    # '<internal_experiment_root>/imotion_04/checkpoint-2000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_04/checkpoint-2000/result/neg1_eval0/data',
    # '<internal_experiment_root>/imotion_04/checkpoint-2000/result/pos1_eval0/data',

    # '<internal_experiment_root>/imotion_05/checkpoint-500/result/gt1_eval0/data'
    # '<internal_experiment_root>/imotion_05_02/checkpoint-1500/result/gt1_eval0/data'
    # '<internal_experiment_root>/imotion_05/checkpoint-1000/result/gt1_eval0/data'

    # '<internal_experiment_root>/imotion_06/checkpoint-500/result/gt1_eval0/data'
    # '<internal_experiment_root>/imotion_06_dec/checkpoint-3000/result/gt1_eval0/data'
    # '<internal_experiment_root>/imotion_06/checkpoint-3500/result/gt1_eval0/data'
    # '<internal_experiment_root>/imotion_06_dec/checkpoint-7500/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_06_dec/checkpoint-500/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_06_dec/checkpoint-3000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion_06_dec_lowlr_highnorm/checkpoint-500/result/gt1_eval0/data',

    # '<internal_model_root>/gameformer/gf_1a_29nov_newData/validation_30sep/gt1results',
    # '<internal_model_root>/gameformer/cgf_1a_29nov_newData/validation_30sep/gt1results',
    # '<internal_experiment_root>/imotion08_r8/checkpoint-1000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8/checkpoint-500/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8/checkpoint-3500/result/gt1_eval0/data'
    # '<internal_experiment_root>/imotion08_r8/checkpoint-3500/result/pos1_eval0/data'
    # '<internal_experiment_root>/imotion08_r8_02/checkpoint-1000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_02_cls/checkpoint-1000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8/checkpoint-3000/result/gt1_eval0/data'
    # '<internal_experiment_root>/imotion08_r8_02_cls/checkpoint-3000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_02/checkpoint-3000/result/gt1_eval0/data'

    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-2000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_cls/checkpoint-2000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_03/checkpoint-2000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-5000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_cls/checkpoint-3000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8/checkpoint-3000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-3000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-14128/result/gt1_eval0/data',

    # '<internal_experiment_root>/imotion08_r8_cls/checkpoint-14128/result/gt1_eval0/data',

    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-14128/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-14128/result/neg1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-14128/result/pos1_eval0/data',
    
    # '<internal_experiment_root>/imotion08_r8_noDrop_cls/checkpoint-14128/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_cls/checkpoint-14128/result/neg1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_cls/checkpoint-14128/result/pos1_eval0/data',

    # '<internal_experiment_root>/imotion08_r8_noDrop_dec/checkpoint-14128/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_dec/checkpoint-14128/result/neg1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_dec/checkpoint-14128/result/pos1_eval0/data',

    # '<internal_experiment_root>/imotion08_r32_noDrop/checkpoint-14128/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r32_noDrop/checkpoint-14128/result/neg1_eval0/data',
    # '<internal_experiment_root>/imotion08_r32_noDrop/checkpoint-14128/result/pos1_eval0/data',
    
    # '<internal_experiment_root>/imotion08_r32_noDrop_cls/checkpoint-14128/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r32_noDrop_cls/checkpoint-14128/result/neg1_eval0/data',
    # '<internal_experiment_root>/imotion08_r32_noDrop_cls/checkpoint-14128/result/pos1_eval0/data',

    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-8000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-8000/result/pos1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop/checkpoint-8000/result/neg1_eval0/data',

    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-3000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-3000/result/pos1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-3000/result/neg1_eval0/data',

    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-10000/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-10000/result/pos1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-10000/result/neg1_eval0/data',
    
    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-14128/result/gt1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-14128/result/pos1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-14128/result/neg1_eval0/data',
    # '<internal_experiment_root>/imotion08_r8_noDrop_synth/checkpoint-14128/result/gt1_eval0/data',

    ### 21 feb 2025
    # '<internal_model_root>/gameformer/feb_18_2025/cgf_l1/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf_l1/validation/neg1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf_l1/validation/pos1results',

    # '<internal_model_root>/gameformer/feb_18_2025/cgf/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf/validation/neg1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf/validation/pos1results',

    # '<internal_model_root>/gameformer/feb_18_2025/gf/validation_30sep/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/gf_l1/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/gf/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf_l1/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf_l1/validation/neg1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf/validation/neg1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf_l1/validation/pos1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf/validation/pos1results',

    # '<internal_model_root>/imotion/feb_2025/imotion/checkpoint-17935/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion/checkpoint-17935/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion/checkpoint-17935/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_noCaption/checkpoint-17935/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e/checkpoint-17935/result/gt1_eval0/data',

    # '<internal_experiment_root>/mtr_models/cmtr_1mar/results/mtr+20_percent_data_act/default/eval/epoch_30/default/gt1/data'
    # '<internal_experiment_root>/mtr_models/mtr_1mar/results/mtr+20_percent_data/default/eval/epoch_30/default/gt1/data'

    

    # '<internal_model_root>/gameformer/feb_18_2025/gf/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/gf_l1/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf_l1/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf/validation/pos1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf_l1/validation/pos1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf/validation/neg1results',
    # '<internal_model_root>/gameformer/feb_18_2025/cgf_l1/validation/neg1results',
    ## DONE
    # '<internal_model_root>/imotion/feb_2025/imotion/checkpoint-5000/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e/checkpoint-5000/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_3epochs/checkpoint-5000/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_largeNorm/checkpoint-5000/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_loraDropout/checkpoint-5000/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e/checkpoint-17935/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion/checkpoint-17935/result/gt1_eval0/data',

    # '<internal_experiment_root>/complex/r8/checkpoint-3600/result/gt1_eval0/data'

    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_q/checkpoint-5000/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_q/checkpoint-10000/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_loraDropout/checkpoint-20000/result/gt1_eval0/data'

    # '<internal_model_root>/imotion/feb_2025/imotion/checkpoint-17935/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_02_q/checkpoint-26902/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_02/checkpoint-26902/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e/checkpoint-17935/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_3epochs/checkpoint-25000/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_largeNorm/checkpoint-17935/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_loraDropout/checkpoint-26902/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_q/checkpoint-26902/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_noCaption/checkpoint-17935/result/gt1_eval0/data',

    # '<internal_model_root>/imotion/feb_2025/imotion/checkpoint-17935/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_02_q/checkpoint-26902/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_02/checkpoint-26902/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e/checkpoint-17935/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_3epochs/checkpoint-25000/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_largeNorm/checkpoint-17935/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_loraDropout/checkpoint-26902/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_q/checkpoint-26902/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_noCaption/checkpoint-17935/pos1_eval0/data',

    # '<internal_model_root>/imotion/feb_2025/imotion/checkpoint-17935/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_02_q/checkpoint-26902/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_02/checkpoint-26902/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e/checkpoint-17935/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_3epochs/checkpoint-25000/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_largeNorm/checkpoint-17935/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_loraDropout/checkpoint-26902/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_q/checkpoint-26902/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_noCaption/checkpoint-17935/result/neg1_eval0/data',

    # '<internal_experiment_root>/mtr_models/mtr_4mar/results/mtr+100_percent_data/default/eval/epoch_12/default/gt1/data',
    # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_12/default/gt1/data',
    # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_12/default/pos1/data',
    # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_12/default/neg1/data',


    # '<internal_experiment_root>/complex/r8/checkpoint-7196/result/safe_no_context_eval0/data',
    # '<internal_experiment_root>/complex/r8/checkpoint-7196/result/safe_with_context_eval0/data',
    # '<internal_experiment_root>/complex/r8/checkpoint-7196/result/unsafe_no_context_eval0/data',
    # '<internal_experiment_root>/complex/r8/checkpoint-7196/result/unsafe_with_context_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_complex/checkpoint-9592/result/safe_no_context_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_complex/checkpoint-9592/result/safe_with_context_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_complex/checkpoint-9592/result/unsafe_no_context_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_complex/checkpoint-9592/result/unsafe_with_context_eval0/data',
    
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_complex_pretrained/checkpoint-9592/result/safe_no_context_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_complex_pretrained/checkpoint-9592/result/safe_with_context_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_complex_pretrained/checkpoint-9592/result/unsafe_no_context_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_complex_pretrained/checkpoint-9592/result/unsafe_with_context_eval0/data',

    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_synth/checkpoint-17935/result/gt1_eval0/data'
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_synth/checkpoint-17935/result/neg1_eval0/data'
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_synth/checkpoint-17935/result/pos1_eval0/data'

    # '<internal_experiment_root>/mtr_models/mtr_4mar/results/mtr+100_percent_data/default/eval/epoch_15/default/gt1/data',
    # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_15/default/pos1/data',
    # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_15/default/gt1/data',
    # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_15/default/neg1/data',

    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_mtr/checkpoint-7500/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_mtr/checkpoint-7500/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/feb_2025/imotion_e2e_mtr/checkpoint-7500/result/neg1_eval0/data',

    # '<internal_model_root>/gameformer/feb_18_2025/gf_l1/validation/gt1results',
    # '<internal_model_root>/gameformer/feb_18_2025/gf_l1/validation/neg1results',
    # '<internal_model_root>/gameformer/feb_18_2025/gf_l1/validation/pos1results',
    # '<internal_experiment_root>/mtr_models/mtr_4mar/results/mtr+100_percent_data/default/eval/epoch_15/default/gt1/data',
    # '<internal_experiment_root>/mtr_models/mtr_4mar/results/mtr+100_percent_data/default/eval/epoch_15/default/pos1/data',
    # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_15/default/gt1/data',
    # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_15/default/pos1/data',

    # '<internal_model_root>/imotion/mar_2025/imotion_e2e/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/imotion_e2e_cb/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/imotion_e2e_decOnly/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/imotion_e2e_llmOnly/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/imotion_e2e_lora01/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/imotion_e2e_lora02/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/imotion_e2e_lr01/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/imotion_e2e_lr02/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/imotion_e2e_simpleTxt/checkpoint-6726/result/gt1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r16a16lr1e4/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r16a16lr1e5/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r16a16lr3e5/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r32a16lr1e4/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r32a16lr1e5/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r32a16lr3e5/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r32a32lr1e4/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r32a32lr1e5/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r32a32lr3e5/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r16a32lr1e4/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r16a32lr1e5/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r16a32lr3e5/checkpoint-6726/result/gt1_eval0/data',
    
    # '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/r32a16lr1e4/checkpoint-6726/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_modeling/llama_2_7b_qOnly/checkpoint-6726/result/gt1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_contrastive/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_1b_instruct/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_3b_instruct/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/mistral_7b_instruct/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/vicuna_7b/checkpoint-6726/result/gt1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b/checkpoint-6726/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_contrastive/checkpoint-6726/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_1b_instruct/checkpoint-6726/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_3b_instruct/checkpoint-6726/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/mistral_7b_instruct/checkpoint-6726/result/pos1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/vicuna_7b/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_contrastive/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_1b_instruct/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_3b_instruct/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/mistral_7b_instruct/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/vicuna_7b/checkpoint-6726/result/neg1_eval0/data',

    # ## 
    # '<internal_model_root>/imotion/mar_2025/ablation_modeling/llama_2_7b_qOnly/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_modeling/llama_2_7b_qOnly/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_modeling/llama_2_7b_qOnly/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_contrastive/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_contrastive/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_contrastive/checkpoint-6726/result/pos1_eval0/data',

    # ## 
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_wSysPrompt/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_wSysPrompt/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_wSysPrompt/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_1b_instruct/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_1b_instruct/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_1b_instruct/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_1b_instruct_contrastive/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_1b_instruct_contrastive/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_1b_instruct_contrastive/checkpoint-6726/result/pos1_eval0/data',

    # ## 
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_3b_instruct/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_3b_instruct/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_3b_instruct/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_3b_instruct_contrastive/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_3b_instruct_contrastive/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_3_2_3b_instruct_contrastive/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/mistral_7b_instruct/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/mistral_7b_instruct/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/mistral_7b_instruct/checkpoint-6726/result/pos1_eval0/data',

    # ## 
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/mistral_7b_instruct_contrastive/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/mistral_7b_instruct_contrastive/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/mistral_7b_instruct_contrastive/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/vicuna_7b/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/vicuna_7b/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/vicuna_7b/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/vicuna_7b_contrastive/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/vicuna_7b_contrastive/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/vicuna_7b_contrastive/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_02/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_02/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/mar_2025/ablation_llms/llama_2_7b_02/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/apr_2025/ablation_llms/1_nussair_llama_2_7b_contrastive_query_only/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/1_nussair_llama_2_7b_contrastive_query_only/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/1_nussair_llama_2_7b_contrastive_query_only/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/apr_2025/ablation_llms/2_nussair_llama_2_7b_contrastive_114_kvs_query/checkpoint-13451/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/2_nussair_llama_2_7b_contrastive_114_kvs_query/checkpoint-13451/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/2_nussair_llama_2_7b_contrastive_114_kvs_query/checkpoint-13451/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/apr_2025/ablation_llms/3_nussair_llama_2_7b_contrastive_34_actors_query/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/3_nussair_llama_2_7b_contrastive_34_actors_query/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/3_nussair_llama_2_7b_contrastive_34_actors_query/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/apr_2025/ablation_llms/4_nussair_llama_2_7b_contrastive_60_lanes_query/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/4_nussair_llama_2_7b_contrastive_60_lanes_query/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/4_nussair_llama_2_7b_contrastive_60_lanes_query/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/apr_2025/ablation_llms/5_nussair_llama_2_7b_contrastive_20_crosswalks_query/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/5_nussair_llama_2_7b_contrastive_20_crosswalks_query/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/5_nussair_llama_2_7b_contrastive_20_crosswalks_query/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/apr_2025/ablation_llms/llama_2_7b_03/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/llama_2_7b_03/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/llama_2_7b_03/checkpoint-6726/result/pos1_eval0/data',

    # '<internal_model_root>/imotion/apr_2025/ablation_llms/llama_2_7b_contrastive_04/checkpoint-6726/result/gt1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/llama_2_7b_contrastive_04/checkpoint-6726/result/neg1_eval0/data',
    # '<internal_model_root>/imotion/apr_2025/ablation_llms/llama_2_7b_contrastive_04/checkpoint-6726/result/pos1_eval0/data',

    "<internal_model_root>/imotion/apr_2025/replacements/1_llama_2_7b_contrastive_34_actors_1e4_r32_a32/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/1_llama_2_7b_contrastive_34_actors_1e4_r32_a32/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/1_llama_2_7b_contrastive_34_actors_1e4_r32_a32/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/2_llama_2_7b_contrastive_34_actors_1e4_r16_a32/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/2_llama_2_7b_contrastive_34_actors_1e4_r16_a32/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/2_llama_2_7b_contrastive_34_actors_1e4_r16_a32/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/3_llama_2_7b_contrastive_34_actors_1e4_r16_a16/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/3_llama_2_7b_contrastive_34_actors_1e4_r16_a16/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/3_llama_2_7b_contrastive_34_actors_1e4_r16_a16/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/4_llama_2_7b_contrastive_34_actors_1e4_r32_a16/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/4_llama_2_7b_contrastive_34_actors_1e4_r32_a16/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/4_llama_2_7b_contrastive_34_actors_1e4_r32_a16/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/5_llama_2_7b_contrastive_34_actors_1e5_r32_a32/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/5_llama_2_7b_contrastive_34_actors_1e5_r32_a32/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/5_llama_2_7b_contrastive_34_actors_1e5_r32_a32/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/6_llama_2_7b_contrastive_34_actors_1e5_r16_a32/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/6_llama_2_7b_contrastive_34_actors_1e5_r16_a32/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/6_llama_2_7b_contrastive_34_actors_1e5_r16_a32/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/7_llama_2_7b_contrastive_34_actors_1e5_r16_a16/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/7_llama_2_7b_contrastive_34_actors_1e5_r16_a16/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/7_llama_2_7b_contrastive_34_actors_1e5_r16_a16/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/8_llama_2_7b_contrastive_34_actors_1e5_r32_a16/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/8_llama_2_7b_contrastive_34_actors_1e5_r32_a16/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/8_llama_2_7b_contrastive_34_actors_1e5_r32_a16/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/9_llama_2_7b_contrastive_34_actors_3e5_r32_a32/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/9_llama_2_7b_contrastive_34_actors_3e5_r32_a32/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/9_llama_2_7b_contrastive_34_actors_3e5_r32_a32/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/10_llama_2_7b_contrastive_34_actors_3e5_r16_a32/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/10_llama_2_7b_contrastive_34_actors_3e5_r16_a32/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/10_llama_2_7b_contrastive_34_actors_3e5_r16_a32/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/11_llama_2_7b_contrastive_34_actors_3e5_r16_a16/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/11_llama_2_7b_contrastive_34_actors_3e5_r16_a16/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/11_llama_2_7b_contrastive_34_actors_3e5_r16_a16/checkpoint-6726/result/pos1_eval0/data",

"<internal_model_root>/imotion/apr_2025/replacements/12_llama_2_7b_contrastive_34_actors_3e5_r32_a16/checkpoint-6726/result/gt1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/12_llama_2_7b_contrastive_34_actors_3e5_r32_a16/checkpoint-6726/result/neg1_eval0/data",
"<internal_model_root>/imotion/apr_2025/replacements/12_llama_2_7b_contrastive_34_actors_3e5_r32_a16/checkpoint-6726/result/pos1_eval0/data",


]


with_classification = False

# Path to the central CSV file
# csv_file_path = '<internal_experiment_root>/all_results.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/feb_all_results_gf_only.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/imotion_e2e_ch5000_gt.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/complex.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/imotion_4mar_01_gt.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/imotion_4mar_01_gt_cls.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/imotion_4mar_01_pos.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/imotion_4mar_01_pos_cls.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/imotion_4mar_01_neg.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/imotion_4mar_01_neg_cls.csv'

# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/mtr.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/complex02_cls.csv'

# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/complex03.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/mtr_v02.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/mtr_imotion.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/gf_01.csv'
# csv_file_path = '<internal_dataset_root>/waymo/gameformer/feb16_2025/mtr_nonconditional.csv'
# csv_file_path = '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/csv/gt.csv'

# csv_file_path = '<internal_model_root>/imotion/mar_2025/ablation_lora_lr/csv/gt.csv'

# csv_file_path = '<internal_model_root>/imotion/mar_2025/ablation_modeling/results.csv'
# csv_file_path = '<internal_model_root>/imotion/mar_2025/ablation_llms/apr_6_results_acc.csv'
csv_file_path = '<internal_model_root>/imotion/apr_2025/ablation_llms/temp.csv'


ade, fde = [], []

save_augmentation_data = False
if save_augmentation_data:
    root_save_dir = '<internal_model_root>/imotion/sep30_imotion_3tokens/train_pos_augmentation/data'
    llm_root_save_dir = root_save_dir + '_templateLLM/'
    # Create the directory
    os.makedirs(root_save_dir)
    os.makedirs(llm_root_save_dir)


# root_data_dir = '<internal_dataset_root>/nuplan/data/cache/train_boston_processed'
# root_data_dir = '<internal_dataset_root>/waymo/gameformer/validation_17may_fullmap_wPlausbility_val_splits/split_1'
# root_data_dir = '<internal_dataset_root>/waymo/gameformer/validation_3jul'
# acts_dir = ''
# root_data_dir = '<internal_dataset_root>/waymo/gameformer/validation_30sep'
# acts_dir = '<internal_dataset_root>/waymo/gameformer/validation_30sep_acts'

# root_data_dir = '<internal_dataset_root>/waymo/gameformer/training_30sep'
# acts_dir = '<internal_dataset_root>/waymo/gameformer/training_30sep_acts'

# root_data_dir = '<internal_dataset_root>/waymo/gameformer/validation_23aug'
# acts_dir = '<internal_dataset_root>/waymo/gameformer/validation_23aug_acts'


# root_data_dir = '<internal_dataset_root>/waymo/gameformer/validation_30sep'
# acts_dir = '<internal_dataset_root>/waymo/gameformer/validation_30sep_acts'

# root_data_dir = '<internal_dataset_root>/waymo/gameformer/feb16_2025/validation'

if 'complex' not in csv_file_path:
    root_data_dir = '<internal_dataset_root>/waymo/gameformer/feb16_2025/validation_more_data/validation'
    acts_dir = '<internal_dataset_root>/waymo/gameformer/feb16_2025/validation_acts'
else:
    root_data_dir = '<internal_dataset_root>/nuplan/test_gpt_prompt_14types/all_types/npz'
    acts_dir = ''

num_classes=5
# num_classes=8

# root_data_dir = '<internal_dataset_root>/nuplan/test/test_boston_processed'
# acts_dir = ''

# num_classes=6


two_agent = False
agent_select = 1 # 1 or 2
eval_accepted_only = with_classification

models_names_and_description = data_dirs


#Needs to evaluate first <internal_model_root>/imotion/gf_15may_fmact_03imotion_shortInstruct/
#Needs to evaluate first <internal_model_root>/imotion/gf_15may_act_04imotion_shortInstruct/



# template_dir = '<internal_dataset_root>/waymo/gameformer/validation_16may_fullmap_wPlausbility_val_templateLLM/dba89f922ecb7de6_1_2_interest.txt'
# with open(template_dir) as f:
#     lines = f.readlines()
# templates = [json.loads(line) for line in lines]

if False:
    if str(REPO_ROOT / "mtr") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "mtr"))
    from mtr.mtr.datasets.waymo.waymo_eval import waymo_evaluation
    # Path to the pickle file
    pkl_dir = '<internal_experiment_root>/mtr_models/mtr_1mar/results/mtr+20_percent_data/default/eval/epoch_30/default/gt1/result.pkl'
    
    print(pkl_dir)
    # Load the pickle file
    with open(pkl_dir, 'rb') as f:
        pred_dicts = pickle.load(f)  # Deserialize the pickle file

    # Apply Waymo evaluation
    print(waymo_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=6)[1])

    pkl_dir = '<internal_experiment_root>/mtr_models/cmtr_1mar/results/mtr+20_percent_data_act/default/eval/epoch_30/default/gt1/result.pkl'
    print(pkl_dir)
    # Load the pickle file
    with open(pkl_dir, 'rb') as f:
        pred_dicts = pickle.load(f)  # Deserialize the pickle file

    # Apply Waymo evaluation
    print(waymo_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=6)[1])

    for pkl_dir in [
        # '<internal_experiment_root>/mtr_models/mtr_4mar/results/mtr+100_percent_data/default/eval/epoch_12/default/gt1/result.pkl',
        # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_12/default/gt1/result.pkl',
        # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_12/default/neg1/result.pkl',
        # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_12/default/pos1/result.pkl',
        # '<internal_experiment_root>/mtr_models/mtr_4mar/results/mtr+100_percent_data/default/eval/epoch_15/default/gt1/result.pkl',
        # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_15/default/gt1/result.pkl',
        # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_15/default/neg1/result.pkl',
        # '<internal_experiment_root>/mtr_models/cmtr_4mar/results/mtr+100_percent_data_act/default/eval/epoch_15/default/pos1/result.pkl',

        '<internal_model_root>/imotion/feb_2025/imotion_e2e_mtr/checkpoint-7500/result/gt1_eval0/result.pkl',
        '<internal_model_root>/imotion/feb_2025/imotion_e2e_mtr/checkpoint-7500/result/pos1_eval0/result.pkl',
        # '<internal_model_root>/imotion/feb_2025/imotion_e2e_mtr/checkpoint-7500/result/neg1_eval0/result.pkl',
        ]:
        print(pkl_dir)
        with open(pkl_dir, 'rb') as f:
            pred_dicts = pickle.load(f)  # Deserialize the pickle file

        # Apply Waymo evaluation
        print(waymo_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=6)[1])
        print('')
    

navigation_extractor = futureNavigation(normalize_track=True, num_classes=num_classes)
direction_classifier = DirectionClassifier(num_classes=num_classes)

wrong_label = 0
for data_i, data_dir in enumerate(data_dirs):
    
    # 
    modalities=6
    hits_stats = {
        'rgif':0, 'any_rgif':0, 'rgif_c':0, 'any_rgif_c':0, 'rgif_class':[0 for _ in range(num_classes)], 'rgif_class_c':[0 for _ in range(num_classes)], 
        # 'rcif':0, 'any_rcif':0, 'rcif_c':0, 'any_rcif_c':0, 'rcif_class':[0 for _ in range(num_classes)], 'rcif_class_c':[0 for _ in range(num_classes)], 
        'gdiversity':0, 'cdiversity':0, 'gdiversity_c':0, 'cdiversity_c':0,
        'gdiversity_class':[0 for _ in range(num_classes)], 'gdiversity_class_c':[0 for _ in range(num_classes)],
        # 'cdiversity_class':[0 for _ in range(num_classes)], 'cdiversity_class_c':[0 for _ in range(num_classes)],
        'g_categorical_diversity': {i+1:0 for i in range(modalities)},
        # 'c_categorical_diversity': {i+1:0 for i in range(modalities)},
        'mode_class_recall':[0 for _ in range(num_classes)], 'mode_class_recall_count':[0 for _ in range(num_classes)],
        'num_examples':0,
        'decision_accuracy':0, 'decision_accuracy_num_examples':0,
        }

    epoch_metrics = MotionMetrics()

    # List all pickle files in the data directory
    pickle_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    np_files = [f for f in os.listdir(root_data_dir) if f.endswith('.npz')]

    # Initialize metrics and statistics
    epoch_loss = []
    size = len(pickle_files)
    print(f"Processing {size} validation samples...")
    # print('###'*30)
    # Loop over all pickle files
    # for filename in tqdm(pickle_files):
    # for filename in tqdm(pickle_files):
    for filename in tqdm(pickle_files):
        file_path = os.path.join(data_dir, filename)
        
        # Load data from pickle file
        with open(file_path, 'rb') as file:
            out_data = pickle.load(file)
        if acts_dir != '' and 'act' in out_data and out_data['act'].item()==-1 and os.path.exists(acts_dir+'/'+filename):
            with open(acts_dir+'/'+filename, 'rb') as file:
                acts = pickle.load(file)
            acts = eval(acts)
            acts = [acts['act0'],acts['act1']]
            if two_agent:
                act = acts[agent_select-1]
            else:
                act = acts[0]
        else:
            if two_agent:
                act = out_data['act'][agent_select-1].item()
            else:
                act = out_data['act'].item() if 'act' in out_data else None
        
        # if act==-1:
        #     continue

        filename = file_path.split('/')[-1].replace('.pkl','')
        
        if 'act' in filename:
            filename = filename[:-5]

        if '_'.join(filename.split('_')[:-1])+'.npz' in np_files:
            filename_ = '_'.join(filename.split('_')[:-1])
            np_filename = root_data_dir+'/'+filename_+'.npz'
            np_data = np.load(np_filename, allow_pickle = True)
            traj_obs = torch.tensor(np_data['ego'][:,:,:2])
            traj_pred = torch.tensor(np_data['gt_future_states'][:,:,:2])
            traj = torch.cat((traj_obs, traj_pred), dim=1)
            object_type = (np_data['ego'][:,-1,8:].argmax(-1)+1) * np_data['ego'][:,-1,8:].sum(-1).astype(int)
            data_object_type = np_data['object_type']

        elif filename+'.npz' in np_files:
            np_filename = root_data_dir+'/'+filename+'.npz'
            np_data = np.load(np_filename, allow_pickle = True)
            traj_obs = torch.tensor(np_data['ego'][:,:,:2])
            traj_pred = torch.tensor(np_data['gt_future_states'][:,:,:2])
            traj = torch.cat((traj_obs, traj_pred), dim=1)
            object_type = (np_data['ego'][:,-1,8:].argmax(-1)+1) * np_data['ego'][:,-1,8:].sum(-1).astype(int)
            data_object_type = np_data['object_type']

            # agent_dir = root_data_dir+'_agentJsons/'+filename+'.json'
            # with open(agent_dir, 'r') as file:
            #     agent_dict = file.read()
            # agent_dict = json.loads(agent_dict)
            # act = agent_dict['Agent-1']['direction 0.1to8_cls']

        if act==-1:
            continue

        hits_stats['decision_accuracy_num_examples'] +=1
        if 'text' in out_data.keys() and with_classification:
            out_txt = out_data['text']
            if two_agent:
                if agent_select==1:
                    if 'Ego Decision' in out_txt:
                        cls_text = out_txt[out_txt.find('Ego Decision'):out_txt.find('Ego Decision')+28]
                    else:
                        cls_text = 'Accepted'
                    example_should_be_accepted = False if ('neg1' in file_path or 'neg12' in file_path) else True
                    # example_should_be_accepted = 'gt1' in file_path or 'pos1' in file_path
                else:
                    if 'Agent-2 Decision' in out_txt:
                        cls_text = out_txt[out_txt.find('Agent-2 Decision'):out_txt.find('Agent-2 Decision')+30]
                    else:
                        cls_text = 'Accepted'
                    example_should_be_accepted = False if ('neg2' in file_path or 'neg12' in file_path) else True
                    # example_should_be_accepted = 'gt2' in file_path or 'pos2' in file_path or 'pos12' in file_path
                text_accept ='Rejected' not in cls_text
            else:
                text_accept = '<Rejected>' not in out_txt and 'cannot' not in out_txt
                # example_should_be_accepted = 'gt' in file_path or '_p_' in file_path or '_pos_' in file_path
                example_should_be_accepted = 'gt' in file_path or '_p_' in file_path or 'pos' in file_path
                if 'complex' in file_path:
                    example_should_be_accepted = False if 'unsafe' in file_path else True

            
            hits_stats['decision_accuracy'] += 1 if text_accept == example_should_be_accepted else 0
            if eval_accepted_only and not text_accept:
                if act is None:
                    get_act_util_ = get_act_util()
                    act = get_act_util_.get_act(history=np_data['ego'][0,:,:5], future=np_data['gt_future_states'][0,:,:5], skip_future_start_end_with_zeros=False, not_skip_invalid_history=False)['Agent-1']['direction 0.1to8_cls']
                hits_stats['rgif_c']+=6 # number of modalities
                hits_stats['any_rgif_c']+=1
                hits_stats['rgif_class_c'][act]+=6
                hits_stats['rgif_class_c'][act]+=6
                # hits_stats[f'gdiversity_class_c'][act]+= 1
                # hits_stats['mode_class_recall_count'][act]+=1
                # hits_stats['gdiversity_c']+=1
                hits_stats['num_examples']+=1
                continue
                    
            # else:
            #     print('?')
        elif 'text' in out_data.keys():
            out_txt = out_data['text']
            text_accept = '<Rejected>' not in out_txt and 'cannot' not in out_txt
            example_should_be_accepted = 'gt' in file_path or '_p_' in file_path or 'pos' in file_path
        ## waymo evaluation
        egos = torch.tensor(out_data['modalities']).to(torch.float32).unsqueeze(0)
        egos = egos.permute(0,2,1,3,4)
        scores = torch.tensor(out_data['scores']).to(torch.float32).unsqueeze(0)
        scores = scores.sum(1)
        scores = F.softmax(scores,dim=-1)
        gt_obs = torch.tensor(np_data['ego'][:,:,:]).unsqueeze(0)
        gt_future = torch.tensor(np_data['gt_future_states'][:,:,:5]).unsqueeze(0)
        ego_ground_truth = torch.cat([gt_obs[...,:5], gt_future], dim=2)
        ego_ground_truth = torch.cat([
            ego_ground_truth[:, :, :, :2], 
            gt_obs[:,:, -1, 5:7].unsqueeze(2).expand(-1,-1, ego_ground_truth.shape[2], -1), 
            ego_ground_truth[:, :, :, 2:]
            ], dim=-1)
        object_type = (np_data['ego'][:,-1,8:].argmax(-1)+1) * np_data['ego'][:,-1,8:].sum(-1).astype(int)
        object_type = torch.tensor(object_type).unsqueeze(0)
        if 'nuplan' in data_dir:
            ego_ground_truth[0,...,-2:] = abs_distance_to_velocity(ego_ground_truth[0,:,:,:2])*10
        
        if egos.shape[2]==1: # one agent predictions only
            ego_ground_truth = ego_ground_truth[:,:1,:,:]
            object_type = object_type[:,:1]


        epoch_metrics.update_state(
            prediction_trajectory=egos, 
            prediction_score=scores,  
            ground_truth_trajectory=ego_ground_truth, 
            ground_truth_is_valid=torch.ne(ego_ground_truth, 0).bool(), 
            object_type=object_type.long(),
            # subsample=True if 'complex' in data_dir else False
            )
        if two_agent:
        #     ego_ground_truth[:,1 if agent_select==1 else 0] = 0 # making it invalid
        #     epoch_metrics.update_state(
        #         prediction_trajectory=egos, 
        #         prediction_score=scores,  
        #         ground_truth_trajectory=ego_ground_truth, 
        #         ground_truth_is_valid=torch.ne(ego_ground_truth, 0).bool(), 
        #         object_type=object_type.long()
        #         )
            # epoch_metrics.update_state(
            #         prediction_trajectory=egos, 
            #         prediction_score=scores,  
            #         ground_truth_trajectory=ego_ground_truth, 
            #         ground_truth_is_valid=torch.ne(ego_ground_truth, 0).bool(), 
            #         object_type=object_type.long()
            #         )
            ego_multimodal = torch.tensor(out_data['modalities']).to(torch.float32)[agent_select-1]
        else:
            if 'nuplan' in data_dir:
                ego_ground_truth[0,...,-2:] = abs_distance_to_velocity(ego_ground_truth[0,:,:,:2])*10
            # epoch_metrics.update_state(
            #     prediction_trajectory=egos, 
            #     prediction_score=scores,  
            #     ground_truth_trajectory=ego_ground_truth,
            #     ground_truth_is_valid=torch.ne(ego_ground_truth, 0).bool(), 
            #     object_type=object_type.long()
            #     )
            ego_multimodal = torch.tensor(out_data['modalities']).to(torch.float32)[0]
            
            # torch.ne(ego_ground_truth, 0).bool()[0,:,11:,:2]
            # valid_ade = (torch.ne(ego_ground_truth, 0).bool()[0,:,11:,0][:,[29,49,79]] * (object_type[0]==1)[:,np.newaxis])
            # egos[0,:,:,[29,49,79]]
            # ade_ = 0
            # for t_ in range(3):
            #     # print((egos[0,:,:,[29,49,79]][:,:,t_] - ego_ground_truth[:,:,[29,49,79],:2][:,:,t_]).norm(dim=-1)[:,valid_ade[:,t_]].mean(dim=-1).min().item())
            #     ade_ += (egos[0,:,:,[29,49,79]][:,:,t_] - ego_ground_truth[:,:,[29,49,79],:2][:,:,t_]).norm(dim=-1)[:,valid_ade[:,t_]].mean(dim=-1).min().item()
            # ade.append(ade_/3)
            # t_=2
            # fde_ = (egos[0,:,:,[29,49,79]][:,:,t_] - ego_ground_truth[:,:,[29,49,79],:2][:,:,t_]).norm(dim=-1)[:,valid_ade[:,t_]].mean(dim=-1).min().item()
            # fde.append(fde_)


        if act!=-1:
            # cls0_ = navigation_extractor.get_navigation_dict(track=torch.tensor(out_data['gt_trajs'][11:]))['direction 0.1to8_cls']
            # cls1_ = navigation_extractor.get_navigation_dict(track=torch.tensor(out_data['gt_trajs'][11:]))['direction 0.1to8_cls']
            # cls2_ = navigation_extractor.get_navigation_dict(track=torch.tensor(np_data['gt_future_states'][:,:,:5][0,:,:2]))['direction 0.1to8_cls']
            get_act_util_ = get_act_util()
            # get_act_util_.get_act(history=out_data['gt_trajs'][:11], future=out_data['gt_trajs'][11:])['Agent-1']['direction 0.1to8_cls']
            # try:
            if False: #enable this for debugging
                cls3_ = get_act_util_.get_act(history=np_data['ego'][0,:,:5], future=np_data['gt_future_states'][0,:,:5], skip_future_start_end_with_zeros=False, not_skip_invalid_history=False)['Agent-1']['direction 0.1to8_cls']
                if 'gt_trajs' in out_data:
                    cls3_ = get_act_util_.get_act(history=out_data['gt_trajs'][:11], future=out_data['gt_trajs'][11:], skip_future_start_end_with_zeros=False, not_skip_invalid_history=False)['Agent-1']['direction 0.1to8_cls']
                # except:
                    # cls3_ = get_act_util_.get_act(history=np_data['ego'][0,:,:5], future=np_data['gt_future_states'][0,:,:5], skip_future_start_end_with_zeros=False, not_skip_invalid_history=False)['Agent-1']['direction 0.1to8_cls']
                if cls3_ != act:
                    wrong_label+=1
            # assert cls3_ == act
            # if act!=cls0_:
            #     print('')
            # if act!=cls1_:
            #     print('')
            # if act!=cls2_:
            #     print('')
            if act is None:
                try:
                    act = get_act_util_.get_act(history=np_data['ego'][0,:,:5], future=np_data['gt_future_states'][0,:,:5], skip_future_start_end_with_zeros=False, not_skip_invalid_history=False)['Agent-1']['direction 0.1to8_cls']
                except:
                    continue # couldn't calculate act
                
            output_act = []
            for modal_i in range(ego_multimodal.shape[0]):
                instructions_out = navigation_extractor.get_navigation_dict(track=ego_multimodal[modal_i])
                # out_cls1 = instructions_out['direction 0.1to8_cls']
                if 'gt_trajs' in out_data:
                    out_cls2 = get_act_util_.get_act(history=out_data['gt_trajs'][:11], future=ego_multimodal[modal_i].numpy(), skip_future_start_end_with_zeros=False, not_skip_invalid_history=False, agent_angle=np_data['gt_future_states'][0,0,2])['Agent-1']['direction 0.1to8_cls']
                    if out_cls2 != get_act_util_.get_act(history=np_data['ego'][0,:,:5], future=ego_multimodal[modal_i].numpy(), skip_future_start_end_with_zeros=False, not_skip_invalid_history=False, agent_angle=np_data['gt_future_states'][0,0,2])['Agent-1']['direction 0.1to8_cls']:
                        print('')
                else:
                    out_cls2 = get_act_util_.get_act(history=np_data['ego'][0,:,:5], future=ego_multimodal[modal_i].numpy(), skip_future_start_end_with_zeros=False, not_skip_invalid_history=False, agent_angle=np_data['gt_future_states'][0,0,2])['Agent-1']['direction 0.1to8_cls']
                # if out_cls1!=act and act==out_cls2:
                #     print('')
                # output_act.append(out_cls1)
                output_act.append(out_cls2)
            output_act = torch.tensor(output_act)

            modal_hits = output_act == act
            
            if True in modal_hits and save_augmentation_data:
                max_hit_score = scores[0][np.where(modal_hits)[0]].max()
                selected_hit_modality_idx = np.where(scores[0]==max_hit_score)[0]
                if len(selected_hit_modality_idx)>1:
                    selected_hit_modality_idx = selected_hit_modality_idx[0].item()
                else:
                    selected_hit_modality_idx = selected_hit_modality_idx.item()
                
                pos_generated_sample = ego_multimodal[selected_hit_modality_idx]

                agent_json = {f"Agent-{1}": get_agent_caption(gt_obs[0,0].numpy().copy(), pos_generated_sample.numpy().copy())}
                llm_json = generate_template_json(agent_json, None, [], [], DirectionClassifier(step_t=1, num_classes=num_classes).classes)
                # {filename.split('/')[-1][:-4]}.txt
                templateLLM_filename = llm_root_save_dir + filename + '.txt'
                with open(templateLLM_filename, 'w') as file:
                    file.write(llm_json)
                
                np_save_dir = root_save_dir + '/' + np_filename.split('/')[-1]
                np.save(np_save_dir, pos_generated_sample.numpy())

                # agent_json = gen_instruct_caption(gt_obs[0,0].numpy().copy(), pos_generated_sample.numpy().copy(), navigation_extractor)
                # filename
                # act
                # text_accept = example_should_be_accepted
                # out_txt
                

            hits_stats['rgif']+= modal_hits.sum().item()
            hits_stats['rgif_c']+=len(modal_hits)
            hits_stats['any_rgif']+= modal_hits.any(dim=-1).sum().item()
            hits_stats['any_rgif_c']+=1
            hits_stats['rgif_class'][act]+= modal_hits.sum().item()
            hits_stats['rgif_class_c'][act]+=len(modal_hits)
            
            class_diversity = len(np.unique(output_act))
            num_modalities=6
            hits_stats[f'gdiversity_class'][act]+= (class_diversity-1)/(num_modalities-1)
            hits_stats[f'gdiversity_class_c'][act]+= 1

            hits_stats['mode_class_recall'][act]+= 1 if stats.mode(output_act, keepdims=False).mode == act else 0 # if the mode in the correct direction
            hits_stats['mode_class_recall_count'][act]+=1
            
            # diversity = (num_modalities - count_)/(num_modalities-1)
            mode_diversity, mode_diversity_count = mode_diversity_metric(output_act)
            hits_stats['gdiversity']+=mode_diversity
            hits_stats['gdiversity_c']+=1
            hits_stats['g_categorical_diversity'][mode_diversity_count]+=1
            categorical_diversity = categorical_diversity_metric(output_act)
            hits_stats['num_examples']+=1

        
        

        

        # template_dir = '<internal_dataset_root>/waymo/gameformer/validation_16may_fullmap_wPlausbility_val_templateLLM/dba89f922ecb7de6_1_2_interest.txt'
        # with open(template_dir) as f:
        #     lines = f.readlines()
        # templates = [json.loads(line) for line in lines]

        # print('')
        # output_act_ = []
        # for modal_i in range(ego_multimodal.shape[2]):
        #     valid_mask = future[sample_i, modal_i, :, 0]!=0
        #     output_act_.append(navigation_extractor.get_navigation_dict(future[sample_i, modal_i, :, :2])['direction 0.1to8_cls'])
        # output_act.append(output_act_)

        # out_data['output_traj'].astype(np.float32)
        # out_data['modalities'].astype(np.float32)
        # out_data['scores'].astype(np.float32)
        # out_data['loss']
        # out_data['text']
        
        # template_dir
        # file_path = os.path.join(data_dir, filename)
        # with open(template_dir) as f:
        #     lines = f.readlines()
        # templates = [json.loads(line) for line in lines]
        # template_select = self.template_select
        # instruct = templates[template_select]['Instruction']
        # instruct = f'<s>[INST] {instruct} '
        # reason = templates[template_select]['Reasoning']
        # decision = templates[template_select]['Decision']
        # caption = f"{reason} Decision: {decision}. Generated trajectories embeddings: "
        # print(act)
        # print('')


    # Compute additional metrics
    metrics_results = epoch_metrics.result()
    rgif = hits_stats['rgif']/hits_stats['rgif_c'] if hits_stats['rgif_c'] != 0 else 0
    any_rgif = hits_stats['any_rgif']/hits_stats['any_rgif_c'] if hits_stats['any_rgif_c'] != 0 else 0
    class_rgif = [hits_stats['rgif_class'][class_i]/hits_stats['rgif_class_c'][class_i] if hits_stats['rgif_class_c'][class_i] != 0 else 0 for class_i in range(num_classes)]
    class_rgif_count_percentage = [hits_stats['rgif_class_c'][class_i]/hits_stats['rgif_c'] if hits_stats['rgif_c'] != 0 else 0 for class_i in range(num_classes)]
    class_avg_rgif = sum(np.array(class_rgif))/len(np.array(class_rgif))
    class_avg_diversity = sum(np.array(hits_stats['gdiversity_class'])/np.array(hits_stats['gdiversity_class_c']))/num_classes
    avg_diversity = hits_stats['gdiversity']/hits_stats['gdiversity_c']
    avg_recall_mode_class = sum(np.array(hits_stats['mode_class_recall']))/sum(np.array(hits_stats['mode_class_recall_count']))
    class_avg_recall_mode_class = sum(np.array(hits_stats['mode_class_recall'])/np.array(hits_stats['mode_class_recall_count']))/num_classes
    accept_reject_accuracy = hits_stats['decision_accuracy']/hits_stats['decision_accuracy_num_examples'] if hits_stats['decision_accuracy_num_examples']!=0 else -1

    # print(f"mADEv: {np.mean(ade)}, mFDEv: {np.mean(fde)}")
    # Prepare the data to write to CSV)
    csv_data = [
        {
            'dir': data_dir+"_cls" if with_classification else data_dir,
            'mADE': metrics_results['minADE'],
            'mADEv': metrics_results['minADE_vehicle'],
            # 'mADEv_': np.mean(ade),
            'mFDE': metrics_results['minFDE'],
            'mFDEv': metrics_results['minFDE_vehicle'],
            # 'mFDEv_': np.mean(fde),
            'IFR(Avg)': rgif * 100,
            'DVS(Avg)': avg_diversity * 100,
            'Acc': accept_reject_accuracy,
            'IFR(CAvg)': class_avg_rgif * 100,
            'DVS(cAVG)': class_avg_diversity * 100,
            'ModeIFR(AVG)': avg_recall_mode_class * 100, #Recall of mode generated direction (avg)
            'ModeIFR(cAVG)': class_avg_recall_mode_class * 100, #Recall of mode generated direction (Class avg)
            'AnyIFR(Avg)': any_rgif * 100,
            # 'c1': class_rgif_count_percentage[0] * 100,
            # 'c2': class_rgif_count_percentage[1] * 100,
            # 'c3': class_rgif_count_percentage[2] * 100,
            # 'c4': class_rgif_count_percentage[3] * 100,
            # 'c5': class_rgif_count_percentage[4] * 100,
            # 'c6': class_rgif_count_percentage[5] * 100,
            # 'c7': class_rgif_count_percentage[6] * 100,
            # 'c8': class_rgif_count_percentage[7] * 100,

        }
    ]

    # Dynamically adding class_rgif_count_percentage with 'c0', 'c1', ... 'cN'
    for i in range(num_classes):
        csv_data[0][f'c{i+1}'] = class_rgif_count_percentage[i] * 100

    # Write the data to CSV
    if with_classification and eval_accepted_only:
        if two_agent:
            csv_file = '/'.join(data_dir.split('/')[:-1])+'/'+data_dir.split('/')[-1]+f"_agent{agent_select}_metrics_acceptedOnly.csv"
        else:
            csv_file = '/'.join(data_dir.split('/')[:-1])+'/'+data_dir.split('/')[-1]+'metrics_acceptedOnly.csv'
    else:
        if two_agent:
            csv_file = '/'.join(data_dir.split('/')[:-1])+'/'+data_dir.split('/')[-1]+f"_agent{agent_select}_metrics.csv"
        else:
            csv_file = '/'.join(data_dir.split('/')[:-1])+'/'+data_dir.split('/')[-1]+'metrics.csv'

    with open(csv_file, 'w', newline='') as csvfile:
        # fieldnames = [
        #     'minADE', 'minADE_vehicle', 'minFDE', 'minFDE_vehicle',
        #     'Recall Instruction Following (Avg)', 'Any Modality Recall Instruction Following (Avg)',
        #     'Recall Instruction Following (Class Avg)', 'Diversity (Class avg)',
        #     'Recall of mode generated direction (Class avg)', 'Diversity (avg)',
        #     'Recall of mode generated direction (avg)', 'class-1', 'class-2', 'class-3',
        #     'class-4', 'class-5', 'class-6', 'class-7', 'class-8', 'accept_reject_accuracy'
        # ]
        fieldnames = list(csv_data[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
        
    # Append or replace the entry in the CSV
    append_or_replace_entry(csv_file_path, csv_data, 'dir')
    print(f"Metrics for {data_dir} have been saved to {csv_file_path}")

    print(f'Data successfully written to {csv_file}')
        

    print(f"*** {models_names_and_description[data_i]}")
    print(f"* Directory: {data_dir}")
    

    metrics_results = epoch_metrics.result()
    if print_all_waymo_metrics:
        # print('---'*15)
        # print('---'*15)
        print('---'*3+f' All Waymo Metrics >>>')
        for k,v in metrics_results.items():
            print(f"> {k}: {v:.4f}")
            
    # print('---'*15)
    # print('---'*15)
    print('---'*3+f' Selected Waymo Metrics >>>')
    print(f">> Number of evaluation examples {hits_stats['num_examples']}")
    print(f">> minADE: {metrics_results['minADE']:.4f}")
    print(f">> minADE_vehicle: {metrics_results['minADE_vehicle']:.4f}")
    print(f">> minFDE: {metrics_results['minFDE']:.4f}")
    print(f">> minFDE_vehicle: {metrics_results['minFDE_vehicle']:.4f}")

    # print('---'*15)
    # print('---'*15)
    print('---'*3+f' RGIF >>>')
    rgif = hits_stats['rgif']/hits_stats['rgif_c'] if hits_stats['rgif_c']!=0 else 0
    any_rgif = hits_stats['any_rgif']/hits_stats['any_rgif_c'] if hits_stats['any_rgif_c']!=0 else 0
    class_rgif = [hits_stats['rgif_class'][class_i]/hits_stats['rgif_class_c'][class_i] if hits_stats['rgif_class_c'][class_i]!=0 else 0 for class_i in range(num_classes)]
    class_rgif_count_percentage = [hits_stats['rgif_class_c'][class_i]/hits_stats['rgif_c'] if hits_stats['rgif_c']!=0 else 0 for class_i in range(num_classes)]
    class_avg_rgif = sum(np.array(class_rgif))/len(np.array(class_rgif))
    print(f">> Recall Instruction Following (Avg) = {rgif*100:.2f} %")
    print(f">> Any Modalitry Recall Instruction Following (Avg) = {any_rgif*100:.2f} %")
    print(f">> Recall Instruction Following (Class Avg) = {class_avg_rgif*100:.2f} %")
    print('>> Evaluation Class Distribution: ')
    for class_i in range(num_classes):
        print(f"> class-{class_i+1}: {class_rgif_count_percentage[class_i]*100:.2f} %")

    # print('---'*15)
    # print('---'*15)
    print('---'*3+f' DIVERSITY >>>')
    class_avg_diversity = sum(np.array(hits_stats[f'gdiversity_class'])/np.array(hits_stats[f'gdiversity_class_c']))/num_classes
    avg_diveristy = hits_stats['gdiversity']/hits_stats['gdiversity_c']
    avg_recall_mode_class = sum(np.array(hits_stats['mode_class_recall']))/sum(np.array(hits_stats['mode_class_recall_count']))
    class_avg_recall_mode_class = sum(np.array(hits_stats['mode_class_recall'])/np.array(hits_stats['mode_class_recall_count']))/num_classes
    print(f'> diversity (Class avg) = {class_avg_diversity*100:.2f} %')
    print(f'> Recall of mode generated direction (Class avg)= {class_avg_recall_mode_class*100:.2f} %')
    print(f'> diversity (avg) = {avg_diveristy*100:.2f} %')
    print(f'> Recall of mode generated direction (avg)= {avg_recall_mode_class*100:.2f} %')
    # hits_stats['g_categorical_diversity'][mode_diversity_count]
    {''}
    print('###'*30)

    print(f'>>>> wrong_label : {wrong_label}')
