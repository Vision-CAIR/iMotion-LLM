import os
from PIL import Image
# import webdataset as wds
# from minigpt4.datasets.datasets.traj_base_dataset import BaseDataset
# from minigpt4.datasets.datasets.traj_caption_datasets import CaptionDataset
import torch
import numpy as np
from traj_utils import *
from torch.utils.data import DataLoader, Dataset
# from generate_meta_caption import *
import random
# from extract_instruct_v2 import extract_simple_turn_5_classes
# from extract_instruct_v3 import ClassifyTrack, get_sample_instruct
from extract_instruct_v3 import *
# from extract_instruct_v4 import *
from inter_pred_utils import *
from scipy import stats
import csv
from tqdm import tqdm

def save_metrics_to_csv(epoch_metrics, epoch_metrics_contrastive, hits_stats, filename, act=False, epoch='', model_name=''):
    output = {}
    output['model'] = model_name
    
    ## Waymo metrics
    if epoch_metrics is not None:
        metrics_results = epoch_metrics.result()
        keys = ['minADE', 'minFDE', 'minADE_vehicle', 'minFDE_vehicle', 'mAP', 'overlap_rate', 'miss_rate']
        for k in keys:
            output[k] = metrics_results[k]    
        if act:
            contrastive_metrics_results = epoch_metrics_contrastive.result()
            for k in keys:
                output['c_'+k] = contrastive_metrics_results[k]

    ## Recall
    output['RGIF'] = hits_stats['rgif']/hits_stats['rgif_c']
    output['any RGIF'] = hits_stats['any_rgif']/hits_stats['any_rgif_c']
    output['RGIF pure'] = hits_stats['rgif']
    output['RGIF count'] = hits_stats['rgif_c']
    output[f'RGIF class weighted Avg'] = 0
    for i in range(len(hits_stats['rgif_class'])):
        output[f'RGIF class-{i}'] = hits_stats['rgif_class'][i]/hits_stats['rgif_class_c'][i] if hits_stats['rgif_class_c'][i]>0 else 0
        output[f'RGIF class-{i} count'] = hits_stats['rgif_class_c'][i]
        output[f'RGIF class-{i} count perc'] = hits_stats['rgif_class_c'][i]/output['RGIF count'] if output['RGIF count']>0 else 0
        output[f'RGIF class weighted Avg'] += output[f'RGIF class-{i}']*output[f'RGIF class-{i} count perc']
    output[f'RGIF class Avg'] = np.average([hits_stats['rgif_class'][i]/hits_stats['rgif_class_c'][i] for i in range(len(hits_stats['rgif_class']))])

    if act:
        output['RCIF'] = hits_stats['rcif']/hits_stats['rcif_c']
        output['any RCIF'] = hits_stats['any_rcif']/hits_stats['any_rcif_c']
        output['RCIF pure'] = hits_stats['rcif']
        output['RCIF count'] = hits_stats['rcif_c']
        output[f'RCIF class weighted Avg'] = 0
        for i in range(len(hits_stats['rcif_class'])):
            output[f'RCIF class-{i}'] = hits_stats['rcif_class'][i]/hits_stats['rcif_class_c'][i]
            output[f'RCIF class-{i} count'] = hits_stats['rcif_class_c'][i]
            output[f'RCIF class weighted Avg'] += output[f'RCIF class-{i}']*output[f'RGIF class-{i} count perc'] # we use the RGIC count perc, as it represent the percentage of test examples per class. Using the RCIF count perc could be misleading in the current implementation where contrastive examples of all classes are generated given one ground truth class
        output[f'RCIF classAvg'] = np.average([hits_stats['rcif_class'][i]/hits_stats['rcif_class_c'][i] for i in range(len(hits_stats['rcif_class']))])
        
    ## Diversity
    output[f'G_diversity'] = hits_stats['gdiversity']/hits_stats['gdiversity_c']
    output[f'G_diversity pure'] = hits_stats['gdiversity']
    output[f'G_diversity count'] = hits_stats['gdiversity_c']
    if act:
        output[f'C_diversity'] = hits_stats['cdiversity']/hits_stats['cdiversity_c']
        output[f'C_diversity pure'] = hits_stats['cdiversity']
        output[f'C_diversity count'] = hits_stats['cdiversity_c']
    
    ## Categorical diversity
    g_categ_diversity = {i: hits_stats['g_categorical_diversity'][i]/sum(hits_stats['g_categorical_diversity'].values()) for i in range(1,7)}
    for i in g_categ_diversity.keys():
        output[f'G_diversity mode-{i} count'] = g_categ_diversity[i]
    if act:
        c_categ_diversity = {i: hits_stats['c_categorical_diversity'][i]/sum(hits_stats['c_categorical_diversity'].values()) for i in range(1,7)}
        for i in c_categ_diversity.keys():
            output[f'C_diversity mode-{i} count'] = c_categ_diversity[i]
    
    correct_diversity = {i:hits_stats['gdiversity_class'][i]/hits_stats['gdiversity_class_c'][i] for i in range(num_classes)}
    for i in range(num_classes):
        output[f'class-{i} diversity (new)'] = correct_diversity[i]
    
    dict_to_csv(output, filename+f'main_metrics_{epoch}.csv')

def dict_to_csv(input_dict, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=input_dict.keys())
        # Write header
        writer.writeheader()
        # Write rows
        writer.writerow(input_dict)


def diversity_metric(acts):
    ## mode class diversity
    num_modalities = acts.shape[1]
    mode_, count_ = stats.mode(acts, axis=1, keepdims=False)
    diversity = (num_modalities - count_)/(num_modalities-1)
    diversity = sum(diversity)
    num_examples = count_.shape[0]
    return diversity, num_examples

def categorical_diversity_metric(acts):
    ## mode class diversity
    num_modalities = acts.shape[1]
    per_categ_count = {i+1:0 for i in range(num_modalities)}
    mode_, count_ = stats.mode(acts, axis=1, keepdims=False)
    for i in per_categ_count.keys():
        per_categ_count[i] = sum(count_==i) # count_ ==  num_modalities, means all modalities in the same direction
    return per_categ_count

def calculate_metrics(gt_act, pred_act, hits_stats, num_classes, g_or_c):
    modal_hits = gt_act[:, None] == pred_act
    ## RGIF
    hits_stats[f'r{g_or_c}if']+= modal_hits.sum().item()
    hits_stats[f'r{g_or_c}if_c']+= int(modal_hits.shape[0]*modal_hits.shape[1])
    ## Any RGIF
    hits_stats[f'any_r{g_or_c}if']+= modal_hits.any(dim=-1).sum().item()
    hits_stats[f'any_r{g_or_c}if_c']+= modal_hits.shape[0]
    ## Class RGIF
    for class_i in range(num_classes):
        class_mask = gt_act == class_i
        num_class_examples = sum(class_mask).item()
        class_modal_hits = pred_act[class_mask,:] == gt_act[class_mask, None]
        hits_stats[f'r{g_or_c}if_class'][class_i] += class_modal_hits.sum().item()
        hits_stats[f'r{g_or_c}if_class_c'][class_i] += num_class_examples*class_modal_hits.shape[1]

        class_diversity = [len(np.unique(pred_act[class_mask][i]))/pred_act[class_mask].shape[1] for i in range(len(pred_act[class_mask]))]
        hits_stats[f'{g_or_c}diversity_class'][class_i] += sum(class_diversity)
        hits_stats[f'{g_or_c}diversity_class_c'][class_i] += len(class_diversity)

    ## mode class diversity
    diversity, diversity_c = diversity_metric(pred_act)
    hits_stats[f'{g_or_c}diversity'] += diversity
    hits_stats[f'{g_or_c}diversity_c'] += diversity_c
    ## categorical class diversity, num_categ = num_modalities
    categorical_diversity = categorical_diversity_metric(pred_act)
    hits_stats[f'{g_or_c}_categorical_diversity'] = {k: hits_stats[f'{g_or_c}_categorical_diversity'][k] + categorical_diversity[k] for k in hits_stats[f'{g_or_c}_categorical_diversity'].keys()}

def mask_dict(dict_in, bool_mask):
    for k in dict_in.keys():
        if isinstance(dict_in[k], list):
            dict_in[k] = [dict_in[k][ii] for ii in range(len(bool_mask)) if bool_mask[ii]]
        else:
            dict_in[k] = dict_in[k][bool_mask]


model_name = 'mar16_04_8c_base'
# model_name = 'mar16_03_actdecl6_8c'
# model_name = 'mar16_03_actdecl6_8c_v2'
# model_name = 'mar16_04_8c_base_cllm_v2'
conditional_model = False
epoch = 100
root_dir = '/ibex/user/felembaa/llm_models/cgf/'
data_dir = root_dir+model_name+f'/eval{epoch}/data/*'
data_list = glob.glob(data_dir)
print(f'Number of found files: {len(data_list)}')

motion_metrics = MotionMetrics()
contrastive_motion_metrics = MotionMetrics()

modalities = 6
num_classes=8
hits_stats = {
    'rgif':0, 'any_rgif':0, 'rgif_c':0, 'any_rgif_c':0, 'rgif_class':[0 for _ in range(num_classes)], 'rgif_class_c':[0 for _ in range(num_classes)], 
    'rcif':0, 'any_rcif':0, 'rcif_c':0, 'any_rcif_c':0, 'rcif_class':[0 for _ in range(num_classes)], 'rcif_class_c':[0 for _ in range(num_classes)], 
    'gdiversity_class':[0 for _ in range(num_classes)], 'gdiversity_class_c':[0 for _ in range(num_classes)],
    'cdiversity_class':[0 for _ in range(num_classes)], 'cdiversity_class_c':[0 for _ in range(num_classes)],
    'gdiversity':0, 'cdiversity':0, 'gdiversity_c':0, 'cdiversity_c':0, 
    'g_categorical_diversity': {i+1:0 for i in range(modalities)},
    'c_categorical_diversity': {i+1:0 for i in range(modalities)},
    }

for i in tqdm(range(len(data_list))):
    data = np.load(data_list[i], allow_pickle = True)
    ## Instruction following and diversity
    llm_classes = ['stay stationary', 'move straight', 'move straight while veering to the right', 'move straight while veering to the left', 'turn right', 'turn left', 'take a right u-turn', 'take a left u-turn', 'unknown']
    gt_act = np.array([llm_classes.index(instruct_i) for instruct_i in data['instruct']])
    if conditional_model:
        contrastive_flag = data['contrastive_flag']
        while len(contrastive_flag)<len(gt_act):
            contrastive_flag = np.append(contrastive_flag, True) # missing value at the end need to be fixed in the original evaluation code in base_task.py
        if len(contrastive_flag)>len(gt_act):
            contrastive_flag = contrastive_flag[:len(gt_act)]
    gt_traj = data['gt']
    pred_traj = data['pred']
    pred_instruct = []
    for mode_i in range(pred_traj.shape[1]):
        pred_instruct_ = get_batch_instruct(torch.tensor(pred_traj[:,mode_i, ...,:2]), None, 8)
        pred_instruct.append(pred_instruct_)
    pred_instruct = torch.stack(pred_instruct, dim=1)
    pred_instruct = pred_instruct[:,:,0].int()
    pred_act = pred_instruct.cpu()
    
    egos, scores, ego_ground_truth, object_type = data['egos'], data['scores'], data['ego_ground_truth'], data['object_type']

    if conditional_model:
        calculate_metrics(torch.tensor(gt_act[~contrastive_flag]), torch.tensor(pred_act[~contrastive_flag]), hits_stats, num_classes, 'g')
        calculate_metrics(torch.tensor(gt_act[contrastive_flag]), torch.tensor(pred_act[contrastive_flag]), hits_stats, 8, 'c')
        
        motion_metrics.update_state(
            prediction_trajectory=torch.tensor(egos[~contrastive_flag]), 
            prediction_score=torch.tensor(scores[~contrastive_flag]), 
            ground_truth_trajectory=torch.tensor(ego_ground_truth[~contrastive_flag]), 
            ground_truth_is_valid=torch.ne(torch.tensor(ego_ground_truth[~contrastive_flag]), 0).bool(), 
            object_type=torch.tensor(object_type[~contrastive_flag]).long()
            )
        contrastive_motion_metrics.update_state(
            prediction_trajectory=torch.tensor(egos[contrastive_flag]), 
            prediction_score=torch.tensor(scores[contrastive_flag]), 
            ground_truth_trajectory=torch.tensor(ego_ground_truth[contrastive_flag]), 
            ground_truth_is_valid=torch.ne(torch.tensor(ego_ground_truth[contrastive_flag]), 0).bool(), 
            object_type=torch.tensor(object_type[contrastive_flag]).long()
            )
    else:
        calculate_metrics(torch.tensor(gt_act), torch.tensor(pred_act), hits_stats, 8, 'g')
        motion_metrics.update_state(
            prediction_trajectory=torch.tensor(egos), 
            prediction_score=torch.tensor(scores), 
            ground_truth_trajectory=torch.tensor(ego_ground_truth), 
            ground_truth_is_valid=torch.ne(torch.tensor(ego_ground_truth), 0).bool(), 
            object_type=torch.tensor(object_type).long()
            )
    
    
    # calculate_metrics(gt_act=None,pred_act=None,hits_stats=None,num_classes=8,g_or_c='g')
print('')

save_metrics_to_csv(epoch_metrics=motion_metrics, epoch_metrics_contrastive= contrastive_motion_metrics, hits_stats=hits_stats, filename='/'.join(data_dir.split('/')[:-2])+'/', act=conditional_model, epoch=0, model_name='baseline')