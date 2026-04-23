# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import pickle
import time

import numpy as np
import torch
import tqdm

from mtr.utils import common_utils
from trajgpt.multimodal_viz import *
# from classifytrack import ClassifyTrack, get_batch_instruct
# import statistics
from scipy import stats
import csv
import sys
import matplotlib.cm as cm
sys.path.append("/home/felembaa/projects/iMotion-LLM-ICLR")
from instructions.extract_instructions import futureNavigation
from gameformer.utils.data_utils import *
# from extract_instruct_v3 import *
def categorical_diversity_metric(acts):
    ## mode class diversity
    num_modalities = acts.shape[1]
    per_categ_count = {i+1:0 for i in range(num_modalities)}
    mode_, count_ = stats.mode(acts, axis=1, keepdims=False)
    for i in per_categ_count.keys():
        per_categ_count[i] = sum(count_==i) # count_ ==  num_modalities, means all modalities in the same direction
    return per_categ_count

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
        output[f'RGIF class-{i}'] = hits_stats['rgif_class'][i]/hits_stats['rgif_class_c'][i]
        output[f'RGIF class-{i} count'] = hits_stats['rgif_class_c'][i]
        output[f'RGIF class-{i} count perc'] = hits_stats['rgif_class_c'][i]/output['RGIF count']
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

def instruction_following_and_diversity(results_dict):
    modalities = 6
    num_classes=8
    hits_stats = {
        'rgif':0, 'any_rgif':0, 'rgif_c':0, 'any_rgif_c':0, 'rgif_class':[0 for _ in range(num_classes)], 'rgif_class_c':[0 for _ in range(num_classes)], 
        'rcif':0, 'any_rcif':0, 'rcif_c':0, 'any_rcif_c':0, 'rcif_class':[0 for _ in range(num_classes)], 'rcif_class_c':[0 for _ in range(num_classes)], 
        'gdiversity':0, 'cdiversity':0, 'gdiversity_c':0, 'cdiversity_c':0, 
        'g_categorical_diversity': {i+1:0 for i in range(modalities)},
        'c_categorical_diversity': {i+1:0 for i in range(modalities)},
        } 
    for i in range(len(results_dict)):
        batch_dict = results_dict[i] 
        
        ## select the ego only if two predcitions are provides
        if len(np.unique(batch_dict['scenario_id']))==len(batch_dict['scenario_id'])/2:
            selected_batch_mask = [ii%2==0 for ii in range(len(batch_dict['scenario_id']))]
            mask_dict(batch_dict, selected_batch_mask)
            
        ## select vehicles only
        vehicle_mask = [type_i == 'TYPE_VEHICLE' for type_i in batch_dict['object_type']]
        mask_dict(batch_dict, vehicle_mask)
        
        ## select valid trajectory
        valid_mask = (batch_dict['gt']!=0).any(dim=-1).any(dim=-1)
        mask_dict(batch_dict, valid_mask)

        gt_act, pred_act = batch_dict['instruct'], batch_dict['pred_instruct']
        
        calculate_metrics(gt_act, pred_act, hits_stats, num_classes, 'g') # ground truth instruction

def var_copy(input):
    if torch.is_tensor(input):
        return input.detach().clone()
    elif isinstance(input, list):
        return input.copy()
    else:
        return input.copy()

def gen_contrastive_samples(samples, instruct, contrastive_instruct): 
    num_contrastive = len(contrastive_instruct[0])
    samples__ = {k: var_copy(samples[k]) for k in samples}
    contrastive_flag = torch.ones((num_contrastive+1)*(len(samples['act']))).bool().to(samples['act'].device)
    contrastive_flag[:len(samples['act'])] = False
    for contrastive_i in range(num_contrastive):
        samples_ = {k: var_copy(samples__[k]) for k in samples__}
        # samples_['act'] = list(contrastive_instruct[:,contrastive_i])
        samples_['act'] = contrastive_instruct[:,contrastive_i]
        for k in samples.keys():
            if torch.is_tensor(samples[k]):
                samples[k] = torch.cat((samples[k], samples_[k]), dim=0)
            elif isinstance(samples[k], list):
                samples[k] = samples[k] + samples_[k]
            else:
                samples[k] = np.append(samples[k],samples_[k],axis=0)

    samples['contrastive_flag'] = contrastive_flag

def eval_one_epoch_(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    pred_dicts = []

    modalities = 6
    num_classes=8
    hits_stats = {
        'rgif':0, 'any_rgif':0, 'rgif_c':0, 'any_rgif_c':0, 'rgif_class':[0 for _ in range(num_classes)], 'rgif_class_c':[0 for _ in range(num_classes)], 
        'rcif':0, 'any_rcif':0, 'rcif_c':0, 'any_rcif_c':0, 'rcif_class':[0 for _ in range(num_classes)], 'rcif_class_c':[0 for _ in range(num_classes)], 
        'gdiversity':0, 'cdiversity':0, 'gdiversity_c':0, 'cdiversity_c':0, 
        'g_categorical_diversity': {i+1:0 for i in range(modalities)},
        'c_categorical_diversity': {i+1:0 for i in range(modalities)},
        }
    center_normalized_pred = []

    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            # generate instruction
            sampling_idx = [ii for ii in range(80)][::10] + [79]
            instruct, contrastive_instruct = get_batch_instruct(batch_dict['input_dict']['center_gt_trajs'][...,:2], sampling_idx=sampling_idx)
            if cfg.MODEL.MOTION_DECODER.ACT:
                batch_dict['input_dict']['act'] = instruct
                gen_contrastive_samples(batch_dict['input_dict'], instruct, contrastive_instruct)
                
                # we need to repeat the values of the following to match the increased number of contrastive examples
                # batch_dict['batch_sample_count'] # TODO: expand this to match the expansion when contrastive examples used
                scale_batch_sample_count = batch_dict['input_dict']['act'].shape[0]/instruct.shape[0]
                batch_sample_count = []
                for ii in range(int(scale_batch_sample_count)):
                    batch_sample_count += batch_dict['batch_sample_count'].copy()
                batch_dict['batch_sample_count'] = batch_sample_count
                batch_dict['batch_size'] = len(batch_dict['batch_sample_count'])
                batch_dict['input_dict']['act'] = batch_dict['input_dict']['act'][:,0] # ego only
            else:
                batch_dict['input_dict']['act'] = instruct[:, 0]
            
            odd_idx = [ii for ii in range(batch_dict['input_dict']['act'].shape[0]) if ii%2!=0] # odd indices, because this baseline view both interactive agents in the same dimension
            batch_dict['input_dict']['act'][odd_idx] = -1
            not_vehicle_idx = [ii!='TYPE_VEHICLE' for ii in batch_dict['input_dict']['center_objects_type']]
            batch_dict['input_dict']['act'][not_vehicle_idx] = -1
                
            batch_pred_dicts = model(batch_dict)
            final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=final_output_dir if save_to_file else None)
            pred_dicts += final_pred_dicts
            
            
            ## Instruction following data
            # sampling_idx = [ii for ii in range(80)][::10] + [79]
            # gt_instruct, _ = get_batch_instruct(batch_dict['input_dict']['center_gt_trajs'][...,:2], sampling_idx=sampling_idx)
            # gt_instruct = gt_instruct[:,0].int()
            
            pred_instruct = []
            for mode_i in range(batch_pred_dicts['pred_trajs'].shape[1]):
                pred_instruct_, _ = get_batch_instruct(batch_pred_dicts['pred_trajs'][:,mode_i, ...,:2], sampling_idx=sampling_idx)
                pred_instruct.append(pred_instruct_)
            pred_instruct = torch.stack(pred_instruct, dim=1)
            pred_instruct = pred_instruct[:,:,0].int()
            pred_act = pred_instruct.cpu()
            gt_act = batch_dict['input_dict']['act'].cpu()
            if cfg.MODEL.MOTION_DECODER.ACT:
                calculate_metrics(gt_act[~batch_dict['input_dict']['contrastive_flag']], pred_act[~batch_dict['input_dict']['contrastive_flag']], hits_stats, 8, 'g')
                calculate_metrics(gt_act[batch_dict['input_dict']['contrastive_flag']], pred_act[batch_dict['input_dict']['contrastive_flag']], hits_stats, 8, 'c')
            else:
                calculate_metrics(gt_act, pred_act, hits_stats, 8, 'g')

            # center_normalized_pred += [{'gt': batch_dict['input_dict']['center_gt_trajs'].cpu(), 
            # 'pred': batch_dict['input_dict']['center_gt_trajs'].cpu(),
            # 'scenario_id': list(batch_dict['input_dict']['scenario_id']),
            # 'object_type': list(batch_dict['input_dict']['center_objects_type']),
            # 'instruct': gt_instruct.cpu(),
            # 'pred_instruct': pred_instruct.cpu(),
            # }]

            ## Plotting
            if False:
                for batch_i in [ii for ii in range(len(gt_instruct))][::2]:
                    fig, ax = plt.subplots()
                    plt.plot(batch_dict['input_dict']['center_gt_trajs'][batch_i,sampling_idx,0], batch_dict['input_dict']['center_gt_trajs'][batch_i,sampling_idx,1], linestyle='--', color='b', alpha=0.5)
                    mode_i = 0
                    plt.plot(batch_pred_dicts['pred_trajs'][batch_i,mode_i,sampling_idx,0].cpu(), batch_pred_dicts['pred_trajs'][batch_i,mode_i,sampling_idx,1].cpu(), linestyle='--', color='r', alpha=0.5)
                    turn_classes = ['STATIONARY', 'STRAIGHT', 'STRAIGHT_RIGHT', 'STRAIGHT_LEFT', 'RIGHT_TURN', 'LEFT_TURN', 'RIGHT_U_TURN', 'LEFT_U_TURN']
                    ax.set_title(f"gt: {turn_classes[gt_instruct[batch_i]]}, pred: {turn_classes[pred_instruct[batch_i, mode_i]]}, batch: {batch_i}")
                    max_value = max(max(batch_pred_dicts['pred_trajs'][batch_i, mode_i, sampling_idx,0]), max(batch_dict['input_dict']['center_gt_trajs'][batch_i,sampling_idx,1])).cpu().item()
                    min_value = min(min(batch_pred_dicts['pred_trajs'][batch_i, mode_i, sampling_idx,0]), min(batch_dict['input_dict']['center_gt_trajs'][batch_i,sampling_idx,1])).cpu().item()
                    ax.set_xlim([min_value, max_value])
                    ax.set_ylim([min_value, max_value])
                    plt.savefig('ex.png')
                    print('')


        disp_dict = {}

        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')
        # if i == 10:
        #     break # comment
    
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
        pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')
        logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}

    # # new evaluation
    # ego_future = np.vstack([pred_dicts[i][0]['pred_trajs'][None,...] for i in range(len(pred_dicts))])
    # ego_gt = np.vstack([pred_dicts[i][0]['gt_trajs'][None,...] for i in range(len(pred_dicts))])
    # ego_history = ego_gt[:,:11]
    # ego_future_gt = ego_gt[:,11:]
    # num_classes = 8
    
    # gt_act = get_batch_instruct(torch.tensor(ego_future_gt[:,:,:2]), torch.tensor(ego_history[:,:,:2]), num_classes=num_classes)
    # pred_act = torch.stack([get_batch_instruct(torch.tensor(ego_future[:,modal_i,:,:]), torch.tensor(ego_history[:,:,:2]), num_classes=num_classes) for modal_i in range(ego_future.shape[1])], dim=1)
    
    # vehicle_mask = torch.zeros(len(pred_dicts)).bool()
    # for i in range(len(pred_dicts)):
    #     if pred_dicts[i][0]['object_type'] == 'TYPE_VEHICLE':
    #         vehicle_mask[i] = True
    #     else:
    #         vehicle_mask[i] = False
    # # TODO: Check valid gt trajectory before calculating anything
    # gt_act, pred_act = gt_act[vehicle_mask], pred_act[vehicle_mask]
    # valid_act_mask = gt_act[:,0]!=-1
    # gt_act, pred_act = gt_act[valid_act_mask], pred_act[valid_act_mask]
    # calculate_metrics(gt_act[...,0], pred_act[...,0], hits_stats)
    if cfg.MODEL.MOTION_DECODER.ACT:
        save_metrics_to_csv(epoch_metrics=None, epoch_metrics_contrastive= None, hits_stats=hits_stats, filename='/ibex/user/felembaa/mtr_models/cmtr+20_percent_data/run01/', act=True, epoch=30, model_name='baseline')
    else:
        save_metrics_to_csv(epoch_metrics=None, epoch_metrics_contrastive= None, hits_stats=hits_stats, filename='/ibex/user/felembaa/mtr_models/mtr+20_percent_data/run01/', act=False, epoch=30, model_name='baseline')
    # dict_to_csv(hits_stats, str(result_dir)+f'main_metrics.csv')
    # target_trajs = np.vstack(target_trajs)
    # target_trajs = target_trajs[vehicle_mask]

    # batch_i_sample = 0
    
    # colors = ['r', 'b', 'g', 'y', 'k', 'c']
    # for j in range(len(target_trajs)):
    #     fig, ax = plt.subplots()
    #     for i, traj in enumerate(target_trajs[j]):
    #         plt.plot(traj[:,0], traj[:,1], linestyle='--', color=colors[i], alpha=0.5) 
    #     plt.savefig('ex.png')
    #     print('')
    # # end of new evaluation

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(pred_dicts, f)

    result_str, result_dict = dataset.evaluation(
        pred_dicts,
        output_path=final_output_dir, 
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50, act=False, num_classes=5):
    if act:
        navigation_extractor = futureNavigation(normalize_track=True, num_classes=num_classes)
    
    result_dir = result_dir / cfg.DATA_CONFIG.eval_mode
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'data'
    save_to_file = True
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    pred_dicts = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            ## TODO: Add Act
            if act:
                if 'act' not in batch['input_dict']:
                    # Generate ground truth actions/instructions
                    vehicle_mask = torch.tensor([type_i == 'TYPE_VEHICLE' for type_i in batch['input_dict']['center_objects_type']], device=batch['input_dict']['center_gt_trajs'].device)
                    sampling_idx = [ii for ii in range(80)][::10] + [79]
                    ## TODO: Fix missing values
                    instructs_ego_dicts = []
                    for batch_i in range(len(batch['input_dict']['center_gt_trajs'][..., :2])):
                        try:
                            # Extract the navigation dictionary and direction classification
                            direction = navigation_extractor.get_navigation_dict(
                                batch['input_dict']['center_gt_trajs'][batch_i, ..., :2].detach().clone()
                            )['direction 0.1to8_cls']
                            # Validate the direction value
                            if isinstance(direction, int) and 0 <= direction <= 7:
                                instructs_ego_dicts.append(direction)
                            else:
                                instructs_ego_dicts.append(-1)  # Replace invalid values with -1
                        except Exception:
                            # Handle any errors and fill with -1
                            instructs_ego_dicts.append(-1)
                    instructs_ego_dicts = torch.tensor(instructs_ego_dicts, device=batch['input_dict']['center_gt_trajs'].device)
                    batch['input_dict']['act'] = instructs_ego_dicts
                    batch['input_dict']['act'][~vehicle_mask] = -1

            batch_dict = batch
            batch_pred_dicts = model(batch_dict)
            final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=final_output_dir if save_to_file else None)
            pred_dicts += final_pred_dicts

            if save_to_file:
                for file_i, filename in enumerate(batch['input_dict']['filename']):
                    to_save_dict = {}
                    save_dir = final_output_dir / (filename+'.pkl')
                    # get the center and heading (local view)
                    #[np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading, x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
                    to_save_dict['gt_trajs'] = final_pred_dicts[file_i][0]['gt_trajs'][:, [0,1,6,7,8]].copy()
                    center, angle = to_save_dict['gt_trajs'][10][:2].copy(), to_save_dict['gt_trajs'][10][2].copy()
                    to_save_dict['gt_trajs'] = agent_norm(to_save_dict['gt_trajs'], center, angle, impute=True).astype(np.float16)
                    to_save_dict['modalities'] = final_pred_dicts[file_i][0]['pred_trajs'].astype(np.float16)
                    for m_i, m in enumerate(to_save_dict['modalities']):
                        to_save_dict['modalities'][m_i] = agent_norm(m.copy(), center, angle)
                    to_save_dict['scores'] = final_pred_dicts[file_i][0]['pred_scores'].astype(np.float16)
                    to_save_dict['act'] = batch['input_dict']['act'][file_i]
                    
                    to_save_dict['modalities'] = to_save_dict['modalities'][None]
                    to_save_dict['scores'] = to_save_dict['scores'][None]
                    # Save the dictionary as a .pkl file
                    with open(save_dir, 'wb') as f:
                        pickle.dump(to_save_dict, f)


                    
            

            if False:
                act1 = batch['input_dict']['act'].clone().detach()
                trajectory1 = batch_pred_dicts['pred_trajs'].clone().detach().cpu()
                if False: # to visualize if contrastive is working, with 8 classes setup
                    ordered_acts = ['stationary', 'move straight', 'move straight veering right', 'move straight veering left', 'turn right', 'turn left', 'take right U-turn', 'take left U-turn']
                    invert_act = {0:1, 1:4, 2:0, 3:0, 4:5, 5:4, 6:1, 7:1, -1:-1}
                    if num_classes==5:
                        raise 'Not implemented'
                    batch['input_dict']['act'] = torch.tensor([invert_act.get(a.item(), a.item()) for a in batch['input_dict']['act']])
                    batch_pred_dicts = model(batch_dict)
                    act2 = batch['input_dict']['act'].clone().detach()
                    trajectory2 = batch_pred_dicts['pred_trajs'].clone().detach().cpu()
                    # for ii in range(trajectory2.shape[0]):
                    if True:
                        ii=0
                        # Create figure
                        plt.figure(figsize=(8, 6))

                        # Define colormap for gradients
                        cmap1 = cm.get_cmap("Blues", 100)  # Gradient shades of blue for the first set
                        cmap2 = cm.get_cmap("Reds", 100)   # Gradient shades of red for the second set

                        # Plot first set of trajectories with a gradient color
                        for iii in range(6):
                            plt.plot(trajectory1[ii][iii, :, 0], trajectory1[ii][iii, :, 1], color=cmap1(iii+50), alpha=0.8, label=f'{ordered_acts[act1[ii]]}' if iii == 0 else "")

                        # Plot second set of trajectories with a gradient color
                        for iii in range(6):
                            plt.plot(trajectory2[ii][iii, :, 0], trajectory2[ii][iii, :, 1], color=cmap2(iii+50), alpha=0.8, label=f'{ordered_acts[act2[ii]]}' if iii == 0 else "")

                        # Save and show the figure
                        plt.xlabel("X-axis")
                        plt.ylabel("Y-axis")
                        plt.title("Gradient Overlayed Trajectories")
                        plt.grid(True)
                        plt.legend()
                        plt.savefig("ex.png")
                        plt.show()
        if True:
            disp_dict = {}

            if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
                past_time = progress_bar.format_dict['elapsed']
                second_each_iter = past_time / max(i, 1.0)
                remaining_time = second_each_iter * (len(dataloader) - i)
                disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
                batch_size = batch_dict.get('batch_size', None)
                logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                            f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                            f'{disp_str}')
    if True:
        if cfg.LOCAL_RANK == 0:
            progress_bar.close()

        if dist_test:
            logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
            pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')
            logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

        logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
        sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
        logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

        if cfg.LOCAL_RANK != 0:
            return {}

        ret_dict = {}

        with open(result_dir / 'result.pkl', 'wb') as f:
            pickle.dump(pred_dicts, f)

        result_str, result_dict = dataset.evaluation(
            pred_dicts,
            output_path=final_output_dir, 
        )

        logger.info(result_str)
        ret_dict.update(result_dict)

        logger.info('Result is save to %s' % result_dir)
        logger.info('****************Evaluation done.*****************')

        return ret_dict
    return {}


if __name__ == '__main__':
    pass
