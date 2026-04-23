import torch
import sys
sys.path.append('.')
import csv
import argparse
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
import time
# from model.GameFormer import GameFormer
# from model.GameFormer import GameFormer_
from gameformer.model.GameFormer import GameFormer_2
from gameformer.utils.inter_pred_utils import *
from gameformer.utils.data_utils import *
import wandb

from gameformer.interaction_prediction.dist_utils import get_rank, init_distributed_mode
from gameformer.interaction_prediction.logger import setup_logger
from gameformer.interaction_prediction.logger import MetricLogger, SmoothedValue
import pickle
# from multimodal_viz import *

# from exctract_instruct import *
# from classifytrack import ClassifyTrack, get_batch_instruct

# import statistics
from scipy import stats
import csv
from tqdm import tqdm

from instructions.extract_instructions import futureNavigation
from evaluate_pkl_data_utils import get_act_util

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


def save_metrics_to_file(epoch_metrics, epoch_metrics_contrastive, hits_stats, filename, act=False, epoch=''):
    text_output = ""
    # text_output += '--'*50 + '\n'
    # text_output += '**'*50 + '\n'
    # text_output += '--'*50 + '\n'
    metrics_results = epoch_metrics.result()
    text_output += 'Ground truth metrics:\n'
    text_output += '--'*10 + '\n'
    for k, v in metrics_results.items():
        text_output += f"> {k}: {v}\n"
    
    # Saving the text_output to a file
    with open(filename+f"gt_results_{epoch}.txt", "w") as file:
        file.write(text_output)
    print(f"Results saved to {filename}")
    # text_output += '--'*50 + '\n'
    # text_output += '**'*50 + '\n'
    # text_output += '--'*50 + '\n'
    # text_output += '--'*50 + '\n'
    # text_output += '**'*50 + '\n'
    # text_output += '--'*50 + '\n'
    # if act:
    #     text_output = ""
    #     contrastive_metrics_results = epoch_metrics_contrastive.result()
    #     text_output += 'Contrastive instruction metrics:\n'
    #     text_output += '--'*10 + '\n'
    #     for k, v in contrastive_metrics_results.items():
    #         text_output += f"> {k}: {v}\n"

    #     # text_output += '\n'
    #     # text_output += '--'*50 + '\n'
    #     # text_output += '**'*50 + '\n'
    #     # text_output += '--'*50 + '\n'
    #     # text_output += '\n'
    #     # text_output += "Instruct following metrics:\n"
    #     # text_output += '--'*10 + '\n'
    #     # text_output += f"> Ground truth instruct recall: {hits_stats['any_modal_hits_gt']/hits_stats['count_any_modal_hits_gt']*100:.2f} %\n"
    #     # text_output += f"> Ground truth instruct recall: {hits_stats['any_modal_hits_gt']/hits_stats['count_any_modal_hits_gt']*100:.2f} %\n"
    #     # text_output += f"> Contrastive instruct recall: {hits_stats['any_modal_hits_atleast_one_contrastive']/hits_stats['count_any_modal_hits_atleast_one_contrastive']*100:.2f}%\n"
    #     # text_output += f"> Following instruct recall: {hits_stats['any_modal_hits']/hits_stats['count_any_modal_hits']*100:.2f}%\n"

    #     # Saving the text_output to a file
    #     with open(filename+f"Contrastive_results_{epoch}.txt", "w") as file:
    #         file.write(text_output)
    #     print(f"Results saved to {filename}")
    # else:
    text_output = "Recall\n"
    text_output += '--'*10 + '\n'
    rgif = hits_stats['rgif']/hits_stats['rgif_c']*100 if hits_stats['rgif_c']!=0 else 0
    text_output += f"> RGIF: {rgif:.2f} %, /{hits_stats['rgif_c']} examples\n"
    any_rgif = hits_stats['any_rgif']/hits_stats['any_rgif_c']*100 if hits_stats['any_rgif_c']!=0 else 0
    text_output += f"> Any modality RGIF: {any_rgif:.2f} %\n"
    for i in range(len(hits_stats['rgif_class'])):
        rgif_class = hits_stats['rgif_class'][i]/hits_stats['rgif_class_c'][i]*100 if hits_stats['rgif_class_c'][i]!=0 else 0
        text_output += f"> RGIF class-{i}: {rgif_class:.2f} %, /{hits_stats['rgif_class_c'][i]} examples\n"
    
    avg_rgif = np.zeros(len(hits_stats['rgif_class']))
    for i in range(len(hits_stats['rgif_class'])):
        avg_rgif[i] = hits_stats['rgif_class'][i]/hits_stats['rgif_class_c'][i] if hits_stats['rgif_class_c'][i]!=0 else 0
    avg_rgif = np.average(avg_rgif)
    text_output += f"> Average RGIF X-classes: {avg_rgif*100:.2f}\n"
    # if act:
    #     text_output += "--"*2 +'\n'
    #     text_output += f"# RCIF: {hits_stats['rcif']/hits_stats['rcif_c']*100:.2f} %, /{hits_stats['rcif_c']} examples\n"
    #     text_output += f"# Any modality RCIF: {hits_stats['any_rcif']/hits_stats['any_rcif_c']*100:.2f} %\n"
    #     for i in range(len(hits_stats['rcif_class'])):
    #         text_output += f"# RCIF class-{i}: {hits_stats['rcif_class'][i]/hits_stats['rcif_class_c'][i]*100:.2f} %, /{hits_stats['rcif_class_c'][i]} examples\n"
    #     text_output += f"# Average RCIF X-classes: {np.average([hits_stats['rcif_class'][i]/hits_stats['rcif_class_c'][i] for i in range(len(hits_stats['rcif_class']))])*100:.2f}\n"
    with open(filename+f"recall_{epoch}.txt", "w") as file:
        file.write(text_output)
    print(f"Results saved to {filename}")

    text_output = "Diversity\n"
    text_output += '--'*10 + '\n'
    gdiversity = hits_stats['gdiversity']/hits_stats['gdiversity_c']*100 if hits_stats['gdiversity_c']!=0 else 0
    text_output += f"G-Diversity: {gdiversity:.6f} %, /{hits_stats['gdiversity_c']} examples\n"
    # if act:
    #     text_output += f"C-Diversity: {hits_stats['cdiversity']/hits_stats['cdiversity_c']*100:.6f} %, /{hits_stats['cdiversity_c']} examples\n"
    with open(filename+f"diversity_{epoch}.txt", "w") as file:
        file.write(text_output)


def save_metrics_to_csv(epoch_metrics, epoch_metrics_contrastive, hits_stats, filename, act=False, epoch='', model_name='', contrastive=False):
    output = {}
    output['model'] = model_name
    
    ## Waymo metrics
    metrics_results = epoch_metrics.result()
    keys = ['minADE', 'minFDE', 'minADE_vehicle', 'minFDE_vehicle', 'mAP', 'overlap_rate', 'miss_rate']
    for k in keys:
        output[k] = metrics_results[k]    
    if contrastive:
        contrastive_metrics_results = epoch_metrics_contrastive.result()
        for k in keys:
            output['c_'+k] = contrastive_metrics_results[k]

    ## Recall
    output['RGIF'] = hits_stats['rgif']/hits_stats['rgif_c'] if hits_stats['rgif_c']!=0 else 0
    output['any RGIF'] = hits_stats['any_rgif']/hits_stats['any_rgif_c'] if hits_stats['any_rgif_c']!=0 else 0
    output['RGIF pure'] = hits_stats['rgif']
    output['RGIF count'] = hits_stats['rgif_c']
    output[f'RGIF class weighted Avg'] = 0
    for i in range(len(hits_stats['rgif_class'])):
        output[f'RGIF class-{i}'] = hits_stats['rgif_class'][i]/hits_stats['rgif_class_c'][i] if hits_stats['rgif_class_c'][i]!=0 else 0
        output[f'RGIF class-{i} count'] = hits_stats['rgif_class_c'][i]
        output[f'RGIF class-{i} count perc'] = hits_stats['rgif_class_c'][i]/output['RGIF count'] if output['RGIF count']!=0 else 0
        output[f'RGIF class weighted Avg'] += output[f'RGIF class-{i}']*output[f'RGIF class-{i} count perc']
    class_avg_rgif = np.zeros(len(hits_stats['rgif_class']))
    for i in range(len(hits_stats['rgif_class'])):
        class_avg_rgif[i] = hits_stats['rgif_class'][i]/hits_stats['rgif_class_c'][i] if hits_stats['rgif_class_c'][i]!=0 else 0
    output[f'RGIF class Avg'] = np.average(class_avg_rgif)
    # output[f'RGIF class Avg'] = np.average([hits_stats['rgif_class'][i]/hits_stats['rgif_class_c'][i] for i in range(len(hits_stats['rgif_class']))])

    if contrastive:
        output['RCIF'] = hits_stats['rcif']/hits_stats['rcif_c'] if hits_stats['rcif_c']!=0 else 0
        output['any RCIF'] = hits_stats['any_rcif']/hits_stats['any_rcif_c'] if hits_stats['any_rcif_c']!=0 else 0
        output['RCIF pure'] = hits_stats['rcif']
        output['RCIF count'] = hits_stats['rcif_c']
        output[f'RCIF class weighted Avg'] = 0
        for i in range(len(hits_stats['rcif_class'])):
            output[f'RCIF class-{i}'] = hits_stats['rcif_class'][i]/hits_stats['rcif_class_c'][i] if hits_stats['rcif_class_c'][i]!=0 else 0
            output[f'RCIF class-{i} count'] = hits_stats['rcif_class_c'][i]
            output[f'RCIF class weighted Avg'] += output[f'RCIF class-{i}']*output[f'RGIF class-{i} count perc'] # we use the RGIC count perc, as it represent the percentage of test examples per class. Using the RCIF count perc could be misleading in the current implementation where contrastive examples of all classes are generated given one ground truth class
        class_avg_rcif = np.zeros(len(hits_stats['rcif_class']))
        for i in range(len(hits_stats['rcif_class'])):
            class_avg_rcif[i] = hits_stats['rcif_class'][i]/hits_stats['rcif_class_c'][i] if hits_stats['rcif_class_c'][i]!=0 else 0
        output[f'RCIF classAvg'] = np.average(class_avg_rcif)
        
    ## Diversity
    output[f'G_diversity'] = hits_stats['gdiversity']/hits_stats['gdiversity_c'] if hits_stats['gdiversity_c']!=0 else 0
    output[f'G_diversity pure'] = hits_stats['gdiversity']
    output[f'G_diversity count'] = hits_stats['gdiversity_c']
    if contrastive:
        output[f'C_diversity'] = hits_stats['cdiversity']/hits_stats['cdiversity_c'] if hits_stats['cdiversity_c']!=0 else 0
        output[f'C_diversity pure'] = hits_stats['cdiversity']
        output[f'C_diversity count'] = hits_stats['cdiversity_c']
    
    ## Categorical diversity
    g_categ_diversity = {i: hits_stats['g_categorical_diversity'][i]/sum(hits_stats['g_categorical_diversity'].values()) for i in range(1,7)}
    for i in g_categ_diversity.keys():
        output[f'G_diversity mode-{i} count'] = g_categ_diversity[i]
    if contrastive:
        c_categ_diversity = {i: hits_stats['c_categorical_diversity'][i]/sum(hits_stats['c_categorical_diversity'].values()) for i in range(1,7)}
        for i in c_categ_diversity.keys():
            output[f'C_diversity mode-{i} count'] = c_categ_diversity[i]
    
    num_classes = 8
    # correct_diversity = {i:hits_stats['gdiversity_class'][i]/hits_stats['gdiversity_class_c'][i] for i in range(num_classes)}
    correct_diversity ={i: hits_stats['gdiversity_class'][i]/hits_stats['gdiversity_class_c'][i] if hits_stats['gdiversity_class_c'][i] != 0 else 0 for i in range(num_classes)}
    for i in range(num_classes):
        output[f'class-{i} diversity (new)'] = correct_diversity[i]
    
    dict_to_csv(output, filename+f'main_metrics_{epoch}.csv')

    return output

def any_modal_follow_instruct_sum(target, pred):
    ## Calculates if any of the predicted modalities follow the instruction
    hits = pred == target[:,None]
    if len(hits.shape)>1:
        hits = hits.any(dim=-1)
    return hits

def modal_follow_instruct(target, pred):
    ## Calculates if any of the predicted modalities follow the instruction
    hits = pred == target[:,None]
    return hits
# def mode_modal_follow_instruct(target, preds):
    ## Hit if the mode modalities action follow the instruction


def unwrap_dist_model(model, distributed=False):
        if distributed:
            return model.module
        else:
            return model

def convert_cfg_to_wandb_cfg(input_dict):
    converted_dict = {}
    for key, value in input_dict.items():
        converted_dict[key] = {'value': value}
    return converted_dict

# define model validation epoch
def validation_epoch(valid_data, model, epoch, act=False, two_agent_act=False, save_dir="", num_classes=8, args=None, contrastive=False):
    navigation_extractor = futureNavigation(normalize_track=True)
    epoch_metrics = MotionMetrics()
    if act and contrastive:
        epoch_metrics_contrastive = MotionMetrics()

    model.eval()
    current = 0
    start_time = time.time()
    size = len(valid_data)
    epoch_loss = []

    local_rank = 0

    hits_stats = {
        'rgif':0, 'any_rgif':0, 'rgif_c':0, 'any_rgif_c':0, 'rgif_class':[0 for _ in range(num_classes)], 'rgif_class_c':[0 for _ in range(num_classes)], 
        'rcif':0, 'any_rcif':0, 'rcif_c':0, 'any_rcif_c':0, 'rcif_class':[0 for _ in range(num_classes)], 'rcif_class_c':[0 for _ in range(num_classes)], 
        'gdiversity':0, 'cdiversity':0, 'gdiversity_c':0, 'cdiversity_c':0,
        'gdiversity_class':[0 for _ in range(num_classes)], 'gdiversity_class_c':[0 for _ in range(num_classes)],
        'cdiversity_class':[0 for _ in range(num_classes)], 'cdiversity_class_c':[0 for _ in range(num_classes)],
        'g_categorical_diversity': {i+1:0 for i in range(args.modalities)},
        'c_categorical_diversity': {i+1:0 for i in range(args.modalities)},
        # 'any_modal_hits_gt':0, 'count_any_modal_hits_gt':0, 
        # 'any_modal_hits_atleast_one_contrastive':0, 'count_any_modal_hits_atleast_one_contrastive':0,
        }

    # hits_stats = {,
    #     'any_modal_hits':0, 'count_any_modal_hits':0, 
    #     'any_modal_hits_gt':0, 'count_any_modal_hits_gt':0, 
    #     'any_modal_hits_atleast_one_contrastive':0, 'count_any_modal_hits_atleast_one_contrastive':0,
    #     }

    for i, batch in enumerate(tqdm(valid_data)):
        if args.full_map:
            inputs = {
                'ego_state': batch[0].to(local_rank),
                'neighbors_state': batch[1].to(local_rank),
                'map_lanes': batch[2].to(local_rank),
                'map_crosswalks': batch[3].to(local_rank),
                'act': batch[7].to(local_rank),
                'additional_map_lanes': batch[8].to(local_rank),
                'additional_map_crosswalks': batch[9].to(local_rank),
                'additional_boundaries':batch[10].to(local_rank),
                'traffic_lights':batch[11].to(local_rank),
                'stop_signs':batch[12].to(local_rank),
                'speed_bumps':batch[13].to(local_rank),
            }   
            ego_future = batch[4].to(local_rank)
            neighbor_future = batch[5].to(local_rank)
        else:
            inputs = {
                'ego_state': batch[0].to(local_rank),
                'neighbors_state': batch[1].to(local_rank),
                'map_lanes': batch[2].to(local_rank),
                'map_crosswalks': batch[3].to(local_rank),
                'act':batch[7].to(local_rank),
            }   
            ego_future = batch[4].to(local_rank)
            neighbor_future = batch[5].to(local_rank)
        object_type = batch[6].to(local_rank) if torch.cuda.is_available() else batch[6]
        
        ## use car ego samples only
        car_mask = object_type[:,0]==1

        if -1 in inputs['act'][:,0]:
            valid_sample_mask = inputs['act'][:,0]!=-1
            inputs_ = {}
            for k in inputs.keys():
                inputs_[k] = inputs[k][valid_sample_mask]
            inputs = inputs_
            # inputs = {k: inputs[k][valid_sample_mask] for k in inputs}
            ego_future = ego_future[valid_sample_mask]
            neighbor_future = neighbor_future[valid_sample_mask]
            object_type = object_type[valid_sample_mask]
            car_mask = car_mask[valid_sample_mask]
        batch_size = len(inputs['act'])
        # if contrastive:
        #     contrastive_mapping = {class_i:[class_j for class_j in range(num_classes) if class_j!=class_i] for class_i in range(num_classes)}
        #     num_contrastive = num_classes-1
        #     samples__ = {k: inputs[k].detach().clone() for k in inputs}
        #     for contrastive_i in range(num_contrastive):
        #         samples_ = {k: samples__[k].detach().clone() for k in samples__}
        #         for instruct_i, instruct in enumerate(samples_['act']):
        #             samples_['act'][instruct_i][0] = contrastive_mapping[samples_['act'][instruct_i][0].int().cpu().item()][contrastive_i]
            
        #         for k in inputs.keys():
        #             if torch.is_tensor(inputs[k]):
        #                 inputs[k] = torch.cat((inputs[k], samples_[k]), dim=0)
        #             else:
        #                 inputs[k] = inputs[k] + samples_[k]

        #     ego_future = torch.cat([ego_future.detach().clone() for _ in range(num_contrastive+1)])
        #     neighbor_future = torch.cat([neighbor_future.detach().clone() for _ in range(num_contrastive+1)])
        #     object_type = torch.cat([object_type.detach().clone() for _ in range(num_contrastive+1)])

        # query the model
        with torch.no_grad():
            outputs = model(inputs)
            loss,future, ego_multimodal = level_k_loss(outputs, ego_future, neighbor_future, args.level, gmm=True, subsample=args.subsample)
        
        if ego_multimodal[1] is not None:
            ego_multimodal = torch.stack(ego_multimodal, dim=1) # two agents, shape = [batch, 2, modalities, T_out, 2]
        else:
            ego_multimodal = ego_multimodal[0].unsqueeze(1) # one agent agents, shape = [batch, 1, modalities, T_out, 2]
        
        future = ego_multimodal.cpu().detach().clone()
        agent_i=0
        future = future[:,agent_i]
        output_act = []
        for sample_i in range(ego_multimodal.shape[0]):
            output_act_ = []
            for modal_i in range(ego_multimodal.shape[2]):
                valid_mask = future[sample_i, modal_i, :, 0]!=0
                output_act_.append(navigation_extractor.get_navigation_dict(future[sample_i, modal_i, :, :2])['direction 0.1to8_cls'])
            output_act.append(output_act_)
        output_act = torch.tensor(output_act, device=inputs['act'].device).unsqueeze(2)

        viz = args.viz
        if viz:
            output_figs = {}
            # for batch_i_sample in range(ego_multimodal.shape[0]):
            batch_i_sample = 1
            output_figs[batch_i_sample] = viz_multimodal(inputs, ego_multimodal[:,0], ego_future, batch_i_sample)
            # output_figs[batch_i_sample].savefig('ex.png')
            plt.xlim(-5,90)
            plt.ylim(-15,15)
            output_figs[batch_i_sample].savefig('ex.png', dpi=300)
            # plt.ylim(-50,50)
            # output_figs[batch_i_sample] = viz_multimodal2(inputs, ego_multimodal, ego_future, neighbor_future, batch_i_sample)
            # output_figs[batch_i_sample].savefig('ex2.png')
            print(f"Instruct: {inputs['act'][batch_i_sample]}")
            print(f"Scores: {future[-1][:,1][batch_i_sample].softmax(-1).cpu().numpy()*100}%")
            print('')
            # batch_i_sample_ = batch_i_sample + int(ego_multimodal.shape[0]/2)
            batch_i_sample_ = batch_i_sample + int(ego_multimodal.shape[0]/num_classes)
            output_figs[batch_i_sample_] = viz_multimodal(inputs, ego_multimodal[:,0], ego_future, batch_i_sample_)
            output_figs[batch_i_sample_].savefig('ex_.png')
            # output_figs[batch_i_sample_] = viz_multimodal2(inputs, ego_multimodal, ego_future, neighbor_future, batch_i_sample_)
            # output_figs[batch_i_sample_].savefig('ex2_.png')
            print(f"Instruct: {inputs['act'][batch_i_sample_]}")
            print(f"Scores: {future[-1][:,1][batch_i_sample_].softmax(-1).cpu().numpy()*100}%")
            print('')
        else:
            # else:
            # compute metrics
            epoch_loss.append(loss.item())
            egos = outputs[f'level_{args.level}_interactions'][:, :, :, :, :2]
            scores = outputs[f'level_{args.level}_scores']
        
            
            ego = inputs['ego_state']
            actors = torch.stack([ego,inputs['neighbors_state'][:, 0]],dim=1)
            actors_future = torch.stack([ego_future, neighbor_future],dim=1)
            ego_ground_truth = torch.cat([actors[:, :, :, :5], actors_future], dim=2)
            ego_ground_truth = torch.cat([
                ego_ground_truth[:, :, :, :2], 
                actors[:,:, -1, 5:7].unsqueeze(2).expand(-1,-1, ego_ground_truth.shape[2], -1), 
                ego_ground_truth[:, :, :, 2:]
                ], dim=-1)
            
            egos = egos.permute(0,2,1,3,4)
            scores = scores.sum(1)
            scores = F.softmax(scores,dim=-1)
            

            # GT instruct following metric
            valid_gt = torch.ne(ego_ground_truth[:batch_size], 0).bool()[:,0,:11].any(-1).any(-1)
            
            input_act_cpu = inputs['act'][:batch_size,0][valid_gt].cpu()
            output_act_cpu = output_act[:batch_size,...,0][valid_gt].cpu()
            modal_hits = output_act_cpu == inputs['act'][:batch_size,0][valid_gt].cpu()[:, None]
            
            hits_stats['rgif']+= modal_hits.sum().item()
            hits_stats['rgif_c']+= int(modal_hits.shape[0]*modal_hits.shape[1]) # batch_size = modal_hits.shape[0] if all valid
            
            hits_stats['any_rgif']+= modal_hits.any(dim=-1).sum().item()
            hits_stats['any_rgif_c']+= modal_hits.shape[0] # batch_size = modal_hits.shape[0] if all valid
            
            for ii in range(num_classes):
                class_mask = inputs['act'][:batch_size,0][valid_gt].cpu() == ii
                num_class_examples = sum(class_mask).item()
                class_modal_hits = output_act_cpu[class_mask,:] == input_act_cpu[class_mask, None]
                hits_stats['rgif_class'][ii] += class_modal_hits.sum().item()
                hits_stats['rgif_class_c'][ii] += num_class_examples*class_modal_hits.shape[1]

                class_diversity = [len(np.unique(output_act_cpu[class_mask][jj]))/output_act_cpu[class_mask].shape[1] for jj in range(len(output_act_cpu[class_mask]))]
                hits_stats[f'gdiversity_class'][ii] += sum(class_diversity)
                hits_stats[f'gdiversity_class_c'][ii] += len(class_diversity)
            
            ## mode class diversity
            diversity, diversity_c = diversity_metric(output_act_cpu)
            hits_stats['gdiversity'] += diversity
            hits_stats['gdiversity_c'] += diversity_c
            ## categorical class diversity, num_categ = num_modalities
            categorical_diversity = categorical_diversity_metric(output_act_cpu)
            hits_stats['g_categorical_diversity'] = {k: hits_stats['g_categorical_diversity'][k] + categorical_diversity[k] for k in hits_stats['g_categorical_diversity'].keys()}
            
            epoch_metrics.update_state(
                prediction_trajectory=egos[:batch_size], 
                prediction_score=scores[:batch_size], 
                ground_truth_trajectory=ego_ground_truth[:batch_size], 
                ground_truth_is_valid=torch.ne(ego_ground_truth[:batch_size], 0).bool(), 
                object_type=object_type[:batch_size].long()
                )
            if contrastive:
                # contrastive instruct following metric
                # valid_gt = torch.ne(ego_ground_truth[:batch_size], 0).bool()[:,0,:11].any(-1).any(-1)
                valid_act = torch.ne(ego_ground_truth[batch_size:], 0).bool()[:,0,:11].any(-1).any(-1)
                # valid_all = torch.ne(ego_ground_truth, 0).bool()[:,0,:11].any(-1).any(-1)
                
                valid_output_act_cpu = output_act[batch_size:,...,0][valid_act].cpu()
                valid_input_act_cpu = inputs['act'][batch_size:,0][valid_act].cpu()
                non_gt_modal_hits = valid_output_act_cpu == valid_input_act_cpu[:, None]
                
                # if True:
                hits_stats['rcif']+= non_gt_modal_hits.sum().item()
                hits_stats['rcif_c']+= int(non_gt_modal_hits.shape[0]*non_gt_modal_hits.shape[1])

                hits_stats['any_rcif']+= non_gt_modal_hits.any(dim=-1).sum().item()
                hits_stats['any_rcif_c']+= non_gt_modal_hits.shape[0] # batch_size = modal_hits.shape[0] if all valid
                
                for ii in range(num_classes):
                    class_mask = valid_input_act_cpu == ii
                    num_class_examples = sum(class_mask).item()
                    class_modal_hits = valid_output_act_cpu[class_mask,:] == valid_input_act_cpu[class_mask, None]
                    hits_stats['rcif_class'][ii] += class_modal_hits.sum().item()
                    hits_stats['rcif_class_c'][ii] += num_class_examples*class_modal_hits.shape[1]
                
                ## mode class diversity
                diversity, diversity_c = diversity_metric(valid_output_act_cpu)
                hits_stats['cdiversity'] += diversity
                hits_stats['cdiversity_c'] += diversity_c
                ## categorical class diversity, num_categ = num_modalities
                categorical_diversity = categorical_diversity_metric(valid_output_act_cpu)
                hits_stats['c_categorical_diversity'] = {k: hits_stats['c_categorical_diversity'][k] + categorical_diversity[k] for k in hits_stats['c_categorical_diversity'].keys()}
            
                epoch_metrics_contrastive.update_state(
                    prediction_trajectory=egos[batch_size:], 
                    prediction_score=scores[batch_size:], 
                    ground_truth_trajectory=ego_ground_truth[batch_size:], 
                    ground_truth_is_valid=torch.ne(ego_ground_truth[batch_size:], 0).bool(), 
                    object_type=object_type[batch_size:].long()
                    )

    save_please = True if not args.viz else False
    # if save_please:
        # save_metrics_to_file(epoch_metrics, epoch_metrics_contrastive= None, hits_stats=hits_stats, filename=save_dir, act=act, epoch=epoch)
        # save_metrics_to_csv(epoch_metrics, epoch_metrics_contrastive=None, hits_stats=hits_stats, filename=save_dir, act=act, epoch=epoch, model_name=args.load_dir.split('/')[-2])
    save_metrics_to_file(epoch_metrics, epoch_metrics_contrastive=epoch_metrics_contrastive if contrastive else None, hits_stats=hits_stats, filename=save_dir, act=act, epoch=epoch)
    output_metrics = save_metrics_to_csv(epoch_metrics, epoch_metrics_contrastive=epoch_metrics_contrastive if contrastive else None, hits_stats=hits_stats, filename=save_dir, act=act, epoch=epoch, model_name=args.load_dir.split('/')[-2] if len(args.load_dir)>0 else '')

    # ['minADE'] ['minFDE'] ['RGIF class weighted Avg']
    return output_metrics

# define model validation epoch
def validation_epoch_pkl(valid_data, model, epoch, act=False, two_agent_act=False, save_dir="", num_classes=8, args=None, contrastive=False, pos1_synth=False):
    if pos1_synth:
        navigation_extractor = futureNavigation(normalize_track=True, num_classes=num_classes)
        get_act_util_ = get_act_util()
    model.eval()
    current = 0
    start_time = time.time()
    size = len(valid_data)

    # # local_rank = 0
    # local_rank = int(os.environ["LOCAL_RANK"])
    # Check if LOCAL_RANK is in the environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    files_saved = []
    tt=[]
    for i, batch in enumerate(tqdm(valid_data)):
        if args.full_map:
            inputs = {
                'ego_state': batch[0].to(local_rank),
                'neighbors_state': batch[1].to(local_rank),
                'map_lanes': batch[2].to(local_rank),
                'map_crosswalks': batch[3].to(local_rank),
                'act': batch[7].to(local_rank),
                'additional_map_lanes': batch[8].to(local_rank),
                'additional_map_crosswalks': batch[9].to(local_rank),
                'additional_boundaries':batch[10].to(local_rank),
                'traffic_lights':batch[11].to(local_rank),
                'stop_signs':batch[12].to(local_rank),
                'speed_bumps':batch[13].to(local_rank),
            }   
            ego_future = batch[4].to(local_rank)
            neighbor_future = batch[5].to(local_rank)
            filenames = batch[14]
        else:
            inputs = {
                'ego_state': batch[0].to(local_rank),
                'neighbors_state': batch[1].to(local_rank),
                'map_lanes': batch[2].to(local_rank),
                'map_crosswalks': batch[3].to(local_rank),
                'act':batch[7].to(local_rank),
            }   
            ego_future = batch[4].to(local_rank)
            neighbor_future = batch[5].to(local_rank)
            filenames = batch[8]
        object_type = batch[6].to(local_rank) if torch.cuda.is_available() else batch[6]
        
        
        with torch.no_grad():
            t1 = time.time()
            outputs = model(inputs)
            t2 = time.time() - t1
            tt.append(t2)
            loss, future, trajectories = level_k_loss_trajgpt(outputs, ego_future, neighbor_future, args.level, gmm=True, subsample=args.subsample)
            # trajectories = ego_multimodal
            # import pdb; pdb.set_trace()
            level_keys = [key_i for key_i in outputs.keys() if '_' in key_i and key_i.split('_')[1].isdigit()]
            level = max([int(key.split('_')[1]) for key in level_keys])
            scores = outputs[f'level_{level}_scores']

        to_save_dict = {'gf_loss': loss,'output_traj': future[0], 'modalities': trajectories, 'scores': scores}
        valid_output = to_save_dict
        output_dir_ = save_dir
        # Save dictionary using pickle
        # filenames = []
        if pos1_synth:
            # found_ = 0
            # total_ = 0
            synth_pos_future = torch.zeros_like(future[0])

            # 🔹 Convert all tensors to CPU first to minimize multiple `.cpu()` calls
            # trajectories_cpu = [t[0].cpu() for t in trajectories]  # Move entire batch to CPU
            trajectories_cpu = trajectories.cpu()
            acts_cpu = inputs["act"].cpu()  # Move acts to CPU
            filenames_cpu = filenames  # No need to convert, already list of strings

            # 🔹 Pre-generate static template strings
            template_instruct = "You are generating the future motion plan for the ego vehicle. Reason about and predict the future multimodal trajectory embeddings of two agents—the ego (Agent-1) and another agent it interacts with (Agent-2) —based on the observed scene embeddings and the following ego instruction: Make the ego vehicle "

            for sample_i, (act, filename) in enumerate(zip(acts_cpu, filenames_cpu)):
                # total_ += 1
                valid_generation = False

                for traj_sample_i in trajectories_cpu[sample_i][0]:
                    try:
                        act_generated = get_act_util_.get_act(history=batch[0][sample_i][:,:5].cpu().numpy(), future=traj_sample_i.cpu().numpy(), skip_future_start_end_with_zeros=False, not_skip_invalid_history=False)['Agent-1']['direction 0.1to8_cls']
                    except:
                        act_generated = -1 # couldn't calculate act
                    agent = navigation_extractor.get_navigation_dict(traj_sample_i)
                    # if agent['direction 0.1to8_cls'] == act[0]:  # ✅ Direct comparison without `.cpu()`
                    # if agent['direction 0.1to8_cls'] == act_generated:
                    #     print('')
                    # if act_generated==0:
                    #     print('')
                    if act_generated == act[0]:  # ✅ Direct comparison without `.cpu()`
                        valid_generation = True
                        # found_ += 1
                        break  # ✅ Exit early to save computation time

                if valid_generation:
                    # ✅ String formatting optimized
                    direction = "stay stationary" if "stationary" in ['direction 0.1to8'] else agent['direction 0.1to8']
                    direction1 = "stay stationary" if "stationary" in agent['direction 0.1to4'] else agent['direction 0.1to4']
                    direction2 = "stay stationary" if "stationary" in agent['direction 4to8'] else agent['direction 4to8']
                    speed1 = agent['speed 0.1to4']
                    speed2 = agent['speed 4to8']
                    speed1 = f" with a {speed1} speed" if ("stationary" != speed1 and "INVALID" != speed1) else ""
                    speed2 = f" with a {speed2} speed" if ("stationary" != speed2 and "INVALID" != speed2) else ""
                    accel1 = f" and a {agent['acceleration 0.1to4']}" if agent['acceleration 0.1to4'] != "stationary" else ""
                    accel2 = f" and a {agent['acceleration 4to8']}" if agent['acceleration 4to8'] != "stationary" else ""

                    instruct1 = f"{template_instruct}{direction}."
                    instruct2 = f"{template_instruct}{direction}, where it will first {direction1}{speed1}, then {direction2}{speed2}."
                    instruct3 = f"{template_instruct}{direction}, where it will first {direction1}, then {direction2}."

                    action_reason = f"The ego vehicle can {direction}, where it will first {direction1}{speed1}{accel1}, then {direction2}{speed2}{accel2}."
                    reason = action_reason

                    json_data = [
                        {"Instruction": instruct1, "Reasoning": reason, "Decision": "<Accepted>", "Label": "pos", "Direction_cls": agent['direction 0.1to8_cls']},
                        {"Instruction": instruct2, "Reasoning": reason, "Decision": "<Accepted>", "Label": "pos", "Direction_cls": agent['direction 0.1to8_cls']},
                        {"Instruction": instruct3, "Reasoning": reason, "Decision": "<Accepted>", "Label": "pos", "Direction_cls": agent['direction 0.1to8_cls']}
                    ]

                    llm_json = "\n".join([json.dumps(j) for j in json_data])  # ✅ Use `json.dumps()` for proper formatting

                    # 🔹 Efficient file writing (fewer disk writes)
                    # templateLLM_filename = f"<internal_dataset_root>/waymo/gameformer/training_28nov_synth_templateLLM/{filename}.txt"
                    templateLLM_filename = f"<internal_dataset_root>/waymo/gameformer/feb16_2025/training_synth_templateLLM/{filename}.txt"
                    with open(templateLLM_filename, 'w') as file:
                        file.write(llm_json)

                    # 🔹 Efficiently save `.npz` file (no `.cpu()` calls inside loop)
                    # synth_traj_filename = f"<internal_dataset_root>/waymo/gameformer/training_28nov_synth_npz/{filename}.npz"
                    synth_traj_filename = f"<internal_dataset_root>/waymo/gameformer/feb16_2025/training_synth_npz/{filename}.npz"
                    np.savez_compressed(synth_traj_filename, synth_traj=traj_sample_i.numpy()) 
        # if pos1_synth:
        #     found_=0
        #     total_=0
        #     synth_pos_future = torch.zeros_like(future[0])
        #     for sample_i in range(len(filenames)):
        #         total_+=1
        #         valid_generation=False
        #         for traj_sample_i in trajectories[sample_i][0]:
        #             agent = navigation_extractor.get_navigation_dict(traj_sample_i.cpu())
        #             if agent['direction 0.1to8_cls'] == inputs["act"][sample_i][0]:
        #                 valid_generation = True
        #                 found_+=1
        #                 break
        #         if valid_generation:
        #             template_instruct = "You are generating the future motion plan for the ego vehicle. Reason about and predict the future multimodal trajectory embeddings of two agents—the ego (Agent-1) and another agent it interacts with (Agent-2) —based on the observed scene embeddings and the following ego instruction: Make the ego vehicle "
        #             direction = "stay stationary" if "stationary" in ['direction 0.1to8'] else agent['direction 0.1to8']
        #             direction1 = "stay stationary" if "stationary" in agent['direction 0.1to4'] else agent['direction 0.1to4']
        #             direction2 = "stay stationary" if "stationary" in agent['direction 4to8'] else agent['direction 4to8']
        #             speed1 = agent['speed 0.1to4']
        #             speed2 = agent['speed 4to8']
        #             speed1 = f" with a {speed1} speed" if ("stationary" != speed1 and "INVALID" != speed1) else ""
        #             speed2 = f" with a {speed2} speed" if ("stationary" != speed2 and "INVALID" != speed2) else ""
        #             accel1 = agent['acceleration 0.1to4']
        #             accel1 = f" and a {accel1}" if ("stationary" != accel1) else ""
        #             accel2 = agent['acceleration 4to8']
        #             accel2 = f" and a {accel2}" if ("stationary" != accel2) else ""
        #             instruct1 = template_instruct + f"{direction}."
        #             instruct2 = template_instruct + f"{direction}, where it will first {direction1}{speed1}, then {direction2}{speed2}."
        #             instruct3 = template_instruct + f"{direction}, where it will first {direction1}, then {direction2}."
        #             action_reason = f"The ego vehicle can {direction}, where it will first {direction1}{speed1}{accel1}, then {direction2}{speed2}{accel2}."
        #             reason = action_reason
        #             agent_type = ['Unknown', 'Vehicle', 'Pedestrian', 'Cyclist']
        #             xx = [str({"Instruction": instruct1, "Reasoning": reason, "Decision":"<Accepted>", "Label":"pos", "Direction_cls": agent['direction 0.1to8_cls']}).replace("'", '"')+"\n",
        #             str({"Instruction": instruct2, "Reasoning": reason, "Decision":"<Accepted>", "Label":"pos", "Direction_cls": agent['direction 0.1to8_cls']}).replace("'", '"')+"\n",
        #             str({"Instruction": instruct3, "Reasoning": reason, "Decision":"<Accepted>", "Label":"pos", "Direction_cls": agent['direction 0.1to8_cls']}).replace("'", '"')+"\n",]
        #             llm_json = "".join(xx)
        #             templateLLM_filename = f"<internal_dataset_root>/waymo/gameformer/training_28nov_synth_templateLLM/{filenames[sample_i]}.txt"
        #             with open(templateLLM_filename, 'w') as file:
        #                 file.write(llm_json)
        #             synth_traj_filename = f"<internal_dataset_root>/waymo/gameformer/training_28nov_synth_npz/{filenames[sample_i]}.npz"
        #             np.savez_compressed(synth_traj_filename, synth_traj=np.array(traj_sample_i.cpu()))
        #         else:
        #             continue
            
        #     # print(found_/total_)

        else:
            for i_save in range(len(filenames)):
                sample_valid_output = {k:v[i_save].to(torch.float16).cpu().numpy() for k,v in valid_output.items() if k!='gf_loss' and k!= 'text'}
                sample_valid_output.update({'loss': valid_output['gf_loss'].cpu()})
                # sample_valid_output.update({'text': valid_output['text'][i_save]})
                sample_valid_output.update({'act': inputs['act'][i_save,0].cpu()})
                sample_valid_output.update({'act_2': inputs['act'][i_save,1].cpu()})
                with open(f"{output_dir_}/{filenames[i_save]}.pkl", 'wb') as f:
                    pickle.dump(sample_valid_output, f)
                files_saved.append(filenames[i_save])
    # Nussair_WACV
    if torch.cuda.is_available():
        peak_alloc_mb    = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved()  / (1024**2)
    print(f"[EVAL] peak_mem_alloc={peak_alloc_mb:.2f} MB | peak_mem_reserved={peak_reserved_mb:.2f} MB")
    print(f"Forward-Pass Latency: {np.mean(tt) * 1000:.2f} ms")
    return {}



# Define model training process
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_logger()
    wandb_cfg = {**convert_cfg_to_wandb_cfg(args.__dict__)}
    if args.wandb and get_rank()==0:
        wandb.login()
        wandb.init(project="trajgpt", name=args.name, config=wandb_cfg)

    if args.save_results and get_rank()==0:
        log_path = f"{args.save_results}/"
        os.makedirs(log_path, exist_ok=True)

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    
    set_seed(args.seed)

    model = GameFormer_2(
            modalities=args.modalities,
            encoder_layers=args.encoder_layers,
            decoder_levels=args.level,
            future_len=args.future_len, 
            neighbors_to_predict=args.neighbors_to_predict,
            act = args.act,
            act_dec = args.act_dec,
            full_map=True,
            # shared_act=args.shared_act,
            # num_act_classes=args.num_act_classes,
            )

    if args.load_dir != '':
        model_path = args.load_dir
        model_ckpts = torch.load(model_path, map_location='cpu')
        msg = model.load_state_dict(model_ckpts['model_states'], strict=False)
        logging.info(msg)
    

    model = model.to(device)
    
    # datasets:
    # train_dataset = DrivingData(args.train_set+'/*')
    valid_dataset = DrivingData(args.valid_set+'/*', act=True, full_map=args.full_map)
    # print(f'found {len(train_dataset)} training data')
    print(f'found {len(valid_dataset)} validation data')

    # training_size = len(train_dataset)
    valid_size = len(valid_dataset)
    # logging.info(f'Length train: {training_size}; Valid: {valid_size}')
    logging.info(f'Valid: {valid_size}')


    # train_data = DataLoader(
    #     train_dataset, batch_size=args.batch_size,  num_workers=args.workers)
    valid_data = DataLoader(
        valid_dataset, batch_size=args.batch_size,num_workers=args.workers)
    with torch.no_grad():
        validation_epoch(valid_data, model, 30, args.act or args.act_dec, args.two_agent_act,  save_dir="/".join(model_path.split('/')[:-1])+"/", num_classes=args.num_act_classes, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interaction Prediction Training')
    parser.add_argument("--model", default='gameformer')
    parser.add_argument("--batch_size", type=int, help='training batch sizes', default=32)
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)

    parser.add_argument('--load_dir', type=str, help='', default='<internal_user_root>/gameformer_models/gf_5may_fullmap/epochs_last.pth')    
    parser.add_argument('--train_set', type=str, help='path to train data', default='')
    parser.add_argument('--valid_set', type=str, help='path to validation data', default='<internal_dataset_root>/waymo/gameformer/validation_11may_fullmap_val')
    parser.add_argument("--workers", type=int, default=4, help="number of workers used for dataloader")
    parser.add_argument("--level", type=int, help='decoder reasoning levels (K)', default=3) # default: 6
    parser.add_argument("--neighbors_to_predict", type=int, help='neighbors to predict, 1 for Waymo Joint Prediction', default=1)
    parser.add_argument("--modalities", type=int, help='joint num of modalities', default=6)
    parser.add_argument("--future_len", type=int, help='prediction horizons', default=80) # 16 or 80
    parser.add_argument("--encoder_layers", type=int, help='encoder layers', default=6)
    parser.add_argument("--wandb", action="store_true", help='distributed mode', default=False)
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="gameformer_temp")
    parser.add_argument('--subsample', type=bool, help='2Hz if True, 10Hz if false', default=False)
    parser.add_argument("--eval_only", action="store_true", help='', default=True)
    parser.add_argument("--act", action="store_true", help='act', default=False)
    parser.add_argument("--act_dec", action="store_true", help='act', default=True)
    parser.add_argument("--shared_act", action="store_true", help='', default=False)
    parser.add_argument("--two_agent_act", action="store_true", help='', default=False)
    parser.add_argument("--save_results", help='', default='')
    parser.add_argument("--num_act_classes", type=int, help='', default=8)
    parser.add_argument("--viz", default=False)
    parser.add_argument("--full_map", default=True)
    
    
    args = parser.parse_args()
    print(args.load_dir.split('/')[-2])
    main()


