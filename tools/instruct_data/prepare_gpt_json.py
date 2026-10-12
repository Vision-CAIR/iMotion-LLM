import pickle
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
from gameformer.utils.inter_pred_utils import *
from gameformer.utils.data_utils import *
import time
from scipy import stats
import csv
from tqdm import tqdm
import os
from instructions.extract_instructions import futureNavigation
from instructions.direction_instructions import DirectionClassifier

# Set IMOTION_VAL_SPLIT_DIR to the validation split produced by preprocessing.
root_data_dir = os.environ.get(
    'IMOTION_VAL_SPLIT_DIR',
    'data/processed/waymo/gameformer/val_splits/split_1',
)



direction_classifier = DirectionClassifier()
for data_i, data_dir in enumerate(data_dirs):
    
    num_classes=8
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
        'num_examples':0
        }

    epoch_metrics = MotionMetrics()

    # List all pickle files in the data directory
    pickle_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    np_files = [f for f in os.listdir(root_data_dir) if f.endswith('.npz')]

    # Initialize metrics and statistics
    epoch_loss = []
    size = len(pickle_files)
    print(f"Processing {size} validation samples...")
    print('###'*30)
    # Loop over all pickle files
    # for filename in tqdm(pickle_files):
    # for filename in tqdm(pickle_files):
    for filename in pickle_files:
        file_path = os.path.join(data_dir, filename)
        
        # Load data from pickle file
        with open(file_path, 'rb') as file:
            out_data = pickle.load(file)

        act = out_data['act'].item()
        if act==-1:
            continue

        filename = file_path.split('/')[-1].replace('.pkl','')
        
        if filename+'.npz' in np_files:
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
        epoch_metrics.update_state(
            prediction_trajectory=egos, 
            prediction_score=scores,  
            ground_truth_trajectory=ego_ground_truth, 
            ground_truth_is_valid=torch.ne(ego_ground_truth, 0).bool(), 
            object_type=object_type.long()
            )

        

        ego_multimodal = torch.tensor(out_data['modalities']).to(torch.float32)[0]
        output_act = []
        for modal_i in range(ego_multimodal.shape[0]):
            instructions_out = navigation_extractor.get_navigation_dict(track=ego_multimodal[modal_i])
            output_act.append(instructions_out['direction 0.1to8_cls'])
        output_act = torch.tensor(output_act)

        modal_hits = output_act == act
        
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

        # template_dir = 'data/processed/waymo/gameformer/validation_16may_fullmap_wPlausbility_val_templateLLM/dba89f922ecb7de6_1_2_interest.txt'
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

        

    print("***"*30)
    print("***"*5)
    print(f"*** {models_names_and_description[data_i]}")
    print(f"* Directory: {data_dir}")
    

    metrics_results = epoch_metrics.result()
    if print_all_waymo_metrics:
        print('---'*15)
        print('---'*15)
        print('---'*3+f' All Waymo Metrics >>>')
        for k,v in metrics_results.items():
            print(f"> {k}: {v:.4f}")
            
    print('---'*15)
    print('---'*15)
    print('---'*3+f' Selected Waymo Metrics >>>')
    print(f">> Number of evaluation examples {hits_stats['num_examples']}")
    print(f">> minADE: {metrics_results['minADE']:.4f}")
    print(f">> minADE_vehicle: {metrics_results['minADE_vehicle']:.4f}")
    print(f">> minFDE: {metrics_results['minFDE']:.4f}")
    print(f">> minFDE_vehicle: {metrics_results['minFDE_vehicle']:.4f}")

    print('---'*15)
    print('---'*15)
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

    print('---'*15)
    print('---'*15)
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
    
    print('###'*30)
    print()
    print()
    print()