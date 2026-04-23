import os
import torch
import logging
import glob
import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from google.protobuf import text_format
import pickle
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
from instructions.direction_instructions import DirectionClassifier
import ast

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import json
import matplotlib.pyplot as plt



def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DrivingData_temp(Dataset):
    def __init__(self, data_dir, act=False, full_map=False, subset_len=-1, contrastive=False, positive_notgt=False):
        self.data_list = glob.glob(data_dir)
        if subset_len!=-1:
            self.data_list = self.data_list[:subset_len]
        self.act = act
        if act:
            self.mapJsons_dir = f"{data_dir[:-2]}_mapJsons"
            self.mapJsons_list = glob.glob(f"{self.mapJsons_dir}/*")
            self.agentJsons_dir = f"{data_dir[:-2]}_agentJsons"
            self.agentJsons_list = glob.glob(f"{self.agentJsons_dir}/*")
        self.full_map = full_map

        self.positive_notgt=positive_notgt
        self.contrastive=contrastive
        train_contrastive_act_distribution = np.array([18.23, 2.7, 3.88, 3.91, 8.36, 16.04, 23.75, 23.12])
        train_positive_notgt_distribution = np.array([7.04, 6.76, 15.34, 9.14, 39.00, 13.54, 4.32, 4.85])
        train_groundtruth_act_distribution = [1.62, 55.77, 3.25, 3.71, 16.67, 17.49, 0.09, 1.40]
        self.train_contrastive_prob = [0.7, 0.3]
        self.train_prob = [0.7, 0.3] if self.contrastive else [1.0,0.0]
        self.contrastive_prob = (1/train_contrastive_act_distribution)/(sum(1/train_contrastive_act_distribution))
        self.positive_prob = (1/train_positive_notgt_distribution)/(sum(1/train_positive_notgt_distribution))
        direction_classifier = DirectionClassifier()
        self.direction_classes = np.array(direction_classifier.classes)

        # self.mapJsons_list = set()
        # self.agentJsons_list = set()
        # self.preload_json_files()

    def preload_json_files(self):
        # Preload map JSON files
        for file_name in self.data_list:
            base_name = file_name.split('/')[-1].split('.')[0]
            map_json_path = f"{self.mapJsons_dir}/map_{base_name}.json"
            agent_json_path = f"{self.agentJsons_dir}/agent_{base_name}.json"
            try:
                with open(map_json_path, 'r') as file:
                    self.mapJsons_list.add(map_json_path)
                with open(agent_json_path, 'r') as file:
                    self.agentJsons_list.add(agent_json_path)
            except FileNotFoundError:
                continue
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        return data

class DrivingData(Dataset):
    def __init__(self, data_dir, act=False, full_map=False, subset_len=-1, contrastive=False, positive_notgt=False, ego_act_only=True, new_eval=False, new_eval_mode='', nuplan=False, random_drop_act=False, files_with_act_only=False, num_classes=8):
        self.random_drop_act = random_drop_act
        self.nuplan = nuplan
        if nuplan:
            self.data_list_ = []
            data_list_ = glob.glob(data_dir)
            for data_i in data_list_:
                # get the data inside f{'{data_i}/npz/*'} using glob.glob, and append it to self.data_list
                self.data_list_ += glob.glob(f'{data_i}/npz/*')
            
            
            self.act = act
            
            if act or new_eval_mode is not None or new_eval_mode!='':
                act_dict = {}
                self.data_list = []
                str_cls_map = {'remain stationary': 0, 'stop': 0, 'move straight': 1, 'turn right': 2, 'turn right and stop': 2, 'turn left': 3, 'turn left and stop': 3}
                for data_i in self.data_list_:
                    # Read the text file
                    # Read and parse into a dict
                    try:
                        with open(data_i.replace('.npz', '.txt').replace('npz','gpt_meta_prompts'), "r") as f:
                            data_ = json.load(f)  # directly converts JSON string to Python dict
                            act_dict[data_i.split('/')[-1]] = data_['trajectory_description'].split(';')[0].split(',')[0].split(' veering')[0].split(' with')[0]
                            act_dict[data_i.split('/')[-1]] = str_cls_map[act_dict[data_i.split('/')[-1]]]
                        self.data_list += [data_i]
                    except:
                        pass
                self.acts_dict = act_dict
            else:
                self.data_list = self.data_list_
            print(f"Found train data {len(self.data_list)}")
            

        else:
            self.data_list = glob.glob(data_dir)
            print(f"Found train data {len(self.data_list)}")
            if subset_len!=-1:
                self.data_list = self.data_list[:subset_len]
            self.act = act
            if act or new_eval_mode is not None or new_eval_mode!='':
                self.acts_dir = f"{data_dir[:-2]}_acts"
                self.acts_list = glob.glob(f"{self.acts_dir}/*")
                print(f"Found act train data {len(self.acts_list)}")
                if files_with_act_only:
                    # Create a set for faster lookups
                    acts_set = {i.split('/')[-1].replace('.pkl', '.npz') for i in self.acts_list}
                    # Use a list comprehension with the set
                    npz_with_found_acts = [i for i in self.data_list if i.split('/')[-1] in acts_set]
                    self.data_list = npz_with_found_acts
                
                # acts_list_temp = [i.split('/')[-1].replace('.pkl', '.npz') for i in self.acts_list]
                # npz_with_found_acts = [i for i in self.data_list if i.split('/') in acts_list_temp]
                
                # self.mapJsons_dir = f"{data_dir[:-2]}_mapJsons"
                # self.mapJsons_list = glob.glob(f"{self.mapJsons_dir}/*")
                # self.agentJsons_dir = f"{data_dir[:-2]}_agentJsons"
                # self.agentJsons_list = glob.glob(f"{self.agentJsons_dir}/*")
        self.full_map = full_map
        self.positive_notgt=positive_notgt
        self.contrastive=contrastive
        train_contrastive_act_distribution = np.array([18.23, 2.7, 3.88, 3.91, 8.36, 16.04, 23.75, 23.12])
        train_positive_notgt_distribution = np.array([7.04, 6.76, 15.34, 9.14, 39.00, 13.54, 4.32, 4.85])
        train_groundtruth_act_distribution = [1.62, 55.77, 3.25, 3.71, 16.67, 17.49, 0.09, 1.40]
        self.train_contrastive_prob = [0.7, 0.3]
        self.train_prob = [0.7, 0.3] if self.contrastive else [1.0,0.0]
        self.contrastive_prob = (1/train_contrastive_act_distribution)/(sum(1/train_contrastive_act_distribution))
        self.positive_prob = (1/train_positive_notgt_distribution)/(sum(1/train_positive_notgt_distribution))
        direction_classifier = DirectionClassifier(num_classes=num_classes)
        self.direction_classes = np.array(direction_classifier.classes)

        self.ego_act_only = ego_act_only
        # self.mapJsons_list = set()
        # self.agentJsons_list = set()
        # self.preload_json_files()
        self.new_eval = new_eval
        self.new_eval_mode = new_eval_mode
        if new_eval:
            meta_ = '/'.join(data_dir.split('/')[:-1]) + f"_eval_meta/meta_{new_eval_mode}.json"
            with open(meta_, "r") as f:
                meta_ = f.read()
            self.meta_eval = json.loads(meta_)

            self.new_eval_mode = new_eval_mode #'traj_pred', 'gt1', 'pos1', 'neg1',...
            if self.new_eval_mode=='traj_pred':
                self.data_list = [data_i for data_i in self.data_list if data_i.split('/')[-1][-5]=='0'] # ego view only
            else:
                self.data_list = list(self.meta_eval.keys())
                self.root_dir = data_dir[:-1]
            
            self.other_agent_always_gt = True if self.new_eval_mode not in ['pos12', 'neg12'] else False
            self.new_eval_mode = 'neg1'if self.new_eval_mode=='neg12' else self.new_eval_mode
            self.new_eval_mode = 'pos1'if self.new_eval_mode=='pos12' else self.new_eval_mode
        
        elif new_eval_mode is not None  and new_eval_mode!='':
            # Create a set for faster lookups
            files_with_found_acts_set = {i.split('/')[-1].replace('.pkl', '.npz') for i in self.acts_list}
            # Filter the data_list using the set
            self.data_list = [i for i in self.data_list if i.split('/')[-1] in files_with_found_acts_set]
            # files_with_found_acts = [i.split('/')[-1].replace('.pkl','.npz') for i in self.acts_list]
            # self.data_list = [i for i in self.data_list if i.split('/')[-1] in files_with_found_acts]
        

        # if self.act:
        #     self.agent_jsons = self.preload_json_files()
            

    def find_transposed_agent_pars(self):
        for file_name in self.data_list:
            print([self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split('/')[-1].split('_')[1] == file_name.split('/')[-1].split('_')[2]][1].split('/')[-1])
            break

    def preload_json_files(self):
        # Preload map JSON files
        [file_name.split('/')[-1].split('.')[0] for file_name in self.data_list if f"{self.agentJsons_dir}/agent_{file_name.split('/')[-1].split('.')[0]}.json" in self.agentJsons_list]
        for file_name in self.data_list:
            base_name = file_name.split('/')[-1].split('.')[0]
            # map_json_path = f"{self.mapJsons_dir}/map_{base_name}.json"
            agent_json_path = f"{self.agentJsons_dir}/agent_{base_name}.json"
            try:
                with open(map_json_path, 'r') as file:
                    self.mapJsons_list.add(map_json_path)
                with open(agent_json_path, 'r') as file:
                    self.agentJsons_list.add(agent_json_path)
            except FileNotFoundError:
                continue
            
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # self.data_list[idx] = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split('/')[-1].split('_')[1] == file_name.split('/')[-1].split('_')[2] and self.data_list[i].split('/')[-1].split('_')[0] == file_name.split('/')[-1].split('_')[0]][0]
        if self.new_eval:
            np_filename = self.root_dir + '_'.join(self.data_list[idx].split('_')[:-1])+'.npz'
        else:
            np_filename = self.data_list[idx]

        data = np.load(np_filename)

        if self.nuplan: # not valid
            # out_dict = {
            #     'traj': traj,
            #     'ego_state': data['ego'][0],
            #     'neighbors_state': np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0),
            #     # 'map_lanes': data['map_lanes'][:, :, :200:2],
            #     'map_lanes': data['map_lanes'][:, :6, :],
            #     'map_crosswalks': data['map_crosswalks'][:, :, :100:2],
            #     'object_type':object_type,
            #     'ego': data['ego'],
            #     'ground_truth': data['gt_future_states'],
            #     'neighbors': data['neighbors'],
            #     'file_name': filename,
            #     'data_object_type' : data_object_type,
            #     'act': torch.tensor(act),
            #     'contrastive_sample': contrastive_sample or of_sample, # used to indicate ignoring the generated trajectory
            #     'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
            # }
            file_name = data['map_name'].item()
            ego, neighbor = data["ego_agent_past"], data["neighbor_agents_past"]
            ego_future_states = data["ego_agent_future"]
            neighbor_future_states = data["neighbor_agents_future"]
            map_lanes = data['map_lanes'][:, :6, ..., :200:2, :]
            # map_lanes = data["lanes"][None][:, :, :200:2]
            map_crosswalks = data['map_crosswalks'][:, :, :100:2]
            # map_crosswalks = data["crosswalks"][None][:, :, :100:2]
            object_type = (data['ego'][:,-1,8:].argmax(-1)+1) * data['ego'][:,-1,8:].sum(-1).astype(int)
            additional_map_lanes, additional_map_crosswalks, additional_boundaries, traffic_lights, stop_signs, speed_bumps = np.array(-1), np.array(-1), np.array(-1), np.array(-1), np.array(-1), np.array(-1) 
            acts = np.array([self.acts_dict[np_filename.split('/')[-1]], -1])
            return ego, neighbor, map_lanes, map_crosswalks, ego_future_states, neighbor_future_states, object_type, acts,\
            additional_map_lanes, additional_map_crosswalks, additional_boundaries, traffic_lights, stop_signs, speed_bumps, file_name
        ego = data['ego'][0]
        neighbor = np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0)
        
        map_lanes = data['map_lanes'][:, :, :200:2]
        map_crosswalks = data['map_crosswalks'][:, :, :100:2]
        ego_future_states = data['gt_future_states'][0]
        neighbor_future_states = data['gt_future_states'][1]
        object_type = data['object_type']

        # plt.figure()
        # x1, y1 = np.concatenate([ego[:,0], ego_future_states[:,0]]), np.concatenate([ego[:,1], ego_future_states[:,1]])
        # x2, y2 = np.concatenate([neighbor[0, :,0], neighbor_future_states[:,0]]), np.concatenate([neighbor[0, :,1], neighbor_future_states[:,1]])
        # # Plot the first path with an arrow at the end
        # plt.plot(x1, y1, color='blue', label='Agent-1')
        # plt.arrow(x1[-2], y1[-2], x1[-1] - x1[-2], y1[-1] - y1[-2], head_width=0.5, head_length=0.5, fc='blue', ec='blue')
        # # Plot the second path with an arrow at the end
        # plt.plot(x2, y2, color='red', label='Agent-2')
        # plt.arrow(x2[-2], y2[-2], x2[-1] - x2[-2], y2[-1] - y2[-2], head_width=0.5, head_length=0.5, fc='red', ec='red')
        # plt.title('Directional Line Plot')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.legend()
        # plt.savefig('ex.png')
        # plt.close()
        
        if self.new_eval_mode=='pos1_synth':
            acts = np.array([-1, -1])
            valid_positive=True
            root_dir = '/'.join(self.data_list[idx].split('/')[:-1])
            file_name = np_filename.split('/')[-1].split('.')[0]
            template_dir = root_dir+'_templateLLM/'+file_name+'.txt'
            with open(template_dir) as f:
                lines = f.readlines()
            templates = [json.loads(line) for line in lines]
            # contrastive_acts, gt_templates, contrastive_templates = self.get_contrastive_act(templates)
            gt_act, contrastive_acts, positive_acts, _, _, _ = self.get_acts(templates)
            if len(positive_acts)==0:
                valid_positive=False
                act = -1
            else:
                positive_prob = self.positive_prob[positive_acts]/sum(self.positive_prob[positive_acts])
                positive_class_arg = np.random.choice(len(positive_acts), p=positive_prob)
                act = positive_acts[positive_class_arg]
                acts[0] = act

            
        file_name = np_filename.split('/')[-1].split('.')[0]
        if self.new_eval:
            acts, file_name = self.get_new_eval_act(self.data_list[idx])
        elif self.act:
            if self.new_eval_mode!='pos1_synth':
                acts = np.array([-1, -1])
            # map_json_dir = f"{self.mapJsons_dir}/map_{file_name}.json"
            # if map_json_dir not in self.mapJsons_list:
            #     map_json_dir = f"{self.mapJsons_dir}/{file_name}.json"
            # if map_json_dir in self.mapJsons_list:
            #     with open(map_json_dir, 'r') as file:
            #         map_json_data = file.read()
            #     # Parse the JSON string into a Python dictionary
            #     map_data_dict = json.loads(map_json_data)
            #     valid_map = True
            # else:
            #     valid_map = False
            if (not self.random_drop_act or (self.random_drop_act and np.random.choice(2, p=[0.5,0.5])==1)) and self.new_eval_mode!='pos1_synth':
                try:
                    with open(f"{self.acts_dir}/{file_name}.pkl", 'rb') as file:
                        out_data = pickle.load(file)
                    acts_loaded = eval(out_data)
                    acts[0] = acts_loaded['act0'] if data['object_type'][0]==1  else -1
                    if not self.ego_act_only:
                        acts[1] = acts_loaded['act1'] if data['object_type'][1]==1  else -1
                except FileNotFoundError:#(FileNotFoundError, EOFError, TypeError, KeyError, ValueError):
                    # Do nothing and continue
                    pass
            # agent_json_dir = f"{self.agentJsons_dir}/agent_{file_name}.json"
            # if agent_json_dir not in self.agentJsons_list:
            #     agent_json_dir = f"{self.agentJsons_dir}/{file_name}.json"
            
            # if agent_json_dir not in self.agentJsons_list:
            #     valid_agent_json = False
            # else:
            #     valid_agent_json = True
            # if valid_agent_json:
            #     with open(agent_json_dir, 'r') as file:
            #         agent_json_data = file.read()
            #     # Parse the JSON string into a Python dictionary
            #     agent_data_dict = json.loads(agent_json_data)
            
                # acts[0] = agent_data_dict['Agent-1']['direction 0.1to8_cls'] if ('Agent-1' in agent_data_dict.keys() and data['object_type'][0]==1)  else -1
                # if not self.ego_act_only:
                #     acts[1] = agent_data_dict['Agent-2']['direction 0.1to8_cls'] if ('Agent-2' in agent_data_dict.keys() and data['object_type'][1]==1) else -1
            # print(file_name)
            # classes_names = ['stationary', 'move straight', 'move straight veering right', 'move straight veering left', 'turn right', 'turn left', 'take right U-turn', 'take left U-turn']
            # print(f"Agent-1: {acts[0]}")
            # print(f"Agent-2: {acts[1]}")
        else:
            # acts = np.array([-1, -1])
            if self.new_eval_mode!='pos1_synth':
                acts = np.array([-1, -1])
        
        if self.act and (self.contrastive or self.positive_notgt) and self.new_eval_mode!='pos1_synth':
            valid_contrastive=True
            valid_positive=True
            root_dir = '/'.join(self.data_list[idx].split('/')[:-1])
            template_dir = root_dir+'_templateLLM/'+file_name+'.txt'
            with open(template_dir) as f:
                lines = f.readlines()
            templates = [json.loads(line) for line in lines]
            # contrastive_acts, gt_templates, contrastive_templates = self.get_contrastive_act(templates)
            gt_act, contrastive_acts, positive_acts, _, _, _ = self.get_acts(templates)
            if self.contrastive:
                if len(contrastive_acts)==0:
                    valid_contrastive=False
                else:
                    contrastive_prob = self.contrastive_prob[contrastive_acts]/sum(self.contrastive_prob[contrastive_acts])
                    contrastive_class_arg = np.random.choice(len(contrastive_acts), p=contrastive_prob)
            if self.positive_notgt:
                if len(positive_acts)==0:
                    valid_positive=False
                else:
                    positive_prob = self.positive_prob[positive_acts]/sum(self.positive_prob[positive_acts])
                    positive_class_arg = np.random.choice(len(positive_acts), p=positive_prob)
            if self.contrastive and valid_contrastive:
                act = contrastive_acts[contrastive_class_arg]
            elif self.positive_notgt and valid_positive:
                act = positive_acts[positive_class_arg]
            else:
                act = -1
            acts[0] = act
        if not self.full_map:    
            if self.new_eval:
                try:
                    act_2_dir = '<internal_dataset_root>/waymo/gameformer/sep14_2025_2agent/validation_complementary_data_acts/'+np_filename.replace('.npz','.pkl').split('/')[-1]
                    with open(act_2_dir, "rb") as f:
                        act2_data = pickle.load(f)
                    # acts[1]=act2_data['act1']
                    acts[1]=ast.literal_eval(act2_data)['act1']
                except:
                    file_not_found =np_filename.replace('.npz','.pkl').split('/')[-1]
                    print(f"> {file_not_found} not found")
                    pass
                # check if filename
            return ego, neighbor, map_lanes, map_crosswalks, ego_future_states, neighbor_future_states, object_type, acts, file_name
        else:
            # n_lanes, t_samples, t_subsampling = 100, 500, 2
            additional_map_lanes = data['additional_map_lanes']#[:n_lanes, :t_samples:t_subsampling, :]
            additional_map_crosswalks = data['additional_map_crosswalks'][:6, :100:2]
            additional_boundaries = data['additional_boundaries']#[:n_lanes, :t_samples:t_subsampling, :]

            traffic_lights = data['traffic_lights']
            # traffic_lights =np.vstack((traffic_lights[np.newaxis], traffic_lights[np.newaxis]))

            stop_signs = data['stop_signs']
            # stop_signs =np.vstack((stop_signs[np.newaxis], stop_signs[np.newaxis]))

            speed_bumps = data['speed_bumps'][:,:100:2]
            # speed_bumps =np.vstack((speed_bumps[np.newaxis], speed_bumps[np.newaxis]))
            
            return ego, neighbor, map_lanes, map_crosswalks, ego_future_states, neighbor_future_states, object_type, acts,\
            additional_map_lanes, additional_map_crosswalks, additional_boundaries, traffic_lights, stop_signs, speed_bumps, file_name
    
    def get_new_eval_act(self, filename):
        filename_ = filename
        # filename_ = [k for k in self.meta_eval[self.new_eval_mode].keys() if filename.split('/')[-1][:-4] in k][0]
        # self.meta_eval[self.new_eval_mode][filename_]
        act = self.meta_eval[filename_][self.new_eval_mode]

        # act = self.meta_eval[self.new_eval_mode][filename_][self.new_eval_mode]
        if self.ego_act_only:
            acts = np.array([act, -1])
        else:
            other_agent_key = self.new_eval_mode.replace('1','2') if '1' in self.new_eval_mode else self.new_eval_mode.replace('2', '1')
            if self.other_agent_always_gt:
                other_agent_key = 'gt1' if '1' in other_agent_key else 'gt2'
            act2 = self.meta_eval[filename_][other_agent_key]
            # act2 = self.meta_eval[filename_][self.new_eval_mode]
            acts = np.array([act, act2]) if '1' in self.new_eval_mode else np.array([act2, act])
        return acts, filename_

    def get_acts(self, templates):
        ## loading and filtering ground truth and contrastive acts
        gt_templates, contrastive_templates, positive_templates = [], [], []
        direction_classes = self.direction_classes
        gt_act = templates[0]['Direction_cls']
        contrastive_acts = []
        possitive_acts_not_gt = []
        for template_i in templates:
            if template_i['Label']=='gt':
                gt_templates.append(template_i)
            if template_i['Label']=='negative':
                contrastive_acts.append(template_i['Direction_cls'])
                contrastive_templates.append(template_i)
            if template_i['Label']=='possible direction not gt':
                possitive_acts_not_gt.append(template_i['Direction_cls'])
                positive_templates.append(template_i)
        if 'straight' in direction_classes[gt_act] or True in ['straight' in direction_i for direction_i in direction_classes[possitive_acts_not_gt]]:
            contrastive_templates = [contrastive_templates[i] for i in range(len(contrastive_acts)) if 'straight' not in direction_classes[contrastive_acts[i]]]
            contrastive_acts = [act_i for act_i in contrastive_acts if 'straight' not in direction_classes[act_i]]
        
        return gt_act, contrastive_acts, possitive_acts_not_gt, gt_templates, contrastive_templates, positive_templates

        # for i in tqdm(range(len(train_dataset)):
        #     xx += 1 if train_dataset[i][7][0]!=-1 else 0
            # ____________________________________________________________________________
    #         future_states = data['ground_truth']
    #         ego_future_states = future_states[0]
    #         neighbor_future_states = future_states[1]

    #         traffic_lights = data['traffic_lights']
    #         traffic_lights_masks = traffic_lights[:,-1].astype(bool)
    #         traffic_lights[~traffic_lights_masks] = [0,0,0,0]

    #         stop_signs = data['stop_signs']
    #         stop_signs_masks = stop_signs[:,-1].astype(bool)
    #         stop_signs[~stop_signs_masks] = [0,0,0,0]

    #         center_lanes = data['center_lanes']
    #         center_lanes_masks = data['center_lanes_masks'].astype(bool)
    #         center_lanes[~center_lanes_masks] = [0,0,0,0]

    #         boundaries = data['boundaries']
    #         boundaries_masks = data['boundaries_masks'].astype(bool)
    #         boundaries[~boundaries_masks] = [0,0,0,0]

    #         crosswalks = data['crosswalks']
    #         crosswalks_masks = data['crosswalks_masks'].astype(bool)
    #         crosswalks[~crosswalks_masks] = [0,0,0,0]

    #         speed_bumps = data['speed_bumps']
    #         speed_bumps_masks = data['speed_bumps_masks'].astype(bool)
    #         speed_bumps[~speed_bumps_masks] = [0,0,0,0]
            
    #         object_type_one_hot = data['ego'][:,-1,-3:]
    #         object_type = np.array([(object_type_one_hot[i].argmax()+1)* int(object_type_one_hot[i].sum()) for i in range(len(object_type_one_hot))])

    #         file_name = self.data_list[idx].split('/')[-1].split('.')[0]
    #         if self.act:
    #             map_json_dir = f"{self.mapJsons_dir}/map_{file_name}.json"
    #             if map_json_dir in self.mapJsons_list:
    #                 with open(map_json_dir, 'r') as file:
    #                     map_json_data = file.read()
    #                 # Parse the JSON string into a Python dictionary
    #                 map_data_dict = json.loads(map_json_data)
                
    #             agent_json_dir = f"{self.agentJsons_dir}/agent_{file_name}.json"
    #             if agent_json_dir in self.agentJsons_list:
    #                 with open(agent_json_dir, 'r') as file:
    #                     agent_json_data = file.read()
    #                 # Parse the JSON string into a Python dictionary
    #                 agent_data_dict = json.loads(agent_json_data)
                
    #             acts = np.array([-1, -1]).astype(future_states.dtype)
    #             acts[0] = agent_data_dict['Agent-1']['direction 0.1to8_cls']
    #         else:
    #             acts = np.array([-1, -1])

            
    #         center_lanes = self.pad_or_truncate_array_to_shape(center_lanes, desired_shape=self.center_lanes_shape)
    #         boundaries = self.pad_or_truncate_array_to_shape(boundaries, desired_shape=self.boundaries_shape)
    #         crosswalks = self.pad_or_truncate_array_to_shape(crosswalks, desired_shape=self.crosswalks_shape)
    #         speed_bumps = self.pad_or_truncate_array_to_shape(speed_bumps, desired_shape=self.speed_bumps_shape)
    #         traffic_lights = self.pad_or_truncate_array_to_shape(traffic_lights, desired_shape=self.traffic_lights_shape)
    #         stop_signs = self.pad_or_truncate_array_to_shape(stop_signs, desired_shape=self.stop_signs_shape)

    #         return ego, neighbor, ego_future_states, neighbor_future_states, object_type, acts, file_name,\
    #             center_lanes, center_lanes_masks, boundaries, boundaries_masks, crosswalks, crosswalks_masks, speed_bumps, speed_bumps_masks,\
    #                 traffic_lights, traffic_lights_masks, stop_signs, stop_signs_masks
    
    # def pad_or_truncate_array_to_shape(self, array, desired_shape):
    #     # Truncate the array if the desired shape is smaller
    #     truncated_array = array[:desired_shape[0], :desired_shape[1]]
        
    #     # Pad the truncated array if it is smaller than the desired shape
    #     padding_top = max(0, desired_shape[0] - truncated_array.shape[0])
    #     padding_bottom = 0
    #     padding_left = max(0, desired_shape[1] - truncated_array.shape[1])
    #     padding_right = 0
    #     padding_tuple = ((0,padding_top), (0,padding_left), (0,0)) if len(truncated_array.shape)==3 else ((0,padding_top), (0,0))

    #     padded_array = np.pad(truncated_array, padding_tuple, mode='constant')
        
    #     return padded_array

# class DrivingData_tfexample(Dataset):
#     def __init__(self, data_dir, act=False):
#         self.data_list = glob.glob(data_dir)

#     def __len__(self):
#         return len(self.data_list)
    
#     def __getitem__(self, idx):
#         data = np.load(self.data_list[idx])
#         ego = data['ego'][0]
#         neighbor = np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0)

#         map_lanes = data['map_lanes'][:, :, :200:2]
#         map_crosswalks = data['map_crosswalks'][:, :, :100:2]
#         ego_future_states = data['gt_future_states'][0]
#         neighbor_future_states = data['gt_future_states'][1]
#         object_type = data['object_type']

#         if self.act:
#             valid_action = np.load(self.data_list_act_dir+'/'+file_name+'.npy', allow_pickle = True)
#             if bool(valid_action[0]):
#                 valid_action, turn, move1, move2, contrastive_turn = valid_action
#             else:
#                 valid_action, turn, move1, move2, contrastive_turn = valid_action[0], '', '', '', '' # no valid action instruction found
#             return ego, neighbor, map_lanes, map_crosswalks, ego_future_states, neighbor_future_states, object_type, valid_action, turn, move1, move2, contrastive_turn
#         return ego, neighbor, map_lanes, map_crosswalks, ego_future_states, neighbor_future_states, object_type


def imitation_loss(trajectories, ground_truth,gmm=True, subsample=False):
    metric_time = [5, 10, 15] if subsample else [29, 49, 79]
    ade_distance = torch.norm(trajectories[:, :, :, 4::5,:2] - ground_truth[:, :, None, 4::5, :2], dim=-1)
    fde_distance = torch.norm(trajectories[:, :, :, metric_time,:2] - ground_truth[:, :, None, metric_time, :2], dim=-1)
    distance = fde_distance.sum(-1) + ade_distance.mean(-1)
    best_mode = torch.argmin(distance.mean(1), dim=-1)
    B, N = trajectories.shape[0], trajectories.shape[1]
    best_mode_future = trajectories[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]]
    best_mode_future = best_mode_future.squeeze(2)
    de = F.smooth_l1_loss(best_mode_future, ground_truth[:, :, :, :2],reduction='none').sum(-1)
    ade = torch.mean(de[:,:,4::5],dim=-1)
    fde = torch.sum(de[:,:,metric_time],dim=-1)
    loss = fde + ade
    loss = torch.mean(loss.mean(1))
    return loss, best_mode, best_mode_future

def gmm_loss(trajectories, convs, probs, ground_truth, subsample=False):
    metric = [5, 10, 15] if subsample else [29, 49, 79]
    distance = torch.norm(trajectories[:, :, :, : ,:2] - ground_truth[:, :, None, :, :2], dim=-1)
    # trajectories[0,0,0,:,0], trajectories[0,0,0,:,1]
    # import matplotlib.pyplot as plt
    # colors = ['r', 'b', 'k', 'c', 'g', 'y']
    # plt.figure()
    # for i in range(6):
    #     plt.plot(trajectories[0,0,i,:,0].cpu().detach().numpy(), trajectories[0,0,i,:,1].cpu().detach().numpy(), color=colors[i])
    # plt.savefig('ex.png')
    # plt.close()

    ndistance = distance.mean(-1) + distance[...,metric].sum(-1) 
    # for i in range(6):
    #     print(f">Mode-{i}) {distance.mean(-1)[0,0,i]}")
    best_mode = torch.argmin(ndistance.mean(1), dim=-1)
    # print(best_mode)
    B, N = trajectories.shape[0], trajectories.shape[1]
    
    #[b,n,t,2]
    best_mode_future = trajectories[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]].squeeze(2)
    #[b,n,t,3]
    convs = convs[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]].squeeze(2)

    dx = best_mode_future[...,0] - ground_truth[...,0]
    dy = best_mode_future[...,1] - ground_truth[...,1]

    log_std_x = torch.clip(convs[...,0], 0, 5)
    log_std_y = torch.clip(convs[...,1], 0, 5)

    std_x, std_y = torch.exp(log_std_x), torch.exp(log_std_y)

    reg_gmm_log_coefficient = log_std_x + log_std_y  # (batch_size, num_timestamps)
    reg_gmm_exp = 0.5  * ((dx**2) / (std_x**2) + (dy**2) / (std_y**2))
    loss = reg_gmm_log_coefficient + reg_gmm_exp
    loss = loss.mean(-1) + loss[..., metric].sum(-1)

    prob_loss = F.cross_entropy(input=probs, target=best_mode, label_smoothing=0.2)
    loss = loss + 2*prob_loss
    loss = loss.mean()

    return loss, best_mode, best_mode_future, convs

def level_k_loss(outputs, ego_future, neighbor_future, levels, gmm=True, subsample=False):
    loss: torch.tensor = 0
    neighbor_future_valid = torch.ne(neighbor_future[..., :2].sum(-1), 0)
    ego_future_valid = torch.ne(ego_future[..., :2].sum(-1), 0)
    sc_cnt = 0
    
    for k in range(levels+1):
        # print('--'*50)
        # print(f'***Level = {k}')
        trajectories = outputs[f'level_{k}_interactions'][..., :2]
        scores = outputs[f'level_{k}_scores']
        ego = trajectories[:, 0] * ego_future_valid.unsqueeze(1).unsqueeze(-1)
        if trajectories.shape[1]>1:
            neighbor = trajectories[:, 1] * neighbor_future_valid.unsqueeze(1).unsqueeze(-1)
            trajectories = torch.stack([ego, neighbor], dim=1)
            gt_future = torch.stack([ego_future, neighbor_future], dim=1)
        else:
            trajectories = torch.stack([ego], dim=1)
            gt_future = torch.stack([ego_future], dim=1)
            neighbor=None
        
        
        if gmm:
            convs = outputs[f'level_{k}_interactions'][..., 2:]
            gloss, best_mode, future, _ = gmm_loss(trajectories, convs, scores.sum(1), gt_future, subsample)
            loss += gloss
            # print(scores.sum(1)[0].softmax(-1))
        else:
            il_loss, best_mode, future = imitation_loss(trajectories, gt_future, subsample)
            sc_loss = F.cross_entropy(scores.sum(1), best_mode)
            loss += 0.5*il_loss + 2*sc_loss
    
    return loss, (future,best_mode,scores), (ego, neighbor) 


def level_k_loss_trajgpt(outputs, ego_future, neighbor_future, levels, gmm=True, subsample=True):
    loss: torch.tensor = 0
    neighbor_future_valid = torch.ne(neighbor_future[..., :2].sum(-1), 0)
    ego_future_valid = torch.ne(ego_future[..., :2].sum(-1), 0)
    sc_cnt = 0
    
    for k in range(levels+1):
        trajectories = outputs[f'level_{k}_interactions'][..., :2]
        scores = outputs[f'level_{k}_scores']
        ego = trajectories[:, 0] * ego_future_valid.unsqueeze(1).unsqueeze(-1)
        if trajectories.shape[1]>1:
            neighbor = trajectories[:, 1] * neighbor_future_valid.unsqueeze(1).unsqueeze(-1)
            trajectories = torch.stack([ego, neighbor], dim=1)
            gt_future = torch.stack([ego_future, neighbor_future], dim=1)
        else:
            trajectories = torch.stack([ego], dim=1)
            gt_future = torch.stack([ego_future], dim=1)
            neighbor=None
        if gmm:
            convs = outputs[f'level_{k}_interactions'][..., 2:]
            gloss, best_mode, future, _ = gmm_loss(trajectories, convs, scores.sum(1), gt_future, subsample)
            loss += gloss
        else:
            il_loss, best_mode, future = imitation_loss(trajectories, gt_future, subsample)
            sc_loss = F.cross_entropy(scores.sum(1), best_mode)
            loss += 0.5*il_loss + 2*sc_loss

    return loss, (future,best_mode,scores), trajectories

def motion_metrics(trajectories, ego_future, neighbor_future, subsample=False):
    ego_future_valid = torch.ne(ego_future[..., :2], 0)
    ego_trajectory = trajectories[:, 0] * ego_future_valid
    if not subsample:
        distance = torch.norm(ego_trajectory[:, 4::5, :2] - ego_future[:, 4::5, :2], dim=-1)
    else:
        distance = torch.norm(ego_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    egoADE = torch.mean(distance)
    egoFDE = torch.mean(distance[:, -1])    
    neigbhor_future_valid = torch.ne(neighbor_future[..., :2], 0)
    neighbor_trajectory = trajectories[:, 1] * neigbhor_future_valid
    if not subsample:
        distance = torch.norm(neighbor_trajectory[:, 4::5, :2] - neighbor_future[:, 4::5, :2], dim=-1)
    else:
        distance = torch.norm(ego_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    neighborADE = torch.mean(distance)
    neighborFDE = torch.mean(distance[:, -1])

    return egoADE.item(), egoFDE.item(), neighborADE.item(), neighborFDE.item()


# Define metrics to measure the prediction
def default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
        track_steps_per_second: 10
        prediction_steps_per_second: 2
        track_history_samples: 10
        track_future_samples: 80
        speed_lower_bound: 1.4
        speed_upper_bound: 11.0
        speed_scale_lower: 0.5
        speed_scale_upper: 1.0
        step_configurations {
            measurement_step: 5
            lateral_miss_threshold: 1.0
            longitudinal_miss_threshold: 2.0
        }
        step_configurations {
            measurement_step: 9
            lateral_miss_threshold: 1.8
            longitudinal_miss_threshold: 3.6
        }
        step_configurations {
            measurement_step: 15
            lateral_miss_threshold: 3.0
            longitudinal_miss_threshold: 6.0
        }
        max_predictions: 6
    """
    # config_text = """
    #     track_steps_per_second: 2
    #     prediction_steps_per_second: 2
    #     track_history_samples: 2
    #     track_future_samples: 16
    #     speed_lower_bound: 1.4
    #     speed_upper_bound: 11.0
    #     speed_scale_lower: 0.5
    #     speed_scale_upper: 1.0
    #     step_configurations {
    #         measurement_step: 5
    #         lateral_miss_threshold: 1.0
    #         longitudinal_miss_threshold: 2.0
    #     }
    #     step_configurations {
    #         measurement_step: 9
    #         lateral_miss_threshold: 1.8
    #         longitudinal_miss_threshold: 3.6
    #     }
    #     step_configurations {
    #         measurement_step: 15
    #         lateral_miss_threshold: 3.0
    #         longitudinal_miss_threshold: 6.0
    #     }
    #     max_predictions: 6
    # """
    text_format.Parse(config_text, config)

    return config

class MotionMetrics:
    """Wrapper for motion metrics computation."""
    def __init__(self):
        super().__init__()
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []
        self._metrics_config = default_metrics_config()

    def reset_states(self):
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []

    def update_state(self, prediction_trajectory, prediction_score, ground_truth_trajectory, ground_truth_is_valid, object_type, subsample=False):
        if not subsample:   
            self._prediction_trajectory.append(prediction_trajectory[..., 4::5,:].clone().detach().cpu())
        else:
            self._prediction_trajectory.append(prediction_trajectory[..., :,:].clone().detach().cpu())
        self._prediction_score.append(prediction_score.clone().detach().cpu())
        self._ground_truth_trajectory.append(ground_truth_trajectory.cpu())
        self._ground_truth_is_valid.append(ground_truth_is_valid[..., -1].cpu())
        self._object_type.append(object_type.cpu())
    
    def set_single_state(self, prediction_trajectory, prediction_score, ground_truth_trajectory, ground_truth_is_valid, object_type, subsample=False):
        if not subsample:   
            self._prediction_trajectory = [prediction_trajectory[..., 4::5,:].clone().detach().cpu()]
        else:
            self._prediction_trajectory = [prediction_trajectory[..., :,:].clone().detach().cpu()]
        self._prediction_score = [prediction_score.clone().detach().cpu()]
        self._ground_truth_trajectory = [ground_truth_trajectory.cpu()]
        self._ground_truth_is_valid = [ground_truth_is_valid[..., -1].cpu()]
        self._object_type = [object_type.cpu()]

    def result(self):
        # [batch_size, 1, top_k, 2, steps, 2].
        prediction_trajectory = torch.cat(self._prediction_trajectory, dim=0) # [batch_size, 6, agents, steps, 2]
        # [batch_size, 1, top_k].
        prediction_score = torch.cat(self._prediction_score, dim=0) # batch_size, 6
        # [batch_size, 1, 2, gt_steps, 7].
        ground_truth_trajectory = torch.cat(self._ground_truth_trajectory, dim=0) # batch_size, agents, steps, 7
        # [batch_size, 1, gt_steps].
        ground_truth_is_valid = torch.cat(self._ground_truth_is_valid, dim=0) # batch_size, agents, steps
        # [batch_size, 1].
        object_type = torch.cat(self._object_type, dim=0) # batch_size, agents

        # We are predicting more steps than needed by the eval code. Subsample.
        #interval = (self._metrics_config.track_steps_per_second // self._metrics_config.prediction_steps_per_second)
        # Prepare these into shapes expected by the metrics computation.
        # [batch_size, top_k, num_agents_per_joint_prediction, pred_steps, 2].
        # num_agents_per_joint_prediction is 1 here.
        if len(prediction_trajectory.shape)>=4:
            prediction_trajectory = prediction_trajectory.unsqueeze(dim=1).numpy() # batch_size, 1, 6, agents, 16, 2
            prediction_score = prediction_score.unsqueeze(dim=1).numpy() # batch_size, 1, 6
        else:
            prediction_trajectory = prediction_trajectory.numpy()

        # [batch_size, num_agents_per_joint_prediction, gt_steps, 7].

        ground_truth_trajectory = ground_truth_trajectory.numpy()
        # [batch_size, num_agents_per_joint_prediction, gt_steps].
        ground_truth_is_valid = ground_truth_is_valid.numpy()

        # [batch_size, num_agents_per_joint_prediction].
        object_type = object_type.numpy()
        b = ground_truth_trajectory.shape[0]

        prediction_ground_truth_indices = tf.cast(tf.concat([tf.zeros((b, 1, 1)), tf.ones((b, 1, 1))],axis=-1),tf.int64) # batch_size, 1, 2
        # print(object_type.shape)
        ground_truth_is_valid = tf.convert_to_tensor(ground_truth_is_valid)
        prediction_ground_truth_indices_mask = tf.ones_like(prediction_ground_truth_indices, dtype=tf.float32)
        valid_gt_all = tf.cast(tf.math.greater_equal(tf.reduce_sum(tf.cast(ground_truth_is_valid,tf.float32), axis=-1), 1), tf.float32)
        valid_gt_all = valid_gt_all[:, tf.newaxis, :] * prediction_ground_truth_indices_mask
        valid_gt_all = tf.cast(valid_gt_all, tf.bool)

        if prediction_trajectory.shape[3] == 1: # single agent
            prediction_ground_truth_indices, valid_gt_all = prediction_ground_truth_indices[...,:1], valid_gt_all[...,:1]
        try:
            metric_values = py_metrics_ops.motion_metrics(
                    config=self._metrics_config.SerializeToString(),
                    prediction_trajectory=tf.convert_to_tensor(prediction_trajectory), # b,1,6,agents,16,2
                    prediction_score=tf.convert_to_tensor(prediction_score), # b,1,6
                    ground_truth_trajectory=tf.convert_to_tensor(ground_truth_trajectory),#b,agents,91,7
                    ground_truth_is_valid=ground_truth_is_valid,#b,agents,91
                    object_type=tf.convert_to_tensor(object_type),#b,agents
                    prediction_ground_truth_indices=prediction_ground_truth_indices,#b,1,2 agents
                    prediction_ground_truth_indices_mask=valid_gt_all)#b,1,2 agents
        except:
            "Something wrong"


        metric_names = config_util.get_breakdown_names_from_motion_config(self._metrics_config)
        results = {}

        for i, m in enumerate(['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']):
            avg = []
            for j, n in enumerate(metric_names):
                results[f'{m}_{n}'] = metric_values[i][j].numpy()
                avg.append(metric_values[i][j].numpy())
            results[f'{m}_vehicle'] = np.mean(avg[:3])
            results[f'{m}_pedestrian'] = np.mean(avg[3:6])
            results[f'{m}_cyclist'] = np.mean(avg[6:])
            results[f'{m}'] = (results[f'{m}_vehicle']+results[f'{m}_pedestrian']+results[f'{m}_cyclist'])/3

        if results[f'minADE_vehicle'] ==0:
            v_mask_valid = object_type==1
            gt_mask_valid = valid_gt_all.numpy()[:,0]
            gt_mask_valid = gt_mask_valid*v_mask_valid
            gt_ = ground_truth_trajectory[...,11:,:2][...,[29, 49, 79],:][:,np.newaxis]
            # pred_ = prediction_trajectory # 5,9,15
            # pred_ = np.concatenate((prediction_trajectory[:,0,:,:,:,:][...,29,:][...,np.newaxis,:], prediction_trajectory[:,0,:,:,:,:][...,49,:][...,np.newaxis,:], prediction_trajectory[:,0,:,:,:,:][...,79,:][...,np.newaxis,:]), -2)
            pred_ = np.concatenate((prediction_trajectory[:,0,:,:,:,:][...,5,:][...,np.newaxis,:], prediction_trajectory[:,0,:,:,:,:][...,9,:][...,np.newaxis,:], prediction_trajectory[:,0,:,:,:,:][...,15,:][...,np.newaxis,:]), -2)
            norms = np.linalg.norm((gt_-pred_),axis=-1)
            # norms_ = np.zeros(norms[...,0,:].shape)
            # norms_valid = np.ones(norms_.shape[0], dtype=int)
            # for i in range(norms.shape[0]):
            #     if sum(gt_mask_valid[i])>0:
            #         norms_[i] = norms[i][...,gt_mask_valid[i],:].mean(-2)
            #     else:
            #         norms_valid[i] = 0
            # norms_ = norms_[norms_valid]
            # norms = norms_
            norms_min = np.zeros(norms.shape[0])
            for example_ in range(len(norms)):
                for t_ in range(3):
                    norms_min[example_] += norms[example_,:,0,t_].min() # min at this specific point of this specific example
            # for t_ in range(3):
            #     norms_min += norms[...,t_].min(-1)
            # FDE = norms[...,-1].mean(-1).min(-1)
            FDE = norms[...,-1,-1].min(-1).mean()
            ADE = norms_min/3
            ADE = ADE.mean()
            results[f'minADE_vehicle'] = ADE
            results[f'minFDE_vehicle'] = FDE
        return results