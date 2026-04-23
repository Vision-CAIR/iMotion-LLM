import os
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.traj_base_dataset import BaseDataset
# from minigpt4.datasets.datasets.traj_caption_datasets import CaptionDataset
import torch
import numpy as np
from traj_utils import *
from torch.utils.data import DataLoader, Dataset
from generate_meta_caption import *
import random
# from extract_instruct_v2 import extract_simple_turn_5_classes
# from extract_instruct_v3 import ClassifyTrack, get_sample_instruct
from extract_instruct_v3 import *
# from extract_instruct_v4 import *
from instructions.direction_instructions import DirectionClassifier

class TrajDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor==vis_processor, text_processor=text_processor)
        
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }

# class TrajectoryDataset(Dataset):
# # class WaymoDataset(Dataset):
#     """Dataloder for the Trajectory datasets"""
#     def __init__(self, data_dir):
#         # super(TrajectoryDataset, self).__init__()
#         self.data_list = glob.glob(data_dir)
#         self.seq_len = 19
#         self.obs_len = 3
#         self.pred_len = 16
#         self.load_all = True

#     def __len__(self):
#         return len(self.data_list)
    
#     def __getitem__(self, idx):
#         data = np.load(self.data_list[idx], allow_pickle = True)
#         traj_obs = torch.tensor(data['ego'][:,:,:2])
#         traj_pred = torch.tensor(data['gt_future_states'][:,:,:2])
#         rel_obs = torch.tensor(data['rel_obs'])
#         rel_pred = torch.tensor(data['rel_pred'])
#         rel_obs_1d = torch.tensor(data['rel_obs_1d'])
#         rel_pred_1d = torch.tensor(data['rel_pred_1d'])
        
#         if self.load_all:
#             object_type = torch.tensor(data['object_type'])
#             bin_edges = torch.tensor(data['bin_edges'])
#             continous_grid = torch.tensor(data['continous_grid'])
#             euc_mat = torch.tensor(data['euc_mat'])
#             return traj_obs, traj_pred, rel_obs, rel_pred, rel_obs_1d, rel_pred_1d, object_type, bin_edges, continous_grid, euc_mat

#         return traj_obs, traj_pred, rel_obs, rel_pred, rel_obs_1d, rel_pred_1d

class TrajAlignDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, act=False, act_json=True, template_select=4, contrastive=False, train=False, random_select=False, positive_notgt=False, two_agent=False, new_eval=False, new_eval_mode='traj_pred', agents_instructed='', random_drop=True, return_meta_data=False, stage_2=''):
        super(TrajAlignDataset, self).__init__()
        self.return_meta_data = return_meta_data
        self.act = act
        self.act_json = act_json
        self.data_dir = data_dir
        if act and act_json:
            self.data_list = glob.glob(data_dir[:-2]+'_json'+'/*')
        else:
            self.data_list = glob.glob(data_dir)

        if len(stage_2)>1:
            self.pos_data_list_ego_future = glob.glob(stage_2+'/*')
            self.pos_data_list_llm = glob.glob(stage_2+'_templateLLM/*')
            self.stage_2_dir = stage_2
            self.stage_2 = True
            all_file_names = [self.pos_data_list_ego_future[i][:-8].split('/')[-1] for i in range(len(self.pos_data_list_ego_future))]
            self.data_list = [data_dir[:-1]+ file_ +'.npz' for file_ in all_file_names]
            # [self.data_list[i].split('/')[-1][:-4] for i in range(len(self.data_list)) if self.data_list[i].split('/')[-1][:-4] in all_file_names]
            # DONE: Load the same samples for actual scenario
        else:
            self.stage_2 = False

        if template_select==33:
            self.template_select = 3
            self.random_drop=True
        else:
            self.random_drop=False
            self.template_select = template_select
        if random_drop:
            self.random_drop=random_drop

        self.contrastive = contrastive
        self.positive_notgt = positive_notgt
        self.train = train
        self.random_select = random_select

        direction_classifier = DirectionClassifier()
        self.direction_classes = np.array(direction_classifier.classes)

        train_contrastive_act_distribution = np.array([18.23, 2.7, 3.88, 3.91, 8.36, 16.04, 23.75, 23.12])
        train_positive_notgt_distribution = np.array([7.04, 6.76, 15.34, 9.14, 39.00, 13.54, 4.32, 4.85])
        train_groundtruth_act_distribution = [1.62, 55.77, 3.25, 3.71, 16.67, 17.49, 0.09, 1.40]
        self.train_contrastive_prob = [0.7, 0.3]
        self.train_prob = [0.7, 0.3] if self.contrastive else [1.0,0.0]
        self.contrastive_prob = (1/train_contrastive_act_distribution)/(sum(1/train_contrastive_act_distribution))
        self.positive_prob = (1/train_positive_notgt_distribution)/(sum(1/train_positive_notgt_distribution))
        
        self.two_agent=two_agent
        if two_agent:
            with open('/'.join(self.data_dir.split('/')[:-1])+'_filenames_T/filenames_T.json', 'r') as file:
                json_dict = file.read()
                json_dict = json.loads(json_dict)
            self.filenames_T = json_dict
        
        self.new_eval = new_eval
        self.pred_only = False
        if new_eval:
            if new_eval_mode=='pred_only':
                self.pred_only = True
                new_eval_mode='gt1'
            meta_ = '/'.join(data_dir.split('/')[:-1]) + f"_eval_meta/meta_{new_eval_mode}.json"
            if not os.path.exists(meta_):
                meta_ = '/'.join(data_dir.split('/')[:-1]) + f"_train_meta/meta_{new_eval_mode}.json"
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
            self.ego_act_only = True if not two_agent else False
            self.agents_instructed = agents_instructed

            
    #     ['stationary', 'move straight', 'move straight veering right',
    #    'move straight veering left', 'turn right', 'turn left',
    #    'take right U-turn', 'take left U-turn']

    #     [1/len(self.direction_classes) for _ in range(len(self.direction_classes))]
    #     self.contrastive_select_prob = [1/len(self.direction_classes) for _ in range(len(self.direction_classes))]
            
        # if act:
        #     self.data_list_act_dir = data_dir[:-2]+'_json'
        #     self.data_list_act = glob.glob(data_dir[:-2]+'_act'+'/*')
        #     self.act = act
        # else:
        #     self.data_list = glob.glob(data_dir)
        # self.seq_len = 19
        # self.obs_len = 3
        # self.pred_len = 16
        # self.load_all = True

    def __len__(self):
        return len(self.data_list)
   
    def __getitem__(self, idx):
        if self.new_eval:
            np_filename = self.root_dir + '_'.join(self.data_list[idx].split('_')[:-1])+'.npz'
        else:
            np_filename = self.data_list[idx]
        
        
        
            
            

        data = np.load(np_filename, allow_pickle = True)
        traj_obs = torch.tensor(data['ego'][:,:,:2])
        traj_pred = torch.tensor(data['gt_future_states'][:,:,:2])
        traj = torch.cat((traj_obs, traj_pred), dim=1)
        object_type = (data['ego'][:,-1,8:].argmax(-1)+1) * data['ego'][:,-1,8:].sum(-1).astype(int)
        data_object_type = data['object_type']

        if self.stage_2:
            # randomly choosing between actual scenario and other feasible scenario\
            use_pos_augmented_sample = bool(np.random.choice([True, False], p=[0.75, 0.25]))
            if use_pos_augmented_sample:
                np_filename_pos = self.stage_2_dir + '/' +np_filename.split('/')[-1]+'.npy'
                data_ = np.load(np_filename_pos)
                # DONE: Update traj_pred with the other feasible augmentation trajectory   
        else:
            use_pos_augmented_sample = False
        if self.new_eval:
            acts, filename = self.get_new_eval_act(self.data_list[idx])
            root_dir = '/'.join(self.data_dir.split('/')[:-1])
            template_dir = root_dir+'_templateLLM/'+np_filename.split('/')[-1].replace('.npz','.txt')
            instruct, caption, act, contrastive_sample, gt_act, positive_acts, contrastive_acts = self.get_instruct_caption(template_dir, force_caption=self.two_agent, acts=acts, agent='1')
            if self.two_agent:
                if np_filename in self.filenames_T.keys():
                    filename_2 = self.filenames_T[np_filename].split('/')[-1].replace('.npz','')
                    template_dir_2 = root_dir+'_templateLLM/'+filename_2+'.txt'
                    instruct_2, caption_2, act_2, contrastive_sample_2, gt_act2, positive_acts2, contrastive_acts2 = self.get_instruct_caption(template_dir_2, acts=acts, agent='2')
                    instruct_updated, caption_updated = self.include_agent_2(instruct, caption, contrastive_sample, instruct_2, caption_2, contrastive_sample_2)
        else:
            root_dir = '/'.join(self.data_dir.split('/')[:-1])
            filename = self.data_list[idx].split('/')[-1].replace('.npz','')
            # agent_dir = root_dir+'_agentJsons/'+filename+'.json'
            # map_dir = root_dir+'_mapJsons/'+filename+'.json'
            if use_pos_augmented_sample:
                # Done: Change the template LLM to the other feasible template LLM, ensure that the sample loaded will not be "infeasible"
                template_dir = self.stage_2_dir+'_templateLLM/'+filename+'.txt'
                instruct, caption, act, contrastive_sample, gt_act, positive_acts, contrastive_acts = self.get_instruct_caption(template_dir, force_caption=self.two_agent, positive_example_only=True)
                traj_pred[0] = torch.tensor(data_)
                traj[0,11:] = torch.tensor(data_)
                gt_future_state = data['gt_future_states']
                gt_future_state[0,:,:2]= torch.tensor(data_)
                return {
                'traj': traj,
                'ego_state': data['ego'][0],
                'neighbors_state': np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0),
                'map_lanes': data['map_lanes'][:, :, :200:2],
                'map_crosswalks': data['map_crosswalks'][:, :, :100:2],
                # 'map_lanes': data['map_lanes'],
                # 'map_crosswalks': data['map_crosswalks'],
                'object_type':object_type,
                'ego': data['ego'],
                'ground_truth': gt_future_state,
                'neighbors': data['neighbors'],
                'file_name': filename,
                'data_object_type' : data_object_type,
                'caption' : caption,
                'instruct': instruct,
                # 'additional_map_lanes': additional_map_lanes,
                # 'additional_map_crosswalks': additional_map_crosswalks,
                # 'additional_boundaries': additional_boundaries,
                # 'traffic_lights': traffic_lights,
                # 'stop_signs': stop_signs,
                # 'speed_bumps': speed_bumps,
                'act': torch.tensor(act),
                'contrastive_sample': contrastive_sample,
                'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
                }
                
            else:
                template_dir = root_dir+'_templateLLM/'+filename+'.txt'
                instruct, caption, act, contrastive_sample, gt_act, positive_acts, contrastive_acts = self.get_instruct_caption(template_dir, force_caption=self.two_agent)
            
            
            if self.two_agent:
                if np_filename in self.filenames_T.keys():
                    filename_2 = self.filenames_T[np_filename].split('/')[-1].replace('.npz','')
                    template_dir_2 = root_dir+'_templateLLM/'+filename_2+'.txt'
                    instruct_2, caption_2, act_2, contrastive_sample_2, gt_act2, positive_acts2, contrastive_acts2 = self.get_instruct_caption(template_dir_2)
                    # Options: 0:No control, 1:control agent-1, 2:control agent-2, 3:control both agents
                    instruct_updated, caption_updated = self.include_agent_2(instruct, caption, contrastive_sample, instruct_2, caption_2, contrastive_sample_2)
                else:
                    filename_2 = ''
                    instruct_updated = instruct
                    caption_updated = caption.replace(caption[caption.find('Decision'): caption.find('Generated')], ". Ego Decision: <Accepted>. ")
                    contrastive_sample_2 = False
                    act_2 = -1
                    gt_act2, positive_acts2, contrastive_acts2 = -1, [], []

        if 'nuplan' in self.data_dir:
            out_dict = {
                'traj': traj,
                'ego_state': data['ego'][0],
                'neighbors_state': np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0),
                'map_lanes': data['map_lanes'][:, :, :200:2],
                'map_crosswalks': data['map_crosswalks'][:, :, :100:2],
                'object_type':object_type,
                'ego': data['ego'],
                'ground_truth': data['gt_future_states'],
                'neighbors': data['neighbors'],
                'file_name': filename,
                'data_object_type' : data_object_type,
                'caption' : caption,
                'instruct': instruct,
                'act': torch.tensor(act),
                'contrastive_sample': contrastive_sample,
                'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
            }
            return out_dict
        # # n_lanes, t_samples, t_subsampling = 100, 500, 2
        # additional_map_lanes = data['additional_map_lanes']#[:n_lanes, :t_samples:t_subsampling, :]
        # additional_map_crosswalks = data['additional_map_crosswalks'][:6, :100:2]
        # additional_boundaries = data['additional_boundaries']#[:n_lanes, :t_samples:t_subsampling, :]

        # traffic_lights = data['traffic_lights']
        # # traffic_lights =np.vstack((traffic_lights[np.newaxis], traffic_lights[np.newaxis]))

        # stop_signs = data['stop_signs']
        # # stop_signs =np.vstack((stop_signs[np.newaxis], stop_signs[np.newaxis]))

        # speed_bumps = data['speed_bumps'][:,:100:2]
        # # speed_bumps =np.vstack((speed_bumps[np.newaxis], speed_bumps[np.newaxis]))
        
        out_dict = {
                'traj': traj,
                'ego_state': data['ego'][0],
                'neighbors_state': np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0),
                'map_lanes': data['map_lanes'][:, :, :200:2],
                'map_crosswalks': data['map_crosswalks'][:, :, :100:2],
                # 'map_lanes': data['map_lanes'],
                # 'map_crosswalks': data['map_crosswalks'],
                'object_type':object_type,
                'ego': data['ego'],
                'ground_truth': data['gt_future_states'],
                'neighbors': data['neighbors'],
                'file_name': filename,
                'data_object_type' : data_object_type,
                'caption' : caption,
                'instruct': instruct,
                # 'additional_map_lanes': additional_map_lanes,
                # 'additional_map_crosswalks': additional_map_crosswalks,
                # 'additional_boundaries': additional_boundaries,
                # 'traffic_lights': traffic_lights,
                # 'stop_signs': stop_signs,
                # 'speed_bumps': speed_bumps,
                'act': torch.tensor(act),
                'contrastive_sample': contrastive_sample,
                'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
            }
        if self.return_meta_data:
            out_dict.update({'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts)})
        if self.two_agent:
            out_dict['instruct'], out_dict['caption'], out_dict['act'], out_dict['contrastive_sample'] = instruct_updated, caption_updated, torch.tensor([act, act_2]), torch.tensor([contrastive_sample, contrastive_sample_2])
            out_dict.update({'file_name_2':filename_2, 'gt_act2':torch.tensor(gt_act2), 
            # 'positive_acts2':torch.tensor(positive_acts2), 'contrastive_acts2':torch.tensor(contrastive_acts2)
            })
            # out_dict.update(
            #     {
            #         'two_agent_instruct': instruct_updated,
            #         'two_agent_caption': caption_updated,
            #         'two_agent_act': torch.tensor([act, act_2]),
            #         'contrastive_sample': torch.tensor([contrastive_sample, contrastive_sample_2]),
            #     }
            # )
        return out_dict
    
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

    def instruct_select(self, instruct, instruct_2, option='12'):
        # option = 12 (both agents), 1 (ego), 2 (Agent-2), none
        if option=='12':
            instruct_updated = instruct.split(' and the following ego instruction: Make ')[0]+'and the following instructions: '
            instruct_updated += instruct.split('following ego instruction: ')[1].replace('.', ',')+'and make '
            instruct_updated += instruct_2.split('following ego instruction: Make ')[1].replace('the ego', 'Agent-2')
            # instruct_updated + instruct
        elif option=='1':
            instruct_updated = instruct
        elif option=='2':
            instruct_updated = instruct_2.split(' and the following ego instruction:')[0]+' and the following Agent-2 instruction: '
            instruct_updated += instruct_2.split(' and the following ego instruction: ')[1].replace('the ego', 'Agent-2')
        else:
            instruct_updated = instruct.split(' and the following ego instruction: Make ')[0]+'. '
        
        return instruct_updated



    def get_instruct_caption(self, template_dir, force_caption=False, acts=None, agent='1', positive_example_only=False):
        ## default:
        instruct = "Predict trajectories embeddings based on the following given scene embeddings. Give short answer without reasoning."
        instruct = f'<s>[INST] {instruct} '
        act = -1
        gt_act = -1
        positive_acts, contrastive_acts = [], []
        decision = '<Accepted>'
        caption = f"Decision: {decision}. Generated trajectories embeddings: "
        contrastive_sample=False


        try:
        # if True:
            with open(template_dir) as f:
                lines = f.readlines()
            templates = [json.loads(line) for line in lines]
            # contrastive_acts, gt_templates, contrastive_templates = self.get_contrastive_act(templates)
            gt_act, contrastive_acts, positive_acts, gt_templates, contrastive_templates, positive_templates = self.get_acts(templates)
            valid_contrastive=True
            valid_positive=True
            if self.new_eval:
                act = acts[int(agent)-1]
                if act == gt_act:
                    template = gt_templates[self.template_select]
                    contrastive_sample = False
                elif act in contrastive_acts:
                    act_arg = [i for i in range(len(contrastive_acts)) if contrastive_acts[i]==act][0]
                    template = contrastive_templates[act_arg]
                    contrastive_sample = True
                elif act in positive_acts:
                    act_arg = [i for i in range(len(positive_acts)) if positive_acts[i]==act][0]
                    template = positive_templates[act_arg]
                    contrastive_sample = False
                
                instruct = template['Instruction']
                instruct = f'<s>[INST] {instruct} '
                reason = template['Reasoning']
                decision = template['Decision']
                caption = f"{reason} Decision: {decision}. Generated trajectories embeddings: "
                if force_caption and contrastive_sample:
                    if 'Agent-2' in gt_templates[3]['Reasoning']:
                        caption = caption.split('Decision')[0] + gt_templates[3]['Reasoning'][gt_templates[3]['Reasoning'].find('Agent-2'):] + ' ' + caption[caption.find('Decision'):]
                elif force_caption and self.new_eval and self.new_eval_mode in ['pos1', 'pos12']:
                    if 'Agent-2' in gt_templates[3]['Reasoning']:
                        caption = caption.split('Decision')[0] + gt_templates[3]['Reasoning'][gt_templates[3]['Reasoning'].find('Agent-2'):] + ' ' + caption[caption.find('Decision'):]
    
            else:
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
                    
                if not self.train and self.contrastive and valid_contrastive: # contrastive eval
                    act = contrastive_acts[contrastive_class_arg]
                    template = contrastive_templates[contrastive_class_arg]
                    assert template['Direction_cls']==act
                    instruct = template['Instruction']
                    instruct = f'<s>[INST] {instruct} '
                    reason = template['Reasoning']
                    decision = template['Decision']
                    caption = f"{reason} Decision: {decision}."
                elif not self.train and self.positive_notgt and valid_positive: # positive eval
                    act = positive_acts[positive_class_arg]
                    template = positive_templates[positive_class_arg]
                    assert template['Direction_cls']==act
                    instruct = template['Instruction']
                    instruct = f'<s>[INST] {instruct} '
                    reason = template['Reasoning']
                    decision = template['Decision']
                    caption = f"{reason} Decision: {decision}."

                elif (self.train and not self.random_select and not self.contrastive) or (not self.contrastive and not self.positive_notgt) or (self.contrastive and not valid_contrastive) or (self.positive_notgt and not valid_positive): # regular gt training
                    template_select = self.template_select
                    if self.template_select==77:
                        template_select = 0
                    if self.template_select==99:
                        template_select = 3
                        instruct = templates[template_select]['Instruction']
                        instruct = f'<s>[INST] {instruct} '
                        instruct = instruct.split(' and the following')[0]
                        reason = templates[template_select]['Reasoning']
                        decision = templates[template_select]['Decision']
                        caption = f"{reason} Decision: {decision}. Generated trajectories embeddings: "
                        act = templates[template_select]['Direction_cls']
                    else:
                        instruct = templates[template_select]['Instruction']
                        instruct = f'<s>[INST] {instruct} '
                        reason = templates[template_select]['Reasoning']
                        decision = templates[template_select]['Decision']
                        caption = f"{reason} Decision: {decision}. Generated trajectories embeddings: "
                        act = templates[template_select]['Direction_cls']
                    


                if not valid_contrastive and self.contrastive and not self.train:
                    act=-1
                elif not valid_positive and self.positive_notgt and not self.train:
                    act=-1
                # if self.train:
            if True:
                if self.new_eval:
                    pass # template already selected
                else:
                    template_select = np.random.choice(5) if self.random_select else self.template_select
                    if self.template_select==77:
                        template_select = 0
                    template = templates[template_select]
                
                if self.pred_only or not self.act or (self.train and self.random_drop and np.random.choice(2, p=[0.75,0.25])==1): # random drop instructions
                    instruct = "Predict trajectories embeddings based on the following given scene embeddings."
                else:
                    instruct = template['Instruction']
                    instruct = "Generate trajectories embeddings based on the following given instruction and scene embeddings. "+instruct.split(": ")[1]
                instruct = f'<s>[INST] {instruct} '
                reason = template['Reasoning']
                decision = template['Decision']
                caption = f"{reason} Decision: {decision}. Generated trajectories embeddings: "
                act = template['Direction_cls']
                # if self.random_drop: # random drop instructions
                #     if np.random.choice(2, p=[0.5,0.5])==1 and 'Make' in instruct:
                #         instruct = instruct.split('Make')[0] + "No instruction. "
            if self.train and self.contrastive:
                if valid_contrastive and np.random.choice(2, p=self.train_contrastive_prob)==1 and not positive_example_only:
                    contrastive_template = contrastive_templates[contrastive_class_arg]
                    instruct = contrastive_template['Instruction']
                    instruct = "Generate trajectories embeddings based on the following given instruction and scene embeddings. "+instruct.split(": ")[1]
                    instruct = f'<s>[INST] {instruct} '
                    reason = contrastive_template['Reasoning']
                    decision = contrastive_template['Decision']
                    caption = f"{reason} Decision: {decision}. Generated trajectories embeddings: "
                    act = contrastive_template['Direction_cls']
                    contrastive_sample=True
                    if force_caption:
                        if 'Agent-2' in gt_templates[3]['Reasoning']:
                            caption = caption.split('Decision')[0] + gt_templates[3]['Reasoning'][gt_templates[3]['Reasoning'].find('Agent-2'):] + ' ' + caption[caption.find('Decision'):]
                
        except FileNotFoundError:
        #     # Do nothing and continue
            pass

        
        return instruct, caption, act, contrastive_sample, gt_act, positive_acts, contrastive_acts
        

    def include_agent_2(self, instruct, caption, contrastive_sample, instruct_2, caption_2, contrastive_sample_2):
         # caption
        if ' Agent-2' in caption:
            motion1 = caption.split(' Agent-2')[0]
        else:
            motion1 = caption.split(' Decision')[0]
        if ' Agent-2' in caption_2:
            motion2 = caption_2.split(' Agent-2')[0].replace('The ego', 'Agent-2')
        else:
            motion2 = caption_2.split(' Decision')[0].replace('The ego', 'Agent-2')
        caption_updated = motion1 + ' ' + motion2 + ' ' +caption[caption.find('Agent-2'):] # caption is the same
        if not contrastive_sample and not contrastive_sample_2: # ++
            # caption
            caption_updated = caption_updated.replace(caption_updated[caption_updated.find('Decision'): caption_updated.find('Generated')], "Ego Decision: <Accepted>. "+"Agent-2 Decision: <Accepted>. ")
            # instruction
            if self.train:
                who_to_instruct = np.random.choice(4, p=np.array([0.5, 0.2, 0.2, 0.1])) # instructing agents: 12, 1, 2, none
            else:
               who_to_instruct = {'12':0,'1':1,'2':2,'':3}[self.agents_instructed]
            if who_to_instruct==0: # 12
                instruct_updated = self.instruct_select(instruct, instruct_2, option='12')
                # instruct_updated = instruct.split(' and the following ego instruction: Make ')[0]+'and the following instructions: '
                # instruct_updated += instruct.split('following ego instruction: ')[1].replace('.', ',')+'and make '
                # instruct_updated += instruct_2.split('following ego instruction: Make ')[1].replace('the ego', 'Agent-2')
                # instruct_updated + instruct
            elif who_to_instruct==1: # 1
                instruct_updated = self.instruct_select(instruct, instruct_2, option='1')
            elif who_to_instruct==2: # 2
                instruct_updated = self.instruct_select(instruct, instruct_2, option='2')
                # instruct_updated = instruct_2.split(' and the following ego instruction:')[0]+' and the following Agent-2 instruction: '
                # instruct_updated += instruct_2.split(' and the following ego instruction: ')[1].replace('the ego', 'Agent-2')
            elif who_to_instruct==3: # none
                instruct_updated = self.instruct_select(instruct, instruct_2, option='none')
                # instruct_updated = instruct.split(' and the following ego instruction: Make ')[0]+'. '
        elif not contrastive_sample and contrastive_sample_2: # +-, [12, 2]
            caption_updated = caption_updated.replace(caption_updated[caption_updated.find('Decision'): caption_updated.find('Generated')], "Ego Decision: <Accepted>. "+"Agent-2 Decision: <Rejected>. ")
            if self.train:
                who_to_instruct = np.random.choice(2, p=np.array([0.5, 0.5]))
            else:
                who_to_instruct = {'12':0,'1':1,'2':2,'':3}[self.agents_instructed]
            if who_to_instruct==0:
                instruct_updated = self.instruct_select(instruct, instruct_2, option='12')
            else:
                instruct_updated = self.instruct_select(instruct, instruct_2, option='2')
        elif contrastive_sample and not contrastive_sample_2: # -+, [12, 1]
            caption_updated = caption_updated.replace(' Decision', '. Decision').replace(caption_updated[caption_updated.find('Decision'): caption_updated.find('Generated')], "Ego Decision: <Rejected>. "+"Agent-2 Decision: <Accepted>. ")
            if self.train:
                who_to_instruct = np.random.choice(2, p=np.array([0.5, 0.5]))
            else:
                who_to_instruct = {'12':0,'1':1,'2':2,'':3}[self.agents_instructed]
            if who_to_instruct==0:
                instruct_updated = self.instruct_select(instruct, instruct_2, option='12')
            else:
                instruct_updated = self.instruct_select(instruct, instruct_2, option='1')
        elif contrastive_sample and contrastive_sample_2: # -- [12]
            caption_updated = caption_updated.replace(' Decision', '. Decision').replace(caption_updated[caption_updated.find('Decision'): caption_updated.find('Generated')], "Ego Decision: <Rejected>. "+"Agent-2 Decision: <Rejected>. ")
            instruct_updated = self.instruct_select(instruct, instruct_2, option='12')
        
        instruct_updated = instruct_updated.replace('embeddingsand', 'embeddings and')

        return instruct_updated, caption_updated


    def get_acts(self, templates):
        ## loading and filtering ground truth and contrastive acts
        gt_templates, contrastive_templates, positive_templates = [], [], []
        direction_classes = self.direction_classes
        gt_act = templates[0]['Direction_cls']
        contrastive_acts = []
        positive_acts_not_gt = []
        for template_i in templates:
            if template_i['Label']=='gt':
                gt_templates.append(template_i)
            if template_i['Label']=='negative':
                contrastive_acts.append(template_i['Direction_cls'])
                contrastive_templates.append(template_i)
            if template_i['Label']=='possible direction not gt':
                positive_acts_not_gt.append(template_i['Direction_cls'])
                positive_templates.append(template_i)
        if 'straight' in direction_classes[gt_act] or True in ['straight' in direction_i for direction_i in direction_classes[positive_acts_not_gt]]:
            contrastive_templates = [contrastive_templates[i] for i in range(len(contrastive_acts)) if 'straight' not in direction_classes[contrastive_acts[i]]]
            contrastive_acts = [act_i for act_i in contrastive_acts if 'straight' not in direction_classes[act_i]]
        
        return gt_act, contrastive_acts, positive_acts_not_gt, gt_templates, contrastive_templates, positive_templates
    
    def augment_short_instruct(self, instruct):
        return instruct
        
        # traj = torch.tensor(data['traj'])
        # rel = torch.tensor(data['rel'])
        # disc_rel = torch.tensor(data['disc_rel'])
        # disc_traj = torch.tensor(data['disc_traj'])
        # anchors = torch.tensor(data['anchors'])
        # object_type = torch.tensor(data['object_type'])

        # return traj, rel, disc_rel, disc_traj, anchors, object_type
        # file_name = self.data_list[idx].split('/')[-1].split('.npz')[0]
        # out_dict = {
        #         'traj': traj,
        #         'ego_state': data['ego'][0],
        #         'neighbors_state': np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0),
        #         # 'map_lanes': data['map_lanes'][:, :, :200:2],
        #         # 'map_crosswalks': data['map_crosswalks'][:, :, :100:2],
        #         'map_lanes': data['map_lanes'],
        #         'map_crosswalks': data['map_crosswalks'],
        #         'object_type':object_type,
        #         'ego': data['ego'],
        #         'ground_truth': data['gt_future_states'],
        #         'neighbors': data['neighbors'],
        #         'file_name': file_name,
        #         'data_object_type' : data_object_type
        #     }
        # if self.act:
        #     valid_action = np.load(self.data_list_act_dir+'/'+file_name+'.npy', allow_pickle = True)
        #     if bool(valid_action[0]):
        #         valid_action, turn, move1, move2, contrastive_turn = valid_action
        #     else:
        #         valid_action, turn, move1, move2, contrastive_turn = valid_action[0], '', '', '', '' # no valid action instruction found
        #     out_dict.update(
        #         {
        #             'valid_action':bool(valid_action),
        #             'turn': turn, # Turning instruction calculated by comparing the most future step with the observed current step
        #             'move1': move1, # Motion instruction calculated using the FIRST half time lapse of future steps
        #             'move2': move2, # Motion instruction calculated using the SECOND half time lapse of future steps
        #             'contrastive_turn': contrastive_turn,
        #         })
            
        # return out_dict
        # return {
        #     'traj': traj,
        #     'neighbors': data['neighbors'],
        #     'map_lanes': data['map_lanes'],
        #     'map_crosswalks': data['map_crosswalks'],
        #     'object_type':object_type,
        #     'ego': data['ego'],
        #     'ground_truth': data['gt_future_states'],
        # }
        # else:    
        #     return {
        #         'traj':traj,
        #     }
    
        # data = np.load(self.data_list[idx], allow_pickle = True)
        # traj_obs = torch.tensor(data['ego'][:,:,:2])
        # traj_pred = torch.tensor(data['gt_future_states'][:,:,:2])
        # rel_obs = torch.tensor(data['rel_obs'])
        # rel_pred = torch.tensor(data['rel_pred'])
        # rel_obs_1d = torch.tensor(data['rel_obs_1d'])
        # rel_pred_1d = torch.tensor(data['rel_pred_1d'])
        
        # if self.load_all:
        #     object_type = torch.tensor(data['object_type'])
        #     bin_edges = torch.tensor(data['bin_edges'])
        #     continous_grid = torch.tensor(data['continous_grid'])
        #     euc_mat = torch.tensor(data['euc_mat'])
            
        #     rel_1d = torch.cat((rel_obs_1d, rel_pred_1d), dim=1)
        #     rel_disc = get_discrete_from_grid_1d(rel_1d, euc_mat)
        #     rel_cont = continous_grid[rel_disc]
        #     traj_disct = rel2traj(rel_cont, traj_obs[:,0,:])
        #     traj_obs_disc = traj_disct[:,:self.obs_len]
        #     traj_pred_disc = traj_disct[:,self.obs_len:]
        #     return {
        #     'traj_obs':traj_obs,
        #     'traj_pred':traj_pred,
        #     'traj_obs_disc':traj_obs_disc,
        #     'traj_pred_disc':traj_pred_disc,
        #     'rel_obs_1d': rel_obs_1d,
        #     'rel_pred_1d': rel_pred_1d,
        #     'continous_grid': continous_grid,
        #     'euc_mat':euc_mat,
        #     }
        #     # return traj_obs, traj_pred, rel_obs, rel_pred, rel_obs_1d, rel_pred_1d, object_type, bin_edges, continous_grid, euc_mat

        # # return traj_obs, traj_pred, rel_obs, rel_pred, rel_obs_1d, rel_pred_1d

    
        # return {
        #     # 'rel_obs':rel[:,:,:self.obs_len],
        #     # 'rel_pred':rel[:,:,self.obs_len:self.obs_len+self.pred_len],
        #     # 'init_traj':init_traj,
        #     'rel_obs_1d': rel_obs_1d,
        #     # 'rel_pred_1d': rel_1d[:,self.obs_len:self.obs_len+self.pred_len]
        #     'rel_pred_1d': rel_pred_1d
        # }
