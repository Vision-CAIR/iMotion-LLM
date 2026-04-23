import os
import sys
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
import logging
from minigpt4.datasets.datasets.complex_instruction_dataset_helper import *
from typing import Dict, List, Sequence
from instructions.direction_instructions import DirectionClassifier
from instructions.extract_instructions import futureNavigation

from pathlib import Path
import pickle

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imotion_paths import first_existing_path


def _strip_glob(path_str):
    return path_str.split("*")[0].rstrip("/")


def _dataset_dir(path_str):
    return Path(_strip_glob(path_str))


def _sibling_dir(path_str, suffix):
    base_dir = _dataset_dir(path_str)
    return base_dir.with_name(f"{base_dir.name}{suffix}")


def _env_or_existing(env_name, *candidates):
    env_value = os.environ.get(env_name)
    if env_value:
        return str(Path(env_value).expanduser())
    return str(first_existing_path(*candidates))


try:
    from mtr.datasets.dataset import DatasetTemplate
    from mtr.utils import common_utils
    from mtr.config import cfg
    import time
    from mtr.config import cfg_from_list, cfg_from_yaml_file, log_config_to_file
    from mtr.config import cfg as cfg_mtr

    print("✅ mtr module loaded successfully.")

except ImportError:
    print("❌ mtr model is not supported (module not found).")

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


class TrajAlignDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, act=False, act_json=True, template_select=4, contrastive=False, train=False, random_select=False, positive_notgt=False, two_agent=False, new_eval=False, new_eval_mode='traj_pred', agents_instructed='', random_drop=True, return_meta_data=False, stage_2='', of_samples=False, nuplan_complex=False, num_classes=8, weighted_sampling=False, debug=False, two_agent_mode="i2", nuplan_direction=False, new_twoAgent_eval_instruct_select=None):
    # def __init__(self, **kwargs):
        super(TrajAlignDataset, self).__init__()
        self.new_twoAgent_eval_instruct_select =new_twoAgent_eval_instruct_select
        self.nuplan_complex = nuplan_complex
        self.nuplan_direction=nuplan_direction
        # self.nuplan_eval_instruct_select = nuplan_eval_instruct_select
        if nuplan_complex:
            list_of_instructs, list_of_np_dirs = get_nuplan_instructs_list(data_dir, 'gpt_data_101124', nuplan_direction)
            self.data_list = list_of_np_dirs
            self.instruct_list = list_of_instructs
            if new_eval_mode in ['safe_with_context', 'safe_no_context', 'unsafe_with_context', 'unsafe_no_context']:
                self.new_eval_mode = new_eval_mode # safe_with_context, safe_no_context, unsafe_with_context, unsafe_no_context
                list_of_np_dirs_ , list_of_instructs_ = [], []
                for i in range(len(list_of_instructs)):
                    if new_eval_mode=='safe_with_context' and list_of_instructs[i]['safe'] and list_of_instructs[i]['category'] == 'With Context':
                        list_of_np_dirs_.append(list_of_np_dirs[i])
                        list_of_instructs_.append(list_of_instructs[i])
                    elif new_eval_mode=='safe_no_context' and list_of_instructs[i]['safe'] and list_of_instructs[i]['category'] == 'Without Context':
                        list_of_np_dirs_.append(list_of_np_dirs[i])
                        list_of_instructs_.append(list_of_instructs[i])
                    elif new_eval_mode=='unsafe_with_context' and (not list_of_instructs[i]['safe']) and list_of_instructs[i]['category'] == 'With Context':
                        list_of_np_dirs_.append(list_of_np_dirs[i])
                        list_of_instructs_.append(list_of_instructs[i])
                    elif new_eval_mode=='unsafe_no_context' and (not list_of_instructs[i]['safe']) and list_of_instructs[i]['category'] == 'Without Context':
                        list_of_np_dirs_.append(list_of_np_dirs[i])
                        list_of_instructs_.append(list_of_instructs[i])
                self.data_list, self.instruct_list = list_of_np_dirs_, list_of_instructs_
            else:
                self.new_eval_mode = None
            return
        self.return_meta_data = return_meta_data
        self.act = act
        self.act_json = act_json
        self.data_dir = data_dir
        if act and act_json:
            self.data_list = glob.glob(data_dir[:-2]+'_json'+'/*')
        else:
            # self.data_list = glob.glob(data_dir)
            data_list_ = glob.glob('/'.join(data_dir.split('/')[:-1])+'_templateLLM/*')
            self.data_list = [data_dir[:-1] + data_list_[i].split('/')[-1][:-4] + '.npz' for i in range(len(data_list_))]
            # [data_list_[i].split('/')[-1][:-4] + '.npz' for i in range(len(data_list_))]
        
            
        logging.info(f"number of {'train' if train else 'eval'} samples found: {len(self.data_list)}")
            # data_list_

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
        # if random_drop:
        self.random_drop=random_drop
        self.of_samples = of_samples
        if of_samples:
            self.of_samples_llmdir = _env_or_existing(
                "IMOTION_LLM_SYNTH_TEMPLATE_DIR",
                _sibling_dir(data_dir, "_synth_templateLLM"),
            )
            self.of_samples_llm_files = glob.glob(self.of_samples_llmdir+'/*')
            self.of_samples_npz = _env_or_existing(
                "IMOTION_LLM_SYNTH_NPZ_DIR",
                _sibling_dir(data_dir, "_synth_npz"),
            )
            self.of_samples_len = len(self.of_samples_llm_files)
            # self.of_samples_llmdir = "/ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov_synth_templateLLM_temp"
            # self.of_samples_npz = "/ibex/project/c2278/felembaa/datasets/waymo/gameformer/training_28nov_synth_templateLLM"


        self.contrastive = contrastive
        self.positive_notgt = positive_notgt
        self.train = train
        self.random_select = random_select

        direction_classifier = DirectionClassifier(num_classes=num_classes)
        self.direction_classes = np.array(direction_classifier.classes)
        if num_classes==8:
            train_contrastive_act_distribution = np.array([18.23, 2.7, 3.88, 3.91, 8.36, 16.04, 23.75, 23.12])
            train_positive_notgt_distribution = np.array([7.04, 6.76, 15.34, 9.14, 39.00, 13.54, 4.32, 4.85])
            train_groundtruth_act_distribution = [1.62, 55.77, 3.25, 3.71, 16.67, 17.49, 0.09, 1.40]
        elif num_classes==5:
            train_contrastive_act_distribution = np.array([20, 20, 20, 20, 20]) # for simplification assume equal distribution
            train_positive_notgt_distribution = np.array([20, 20, 20, 20, 20])
            train_groundtruth_act_distribution = [1.30, 66.90, 14.27, 17.17, 0.35]
        self.train_contrastive_prob = [0.7, 0.3]

        # self.train_contrastive_prob = [0.5, 0.5]
        # self.train_prob = [0.7, 0.3] if self.contrastive else [1.0,0.0]
        self.contrastive_prob = (1/train_contrastive_act_distribution)/(sum(1/train_contrastive_act_distribution))
        self.positive_prob = (1/train_positive_notgt_distribution)/(sum(1/train_positive_notgt_distribution))
        
        self.two_agent=two_agent
        self.two_agent_mode = two_agent_mode
        if two_agent and ('i1' not in two_agent_mode and 'i2' not in two_agent_mode):
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
            print(f">>> Eval mode: {self.new_eval_mode}")
            # print('---')
        if weighted_sampling:
            self.initialize_sample_weights_with_known_distribution()


    def _extract_act_for_sample(self, idx):
        """
        Extracts the ground-truth action class from the template file for the sample at index `idx`.
        Uses same file loading logic as get_item method.
        
        Args:
            idx (int): Index of the sample in self.data_list
        
        Returns:
            int: The ground-truth action class or -1 if not found
        """
        try:
            np_filename = self.data_list[idx]
            root_dir = '/'.join(self.data_dir.split('/')[:-1])
            filename = np_filename.split('/')[-1].replace('.npz', '')
            template_dir = root_dir + '_templateLLM/' + filename + '.txt'
            
            with open(template_dir, 'r') as f:
                for line in f:
                    template = json.loads(line)
                    if template.get('Label') == 'gt':
                        return template.get('Direction_cls', -1)
            
            return -1
        except Exception as e:
            if idx % 10000 == 0:  # Limit logging to avoid spam
                print(f"Warning for idx {idx}: {str(e)}")
            return -1


    def initialize_sample_weights_with_known_distribution(self):
        """
        Initializes sample weights for weighted sampling using known class distribution.
        Uses log-scale weighting to handle extreme class imbalance.
        """
        print("Initializing sample weights with known class distribution (log scale)...")
        
        # Known class distribution for 5 classes
        act_counts = np.array([16748, 863910, 184286, 221762, 4575], dtype=np.float32)
        total_samples = np.sum(act_counts)
        
        # Calculate class probabilities
        class_probs = act_counts / total_samples
        
        # Calculate log-scale weights
        # Using log(1/p) = -log(p) as the weight
        if True:
            # Adding 1 inside log to ensure positive values
            log_weights = -np.log(class_probs + 1e-10)
        if False:
            # For debugging:
            log_weights = 1.0 / (class_probs ** 5)  # Using power of 5 makes weights even more extreme
        
        # Alternative: Use log(N/n_i) where N is total samples and n_i is class count
        # log_weights = np.log(total_samples / act_counts)
        
        # Normalize weights to sum to 1
        log_weights = log_weights / np.sum(log_weights)
        
        # Print class distribution and statistics
        print("\nClass Distribution:")
        for i in range(len(act_counts)):
            print(f"Class {i}: {act_counts[i]:.0f} samples ({class_probs[i]*100:.2f}%), log weight: {log_weights[i]:.6f}")
        
        # Default weight (use median of log weights)
        default_weight = np.median(log_weights)
        print(f"Default weight for unidentified classes: {default_weight:.6f}")
        
        print("\nAssigning sample weights...")
        
        # Pre-allocate numpy array with default weights
        self.sample_weights = np.full(len(self.data_list), default_weight, dtype=np.float32)
        
        # Process samples in batches for efficiency
        batch_size = 1000
        num_batches = (len(self.data_list) + batch_size - 1) // batch_size
        
        # Cache for template results
        template_cache = {}
        valid_assignments = 0
        for batch_idx in tqdm(range(num_batches), desc="Computing sample weights"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.data_list))
            
            for idx in range(start_idx, end_idx):
                np_filename = self.data_list[idx]
                filename = np_filename.split('/')[-1].replace('.npz', '')
                
                # Use cache for efficiency
                if filename in template_cache:
                    act_class = template_cache[filename]
                else:
                    act_class = self._extract_act_for_sample(idx)
                    template_cache[filename] = act_class
                    
                if 0 <= act_class < len(act_counts):
                    self.sample_weights[idx] = log_weights[act_class]
                    valid_assignments += 1
            
            # Clear cache periodically to manage memory
            if len(template_cache) > 100000:
                template_cache.clear()
        # import pdb; pdb.set_trace()
        # Final normalization
        weight_sum = np.sum(self.sample_weights)
        if weight_sum > 0:
            self.sample_weights = self.sample_weights / weight_sum
        
        print(f"Weighted sampling initialized with {len(self.sample_weights)} weights")
        print(f"Valid class assignments: {valid_assignments} ({valid_assignments/len(self.data_list)*100:.2f}%)")
        print(f"Min weight: {np.min(self.sample_weights)}, Max weight: {np.max(self.sample_weights)}")
        print(f"Mean weight: {np.mean(self.sample_weights)}, Std dev: {np.std(self.sample_weights)}")
        print(f"Weight distribution: min={np.min(log_weights):.4f}, max={np.max(log_weights):.4f}, ratio={np.max(log_weights)/np.min(log_weights):.2f}x")

    def nuplan_complex_init(self, data_dir, act, act_json, template_select, contrastive, train, random_select, positive_notgt, two_agent, new_eval, new_eval_mode, agents_instructed, random_drop, return_meta_data, stage_2, of_samples, nuplan_complex):
        self.return_meta_data = return_meta_data
        self.act = act
        self.act_json = act_json
        self.data_dir = data_dir
        self.data_list = []
        data_list_1 = glob.glob(data_dir)
        for dir1 in data_list_1:
            dir1_list_json = glob.glob(dir1+'/gpt_data_101124/*')
            dir1_list_npz = glob.glob(dir1+'/npz/*')
            for json_file in dir1_list_json:
                npz_file = json_file.replace(json_file.split('/')[-2], 'npz').replace('.json','.npz')
                if npz_file in dir1_list_npz:
                    self.data_list.append(npz_file)
                else:
                    raise f"NOT FOUND: {npz_file}"
        print(f'Found {len(self.data_list)} complex instruction scenarios (could contain multiple instruction-caption pairs)')
        self.gpt_dir_name = 'gpt_data_101124'


    def nuplan_complex_getitem(self, idx):

        np_filename = self.data_list[idx]
        # json_filename = np_filename.replace('.npz', '.json').replace('npz', self.gpt_dir_name)
        
        data = np.load(np_filename, allow_pickle = True)
        traj_obs = torch.tensor(data['ego'][:,:,:2])
        traj_pred = torch.tensor(data['gt_future_states'][:,:,:2])
        traj = torch.cat((traj_obs, traj_pred), dim=1)
        object_type = (data['ego'][:,-1,8:].argmax(-1)+1) * data['ego'][:,-1,8:].sum(-1).astype(int)
        data_object_type = data['object_type']
        map_lanes = data['map_lanes'][:, :6, :][...,:200:2,:]
        map_crosswalks = data['map_crosswalks'][:, :, :100:2]
        ego = data['ego']
        gt_future_states = data['gt_future_states']
        neighbors = data['neighbors']

        # if True: # single agent only data
        #     traj_obs = traj_obs[:1]
        #     traj_pred = traj_pred[:1]
        #     traj = traj[:1]
        #     object_type = object_type[:1]
        #     data_object_type = data_object_type[:1]
        #     map_lanes = map_lanes[:1]
        #     map_crosswalks = map_crosswalks[:1]
        #     ego = ego[:1]
        #     gt_future_states = gt_future_states[:1]
        #     neighbors = np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0)
            

        out_dict = {
            'traj': traj,
            'ego_state': data['ego'][0],
            'neighbors_state': np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0),
            # 'map_lanes': data['map_lanes'][:, :, :200:2],
            'map_lanes': map_lanes,
            'map_crosswalks': map_crosswalks,
            'object_type':object_type,
            'ego': ego,
            'ground_truth': gt_future_states,
            'neighbors': neighbors,
            'data_object_type' : data_object_type,
        }

        instruct = self.instruct_list[idx]['instruction']
        caption = self.instruct_list[idx]['reasoning']
        safe = self.instruct_list[idx]['safe']
        category = self.instruct_list[idx]['category']
        instruct = f"<s>[INST] Generate trajectories embeddings based on the following given instruction and scene embeddings. {instruct} "
        accepted_or_rejected = "<Accepted>" if safe else "<Rejected>"
        caption = f"{caption} Decision: {accepted_or_rejected}. Generated trajectories embeddings: "
        
        # '<s>[INST] ' ' '
        # ' Decision: <Rejected><Accepted>. Generated trajectories embeddings: '

        if self.nuplan_direction:
            navigation_extractor = futureNavigation(normalize_track=True, num_classes=5)
            direction_classes = DirectionClassifier(num_classes=5).classes
            act_dict = navigation_extractor.get_navigation_dict(torch.tensor(traj_pred[0]))
            act_cls = act_dict['direction 0.1to8_cls']
            direction_classes = ['stay stationary', 'move straight', 'turn right', 'turn left', 'take left U-turn']
            instruct = instruct.split('embeddings.')[0]+'embeddings. ' + direction_classes[act_cls]+'. '

        out_dict.update(
            {
                'instruct': instruct,
                'caption': caption,
                'contrastive_sample' : not safe,
                'file_name': self.data_list[idx].split('/')[-1].replace('.npz',f'_{idx}')
            }
        )
        # out_dict.update({
        #     'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
        #     'contrastive_sample': contrastive_sample or of_sample, # used to indicate ignoring the generated trajectory
        #     'act': torch.tensor(act),
        #     'instruct': instruct,
        #     'file_name': filename, #*
        #     'caption' : caption, #*
        # })
        return out_dict


    def __len__(self):
        return len(self.data_list)
   
    def __getitem__(self, idx):
        if self.nuplan_complex:
            return self.nuplan_complex_getitem(idx)

        if self.new_eval:
            np_filename = self.root_dir + '_'.join(self.data_list[idx].split('_')[:-1])+'.npz'
        else:
            np_filename = self.data_list[idx]
        
        
        # if self.of_samples and random.randint(0,3)==2:
        if self.of_samples and np.random.choice(2, p=[0.7,0.3])==1: # 0.3*0.7 = 0.21 probability of pos, 0.49 GT, 0.3 neg
            of_idx = int(idx/self.__len__()*self.of_samples_len + random.randint(-100, 100))
            of_idx = max(0, min(of_idx, self.of_samples_len-1))
            of_filename = self.of_samples_llm_files[of_idx]
            np_filename = '/'.join(np_filename.split('/')[:-1])+"/"+of_filename.replace('.txt', '.npz').split('/')[-1]
            use_of_samples = True
        else:
            use_of_samples = False
            
            

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
            instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts, act2 = self.get_instruct_caption(template_dir, force_caption=False, acts=acts, agent='1')

            if self.two_agent:
                if acts[1]==-1:
                    try:
                        two_agent_template_dir = os.environ.get("IMOTION_LLM_TWO_AGENT_TEMPLATE_DIR", "")
                        if not two_agent_template_dir:
                            raise FileNotFoundError("Set IMOTION_LLM_TWO_AGENT_TEMPLATE_DIR for two-agent evaluation.")
                        act_2_dir = str(Path(two_agent_template_dir) / f"{filename.split('_act')[0]}.txt")
                        # Read:
                        template_a2 = []
                        with open(act_2_dir, 'r') as f:
                            for line in f:
                                template_a2.append(json.loads(line))

                        if self.new_twoAgent_eval_instruct_select=='both':
                                instruct_updated = template_a2[0]['Instruction']
                                caption_updated = template_a2[0]['Reasoning']
                                if self.new_eval and self.new_eval_mode in ['pos1', 'neg1']:
                                    if 'while' in instruct_updated:
                                        instruct = instruct[:-2]+' while'+instruct_updated.split('while')[1]+' '
                                    # instruct = instruct.split(' Make')[0] + ' Make' + instruct_updated.split(' Make')[1]+' '
                                    # caption = caption_updated + ' Decision' + caption.split('. Decision')[1]
                                else:
                                    instruct = instruct.split(' Make')[0] + ' Make' + instruct_updated.split(' Make')[1]+' '
                                    caption = caption_updated + ' Decision' + caption.split('. Decision')[1]
                                acts[1] = template_a2[0].get('Direction_cls', -1)
                                act_2 = acts[1]
                    except:
                        act_2 = -1
                        print(f"{filename.split('_act')[0]+'.txt'} not found")
                        pass
                # if np_filename in self.filenames_T.keys():
                #     filename_2 = self.filenames_T[np_filename].split('/')[-1].replace('.npz','')
                #     template_dir_2 = root_dir+'_templateLLM/'+filename_2+'.txt'
                #     instruct_2, caption_2, act_2, contrastive_sample_2, of_sample_2, gt_act2, positive_acts2, contrastive_acts2 = self.get_instruct_caption(template_dir_2, acts=acts, agent='2')
                #     instruct_updated, caption_updated = self.include_agent_2(instruct, caption, contrastive_sample, instruct_2, caption_2, contrastive_sample_2)
        else:
            root_dir = '/'.join(self.data_dir.split('/')[:-1])
            if not use_of_samples:
                filename = self.data_list[idx].split('/')[-1].replace('.npz','')
            else:
                filename = np_filename.split('/')[-1].replace('.npz','')
            # agent_dir = root_dir+'_agentJsons/'+filename+'.json'
            # map_dir = root_dir+'_mapJsons/'+filename+'.json'
            if use_pos_augmented_sample or use_of_samples:
                # Done: Change the template LLM to the other feasible template LLM, ensure that the sample loaded will not be "infeasible"
                # template_dir = self.stage_2_dir+'_templateLLM/'+filename+'.txt'
                template_dir = of_filename
                instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts = self.get_instruct_caption(template_dir, force_caption=self.two_agent, positive_example_only=True)
                
                data_ = np.load(self.of_samples_npz + '/' + of_filename.replace('.txt', '.npz').split('/')[-1])
                data_ = data_['synth_traj']
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
                instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts, act_2 = self.get_instruct_caption(template_dir, force_caption=self.two_agent)

            if False:
                if self.two_agent:
                    if np_filename in self.filenames_T.keys():
                        filename_2 = self.filenames_T[np_filename].split('/')[-1].replace('.npz','')
                        template_dir_2 = root_dir+'_templateLLM/'+filename_2+'.txt'
                        instruct_2, caption_2, act_2, contrastive_sample_2, of_sample_2, gt_act2, positive_acts2, contrastive_acts2 = self.get_instruct_caption(template_dir_2)
                        # Options: 0:No control, 1:control agent-1, 2:control agent-2, 3:control both agents
                        instruct_updated, caption_updated = self.include_agent_2(instruct, caption, contrastive_sample, instruct_2, caption_2, contrastive_sample_2)
                    else:
                        filename_2 = ''
                        instruct_updated = instruct
                        caption_updated = caption.replace(caption[caption.find('Decision'): caption.find('Generated')], ". Ego Decision: <Accepted>. ")
                        contrastive_sample_2 = False
                        act_2 = -1
                        gt_act2, positive_acts2, contrastive_acts2 = -1, [], []
            gt_act2 = act_2

        if 'nuplan' in self.data_dir:
            out_dict = {
                'traj': traj,
                'ego_state': data['ego'][0],
                'neighbors_state': np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0),
                # 'map_lanes': data['map_lanes'][:, :, :200:2],
                'map_lanes': data['map_lanes'][:, :6, :],
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
                'contrastive_sample': contrastive_sample or of_sample, # used to indicate ignoring the generated trajectory
                'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
            }
            # return out_dict
        else:
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
                    'act': torch.tensor(act),
                    'contrastive_sample': contrastive_sample or of_sample, # used to indicate ignoring the generated trajectory
                    'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
                    'act_2': torch.tensor(act_2)
                }
        if self.return_meta_data:
            out_dict.update({'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts)})
        if False:
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
        # return out_dict['instruct']

        if self.two_agent and '1of2' in self.two_agent_mode:
            if 'i2' in self.two_agent_mode:
                instruct_both_agents = np.random.choice(2, p=np.array([0.5, 0.5]))==1
            else:
                instruct_both_agents = False
            if not instruct_both_agents:
                instruct_agent_select = np.random.choice(2, p=np.array([0.5, 0.5]))
                if instruct_agent_select==0 or out_dict['contrastive_sample']:
                    out_dict['instruct'] = out_dict['instruct'].split(' while')[0]+'. ' # ego instruct
                    out_dict['act_2'] = torch.tensor(-1)
                    out_dict['caption'] = out_dict['caption'].split(', while Agent-2')[0]+'. Decision' + out_dict['caption'].split('. Decision')[1]
                else:
                    out_dict['instruct'] = out_dict['instruct'].split(' Make')[0] + out_dict['instruct'].split(' while')[1] # Agent-2 instruct
                    out_dict['act'] = torch.tensor(-1)
                    out_dict['contrastive_sample']=False
                    out_dict['caption'] = 'Agent-2' + out_dict['caption'].split(', while Agent-2')[1]
            
            if 'iNone' in self.two_agent_mode:
                instruct_none = np.random.choice(2, p=np.array([0.5, 0.5]))==1
                if instruct_none:
                    out_dict['instruct'] = out_dict['instruct'].split(' given instruction and')[0] + out_dict['instruct'].split(' based on the following given instruction and')[1].split('. ')[0]+'.'
                    out_dict['caption'] = out_dict['caption'].split('<Accepted>. ')[1] if '<Accepted>' in  out_dict['caption'] else out_dict['caption'].split('<Rejected>. ')[1]
                    out_dict['contrastive_sample']=False


        return out_dict
    
    def get_new_eval_act(self, filename):
        filename_ = filename
        # filename_ = [k for k in self.meta_eval[self.new_eval_mode].keys() if filename.split('/')[-1][:-4] in k][0]
        # self.meta_eval[self.new_eval_mode][filename_]
        act = self.meta_eval[filename_][self.new_eval_mode]

        # act = self.meta_eval[self.new_eval_mode][filename_][self.new_eval_mode]
        if self.ego_act_only or self.new_twoAgent_eval_instruct_select:
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

    # Function to load and parse the JSON data with filtering
    # def load_complex_data(self, json_dir):
    #     output_dir = {"safe":[], "safe_no_context":[], "unsafe":[], "unsafe_no_context":[]}
    #     with open(json_dir, 'r') as file:
    #         data = json.load(file)
        
    #     # safe
    #     for entry in data[0]['data']:
    #         entry['instructions'][0]
    #         if 'without' in entry['instructions'][0]['category'].lower():
            
    #         else:


    #         break
    #     # List to store filtered data samples
    #     filtered_samples = []
        
    #     # Iterate through each entry in the JSON data
    #     for entry in data:
    #         break
    #         for sample in entry["data"]:
    #             for instruction_set in sample["instructions"]:
    #                 # Check if 'safe' matches 'safe_instruction'
    #                 if instruction_set["safe"] == instruction_set["safe_instruction"]:
    #                     # Extract relevant data into a dictionary if they match
    #                     sample_dict = {
    #                         "instruction": instruction_set["instruction"],
    #                         "reasoning": instruction_set["reasoning"],
    #                         "safe": instruction_set["safe"],
    #                         "safe_instruction": instruction_set["safe_instruction"],
    #                         "category": instruction_set["category"]
    #                     }
    #                     filtered_samples.append(sample_dict)
        
    #     return filtered_samples

    def get_instruct_caption(self, template_dir, force_caption=False, acts=None, agent='1', positive_example_only=False):
        ## default:
        instruct = "Predict trajectories embeddings based on the following given scene embeddings. Give short answer without reasoning."
        instruct = f'<s>[INST] {instruct} '
        act = -1
        act2 = -1
        gt_act = -1
        positive_acts, contrastive_acts = [], []
        decision = '<Accepted>'
        caption = f"Decision: {decision}. Generated trajectories embeddings: "
        contrastive_sample=False
        of_sample=False


        try:
        # if True:
            with open(template_dir) as f:
                lines = f.readlines()
            # print('1')
            templates = [json.loads(line) for line in lines]
            # contrastive_acts, gt_templates, contrastive_templates = self.get_contrastive_act(templates)
            gt_act, contrastive_acts, positive_acts, gt_templates, contrastive_templates, positive_templates, gt_act2 = self.get_acts(templates)
            valid_contrastive=True
            # valid_positive=True
            valid_positive=False
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
                if self.of_samples:
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
                    if self.two_agent:
                        raise "not impelemented"
                else:
                    instruct = template['Instruction']
                    instruct = "Generate trajectories embeddings based on the following given instruction and scene embeddings. "+instruct.split(": ")[1]
                instruct = f'<s>[INST] {instruct} '
                reason = template['Reasoning']
                decision = template['Decision']
                caption = f"{reason} Decision: {decision}. Generated trajectories embeddings: "
                act = template['Direction_cls']
                if self.two_agent and 'Direction_cls_a2' in template:
                    act2 = template['Direction_cls_a2']
                else:
                    act2=-1
                # if self.random_drop: # random drop instructions
                #     if np.random.choice(2, p=[0.5,0.5])==1 and 'Make' in instruct:
                #         instruct = instruct.split('Make')[0] + "No instruction. "
            # of_sample = False
            if self.train and self.contrastive and self.positive_notgt and (not positive_example_only):
                scenario_type_select = np.random.choice(3, p=[1/3, 1/3, 1/3])
                if valid_positive and scenario_type_select == 1:
                    selected_template = positive_templates[positive_class_arg]
                    of_sample = True
                elif valid_contrastive and scenario_type_select == 2:
                    selected_template = contrastive_templates[contrastive_class_arg]
                    contrastive_sample=True
                else:
                    selected_template=None
                if selected_template is not None:
                    instruct = selected_template['Instruction']
                    instruct = "Generate trajectories embeddings based on the following given instruction and scene embeddings. "+instruct.split(": ")[1]
                    instruct = f'<s>[INST] {instruct} '
                    reason = selected_template['Reasoning']
                    decision = selected_template['Decision']
                    caption = f"{reason} Decision: {decision}. Generated trajectories embeddings: "
                    act = selected_template['Direction_cls']
            elif self.train and self.contrastive:
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
                    act
                    if False:
                        if force_caption:
                            if 'Agent-2' in gt_templates[3]['Reasoning']:
                                caption = caption.split('Decision')[0] + gt_templates[3]['Reasoning'][gt_templates[3]['Reasoning'].find('Agent-2'):] + ' ' + caption[caption.find('Decision'):]
                
        except FileNotFoundError:
            # print('0')
        #     # Do nothing and continue
            pass

        
        return instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts, act2
        

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
        if self.two_agent and 'Direction_cls_a2' in templates[0]:
            gt_act2 = templates[0]['Direction_cls_a2']
        else:
            gt_act2 = -1
        return gt_act, contrastive_acts, positive_acts_not_gt, gt_templates, contrastive_templates, positive_templates, gt_act2
    
    def augment_short_instruct(self, instruct):
        return instruct
        
    @property
    def modality_lengths(self) -> List[int]:
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list



class TrajAlignDataset_mtr(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, act=False, act_json=True, template_select=4, contrastive=False, train=False, random_select=False, positive_notgt=False, two_agent=False, new_eval=False, new_eval_mode='traj_pred', agents_instructed='', random_drop=True, return_meta_data=False, stage_2='', of_samples=False, nuplan_complex=False, debug=False, num_classes=8, weighted_sampling=False):
    # def __init__(self, **kwargs):
        super(TrajAlignDataset_mtr, self).__init__()
        self.new_eval_mode = new_eval_mode #'traj_pred', 'gt1', 'pos1', 'neg1',...
        self.num_classes=num_classes
        cfg_from_yaml_file(str(REPO_ROOT / "mtr" / "tools" / "cfgs" / "waymo" / "mtr+100_percent_data_act.yaml"), cfg_mtr)
        self.dataset_cfg = cfg_mtr.DATA_CONFIG
        self.training = train
        self.logger = None
        if os.environ.get("IMOTION_LLM_MTR_DATA_ROOT"):
            self.dataset_cfg.DATA_ROOT = os.environ["IMOTION_LLM_MTR_DATA_ROOT"]
        self.data_root = cfg.ROOT_DIR / self.dataset_cfg.DATA_ROOT
        self.data_path = self.data_root / self.dataset_cfg.SPLIT_DIR[self.mode]

        self.dataset_cfg.SAMPLE_INTERVAL['train'] = 1 if not debug else 5
        self.infos = self.get_all_infos(self.data_root / self.dataset_cfg.INFO_FILE[self.mode])
        if self.logger is not None:
            self.logger.info(f'Total scenes after filters: {len(self.infos)}')
        else:
            print(f'Total scenes after filters: {len(self.infos)}')
        
        if train:
            self.selected_data_indexed = self.filter_infos_based_on_gf_data(debug=debug)
        else:
            self.selected_data_indexed = self.test_filter_infos_based_on_gf_data(debug=debug)
        print(f'Total scenes after gameformer prompt filters: {len(self.selected_data_indexed)}')


        # return
        print('')
        self.nuplan_complex = nuplan_complex
        if nuplan_complex:
            list_of_instructs, list_of_np_dirs = get_nuplan_instructs_list(data_dir, 'gpt_data_101124')
            self.data_list = list_of_np_dirs
            self.instruct_list = list_of_instructs
            return
        self.return_meta_data = return_meta_data
        self.act = act
        self.act_json = act_json
        self.data_dir = data_dir
        # if act and act_json:
        #     self.data_list = glob.glob(data_dir[:-2]+'_json'+'/*')
        # else:
        #     # self.data_list = glob.glob(data_dir)
        #     data_list_ = glob.glob('/'.join(data_dir.split('/')[:-1])+'_templateLLM/*')
        #     self.data_list = [data_dir[:-1] + data_list_[i].split('/')[-1][:-4] + '.npz' for i in range(len(data_list_))]
        #     # [data_list_[i].split('/')[-1][:-4] + '.npz' for i in range(len(data_list_))]
        
            
        # logging.info(f"number of {'train' if train else 'eval'} samples found: {len(self.data_list)}")
            # data_list_


        self.stage_2 = False

        if template_select==33:
            self.template_select = 3
            self.random_drop=True
        else:
            self.random_drop=False
            self.template_select = template_select
        # if random_drop:
        self.random_drop=random_drop
        self.of_samples = of_samples

        self.contrastive = contrastive
        self.positive_notgt = positive_notgt
        self.train = train
        self.random_select = random_select

        direction_classifier = DirectionClassifier(num_classes=self.num_classes)
        self.direction_classes = np.array(direction_classifier.classes)

        if num_classes==8:
            train_contrastive_act_distribution = np.array([18.23, 2.7, 3.88, 3.91, 8.36, 16.04, 23.75, 23.12])
            train_positive_notgt_distribution = np.array([7.04, 6.76, 15.34, 9.14, 39.00, 13.54, 4.32, 4.85])
            train_groundtruth_act_distribution = [1.62, 55.77, 3.25, 3.71, 16.67, 17.49, 0.09, 1.40]
        elif num_classes==5:
            train_contrastive_act_distribution = np.array([20, 20, 20, 20, 20]) # for simplification assume equal distribution
            train_positive_notgt_distribution = np.array([20, 20, 20, 20, 20])
            train_groundtruth_act_distribution = [1.30, 66.90, 14.27, 17.17, 0.35]

        self.train_contrastive_prob = [0.7, 0.3]
        # self.train_contrastive_prob = [0.5, 0.5]
        # self.train_prob = [0.7, 0.3] if self.contrastive else [1.0,0.0]
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
            print(f">>> Eval mode: {self.new_eval_mode}")
        if weighted_sampling:
            self.initialize_sample_weights_with_known_distribution()
            # print('---')

    def initialize_sample_weights_with_known_distribution(self):
        """
        Initializes sample weights for weighted sampling using known class distribution.
        Uses log-scale weighting to handle extreme class imbalance.
        """
        print("Initializing sample weights with known class distribution (log scale)...")
        
        # Known class distribution for 5 classes
        act_counts = np.array([16748, 863910, 184286, 221762, 4575], dtype=np.float32)
        total_samples = np.sum(act_counts)
        
        # Calculate class probabilities
        class_probs = act_counts / total_samples
        
        # Calculate log-scale weights
        # Using log(1/p) = -log(p) as the weight
        if True:
            # Adding 1 inside log to ensure positive values
            log_weights = -np.log(class_probs + 1e-10)
        if False:
            # For debugging:
            log_weights = 1.0 / (class_probs ** 5)  # Using power of 5 makes weights even more extreme
        
        # Alternative: Use log(N/n_i) where N is total samples and n_i is class count
        # log_weights = np.log(total_samples / act_counts)
        
        # Normalize weights to sum to 1
        log_weights = log_weights / np.sum(log_weights)
        
        # Print class distribution and statistics
        print("\nClass Distribution:")
        for i in range(len(act_counts)):
            print(f"Class {i}: {act_counts[i]:.0f} samples ({class_probs[i]*100:.2f}%), log weight: {log_weights[i]:.6f}")
        
        # Default weight (use median of log weights)
        default_weight = np.median(log_weights)
        print(f"Default weight for unidentified classes: {default_weight:.6f}")
        
        print("\nAssigning sample weights...")
        
        # Pre-allocate numpy array with default weights
        
        self.sample_weights = np.full(len(self.selected_data_indexed), default_weight, dtype=np.float32)
        # self.sample_weights = np.full(len(self.data_list), default_weight, dtype=np.float32)
        
        # Process samples in batches for efficiency
        batch_size = 1000
        # num_batches = (len(self.data_list) + batch_size - 1) // batch_size
        num_batches = (len(self.selected_data_indexed) + batch_size - 1) // batch_size
        
        # Cache for template results
        template_cache = {}
        valid_assignments = 0
        
        for batch_idx in tqdm(range(num_batches), desc="Computing sample weights"):
            start_idx = batch_idx * batch_size
            # end_idx = min(start_idx + batch_size, len(self.data_list))
            end_idx = min(start_idx + batch_size, len(self.selected_data_indexed))
            
            for idx in range(start_idx, end_idx):
                try:
                    instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts = self.get_instruct_caption(self.selected_data_indexed[idx]['prompt_dir'], force_caption=False, acts=None)
                except:
                    gt_act = -1
                # np_filename = self.data_list[idx]
                # filename = np_filename.split('/')[-1].replace('.npz', '')
                # self.selected_data_indexed[0]['prompt_dir']
                
                # Use cache for efficiency
                # if filename in template_cache:
                #     act_class = template_cache[filename]
                # else:
                #     act_class = self._extract_act_for_sample(idx)
                #     template_cache[filename] = act_class
                
                act_class = gt_act
                filename= self.selected_data_indexed[idx]['mtr_id']
                template_cache[filename] = act_class
                    
                if 0 <= act_class < len(act_counts):
                    self.sample_weights[idx] = log_weights[act_class]
                    valid_assignments += 1
            
            # Clear cache periodically to manage memory
            if len(template_cache) > 100000:
                template_cache.clear()
        
        # Final normalization
        weight_sum = np.sum(self.sample_weights)
        if weight_sum > 0:
            self.sample_weights = self.sample_weights / weight_sum
        
        print(f"Weighted sampling initialized with {len(self.sample_weights)} weights")
        print(f"Valid class assignments: {valid_assignments} ({valid_assignments/len(self.selected_data_indexed)*100:.2f}%)")
        print(f"Min weight: {np.min(self.sample_weights)}, Max weight: {np.max(self.sample_weights)}")
        print(f"Mean weight: {np.mean(self.sample_weights)}, Std dev: {np.std(self.sample_weights)}")
        print(f"Weight distribution: min={np.min(log_weights):.4f}, max={np.max(log_weights):.4f}, ratio={np.max(log_weights)/np.min(log_weights):.2f}x")

    def test_filter_infos_based_on_gf_data(self, debug=False):
        mapping_root = _env_or_existing(
            "IMOTION_LLM_MTR_EVAL_MAPPING_DIR",
            _dataset_dir(self.data_dir).parent / "gf_mtr_mapping_test" / "gf_templatellm_maps",
        )
        mapping_dir = str(Path(mapping_root) / f"{self.new_eval_mode}.json")
        with open(mapping_dir, "r") as f:
            test_mapping = json.load(f)

        json_dir = mapping_root
        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        # Iterate through all JSON files in the directory
        merged_data = {}
        for json_file in json_files:
            with open(json_file, 'r') as file:
                data = json.load(file)  # Load JSON content
                merged_data.update(data)  # Merge into the main dictionary
        merged_data_keys = set(merged_data.keys())

        selected_data_indexed = {}
        global_index = 0
        for index in tqdm(range(len(self.infos))):
            info = self.infos[index]
            scene_id = info['scenario_id']
            with open(self.data_path / f'sample_{scene_id}.pkl', 'rb') as f:
                info = pickle.load(f)
            current_time_index = info['current_time_index']
            track_infos = info['track_infos']
            track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
            obj_types = np.array(track_infos['object_type'])
            obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
            center_objects, track_index_to_predict = self.get_interested_agents(
                track_index_to_predict=track_index_to_predict,
                obj_trajs_full=obj_trajs_full,
                current_time_index=current_time_index,
                obj_types=obj_types, scene_id=scene_id
            )
            center_objects_id = np.array(track_infos['object_id'])[track_index_to_predict]
            
            # Generate mtr_ids for lookup
            mtr_ids = [f"{scene_id}_{id_}" for id_ in list(center_objects_id)]
            # Filter based on existing `merged_data`
            for obj_index, mtr_id in enumerate(mtr_ids):
                if mtr_id in test_mapping.values():  # **Fast O(1) set lookup**
                    # Find all keys corresponding to the current mtr_id
                    matching_keys = [key for key, value in test_mapping.items() if value == mtr_id]

                    for key_index, key in enumerate(matching_keys):  # Include all matching keys separately
                        selected_data_indexed[global_index] = {
                            "infos_index": index,
                            "obj_index": obj_index,
                            "mtr_id": mtr_id,
                            # "prompt_dir": key,  # Now correctly set to each matching key
                            "filename": key,  # Now correctly set to each matching key
                            # "prompt_dir": merged_data[mtr_id],
                            "prompt_dir": str(_sibling_dir(self.data_dir, "_templateLLM") / f"{key.split('_act')[0]}.txt"),
                        }
                        global_index += 1  # Ensure unique indexing
                if len(selected_data_indexed) >= len(test_mapping) or debug and len(selected_data_indexed)==100:
                    break
            if len(selected_data_indexed) >= len(test_mapping) or debug and len(selected_data_indexed)==100:
                    break
        
        return selected_data_indexed

    def filter_infos_based_on_gf_data(self, debug):
        # print('loading prompts gameformer mapping jsons')
        # Define the directory containing JSON files
        ## generated using /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/generate_name_match_with_mtr_data.py
        # json_dir = "/ibex/project/c2278/felembaa/datasets/waymo/gameformer/gf_mtr_mapping/gf_templatellm_maps/"
        json_dir = _env_or_existing(
            "IMOTION_LLM_MTR_TRAIN_MAPPING_DIR",
            _dataset_dir(self.data_dir).parent / "gf_mtr_mapping" / "gf_templatellm_maps",
        )
        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        # Iterate through all JSON files in the directory
        merged_data = {}
        for json_file in json_files:
            with open(json_file, 'r') as file:
                data = json.load(file)  # Load JSON content
                merged_data.update(data)  # Merge into the main dictionary
        # print('filtering data based on existing gameformer data samples')
        # Convert keys of merged_data to a set for fast lookup
        merged_data_keys = set(merged_data.keys())
        # Dictionary indexed by global_index
        selected_data_indexed = {}
        global_index = 0
        for index in tqdm(range(len(self.infos))):
            info = self.infos[index]
            scene_id = info['scenario_id']
            # if scene_id == '7e74537718057728':
            #     print("Found")
            #     break
            with open(self.data_path / f'sample_{scene_id}.pkl', 'rb') as f:
                info = pickle.load(f)
            current_time_index = info['current_time_index']
            track_infos = info['track_infos']
            track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
            obj_types = np.array(track_infos['object_type'])
            obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
            center_objects, track_index_to_predict = self.get_interested_agents(
                track_index_to_predict=track_index_to_predict,
                obj_trajs_full=obj_trajs_full,
                current_time_index=current_time_index,
                obj_types=obj_types, scene_id=scene_id
            )
            center_objects_id = np.array(track_infos['object_id'])[track_index_to_predict]
            
            # Generate mtr_ids for lookup
            mtr_ids = [f"{scene_id}_{id_}" for id_ in list(center_objects_id)]
            # Filter based on existing `merged_data`
            for obj_index, mtr_id in enumerate(mtr_ids):
                if mtr_id in merged_data_keys:  # **Fast O(1) set lookup**
                    selected_data_indexed[global_index] = {
                        "infos_index": index,
                        "obj_index": obj_index,
                        "mtr_id": mtr_id,
                        "prompt_dir": merged_data[mtr_id],
                    }
                    global_index += 1  # Ensure unique indexing
            if debug and global_index>1000:
                break

        # Convert list to dictionary at the end (Reduces overhead of frequent dict updates)
        # selected_data_indexed = {i: data for i, data in enumerate(selected_data_list)}
        return selected_data_indexed

    def nuplan_complex_init(self, data_dir, act, act_json, template_select, contrastive, train, random_select, positive_notgt, two_agent, new_eval, new_eval_mode, agents_instructed, random_drop, return_meta_data, stage_2, of_samples, nuplan_complex):
        self.return_meta_data = return_meta_data
        self.act = act
        self.act_json = act_json
        self.data_dir = data_dir
        self.data_list = []
        data_list_1 = glob.glob(data_dir)
        for dir1 in data_list_1:
            dir1_list_json = glob.glob(dir1+'/gpt_data_101124/*')
            dir1_list_npz = glob.glob(dir1+'/npz/*')
            for json_file in dir1_list_json:
                npz_file = json_file.replace(json_file.split('/')[-2], 'npz').replace('.json','.npz')
                if npz_file in dir1_list_npz:
                    self.data_list.append(npz_file)
                else:
                    raise f"NOT FOUND: {npz_file}"
        print(f'Found {len(self.data_list)} complex instruction scenarios (could contain multiple instruction-caption pairs)')
        self.gpt_dir_name = 'gpt_data_101124'


    def nuplan_complex_getitem(self, idx):
        np_filename = self.data_list[idx]
        # json_filename = np_filename.replace('.npz', '.json').replace('npz', self.gpt_dir_name)
        
        data = np.load(np_filename, allow_pickle = True)
        traj_obs = torch.tensor(data['ego'][:,:,:2])
        traj_pred = torch.tensor(data['gt_future_states'][:,:,:2])
        traj = torch.cat((traj_obs, traj_pred), dim=1)
        object_type = (data['ego'][:,-1,8:].argmax(-1)+1) * data['ego'][:,-1,8:].sum(-1).astype(int)
        data_object_type = data['object_type']
        map_lanes = data['map_lanes'][:, :6, :][...,:200:2,:]
        map_crosswalks = data['map_crosswalks'][:, :, :100:2]
        ego = data['ego']
        gt_future_states = data['gt_future_states']
        neighbors = data['neighbors']

        # if True: # single agent only data
        #     traj_obs = traj_obs[:1]
        #     traj_pred = traj_pred[:1]
        #     traj = traj[:1]
        #     object_type = object_type[:1]
        #     data_object_type = data_object_type[:1]
        #     map_lanes = map_lanes[:1]
        #     map_crosswalks = map_crosswalks[:1]
        #     ego = ego[:1]
        #     gt_future_states = gt_future_states[:1]
        #     neighbors = np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0)
            

        out_dict = {
            'traj': traj,
            'ego_state': data['ego'][0],
            'neighbors_state': np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0),
            # 'map_lanes': data['map_lanes'][:, :, :200:2],
            'map_lanes': map_lanes,
            'map_crosswalks': map_crosswalks,
            'object_type':object_type,
            'ego': ego,
            'ground_truth': gt_future_states,
            'neighbors': neighbors,
            'data_object_type' : data_object_type,
        }

        instruct = self.instruct_list[idx]['instruction']
        caption = self.instruct_list[idx]['reasoning']
        safe = self.instruct_list[idx]['safe']
        category = self.instruct_list[idx]['category']
        instruct = f"<s>[INST] Generate trajectories embeddings based on the following given instruction and scene embeddings. {instruct} "
        accepted_or_rejected = "<Accepted>" if safe else "<Rejected>"
        caption = f"{caption} Decision: {accepted_or_rejected}. Generated trajectories embeddings: "
        
        # '<s>[INST] ' ' '
        # ' Decision: <Rejected><Accepted>. Generated trajectories embeddings: '

        out_dict.update(
            {
                'instruct': instruct,
                'caption': caption,
                'contrastive_sample' : not safe,
            }
        )
        # out_dict.update({
        #     'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
        #     'contrastive_sample': contrastive_sample or of_sample, # used to indicate ignoring the generated trajectory
        #     'act': torch.tensor(act),
        #     'instruct': instruct,
        #     'file_name': filename, #*
        #     'caption' : caption, #*
        # })
        return out_dict

    def __len__(self):
        return len(self.selected_data_indexed)
        
    def __getitem__(self, index):
        metadata = self.selected_data_indexed[index]
        if self.new_eval:
            acts = [self.meta_eval[metadata['filename']][self.new_eval_mode]]
        else:
            acts = None
        ret_infos = self.create_scene_level_data(metadata['infos_index'], metadata['obj_index'])
        template_dir = metadata['prompt_dir']
        instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts = self.get_instruct_caption(template_dir, force_caption=self.two_agent, acts=acts)

        out_dict = {
                    'caption' : caption,
                    'instruct': instruct,
                    'act': torch.tensor(act),
                    'contrastive_sample': contrastive_sample or of_sample, # used to indicate ignoring the generated trajectory
                    'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
                }
        out_dict.update(metadata)
        out_dict.update(ret_infos) # trajectory data for MTR
        # if not self.train:
        #     out_dict['scenario_id'] = str(out_dict['scenario_id'][0])
        return out_dict

    def old__len__(self):
        return len(self.data_list)
   
    def old__getitem__(self, idx):
        if self.nuplan_complex:
            return self.nuplan_complex_getitem(idx)

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
            instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts = self.get_instruct_caption(template_dir, force_caption=self.two_agent, acts=acts, agent='1')
            if self.two_agent:
                if np_filename in self.filenames_T.keys():
                    filename_2 = self.filenames_T[np_filename].split('/')[-1].replace('.npz','')
                    template_dir_2 = root_dir+'_templateLLM/'+filename_2+'.txt'
                    instruct_2, caption_2, act_2, contrastive_sample_2, of_sample_2, gt_act2, positive_acts2, contrastive_acts2 = self.get_instruct_caption(template_dir_2, acts=acts, agent='2')
                    instruct_updated, caption_updated = self.include_agent_2(instruct, caption, contrastive_sample, instruct_2, caption_2, contrastive_sample_2)
        else:
            root_dir = '/'.join(self.data_dir.split('/')[:-1])
            filename = self.data_list[idx].split('/')[-1].replace('.npz','')
            # agent_dir = root_dir+'_agentJsons/'+filename+'.json'
            # map_dir = root_dir+'_mapJsons/'+filename+'.json'
            if use_pos_augmented_sample:
                # Done: Change the template LLM to the other feasible template LLM, ensure that the sample loaded will not be "infeasible"
                template_dir = self.stage_2_dir+'_templateLLM/'+filename+'.txt'
                instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts = self.get_instruct_caption(template_dir, force_caption=self.two_agent, positive_example_only=True)
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
                instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts = self.get_instruct_caption(template_dir, force_caption=self.two_agent)

            
            if self.two_agent:
                if np_filename in self.filenames_T.keys():
                    filename_2 = self.filenames_T[np_filename].split('/')[-1].replace('.npz','')
                    template_dir_2 = root_dir+'_templateLLM/'+filename_2+'.txt'
                    instruct_2, caption_2, act_2, contrastive_sample_2, of_sample_2, gt_act2, positive_acts2, contrastive_acts2 = self.get_instruct_caption(template_dir_2)
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
                # 'map_lanes': data['map_lanes'][:, :, :200:2],
                'map_lanes': data['map_lanes'][:, :6, :],
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
                'contrastive_sample': contrastive_sample or of_sample, # used to indicate ignoring the generated trajectory
                'gt_act':torch.tensor(gt_act), #'positive_acts':torch.tensor(positive_acts), 'contrastive_acts':torch.tensor(contrastive_acts),
            }
            # return out_dict
        else:
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
                    'act': torch.tensor(act),
                    'contrastive_sample': contrastive_sample or of_sample, # used to indicate ignoring the generated trajectory
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
        # return out_dict['instruct']
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

    # Function to load and parse the JSON data with filtering
    # def load_complex_data(self, json_dir):
    #     output_dir = {"safe":[], "safe_no_context":[], "unsafe":[], "unsafe_no_context":[]}
    #     with open(json_dir, 'r') as file:
    #         data = json.load(file)
        
    #     # safe
    #     for entry in data[0]['data']:
    #         entry['instructions'][0]
    #         if 'without' in entry['instructions'][0]['category'].lower():
            
    #         else:


    #         break
    #     # List to store filtered data samples
    #     filtered_samples = []
        
    #     # Iterate through each entry in the JSON data
    #     for entry in data:
    #         break
    #         for sample in entry["data"]:
    #             for instruction_set in sample["instructions"]:
    #                 # Check if 'safe' matches 'safe_instruction'
    #                 if instruction_set["safe"] == instruction_set["safe_instruction"]:
    #                     # Extract relevant data into a dictionary if they match
    #                     sample_dict = {
    #                         "instruction": instruction_set["instruction"],
    #                         "reasoning": instruction_set["reasoning"],
    #                         "safe": instruction_set["safe"],
    #                         "safe_instruction": instruction_set["safe_instruction"],
    #                         "category": instruction_set["category"]
    #                     }
    #                     filtered_samples.append(sample_dict)
        
    #     return filtered_samples

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
        of_sample=False


        try:
        # if True:
            with open(template_dir) as f:
                lines = f.readlines()
            # print('1')
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
                if self.of_samples:
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
            of_sample = False
            if self.train and self.contrastive and self.of_samples and not positive_example_only:
                scenario_type_select = np.random.choice(3, p=[1/3, 1/3, 1/3])
                if valid_positive and scenario_type_select == 1:
                    selected_template = positive_templates[positive_class_arg]
                    of_sample = True
                elif valid_contrastive and scenario_type_select == 2:
                    selected_template = contrastive_templates[contrastive_class_arg]
                    contrastive_sample=True
                else:
                    selected_template=None
                if selected_template is not None:
                    instruct = selected_template['Instruction']
                    instruct = "Generate trajectories embeddings based on the following given instruction and scene embeddings. "+instruct.split(": ")[1]
                    instruct = f'<s>[INST] {instruct} '
                    reason = selected_template['Reasoning']
                    decision = selected_template['Decision']
                    caption = f"{reason} Decision: {decision}. Generated trajectories embeddings: "
                    act = selected_template['Direction_cls']
            elif self.train and self.contrastive:
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
            # print('0')
        #     # Do nothing and continue
            pass

        
        return instruct, caption, act, contrastive_sample, of_sample, gt_act, positive_acts, contrastive_acts
        

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
        
    @property
    def modality_lengths(self) -> List[int]:
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    def collate_batch(self, batch_list):
        """
        Args:
        batch_list:
            scenario_id: (num_center_objects)
            track_index_to_predict (num_center_objects):

            obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            obj_trajs_mask (num_center_objects, num_objects, num_timestamps):
            map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_polylines, num_points_each_polyline)

            obj_trajs_pos: (num_center_objects, num_objects, num_timestamps, 3)
            obj_trajs_last_pos: (num_center_objects, num_objects, 3)
            obj_types: (num_objects)
            obj_ids: (num_objects)

            center_objects_world: (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            center_objects_type: (num_center_objects)
            center_objects_id: (num_center_objects)

            obj_trajs_future_state (num_center_objects, num_objects, num_future_timestamps, 4): [x, y, vx, vy]
            obj_trajs_future_mask (num_center_objects, num_objects, num_future_timestamps):
            center_gt_trajs (num_center_objects, num_future_timestamps, 4): [x, y, vx, vy]
            center_gt_trajs_mask (num_center_objects, num_future_timestamps):
            center_gt_final_valid_idx (num_center_objects): the final valid timestamp in num_future_timestamps
        """
        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():

            if key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask', 'map_polylines_center',
                'obj_trajs_pos', 'obj_trajs_last_pos', 'obj_trajs_future_state', 'obj_trajs_future_mask']:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list)
            elif key in ['scenario_id', 'obj_types', 'obj_ids', 'center_objects_type', 'center_objects_id']:
                input_dict[key] = np.concatenate(val_list, axis=0)
            else:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = torch.cat(val_list, dim=0)

        batch_sample_count = [len(x['track_index_to_predict']) for x in batch_list]
        batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_sample_count}
        return batch_dict
    
    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def get_all_infos(self, info_path):
        # self.logger.info(f'Start to load infos from {info_path}')
        print(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::self.dataset_cfg.SAMPLE_INTERVAL[self.mode]]
        # self.logger.info(f'Total scenes before filters: {len(infos)}')
        print(f'Total scenes before filters: {len(infos)}')

        for func_name, val in self.dataset_cfg.INFO_FILTER_DICT.items():
            infos = getattr(self, func_name)(infos, val)

        return infos

    def filter_info_by_object_type(self, infos, valid_object_types=None):
        ret_infos = []
        for cur_info in infos:
            num_interested_agents = cur_info['tracks_to_predict']['track_index'].__len__()
            if num_interested_agents == 0:
                continue

            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)

            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue

            assert len(cur_info['tracks_to_predict'].keys()) == 3, f"{cur_info['tracks_to_predict'].keys()}"
            cur_info['tracks_to_predict']['track_index'] = list(np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])

            ret_infos.append(cur_info)
        # self.logger.info(f'Total scenes after filter_info_by_object_type: {len(ret_infos)}')
        print(f'Total scenes after filter_info_by_object_type: {len(ret_infos)}')
        return ret_infos

    def create_scene_level_data(self, index, obj_index):
        """
        Args:
            index (index):

        Returns:

        """
        info = self.infos[index]
        scene_id = info['scenario_id']
        with open(self.data_path / f'sample_{scene_id}.pkl', 'rb') as f:
            info = pickle.load(f)

        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

        track_infos = info['track_infos']

        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_ids = np.array(track_infos['object_id'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
            center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids) = self.create_agent_data_for_center_objects(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types, obj_ids=obj_ids
        )

        ret_dict = {
            # 'scenario_id': scene_id,
            'scenario_id': np.array([scene_id]),
            # 'scenario_id': np.array([scene_id]),
            'obj_trajs': obj_trajs_data[obj_index:obj_index+1], # center normalized
            'obj_trajs_mask': obj_trajs_mask[obj_index:obj_index+1],
            'track_index_to_predict': track_index_to_predict_new[obj_index:obj_index+1],  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos[obj_index:obj_index+1],
            'obj_trajs_last_pos': obj_trajs_last_pos[obj_index:obj_index+1],
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects[obj_index:obj_index+1],
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict][obj_index:obj_index+1],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict][obj_index:obj_index+1],

            'obj_trajs_future_state': obj_trajs_future_state[obj_index:obj_index+1],
            'obj_trajs_future_mask': obj_trajs_future_mask[obj_index:obj_index+1],
            'center_gt_trajs': center_gt_trajs[obj_index:obj_index+1],
            'center_gt_trajs_mask': center_gt_trajs_mask[obj_index:obj_index+1],
            'center_gt_final_valid_idx': center_gt_final_valid_idx[obj_index:obj_index+1],
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict][obj_index:obj_index+1]
        }

        if not self.dataset_cfg.get('WITHOUT_HDMAP', False):
            if info['map_infos']['all_polylines'].__len__() == 0:
                info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
                print(f'Warning: empty HDMap {scene_id}')

            map_polylines_data, map_polylines_mask, map_polylines_center = self.create_map_data_for_center_objects(
                center_objects=center_objects, map_infos=info['map_infos'],
                center_offset=self.dataset_cfg.get('CENTER_OFFSET_OF_MAP', (30.0, 0)),
            )   # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)

            ret_dict['map_polylines'] = map_polylines_data[obj_index:obj_index+1]
            ret_dict['map_polylines_mask'] = (map_polylines_mask[obj_index:obj_index+1] > 0)
            ret_dict['map_polylines_center'] = map_polylines_center[obj_index:obj_index+1]
        
        # ret_dict['scenario_id'][0]
        # mtr_ids = [f"{ret_dict['scenario_id'][0]}_{ret_dict['center_objects_id'][0]}" for i in range(len(ret_dict['center_objects_id']))]
        # mtr_id = f"{using batch['input_dict']['scenario_id'][0]}_{batch['input_dict']['center_objects_id'][0]}"
        ## Used to match with data samples preprocessed by GameFomrer preprocessing code
        mtr_ids = np.array([f"{ret_dict['scenario_id'][i]}_{ret_dict['center_objects_id'][i]}" for i in range(len(ret_dict['center_objects_id']))])
        return ret_dict

    def create_agent_data_for_center_objects(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types, obj_ids
        ):
        obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask = self.generate_centered_trajs_for_agents(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past,
            obj_types=obj_types, center_indices=track_index_to_predict,
            sdc_index=sdc_track_index, timestamps=timestamps, obj_trajs_future=obj_trajs_future
        )

        # generate the labels of track_objects for training
        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps, 4)
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps)
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        # filter invalid past trajs
        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 4):  [x, y, vx, vy]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]  # (num_center_objects, num_objects, num_timestamps_future):
        obj_types = obj_types[valid_past_mask]
        obj_ids = obj_ids[valid_past_mask]

        valid_index_cnt = valid_past_mask.cumsum(axis=0)
        track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
        sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

        assert obj_trajs_future_state.shape[1] == obj_trajs_data.shape[1]
        assert len(obj_types) == obj_trajs_future_mask.shape[1]
        assert len(obj_ids) == obj_trajs_future_mask.shape[1]

        # generate the final valid position of each object
        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
            center_gt_final_valid_idx[cur_valid_mask] = k

        return (obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
            obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids)

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        track_index_to_predict_selected = []

        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            assert obj_trajs_full[obj_idx, current_time_index, -1] > 0, f'obj_idx={obj_idx}, scene_id={scene_id}'

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)

        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    @staticmethod
    def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
                angle=-center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def generate_centered_trajs_for_agents(self, center_objects, obj_trajs_past, obj_types, center_indices, sdc_index, timestamps, obj_trajs_future):
        """[summary]

        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_trajs_past (num_objects, num_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_types (num_objects):
            center_indices (num_center_objects): the index of center objects in obj_trajs_past
            centered_valid_time_indices (num_center_objects), the last valid time index of center objects
            timestamps ([type]): [description]
            obj_trajs_future (num_objects, num_future_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        Returns:
            ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
            ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
            ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
        """
        assert obj_trajs_past.shape[-1] == 10
        assert center_objects.shape[-1] == 10
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        # transform to cpu torch tensor
        center_objects = torch.from_numpy(center_objects).float()
        obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
        timestamps = torch.from_numpy(timestamps)

        # transform coordinates to the centered objects
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        ## generate the attributes for each object
        object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
        object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # TODO: CHECK THIS TYPO
        object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
        object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
        object_onehot_mask[:, sdc_index, :, 4] = 1

        object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
        object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

        object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
        vel_pre = torch.roll(vel, shifts=1, dims=2)
        acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        acce[:, :, 0, :] = acce[:, :, 1, :]

        ret_obj_trajs = torch.cat((
            obj_trajs[:, :, :, 0:6], 
            object_onehot_mask,
            object_time_embedding, 
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9], 
            acce,
        ), dim=-1)

        ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs[ret_obj_valid_mask == 0] = 0

        ##  generate label for future trajectories
        obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0
        
        
        
        # fig, ax = plt.subplots()
        # plt.plot(ret_obj_trajs[1,3,:,0], ret_obj_trajs[1,3,:,1], linestyle='--', color='r', alpha=0.5) 
        # plt.plot(ret_obj_trajs_future[1,3,:,0], ret_obj_trajs_future[1,3,:,1], linestyle='--', color='b', alpha=0.5) 
        # plt.savefig('ex.png')
        return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()

    @staticmethod
    def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
        """
        Args:
            polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

        Returns:
            ret_polylines: (num_polylines, num_points_each_polyline, 7)
            ret_polylines_mask: (num_polylines, num_points_each_polyline)
        """
        point_dim = polylines.shape[-1]

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        ret_polylines = torch.from_numpy(ret_polylines)
        ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

        # # CHECK the results
        # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
        # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
        # assert center_dist.max() < 10
        return ret_polylines, ret_polylines_mask

    def create_map_data_for_center_objects(self, center_objects, map_infos, center_offset):
        """
        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict):
                all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_offset (2):, [offset_x, offset_y]
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
        """
        num_center_objects = center_objects.shape[0]

        # transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).view(num_center_objects, -1, batch_polylines.shape[1], 2)

            # use pre points to map
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
            xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask

        polylines = torch.from_numpy(map_infos['all_polylines'].copy())
        center_objects = torch.from_numpy(center_objects)

        batch_polylines, batch_polylines_mask = self.generate_batch_polylines_from_map(
            polylines=polylines.numpy(), point_sampled_interval=self.dataset_cfg.get('POINT_SAMPLED_INTERVAL', 1),
            vector_break_dist_thresh=self.dataset_cfg.get('VECTOR_BREAK_DIST_THRESH', 1.0),
            num_points_each_polyline=self.dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20),
        )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)

        # collect a number of closest polylines for each center objects
        num_of_src_polylines = self.dataset_cfg.NUM_OF_SRC_POLYLINES

        if len(batch_polylines) > num_of_src_polylines:
            polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
            center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot.view(num_center_objects, 1, 2),
                angle=center_objects[:, 6]
            ).view(num_center_objects, 2)

            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

            dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
            topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
            map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
        else:
            map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)

        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  # (num_center_objects, num_polylines, 3)

        map_polylines = map_polylines.numpy()
        map_polylines_mask = map_polylines_mask.numpy()
        map_polylines_center = map_polylines_center.numpy()

        return map_polylines, map_polylines_mask, map_polylines_center

    def generate_prediction_dicts(self, batch_dict, output_path=None):
        """

        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 10)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_dict['input_dict']

        pred_scores = batch_dict['pred_scores']
        pred_trajs = batch_dict['pred_trajs']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 7

        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=center_objects_world[:, 6].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]
        
        
        # fig, ax = plt.subplots()
        # plt.plot(input_dict['center_gt_trajs_src'].cpu()[0,:,0], input_dict['center_gt_trajs_src'].cpu()[0,:,1], linestyle='--', color='r', alpha=0.5) 
        # # plt.plot(batch_dict['pred_trajs'].cpu()[0,0,:,0], batch_dict['pred_trajs'].cpu()[0,0,:,1], linestyle='--', color='r', alpha=0.5) 
        # # plt.plot(pred_trajs_world[0,0,:,0].cpu(), pred_trajs_world[0,0,:,1].cpu(), linestyle='--', color='b', alpha=0.5) 
        # plt.savefig('ex.png')

        pred_dict_list = []
        batch_sample_count = batch_dict['batch_sample_count']
        start_obj_idx = 0
        for bs_idx in range(batch_dict['batch_size']):
            cur_scene_pred_list = []
            for obj_idx in range(start_obj_idx, start_obj_idx + batch_sample_count[bs_idx]):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][obj_idx],
                    'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].to(torch.float32).cpu().numpy(),
                    'pred_scores': pred_scores[obj_idx, :].to(torch.float32).cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][obj_idx],
                    'object_type': input_dict['center_objects_type'][obj_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][obj_idx].to(torch.float32).cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].to(torch.float32).cpu().numpy()
                }
                cur_scene_pred_list.append(single_pred_dict)

            pred_dict_list.append(cur_scene_pred_list)
            start_obj_idx += batch_sample_count[bs_idx]

        assert start_obj_idx == num_center_objects
        assert len(pred_dict_list) == batch_dict['batch_size']

        return pred_dict_list

    def evaluation(self, pred_dicts, output_path=None, eval_method='waymo', **kwargs):
        if eval_method == 'waymo':
            from mtr.mtr.datasets.waymo.waymo_eval import waymo_evaluation
            try:
                num_modes_for_eval = pred_dicts[0][0]['pred_trajs'].shape[0]
            except:
                num_modes_for_eval = 6
            metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=num_modes_for_eval)

            metric_result_str = '\n'
            for key in metric_results:
                metric_results[key] = metric_results[key]
                metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
            metric_result_str += '\n'
            metric_result_str += result_format_str
        else:
            raise NotImplementedError

        return metric_result_str, metric_results
