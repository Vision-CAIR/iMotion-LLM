from minigpt4.datasets.datasets.traj_dataset import TrajAlignDataset as TrajDataset
from minigpt4.models import mini_gpt4
from torch.utils.data import DataLoader
from minigpt4.common.registry import registry
from minigpt4.common.config import Config
# from minigpt4.tasks.base_task import BaseTask as tasks
import minigpt4.tasks as tasks
import torch
from torch.cuda.amp import autocast as autocast
# from vizualize_example import *
# import matplotlib.pyplot as plt
# import time
import json
import glob
from tqdm import tqdm
import os
import time

    
validation_subset_num = 6



class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")
    
    def __setattr__(self, key, value):
        self[key] = value


json_path = "<legacy_trajgpt_repo>/scenario_id_map.json"
with open(json_path, 'r') as file:
    scenario_ids_list = json.load(file)



args = AttrDict()
args.options = ['']
args.cfg_path = '<legacy_trajgpt_repo>/train_configs/mar1/exp01.yaml'
cfg = Config(args)

device = 'cuda'

task = tasks.setup_task(cfg)
model = task.build_model(cfg)
model = model.to(device)
model.eval()
checkpoint = torch.load(cfg.run_cfg.output_dir + "/checkpoint_last.pth", map_location=device)
state_dict = checkpoint["model"]
model.load_state_dict(state_dict, strict=False)

validation_splits = glob.glob('<internal_waymo_dataset_root>/validation_interactive_p_29feb_splits/*_json')
print(len(validation_splits))


filename = validation_splits[validation_subset_num].split('_json')[0]
dataset = TrajDataset(f'{filename}/*', act=cfg.model_cfg.act)
loader = DataLoader(
                    dataset,
                    batch_size=16,
                    num_workers=0,
                    drop_last=False,
                    # pin_memory=True,
                    shuffle=False,
                ) 

output_save_dir = cfg.run_cfg.output_dir+'eval/'+filename.split('/')[-1]+'/'
os.makedirs(output_save_dir, exist_ok=True)


l_count = 0
r_count = 0
s_count = 0
u_count = 0
else_count = 0

for batch_i, samples in enumerate(tqdm(loader)):
    start_time = time.time()
    for k in list(samples.keys()):
        if torch.is_tensor(samples[k]):
            samples[k] = samples[k].to(device)
    samples['instruct'] = samples['all_captions'][1]
    samples_ = samples.copy()
    samples_['instruct'] = list(samples_['instruct'])
    for instruct_i, instruct in enumerate(samples_['instruct']):
        if 'U-turn' in instruct:
            samples_['instruct'][instruct_i] = instruct.split('take')[0]+'move straight.'
        elif 'straight' in instruct:
            samples_['instruct'][instruct_i] = instruct.split('move')[0]+'take a right turn.'
        elif 'right' in instruct:
            samples_['instruct'][instruct_i] = instruct.split('take')[0]+'move straight.'
        elif 'left' in instruct:
            samples_['instruct'][instruct_i] = instruct.split('take')[0]+'move straight.'
        elif 'not' in instruct:
            samples_['instruct'][instruct_i] = instruct.split('not')[0]+'move straight.'
        else:
            print(instruct)
            else_count+=1
            # raise 'undefined'
    samples_['instruct'] = tuple(samples_['instruct'])
    samples__ = {}
    for k in samples.keys():
        if torch.is_tensor(samples[k]):
            samples__[k] = torch.cat((samples[k], samples_[k]), dim=0)
        else:
            samples__[k] = samples[k] + samples_[k]

    samples = samples__

    for k in list(samples.keys()):
        if torch.is_tensor(samples[k]):
            samples[k] = samples[k].to(device)
    
    output_dict = model(samples, False)

    output_dict.update(samples)
    for k in list(output_dict.keys()):
        if torch.is_tensor(output_dict[k]):
            output_dict[k] = output_dict[k].to('cpu')
    torch.save(output_dict, output_save_dir+f'batch_{batch_i}.pth')
    # print(output_dict['text'][0])
    # print(output_dict['text'][int(len(output_dict['text'])/2)])

