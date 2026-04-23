import shutil
import os
import json
from tqdm import tqdm
import glob

# target_file_idx = 3
for target_file_idx in [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]:
    json_path = "/home/felembaa/projects/trajgpt/scenario_id_map.json"
    with open(json_path, 'r') as file:
        scenario_ids_list = json.load(file)

    scenario_ids_select = scenario_ids_list[list(scenario_ids_list.keys())[target_file_idx]]
    save_dir = list(scenario_ids_list.keys())[target_file_idx].replace('.','_').replace('-','_')
    save_dir = '/ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_p_29feb_splits/'+save_dir+'_json/'
    os.makedirs(save_dir, exist_ok=True)
    source_files = glob.glob('/ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_p_29feb_json/*')
    for file_i in tqdm(source_files):
        scenario_id = file_i.split('/')[-1].split('_')[0]
        if scenario_id in scenario_ids_select:
            shutil.copy(file_i, save_dir)

    save_dir = list(scenario_ids_list.keys())[target_file_idx].replace('.','_').replace('-','_')
    save_dir = '/ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_p_29feb_splits/'+save_dir+'/'
    os.makedirs(save_dir, exist_ok=True)
    source_files = glob.glob('/ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_p_29feb/*')
    for file_i in tqdm(source_files):
        scenario_id = file_i.split('/')[-1].split('_')[0]
        if scenario_id in scenario_ids_select:
            shutil.copy(file_i, save_dir)