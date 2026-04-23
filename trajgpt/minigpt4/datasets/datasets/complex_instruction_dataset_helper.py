import os
import torch
import numpy as np
# from traj_utils import *
# from torch.utils.data import DataLoader, Dataset
# from generate_meta_caption import *
import random
# from extract_instruct_v2 import extract_simple_turn_5_classes
# from extract_instruct_v3 import ClassifyTrack, get_sample_instruct
# from extract_instruct_v3 import *
# from extract_instruct_v4 import *
# from instructions.direction_instructions import DirectionClassifier
import logging
import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NUPLAN_PROMPT_GLOB = os.environ.get(
    "IMOTION_LLM_NUPLAN_PROMPT_GLOB",
    str(REPO_ROOT / "data" / "processed" / "nuplan" / "gpt_prompt_14types" / "*"),
)

def get_nuplan_instructs_list(data_dir=None, gpt_dir_name='gpt_data_101124', nuplan_direction=False):
    data_dir = data_dir or DEFAULT_NUPLAN_PROMPT_GLOB
    data_list = []
    data_list_1 = glob.glob(data_dir)
    for dir1 in data_list_1:
        dir1_list_json = glob.glob(dir1+f'/{gpt_dir_name}/*')
        dir1_list_npz = glob.glob(dir1+'/npz/*')
        for json_file in dir1_list_json:
            npz_file = json_file.replace(json_file.split('/')[-2], 'npz').replace('.json','.npz')
            if npz_file in dir1_list_npz:
                data_list.append(npz_file)
            else:
                raise f"NOT FOUND: {npz_file}"
    print(f'Found {len(data_list)} complex instruction scenarios (could contain multiple instruction-caption pairs)')

    list_of_instructs = []
    list_of_np_dirs = []
    for np_filename in data_list:
        json_filename = np_filename.replace('.npz', '.json').replace('npz', gpt_dir_name)
        instructs_ = load_complex_data(json_filename)
        if not nuplan_direction:
            for instruct_ in instructs_:
                list_of_instructs.append(instruct_)
                list_of_np_dirs.append(np_filename)
        else:
            simple_instruct_dir = np_filename.replace('.npz', '.txt').replace('npz', 'gpt_meta_prompts')
            with open(simple_instruct_dir, "r") as f:
                data_ = json.load(f)  # directly converts JSON string to Python dict
                to_append_instruct = data_['trajectory_description'].split(';')[0].split(',')[0].split(' veering')[0].split(' with')[0]
                to_append= {
                    'instruction': f"Make the ego vehicle {to_append_instruct}.", 
                    'reasoning':f"The ego vehicle can safely {data_['trajectory_description']}.",
                    'safe': True,
                    'category': 'Without Context'
                    }
                list_of_instructs.append(to_append)
                list_of_np_dirs.append(np_filename)
    
    print(f"Found {len(list_of_instructs)} number of instruction-reasoning")
    return list_of_instructs, list_of_np_dirs
    
def load_complex_data(json_dir):
    # output_dir = {"safe":[], "safe_no_context":[], "unsafe":[], "unsafe_no_context":[]}
    # json_dir = json_filename
    with open(json_dir, 'r') as file:
        data = json.load(file)
    
    # Iterate through each entry in the JSON data
    filtered_samples = []
    for i, entry in enumerate(data):
        for sample in entry["data"]:
            for instruction_set in sample["instructions"]:
                # Check if 'safe' matches 'safe_instruction'
                if "safe" not in instruction_set or instruction_set['safe'] is None:
                    if "safe_instruction" in instruction_set and ((i==0 and instruction_set["safe_instruction"]) or (i==1 and not instruction_set["safe_instruction"])):
                        valid_instruct = True
                    else:
                        valid_instruct = False                  
                elif not isinstance(instruction_set['safe'], bool) or not isinstance(instruction_set['safe_instruction'], bool):
                        valid_instruct = False
                elif (i==0 and instruction_set["safe"] and instruction_set["safe_instruction"]) or (i==1 and not instruction_set["safe"] and not instruction_set["safe_instruction"]):
                    valid_instruct=True
                else:
                    valid_instruct=False
                # if "safe" in instruction_set and instruction_set["safe"] == instruction_set["safe_instruction"]:
                    # Extract relevant data into a dictionary if they match
                if valid_instruct:
                    sample_dict = {
                        "instruction": instruction_set["instruction"],
                        "reasoning": instruction_set["reasoning"],
                        "safe": True if i==0 else False,
                        "category": instruction_set["category"]
                    }
                    filtered_samples.append(sample_dict)
                    
    
    return filtered_samples


    

if __name__ == "__main__":
    data_dir = DEFAULT_NUPLAN_PROMPT_GLOB
    gpt_dir_name = 'gpt_data_101124'
    list_of_instructs, list_of_np_dirs = get_nuplan_instructs_list()
    print('')
    
    # data_list = []
    # data_list_1 = glob.glob(data_dir)
    # for dir1 in data_list_1:
    #     dir1_list_json = glob.glob(dir1+'/gpt_data_101124/*')
    #     dir1_list_npz = glob.glob(dir1+'/npz/*')
    #     for json_file in dir1_list_json:
    #         npz_file = json_file.replace(json_file.split('/')[-2], 'npz').replace('.json','.npz')
    #         if npz_file in dir1_list_npz:
    #             data_list.append(npz_file)
    #         else:
    #             raise f"NOT FOUND: {npz_file}"
    # print(f'Found {len(data_list)} complex instruction scenarios (could contain multiple instruction-caption pairs)')
    
    # list_of_instructs = []
    # list_of_np_dirs = []
    # for np_filename in data_list:
    #     json_filename = np_filename.replace('.npz', '.json').replace('npz', gpt_dir_name)
    #     instructs_ = load_complex_data(json_filename)
    #     for instruct_ in instructs_:
    #         list_of_instructs.append(instruct_)
    #         list_of_np_dirs.append(np_filename)
    
    # print(f"Found {len(list_of_instructs)} number of instruction-reasoning")
    # return list_of_instructs, list_of_np_dirs
