# Title: Validate Instructions
# Description: Code to validate each category of the defined instructions, 
# the code should sample equal number of examples per category so the evaluator 
# can manually inspect and record accuracy score

# Import necessary libraries and modules
from vizualize_instructions import viz_instruction_sample
import torch
from torch.utils.data import DataLoader
from direction_instructions import DirectionClassifier
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
# Add the project path to the system path for importing project-specific modules
sys.path.append('/home/felembaa/projects/iMotion-LLM-ICLR/gameformer')
from utils.inter_pred_utils import DrivingData
import json

def init_or_read_json(json_dir):
    exist=False
    # Writing JSON data to a file
    # if not os.path.exists(json_output_dir):
    if True:
        with open(json_output_dir, 'w') as file:
            json_data = {}
            json.dump(json_data, file)
    # else:
    #      # load JSON data
    #     with open(json_output_dir, 'r') as file:
    #         json_data = json.load(file)
    #     exist=True
    return json_data, exist

def get_direction_instructions(ego_future, direction_classifier):
    # Sample specific time steps for direction instructions (1Hz sampling from 0 to 70 and 79)
    sampling_idx = [ii for ii in range(80)][::10] + [79]  # 80 steps sampled at 10Hz + the last step
    sampling_idx_1st_half = sampling_idx[:5] # 0.1s to 4s in the future
    sampling_idx_2nd_half = sampling_idx[4:] # 4s to 8s in the future

    # Get the batch (batch size of 1 in this code) instruction for the current ego vehicle's future trajectory
    instruct, contrastive_instructs = direction_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx)
    instruct_1st_half, _ = direction_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_1st_half)
    instruct_2nd_half, _ = direction_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_2nd_half)

    # Determine the instruction string
    if instruct[0][0] != -1:
        instruct_str = direction_classifier.classes[int(instruct[0][0])]
    else:
        instruct_str = 'UNKNOWN'
    if instruct_1st_half[0][0] != -1:
        instruct_str_1st_half = direction_classifier.classes[int(instruct_1st_half[0][0])]
    else:
        instruct_str_1st_half = 'UNKNOWN'
    if instruct_2nd_half[0][0] != -1:
        instruct_str_2nd_half = direction_classifier.classes[int(instruct_2nd_half[0][0])]
    else:
        instruct_str_2nd_half = 'UNKNOWN'
    
    return [int(instruct[0][0]), int(instruct_1st_half[0][0]), int(instruct_2nd_half[0][0])], [instruct_str, instruct_str_1st_half, instruct_str_2nd_half]



def process_batch(batch, output_dir, direction_classifier):
    # Initialize the direction classifier to categorize instructions
    # direction_classifier = DirectionClassifier()

    ## Data Statistics
    # for data_dir in [train_data_dir, valid_data_dir]:
        
        # JSON output directory
        # output_dir = f"{save_dir_json}{data_dir.split('/')[-2]}/"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        # json_output_dir = directory_path+'meta.json'
        # json_data, json_data_exists = init_or_read_json(json_output_dir)


        # print("#"*10)
        # print(f"Processing: {data_dir}")
        # dataset_ = DrivingData(data_dir+'*')
        # print(f'Found {len(dataset_)} data files')
        # dataloader_ = DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=16)        
        
        
        
    # Prepare an instruction set with -1 representing unknown instructions
    # + the range of classifier's known classes
    instructions_set = [-1] + [i for i in range(direction_classifier.num_classes)]
    # Initialize a list to keep track of found instructions, can be used to calculate the histogram of classes if not limited by max number of entries after which the code stops searching.
    found_instruct = [0] * len(instructions_set)
    found_instruct_1st_half = [0] * len(instructions_set)
    found_instruct_2nd_half = [0] * len(instructions_set)
        
        # Iterate over the data batches
        # for batch in tqdm(dataloader_):
            # Extract data components from the current batch
    ego_state, neighbors, map_lanes, map_crosswalks, ego_future, object_type, file_name = batch[0][0], batch[1][0], batch[2][0], batch[3][0], batch[4][0], batch[6][0], batch[7][0]
    json_output_dir = f"{output_dir}{file_name}.json"
            # json_data, json_data_exists = init_or_read_json(json_output_dir)
            # if not json_data_exists:
            #     json_data[file_name] = {}
            # instructions_int, instructions_str = get_direction_instructions(ego_future, direction_classifier)
            # json_data_to_save = {
            #     'direction_instruct_0s_to_8s_int': str(instructions_int[0]),
            #     'direction_instruct_0s_to_4s_int': str(instructions_int[1]),
            #     'direction_instruct_4s_to_8s_int': str(instructions_int[2]),
            #     'direction_instruct_0s_to_8s_str': str(instructions_str[0]),
            #     'direction_instruct_0s_to_4s_str': str(instructions_str[1]),
            #     'direction_instruct_4s_to_8s_str': str(instructions_str[2]),
            # }
            
            # with open(json_output_dir, 'w') as file:
            #     json.dump(json_data, file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process driving data in parallel.")
    parser.add_argument("--mp", type=bool, required=False, help="use multiprocessing", default=False)
    parser.add_argument("--data_dir", type=bool, required=False, help="", default='/ibex/project/c2253/felembaa/waymo_dataset/training_interactive_32_full_14mar/')
    # parser.add_argument("--save_dir", type=bool, required=False, help="", default='/ibex/project/c2253/felembaa/waymo_dataset/instructions_samples/direction/')
    parser.add_argument("--save_dir_json", type=bool, required=False, help="", default='/ibex/project/c2253/felembaa/waymo_dataset/instructions_json/')
    # '/ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_32_small_13mar/'
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir_json = args.save_dir_json
    # JSON output directory
    output_dir = f"{save_dir_json}{data_dir.split('/')[-2]}/"

    num_cpus = 1

    print("#"*10)
    print(f"Processing: {data_dir}")
    dataset = DrivingData(data_dir+'*')
    print(f'Found {len(dataset)} data files')
    
    # Initialize the direction classifier to categorize instructions
    direction_classifier = DirectionClassifier()

    if not args.mp:
        data_loader = DataLoader(
        dataset, batch_size=1, 
        sampler= None, num_workers=4,
        shuffle=False,
        )
        for batch in tqdm(data_loader):
            process_batch(batch, output_dir, direction_classifier)
    else:
        num_workers = 0
        num_cpus = cpu_count()
        num_cpus = int(cpu_count()/(num_workers+1))
        print(f"MP WILL START USING {num_cpus} cores")
        data_loaders = []
        subset_len = int(len(dataset)/num_cpus)
        data_loaders = [
            DataLoader(
                Subset(dataset, indices=np.arange(i*subset_len, (i+1)*subset_len)),
                batch_size=1, 
                sampler= None, 
                num_workers=num_workers,
                shuffle=False,)
                for i in range(num_cpus-1)
        ]
        data_loaders.append(
            DataLoader(
                Subset(dataset, indices=np.arange((num_cpus-1)*subset_len, len(dataset))),
                batch_size=1, 
                sampler= None, 
                num_workers=num_workers,
                shuffle=False,)
                )
        args_list = [(loader, output_dir, direction_classifier) for loader in data_loaders]
        with Pool(processes=num_cpus) as pool:
            pool.map(process_batch_loader, args_list)
    

def process_batch_loader(args_list):
    loader, output_dir, direction_classifier = args_list
    for batch in tqdm(loader):
        process_batch(batch, output_dir, direction_classifier)
