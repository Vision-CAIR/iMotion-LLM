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


save_figures = False
generate_json_and_statistics = True
train_data_dir = '/ibex/project/c2253/felembaa/waymo_dataset/training_interactive_32_full_14mar/'
valid_data_dir = '/ibex/project/c2253/felembaa/waymo_dataset/validation_interactive_32_small_13mar/'
save_dir = '/ibex/project/c2253/felembaa/waymo_dataset/instructions_samples/direction/'
save_dir_json = '/ibex/project/c2253/felembaa/waymo_dataset/instructions_json/'

if save_figures:
    # Directory where the samples will be saved
    
    # Maximum number of examples to be saved per instruction category
    max_examples_per_instruct = 100

    # Load the dataset and create a DataLoader for batch processing
    dataset_ = DrivingData(train_data_dir+'*')
    print(f'Found {len(dataset_)} data')
    dataloader_ = DataLoader(dataset_, batch_size=1, shuffle=True)

    # Initialize the direction classifier to categorize instructions
    direction_classifier = DirectionClassifier()

    # Prepare an instruction set with -1 representing unknown instructions
    # + the range of classifier's known classes
    instructions_set = [-1] + [i for i in range(direction_classifier.num_classes)]
    # Initialize a list to keep track of found instructions, can be used to calculate the histogram of classes if not limited by max number of entries after which the code stops searching.
    found_instruct = [0] * len(instructions_set)

    # Iterate over the data batches
    for batch in tqdm(dataloader_):
        # Extract data components from the current batch
        ego_state, neighbors, map_lanes, map_crosswalks, ego_future, object_type = batch[0][0], batch[1][0], batch[2][0], batch[3][0], batch[4][0], batch[6][0]
        
        # Filter to keep car objects based on the object_type mask
        car_mask = object_type[0] == 1
        if car_mask:
            ego_state = ego_state[:, :]
            num_neighbors = 10  # Consider the first 10 neighbors for visualization
            neighbors = neighbors[:num_neighbors, :, :]
            
            # Sample specific time steps for direction instructions (1Hz sampling from 0 to 70 and 79)
            sampling_idx = [ii for ii in range(80)][::10] + [79]  # 80 steps sampled at 10Hz + the last step
            sampling_idx_1st_half = sampling_idx[:5] # 0.1s to 4s in the future
            sampling_idx_2nd_half = sampling_idx[4:] # 4s to 8s in the future

            # Get the batch (batch size of 1 in this code) instruction for the current ego vehicle's future trajectory
            instruct, contrastive_instructs = direction_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx)
            instruct_1st_half, _ = direction_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_1st_half)
            instruct_2nd_half, _ = direction_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_2nd_half)
            
            # Check if the instruction is within our set and not exceeded the max examples per instruction
            if instruct[0][0] in instructions_set and found_instruct[int(instruct[0][0])] < max_examples_per_instruct:
                found_instruct[int(instruct[0][0])] += 1  # Increment the counter for this instruction
                
                # Determine the instruction string for title
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
                    
                # Visualize the instruction sample
                fig = viz_instruction_sample(ego_state, neighbors, map_lanes, map_crosswalks, ego_future)
                fig.suptitle(f"{instruct_str} ({instruct_str_1st_half} -> {instruct_str_2nd_half})")  # Set the figure title to the instruction
                
                # Prepare the directory path for saving the figure
                directory_path = f"{save_dir}/{instruct_str.replace(' ', '_').replace('-', '_')}/"
                os.makedirs(directory_path, exist_ok=True)  # Create the directory if it doesn't exist
                
                # Save the figure to the specified directory
                fig.savefig(f"{directory_path}ex{found_instruct[int(instruct[0][0])]}_{instruct_str_1st_half}_THEN_{instruct_str_2nd_half}.png", dpi=300)
                plt.close()
            else:
                continue  # Skip to the next batch if the instruction is not in our set or max examples reached
            
        # Check if we've collected enough examples for each instruction
        if sum(found_instruct) >= max_examples_per_instruct * (len(found_instruct)-1):
            break  # Exit the loop if we have enough samples for each instruction

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


def generate_json_instructions(dataloader_):
    # Initialize the direction classifier to categorize instructions
    direction_classifier = DirectionClassifier()

    ## Data Statistics
    for data_dir in [train_data_dir, valid_data_dir]:
        
        # JSON output directory
        output_dir = f"{save_dir_json}{data_dir.split('/')[-2]}/"
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        # json_output_dir = directory_path+'meta.json'
        # json_data, json_data_exists = init_or_read_json(json_output_dir)


        print("#"*10)
        print(f"Processing: {data_dir}")
        dataset_ = DrivingData(data_dir+'*')
        print(f'Found {len(dataset_)} data files')
        dataloader_ = DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=16)        
        
        
        
        # Prepare an instruction set with -1 representing unknown instructions
        # + the range of classifier's known classes
        instructions_set = [-1] + [i for i in range(direction_classifier.num_classes)]
        # Initialize a list to keep track of found instructions, can be used to calculate the histogram of classes if not limited by max number of entries after which the code stops searching.
        found_instruct = [0] * len(instructions_set)
        found_instruct_1st_half = [0] * len(instructions_set)
        found_instruct_2nd_half = [0] * len(instructions_set)
        
        # Iterate over the data batches
        for batch in tqdm(dataloader_):
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


