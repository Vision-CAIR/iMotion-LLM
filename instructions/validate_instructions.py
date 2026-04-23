# Title: Validate Instructions
# Description: Code to validate each category of the defined instructions, 
# the code should sample equal number of examples per category so the evaluator 
# can manually inspect and record accuracy score

# Import necessary libraries and modules
from vizualize_instructions import viz_instruction_sample
import torch
from torch.utils.data import DataLoader
from direction_instructions import DirectionClassifier
from acceleration_instructions import AccelerationClassifier
from speed_instructions import SpeedClassifier
from shape_instructions import ShapeClassifier
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
REPO_ROOT = Path(__file__).resolve().parents[1]
GAMEFORMER_ROOT = REPO_ROOT / "gameformer"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GAMEFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(GAMEFORMER_ROOT))
from utils.inter_pred_utils import DrivingData
import json
import shutil
from pathlib import Path


save_figures = True
generate_json_and_statistics = False
train_data_dir = str(REPO_ROOT / "data" / "processed" / "waymo" / "gameformer" / "train") + "/"
valid_data_dir = str(REPO_ROOT / "data" / "processed" / "waymo" / "gameformer" / "val") + "/"
save_dir = str(REPO_ROOT / "outputs" / "instruction_samples")
save_dir_json = str(REPO_ROOT / "outputs" / "instruction_json") + "/"

if save_figures:
    # Clean up the directory
    shutil.rmtree(Path(save_dir), ignore_errors=True)

    # Maximum number of examples to be saved per instruction category
    max_examples_per_instruct = 20

    # Load the dataset and create a DataLoader for batch processing
    data_dir = valid_data_dir
    dataset_ = DrivingData(data_dir+'*')
    print(f'Found {len(dataset_)} data')
    dataloader_ = DataLoader(dataset_, batch_size=1, shuffle=True)

    # Initialize the direction classifier to categorize instructions
    direction_classifier = DirectionClassifier()
    acceleration_classifier = AccelerationClassifier()
    speed_classifier = SpeedClassifier()
    shape_classifier = ShapeClassifier()

    # Prepare an instruction set with -1 representing unknown instructions
    # + the range of classifier's known classes
    # direction_set = [-1] + [i for i in range(direction_classifier.num_classes)]
    # movement_set = [-1] + [i for i in range(acceleration_classifier.num_movement_classes)]
    # acceleration_set = [-1] + [i for i in range(acceleration_classifier.num_acceleration_classes)]
    direction_set = [i for i in range(direction_classifier.num_classes)]
    movement_set = [i for i in range(acceleration_classifier.num_movement_classes)]
    acceleration_set = [i for i in range(acceleration_classifier.num_acceleration_classes)]
    speed_set = [i for i in range(speed_classifier.num_classes)]
    shape_set = [i for i in range(shape_classifier.num_classes)]
    # Initialize a list to keep track of found instructions, can be used to calculate the histogram of classes if not limited by max number of entries after which the code stops searching.
    found_direction_count = [0] * len(direction_set)
    found_movement_count = [0] * len(movement_set)
    found_acceleration_count = [0] * len(acceleration_set)
    found_speed_count = [0] * (len(speed_set)-1) # ignoring the invalid
    found_shape_count = [0] * len(shape_set)


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

            # Get shape instructions
            shape_instruct = shape_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx)

            # Get the batch (batch size of 1 in this code) instruction for the current ego vehicle's future trajectory
            direction_instruct, _ = direction_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx)
            direction_instruct_1st_half, _ = direction_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_1st_half)
            direction_instruct_2nd_half, _ = direction_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_2nd_half)
            
            # Get acceleration instructions
            movement_instruct, acceleration_instruct = acceleration_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx)
            movement_instruct_1st_half, acceleration_instruct_1st_half = acceleration_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_1st_half)
            movement_instruct_2nd_half, acceleration_instruct_2nd_half = acceleration_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_2nd_half)

            # Get speed instructions
            speed_instruct, speeds_0to8 = speed_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx)
            speed_instruct_1st_half, speeds_4to8 = speed_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_1st_half)
            speed_instruct_2nd_half, _ = speed_classifier.get_batch_instruct(ego_future=ego_future[None], sampling_idx=sampling_idx_2nd_half)
            speed_0, speed_8, speed_4 = speeds_0to8[:,0], speeds_0to8[:,-1], speeds_4to8[:,0]



            # for single instruction, needs to be modified to support batch data
            direction_instruct, direction_instruct_1st_half, direction_instruct_2nd_half = int(direction_instruct[0][0]), int(direction_instruct_1st_half[0][0]), int(direction_instruct_2nd_half[0][0])
            movement_instruct, movement_instruct_1st_half, movement_instruct_2nd_half = int(movement_instruct[0][0]), int(movement_instruct_1st_half[0][0]), int(movement_instruct_2nd_half[0][0])
            acceleration_instruct, acceleration_instruct_1st_half, acceleration_instruct_2nd_half = int(acceleration_instruct[0][0]), int(acceleration_instruct_1st_half[0][0]), int(acceleration_instruct_2nd_half[0][0])
            speed_instruct, speed_instruct_1st_half, speed_instruct_2nd_half = int(speed_instruct[0][0]), int(speed_instruct_1st_half[0][0]), int(speed_instruct_2nd_half[0][0])
            speed_0, speed_8, speed_4 = speed_0[0], speed_8[0], speed_4[0]
            # shape_instruct = int(shape_instruct[0][0])
            
            
            instructions_dict = {
                'direction':{
                    '0to8': direction_instruct,
                    '0to4': direction_instruct_1st_half,
                    '4to8': direction_instruct_2nd_half,
                }, 
                'movement':{
                    '0to8':movement_instruct,
                    '0to4':movement_instruct_1st_half,
                    '4to8':movement_instruct_2nd_half,
                }, 
                'acceleration':{
                    '0to8':acceleration_instruct,
                    '0to4':acceleration_instruct_1st_half,
                    '4to8':acceleration_instruct_2nd_half,
                }, 
                'speed':{
                    '0to8':speed_instruct,
                    '0to4':speed_instruct_1st_half,
                    '4to8':speed_instruct_2nd_half,
                    # 'speed_0':f"{speed_0:.2f} m/s",
                    # 'speed_4':f"{speed_4:.2f} m/s",
                    # 'speed_8':f"{speed_8:.2f} m/s",
                    'speed_0':f"{int(speed_0*3.6)} km/h",
                    'speed_4':f"{int(speed_4*3.6)} km/h",
                    'speed_8':f"{int(speed_8*3.6)} km/h",
                }, 
                # 'shape':{
                #     'type': shape_instruct
                # }
                }
            
            # for instruct_category, instruct_category_value in instructions_dict.items():
            #     for time_category, time_category_value in instruct_category_value.items():
            #         instructions_dict[instruct_category][time_category]
            
            for instruct_category, instruct_category_value in instructions_dict.items():
                if instruct_category == 'shape':
                    current_instruct = instruct_category_value['type']
                else:
                    current_instruct = instruct_category_value['0to8']
                # Check if the instruction is within our set and not exceeded the max examples per instruction
                if current_instruct != -1:
                    if instruct_category == 'direction':
                        instruct_str = direction_classifier.classes[current_instruct]
                        if instruct_category_value['0to4'] != -1:
                            instruct_str_1st_half = direction_classifier.classes[instruct_category_value['0to4']]
                        else:
                            continue
                        if instruct_category_value['4to8'] != -1:
                            instruct_str_2nd_half = direction_classifier.classes[instruct_category_value['4to8']]
                        else:
                            continue
                        if found_direction_count[current_instruct]>=max_examples_per_instruct:
                            continue
                        found_direction_count[current_instruct] += 1
                        example_number = found_direction_count[current_instruct]
                    elif instruct_category == 'movement':
                        instruct_str = acceleration_classifier.movement_classes[current_instruct]
                        if instruct_category_value['0to4'] != -1:
                            instruct_str_1st_half = acceleration_classifier.movement_classes[instruct_category_value['0to4']]
                        else:
                            continue
                        if instruct_category_value['4to8'] != -1:
                            instruct_str_2nd_half = acceleration_classifier.movement_classes[instruct_category_value['4to8']]
                        else:
                            continue
                        if found_movement_count[current_instruct]>=max_examples_per_instruct:
                            continue
                        found_movement_count[current_instruct] += 1
                        example_number = found_movement_count[current_instruct]
                    elif instruct_category == 'acceleration':
                        instruct_str = acceleration_classifier.acceleration_classes[current_instruct]
                        if instruct_category_value['0to4'] != -1:
                            instruct_str_1st_half = acceleration_classifier.acceleration_classes[instruct_category_value['0to4']]
                        else:
                            continue
                        if instruct_category_value['4to8'] != -1:
                            instruct_str_2nd_half = acceleration_classifier.acceleration_classes[instruct_category_value['4to8']]
                        else:
                            continue
                        if found_acceleration_count[current_instruct]>=max_examples_per_instruct:
                            continue
                        found_acceleration_count[current_instruct] += 1
                        example_number = found_acceleration_count[current_instruct]
                    elif instruct_category == 'speed':
                        instruct_str = speed_classifier.classes[current_instruct]
                        speed_0_str = instruct_category_value['speed_0']
                        speed_8_str = instruct_category_value['speed_8']
                        speed_4_str = instruct_category_value['speed_4']
                        if instruct_category_value['0to4'] != -1:
                            instruct_str_1st_half = speed_classifier.classes[instruct_category_value['0to4']]
                        else:
                            continue
                        if instruct_category_value['4to8'] != -1:
                            instruct_str_2nd_half = speed_classifier.classes[instruct_category_value['4to8']]
                        else:
                            continue
                        if instruct_str == 'INVALID' or instruct_str_1st_half == 'INVALID' or instruct_str_2nd_half == 'INVALID':
                            continue
                        if found_speed_count[current_instruct]>=max_examples_per_instruct:
                            continue
                        found_speed_count[current_instruct] += 1
                        example_number = found_speed_count[current_instruct]
                    elif instruct_category == 'shape':
                        # ego_future_sampled = torch.cat((ego_future[::10],ego_future[-1:]), dim=0)
                        # fig = viz_instruction_sample(ego_state, neighbors, map_lanes, map_crosswalks, ego_future_sampled)
                        # fig.savefig('ex.png')
                        instruct_str = shape_classifier.classes[current_instruct]
                        if found_shape_count[current_instruct]>=max_examples_per_instruct:
                            continue
                        found_shape_count[current_instruct] += 1
                        example_number = found_shape_count[current_instruct]

                else:
                    instruct_str = 'INVALID'
                    instruct_str_1st_half = 'INVALID'
                    instruct_str_2nd_half = 'INVALID'
                    continue
                
                # Visualize the instruction sample
                ego_future_sampled = torch.cat((ego_future[::10],ego_future[-1:]), dim=0)
                # fig = viz_instruction_sample(ego_state, neighbors, map_lanes, map_crosswalks, ego_future)
                fig = viz_instruction_sample(ego_state, neighbors, map_lanes, map_crosswalks, ego_future_sampled)
                # plt.title(f"{instruct_str} ({instruct_str_1st_half} -> {instruct_str_2nd_half})")  # Set the figure title to the instruction
                if instruct_category == 'shape':
                    title_str = f"{instruct_str}"
                else:
                    title_str = f"{instruct_str}\n({instruct_str_1st_half}->{instruct_str_2nd_half})"
                    if instruct_category == 'speed':
                        title_str = title_str + f"\nspeed @t=0.1s: {speed_0_str} @t=4s: {speed_4_str} @t=8s: {speed_8_str}"
                fig.suptitle(title_str)  # Set the figure title to the instruction
                plt.tight_layout()
                # Prepare the directory path for saving the figure
                directory_path = f"{save_dir}/{instruct_category}/{instruct_str.replace(' ', '_').replace('-', '_')}/"
                os.makedirs(directory_path, exist_ok=True)  # Create the directory if it doesn't exist
                # Save the figure to the specified directory
                if instruct_category == 'shape':
                    fig.savefig(f"{directory_path}ex{example_number}.png", dpi=300)
                else:
                    fig.savefig(f"{directory_path}ex{example_number}_{instruct_str_1st_half}_THEN_{instruct_str_2nd_half}.png", dpi=300)
                plt.close()
            
        # Check if we've collected enough examples for each instruction
        if sum(found_direction_count) + sum(found_movement_count) + sum(found_acceleration_count) + sum(found_speed_count) + sum(found_shape_count) >= max_examples_per_instruct* (len(found_direction_count)+len(found_movement_count)+len(found_acceleration_count) + len(found_speed_count) + len(found_shape_count)):
            break  # Exit the loop if we have enough samples for each instruction
            
