# python extract_instruct_future_ddp.py --input_folder '<internal_user_root>/waymo_dataset/validation_interactive_original'
import os
import numpy as np
import torch
# from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset, Dataset
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

from traj_utils import *
from mm_viz import *

def calculate_distance(x,y):
    return (x - y).norm().item()
    # return (torch.tensor(x) - torch.tensor(y)).norm().item()

def get_car_motion_action(ego_future, straight_threshold = 0.4, soft_turn_threshold = 0.5, turn_threshold = 0.8, sharp_turn_threshold = np.pi/2 + 0.6, uturn_threshold = 2.4, directions=[]):
        # turning direction of ego vehicle
        #['unknown', 'not move', 'move straight', 'take a soft left turn', 'take a soft right turn', 'take a left turn', 'take a right turn', 'take a sharp left turn', 'take a sharp right turn', 'take a left U-turn', 'take a right U-turn', 'invalid']
    output_actions = []
    if calculate_distance(ego_future[0,0,:2], ego_future[0,-1,:2]) < 0.5: #movement between first step and last step is less than 0.5m
        if calculate_distance(ego_future[0,0,:2], ego_future[0,int(ego_future.shape[1]/2),:2]) > 1.0: #but if also half way in the future the movement is more than 2m, this is flagged as invalid movement. (Some noisy examples where the object move in a weird way then go back to its original position)
            output_actions.append(directions[-1])
            # return directions[-1]
        else:
            output_actions.append(directions[1])
            # return directions[1] # not moving
    else:
        if abs(ego_future[0,-1,2])< straight_threshold:
            output_actions.append(directions[2])
            # return directions[2] # straight
        elif abs(ego_future[0,-1,2])>soft_turn_threshold and abs(ego_future[0,-1,2])<turn_threshold:
            if ego_future[0,-1,2]>0:
                output_actions.append(directions[3])
                # return directions[3] # 'soft left turn'
            else:
                output_actions.append(directions[4])
                # return directions[4] # 'soft right turn'
        elif abs(ego_future[0,-1,2])>turn_threshold and abs(ego_future[0,-1,2])<sharp_turn_threshold:
            if ego_future[0,-1,2]>0:
                output_actions.append(directions[5])
                # return directions[5] # 'left turn'
            else:
                output_actions.append(directions[6])
                # return directions[6] # 'right turn'
        elif abs(ego_future[0,-1,2])>sharp_turn_threshold and abs(ego_future[0,-1,2])<uturn_threshold:
            if ego_future[0,-1,2]>0:
                output_actions.append(directions[7])
                # return directions[7] # 'sharp left turn'
            else:
                output_actions.append(directions[8])
                # return directions[8] # 'sharp right turn'
        elif abs(ego_future[0,-1,2])>uturn_threshold:
            if ego_future[0,int(ego_future.shape[1]/2),2]>0: # comparison show be made on half way heading, not final heading as the car could flip its turning angle sign
                output_actions.append(directions[9])
                # return directions[9] # 'left U-turn'
            else:
                output_actions.append(directions[10])
                # return directions[10] # 'right U-turn'

    # Acceleration, deceleration:
    
    
    
    if len(output_actions)==0:
        output_actions.append(directions[0])
    return output_actions
        # return directions[0] # unknown

def classify_movement(traj):
    traj_ = traj[traj.mean(-1)!=0] # remove zero entries
    # vel_ = vel[traj.mean(-1)!=0]
    if len(traj_) > len(traj)/2:
        if calculate_distance(traj_[0], traj_[-1]) < 0.5: #movement between first step and last step is less than 0.5m
            if calculate_distance(traj_[0], traj_[int(traj_.shape[0]/2)]) < 1.0: #but if also half way in the future the movement is more than 2m, this is flagged as invalid movement. (Some noisy examples where the object move in a weird way then go back to its original position)
                status = "not move"
            else:
                status = "invalid"
            return status
        vel = abs_distance_to_velocity(traj_)[1:][::10]
    else:
        # status = "Missing traj values"
        status = "invalid"
        return status
    # Calculate the magnitude of velocity for each step
    velocity_magnitudes = np.linalg.norm(vel, axis=-1)
    
    # # Calculate the change in velocity magnitude between each step
    # velocity_changes = np.diff(velocity_magnitudes)
    # # Determine if the agent is generally accelerating or decelerating
    # average_change = np.mean(velocity_changes)

    # Calculate the change in velocity magnitude between each first and last step velocity
    velocity_changes = velocity_magnitudes[-1]-velocity_magnitudes[0]
    # Determine if the agent is generally accelerating or decelerating
    average_change = velocity_changes

    # Define thresholds for differentiating between movement states
    acceleration_threshold = 0.13  # Adjust as needed
    deceleration_threshold = -0.13 # Adjust as needed
    moving_threshold = 0.1         # Adjust as needed to distinguish between stopping/starting/not moving
    
    # Initial status assuming the agent is not moving
    status = "unknown"
    
    # Check if the agent is moving
    if np.any(velocity_magnitudes > moving_threshold):
        # Determine if the agent is generally accelerating or decelerating
        average_change = np.mean(velocity_changes)
        # print(f"Average Acceleration: {average_change}")
        if average_change > acceleration_threshold:
            status = "accelerate"
        elif average_change < deceleration_threshold:
            status = "decelerate"
        else:
            status = "move with a constant velocity"
        
        # Further checks for stopping or starting by considering the beginning and end velocities
        # if velocity_magnitudes[:1].mean() < moving_threshold and np.any(velocity_magnitudes[1:] > moving_threshold):
        if velocity_magnitudes[0].mean() < moving_threshold and np.any(velocity_magnitudes > moving_threshold):
            status = "start to move"
        elif velocity_magnitudes[-1] < moving_threshold and np.any(velocity_magnitudes > moving_threshold):
            status = "stop"
    else:
        # If none of the above conditions are met, the agent is not moving
        status = "not move"
    

    return status

def get_prompt(turn_directions, action_):
    text_ = "I am developing a set of instructions to guide an autonomous vehicle, referred to as the ego vehicle or Agent A, through a predefined trajectory. These instructions are segmented into three distinct parts to detail the vehicle's intended movements:"
    text_ += "\n1) Overall Turn Action: This defines the vehicle's direction or change in direction."
    text_ += "\n2) Motion Action in the First Half of the Forecast Period: Specifies the vehicle's movement within the first 4 seconds into the future."
    text_ += "\n3) Motion Action in the Second Half of the Forecast Period: Details the vehicle's movement from the 4th to the 8th second into the future."
    text_ += f"\nThe Overall Turn Action can be any of the following: {turn_directions[1:-1]}."
    text_ += f"\nThe Motion Action encompasses: {action_}."
    text_ += "\nExamples of basic instructions, directly employing the predefined sets without modification, include:"
    text_ += "\n- Instructing the ego vehicle to remain stationary."
    text_ += "\n- Commanding the ego vehicle to execute a soft left turn while maintaining constant velocity."
    text_ += "\n- Directing the ego vehicle to perform a soft right turn, initially accelerating then decelerating."
    text_ += "\n- Ordering the ego vehicle to proceed straight, commencing movement from a stationary position."
    text_ += "\n- Guiding the ego vehicle through a right U-turn, decelerating before accelerating."
    text_ += "\n- Advising the ego vehicle to undertake a soft right turn at a constant speed."
    text_ += "\nObjective: Refine the following instruction to enhance clarity and facilitate compliance by the driver."
    return text_



def process_batch(batch, output_folder, object_type_map, turn_directions, action_):
    inputs = {
        'ego_state': batch[0],
        'neighbors_state': batch[1],
        'map_lanes': batch[2],
        'map_crosswalks': batch[3],
    }
    ego_future = batch[4]
    neighbor_future = batch[5]
    filename = batch[7][0]

    object_type_int = int(((inputs['ego_state'][0, -1, 8:].argmax(-1) + 1) * inputs['ego_state'][0, -1, 8:].sum(-1)).item())

    turn = get_car_motion_action(ego_future, directions=turn_directions)[0]
    move1 = classify_movement(ego_future[0, :40, 0:2])
    move2 = classify_movement(ego_future[0, 40:, 0:2])
    contrastive_turn = get_contrastive_turn(turn, turn_directions)
    valid_action = turn != 'unknown' and turn != 'invalid' and move1 != 'unknown' and move1 != 'invalid' and move2 != 'unknown' and move2 != 'invalid' and object_type_map[object_type_int] == 'car'
    # context_prompt = get_prompt(turn_directions, action_)
    # instruction = "Make the ego vehicle (the first agent) [turn], [move]."
    debug = True
    if debug:
        if valid_action:
            viz_multimodal(inputs, ego_future.unsqueeze(0)[...,:2], ego_future).savefig('ex.png')
            print(turn)
            print(contrastive_turn[1])
            ''
    else:
        if valid_action:
            # contrastive_turn=turn
            # contrastive_move1=move1
            # contrastive_move2=move2
            np.save(output_folder + '/' + filename + '.npy', np.array([valid_action, turn, move1, move2, contrastive_turn[1]]))
        else:
            np.save(output_folder + '/' + filename + '.npy', np.array([valid_action]))
    
    ''

def get_contrastive_turn(turn, turn_directions):
    turn_idx = [i for i in range(len(turn_directions)) if turn_directions[i]== turn][0]
    if turn_idx==0 or turn_idx==len(turn_directions): # unknown, invalid
        return False, None
    else:
        if turn_idx==1: # not move,
            return True, turn 
        elif turn_idx==2: # 'move straight'
            return True, turn_directions[6]# turn_directions[5:7] + turn_directions[9:11] # (True, ['take a left turn', 'take a right turn', 'take a left U-turn', 'take a right U-turn'])
        elif turn_idx==3:
            return True, turn_directions[turn_idx+1], 
        elif turn_idx==4:
            return True, turn_directions[turn_idx-1]
        else:
            return True, turn_directions[2] # straight
        # elif turn_idx==5: # right
        #     return True, turn_directions[2] # straight
        # elif turn_idx==6:
        #     return True, turn_directions[2] # straight
        # elif turn_idx==7:
        #     return True, turn_directions[2]
        # elif turn_idx==8:
        #     return True, turn_directions[2]
        # elif turn_idx==7:
        #     return True, turn_directions[2]
        # elif turn_idx==8:
        #     return True, turn_directions[2]
        

    if samples['valid_action'][batch_i]:
        # print(f"Original: {samples['turn'][batch_i]}")
        turn_idx = [i_ for i_ in range(len(turn_acts)) if turn_acts[i_]== samples['turn'][batch_i]][0]
        if turn_acts[turn_idx] == 'move straight':
            samples['turn'][batch_i] = 'take a right turn'
        elif turn_acts[turn_idx] == 'U-turn':
            samples['turn'][batch_i] = 'move straight'
        elif 'left' in turn_acts[turn_idx]:
            samples['turn'][batch_i] = turn_acts[turn_idx+1]
        elif 'right' in turn_acts[turn_idx]:
            samples['turn'][batch_i] = turn_acts[turn_idx-1]
        # print(f"Updated: {samples['turn'][batch_i]}")
        data_name += "\nUsed: "+samples['turn'][batch_i]
def get_plausible_position(map, current_position, current_heading, relative_future_location_angle, max_ms = 32):
    ## map: drivable lanes
    ## current_position: Agent's current 2D location
    ## current_heading: Agent's current heading angle
    ## relative_future_location_angle: Angle to future location based on current location (or possible set of angles)
    ## return False if not plausible, future position and angle to future direction. The angle to future direction is what we need from this
    return True

def on_map(map, position):
    ## Check if position is on map
    return True
def main():
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser(description="Process driving data in parallel.")
    # parser.add_argument("--input_folder", type=str, required=True,
    #                     help="Path to the input dataset directory.")
    parser.add_argument("--input_folder", type=str, required=False,
                        help="Path to the input dataset directory.", default='<internal_waymo_dataset_root>/validation_interactive_original_20')
    # <internal_waymo_dataset_root>/training_interactive_original_20
    # parser.add_argument("--debug", type=str, required=False,
    #                     help="", default=True)
    parser.add_argument("--mp", type=bool, required=False,
                        help="", default=False)
    args = parser.parse_args()
    
    debug = True
    input_path = args.input_folder
    output_folder = input_path+'_act'
    if not debug:
        os.makedirs(output_folder, exist_ok=True)

    dataset = DrivingData(input_path+'/*')
    print(f'found {len(dataset)} scenes.')


    object_type_map = ['unknown','car', 'pedestrian', 'cyclist']
    turn_directions = ['unknown', 'not move', 'move straight', 'take a soft left turn', 'take a soft right turn', 'take a left turn', 'take a right turn', 'take a sharp left turn', 'take a sharp right turn', 'take a left U-turn', 'take a right U-turn', 'invalid']
    action_ = ['not move', 'start to move', 'stop', 'move with a constant velocity', 'accelerate', 'decelerate']

    # if num_processes is None:
    #     num_processes = min(cpu_count(), 8)
    if not args.mp:
        data_loader = DataLoader(
        dataset, batch_size=1, 
        sampler= None, num_workers=4,
        shuffle=False,
        )
        for batch in tqdm(data_loader):
            process_batch(batch, output_folder, object_type_map, turn_directions, action_)
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
        args_list = [(loader, output_folder, object_type_map, turn_directions, action_) for loader in data_loaders]
        with Pool(processes=num_cpus) as pool:
            pool.map(process_batch_loader, args_list)

def process_batch_loader(args_list):
    loader, output_folder, object_type_map, turn_directions, action_ = args_list
    for batch in tqdm(loader):
        process_batch(batch, output_folder, object_type_map, turn_directions, action_)


if __name__ == "__main__":
    main()