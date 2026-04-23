# python extract_instruct_future_ddp.py --input_folder '/ibex/user/felembaa/waymo_dataset/validation_interactive_original'
import os
import numpy as np
import torch
# from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset, Dataset
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm
import math


def get_unit_vector(pointA, pointB):
    # Vector from A to B
    vector_AB = pointB - pointA
    # Normalize the vector to get the unit vector
    magnitude_AB = np.linalg.norm(vector_AB)
    unit_vector_AB = vector_AB / magnitude_AB if magnitude_AB != 0 else vector_AB
    return unit_vector_AB, magnitude_AB


def rel_angle(point1, point2):
    if point1.sum(-1)==0:
        angle = np.arctan2(point2[1], point2[0]) 
    else:
        point_ = point2-point1
        angle = np.arctan2(point_[1], point_[0])
    return minus_2pi(angle)

def minus_2pi(angle):
    while angle > np.pi and angle>0:
        angle = 2*np.pi - angle
    while angle < -np.pi and angle<0:
        angle = angle + 2*np.pi
    return angle


def get_car_motion_action(future, history=None, straight_threshold = 10*np.pi/180, soft_turn_threshold = 10*np.pi/180, turn_threshold = 30*np.pi/180, sharp_turn_threshold = 110*np.pi/180, uturn_threshold = 115*np.pi/180, directions=[]):
        # turning direction of ego vehicle
        #['unknown', 'not move', 'move straight', 'take a soft left turn', 'take a soft right turn', 'take a left turn', 'take a right turn', 'take a sharp left turn', 'take a sharp right turn', 'take a left U-turn', 'take a right U-turn', 'invalid']
    
    # history used to get the correction angle if provided
    output_actions = []
    # if global_angle == 0:
    if history is not None:
        correction_angle = rel_angle(history[-2,:2],history[-1,:2])
    else:
        correction_angle = rel_angle(future[0,:2],future[1,:2]) # use the first motion angle as a reference
    
    # else:
        # correction_angle = global_angle
    angle = rel_angle(future[0,:2],future[-1,:2]) - correction_angle


    # print(f'ANGLE: {math.degrees(angle)}')
    
    halfway_angle = rel_angle(future[0,:2],future[int(future.shape[0]/2),:2]) - correction_angle # -global_angle
    if np.linalg.norm(future[0,:2]-future[-1,:2]) < 0.5: #movement between first step and last step is less than 0.5m
        if np.linalg.norm(future[0,:2]- future[int(future.shape[0]/2),:2]) > 1.0: #but if also half way in the future the movement is more than 2m, this is flagged as invalid movement. (Some noisy examples where the object move in a weird way then go back to its original position)
            output_actions.append(directions[-1])
            # return directions[-1]
        else:
            output_actions.append(directions[1])
            # return directions[1] # not moving
    else:
        if abs(angle)< straight_threshold:
            output_actions.append(directions[2])
            # return directions[2] # straight
        elif abs(angle)>soft_turn_threshold and abs(angle)<turn_threshold:
            if angle>0:
                output_actions.append(directions[3])
                # return directions[3] # 'soft left turn'
            else:
                output_actions.append(directions[4])
                # return directions[4] # 'soft right turn'
        elif abs(angle)>turn_threshold and abs(angle)<sharp_turn_threshold:
            possible_uturn_angle = rel_angle(future[-2,:2],future[-1,:2]) - correction_angle
            if abs(possible_uturn_angle)>uturn_threshold:
                if halfway_angle>0: # comparison show be made on half way heading, not final heading as the car could flip its turning angle sign
                    output_actions.append(directions[9])
                    # return directions[9] # 'left U-turn'
                else:
                    output_actions.append(directions[10])
                    # return directions[10] # 'right U-turn'
            elif angle>0:
                output_actions.append(directions[5])
                # return directions[5] # 'left turn'
            else:
                output_actions.append(directions[6])
                # return directions[6] # 'right turn'
        elif abs(angle)>sharp_turn_threshold and abs(angle)<uturn_threshold:
            if future[-1,2]>0:
                output_actions.append(directions[7])
                # return directions[7] # 'sharp left turn'
            else:
                output_actions.append(directions[8])
                # return directions[8] # 'sharp right turn'
        elif abs(angle)>uturn_threshold:
            if halfway_angle>0: # comparison show be made on half way heading, not final heading as the car could flip its turning angle sign
                output_actions.append(directions[9])
                # return directions[9] # 'left U-turn'
            else:
                output_actions.append(directions[10])
                # return directions[10] # 'right U-turn'

    if len(output_actions)==0:
        output_actions.append(directions[0])
    return output_actions, angle

def abs_distance_to_velocity(abs_distance):
    convert_to_np=False
    if 'numpy' in str(type(abs_distance)):
        abs_distance = torch.tensor(abs_distance)
        convert_to_np=True
    vel = torch.cat((torch.zeros_like(abs_distance[...,0:1,:]), abs_distance[...,1:,:]-abs_distance[...,:-1,:]), dim=-2)
    if convert_to_np:
        vel = vel.numpy()
    return vel

def classify_movement(traj, history):
    traj_ = traj[traj.mean(-1)!=0] # remove zero entries
    hist_traj_ = history[history.mean(-1)!=0] # remove zero entries
    # vel_ = vel[traj.mean(-1)!=0]
    if len(traj_) > len(traj)/2:
        if np.linalg.norm(traj_[0] - traj_[-1]) < 0.5: #movement between first step and last step is less than 0.5m
            if np.linalg.norm(traj_[0] - traj_[int(traj_.shape[0]/2)]) < 1.0: #but if also half way in the future the movement is more than 2m, this is flagged as invalid movement. (Some noisy examples where the object move in a weird way then go back to its original position)
                status = "not moving"
            else:
                status = "unknown motion"
            return status
        
        dtype_to_torch = False
        if 'ndarray' in str(type(traj_)):
            dtype_to_torch = True
            traj_ = torch.tensor(traj_)
            hist_traj_ = torch.tensor(hist_traj_)
        # vel = abs_distance_to_velocity(traj_)[1:][::10]
        vel = abs_distance_to_velocity(traj_[::10])[1:]
        hist_vel = abs_distance_to_velocity(hist_traj_)[1:]*10

    else:
        # status = "Missing traj values"
        status = "unknown motion"
        return status
    # Calculate the magnitude of velocity for each step
    velocity_magnitudes = np.linalg.norm(vel, axis=-1)
    hist_vel_mag = np.linalg.norm(hist_vel, axis=-1)
    velocity_magnitudes = np.append(hist_vel_mag[-1],velocity_magnitudes)

    
    # # Calculate the change in velocity magnitude between each step
    # velocity_changes = np.diff(velocity_magnitudes)
    # # Determine if the agent is generally accelerating or decelerating
    # average_change = np.mean(velocity_changes)

    # Calculate the change in velocity magnitude between each first and last step velocity
    velocity_changes = [velocity_magnitudes[i+1]-velocity_magnitudes[i] for i in range(len(velocity_magnitudes)-1)]
    # Determine if the agent is generally accelerating or decelerating
    # average_change = velocity_changes

    # Define thresholds for differentiating between movement states
    moving_threshold = 0.5
    constant_vel_threshold = 0.2
    accel_threshold = 1.5
    abrupt_accel_threshold = 3
    

    
    # Initial status assuming the agent is not moving
    # status = "unknown"
    
    # Check if the agent is moving
    if not np.any(velocity_magnitudes > moving_threshold):
        status = ['not moving']
    else:
        # Determine if the agent is generally accelerating or decelerating
        average_change = np.mean(velocity_changes)
        # print(f"Average Acceleration: {average_change}")
        if abs(average_change) < constant_vel_threshold:
            status = "moving with a constant velocity"
        elif abs(average_change) < accel_threshold:
            status = "gradually accelerating" if average_change>0 else "gradually decelerating"
        elif abs(average_change) < abrupt_accel_threshold:
            status = "abruptly accelerating" if average_change>0 else "abruptly decelerating"
        else:
            status = "extremely accelerating" if average_change>0 else "extremely decelerating"

        if 'decelerating' in status:
            if velocity_magnitudes[-1] < moving_threshold:
                if len(velocity_magnitudes)>4:
                    status = "stopping"
                else:
                    status = status+" to stop" 
        elif 'accelerating' in status:
            if hist_vel_mag[-1] < moving_threshold:
                status = status + " starting to move"

    return status

def extract_instruct(future, history):

    object_type_map = ['unknown','car', 'pedestrian', 'cyclist']
    turn_directions = ['unknown', 'not move', 'move straight', 'take a soft left turn', 'take a soft right turn', 'take a left turn', 'take a right turn', 'take a sharp left turn', 'take a sharp right turn', 'take a left U-turn', 'take a right U-turn', 'invalid']
    # action_ = ['not move', 'start to move', 'stop', 'move with a constant velocity', 'accelerate', 'decelerate']

        #     turn = get_car_motion_action(agents_future[i], directions=turn_directions)[0]
        # move1 = classify_movement(agents_future[i, :40, 0:2])
        # move2 = classify_movement(agents_future[i, 40:, 0:2])
        # contrastive_turn = get_contrastive_turn(turn, turn_directions)
    
    
    
    turn1, global_angle = get_car_motion_action(future[:40], history,directions=turn_directions)
    turn1 = turn1[0]
    turn2, global_angle = get_car_motion_action(future[40:], directions=turn_directions)
    turn2 = turn2[0]
    turn, global_angle = get_car_motion_action(future[:], history, directions=turn_directions)
    turn = turn[0]

    # print('move')
    move = classify_movement(future[:, :2], history=history[:, :2])
    # print('move1')
    move1 = classify_movement(future[:40, :2], history=history[:, :2])
    # print('move2')
    move2 = classify_movement(future[40:, :2], history=future[:40, :2])

    # valid_action = turn != 'unknown' and turn != 'invalid' and move1 != 'unknown' and move1 != 'invalid' and move2 != 'unknown' and move2 != 'invalid' and object_type_map[object_type_int] == 'car'
    instructs = {'turn':turn,
    'turn_1':turn1,
    'turn_2':turn2,
    'move':move,
    'move_1':move1,
    'move_2':move2}
    return instructs


def extract_instruct_class(future, history):

    object_type_map = ['unknown','car', 'pedestrian', 'cyclist']
    turn_directions = ['unknown', 'not move', 'move straight', 'take a soft left turn', 'take a soft right turn', 'take a left turn', 'take a right turn', 'take a sharp left turn', 'take a sharp right turn', 'take a left U-turn', 'take a right U-turn', 'invalid']
    

    turn, global_angle = get_car_motion_action(future[:], history, directions=turn_directions)
    turn = turn[0]
    if 'not move' in turn:
        turn_cls = 1
    elif 'straight' in turn:
        turn_cls = 2
    elif 'right' in turn:
        turn_cls = 3
    elif 'left' in turn:
        turn_cls = 4
    else:
        turn_cls = 0
    return turn_cls


def extract_simple_turn_5_classes(future, history):

    object_type_map = ['unknown','car', 'pedestrian', 'cyclist']
    turn_directions = ['unknown', 'not move', 'move straight', 'take a soft left turn', 'take a soft right turn', 'take a left turn', 'take a right turn', 'take a sharp left turn', 'take a sharp right turn', 'take a left U-turn', 'take a right U-turn', 'invalid']
    turn, global_angle = get_car_motion_action(future[:], history, directions=turn_directions)
    turn = turn[0]
    if 'not move' in turn:
        turn_cls = 'not move'
        # turn_cls = 1
    elif 'straight' in turn:
        turn_cls = 'move straight'
        # turn_cls = 2
    elif 'right' in turn:
        turn_cls = 'turn right'
        # turn_cls = 3
    elif 'left' in turn:
        turn_cls = 'turn left'
        # turn_cls = 4
    else:
        turn_cls = 'move safely'
    return turn_cls