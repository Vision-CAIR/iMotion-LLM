import math
import numpy as np
import torch
from instructions.direction_instructions import DirectionClassifier
from instructions.acceleration_instructions import AccelerationClassifier
from instructions.speed_instructions import SpeedClassifier
from utils.data_utils import *

class futureNavigation():
    def __init__(
        self, 
        normalize_track=False,
        interpolate_track=False,
        num_classes=8,
        ):
        # Initialize the direction classifier to categorize instructions
        self.direction_classifier = DirectionClassifier(num_classes=num_classes)
        self.acceleration_classifier = AccelerationClassifier()
        self.speed_classifier = SpeedClassifier()

        # Sample specific time steps for direction instructions (1Hz sampling from 0 to 70 and 79)
        sampling_idx = [ii for ii in range(80)][::10] + [79]  # 80 steps sampled at 10Hz + the last step
        self.sampling_idx = sampling_idx
        self.sampling_idx_1st_half = sampling_idx[:5] # 0.1s to 4s in the future
        self.sampling_idx_2nd_half = sampling_idx[4:] # 4s to 8s in the future

        self.normalize_track = normalize_track
        self.interpolate_track = interpolate_track

    def get_navigation_dict(self, track):
        ## No batch support
        track = track.detach().clone()[None]
        if self.normalize_track:
            center, angle = track[0,self.sampling_idx[0],:2], self.get_heading_from_traj(track[0,0], track[0,1])
            # valid_mask = track[0, :,0]!=0
            track[0] = torch.tensor(agent_norm(track[0].numpy(), center.numpy(), angle, impute=False))
            
        # Get the batch (batch size of 1 in this code) instruction for the current ego vehicle's future trajectory
        direction_instruct, _ = self.direction_classifier.get_batch_instruct(ego_future=track, sampling_idx=self.sampling_idx)
        direction_instruct_1st_half, _ = self.direction_classifier.get_batch_instruct(ego_future=track, sampling_idx=self.sampling_idx_1st_half)
        direction_instruct_2nd_half, _ = self.direction_classifier.get_batch_instruct(ego_future=track, sampling_idx=self.sampling_idx_2nd_half)
        
        # Get acceleration instructions
        movement_instruct, acceleration_instruct = self.acceleration_classifier.get_batch_instruct(ego_future=track, sampling_idx=self.sampling_idx)
        movement_instruct_1st_half, acceleration_instruct_1st_half = self.acceleration_classifier.get_batch_instruct(ego_future=track, sampling_idx=self.sampling_idx_1st_half)
        movement_instruct_2nd_half, acceleration_instruct_2nd_half = self.acceleration_classifier.get_batch_instruct(ego_future=track, sampling_idx=self.sampling_idx_2nd_half)

        # Get speed instructions
        speed_instruct, speeds_0to8 = self.speed_classifier.get_batch_instruct(ego_future=track, sampling_idx=self.sampling_idx)
        speed_instruct_1st_half, speeds_4to8 = self.speed_classifier.get_batch_instruct(ego_future=track, sampling_idx=self.sampling_idx_1st_half)
        speed_instruct_2nd_half, _ = self.speed_classifier.get_batch_instruct(ego_future=track, sampling_idx=self.sampling_idx_2nd_half)
        speed_0, speed_8, speed_4 = speeds_0to8[:,0], speeds_0to8[:,-1], speeds_4to8[:,0]

        # for single instruction, needs to be modified to support batch data
        direction_instruct, direction_instruct_1st_half, direction_instruct_2nd_half = int(direction_instruct[0][0]), int(direction_instruct_1st_half[0][0]), int(direction_instruct_2nd_half[0][0])
        movement_instruct, movement_instruct_1st_half, movement_instruct_2nd_half = int(movement_instruct[0][0]), int(movement_instruct_1st_half[0][0]), int(movement_instruct_2nd_half[0][0])
        acceleration_instruct, acceleration_instruct_1st_half, acceleration_instruct_2nd_half = int(acceleration_instruct[0][0]), int(acceleration_instruct_1st_half[0][0]), int(acceleration_instruct_2nd_half[0][0])
        speed_instruct, speed_instruct_1st_half, speed_instruct_2nd_half = int(speed_instruct[0][0]), int(speed_instruct_1st_half[0][0]), int(speed_instruct_2nd_half[0][0])
        speed_0, speed_8, speed_4 = speed_0[0], speed_8[0], speed_4[0]
        
        instructions_dict = {
            'direction':{
                '0.1to8_cls': direction_instruct,
                '0.1to4_cls': direction_instruct_1st_half,
                '4to8_cls': direction_instruct_2nd_half,
                '0.1to8': self.direction_classifier.classes[direction_instruct] if direction_instruct!=-1 else 'INVALID',
                '0.1to4': self.direction_classifier.classes[direction_instruct_1st_half] if direction_instruct_1st_half!=-1 else 'INVALID',
                '4to8': self.direction_classifier.classes[direction_instruct_2nd_half] if direction_instruct_2nd_half!=-1 else 'INVALID',
                }, 
            'movement':{
                '0.1to8_cls':movement_instruct,
                '0.1to4_cls':movement_instruct_1st_half,
                '4to8_cls':movement_instruct_2nd_half,
                '0.1to8':self.acceleration_classifier.movement_classes[movement_instruct] if movement_instruct!=-1 else 'INVALID',
                '0.1to4':self.acceleration_classifier.movement_classes[movement_instruct_1st_half] if movement_instruct_1st_half!=-1 else 'INVALID',
                '4to8':self.acceleration_classifier.movement_classes[movement_instruct_2nd_half] if movement_instruct_2nd_half!=-1 else 'INVALID',
                }, 
            'acceleration':{
                '0.1to8_cls':acceleration_instruct,
                '0.1to4_cls':acceleration_instruct_1st_half,
                '4to8_cls':acceleration_instruct_2nd_half,
                '0.1to8':self.acceleration_classifier.acceleration_classes[acceleration_instruct] if acceleration_instruct!=-1 else 'INVALID',
                '0.1to4':self.acceleration_classifier.acceleration_classes[acceleration_instruct_1st_half] if acceleration_instruct_1st_half!=-1 else 'INVALID',
                '4to8':self.acceleration_classifier.acceleration_classes[acceleration_instruct_2nd_half] if acceleration_instruct_2nd_half!=-1 else 'INVALID',
                }, 
            'speed':{
                '0.1to8_cls':speed_instruct,
                '0.1to4_cls':speed_instruct_1st_half,
                '4to8_cls':speed_instruct_2nd_half,
                '0.1to8':self.speed_classifier.classes[speed_instruct] if speed_instruct!=-1 else 'INVALID',
                '0.1to4':self.speed_classifier.classes[speed_instruct_1st_half] if speed_instruct_1st_half !=-1 else 'INVALID',
                '4to8':self.speed_classifier.classes[speed_instruct_2nd_half] if speed_instruct_2nd_half!=-1 else 'INVALID',
                # '0.1':f"{speed_0:.2f} m/s",
                # '4':f"{speed_4:.2f} m/s",
                # '8':f"{speed_8:.2f} m/s",
                '0.1':f"{int(speed_0*3.6)} km/h",
                '4':f"{int(speed_4*3.6)} km/h",
                '8':f"{int(speed_8*3.6)} km/h",
                }, 
            }
        
        instructions_dict = {f"{k} {k_}":v[k_] for k,v in instructions_dict.items() for k_ in instructions_dict[k].keys()}
        # for k,v in instructions_dict.items():
        #     print(f"{k}: {v}")
        return instructions_dict
        
    def get_navigation_dict_history(self, track, sampling_idx=None):
        if sampling_idx is None:
            sampling_idx = [0,10]
        ## No batch support
        track = track.detach().clone()[None]
        # Get the batch (batch size of 1 in this code) instruction for the current ego vehicle's future trajectory
        direction_instruct, _ = self.direction_classifier.get_batch_instruct(ego_future=track, sampling_idx=sampling_idx)
        
        # Get acceleration instructions
        movement_instruct, acceleration_instruct = self.acceleration_classifier.get_batch_instruct(ego_future=track, sampling_idx=sampling_idx)

        # Get speed instructions
        speed_instruct, speeds_0to8 = self.speed_classifier.get_batch_instruct(ego_future=track, sampling_idx=sampling_idx)
        speed_0 = speeds_0to8[:,0]

        # for single instruction, needs to be modified to support batch data
        direction_instruct = int(direction_instruct[0][0])
        movement_instruct = int(movement_instruct[0][0])
        acceleration_instruct = int(acceleration_instruct[0][0])
        speed_instruct = int(speed_instruct[0][0])
        speed_0 = speed_0[0]

        if direction_instruct==0:
            movement_instruct, acceleration_instruct, speed_instruct = 0, 0, 0

        
        instructions_dict = {
            'direction':{
                'past_1s_cls': direction_instruct,
                'past_1s': self.direction_classifier.classes[direction_instruct] if direction_instruct!=-1 else 'INVALID',
                }, 
            'movement':{
                'past_1s_cls':movement_instruct,
                'past_1s':self.acceleration_classifier.movement_classes[movement_instruct] if movement_instruct!=-1 else 'INVALID',
                }, 
            'acceleration':{
                'past_1s_cls':acceleration_instruct,
                'past_1s':self.acceleration_classifier.acceleration_classes[acceleration_instruct] if acceleration_instruct!=-1 else 'INVALID',
                }, 
            'speed':{
                'past_1s_cls':speed_instruct,
                'past_1s':self.speed_classifier.classes[speed_instruct] if speed_instruct!=-1 else 'INVALID',
                'past_1s_speed':f"{int(speed_0*3.6)} km/h",
                }, 
            }
        
        instructions_dict = {f"{k} {k_}":v[k_] for k,v in instructions_dict.items() for k_ in instructions_dict[k].keys()}
        # for k,v in instructions_dict.items():
        #     print(f"{k}: {v}")
        return instructions_dict


    def get_heading_from_traj(self, step1xy, step2xy):
        x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
        dx, dy = x2 - x1, y2 - y1
        heading_angle = math.atan2(dy, dx)
        return heading_angle