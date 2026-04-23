# Title: direction instructions
# Description: Code to annotate given trajectories with direction instructions
# Original code used: https://github.com/waymo-research/waymo-open-dataset/issues/755

import math
import numpy as np
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GAMEFORMER_ROOT = REPO_ROOT / "gameformer"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GAMEFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(GAMEFORMER_ROOT))
from utils.data_utils import *

class DirectionClassifier:
    def __init__(
        self, 
        num_classes=8, 
        kMaxSpeedForStationary=2.0,  # m/s
        stepSpeedForStationary=0.5, # m/s
        kMaxDisplacementForStationary=5.0, # m
        
        kMaxLateralDisplacementForStraight=5.0, # m 
        kMinLongitudinalDisplacementForUTurn=-5.0, # m
        kMinLateralDisplacementForUTurn = 20, # m, updated as taking two lefts can be misclassified as left u-turn
        kMaxAbsHeadingDiffForStraight=math.pi / 6.0, # rad
        step_t=0.1, # s
        # delta_correction=[1.0, 1.02]
        delta_correction=[1.0, 1.0]
        ):

        self.num_classes = num_classes
        if num_classes == 8:
            # self.classes = ['STATIONARY', 'STRAIGHT', 'STRAIGHT_RIGHT', 'STRAIGHT_LEFT', 'RIGHT_TURN', 'LEFT_TURN', 'RIGHT_U_TURN', 'LEFT_U_TURN']
            self.classes = ['stationary', 'move straight', 'move straight veering right', 'move straight veering left', 'turn right', 'turn left', 'take right U-turn', 'take left U-turn']
            # self.classes = ['stationary', 'straight', 'straight veering right', 'straight veering left', 'turn right', 'turn left', 'take right U-turn', 'take left U-turn']
            self.llm_classes = ['stay stationary', 'move straight', 'move straight while veering to the right', 'move straight while veering to the left', 'turn right', 'turn left', 'take a right u-turn', 'take a left u-turn']
            self.merge_straight_classes = False
            self.use_right_uturn = True
            self.use_left_uturn = True
        else:
            # self.classes = ['STATIONARY', 'STRAIGHT', 'RIGHT', 'LEFT']
            self.classes = ['stationary', 'move straight', 'turn right', 'turn left', 'take left U-turn']
            self.llm_classes = ['stay stationary', 'move straight', 'turn right', 'turn left', 'take a left u-turn']
            self.merge_straight_classes = True
            self.use_right_uturn = False
            self.use_left_uturn = True

        # Constants for classification with default values as specified
        self.kMaxSpeedForStationary = kMaxSpeedForStationary
        self.kMaxDisplacementForStationary = kMaxDisplacementForStationary
        self.kMaxLateralDisplacementForStraight = kMaxLateralDisplacementForStraight
        self.kMinLongitudinalDisplacementForUTurn = kMinLongitudinalDisplacementForUTurn
        self.kMinLateralDisplacementForUTurn = kMinLateralDisplacementForUTurn
        self.kMaxAbsHeadingDiffForStraight = kMaxAbsHeadingDiffForStraight
        self.step_t = step_t
        self.delta_correction = delta_correction
        self.stepSpeedForStationary = stepSpeedForStationary

        
        
    @staticmethod
    def get_heading_from_traj(step1xy, step2xy):
        x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
        dx, dy = x2 - x1, y2 - y1
        heading_angle = math.atan2(dy, dx)
        return heading_angle

    @staticmethod
    def get_vel_from_traj(step1xy, step2xy, time_difference=0.1):
        x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocity = distance / time_difference
        return velocity

    @staticmethod
    def get_standard_angle(angle):
        while angle < 0 or angle >= 2 * math.pi:
            if angle < 0:
                angle += 2 * math.pi
            else:
                angle -= 2 * math.pi
        return angle

    @staticmethod
    def get_standard_heading_diff(angle):
        if abs(angle) > math.pi:
            if angle < 0:
                return 2 * math.pi - abs(angle)
            else:
                return -(2 * math.pi - abs(angle))
        else:
            return angle

    def get_batch_instruct(self, ego_future, sampling_idx=None, ignore_neighbor=True):
        # This method's implementation depends on specifics of ego_future structure and usage of torch
        num_classes = self.num_classes
        if not ignore_neighbor:
            raise 'Not implemented'
        device = ego_future.device
        # valid_mask = torch.ne(ego_future[:,:, :2], 0).bool().any(-1).any(-1)
        turn_cls = torch.zeros((ego_future.shape[0], 1), device=device)
        turn_cls_contrastive = torch.zeros((ego_future.shape[0], num_classes-1), device=device)
        ego_future_cpu = ego_future.detach().clone().cpu().numpy()
        for batch_i in range(ego_future.shape[0]):
            # if valid_mask[batch_i]:
            classify_track_out, turn_cls[batch_i], contrastive, contrastive_cls = self.classify_track(ego_future_cpu[batch_i], sampling_idx=sampling_idx)
            turn_cls_contrastive[batch_i] = torch.tensor(contrastive_cls, device=device)
            # else:
                # turn_cls[batch_i] = 0
                # turn_cls_contrastive[batch_i] = torch.ones_like(turn_cls_contrastive[batch_i])*-1
                # turn_cls_contrastive[batch_i] = torch.tensor(contrastive_cls, device=device) turn_cls_contrastive[batch_i])*-1
        return torch.cat((turn_cls, torch.ones_like(turn_cls)*-1), dim=-1), torch.cat((turn_cls_contrastive[...,None], torch.ones_like(turn_cls_contrastive[...,None])*-1), dim=-1)

    def get_sample_instruct(self, ego_future, sampling_idx=None):
        # Depends on ego_future structure; intended for numpy usage
        if sampling_idx is not None:
            raise 'not implemented'
        num_classes = self.num_classes
        valid_mask = np.any(ego_future[:,0] !=0, axis=-1)
        if valid_mask:
            return ClassifyTrack(ego_future.copy(), None, num_classes)
        else:
            return 'INVALID',-1, ['INVALID']*7, -1


    def classify_track(self, track, sampling_idx=None):
        track = track.copy()
        # direction classification logic involved in classifying tracks

        # time of each step
        time_step = np.array([i*self.step_t for i in range(1, len(track)+1)])
        
        # sampling the track and the time using the same sampling index, useful when downsampling from 10Hz to 1Hz
        if sampling_idx is not None:
            time_step = time_step[sampling_idx]
            track = track[sampling_idx]

        # Valid mask if the x value is not 0, note that this is only valid for future trajectories, 
        # where the last step of the observed trajectories when normalized could be [0,0], 
        # yet still it is a valid entry
        # valid_mask = track[1:,0] != 0
        valid_mask = np.hstack((True, track[1:,0] != 0)) # if first step is 0 its ok
        valid_states = track[valid_mask]
        time_step = time_step[valid_mask]

        


        # No enough valid entries -> unknown class
        if sum(valid_mask)<=1:
            return_class = "stationary"
            # return_class = "INVALID"
            # return 'INVALID', -1, ['INVALID']*(self.num_classes-1), [-1]*(self.num_classes-1)
            # return 'unknown', -1, ['unknown']*(self.num_classes-1), [-1]*(self.num_classes-1)
        else:
            # Less than 1 second window of valid entries -> unknown class
            valid_time_window = time_step[-1]-time_step[0]
            # if valid_time_window<=1:
            if valid_time_window<1:
                return 'INVALID', -1, ['INVALID']*(self.num_classes-1), [-1]*(self.num_classes-1)

            # # normalize the trajectory (center and heading)
            # center, angle = valid_states[0].copy()[:2], valid_states[0].copy()[2]
            # valid_states = agent_norm(valid_states, center, angle, impute=True)

            # Start and end state
            start_state = valid_states[0]
            end_state = valid_states[-1]

            # Compute deltas and displacements
            x_delta = end_state[0] - start_state[0]  # cx
            y_delta = end_state[1] - start_state[1]  # cy
            final_displacement = math.hypot(x_delta, y_delta)

            # First check if stationary
            speeds = [self.get_vel_from_traj(valid_states[ii], valid_states[ii+1], time_difference=time_step[ii+1] - time_step[ii]) for ii in range(len(valid_states)-1)]
            max_speed = max(speeds)
            if max_speed < self.kMaxSpeedForStationary and final_displacement < self.kMaxDisplacementForStationary:
                return_class = "stationary"
            else:
                # To get the correct heading, we need to measure it when the agent is moving only, stationary should be ignored
                moving_indx = [i for i in range(len(speeds)) if speeds[i]>self.stepSpeedForStationary]

                # Didn't work: If the agent moved only once, we can't know his relative heading
                # if len(moving_indx)==1:
                if len(moving_indx)==0:
                    return 'unknown', -1, ['unknown']*(self.num_classes-1), [-1]*(self.num_classes-1)
                else:
                    # the heading from state[0] to state[moving_indx[0]+1], where moving_indx[0] is the first source state at which the agent will move after
                    start_state_heading = self.get_standard_angle(self.get_heading_from_traj(valid_states[0], valid_states[moving_indx[0]+1]))
                    # similarly for end_state_heading, moving_indx[-1] have an upper cap of len(valid_states)-1
                    end_state_heading = self.get_standard_angle(self.get_heading_from_traj(valid_states[moving_indx[-1]], valid_states[-1]))
                    # if start and end movement speed needed 
                    # (not this does not mean the first and final observed speed, but the first and last "movement" observed speed)
                    # meaning the agent could be not moving at the start or the end and it is not measured by the following
                    start_speed, end_speed = speeds[moving_indx[0]], speeds[moving_indx[-1]]
                
                # normalize the angle
                heading_diff = self.get_standard_angle(end_state_heading) - self.get_standard_angle(start_state_heading)  # heading
                heading_diff = self.get_standard_heading_diff(heading_diff)

                dx, dy = x_delta*self.delta_correction[0], y_delta*self.delta_correction[1]
                
                if abs(heading_diff) < self.kMaxAbsHeadingDiffForStraight:
                    if abs(dy) < self.kMaxLateralDisplacementForStraight or self.merge_straight_classes:
                        return_class = "move straight"
                    else:
                        return_class = "move straight veering right" if dy < 0 else "move straight veering left"
                elif dy < 0:
                    if self.use_right_uturn:
                        return_class = "take right U-turn" if (dx < self.kMinLongitudinalDisplacementForUTurn and abs(dy) < self.kMinLateralDisplacementForUTurn) else "turn right"
                    else:
                        return_class = "turn right"
                elif dx < self.kMinLongitudinalDisplacementForUTurn and self.use_left_uturn and abs(dy) < self.kMinLateralDisplacementForUTurn:
                    return_class = "take left U-turn"
                else:
                    return_class = "turn left"

        return_class_idx = [i for i, class_i in enumerate(self.classes) if class_i==return_class][0]
        ground_truth_classes = self.llm_classes[return_class_idx]
        contrastive_classes_idx = [i for i, class_i in enumerate(self.classes) if class_i!=return_class]
        contrastive_classes = [self.llm_classes[i] for i in contrastive_classes_idx]

        return ground_truth_classes, return_class_idx, contrastive_classes, contrastive_classes_idx


    def classify_lane(self, track):
        track = track.copy()
        # direction classification logic involved in classifying tracks

    
        # Start and end state
        start_state = track[0]
        end_state = track[-1]

        # Compute deltas and displacements
        x_delta = end_state[0] - start_state[0]  # cx
        y_delta = end_state[1] - start_state[1]  # cy
        final_displacement = math.hypot(x_delta, y_delta)

        
        if final_displacement < self.kMaxDisplacementForStationary:
            return_class = "stationary"
        else:

            # the heading from state[0] to state[moving_indx[0]+1], where moving_indx[0] is the first source state at which the agent will move after
            start_state_heading = start_state[2]
            # similarly for end_state_heading, moving_indx[-1] have an upper cap of len(valid_states)-1
            end_state_heading = end_state[2]
            # if start and end movement speed needed 
            
            # normalize the angle
            heading_diff = self.get_standard_angle(end_state_heading) - self.get_standard_angle(start_state_heading)  # heading
            heading_diff = self.get_standard_heading_diff(heading_diff)

            dx, dy = x_delta*self.delta_correction[0], y_delta*self.delta_correction[1]
            
            if abs(heading_diff) < self.kMaxAbsHeadingDiffForStraight:
                if abs(dy) < self.kMaxLateralDisplacementForStraight or self.merge_straight_classes:
                    return_class = "move straight"
                else:
                    return_class = "move straight veering right" if dy < 0 else "move straight veering left"
            elif dy < 0:
                if self.use_right_uturn:
                    return_class = "take right U-turn" if (dx < self.kMinLongitudinalDisplacementForUTurn and abs(dy) < self.kMinLateralDisplacementForUTurn) else "turn right"
                else:
                    return_class = "turn right"
            elif dx < self.kMinLongitudinalDisplacementForUTurn and self.use_left_uturn and abs(dy) < self.kMinLateralDisplacementForUTurn:
                return_class = "take left U-turn"
            else:
                return_class = "turn left"

        return_class_idx = [i for i, class_i in enumerate(self.classes) if class_i==return_class][0]
        ground_truth_classes = self.llm_classes[return_class_idx]
        contrastive_classes_idx = [i for i, class_i in enumerate(self.classes) if class_i!=return_class]
        contrastive_classes = [self.llm_classes[i] for i in contrastive_classes_idx]

        return ground_truth_classes, return_class_idx, contrastive_classes, contrastive_classes_idx
