# Title: acceleration instructions

import math
import numpy as np
import torch

class SpeedClassifier:
    def __init__(
        self, 
        kMaxSpeedForStationary=2.0,  # m/s
        stepSpeedForStationary=0.5, # m/s
        kMaxDisplacementForStationary=5.0, # m
        very_slow = 5.56, 
        slow = 11.11, # m/s
        moderate = 25.0, # m/s
        fast = 30.56,
        very_fast = 38.89,
        extremely_fast = 70.0,
        step_t=0.1, # s
        ):
        # 1 m/s -> 3.6 km/h
        # not moving cap:  2 m/s -> 7.2 km/h & final_displacement < 5m
        # very slow cap: 20 km/h -> 5.56 m/s
        # slow cap: 40 km/h -> 11.11 m/s
        # medium: 90 km/h ->  25 m/s
        # fast: 120 km/h -> 30.56 m/s
        # very fast: 140 km/h -> 38.89 m/s
        # extremely fast: 252 km/h -> 70 m/s
        # hypersonic: > 252 km/h
        # self.classes = ['STATIONARY', 'VERY_SLOW', 'SLOW', 'MODERATE', 'FAST', 'VERY_FAST', 'INVALID']
        self.classes = ['stationary', 'very slow', 'slow', 'moderate', 'fast', 'very fast', 'INVALID']
        self.num_classes = len(self.classes)
        # Constants for classification with default values as specified
        self.kMaxSpeedForStationary = kMaxSpeedForStationary
        self.kMaxDisplacementForStationary = kMaxDisplacementForStationary
        self.step_t = step_t
        self.stepSpeedForStationary = stepSpeedForStationary

        # velocity constants
        self.very_slow = very_slow
        self.slow = slow
        self.moderate = moderate
        self.fast = fast
        self.very_fast = very_fast
        self.extremely_fast = extremely_fast

    @staticmethod
    def get_vel_from_traj(step1xy, step2xy, time_difference=0.1):
        x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocity = distance / time_difference
        return velocity


    def get_batch_instruct(self, ego_future, sampling_idx=None, ignore_neighbor=True):
        # This method's implementation depends on specifics of ego_future structure and usage of torch
        # num_classes = self.num_classes
        if not ignore_neighbor:
            raise 'Not implemented'
        device = ego_future.device
        valid_mask = torch.ne(ego_future[:,:, :2], 0).bool().any(-1).any(-1)
        cls_idx = torch.zeros((ego_future.shape[0], 1), device=device)
        speeds = torch.zeros((ego_future.shape[0], 2), device=device)
        ego_future_cpu = ego_future.detach().clone().cpu().numpy()
        for batch_i in range(ego_future.shape[0]):
            # note that idx only stored, the string class is ignored but can be retrieved using the class arguments
            speed_cls, cls_idx[batch_i], speeds_ = self.classify_track(ego_future_cpu[batch_i], sampling_idx=sampling_idx)
            speeds[batch_i] = torch.tensor(speeds_, device=device)
        return cls_idx, speeds


    def classify_track(self, track, sampling_idx=None):
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
        valid_mask = np.hstack((True, track[1:,0] != 0))
        valid_states = track[valid_mask]
        time_step = time_step[valid_mask]

        # No enough valid entries -> unknown class
        # if sum(valid_mask) == 0:
        #     speed_cls = 'INVALID'
        #     speed_cls = 'INVALID'
        if sum(valid_mask)<=1:
            speed_cls = 'INVALID'
            speed_cls = 'INVALID'
        else:
            # Less than 1 second window of valid entries -> unknown class
            valid_time_window = time_step[-1]-time_step[0]
            # if valid_time_window<=1:
            if valid_time_window<1:
                # return 'unknown', -1, ['unknown']*(self.num_classes-1), [-1]*(self.num_classes-1)
                speed_cls = 'stationary'
                speed_cls = 'stationary'
                # return 'UNKNOWN', -1, 'UNKNOWN', -1

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
                speed_cls = "stationary"
            else:
                max_speed = max(speeds)
                if max_speed < self.very_slow:
                    speed_cls = "very slow"
                elif max_speed < self.slow:
                    speed_cls = "slow"
                elif max_speed < self.moderate:
                    speed_cls = "moderate"
                elif max_speed < self.fast:
                    speed_cls = "fast"
                elif max_speed < self.very_fast:
                    speed_cls = "very fast"
                elif max_speed < self.extremely_fast:
                    speed_cls = "INVALID" #"EXTREMELY_FAST"
                else:
                    speed_cls = "INVALID"

            #     initial_speed = speeds[0] # m/s
            #     final_speed = speeds[-1] # m/s
            # if speed_cls == "STATIONARY" or speed_cls = "INVALID":
        if speed_cls == "INVALID":
            initial_speed = 0.0
            final_speed = 0.0
        else:
            initial_speed = speeds[0] # m/s
            final_speed = speeds[-1] # m/s
        cls_idx = [i for i, class_i in enumerate(self.classes) if class_i==speed_cls][0]

        return speed_cls, cls_idx, [initial_speed, final_speed]