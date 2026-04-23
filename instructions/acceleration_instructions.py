# Title: acceleration instructions

import math
import numpy as np
import torch

class AccelerationClassifier:
    def __init__(
        self, 
        kMaxSpeedForStationary=2.0,  # m/s
        stepSpeedForStationary=0.5, # m/s
        kMaxDisplacementForStationary=5.0, # m
        constant_vel_threshold = 0.2, # m/s^2, cap for constant velocity
        accel_threshold = 0.87, # m/s^2, cap for mild acceleration/deceleration
        mild_accel_threshold = 1.6, # m/s^2, cap for abrupt acceleration/deceleration
        aggressive_accel_threshold = 2.26, # m/s^2, cap for aggressive acceleration/deceleration
        step_t=0.1, # s
        ):
        # 1 m/s^2 = 3.6 km/h/s, i.e. the km/h achived in 1 second, 
        # fasy cars usually have a maximum accelearion below 100 km/h/8s, 
        # meaninng reaching 100 km/h in 8 seconds
        # 0.2 m/s^2 -> 5.76 km/h/8s
        # 0.87 m/s^2 -> 25.056 km/h/8s
        # 1.05 m/s^2 -> 30.24 km/h/8s
        # 1.4 m/s^2 -> 40.32 km/h/8s
        # 1.5 m/s^2 -> 43.2 km/h/8s
        # 1.6 m/s^2 -> 46.08 km/h/8s
        # 1.39 m/s^2 -> 40.032 km/h/8s
        # 1.57 m/s^2 -> 45.216 km/h/8s
        # 2.1 m/s^2 -> 60.48 km/h/8s
        # 2.26 m/s^2 -> 65.09 km/h/8s
        # 2.4 m/s^2 -> 69.12 km/h/8s
        # 3 m/s^2 -> 86.4 km/h/8s

        # self.movement_classes = ['STATIONARY', 'MOVING', 'STARTING_TO_MOVE', 'STOPPING']
        self.movement_classes = ['stationary', 'moving', 'starting to move', 'stopping']
        # self.acceleration_classes = ['STATIONARY', 'CONSTANT_VELOCITY', 'MILD_ACCELERATION', 'MILD_DECELERATION', 'MODERATE_ACCELERATION', 'MODERATE_DECELERATION', 'AGGRESSIVE_ACCELERATION', 'AGGRESSIVE_DECELERATION', 'EXTREME_ACCELERATION', 'EXTREME_DECELERATION']
        self.acceleration_classes = ['stationary', 'constant velocity', 'mild acceleration', 'mild deceleration', 'moderate acceleration', 'moderate deceleration', 'aggressive acceleration', 'aggressive deceleration', 'extreme acceleration', 'extreme deceleration']
        self.num_movement_classes = len(self.movement_classes)
        self.num_acceleration_classes = len(self.acceleration_classes)
        # Constants for classification with default values as specified
        self.kMaxSpeedForStationary = kMaxSpeedForStationary
        self.kMaxDisplacementForStationary = kMaxDisplacementForStationary
        self.step_t = step_t
        self.stepSpeedForStationary = stepSpeedForStationary

        # acceleration constants
        self.constant_vel_threshold = constant_vel_threshold
        self.accel_threshold = accel_threshold
        self.mild_accel_threshold = mild_accel_threshold
        self.aggressive_accel_threshold = aggressive_accel_threshold

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
        # valid_mask = torch.ne(ego_future[:,:, :2], 0).bool().any(-1).any(-1)
        movement_classes = torch.zeros((ego_future.shape[0], 1), device=device)
        accel_classes = torch.zeros((ego_future.shape[0], 1), device=device)
        ego_future_cpu = ego_future.detach().clone().cpu().numpy()
        for batch_i in range(ego_future.shape[0]):
            # note that idx only stored, the string class is ignored but can be retrieved using the class arguments
            movement_cls, movement_classes[batch_i], accel_cls, accel_classes[batch_i] = self.classify_track(ego_future_cpu[batch_i], sampling_idx=sampling_idx)
        return movement_classes, accel_classes


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
        if sum(valid_mask)<=1:
            movement_class = 'INVALID'
            accel_class = 'INVALID'
        else:
            # Less than 1 second window of valid entries -> unknown class
            valid_time_window = time_step[-1]-time_step[0]
            # if valid_time_window<=1:
            if valid_time_window<1:
                # return 'unknown', -1, ['unknown']*(self.num_classes-1), [-1]*(self.num_classes-1)
                movement_class = 'stationary'
                accel_class = 'stationary'
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
                movement_class = "stationary"
                accel_class = "stationary"
            else:
                velocity_magnitudes = speeds
                velocity_changes = [velocity_magnitudes[i+1]-velocity_magnitudes[i] for i in range(len(velocity_magnitudes)-1)]
                average_change = np.mean(velocity_changes)

                if abs(average_change) < self.constant_vel_threshold:
                    accel_class = "constant velocity"
                elif abs(average_change) < self.accel_threshold:
                    accel_class = "mild acceleration" if average_change>0 else "mild deceleration"
                elif abs(average_change) < self.mild_accel_threshold:
                    accel_class = "moderate acceleration" if average_change>0 else "moderate deceleration"
                elif abs(average_change) < self.aggressive_accel_threshold:
                    accel_class = "aggressive acceleration" if average_change>0 else "aggressive deceleration"
                else:
                    accel_class = "extreme acceleration" if average_change>0 else "extreme deceleration"

                if 'deceleration' in accel_class and velocity_magnitudes[-1] < self.kMaxSpeedForStationary:
                    movement_class = 'stopping' 
                elif 'acceleration' in accel_class and velocity_magnitudes[-1] < self.kMaxSpeedForStationary:
                    movement_class = 'starting to move' 
                else:
                    movement_class = 'moving' 
        if movement_class == 'INVALID':
            movement_class_idx = -1
            accel_class_idx = -1
        else:
            movement_class_idx = [i for i, class_i in enumerate(self.movement_classes) if class_i==movement_class][0]
            accel_class_idx = [i for i, class_i in enumerate(self.acceleration_classes) if class_i==accel_class][0]

        return movement_class, movement_class_idx, accel_class, accel_class_idx