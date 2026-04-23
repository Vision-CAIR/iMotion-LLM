# Title: acceleration instructions

import math
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter

class ShapeClassifier:
    def __init__(
        self, 
        kMaxSpeedForStationary=2.0,  # m/s
        stepSpeedForStationary=0.5, # m/s
        kMaxDisplacementForStationary=5.0, # m
        curvature_threshold = 0.01,
        step_t=0.1, # s
        kMaxLateralDisplacementForStraight=5.0, # m 
        kMaxAbsHeadingDiffForStraight=math.pi / 6.0, # rad
        ):
        self.kMaxLateralDisplacementForStraight = kMaxLateralDisplacementForStraight
        self.kMaxAbsHeadingDiffForStraight = kMaxAbsHeadingDiffForStraight

        # 1 m/s -> 3.6 km/h
        # not moving cap:  2 m/s -> 7.2 km/h & final_displacement < 5m
        # very slow cap: 20 km/h -> 5.56 m/s
        # slow cap: 40 km/h -> 11.11 m/s
        # medium: 90 km/h ->  25 m/s
        # fast: 120 km/h -> 30.56 m/s
        # very fast: 140 km/h -> 38.89 m/s
        # extremely fast: 252 km/h -> 70 m/s
        # hypersonic: > 252 km/h
        self.classes = ['STATIONARY', 'LINE', 'SINGLE_CURVE', 'DOUBLE_CURVE', 'TRIPLE_CURVE', 'HIGHER_DEGREE_CURVE']
        self.num_classes = len(self.classes)
        # Constants for classification with default values as specified
        self.kMaxSpeedForStationary = kMaxSpeedForStationary
        self.kMaxDisplacementForStationary = kMaxDisplacementForStationary
        self.step_t = step_t
        self.stepSpeedForStationary = stepSpeedForStationary

        # Define a curvature threshold
        self.curvature_threshold = curvature_threshold  # Adjust this value based on your specific criteria

    def get_batch_instruct(self, ego_future, sampling_idx=None, ignore_neighbor=True):
        # This method's implementation depends on specifics of ego_future structure and usage of torch
        # num_classes = self.num_classes
        if not ignore_neighbor:
            raise 'Not implemented'
        device = ego_future.device
        valid_mask = torch.ne(ego_future[:,:, :2], 0).bool().any(-1).any(-1)
        cls_idx = torch.zeros((ego_future.shape[0], 1), device=device)
        ego_future_cpu = ego_future.detach().clone().cpu().numpy()
        for batch_i in range(ego_future.shape[0]):
            # note that idx only stored, the string class is ignored but can be retrieved using the class arguments
            str_cls, cls_idx[batch_i] = self.classify_track(ego_future_cpu[batch_i], sampling_idx=sampling_idx)
        return cls_idx

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
    
    @staticmethod
    def check_collinearity(points, tolerance=0.1):
        # Check if all points are on the same line with a tolerance for minor deviations
        for i in range(len(points)-2):
            p1, p2, p3 = points[i], points[i+1], points[i+2]
            area = 0.5 * np.abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            if area > tolerance:
                return False
        return True

    @staticmethod
    def count_curves(points, angle_threshold_deg=25):
        vectors = np.diff(points, axis=0)
        angles = []
        for i in range(len(vectors)-1):
            v1, v2 = vectors[i], vectors[i+1]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            theta = np.arccos(np.clip(cos_theta, -1, 1))
            angles.append(theta)
        # Count significant changes in direction with a higher angle threshold
        angle_threshold_rad = np.radians(angle_threshold_deg)
        return sum(1 for x in angles if x > angle_threshold_rad)


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
        valid_mask = track[:,0] != 0
        valid_states = track[valid_mask]
        time_step = time_step[valid_mask]

        # No enough valid entries -> unknown class
        # if sum(valid_mask) == 0:
        #     speed_cls = 'INVALID'
        #     speed_cls = 'INVALID'
        if sum(valid_mask)<=1:
            return_cls = 'INVALID'
        else:
            # Less than 1 second window of valid entries -> unknown class
            valid_time_window = time_step[-1]-time_step[0]
            # if valid_time_window<=1:
            if valid_time_window<1:
                # return 'unknown', -1, ['unknown']*(self.num_classes-1), [-1]*(self.num_classes-1)
                return_cls = 'STATIONARY'
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
                return_cls = "STATIONARY"
            else:
                # To get the correct heading, we need to measure it when the agent is moving only, stationary should be ignored
                moving_indx = [i for i in range(len(speeds)) if speeds[i]>self.stepSpeedForStationary]
                # Didn't work: If the agent moved only once, we can't know his relative heading
                # if len(moving_indx)==1:
                if len(moving_indx)==0:
                    return 'unknown', -1, ['unknown']*(self.num_classes-1), [-1]*(self.num_classes-1)
                else:
                    start_state_heading = self.get_standard_angle(self.get_heading_from_traj(valid_states[0], valid_states[moving_indx[0]+1]))
                    end_state_heading = self.get_standard_angle(self.get_heading_from_traj(valid_states[moving_indx[-1]], valid_states[-1]))
                    # normalize the angle
                    heading_diff = self.get_standard_angle(end_state_heading) - self.get_standard_angle(start_state_heading)  # heading
                    heading_diff = self.get_standard_heading_diff(heading_diff)
                    dx, dy = x_delta, y_delta
                    if abs(heading_diff) < self.kMaxAbsHeadingDiffForStraight:
                        cls_idx = 1 # LINE
                        return_cls = self.classes[cls_idx] 
                    else:
                        cls_idx = 2 # Curve
                        return_cls = self.classes[cls_idx] 

                # points = track[:,:2]
                # is_line = self.check_collinearity(points)
                # if is_line:
                #     cls_idx = 1
                #     return_cls = self.classes[cls_idx]
                # else:
                #     num_curves = self.count_curves(points)
                #     cls_idx = num_curves+1
                #     return_cls = self.classes[cls_idx]
                # print(return_cls)
                # print('***'*10)
            if False:
                x = track[:,0]
                y = track[:,1]
                # Applying Gaussian smoothing
                smoothed_x = gaussian_filter(x, sigma=2)
                x = smoothed_x
                smoothed_y = gaussian_filter(y, sigma=2)
                y = smoothed_y
                # First derivatives
                dydx = np.gradient(y, x)  # Change in y with respect to x
                dxdt = np.gradient(x)     # Change in x with respect to an arbitrary parameter t
                dydt = np.gradient(y)     # Change in y with respect to t

                # Second derivatives
                d2ydx2 = np.gradient(dydx, x)  # Second derivative of y with respect to x
                d2xdt2 = np.gradient(dxdt)     # Second derivative of x with respect to t
                d2ydt2 = np.gradient(dydt)     # Second derivative of y with respect to t

                # Curvature calculation (estimated curvature formula)
                curvature = np.abs(dxdt * d2ydt2 - dydt * d2xdt2) / np.power(dxdt**2 + dydt**2, 1.5)

                # Identify where curvature is above the threshold (significant curvature)
                significant_curvature_indices = np.where(curvature > self.curvature_threshold)[0]

                if len(significant_curvature_indices)==0:
                    cls_idx = 1
                    return_cls = self.classes[cls_idx]
                else:
                    # Calculate sign changes only where curvature is significant
                    sign_changes = np.where(np.diff(np.sign(curvature[significant_curvature_indices])))[0]
                    num_inflection_points = len(sign_changes)
                    if num_inflection_points>3:
                        num_inflection_points = self.num_classes-1
                    else:
                        num_inflection_points += 1
                        cls_idx = num_inflection_points+1
                        return_cls = self.classes[cls_idx]
                    
        
        if return_cls == "STATIONARY":
            cls_idx = 0
        if return_cls == "INVALID":
            cls_idx = -1
        return return_cls, cls_idx

        #         max_speed = max(speeds)
        #         if max_speed < self.very_slow:
        #             speed_cls = "VERY_SLOW"
        #         elif max_speed < self.slow:
        #             speed_cls = "SLOW"
        #         elif max_speed < self.moderate:
        #             speed_cls = "MODERATE"
        #         elif max_speed < self.fast:
        #             speed_cls = "FAST"
        #         elif max_speed < self.very_fast:
        #             speed_cls = "VERY_FAST"
        #         elif max_speed < self.extremely_fast:
        #             speed_cls = "INVALID" #"EXTREMELY_FAST"
        #         else:
        #             speed_cls = "INVALID"

        #     #     initial_speed = speeds[0] # m/s
        #     #     final_speed = speeds[-1] # m/s
        #     # if speed_cls == "STATIONARY" or speed_cls = "INVALID":
        # if speed_cls == "INVALID":
        #     initial_speed = 0.0
        #     final_speed = 0.0
        # else:
        #     initial_speed = speeds[0] # m/s
        #     final_speed = speeds[-1] # m/s
        # cls_idx = [i for i, class_i in enumerate(self.classes) if class_i==speed_cls][0]

        # return speed_cls, cls_idx, [initial_speed, final_speed]










