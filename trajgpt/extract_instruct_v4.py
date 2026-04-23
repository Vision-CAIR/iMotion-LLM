## https://github.com/waymo-research/waymo-open-dataset/issues/755
import math
import torch
import numpy as np
import statistics

def get_heading_from_traj(step1xy, step2xy):
    x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
    dx, dy = x2-x1, y2-y1
    heading_angle = math.atan2(dy, dx)
    return heading_angle

def get_vel_from_traj(step1xy, step2xy, time_difference=0.1):
    x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    velocity = distance / time_difference
    return velocity

def get_batch_instruct(ego_future, num_classes=8, sampling_idx=None):
    device = ego_future.device
    # acts = torch.zeros((ego_future.shape[0], 1))
    valid_mask = torch.ne(ego_future[:,:, :2], 0).bool().any(-1).any(-1)
    turn_cls = torch.zeros((ego_future.shape[0], 1), device=device)
    turn_cls_contrastive = torch.zeros((ego_future.shape[0], num_classes-1), device=device)
    ego_future_cpu = ego_future.detach().clone().cpu().numpy()
    for batch_i in range(ego_future.shape[0]):
        if valid_mask[batch_i]:
            classify_track_out, turn_cls[batch_i], contrastive, contrastive_cls = ClassifyTrack(ego_future_cpu[batch_i], None, num_classes, sampling_idx=sampling_idx)
            turn_cls_contrastive[batch_i] = torch.tensor(contrastive_cls, device=device)
        else:
            turn_cls[batch_i] = -1
            turn_cls_contrastive[batch_i] = torch.ones_like(turn_cls_contrastive[batch_i])*-1
    return torch.cat((turn_cls, torch.ones_like(turn_cls)*-1), dim=-1), torch.cat((turn_cls_contrastive[...,None], torch.ones_like(turn_cls_contrastive[...,None])*-1), dim=-1)


def get_sample_instruct(ego_future, num_classes=8):
    valid_mask = np.any(ego_future[:,0] !=0, axis=-1)
    if valid_mask:
        return ClassifyTrack(ego_future.copy(), None, num_classes)[0]
    else:
        return 'unknown'

def get_sample_instruct_c(ego_future, num_classes=8):
    valid_mask = np.any(ego_future[:,0] !=0, axis=-1)
    if valid_mask:
        return ClassifyTrack(ego_future.copy(), None, num_classes)
    else:
        return 'unknown',-1, ['unknown']*7, -1

def get_sample_instruct_torch(ego_future, ego_state, num_classes=8):
    valid_mask = torch.ne(ego_future[:,:, :2], 0).bool().any(-1).any(-1)
    turn_cls = torch.zeros((ego_future.shape[0], 1))
    ego_future_cpu = ego_future.detach().clone().numpy()
    for batch_i in range(ego_future.shape[0]):
        if valid_mask[batch_i]:
            classify_track_out, turn_cls[batch_i] = ClassifyTrack(ego_future_cpu[batch_i], None, num_classes)
        else:
            turn_cls[batch_i] = -1
    return torch.cat((turn_cls, torch.ones_like(turn_cls)*-1), dim=-1)

def GetStandardAngle(angle):
    while angle < 0 or angle >= 2 * math.pi:
        if angle < 0:
            angle += 2 * math.pi
        else:
            angle -= 2 * math.pi
    return angle

def GetStandardHeadingDiff(angle):
    if abs(angle) > math.pi:
        # return -(2 * math.pi - abs(angle))
        if angle < 0:
            return 2 * math.pi - abs(angle)
        else:
            return -(2 * math.pi - abs(angle))
    else:
        return angle

def ClassifyTrack(track, valid_state_bit, num_classes=8, sampling_idx=None):
    classes = ['STATIONARY', 'STRAIGHT', 'STRAIGHT_RIGHT', 'STRAIGHT_LEFT', 'RIGHT_TURN', 'LEFT_TURN', 'RIGHT_U_TURN', 'LEFT_U_TURN']
    llm_classes = ['stay stationary', 'move straight', 'move straight while veering to the right', 'move straight while veering to the left', 'turn right', 'turn left', 'take a right u-turn', 'take a left u-turn']
    classes_4 = ['STATIONARY', 'STRAIGHT', 'RIGHT', 'LEFT']
    llm_classes_4 = ['stay stationary', 'move straight', 'turn right', 'turn left']
    # Constants for classification
    kMaxSpeedForStationary = 2.0  # m/s
    kMaxDisplacementForStationary = 5.0  # m
    kMaxLateralDisplacementForStraight = 5.0  # m
    kMinLongitudinalDisplacementForUTurn = -5.0  # m
    kMaxAbsHeadingDiffForStraight = math.pi / 6.0  # rad

    # Extract valid states
    # valid_states = [state for state in track if state[9]]  # state[9] is the valid bit # use this with original Waymo format
    # valid_states = [state for state_i, state in enumerate(track) if valid_state_bit[state_i]]  # state[9] is the valid bit
    time_step = np.array([i*0.1 for i in range(1, len(track)+1)])

    if sampling_idx is not None:
        time_step = time_step[sampling_idx]
        track = track[sampling_idx]
    
    valid_mask = track[:,0] != 0
    valid_states = track[valid_mask]
    time_step = time_step[valid_mask]
    if sum(valid_mask)<=1:
        return 'unknown', -1, ['unknown']*(num_classes-1), [-1]*(num_classes-1)
    valid_time_window = time_step[-1]-time_step[0]
    if valid_time_window<=1:
        return 'unknown', -1, ['unknown']*(num_classes-1), [-1]*(num_classes-1)

    # if not valid_states:
    #     return "Invalid"

    # Start and end state
    start_state = valid_states[0]
    end_state = valid_states[-1]

    # Compute deltas and displacements
    x_delta = end_state[0] - start_state[0]  # cx
    y_delta = end_state[1] - start_state[1]  # cy
    final_displacement = math.hypot(x_delta, y_delta)
    # heading_diff = GetStandardAngle(end_state[6]) - GetStandardAngle(start_state[6])  # heading # use this with original Waymo format
    if len(start_state)>2:
        heading_diff = GetStandardAngle(end_state[2]) - GetStandardAngle(start_state[2])  # heading
        heading_diff = GetStandardHeadingDiff(heading_diff)
        # print(f"heading_diff: {heading_diff}")
        # Normalized deltas
        # dx = x_delta * math.cos(-start_state[6]) - y_delta * math.sin(-start_state[6]) # use this with original Waymo format
        # dy = x_delta * math.sin(-start_state[6]) + y_delta * math.cos(-start_state[6]) # use this with original Waymo format
        dx = x_delta * math.cos(-start_state[2]) - y_delta * math.sin(-start_state[2])
        dy = x_delta * math.sin(-start_state[2]) + y_delta * math.cos(-start_state[2])
        # Speed calculations
        # start_speed = math.hypot(start_state[7], start_state[8])  # vel_x, vel_y # use this with original Waymo format
        # end_speed = math.hypot(end_state[7], end_state[8])  # vel_x, vel_y # use this with original Waymo format
        start_speed = math.hypot(start_state[3], start_state[4])  # vel_x, vel_y
        end_speed = math.hypot(end_state[3], end_state[4])  # vel_x, vel_y
        max_speed = max(start_speed, end_speed)
        # print(f"max_speed: {max_speed}")
    else: ##### !!!! Trajectory should be rotated to a normalized view, aligned on x,y axis !!!! #####
        # start_state_heading = get_heading_from_traj(valid_states[0], valid_states[1])
        # end_state_heading = get_heading_from_traj(valid_states[0], valid_states[-2])
        # heading_diff = GetStandardAngle(end_state_heading) - GetStandardAngle(start_state_heading)  # heading
        ii=0
        # while ii<len(valid_states)-2 and get_vel_from_traj(valid_states[ii], valid_states[ii+1], time_difference=time_step[ii+1] - time_step[ii]) < kMaxSpeedForStationary:
        while ii<len(valid_states)-2 and get_vel_from_traj(valid_states[ii], valid_states[ii+1], time_difference=time_step[ii+1] - time_step[ii]) < kMaxSpeedForStationary:
            ii+=1
        start_state_heading = GetStandardAngle(get_heading_from_traj(valid_states[0], valid_states[ii+1]))
        start_speed = get_vel_from_traj(valid_states[0], valid_states[ii+1], time_difference=time_step[ii+1] - time_step[0])
        # start_state_heading = GetStandardAngle(get_heading_from_traj(valid_states[ii], valid_states[ii+1]))
        # start_speed = get_vel_from_traj(valid_states[ii], valid_states[ii+1], time_difference=time_step[ii+1] - time_step[ii])
        ii=1
        # while ii<len(valid_states)-2 and get_vel_from_traj(valid_states[-ii-1], valid_states[-ii], time_difference=time_step[-ii] - time_step[-ii-1]) < kMaxSpeedForStationary:
        while ii<len(valid_states)-2 and get_vel_from_traj(valid_states[-ii-1], valid_states[-ii], time_difference=time_step[-ii] - time_step[-ii-1]) < kMaxSpeedForStationary:
            # print(GetStandardAngle(get_heading_from_traj(valid_states[ii-1], valid_states[ii])))
            ii+= 1
        # end_state_heading = GetStandardAngle(get_heading_from_traj(valid_states[-1], valid_states[-ii]))
        # end_speed = get_vel_from_traj(valid_states[-ii], valid_states[-ii-1], time_difference=time_step[-ii] - time_step[-ii-1])
        end_state_heading = GetStandardAngle(get_heading_from_traj(valid_states[-ii-1], valid_states[-ii]))
        end_speed = get_vel_from_traj(valid_states[-ii-1], valid_states[-ii], time_difference=time_step[-ii] - time_step[-ii-1])
        # end_speed = get_vel_from_traj(valid_states[-2], valid_states[-1], time_difference=time_step[-2] - time_step[-1])

        # start_state_heading = statistics.median([GetStandardAngle(get_heading_from_traj(valid_states[i], valid_states[i+1])) for i in range(1)])
        # start_state_heading = statistics.median([GetStandardAngle(get_heading_from_traj(valid_states[i], valid_states[i+5])) for i in [0,5]]) # avg first 3 steps
        # end_state_heading = statistics.median([GetStandardAngle(get_heading_from_traj(valid_states[i-10], valid_states[i])) for i in [-1,-10, -20]])
        # end_state_heading = statistics.median([GetStandardAngle(get_heading_from_traj(valid_states[-10+i], valid_states[-10+i+1])) for i in range(1,9)])
        # end_state_heading = sum([GetStandardAngle(get_heading_from_traj(valid_states[i-10], valid_states[i])) for i in [-1,-10]])/2
        heading_diff = GetStandardAngle(end_state_heading) - GetStandardAngle(start_state_heading)  # heading
        heading_diff = GetStandardHeadingDiff(heading_diff)
        dx = x_delta
        dy = y_delta*1.02
        # dx = x_delta * math.cos(-start_state_heading) - y_delta * math.sin(-start_state_heading)
        # dy = x_delta * math.sin(-start_state_heading) + y_delta * math.cos(-start_state_heading)
        
        
        max_speed = max(start_speed, end_speed)

    # Trajectory type classification
    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return_class = "STATIONARY"
    elif abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if abs(dy) < kMaxLateralDisplacementForStraight:
            return_class = "STRAIGHT"
        else:
            return_class = "STRAIGHT_RIGHT" if dy < 0 else "STRAIGHT_LEFT"
    # elif heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
    elif dy < 0:
        return_class = "RIGHT_U_TURN" if dx < kMinLongitudinalDisplacementForUTurn else "RIGHT_TURN"
    elif dx < kMinLongitudinalDisplacementForUTurn:
        return_class = "LEFT_U_TURN"
    else:
        return_class = "LEFT_TURN"
    
    if num_classes==8:
        # return return_class, [i for i, class_i in enumerate(classes) if class_i==return_class][0]
        return_class_idx = [i for i, class_i in enumerate(classes) if class_i==return_class][0]
        contrastive_classes_idx = [i for i, class_i in enumerate(classes) if class_i!=return_class]
        contrastive_classes = [llm_classes[i] for i in contrastive_classes_idx]
        return llm_classes[return_class_idx], return_class_idx, contrastive_classes, contrastive_classes_idx
    else:
        if 'STRAIGHT' in return_class:
            return_class = 'STRAIGHT'
        elif 'LEFT' in return_class:
            return_class = 'LEFT'
        elif 'RIGHT' in return_class:
            return_class = 'RIGHT'
        else:
            return_class = 'STATIONARY' 
        # return return_class, [i for i, class_i in enumerate(classes_4) if class_i==return_class][0]
        return_class_idx = [i for i, class_i in enumerate(classes_4) if class_i==return_class][0]
        raise 'not implemented'
        return llm_classes_4[return_class_idx], return_class_idx
        
    # ['STATIONARY', 'STRAIGHT', 'RIGHT', 'LEFT']
    # ['STATIONARY', 'STRAIGHT', 'STRAIGHT_RIGHT', 'STRAIGHT_LEFT', 'RIGHT_TURN', 'LEFT_TURN', 'RIGHT_U_TURN', 'LEFT_U_TURN']