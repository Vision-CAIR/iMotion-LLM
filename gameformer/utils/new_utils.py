import numpy as np
import torch
from utils.data_utils import *


def interpolate_missing_traj(traj):
    missing_indices = np.where(traj[:, :2].sum(axis=-1) == 0)[0]
    if len(missing_indices)>traj.shape[0]/2:
        return np.array([])
    if len(missing_indices) > 0:
        for i in range(2):  # Assuming we only need to interpolate the first two columns
            valid_indices = np.where(traj[:, i] != 0)[0]
            if len(valid_indices) == traj[:,i].shape[0]:  # If no missing values, we don't interpolate
                continue
            interpolator = interp1d(valid_indices, traj[valid_indices, i], fill_value="extrapolate")
            traj[missing_indices, i] = interpolator(missing_indices)
    return traj

def interpolate_missing_data(ego, ground_truth, neighbors):
    # interpolate missing values
    future_start_end_with_zeros = False
    for i in range(len(ego)):
        if np.any(ego[i,:,0]==0):
            start_non_zero = np.where(ego[i][:,0]!=0)[0][0]
            end_non_zero = np.where(ego[i][:,0]!=0)[0][-1]
            ego[i, start_non_zero:end_non_zero] = interpolate_missing_traj(ego[i, start_non_zero:end_non_zero])
        if np.any(ground_truth[i,:,0]==0):
            start_non_zero = np.where(ground_truth[i][:,0]!=0)[0][0]
            end_non_zero = np.where(ground_truth[i][:,0]!=0)[0][-1]
            if start_non_zero > 0 or end_non_zero < len(ground_truth[i])-1:
                future_start_end_with_zeros = True
            ground_truth[i, start_non_zero:end_non_zero] = interpolate_missing_traj(ground_truth[i, start_non_zero:end_non_zero])
    for i in range(len(neighbors)):
        if np.any(neighbors[-1,:,0]==0) and sum(neighbors[-1,:,0]!=0)>4:
            start_non_zero = np.where(neighbors[i][:,0]!=0)[0][0]
            end_non_zero = np.where(neighbors[i][:,0]!=0)[0][-1]
            neighbors_interpolated = interpolate_missing_traj(neighbors[i, start_non_zero:end_non_zero])
            if len(neighbors_interpolated)==0:
                continue
            neighbors[i, start_non_zero:end_non_zero] = interpolate_missing_traj(neighbors[i, start_non_zero:end_non_zero])

    return ego, ground_truth, neighbors, future_start_end_with_zeros


def get_agent_caption(agent_name, history, future, navigation_extractor):
    # normalizing the view with respect to the agent, to cacluate directional feature. Irrespective of other agents, so using the location information with the normalized version is not correct
    history_normalized_view = history.copy()
    agent_center, agent_angle = history.copy()[0,:2], history.copy()[0,2]
    valid_mask = history[:,0]!=0
    # print(history[:,:2])
    history_normalized_view[valid_mask,:5] = agent_norm(history.copy()[valid_mask,:], agent_center, agent_angle, impute=True) # with respect to itself
    history_instructs = navigation_extractor.get_navigation_dict_history(torch.tensor(history_normalized_view[:,:5]))

    if future is not None:
        future_normalized_view = future.copy()
        agent_center, agent_angle = future.copy()[0,:2], future.copy()[0,2]
        valid_mask = future[:,0]!=0
        future_normalized_view[valid_mask,:5] = agent_norm(future.copy()[valid_mask,:], agent_center, agent_angle, impute=True) # with respect to itself
        future_instructs = navigation_extractor.get_navigation_dict(torch.tensor(future_normalized_view))
    else:
        return history_instructs
    return {
        **history_instructs,
        **future_instructs
        }

def gen_instruct_caption_01(history, future, navigation_extractor, vizualize=True, normalization_param=None):
    instruct_dict = {}
    for i in range(history.shape[0]):
        if not np.any(history[i][:,0] == 0):
            instruct_dict[f"Agent-{i+1}"] = get_agent_caption(i, history=history[i], future=future[i] if i<2 else None, navigation_extractor=navigation_extractor)
    return instruct_dict

def gen_instruct_caption_02(history, future, navigation_extractor, stop_signs_points, traffic_light_points, traffic_light_states, normalization_param=None):
    
    agent_features = {}
    for i in range(history.shape[0]):
        # if history[i].sum()!=0:
        if not np.any(history[i][:,0] == 0) or i in [0,1]:
            if history.shape[-1]==9:
                agent_type = int(history[i,-1,8])
            else:
                agent_type = int((history[i,-1,-3:].argmax(-1)+1)*history[i,-1,-3:].sum()) # agent_type = [1:car, 2:pedestrian, 3:cylcist], if 0 then undefined}) 
            location_before_1s = f"({history[i,0,0]:.2f}, {history[i,0,1]:.2f})"
            location_current = f"({history[i,-1,0]:.2f}, {history[i,-1,1]:.2f})"
            heading_before_1 = f"{np.rad2deg(wrap_to_pi(history[i,0,2])):.0f} degrees"
            heading_current = f"{np.rad2deg(wrap_to_pi(history[i,-1,2])):.0f} degrees"
            if i==0:
                speeds = np.linalg.norm(abs_distance_to_velocity(history[i,:,:2]),axis=-1)*10
            else:
                speeds = np.linalg.norm(abs_distance_to_velocity(history[i,:,:2][history[i,:,0]!=0]),axis=-1)*10
            speeds = speeds[1:]
            speed_before_1 = f"{int(speeds[0]*3.6)} km/h"
            speed_current = f"{int(speeds[-1]*3.6)} km/h"
            zone_1s, zone_3s, zone_8s, zone_16s = f"{max(speeds):.2f} m", f"{max(speeds)*3:.2f} m", f"{max(speeds)*8:.2f} m", f"{max(speeds)*16:.2f} m"

            if i<2:
                time_indx = int((len(future[i,:]))/2)-1
                speeds = np.linalg.norm(abs_distance_to_velocity(future[i,:,:2][future[i,:,0]!=0]),axis=-1)*10*3.6
                speeds = speeds[1:]
                location_future_4s = f"({future[i,time_indx,0]:.2f}, {future[i,time_indx,1]:.2f})"
                heading_future_4s = f"{np.rad2deg(wrap_to_pi(future[i,time_indx,2])):.0f} degrees"
                if time_indx<len(speeds):
                    speed_future_4s = f"{int(speeds[time_indx-1])} km/h"
                else:
                    speed_future_4s = "Unknown"
                time_indx = int((len(future[i,:])))-1
                location_future_8s = f"({future[i,time_indx,0]:.2f}, {future[i,time_indx,1]:.2f})"
                heading_future_8s = f"{np.rad2deg(wrap_to_pi(future[i,time_indx,2])):.0f} degrees"
                if time_indx<len(speeds):
                    speed_future_8s = f"{int(speeds[time_indx-1])} km/h"
                else:
                    speed_future_8s = "Unknown"
                
                
            
            agent_features.update({f"Agent-{i+1}": {
                "Type": agent_type,
                "Location before 1s": location_before_1s,
                "Current Location": location_current,
                "Heading before 1s": heading_before_1,
                "Current Heading": heading_current,
                "Speed before 1s": speed_before_1,
                "Current Speed": speed_current,
                "Distance in 1s": zone_1s,
                "Distance in 3s": zone_3s,
                "Distance in 8s": zone_8s,
                "Distance in 16s": zone_16s,
                "Future Location in 4s": location_future_4s if i < 2 else None,
                "Future Heading in 4s": heading_future_4s if i < 2 else None,
                "Future Speed in 4s": speed_future_4s if i < 2 else None,
                "Future Location in 8s": location_future_8s if i < 2 else None,
                "Future Heading in 8s": heading_future_8s if i < 2 else None,
                "Future Speed in 8s": speed_future_8s if i < 2 else None
            }})

    map_features = {}
    for i in range(len(stop_signs_points)):
        map_features[f"Stop sign {i+1} location"] = f"({stop_signs_points[i,0]:.2f}, {stop_signs_points[i,1]:.2f})"
        
    states_name = ['unknown', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution']
    for i in range(len(traffic_light_points)):
        if traffic_light_states[i] == 0:
            continue
        map_features[f"Traffic light {i+1} location"] = f"({traffic_light_points[i,0]:.2f}, {traffic_light_points[i,1]:.2f})"
        map_features[f"Traffic light {i+1} state"] = states_name[int(traffic_light_states[i])]


    ## crosswalks
    
    # cross_walks = self.get_crosswalk_polylines()
    # for i, cross_walk_id in enumerate(self.get_crosswalk_polylines()):
    #     print(i)

    ## speedpumps

    ## road edges

    ## plausible and non-plausible sets
    
    return agent_features, map_features

