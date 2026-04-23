import torch
import glob
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("/home/felembaa/projects/iMotion-LLM-ICLR")
import argparse
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
from waymo_open_dataset.protos import scenario_pb2
from utils.data_utils import *
import os
import pickle
from utils.mm_viz import *
import time
from scipy.interpolate import interp1d
from matplotlib.patches import Circle
from instructions.extract_instructions import futureNavigation
import json
from datetime import datetime
from copy import deepcopy
from utils.new_utils import *
from chatgpt_instruct_v02 import *
import shutil
from instructions.direction_instructions import DirectionClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

import numpy as np

def remove_entries_after_turn(entries):
   
    filtered_entries = []
    turn_found = False  # Flag to check if we've encountered a "turn"

    for entry in entries:
        instruction, _ = entry
        
        if "turn" in instruction:
            turn_found = True
            filtered_entries.append(entry)
        elif "straight" in instruction and turn_found:
            continue
        else:
            filtered_entries.append(entry)
    
    return filtered_entries

def remove_uturn_after_opposite_turn(entries):
    
    filtered_entries = []
    turn_direction = "none"  # Track if last turn was "right" or "left"

    for entry in entries:
        instruction, _ = entry

        if "turn right" in instruction and turn_direction=="none":
            turn_direction = "right"
            filtered_entries.append(entry)
        elif "turn left" in instruction and turn_direction=="none":
            turn_direction = "left"
            filtered_entries.append(entry)
        elif "left u-turn" in instruction and turn_direction == "right":
            break
        elif "right u-turn" in instruction and turn_direction == "left":
            break
        else:
            filtered_entries.append(entry)

    return filtered_entries


def pad_and_stack_arrays(array_list, pad_value=0):
   
    max_T = max(array.shape[0] for array in array_list)
    padded_arrays = []
    for array in array_list:
        T = array.shape[0]  # Current number of rows (T)
        padding = ((0, max_T - T), (0, 0))  # Pad only along the first dimension (rows)
        padded_array = np.pad(array, padding, mode='constant', constant_values=pad_value)
        padded_arrays.append(padded_array)
    return np.stack(padded_arrays)

def remove_leading_zeros(lanes):
    """
    Removes all leading rows that consist entirely of zeros from each lane.
    """
    cleaned_lanes = []
    for lane in lanes:
        non_zero_idx = np.where(np.any(lane != 0, axis=1))[0]
        
        if len(non_zero_idx) > 0:
            first_non_zero_idx = non_zero_idx[0]
            cleaned_lanes.append(lane[first_non_zero_idx:])
        else:
            cleaned_lanes.append(np.array([]))

    return cleaned_lanes

def truncate_lanes_vectorized(drivable_lanes_new, threshold=10.0):
    drivable_lanes_new = remove_leading_zeros(drivable_lanes_new)
    truncated_lanes = []
    for lane in drivable_lanes_new:
        if len(lane)<=1:
            continue
        diffs = lane[1:] - lane[:-1]  # Differences between consecutive points
        distances = np.linalg.norm(diffs, axis=1)  # Euclidean distance
        exceed_indices = np.where(distances > threshold)[0]
        if len(exceed_indices) == 0:
            truncated_lanes.append(lane)
        else:
            truncated_lanes.append(lane[:exceed_indices[0] + 1])
    if len(truncated_lanes)==0:
        return []
    else:
        return pad_and_stack_arrays(truncated_lanes)

def truncate_lanes(drivable_lanes_new, threshold=10.0):
    truncated_lanes = []
    for lane in drivable_lanes_new:
        valid_lane = []
        for i in range(len(lane) - 1):
            point1 = lane[i]
            point2 = lane[i + 1]
            distance = np.linalg.norm(point2 - point1)
            valid_lane.append(point1)
            
            if distance > threshold:
                break  # Stop adding points if the distance exceeds the threshold
        valid_lane.append(lane[i + 1])
        truncated_lanes.append(np.array(valid_lane))
    
    return np.array(truncated_lanes)

def connect_lanes_within_distance(lanes, ego_lane, max_distance, max_depth=10, verbose=False):
   
    all_discrete_paths = []  # To store the discrete paths for each lane
  
    all_lane_ids = []  # To store lane IDs
    visited_lanes = set()  # Keep track of visited lanes to avoid cycles

    def traverse_lane(lanes, lane, original_lane, depth, current_discrete_path, current_lane_ids):
        lane_id = list(lane.keys())[0]
    
        if depth > max_depth:
            if verbose:
                print(f"Max depth reached at lane {lane_id}")
            if len(current_lane_ids)>1:
                current_discrete_path.pop()
               
                current_lane_ids.pop()
                all_discrete_paths.append(list(current_discrete_path))
            
                all_lane_ids.append(list(current_lane_ids))
            return
        if lane_id in visited_lanes:
            if verbose:
                print(f"Cycle detected at lane {lane_id}, stopping traversal")
            return
        visited_lanes.add(lane_id)
        if lane != original_lane or depth > 0:
            current_discrete_path.append(list(lane.values())[0])
            current_lane_ids.append(lane_id)
        original_polyline = list(get_polylines(original_lane).values())[0][...,:2]
        lane_polyline = list(get_polylines(lane).values())[0][...,:2]
        if len(original_polyline)>1 and len(lane_polyline)>1:
            original_linestring = LineString(original_polyline)
            lane_linestring = LineString(lane_polyline)
            distance = lane_linestring.distance(original_linestring)
        elif len(original_polyline)==1 and len(lane_polyline)==1:
            distance = np.linalg.norm(original_polyline[0] - lane_polyline[0])
            lane_linestring = None
            original_linestring = None
        else:
            lane_linestring = None
            original_linestring = None
            print('')


        if distance > max_distance:
            if verbose:
                print(f"Stopping traversal for lane {lane_id} due to distance: {distance}")
                print("******************")
            current_discrete_path.pop()
           
            current_lane_ids.pop()
            all_discrete_paths.append(list(current_discrete_path))
        
            all_lane_ids.append(list(current_lane_ids))
            
            return
        outgoing_edges = list(lane.values())[0].exit_lanes
        if len(outgoing_edges)==0:
            if verbose:
                print(f"Reached a leaf lane {lane_id} with no outgoing edges")
                print("###################")
            all_discrete_paths.append(list(current_discrete_path))
            all_lane_ids.append(list(current_lane_ids))
            return
        for i, outgoing_lane_connector_id in enumerate(outgoing_edges):
            next_lane_id = outgoing_lane_connector_id
            next_lane = {next_lane_id: lanes[next_lane_id]}
            next_lane_polyline = list(get_polylines(next_lane).values())[0][...,:2]
            if len(next_lane_polyline)<=1:
                continue
            next_lane_linestring = LineString(next_lane_polyline)
            if lane_linestring is None:
                continue # needs to be corrected, might cause some errors
            new_lane_to_current_lane_distance = lane_linestring.distance(next_lane_linestring)
            if verbose:
                print(f"Processing outgoing lane {i} from lane {lane_id}, next lane ID: {next_lane_id}")
            traverse_lane(lanes, next_lane, original_lane, depth + 1, current_discrete_path, current_lane_ids)
        if lane != original_lane or depth > 0:
            if len(current_lane_ids)>0:
                current_discrete_path.pop()
                current_lane_ids.pop()
    traverse_lane(lanes, ego_lane, ego_lane, 0, current_discrete_path=list(ego_lane.values()),
                  current_lane_ids=[list(ego_lane.keys())[0]])
    unique_data = {
        tuple(lane_id): (list(discrete))
        for lane_id, discrete in zip(all_lane_ids, all_discrete_paths)
    }

    if len(unique_data)==0:
        return [], [], []
    all_lane_ids, paths = zip(*unique_data.items())
    all_lane_ids = [list(lane_id) for lane_id in all_lane_ids]  # Convert tuples back to lists
    all_discrete_paths = list(unique_data.values())
    all_coords = []
    for i in range(len(all_discrete_paths)):
        all_coords_ = []
        for j in range(len(all_discrete_paths[i])):
            polyline_to_append = list(get_polylines({0: all_discrete_paths[i][j]}).values())[0]
            if j>0:
                previous_polyline = all_coords_[j-1]
                distances = np.linalg.norm(polyline_to_append[:, :2, np.newaxis] - previous_polyline[:, :2].T, axis=1)
                closest_index = np.argmin(np.min(distances, axis=1))
                sliced_polyline = polyline_to_append[closest_index:]
                polyline_to_append = sliced_polyline

            all_coords_.append(polyline_to_append)
        all_coords.append(all_coords_)
    
    return all_coords, all_discrete_paths, all_lane_ids
 
def get_vel_from_traj(step1xy, step2xy, time_difference=0.1):
        x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocity = distance / time_difference
        return velocity

def get_distance_between_map_object_and_point(point, polyline):
   
    polyline = np.array([p for p in polyline if not np.any(np.isinf(p)) and not np.any(np.isnan(p))])
    if len(polyline)>1:
        return float(Point(point).distance(LineString(polyline)))
    elif len(polyline)==1:
        return float(Point(point).distance(Point(polyline)))
    else:
        return 10000

def get_rel_direction(angle_, correction_angle):
    angle = minus_2pi(angle_ - correction_angle)
    directly_ahead_threshold = np.radians(0.5)  # Small range around 0 degrees
    directly_behind_threshold = np.radians(179.5)  # Small range around 180 degrees
    front_threshold = np.radians(10)  # Small range around 180 degrees
    behind_threshold = np.radians(110)  # Small range around 180 degrees
    if abs(angle) < directly_ahead_threshold:
        category = "Directly ahead"
    elif abs(angle) > directly_behind_threshold:
        category = "Directly behind"
    elif abs(angle) < front_threshold:
        category = "Ahead"
    elif abs(angle) > behind_threshold:
        category = "Behind"
    else:
        category = "on the left" if angle>0 else "on the right"
       
    return category


def get_unit_vector(pointA, pointB):
    vector_AB = pointB - pointA
    magnitude_AB = np.linalg.norm(vector_AB)
    unit_vector_AB = vector_AB / magnitude_AB if magnitude_AB != 0 else vector_AB
    return unit_vector_AB, magnitude_AB

def search_nearest_line(lanes, point_):
    point = Point(point_)
    nearest = 1000
    nearest_vec = np.array([0,0])
    for lane in lanes.keys(): 
        if lanes[lane].shape[0] > 1:
            line = LineString(lanes[lane][:, :2])
            closest_point_on_line = np.array(line.interpolate(line.project(point)).coords[0])
            unit_vector, mag = get_unit_vector(np.array(point.coords[0]), closest_point_on_line)
            if mag < nearest:
                nearest_lane = lane
                nearest = mag
                nearest_vec = unit_vector
    return nearest, nearest_vec, nearest_lane

def vizualize_background(segment, segment_type, fig, categ, ax, force_color=None):
   
    center_plt_info = [('lightgray','-', 0.1),('gray','solid', 0.4), ('red','solid', 0.4), ('b','dashed', 0.2),]
   
    boundry_plt_info = [('lightgray','-', 0.1), ('w','dashed', 0.4),  ('w','solid', 0.4), ('w','solid', 0.4), ('xkcd:yellow','dashed', 0.4), ('xkcd:yellow','dashed', 0.3), ('xkcd:yellow','solid', 0.3), ('xkcd:yellow','solid', 0.3), ('xkcd:yellow','dotted', 0.4),('k', '-', 1.0), ('k', '-', 1.0),]
    edge_plt_info = [('k', '-', 1.0), ('k', '-', 1.0), ('k', '-', 1.0),]
    stopsign_plt_info = [('lightgray','-', 0.1),('r', 'solid',1)]
    crosswalk_plt_info = [('white','-', 1.0),('white', '-',1.0)]
    speedbump_plt_info = [('orange','-', 1.0),('orange', '-',1.0)]
    plt_info = [center_plt_info, boundry_plt_info, edge_plt_info, stopsign_plt_info, crosswalk_plt_info, speedbump_plt_info]
    z_orders = [2,2,2,3,1,1,3]
    if categ==4 or categ==5:
        segment_type=1
    if not fig is not None:
        fig, ax = plt.subplots(dpi=300)
       
        fig.set_tight_layout(True)
        plt.gca().set_facecolor('silver')
        plt.gca().margins(0)  
       

    if categ!=3 and categ!=6:
        if force_color is not None:
            plt.plot(segment[:,0], segment[:,1], (force_color,'solid', 1)[0], linestyle=(force_color,'solid', 2)[1], linewidth=1, alpha=(force_color,'solid', 0.5)[2], zorder=z_orders[categ])
        else:
            plt.plot(segment[:,0], segment[:,1], plt_info[categ][segment_type][0], linestyle=plt_info[categ][segment_type][1], linewidth=1, alpha=plt_info[categ][segment_type][2], zorder=z_orders[categ])
    elif categ==3:
        circle = Circle((segment[0], segment[1]), 2, color='red', fill=False, zorder=z_orders[categ])
        plt.gca().add_patch(circle)
        plt.text(segment[0], segment[1], 'STOP', color='red', fontsize=1, ha='center', va='center', zorder=z_orders[categ])
    elif categ==6:
        if segment_type in [1, 4, 7]:
            color = 'red'
            circle = Circle((segment[0], segment[1]), 1.0, color=color, zorder=z_orders[categ])
            plt.gca().add_patch(circle)
        elif segment_type in [2, 5, 8]:
            color = 'orange'
            circle = Circle((segment[0], segment[1]), 1.0, color=color, zorder=z_orders[categ])
            plt.gca().add_patch(circle)
        elif segment_type in [3, 6]:
            color = 'green'
            circle = Circle((segment[0], segment[1]), 1.0, color=color, zorder=z_orders[categ])
            plt.gca().add_patch(circle)
    return fig, ax

def vizualize_traj_arrow(history_traj, traj, fig, ax, color, linestyle='solid', add_arrow=True, alpha=0.5, arrcolor='b'): # single agent, single modality
    if not add_arrow:
        plt.scatter(traj[:, 0], traj[:, 1], color=color, alpha=alpha, s=10, marker='o')
    if add_arrow:
        plt.plot(traj[:-1,0], traj[:-1,1], linestyle=linestyle, color=color, alpha=alpha, linewidth=1)
        if np.linalg.norm(traj[-1]-traj[0]) > 2: # if more than 2 meter travel, draw an arrow
            ax.annotate('', xy=traj[-1], xytext=traj[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1, linestyle='solid', alpha=0.5))   

def rect_(x_pos, y_pos, width, height, heading, object_type, color='r', alpha=0.8, fontsize=3):
    rect_x = x_pos - width / 2
    rect_y = y_pos - height / 2
    rect = plt.Rectangle((rect_x, rect_y), width, height, linewidth=2, color=color, alpha=alpha, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_pos, y_pos), heading) + plt.gca().transData)
    plt.gca().add_patch(rect)
    plt.text(x_pos, y_pos, object_type, color='black', fontsize=fontsize, ha='center', va='center', fontweight='bold')
    
def vizualize_agent_rect(fig, past, object_type, color, agent_prefix=''):
    agent_i = past
    object_type_str = ['Unset', 'Car', 'Pedestrian', 'Cyclist']
    object_type_str = [str(agent_prefix) for object_type_str_i in object_type_str]
    rect_(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type_str[int(object_type)], color=color)


class DataProcess(object):
    def __init__(
                self, 
                root_dir=[''],
                point_dir='',
                save_dir='',
                num_neighbors=32,
                ):
        self.num_neighbors = num_neighbors
        self.hist_len = 11
        self.future_len = 80
        self.data_files = root_dir
        self.point_dir = point_dir
        self.save_dir = save_dir
        self.drivable_lanes = 500
        self.num_crosswalks = 12
        self.global_counter = 0
    
    
    def build_points(self):
        self.points_dict = {}
        for obj_type in ['vehicle','pedestrian','cyclist']:
            for c in [6,32,64]:
                with open(self.point_dir + f'{obj_type}_{c}.pkl','rb') as reader:
                    data = pickle.load(reader)
                assert data.shape[0]==c
                self.points_dict[f'{obj_type}_{c}'] = data

    def build_map(self, map_features, dynamic_map_states):
        self.lanes = {}
        self.roads = {}
        self.stop_signs = {}
        self.crosswalks = {}
        self.speed_bumps = {}
        self.roads_boundary = {}
        self.roads_edge = {}
        for map in map_features:
            map_type = map.WhichOneof("feature_data")
            map_id = map.id
            map = getattr(map, map_type)

            if map_type == 'lane':
                self.lanes[map_id] = map
            elif map_type == 'road_line' or map_type == 'road_edge':
                self.roads[map_id] = map
                if map_type == 'road_edge':
                    self.roads_edge[map_id] = map
                else:
                    self.roads_boundary[map_id] = map
            elif map_type == 'stop_sign': 
                self.stop_signs[map_id] = map # TODO: This to be used to generate captions, similar to what will be showed later (check for the later TODOs)
            elif map_type == 'crosswalk': 
                self.crosswalks[map_id] = map # TODO: This to be used to generate captions
            elif map_type == 'speed_bump':
                self.speed_bumps[map_id] = map # TODO: This to be used to generate captions
            else:
                continue
        self.traffic_signals = dynamic_map_states # TODO: This to be used to generate captions

    def plt_scene_lanes_segments(self, agent_1=None, agent_2=None, other_agents=None, figure_center=None, max_num_lanes=10):
        lane_polylines = get_polylines(self.lanes)
        lane_types = [value.type for value in self.lanes.values()]
        fig=None
        ax=None
        num_lanes = 0
        colors_list = [
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 
            'cyan', 'magenta', 'lime', 'teal', 'indigo', 'violet', 'gold', 
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 
            'cyan', 'magenta', 'lime', 'teal', 'indigo', 'violet', 'gold', 
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 
            'cyan', 'magenta', 'lime', 'teal', 'indigo', 'violet', 'gold', 
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 
            'cyan', 'magenta', 'lime', 'teal', 'indigo', 'violet', 'gold', 
        ]

        for i, lane in enumerate(lane_polylines.values()):
            if lane_types[i] in [0,1,2]:
                fig, ax = vizualize_background(segment=lane, segment_type=lane_types[i], fig=fig, categ=0, ax=ax, force_color = colors_list[i])
                num_lanes+=1
                if num_lanes>max_num_lanes:
                    break
            else:
                continue

       
        if figure_center is not None:
            center_x = figure_center[0]  # Replace with your actual center x-coordinate
            center_y = figure_center[1]  # Replace with your actual center y-coordinate
            range_crop = 75  # Crop range around the center
            ax.set_xlim(center_x - range_crop, center_x + range_crop)
            ax.set_ylim(center_y - range_crop, center_y + range_crop)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return fig

    def plt_scene(self, agent_1=None, agent_2=None, other_agents=None, figure_center=None, max_num_lanes=10, direction_lanes = None, map_lanes=None, drivable_lanes=None):
        fig=None
        ax=None
        num_lanes = 0
        colors_list = [
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 
            'cyan', 'magenta', 'lime', 'teal', 'indigo', 'violet', 'gold', 
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 
            'cyan', 'magenta', 'lime', 'teal', 'indigo', 'violet', 'gold', 
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 
            'cyan', 'magenta', 'lime', 'teal', 'indigo', 'violet', 'gold', 
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 
            'cyan', 'magenta', 'lime', 'teal', 'indigo', 'violet', 'gold', 
        ]

        if map_lanes is None:
            lane_polylines = get_polylines(self.lanes)
            lane_types = [value.type for value in self.lanes.values()]

            for i, lane in enumerate(lane_polylines.values()):
                if lane_types[i] in [0,1,2]:
                    fig, ax = vizualize_background(segment=lane, segment_type=lane_types[i], fig=fig, categ=0, ax=ax, force_color = 'k')
                    num_lanes+=1
                    if num_lanes>max_num_lanes:
                        break
                else:
                    continue
        else:
            for i, lane in enumerate(map_lanes):
                lane = lane[[lane[:,10][k] in [0,1,2] for k in range(len(lane[:,10]))]]
                lane = lane[lane[:,0]!=0]
                fig, ax = vizualize_background(segment=lane, segment_type=2, fig=fig, categ=0, ax=ax, force_color = 'k')
                num_lanes+=1
                if num_lanes>max_num_lanes:
                    break
                else:
                    continue
        
        if drivable_lanes is not None:
            for i, lane in enumerate(drivable_lanes):
                lane = lane[lane[:,0]!=0]
                fig, ax = vizualize_background(segment=lane, segment_type=2, fig=fig, categ=0, ax=ax, force_color = 'r')
        
        if direction_lanes is not None:
            special_lanes = direction_lanes
            unique_tuples = list({class_label: array for class_label, array in special_lanes.values()}.items())
            special_lanes = unique_tuples
            classes_colors = ['brown', 'b', 'c', 'orange', 'yellow', 'purple', 'pink', 'brown', 'gray']
            classes_names = ['stationary', 'straight', 'straight-right', 'straight-left', 'right', 'left', 'right-u-turn', 'left-u-turn']
            for lane_class, lane in special_lanes:
                plt.plot(lane[:, 0], lane[:, 1], color=classes_colors[lane_class], 
                        linewidth=5, alpha=0.9, label=classes_names[lane_class])
                plt.legend()
    

        if True:
            if other_agents is not None:
                alphabets = [i for i in range(50)]
                for agent_idx, agent_ in enumerate(other_agents):
                    agent_type = int((agent_[-1,-3:].argmax(-1)+1)*agent_[-1,-3:].sum())
                    if agent_type>0:
                        colors = ['','dimgrey', 'sandybrown', 'dimgrey']
                        vizualize_agent_rect(fig=fig, past=agent_, object_type=agent_type, color=colors[agent_type], agent_prefix=alphabets[agent_idx+3])
            if agent_1 is not None:            
                agent_ = agent_1
                agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                colors = ['','darkviolet', 'darkviolet', 'darkviolet']
                vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='1') # 'Ego'
                vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
            if agent_2 is not None:            
                agent_ = agent_2
                agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                colors = ['','blue', 'blue', 'blue']
                vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='2') # 'Interactive'
                vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
            

        if figure_center is not None:
            center_x = figure_center[0]  # Replace with your actual center x-coordinate
            center_y = figure_center[1]  # Replace with your actual center y-coordinate
            range_crop = 75  # Crop range around the center
            ax.set_xlim(center_x - range_crop, center_x + range_crop)
            ax.set_ylim(center_y - range_crop, center_y + range_crop)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return fig


    def get_stop_signs_locations(self):
        stop_sign_lanes = []
        locations = []
        for i, sign in self.stop_signs.items():
            stop_sign_lanes.append(sign.lane)
            locations.append([sign.position.x,sign.position.y])
        locations, idx = np.unique(locations, axis=0, return_index=True)
        stop_sign_lanes = [stop_sign_lanes[i] for i in idx]
        return locations, stop_sign_lanes

    def get_traffic_light_control(self, current_location, current_lane):
        
        traffic_light_location = None # it could be None, if no traffic light control the lane
        traffic_light_state = None
        distance_to_traffic_light = None # by comparing the current_location
        return traffic_light_location, traffic_light_state, distance_to_traffic_light

    def get_traffic_light_locations(self, unique_only=False):
        traffic_light_lanes = {}

        for signal in self.traffic_signals[self.hist_len-1].lane_states:
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self.lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        locations = []
        states = []
        for lane in traffic_light_lanes.keys():
            state = traffic_light_lanes[lane][0]
            states.append(state)
            traffic_light_location = traffic_light_lanes[lane][1:]
            locations.append(list(traffic_light_location))

        controlled_lanes = list(traffic_light_lanes.keys())

        if unique_only:
            locations, idx = np.unique(locations, axis=0, return_index=True)
            states = np.array(states)[idx]
            controlled_lanes = np.array(controlled_lanes)[idx]
        
        return locations, states, controlled_lanes

    def get_crosswalk_polylines(self):
        polylines = {}
        for k, crosswalk in self.crosswalks.items():
            polygon = Polygon([(point.x, point.y) for point in crosswalk.polygon])
            polyline = polygon_completion(crosswalk.polygon)
            polyline = polyline[np.linspace(0, polyline.shape[0], num=100, endpoint=False, dtype=np.int32)]
            polylines[k] = polyline
        return polylines

    def onoffroad(self, traj, future_traj_):
        lane_polylines = get_polylines(self.lanes) # get the xy information, center lane
        lane_types = [value.type for value in self.lanes.values()] # get the type, # 1:  
        road_polylines = get_polylines(self.roads)
        road_types = [value.type for value in self.roads.values()]
        edges_polylines = get_polylines(self.roads_edge)
        edges_types = [value.type for value in self.roads_edge.values()]

        current_xyh = traj[-1,:3] # our current time step xyh, where h is heading angle
        final_xyh = future_traj_[-1,:3] # the last step in the future xyh
        current_nearest_lane_distance, current_nearest_lane_vec, nearest_lane = search_nearest_line(lane_polylines, current_xyh[:2])
        final_nearest_lane_distance, final_nearest_lane_vec, nearest_lane = search_nearest_line(lane_polylines, final_xyh[:2])
        current_nearest_road_distance, current_nearest_road_vec, nearest_lane = search_nearest_line(edges_polylines, current_xyh[:2])
        final_nearest_road_distance, final_nearest_road_vec, nearest_lane = search_nearest_line(edges_polylines, final_xyh[:2])
        current_in= current_nearest_lane_distance<2
        final_in= final_nearest_lane_distance<2
        if not current_in:                
            if current_nearest_road_distance>current_nearest_lane_distance:
                if np.dot(current_nearest_road_vec,current_nearest_lane_vec)>0.2:
                    current_in=True
        if not final_in:
            if final_nearest_road_distance>final_nearest_lane_distance:
                if np.dot(final_nearest_road_vec,final_nearest_lane_vec)>0.2:
                    final_in=True
        if not current_in:
            if np.dot(current_nearest_road_vec,current_nearest_lane_vec)<-0.5:
                current_in=True
        if not final_in:
            if np.dot(final_nearest_road_vec,final_nearest_lane_vec)<-0.5:
                final_in=True
        if current_nearest_lane_distance>25:
            current_in=False
        if final_nearest_lane_distance>25:
            final_in=False

        if not current_in and not final_in:
            inroad_state = {'current map state': 'off-road', 'future map state':'stay off-road', 'on-off map caution': 'be careful driving off-road'}
        if not current_in and final_in:
            inroad_state = {'current map state': 'off-road', 'future map state':'merge to the road'}
        if current_in and not final_in:
            inroad_state = {'current map state': 'on-road', 'future map state':'go off-road', 'on-off map caution': 'be careful going from on-road to off-road'}
        else:
            inroad_state = {'current map state': 'on-road', 'future map state':'stay on-road'}
        
        return inroad_state
    
    def get_agent_caption(self, agent_name, history, future, navigation_extractor):
        history_normalized_view = history.copy()
        
        agent_center, agent_angle = history[valid_mask].copy()[0,:2], history[valid_mask].copy()[0,2]
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
        

    def gen_instruct_caption_02(self, history, future, navigation_extractor, vizualize=True, normalization_param=None):
        
        
        agent_features = {}
        for i in range(history.shape[0]):
            if not np.any(history[i][:,0] == 0) or i in [0,1]:
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
        stop_signs_points, stop_signs_lanes  = self.get_stop_signs_locations()
        stop_i = 0
        for i in range(len(stop_signs_points)):
            stop_signs_point_local_map = map_norm(stop_signs_points[i][None], center=normalization_param[0], angle=normalization_param[1])[0,:2]
            if stop_i!=0:
                if f"({stop_signs_point_local_map[0]:.2f}, {stop_signs_point_local_map[1]:.2f})" == map_features[f"Stop sign {stop_i} location"]:
                    continue
            map_features[f"Stop sign {stop_i+1} location"] = f"({stop_signs_point_local_map[0]:.2f}, {stop_signs_point_local_map[1]:.2f})"
            stop_i += 1
            
        signs_points, signs_states, signs_lanes  = self.get_traffic_light_locations()
        states_name = ['unknown', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution']
        traffic_i = 0
        for i in range(len(signs_points)):
            if signs_states[i] == 0:
                continue # print('unknown traffic light')
            traffic_point_local_map = map_norm(np.array(signs_points[i])[None], center=normalization_param[0], angle=normalization_param[1])[0,:2]
            if traffic_i!=0:
                if f"({traffic_point_local_map[0]:.2f}, {traffic_point_local_map[1]:.2f})" == map_features[f"Traffic light {traffic_i} location"]:
                    continue
            map_features[f"Traffic light {traffic_i+1} location"] = f"({traffic_point_local_map[0]:.2f}, {traffic_point_local_map[1]:.2f})"
            map_features[f"Traffic light {traffic_i+1} state"] = states_name[signs_states[i]]
            traffic_i+=1
        return agent_features, map_features

    def gen_instruct_caption_01(self, history, future, navigation_extractor, vizualize=True, normalization_param=None, viz_dir=None):
        
        instruct_dict = {}
        
        for i in range(history.shape[0]):
            if sum(history[i][:,0]!=0)>=2:
                if not np.any(history[i][:,0] == 0):
                    instruct_dict[f"Agent-{i+1}"] = self.get_agent_caption(i, history=history[i], future=future[i] if i<2 else None, navigation_extractor=navigation_extractor)
                else:
                    instruct_dict[f"Agent-{i+1}"] = self.get_agent_caption(i, history=history[i], future=future[i] if i<2 else None, navigation_extractor=navigation_extractor)
                   
            
      
        vizualize=False
        if vizualize:
            history_global_frame = history
            future_global_frame = future
            subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int) # subsample to sample/second
            fig = self.plt_scene((history_global_frame[0], future_global_frame[0, subsample_indices]), max_num_lanes=50)
            fig.savefig('ex.png')
            plt.close()
        
        return instruct_dict

    def gen_instruct_caption_v01(self, history_ego, future_ego, history_interactive_neighbor, future_interactive_neighbor, other_neighbors, navigation_extractor=None):
        traj = self.interpolate_missing_traj(history_ego.copy())
        future_traj_ = self.interpolate_missing_traj(future_ego.copy()) # correction
        if len(traj)==0 or len(future_traj_)==0:
            return {}
        

        inter_traj = self.interpolate_missing_traj(history_interactive_neighbor.copy())
        inter_future_traj_ = self.interpolate_missing_traj(future_interactive_neighbor.copy()) # correction

        alphabets = [i+1 for i in range(50)]
        alphabets_idx = [i for i in range(len(alphabets))]
        
        lane_polylines = get_polylines(self.lanes)
        subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int) # subsample to sample/second
        future_traj_subsampled = future_traj_[subsample_indices]
        inter_future_traj_subsampled = inter_future_traj_[subsample_indices]
        
        json_dict = {}
        agent_type = int((history_ego[-1,-3:].argmax(-1)+1)*history_ego[-1,-3:].sum()) # agent_type = [1:car, 2:pedestrian, 3:cylcist], if 0 then undefined
        json_dict['Agent-1'] = {'type': agent_type}
        if len(history_ego[:,:2][history_ego[:,0]!=0])>1:
            speeds = np.linalg.norm(abs_distance_to_velocity(history_ego[:,:2][history_ego[:,0]!=0]),axis=-1)*10*3.6
            speed_past, speed_current = int(speeds[1]), int(speeds[-1])
            current_instructs = navigation_extractor.get_navigation_dict(torch.tensor(future_traj_))
            history_instructs = navigation_extractor.get_navigation_dict_history(torch.tensor(history_ego[:,:5]))
        agent_type_str_map = ['Unknown', 'Vehicle', 'Pedestrian', 'Cyclist']
        if agent_type != 0:
            state_features = {
                'Agent-1 speed': str(np.linalg.norm(traj[-1,3:5]).round(2)),
                'Agent-1 heading angle in radians': str(traj[-1,2].round(2)),
                'Agent-1 location': "(0.00, 0.00)",
                }
            agent_name = '1'
            ego_num = 0
            inter_num = 1
            onroad_state_ego = self.onoffroad(traj, future_traj_)
            instructs_ego = navigation_extractor.get_navigation_dict(torch.tensor(future_traj_))
            t_sceond_rule={}
            for ts in [1,3,8]:
                t_sceond_rule[f"Agent-{agent_name} {ts}sec agents"] = []
                ts_neighbors = self.t_seconds_rule(traj_future=future_traj_, traj_history=traj, neighbors=other_neighbors, seconds=ts)
                if len(inter_traj)==0:
                    ts_interactive = {}
                else:
                    ts_interactive = self.t_seconds_rule(traj_future=future_traj_, traj_history=traj, neighbors=inter_traj[np.newaxis], seconds=ts)
                    t_sceond_rule[f"Agent-{agent_name} {ts}sec agents"].extend([f"Agent-{alphabets[inter_num]} {ts_interactive[1][ii]}" for ii, ni in enumerate(ts_interactive[-1])])

                t_sceond_rule[f"Agent-{agent_name} {ts}sec agents"].extend([f"Agent-{alphabets[ni+2]} {ts_neighbors[1][ii]}" for ii, ni in enumerate(ts_neighbors[-1])])
                t_sceond_rule[f"Agent-{agent_name} {ts}sec distance"] = str(ts_neighbors[0].round(2))
            
            dict1 = {f"Agent-1 {k}":v for k,v in onroad_state_ego.items()}
            dict2 = {f"Agent-1 {k}":v for k,v in instructs_ego.items()}
            
            if len(inter_traj)!=0:
                neighbor_vector, distance_ = get_unit_vector(traj[-1,:2], inter_traj[-1,:2])
                angle = rel_angle(traj[-1,:2], inter_traj[-1,:2])
                category = get_rel_direction(angle, correction_angle=rel_angle(traj[-2,:2], traj[-1,:2]))
                if 'left' in category or 'right' in category:
                    my_vector, _ = get_unit_vector(traj[-2,:2], traj[-1,:2])
                    category = "on the left" if np.cross(my_vector, neighbor_vector)>0 else "on the right"
                rel_inter_2_ego = {"Agent-2 location to Agent-1": f"The interactive neighbor, termed Agent-2, is {distance_:.2f}m away {category.lower()} of the ego Agent-1."}
            else:
                rel_inter_2_ego = {}
            stop_sign_found = self.get_relevant_stop_sign(traj, detection_threshold=150)
            dict_stop = {f"Agent-1 stop sign {k}":v for k,v in stop_sign_found.items()}
            traffic_sign_found = self.get_relevant_traffic_sign(traj, detection_threshold=150)
            dict_traffic = {f"Agent-1 traffic sign {k}":v for k,v in traffic_sign_found.items()}
            agents_within = self.get_agents_within(traj, inter_traj, other_neighbors, inter_num, 20)
            if len(agents_within)>0:
                agents_within_dict = {f"Agents within 20 meters from Agent-1": agents_within}
            else:
                agents_within_dict = {f"Agents within 20 meters from Agent-1": "No agents"}

            agent_type_ = int((history_interactive_neighbor[-1,-3:].argmax(-1)+1)*history_interactive_neighbor[-1,-3:].sum()) # agent_type = [1:car, 2:pedestrian, 3:cylcist], if 0 then undefined
            if agent_type_!=0:
                if len(inter_traj)==0 or len(inter_future_traj_)==0:
                    caption_json = {
                        **state_features,
                        **dict1,
                        **t_sceond_rule,
                        **dict_traffic,
                        **dict_stop,
                        **agents_within_dict,
                        **rel_inter_2_ego,
                        **dict2,
                    }
                    return caption_json
                agent_name = '2'
                ego_num = 1
                inter_num = 0
                local_location = inter_traj[-1,:2] - traj[-1,:2]
                local_location = local_location.round(1)
                state_features.update(
                    {
                    'Agent-2 speed': str(np.linalg.norm(inter_traj[-1,3:5]).round(2)),
                    'Agent-2 heading angle in radians': str(inter_traj[-1,2].round(2)),
                    'Agent-2 location': f"({local_location[0]:.2f}, {local_location[1]:.2f})",
                    })
                traj, future_traj_, inter_traj = inter_traj, inter_future_traj_, traj
                onroad_state_inter = self.onoffroad(traj, future_traj_)
                instructs_inter = navigation_extractor.get_navigation_dict(torch.tensor(future_traj_))
                for ts in [1,3,8]:
                    t_sceond_rule[f"Agent-{agent_name} {ts}sec agents"] = []
                    ts_neighbors = self.t_seconds_rule(traj_future=future_traj_, traj_history=traj, neighbors=other_neighbors, seconds=ts)
                    if len(inter_traj)==0:
                        ts_interactive = {}
                    else:
                        ts_interactive = self.t_seconds_rule(traj_future=future_traj_, traj_history=traj, neighbors=inter_traj[np.newaxis], seconds=ts)
                        t_sceond_rule[f"Agent-{agent_name} {ts}sec agents"].extend([f"Agent-{alphabets[inter_num]} {ts_interactive[1][ii]}" for ii, ni in enumerate(ts_interactive[-1])])

                    t_sceond_rule[f"Agent-{agent_name} {ts}sec agents"].extend([f"Agent-{alphabets[ni+2]} {ts_neighbors[1][ii]}" for ii, ni in enumerate(ts_neighbors[-1])])
                    t_sceond_rule[f"Agent-{agent_name} {ts}sec distance"] = str(ts_neighbors[0].round(2))

                if len(inter_traj)!=0:
                    neighbor_vector, distance_ = get_unit_vector(traj[-1,:2], inter_traj[-1,:2])
                    angle = rel_angle(traj[-1,:2], inter_traj[-1,:2])
                    category = get_rel_direction(angle, correction_angle=rel_angle(traj[-2,:2], traj[-1,:2]))
                    if 'left' in category or 'right' in category:
                        my_vector, _ = get_unit_vector(traj[-2,:2], traj[-1,:2])
                        category = "on the left" if np.cross(my_vector, neighbor_vector)>0 else "on the right"
                    rel_inter_2_ego.update({"Agent-2 view":f"The ego agent, termed Agent-1, is {distance_:.2f}m away {category.lower()} of the interactive neighbor Agent-2."})

            
                
                stop_sign_found = self.get_relevant_stop_sign(traj, detection_threshold=150)
                dict_stop.update({f"Agent-2 stop sign {k}":v for k,v in stop_sign_found.items()})
                traffic_sign_found = self.get_relevant_traffic_sign(traj, detection_threshold=150)
                dict_traffic.update({f"Agent-2 traffic sign {k}":v for k,v in traffic_sign_found.items()})

                agents_within = self.get_agents_within(traj, inter_traj, other_neighbors, inter_num, 20)
                if len(agents_within)>0:
                    agents_within_dict.update({f"Agents within 20 meters from Agent-2": agents_within})
                else:
                    agents_within_dict.update({f"Agents within 20 meters from Agent-2": "No agents"})

                dict3 = {f"Agent-2 {k}":v for k,v in onroad_state_inter.items()}
                dict4 = {f"Agent-2 {k}":v for k,v in instructs_inter.items()}
            else:
                dict3 = {}
                dict4 = {}
            
            agents_types_list = (other_neighbors[:,-1,-3:].argmax(-1)+1) * other_neighbors[:,-1,-3:].sum(-1)
            agents_types_dict = {
                'Agent-1 type': agent_type_str_map[agent_type],
                'Agent-2 type': agent_type_str_map[agent_type_],
            }
            agents_types_dict.update({
                f"Agent-{agent_i+2} type":agent_type_str_map[int(agents_types_list[agent_i])] for agent_i in range(len(agents_types_list)) if int(agents_types_list[agent_i])!=0
                })
            neighbors_distance_list = {}
            for agent_i, neighbor in enumerate(other_neighbors):
                if neighbor[-1,0]!=0:
                    neighbors_distance_list[f"Agent-{agent_i+2} distance to Agent-1 at current time step"] \
                        = f"{get_unit_vector(traj[-1,:2], neighbor[-1,:2])[1]:.2f}m" if traj[-1,0] != 0 else "Unknown"
                    neighbors_distance_list[f"Agent-{agent_i+2} distance to Agent-1 at future time step (8 seconds in the future)"] \
                        = f"{get_unit_vector(future_traj_subsampled[-1,:2], neighbor[-1,:2])[1]:.2f}m" if future_traj_subsampled[-1,0] != 0 else "Unknown"
            

            caption_json = {
                **state_features,
                **dict1,
                **dict3,
                **dict4,
                **t_sceond_rule,
                **dict_traffic,
                **dict_stop,
                **agents_within_dict,
                **rel_inter_2_ego,
                **dict2,
                **agents_types_dict,
                **neighbors_distance_list
            }
            
        else:
            caption_json = {}

        return caption_json

    def get_agents_within(self, traj_history, inter_traj, other_neighbors, inter_traj_num, detection_range=20):
        alphabets = [i+1 for i in range(50)]
        detected = []
        neighbors_ = other_neighbors[:,-1,:2][other_neighbors[:,-1,:2].sum(-1)!=0] 
        if len(inter_traj)!=0:
            if len(neighbors_)!=0:
                neighbors_ = np.concatenate((inter_traj[-1,:2][np.newaxis], neighbors_))
            else:
                neighbors_ = inter_traj[-1,:2][np.newaxis]
        for i, neighbor in enumerate(neighbors_):
            neighbor_vector, distance = get_unit_vector(traj_history[-1,:2], neighbor)
            if distance<detection_range:
                angle = rel_angle(traj_history[-1,:2], neighbor)
                category = get_rel_direction(angle, correction_angle=rel_angle(traj_history[-2,:2], traj_history[-1,:2]))
                if 'left' in category or 'right' in category:
                    my_vector, _ = get_unit_vector(traj_history[-2,:2], traj_history[-1,:2])
                    category = "on the left" if np.cross(my_vector, neighbor_vector)>0 else "on the right"
                if i==0:
                    detected.append(f"Agent-{alphabets[inter_traj_num]} {category}")
                else:
                    if len(inter_traj)!=0:
                        detected.append(f"Agent-{alphabets[i+1]} {category}")
                    else:
                        detected.append(f"Agent-{alphabets[i+2]} {category}")
        return detected

    
    def get_relevant_traffic_sign(self, traj_history, detection_threshold=150):
        lane_polylines = get_polylines(self.lanes)
        current_nearest_lane_distance, current_nearest_lane_vec, nearest_lane = search_nearest_line(lane_polylines, traj_history[-1,:2])
        signs_points, states, signs_lanes  = self.get_traffic_light_locations()
        if len(signs_points)==0:
            return {}
        signs_dist = np.linalg.norm(traj_history[-1,:2] - signs_points, axis=-1)
        idx = [i for i in range(len(signs_lanes)) if nearest_lane == signs_lanes[i]]
        if len(idx)==0:
            idx = [i for i in range(len(signs_lanes)) if signs_dist[i]<detection_threshold]
        valid_signs = {}
        if len(idx)==0:
            valid_signs = {}
        else:
            distances = []
            for i in idx:
                sign_unit_vector, distance = get_unit_vector(traj_history[-1,:2], signs_points[i])
                angle = rel_angle(traj_history[-1,:2], signs_points[i])
                category = get_rel_direction(angle, correction_angle=rel_angle(traj_history[-2,:2], traj_history[-1,:2]))
                if 'ahead' in category.lower() or (distance<10 and not 'behind' in category.lower()):
                    states_name = ['unknown', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution']
                    if states[i] == 0:
                        continue
                    valid_signs[len(distances)] = {'state':states_name[states[i]], 'distance': distance.round(2), 'relative location': 'just in front of the car' if distance<10 else 'ahead'}
                    distances.append(distance)
            if len(distances)>0:
                if nearest_lane in signs_lanes:
                    valid_signs = valid_signs[np.argmin(distances)]
                elif min(distances)<150:
                    valid_signs = valid_signs[np.argmin(distances)]
                else:
                    x=0
            
        return valid_signs

    def get_relevant_stop_sign(self, traj_history, detection_threshold=150):
        lane_polylines = get_polylines(self.lanes)
        current_nearest_lane_distance, current_nearest_lane_vec, nearest_lane = search_nearest_line(lane_polylines, traj_history[-1,:2])
        stop_signs_points, stop_signs_lanes  = self.get_stop_signs_locations()
        if len(stop_signs_points)==0:
            return {}
        stop_signs_dist = np.linalg.norm(traj_history[-1,:2] - stop_signs_points, axis=-1)
        idx = [i for i in range(len(stop_signs_lanes)) if nearest_lane in stop_signs_lanes[i] or stop_signs_dist[i]<detection_threshold]
        valid_signs = {}
        if len(idx)==0:
            valid_signs = {}
        else:
            distances = []
            for i in idx:
                stop_sign_unit_vector, distance = get_unit_vector(traj_history[-1,:2], stop_signs_points[i])
                angle = rel_angle(traj_history[-1,:2], stop_signs_points[i])
                category = get_rel_direction(angle, correction_angle=rel_angle(traj_history[-2,:2], traj_history[-1,:2]))
                if 'left' in category or 'right' in category:
                    my_vector, _ = get_unit_vector(traj_history[-2,:2], traj_history[-1,:2])
                    category = "on the left" if np.cross(my_vector, stop_sign_unit_vector)>0 else "on the right"
                if 'behind' not in category.lower(): # if in front of me or on my right or left
                    valid_signs[len(distances)] = {'state':1, 'direction': category, 'distance': distance.round(2), 'relative location': 'next to the car' if distance<5 else 'Ahead'}
                    distances.append(distance)
            if len(distances)>0:
                valid_signs = valid_signs[np.argmin(distances)]
            
        return valid_signs
        


    def t_seconds_rule(self, traj_future, traj_history, neighbors, seconds=3):
        
        hist_vel = abs_distance_to_velocity(traj_history[:,:2])[1:]*10 # *10 to convert it to m/s
        current_velocity = hist_vel[-1] # speed in m/s
        safe_relative_point = current_velocity*seconds # the relative future point compared to our reference point 
        safe_point = traj_history[-1,:2]+safe_relative_point # the absolute safe point
        safe_vector, safe_distance = get_unit_vector(traj_history[-1,:2], safe_point) # directional vector to the safe point and safe range
        safe_agents = []
        safe_agents_direction = []
        safe_agents_distance = []
        risk_agents = []
        risk_agents_direction = []
        risk_agents_distance = []
        neighbors_ = neighbors[neighbors[:,-1,:2].sum(-1)!=0]
        if len(neighbors_)==0:
            return safe_distance, risk_agents_direction, risk_agents
        for i, neighbor in enumerate(neighbors):
            risk=False
            neighbor_vector, neighbor_distance = get_unit_vector(traj_history[-1,:2], neighbor[-1,:2])
            angle = rel_angle(traj_history[-1,:2], neighbor[-1,:2])
            category = get_rel_direction(angle, correction_angle=rel_angle(traj_history[-2,:2], traj_history[-1,:2]))
            if 'left' in category or 'right' in category:
                my_vector, _ = get_unit_vector(traj_history[-2,:2], traj_history[-1,:2])
                category = "on the left" if np.cross(my_vector, neighbor_vector)>0 else "on the right"
            if neighbor_distance <= safe_distance:
                if category == "Directly ahead":
                    risk=True
                elif category == "Ahead":
                    risk=True
                elif category == "on the left" or category == "on the right":
                    if neighbor_distance<=10:
                        risk=True
                else:
                    risk=False
            if risk:
                risk_agents.append(i)  # Neighbor is within the risk zone
                risk_agents_direction.append(category)
                risk_agents_distance.append(neighbor_distance)
            else:
                safe_agents.append(i)  # Neighbor is outside the risk zone
                safe_agents_direction.append(category)
                safe_agents_distance.append(neighbor_distance)
        return safe_distance, risk_agents_direction, risk_agents

    def interpolate_missing_traj(self, traj):
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

    def interpolate_missing_traj_v01(self, traj):
        missing_indices = np.where(traj[:, :2].sum(axis=-1) == 0)[0]
        if len(missing_indices)>traj.shape[0]/2:
            return np.array([])
        if len(missing_indices) > 0:
            for i in range(2):  # Assuming we only need to interpolate the first two columns
                valid_indices = np.where(traj[:, i] != 0)[0]
                valid_values = traj[valid_indices, i]
                interpolator = interp1d(valid_indices, valid_values, fill_value="extrapolate")
                traj[missing_indices, i] = interpolator(missing_indices)
        return traj

    def get_relative_direction(self, reference_traj, other_traj, navigation_extractor=None, same_direction_threshold = 20/180*np.pi, opposite_direction_threshold = (180-20)/180*np.pi, crosswise_threshold_low = (90-20)/180*np.pi, crosswise_threshold_high = (90+20)/180*np.pi,
        directions = []):
        center, angle = reference_traj.copy()[-1][:2], reference_traj.copy()[-1][2]
        normalization_param = [center, angle]
        reference_traj[:,:5] = agent_norm(reference_traj[:,:], center, angle)
        other_traj[:,:5] = agent_norm(other_traj[:,:], center, angle)
        if sum(reference_traj[:,0] != 0)<=1:
            return "Unknown"
        other_velocity = get_vel_from_traj()
        other_velocity = abs_distance_to_velocity(other_traj[:,:2])
        max_speed = max(other_velocity)
        if sum(other_velocity)==0:
            return 'Not moving'
        elif abs(other_heading) <= same_direction_threshold:
            return 'Moving in the same direction'
        elif abs(other_heading) >= opposite_direction_threshold:
            return 'Moving in the opposite direction'
        elif abs(other_heading) >= crosswise_threshold_low and abs(other_heading) <= crosswise_threshold_high:
            if other_heading<0:
                return 'Moving crosswise from the left to the right of me'
            else:
                return 'Moving crosswise from the right to the left of me'
        else:
            return 'Moving with an angle'

    def map_process_original(self, traj):
    
        vectorized_map = np.zeros(shape=(6, 300, 17))
        vectorized_crosswalks = np.zeros(shape=(4, 100, 3))
        agent_type = int(traj[-1][-1])
        lane_polylines = get_polylines(self.lanes)
        road_polylines = get_polylines(self.roads)
        ref_lane_ids = find_reference_lanes(agent_type, traj, lane_polylines)
        ref_lanes = []
        for curr_lane, start in ref_lane_ids.items():
            candidate = depth_first_search(curr_lane, self.lanes, dist=lane_polylines[curr_lane][start:].shape[0], threshold=300)
            ref_lanes.extend(candidate)
        
        if agent_type != 2:
            neighbor_lane_ids = find_neighbor_lanes(ref_lane_ids, traj, self.lanes, lane_polylines)
            for neighbor_lane, start in neighbor_lane_ids.items():
                candidate = depth_first_search(neighbor_lane, self.lanes, dist=lane_polylines[neighbor_lane][start:].shape[0], threshold=300)
                ref_lanes.extend(candidate)
            ref_lane_ids.update(neighbor_lane_ids)
        ref_lanes = remove_overlapping_lane_seq(ref_lanes)
        traffic_light_lanes = {}
        stop_sign_lanes = []

        for signal in self.traffic_signals[self.hist_len-1].lane_states:
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self.lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        for i, sign in self.stop_signs.items():
            stop_sign_lanes.extend(sign.lane)
        added_lanes = 0
        for i, s_lane in enumerate(ref_lanes):
            added_points = 0
            if i > 5:
                break
            cache_lane = np.zeros(shape=(500, 17))

            for lane in s_lane:
                curr_index = ref_lane_ids[lane] if lane in ref_lane_ids else 0
                self_line = lane_polylines[lane][curr_index:]

                if added_points >= 500:
                    break      
                for point in self_line:
                    cache_lane[added_points, 0:3] = point
                    cache_lane[added_points, 10] = self.lanes[lane].type
                    for left_boundary in self.lanes[lane].left_boundaries:
                        left_boundary_id = left_boundary.boundary_feature_id
                        left_start = left_boundary.lane_start_index
                        left_end = left_boundary.lane_end_index
                        left_boundary_type = left_boundary.boundary_type # road line type
                        if left_boundary_type == 0:
                            left_boundary_type = self.roads[left_boundary_id].type + 8 # road edge type
                        
                        if left_start <= curr_index <= left_end:
                            left_boundary_line = road_polylines[left_boundary_id]
                            nearest_point = find_neareast_point(point, left_boundary_line)
                            cache_lane[added_points, 3:6] = nearest_point
                            cache_lane[added_points, 11] = left_boundary_type
                    for right_boundary in self.lanes[lane].right_boundaries:
                        right_boundary_id = right_boundary.boundary_feature_id
                        right_start = right_boundary.lane_start_index
                        right_end = right_boundary.lane_end_index
                        right_boundary_type = right_boundary.boundary_type # road line type
                        if right_boundary_type == 0:
                            right_boundary_type = self.roads[right_boundary_id].type + 8 # road edge type

                        if right_start <= curr_index <= right_end:
                            right_boundary_line = road_polylines[right_boundary_id]
                            nearest_point = find_neareast_point(point, right_boundary_line)
                            cache_lane[added_points, 6:9] = nearest_point
                            cache_lane[added_points, 12] = right_boundary_type
                    cache_lane[added_points, 9] = self.lanes[lane].speed_limit_mph / 2.237
                    cache_lane[added_points, 15] = self.lanes[lane].interpolating
                    if lane in traffic_light_lanes.keys():
                        cache_lane[added_points, 13] = traffic_light_lanes[lane][0]
                        if np.linalg.norm(traffic_light_lanes[lane][1:] - point[:2]) < 3:
                            cache_lane[added_points, 14] = True
                    if lane in stop_sign_lanes:
                        cache_lane[added_points, 16] = True
                    added_points += 1
                    curr_index += 1

                    if added_points >= 500:
                        break             
            vectorized_map[i] = cache_lane[np.linspace(0, added_points, num=300, endpoint=False, dtype=np.int32)]
            added_lanes += 1
        added_cross_walks = 0
        detection = Polygon([(0, -5), (50, -20), (50, 20), (0, 5)])
        detection = affine_transform(detection, [1, 0, 0, 1, traj[-1][0], traj[-1][1]])
        detection = rotate(detection, traj[-1][2], origin=(traj[-1][0], traj[-1][1]), use_radians=True)

        for _, crosswalk in self.crosswalks.items():
            polygon = Polygon([(point.x, point.y) for point in crosswalk.polygon])
            polyline = polygon_completion(crosswalk.polygon)
            polyline = polyline[np.linspace(0, polyline.shape[0], num=100, endpoint=False, dtype=np.int32)]

            if detection.intersects(polygon):
                vectorized_crosswalks[added_cross_walks, :polyline.shape[0]] = polyline
                added_cross_walks += 1
            
            if added_cross_walks >= 4:
                break

        return vectorized_map.astype(np.float32), vectorized_crosswalks.astype(np.float32)

    def map_process(self, traj):
      
        max_num_points = 500
        vectorized_map = np.zeros(shape=(self.drivable_lanes, max_num_points, 17))
        
        additional_boundaries_ = np.zeros(shape=(self.drivable_lanes, max_num_points, 4))
        vectorized_crosswalks = np.zeros(shape=(self.num_crosswalks, 100, 3))
        vectorized_traffic_lights = np.zeros(shape=(32, 3))
        vectorized_stop_signs = np.zeros(shape=(16, 2))
        vectorized_speed_bumps = np.zeros(shape=(self.num_crosswalks, 100, 3))
        agent_type = int(traj[-1][-1])
        lane_polylines = get_polylines(self.lanes)
        road_polylines = get_polylines(self.roads)
        ref_lane_ids = find_all_lanes(agent_type, traj, lane_polylines)
        ref_lanes = ref_lane_ids
        
        if agent_type != 2:
            neighbor_lane_ids = find_neighbor_lanes_fullmap(ref_lane_ids, traj, self.lanes, lane_polylines)
            for neighbor_lane, start in neighbor_lane_ids.items():
                candidate = depth_first_search(neighbor_lane, self.lanes, dist=lane_polylines[neighbor_lane][start:].shape[0], threshold=300)
                ref_lanes.extend(candidate)
            ref_lane_ids.update(neighbor_lane_ids)
        traffic_light_lanes = {}
        traffic_light_points = []
        stop_sign_lanes = []

        for signal in self.traffic_signals[self.hist_len-1].lane_states:
            traffic_light_points.append([signal.stop_point.x, signal.stop_point.y, signal.state])
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self.lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        for i, sign in self.stop_signs.items():
            stop_sign_lanes.extend(sign.lane)
        added_lanes = 0
        
        for i, s_lane in enumerate(ref_lanes):

            added_points = 0
            if i > (self.drivable_lanes-1): # DONE: make this 5 (that mean 6 drivable lanes variable)
                break
            cache_lane = np.zeros(shape=(max_num_points, 17))
            if True:
                lane = s_lane
                curr_index = ref_lane_ids[lane] if lane in ref_lane_ids else 0
                self_line = lane_polylines[lane][curr_index:]
                if added_points >= max_num_points:
                    break
                for point in self_line:
                    cache_lane[added_points, 0:3] = point
                    cache_lane[added_points, 10] = self.lanes[lane].type

            
                    if len(self.lanes[lane].left_boundaries)>0:
                        i_left_boundary, left_boundary = 0, self.lanes[lane].left_boundaries[0]
                        left_boundary_id = left_boundary.boundary_feature_id
                        left_start = left_boundary.lane_start_index
                        left_end = left_boundary.lane_end_index
                        left_boundary_type = left_boundary.boundary_type # road line type
                        if left_boundary_type == 0:
                            left_boundary_type = self.roads[left_boundary_id].type + 8 # road edge type
                        left_boundary_line = road_polylines[left_boundary_id]
                        nearest_point = find_neareast_point(point, left_boundary_line)
                        if i_left_boundary==0:
                            cache_lane[added_points, 3:6] = nearest_point
                            cache_lane[added_points, 11] = left_boundary_type
                
                    if len(self.lanes[lane].right_boundaries)>0:
                        i_right_boundary, right_boundary = 0, self.lanes[lane].right_boundaries[0]
                        right_boundary_id = right_boundary.boundary_feature_id
                        right_start = right_boundary.lane_start_index
                        right_end = right_boundary.lane_end_index
                        right_boundary_type = right_boundary.boundary_type # road line type
                        if right_boundary_type == 0:
                            right_boundary_type = self.roads[right_boundary_id].type + 8 # road edge type
                        right_boundary_line = road_polylines[right_boundary_id]
                        nearest_point = find_neareast_point(point, right_boundary_line)
                        if i_right_boundary==0:
                            cache_lane[added_points, 6:9] = nearest_point
                            cache_lane[added_points, 12] = right_boundary_type
                    cache_lane[added_points, 9] = self.lanes[lane].speed_limit_mph / 2.237
                    cache_lane[added_points, 15] = self.lanes[lane].interpolating
                    if lane in traffic_light_lanes.keys():
                        cache_lane[added_points, 13] = traffic_light_lanes[lane][0]
                        if np.linalg.norm(traffic_light_lanes[lane][1:] - point[:2]) < 3:
                            cache_lane[added_points, 14] = True
                    if lane in stop_sign_lanes:
                        cache_lane[added_points, 16] = True
                    added_points += 1
                    curr_index += 1

                    if added_points >= max_num_points:
                        break             

    
            vectorized_map[i] = cache_lane
            added_lanes += 1
        added_cross_walks = 0
        added_speed_bumps = 0
        detection = Polygon([(0, -5), (50, -20), (50, 20), (0, 5)])
        detection = affine_transform(detection, [1, 0, 0, 1, traj[-1][0], traj[-1][1]])
        detection = rotate(detection, traj[-1][2], origin=(traj[-1][0], traj[-1][1]), use_radians=True)

        for _, crosswalk in self.crosswalks.items():
            polygon = Polygon([(point.x, point.y) for point in crosswalk.polygon])
            polyline = polygon_completion(crosswalk.polygon)
            polyline = polyline[np.linspace(0, polyline.shape[0], num=100, endpoint=False, dtype=np.int32)]
            if True:
                vectorized_crosswalks[added_cross_walks, :polyline.shape[0]] = polyline
                added_cross_walks += 1
            
            if added_cross_walks >= self.num_crosswalks:
                break
        
        for _, speed_bump in self.speed_bumps.items():
            polygon = Polygon([(point.x, point.y) for point in speed_bump.polygon])
            polyline = polygon_completion(speed_bump.polygon)
            polyline = polyline[np.linspace(0, polyline.shape[0], num=100, endpoint=False, dtype=np.int32)]
            if True:
                vectorized_speed_bumps[added_speed_bumps, :polyline.shape[0]] = polyline
                added_speed_bumps += 1
            
            if added_speed_bumps >= len(vectorized_speed_bumps):
                break
        missing_boundaries_ids = list(self.roads.keys()) # all boundaries
        if len(missing_boundaries_ids)>0:
            for added_roads, boundary_id in enumerate(missing_boundaries_ids):
                if boundary_id in list(self.roads_boundary.keys()):
                    boundary_type = self.roads[boundary_id].type # road boundary type
                else:
                    boundary_type = self.roads[boundary_id].type + 8 # road edge type
                additional_boundaries_[added_roads, :min(additional_boundaries_.shape[1], road_polylines[boundary_id].shape[0]), :3] = road_polylines[boundary_id][:min(additional_boundaries_.shape[1], road_polylines[boundary_id].shape[0])]
                additional_boundaries_[added_roads, :min(additional_boundaries_.shape[1], road_polylines[boundary_id].shape[0]), 3] = boundary_type
        boundaries_min_distance = np.linalg.norm(traj[-1, :2] - additional_boundaries_[...,:2], axis=-1).min(-1)
        additional_boundaries_ = additional_boundaries_[np.argsort(boundaries_min_distance)]
       
        
        if sum(vectorized_crosswalks[...,0].sum(-1)!=0)>0:
            vectorized_crosswalks_min_distance = np.linalg.norm(traj[-1, :2] - vectorized_crosswalks[...,:2], axis=-1).min(-1)
            vectorized_crosswalks = vectorized_crosswalks[np.argsort(vectorized_crosswalks_min_distance)]
        
        if len(traffic_light_points)>0:
            traffic_light_points = np.array(traffic_light_points)
            traffic_light_points[traffic_light_points[:,-1]==0] = [0,0,0]
            if traffic_light_points[:,-1].sum()>0:
                traffic_light_distance = [np.linalg.norm(traj[-1, :2] - traffic_light_points[i][:2]) for i in range(len(traffic_light_points))]
                traffic_light_sort_idx = np.argsort(traffic_light_distance)
                sorted_traffic_lights = traffic_light_points[traffic_light_sort_idx]
                vectorized_traffic_lights[:min(len(vectorized_traffic_lights), len(sorted_traffic_lights))] = sorted_traffic_lights[:min(len(vectorized_traffic_lights), len(sorted_traffic_lights))]
        
        if len(self.stop_signs)>0:
            stop_signs = np.vstack([[v.position.x, v.position.y] for v in self.stop_signs.values()])
            argsort_stop_signs = np.argsort([np.linalg.norm(traj[-1, :2] - stop_signs[i,:2]) for i in range(len(stop_signs))])
            vectorized_stop_signs[:min(len(vectorized_stop_signs), len(stop_signs))] = stop_signs[argsort_stop_signs][:min(len(vectorized_stop_signs), len(stop_signs))]
        
        if sum(vectorized_speed_bumps[...,0].sum(-1)!=0)>0:
            vectorized_speed_bumps_min_distance = np.linalg.norm(traj[-1, :2] - vectorized_speed_bumps[...,:2], axis=-1).min(-1)
            vectorized_speed_bumps[vectorized_speed_bumps_min_distance>311] = 0.0
            vectorized_speed_bumps = vectorized_speed_bumps[np.argsort(vectorized_speed_bumps_min_distance)]
            
            vectorized_speed_bumps[...,0].sum(-1)==0
            vectorized_speed_bumps[vectorized_speed_bumps[...,0].sum(-1)==0]
            vectorized_speed_bumps[vectorized_speed_bumps[...,0].sum(-1)==0]=[0,0,0]
        return vectorized_map.astype(np.float32), vectorized_crosswalks.astype(np.float32), additional_boundaries_.astype(np.float32), vectorized_traffic_lights.astype(np.float32), vectorized_stop_signs.astype(np.float32), vectorized_speed_bumps.astype(np.float32)

    def get_direction_plausibility_vectorized(self, ego, map_lanes, map_crosswalks, ground_truth, additional_boundaries_, traffic_lights, stop_signs, agent_json_):
        ego_position = ego[0, :, :2]
        lane_positions = map_lanes[:, :, :2]
        
        bike_lanes_mask = map_lanes[:,:,10]!=3
        undefined_lanes_mask = map_lanes[:,:,10]!=0
        valid_lane_mask = bike_lanes_mask * undefined_lanes_mask
        lane_positions[~valid_lane_mask] = 10000
        
        distances = np.linalg.norm(lane_positions, axis=2)
        min_distances_indices = np.argmin(distances, axis=1)
        valid_lanes = distances[np.arange(distances.shape[0]), min_distances_indices] < 1  # 1 meter threshold
        angles = np.degrees(map_lanes[np.arange(map_lanes.shape[0]), min_distances_indices, 2])
        valid_angles = np.abs(angles) < 15  # 15 degrees threshold

        valid_lanes = np.logical_and(valid_lanes, valid_angles)
        possible_map_lanes = map_lanes[valid_lanes]
        direction_classifier = DirectionClassifier(num_classes=5)
        possible_directions = []
        for lane in possible_map_lanes:
            start_point = lane[0]
            end_point = lane[-1]
            two_points_lane = np.vstack((start_point, end_point))
            direction_str, direction_cls, _, _ = direction_classifier.classify_lane(two_points_lane)
            possible_directions.append(direction_classifier.classes[direction_cls])
        
        return np.unique(possible_directions)

    def get_direction_plausibility(self, ego, map_lanes, map_crosswalks, ground_truth, additional_boundaries_, traffic_lights, stop_signs, agent_json_):
        
        ego = ego[0,:,:]
        current_location_on_map = map_lanes
        bike_lanes_mask = map_lanes[:,:,10]!=3
        undefined_lanes_mask = map_lanes[:,:,10]!=0
        valid_lane_mask = bike_lanes_mask * undefined_lanes_mask
        
        closest_current_point_arg = np.linalg.norm(map_lanes[valid_lane_mask][:,:2], axis=1).argmin()
        reference_slice = map_lanes[valid_lane_mask][closest_current_point_arg][:3]
        matches = np.any(map_lanes[:, :, :3] == reference_slice, axis=(1,2))
        current_lane_idx = np.where(matches)[0][0] if np.any(matches) else None
        matches = np.all(map_lanes[current_lane_idx, :, :3] == reference_slice, axis=1)
        current_point_idx = np.where(matches)[0][0] if np.any(matches) else None

        possible_map_lanes = []
        if current_lane_idx is not None and current_lane_idx is not None:
            fill_in_lane = map_lanes[current_lane_idx, current_point_idx:]
            fill_in_mask = valid_lane_mask[current_lane_idx, current_point_idx:]
            possible_map_lanes.append(fill_in_lane[fill_in_mask])
        for i in range(len(map_lanes)):
            if i==current_lane_idx:
                continue
            map_lanes[i][valid_lane_mask[i]]
            array2 = map_lanes[i][valid_lane_mask[i]][:,:3]
            if len(array2)==0:
                continue
            possible_map_lanes_cache = []
            
            for j in possible_map_lanes:
                array1 = j[:,:3]
                distances = np.sqrt(((array1[:,:2][:, np.newaxis] - array2[:,:2])**2).sum(axis=2))
                min_index = np.unravel_index(np.argmin(distances), distances.shape)
                closest_points = (array1[min_index[0]], array2[min_index[1]])
                minimum_distance = distances[min_index]
                if minimum_distance<1: # if 1 meter between two points
                    angle_difference = np.degrees(array2[min_index[1], 2] - array1[min_index[0], 2])
                    if angle_difference<15:
                        fill_in_lane = map_lanes[i, min_index[1]:]
                        fill_in_mask = valid_lane_mask[i, min_index[1]:]
                        possible_map_lanes_cache.append(fill_in_lane[fill_in_mask])
            for cache_lane in possible_map_lanes_cache:
                possible_map_lanes.append(cache_lane)
        max_length = max([len(arr) for arr in possible_map_lanes])
        padded_arrays = [
            np.pad(arr, pad_width=((0, max_length - arr.shape[0]), (0, 0)), mode='constant', constant_values=0)
            for arr in possible_map_lanes
        ]
        possible_map_lanes = np.stack(padded_arrays)
        possible_map_lanes_mask = possible_map_lanes[...,0]!=0

        
        current_speed = np.linalg.norm(ego[-1,3:5]) # m/s
        starting_point = possible_map_lanes[0][possible_map_lanes_mask[0]][0,:3] # x,y,h
        max_possible_speed_increase = 2.5 # m/s maximum possible increase in speed -> 72 km/h/8s
        maximum_future_speeds = [current_speed+(i*max_possible_speed_increase) for i in range(8)] # m/s for 8 the current speed and 7 seconds in the future speeds
        max_travel_distance = sum(maximum_future_speeds)
        max_possible_speed_decrease = max_possible_speed_increase # m/s
        min_future_speeds = [current_speed-(i*max_possible_speed_decrease) for i in range(8) if current_speed-(i*max_possible_speed_decrease)>0] # m/s for 8 the current speed and 7 seconds in the future speeds
        min_travel_distance = sum(min_future_speeds)
        starting_point = possible_map_lanes[0][possible_map_lanes_mask[0]][0,:3] # x,y,h
        map_points_distances = np.linalg.norm(starting_point[:2] - possible_map_lanes[possible_map_lanes_mask][:,:2], axis=1)
        reachable_points_mask = (map_points_distances < max_travel_distance) * (map_points_distances > min_travel_distance)
        possible_map_lanes_mask[possible_map_lanes_mask] = reachable_points_mask * possible_map_lanes_mask[possible_map_lanes_mask]
        direction_classifier = DirectionClassifier(step_t=1, num_classes=5)
        possible_directions = []
        found_lanes = 0
        for i in range(len(possible_map_lanes)):
            if len(possible_map_lanes[i][possible_map_lanes_mask[i]])>0:
                two_points_lane = np.vstack((starting_point, possible_map_lanes[i][possible_map_lanes_mask[i]][-1,:3]))
                direction_str, direction_cls,_,_ = direction_classifier.classify_lane(two_points_lane)
                possible_directions.append(direction_classifier.classes[direction_cls])
                found_lanes+=1
       
        possible_directions = np.unique(possible_directions)
        return possible_directions

    def get_direction_plausibility_03(self, ego_lane, drivable_lanes, ego, map_lanes=None, map_crosswalks=None, ground_truth=None, additional_boundaries_=None, traffic_lights=None, stop_signs=None, agent_json_=None):
        possible_map_lanes = drivable_lanes
        possible_map_lanes[possible_map_lanes[...,0]==0] = np.inf
        ego = ego[0,:,:]
       
        map_lanes_norms = np.linalg.norm(ego_lane[0,...,:2], axis=-1)
        min_index_flat = np.argmin(map_lanes_norms) # lane is already 1D
        reference_slice = ego_lane[0][min_index_flat]
        
        
        if map_lanes_norms[min_index_flat]<1:
            starting_point = reference_slice[:3]# x,y,h
            ego_center_lane = ego_lane[0][min_index_flat:]
        else:
            starting_point = ego[-1, :3] ## 090225: change starting point to ego vehicle position and angle not the closest lane point
            ego_center_lane = ego[-1, :3][None]

        
        direction_classifier = DirectionClassifier(step_t=1, num_classes=5)
        possible_directions = []
        possible_classes = []
        found_lanes = 0
        lines_to_plot_tupple = {}
        possible_colors = ['g', 'b', 'c', 'm', 'yellow', 'purple', 'pink', 'orange']
        for i in range(len(possible_map_lanes)):
            if np.any(possible_map_lanes[i][:,0]!=np.inf):
                target_lane = possible_map_lanes[i][possible_map_lanes[i][:, 0] != np.inf]
                drivable_target_lane = target_lane # The lanes already cropped based on min and max distance
                if len(drivable_target_lane)>4:
                    two_points_lane_maximum_distance = np.vstack((starting_point, drivable_target_lane[-1,:3]))
                    two_points_lane_halfway = np.vstack((starting_point, drivable_target_lane[len(drivable_target_lane)//2,:3]))
                    two_points_lane_quarterway = np.vstack((starting_point, drivable_target_lane[len(drivable_target_lane)//4,:3]))
                    two_points_lane_shortest_path = np.vstack((starting_point, drivable_target_lane[0,:3]))
                    four_splits_directions = []
                    for two_points_lane in [two_points_lane_shortest_path, two_points_lane_quarterway, two_points_lane_halfway, two_points_lane_maximum_distance]:    
                        direction_str, direction_cls,_,_ = direction_classifier.classify_lane(deepcopy(two_points_lane))
                        four_splits_directions.append((direction_str, direction_cls))
                    four_splits_directions = remove_entries_after_turn(four_splits_directions) 
                    four_splits_directions = remove_uturn_after_opposite_turn(four_splits_directions) 
                    
                    for direction_str, direction_cls in four_splits_directions:
                        if direction_cls!=-1:
                            starting_point_idx = np.linalg.norm((drivable_target_lane[:,:3] - starting_point)[:,:2], axis=-1).argmin()
                            lines_to_plot_tupple[found_lanes] = (direction_cls, drivable_target_lane[starting_point_idx:])
                            possible_directions.append(direction_classifier.classes[direction_cls])
                            possible_classes.append(direction_cls)
                            found_lanes+=1

                     


        return np.unique(possible_directions), lines_to_plot_tupple, possible_map_lanes

    
    def ego_process(self, sdc_ids, tracks):
        ego_states = np.zeros(shape=(2, self.hist_len, 11))
        self.current_xyzh = []
        for s,sdc_id in enumerate(sdc_ids):
            sdc_states = tracks[sdc_id].states[:self.hist_len]
            ego_type = tracks[sdc_id].object_type
            self.ego_type = ego_type
            if ego_type<=0 or ego_type>3:
                one_hot_type = [0,0,0]
            else:
                one_hot_type = np.eye(3)[int(ego_type)-1]
            self.current_xyzh.append( (tracks[sdc_id].states[self.hist_len-1].center_x, tracks[sdc_id].states[self.hist_len-1].center_y, 
                                tracks[sdc_id].states[self.hist_len-1].center_z, tracks[sdc_id].states[self.hist_len-1].heading) )
            for i, sdc_state in enumerate(sdc_states):
                if sdc_state.valid:
                    ego_state = np.array([sdc_state.center_x, sdc_state.center_y, sdc_state.heading, sdc_state.velocity_x, 
                                        sdc_state.velocity_y, sdc_state.length, sdc_state.width, sdc_state.height, 
                                        one_hot_type[0],one_hot_type[1],one_hot_type[2]])
                    ego_states[s,i] = ego_state

        return ego_states.astype(np.float32)

    def neighbors_process(self, sdc_ids, tracks):
        neighbors_states = np.zeros(shape=(self.num_neighbors, self.hist_len, 11))
        neighbors = []
        self.neighbors_id = []
        self.neighbors_type = []
        for e in range(len(sdc_ids)):
            for i, track in enumerate(tracks):
                track_states = track.states[:self.hist_len]
                if i not in sdc_ids and track_states[-1].valid:
                    xy = np.stack([track_states[-1].center_x, track_states[-1].center_y], axis=-1)
                    neighbors.append((i, np.linalg.norm(xy - self.current_xyzh[e][:2]))) 
        sorted_neighbors = sorted(neighbors, key=lambda item: item[1])
        added_num = 0
        appended_ids = set()
        for neighbor in sorted_neighbors:
            neighbor_id = neighbor[0]
            if neighbor_id in appended_ids:
                continue
            appended_ids.add(neighbor_id)

            neighbor_states = tracks[neighbor_id].states[:self.hist_len]
            neighbor_type = tracks[neighbor_id].object_type
            self.neighbors_type.append(neighbor_type)
            if neighbor_type<=0 or neighbor_type>3:
                one_hot_type = [0,0,0]
            else:
                one_hot_type = np.eye(3)[int(neighbor_type)-1]
            self.neighbors_id.append(neighbor_id)

            for i, neighbor_state in enumerate(neighbor_states):
                if neighbor_state.valid: 
                    neighbors_states[added_num, i] = np.array([neighbor_state.center_x, neighbor_state.center_y, neighbor_state.heading,  neighbor_state.velocity_x, 
                                                               neighbor_state.velocity_y, neighbor_state.length, neighbor_state.width, neighbor_state.height, 
                                                               one_hot_type[0],one_hot_type[1],one_hot_type[2]])
            added_num += 1
            if added_num >= self.num_neighbors:
                break

        return neighbors_states.astype(np.float32), self.neighbors_id

    def ground_truth_process(self, sdc_ids, tracks):
        ground_truth = np.zeros(shape=(2, self.future_len, 5))
        
        for j, sdc_id in enumerate(sdc_ids):
            track_states = tracks[sdc_id].states[self.hist_len:]
            for i, track_state in enumerate(track_states):
                ground_truth[j, i] = np.stack([track_state.center_x, track_state.center_y, track_state.heading, 
                                            track_state.velocity_x, track_state.velocity_y], axis=-1)

        return ground_truth.astype(np.float32)
    
    def get_static_region(self,ego):
        region_dict = {}
        for c in [6,32,64]:
            region = []
            for i,n_type in enumerate(self.object_type):
                if n_type==2:
                    obj_type = 'pedestrian'
                elif n_type==3:
                    obj_type = 'cyclist'
                else:
                    obj_type = 'vehicle'
                data = self.points_dict[f'{obj_type}_{c}']
                if i==0:
                    region.append(data)
                    continue
                x,y = data[:,0], data[:,1]
                p_x,p_y,theta = ego[i,-1,0],ego[i,-1,1],ego[i,-1,2]
                new_x = np.cos(-theta)*x + np.sin(-theta)*y + p_x
                new_y = -np.sin(-theta)*x + np.cos(-theta)*y + p_y
                region.append(np.stack([new_x,new_y],axis=1))
            region_dict[c] = np.array(region,dtype=np.float32)
        return region_dict

    def reverse_normalize_traj(self, ego, neighbors, ground_truth, normalization_param):
        center, angle = normalization_param
        if ego is not None:
            ego[0, :, :5] = reverse_agent_norm(ego[0], center, angle)
            ego[1, :, :5] = reverse_agent_norm(ego[1], center, angle)
        if ground_truth is not None:
            ground_truth[0] = reverse_agent_norm(ground_truth[0], center, angle)
            ground_truth[1] = reverse_agent_norm(ground_truth[1], center, angle)

        if neighbors is not None:
            for i in range(neighbors.shape[0]):
                if neighbors[i, -1, 0] != 0:
                    neighbors[i, :, :5] = reverse_agent_norm(neighbors[i], center, angle)

        return ego, neighbors, ground_truth

    def interpolate_missing_data(self, ego, ground_truth, neighbors, test_data=False):
        not_valid = False
        future_start_end_with_zeros = False
        threshold_1 = 5 if not test_data else 0 # 5/11
        threshold_2 = 50 if not test_data else 0 # 50/80
        if sum(ego[0,:,0]==0)>threshold_1 or sum(ego[1,:,0]==0)>threshold_1:
       
            not_valid = True 
        elif sum(ground_truth[0,:,0]==0)>threshold_2 or sum(ground_truth[1,:,0]==0)>threshold_2:
        
            not_valid = True 
        else:
            for i in range(len(ego)):
                if np.any(ego[i,:,0]==0):
                    start_non_zero = np.where(ego[i][:,0]!=0)[0][0]
                    end_non_zero = np.where(ego[i][:,0]!=0)[0][-1]
                    ego[i, start_non_zero:end_non_zero] = self.interpolate_missing_traj(ego[i, start_non_zero:end_non_zero])
                if np.any(ground_truth[i,:,0]==0):
                    start_non_zero = np.where(ground_truth[i][:,0]!=0)[0][0]
                    end_non_zero = np.where(ground_truth[i][:,0]!=0)[0][-1]
                    if start_non_zero > 0 or end_non_zero < len(ground_truth[i])-1:
                        future_start_end_with_zeros = True
                    ground_truth[i, start_non_zero:end_non_zero] = self.interpolate_missing_traj(ground_truth[i, start_non_zero:end_non_zero])
            for i in range(len(neighbors)):
                if np.any(neighbors[i,:,0]==0) and sum(neighbors[i,:,0]!=0)>2:
                    start_non_zero = np.where(neighbors[i][:,0]!=0)[0][0]
                    end_non_zero = np.where(neighbors[i][:,0]!=0)[0][-1]
                    neighbors_interpolated = self.interpolate_missing_traj(neighbors[i, start_non_zero:end_non_zero])
                    if len(neighbors_interpolated)==0:
                        continue
                    neighbors[i, start_non_zero:end_non_zero] = neighbors_interpolated
        return ego, ground_truth, neighbors, not_valid

    def normalize_map_points(self, lane_data, center, angle):
        for k in range(lane_data.shape[0]):
            lane_data_ = lane_data[k]
            if lane_data_[0][0] != 0:
                lane_data_[:, :3] = map_norm(lane_data_, center, angle)
        return lane_data

    def normalize_data_original(self, ego, neighbors, map_lanes, map_crosswalks, ground_truth, viz=False):
        center, angle = ego[0].copy()[-1][:2], ego[0].copy()[-1][2]
        ego[0, :, :5] = agent_norm(ego[0], center, angle, impute=True)
        ego[1, :, :5] = agent_norm(ego[1], center, angle, impute=True)

        ground_truth[0] = agent_norm(ground_truth[0], center, angle) 
        ground_truth[1] = agent_norm(ground_truth[1], center, angle) 

        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                neighbors[i, :, :5] = agent_norm(neighbors[i], center, angle, impute=True) 

        if self.point_dir != '':
            region_dict = self.get_static_region(ego) 
        else:
            region_dict = None      
        for i in range(map_lanes.shape[0]):
            lanes = map_lanes[i]
            crosswalks = map_crosswalks[i]

            for j in range(map_lanes.shape[1]):
                lane = lanes[j]
                if lane[0][0] != 0:
                    lane[:, :9] = map_norm(lane, center, angle)

            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk[:, :3] = map_norm(crosswalk, center, angle)

        return ego, neighbors, map_lanes, map_crosswalks, ground_truth,region_dict

    def normalize_data(self, ego, neighbors, map_lanes, map_crosswalks, ground_truth, map_speed_bumps, viz=False):
        for i in range(len(ego)):
            speeds_01 = np.linalg.norm(abs_distance_to_velocity(ego[i,:,:2][ego[i,:,0]!=0]),axis=-1)*10*3.6
            speeds_02 = np.linalg.norm(abs_distance_to_velocity(ground_truth[i,:,:2][ground_truth[i,:,0]!=0]),axis=-1)*10*3.6
            if len(speeds_01)==0 or len(speeds_02)==0:
                return [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]
            max_speed = max(max(speeds_01), max(speeds_02))
            if max_speed>130: # 130 kmh ~= 80 mph
                return [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]
        center, angle = ego[0].copy()[-1][:2], ego[0].copy()[-1][2]
        normalization_param = [center, angle]
        ego[0, :, :5] = agent_norm(ego[0], center, angle, impute=True)
        ego[1, :, :5] = agent_norm(ego[1], center, angle, impute=True)

        ground_truth[0] = agent_norm(ground_truth[0], center, angle)
        ground_truth[1] = agent_norm(ground_truth[1], center, angle)

        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                neighbors[i, :, :5] = agent_norm(neighbors[i], center, angle, impute=True) 

        if self.point_dir != '':
            region_dict = self.get_static_region(ego)
        else:
            region_dict = None
        for i in range(map_lanes.shape[0]):
            lanes = map_lanes[i]
            crosswalks = map_crosswalks[i]

            for j in range(map_lanes.shape[1]):
                lane = lanes[j]
                if lane[0][0] != 0:
                    lane[:, :9] = map_norm(lane, center, angle)
            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk[:, :3] = map_norm(crosswalk, center, angle)
            
            speed_bumps = map_speed_bumps[i]
            for k in range(map_speed_bumps.shape[1]):
                speed_bump = speed_bumps[k]
                if speed_bump[0][0] != 0:
                    speed_bump[:, :3] = map_norm(speed_bump, center, angle)
        if viz:
            for i in range(ego.shape[0]):
                rect = plt.Rectangle((ego[i,-1, 0]-ego[i,-1, 5]/2, ego[i,-1, 1]-ego[i,-1, 6]/2), ego[i,-1, 5], ego[i,-1, 6], linewidth=2, color='r', alpha=0.6, zorder=3,
                                    transform=mpl.transforms.Affine2D().rotate_around(*(ego[i,-1, 0], ego[i,-1, 1]), ego[i,-1, 2]) + plt.gca().transData)
                plt.gca().add_patch(rect)

                future = ground_truth[i][ground_truth[i][:, 0] != 0]
                plt.plot(future[:, 0], future[:, 1], 'r', linewidth=1, zorder=3)
            
            for i in range(neighbors.shape[0]):
                if neighbors[i, -1, 0] != 0:
                    rect = plt.Rectangle((neighbors[i, -1, 0]-neighbors[i, -1, 5]/2, neighbors[i, -1, 1]-neighbors[i, -1, 6]/2), 
                                          neighbors[i, -1, 5], neighbors[i, -1, 6], linewidth=1.5, color='m', alpha=0.6, zorder=3,
                                          transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, -1, 0], neighbors[i, -1, 1]), neighbors[i, -1, 2]) + plt.gca().transData)
                    plt.gca().add_patch(rect)
            
            for i in range(map_lanes.shape[0]):
                lanes = map_lanes[i]
                crosswalks = map_crosswalks[i]

                for j in range(map_lanes.shape[1]):
                    lane = lanes[j]
                    if lane[0][0] != 0:
                        centerline = lane[:, 0:2]
                        centerline = centerline[centerline[:, 0] != 0]
                        left = lane[:, 3:5]
                        left = left[left[:, 0] != 0]
                        right = lane[:, 6:8]
                        right = right[right[:, 0] != 0]
                        plt.plot(centerline[:, 0], centerline[:, 1], 'k', linewidth=0.5) # plot centerline

            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk = crosswalk[crosswalk[:, 0] != 0]
                    plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=1) # plot crosswalk
            
            if self.point_dir != '':
                for i in range(region_dict[32].shape[0]):
                    plt.scatter(region_dict[32][i,:,0],region_dict[32][i,:,1],marker='*',s=10)

            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.close()
        
        return ego, neighbors, map_lanes, map_crosswalks, ground_truth,region_dict, normalization_param, map_speed_bumps
    
    def interactive_process(self,tracks_list,interesting_ids,tracks):
        self.sdc_ids_list = []

        for ego_id in tracks_list:

            ego_state = tracks[ego_id].states[self.hist_len-1]
            ego_xy = np.array([ego_state.center_x, ego_state.center_y])

            candidate_tracks = []
            cnt = 2
            if len(tracks_list)==1:
                for i, track in enumerate(tracks):
                    track_states = track.states[self.hist_len-1]
                    if i != ego_id and track_states.valid:
                        tracks_xy = np.array([track_states.center_x, track_states.center_y])
                        candidate_tracks.append((i, np.linalg.norm(tracks_xy - ego_xy)))
            else:
                for t in tracks_list:
                    if t!=ego_id:
                        if t in interesting_ids and ego_id in interesting_ids:
                            self.sdc_ids_list.append(((ego_id, t), 1))
                            cnt -= 1
                            continue

                        track_states = tracks[t].states[self.hist_len-1]
                        tracks_xy = np.array([track_states.center_x, track_states.center_y])
                        candidate_tracks.append((t, np.linalg.norm(tracks_xy - ego_xy)))
            sorted_candidate = sorted(candidate_tracks, key=lambda item: item[1])[:cnt]

            for can in sorted_candidate:
                self.sdc_ids_list.append(((ego_id, can[0]), 0))

    def get_drivable_lanes(self, ego, use_neighbouring_lanes=True):
        max_num_neighboring_lanes = 0 ## Lanes that connecect to the ego lane
        dist_threshold, angle_threshold = 1, 10 # 1m, 5 degrees
        ego_state = ego[-1, :3]
        ego_state_velocity = np.linalg.norm(ego[-1, 3:5]) # in m/s
        ego_coords = ego_state[:2]
        ego_angle = ego_state[2]
        lanes_polylines = deepcopy(get_polylines(self.lanes))
        lanes_polylines_rel_angles = [np.abs(np.degrees(ego_angle-polyline_v[:,2])) for polyline_v in lanes_polylines.values()]
        angle_mask = [angle_list>angle_threshold for angle_list in lanes_polylines_rel_angles]
        for polyline_v, angle_mask_i in zip(lanes_polylines.values(), angle_mask):
            polyline_v[angle_mask_i] = np.inf
        sorted_lanes = dict(
            sorted(
                self.lanes.items(),  # Sort the items in self.lanes
                key=lambda item: get_distance_between_map_object_and_point(ego_coords, lanes_polylines[item[0]][...,:2])  # Use the minimum value from lanes_polylines as the sorting key
            )
        )
        lanes_polylines = deepcopy(get_polylines(self.lanes))
        sorted_lanes_polylines = {k:lanes_polylines[k] for k in sorted_lanes.keys()}
        ego_lane_id = list(sorted_lanes.keys())[0]
        ego_lane = sorted_lanes[ego_lane_id]
        ego_lane_xyh = sorted_lanes_polylines[ego_lane_id]
        current_speed = ego_state_velocity # m/s
        if current_speed<=18: # since stationary indicate the vehicle does not exceed 5 meters in 8 seconds, we assume this is the max possible speed that the vehicle can break from to be a safe yet firm breaking
            vehicle_can_stop = True
        else:
            vehicle_can_stop = False

        max_possible_speed_increase = 0.52 # m/s² -> 15 km/h over 8s
        speed_limit = ego_lane.speed_limit_mph # mph
        speed_limit = speed_limit * 0.44704 # meter-per-second
        maximum_future_speeds = [min(current_speed+(i*max_possible_speed_increase),speed_limit)  for i in range(8)] # m/s for 8 the current speed and 7 seconds in the future speeds
        max_travel_distance = sum(maximum_future_speeds)
        max_possible_speed_decrease = 2.5 # m/s
        min_future_speeds = [current_speed-(i*max_possible_speed_decrease) for i in range(8) if current_speed-(i*max_possible_speed_decrease)>0] # m/s for 8 the current speed and 7 seconds in the future speeds
        min_travel_distance = sum(min_future_speeds)
        max_distance = min(min(speed_limit*8, max_travel_distance), 60) # max 60 meters
        min_distance = max(0, min_travel_distance)
        if max_distance<=min_distance:
            max_distance = min_distance+20
        lanes_polylines = deepcopy(get_polylines(self.lanes))
        for k, lane in lanes_polylines.items():
            distances = np.linalg.norm(lane[:, :2] - ego_coords, axis=-1)
            exceed_indices = np.where(distances > max_distance)[0]
            if len(exceed_indices) > 0:
                first_exceed_idx = exceed_indices[0]  # Get the first index where distance exceeds max_distance
                lanes_polylines[k] = lane[:first_exceed_idx]  # Crop the polyline
        
        sorted_lanes_polylines = {k:lanes_polylines[k] for k in sorted_lanes.keys()}
        ego_lane_xyh = sorted_lanes_polylines[ego_lane_id]
        if len(ego_lane_xyh)==0:
            for retry in range(5):
                ego_lane_id = list(sorted_lanes.keys())[retry+1]
                ego_lane = sorted_lanes[ego_lane_id]
                ego_lane_xyh = sorted_lanes_polylines[ego_lane_id]
                if len(ego_lane_xyh)!=0:
                    break
        if use_neighbouring_lanes: 
            additional_ego_lanes = {} # to current position of the ego vehicle
            ego_lane_threshold = get_distance_between_map_object_and_point(ego_coords, sorted_lanes_polylines[ego_lane_id]) ## the distance to the current lane + 0.5m buffer
            filtered_lanes = {
                key: value
                for key, value in sorted_lanes_polylines.items()
                if key!= ego_lane_id and len(sorted_lanes_polylines[key])>0 and get_distance_between_map_object_and_point(ego_coords, sorted_lanes_polylines[key][..., :2]) <= ego_lane_threshold and min(np.abs(np.degrees(ego_angle-sorted_lanes_polylines[key][...,-1]))) <= angle_threshold
            }
            if len(filtered_lanes)>0:
                for lane_id in filtered_lanes:
                    object_i_xyh = filtered_lanes[lane_id]
                    if lane_id not in additional_ego_lanes:
                        additional_ego_lanes.update({lane_id:sorted_lanes[lane_id]})
            additional_lanes = {} # neighboring lanes to the ego lane, not to the starting position
            for lane_id in sorted_lanes_polylines:
                if max_num_neighboring_lanes is not None and len(additional_lanes)>=max_num_neighboring_lanes:
                    break
                if lane_id == ego_lane_id:
                    continue
                if len(filtered_lanes)>0:
                    if lane_id in filtered_lanes:
                        continue
                    ego_lane_and_filtered_lanes = {ego_lane_id: sorted_lanes_polylines[ego_lane_id]}
                    ego_lane_and_filtered_lanes.update(filtered_lanes)
                    object_i_xyh = sorted_lanes_polylines[lane_id]
                    if len(object_i_xyh)==0:
                        continue
                    for k,v in ego_lane_and_filtered_lanes.items(): # iterate over all ego lanes
                        diff_to_ego_lane = (v[..., np.newaxis, :] - object_i_xyh)
                        object_i_dist = np.linalg.norm(diff_to_ego_lane[...,:2], axis=2)
                        min_index = np.argmin(object_i_dist)
                        row_idx, col_idx = np.unravel_index(min_index, object_i_dist.shape)
                        if object_i_dist[row_idx,col_idx]<=dist_threshold and np.abs(np.degrees(diff_to_ego_lane[row_idx,col_idx][-1]))<= angle_threshold:
                            additional_lanes.update({lane_id:sorted_lanes[lane_id]})
                else:
                    object_i_xyh = sorted_lanes_polylines[lane_id]
                    if len(object_i_xyh)>4:
                        diff_to_ego_lane = (ego_lane_xyh[..., np.newaxis, :] - object_i_xyh)
                        object_i_dist = np.linalg.norm(diff_to_ego_lane[...,:2], axis=2)
                        min_index = np.argmin(object_i_dist)
                        row_idx, col_idx = np.unravel_index(min_index, object_i_dist.shape)
                        if object_i_dist[row_idx,col_idx]<=dist_threshold and np.abs(np.degrees(diff_to_ego_lane[row_idx,col_idx][-1]))<= angle_threshold:
                            additional_lanes.update({lane_id:sorted_lanes[lane_id]})
        

            additional_lanes.update(additional_ego_lanes)
        
                
                


        coords_connected, lanes_connected, lanes_ids_connected = connect_lanes_within_distance(sorted_lanes, {ego_lane_id:ego_lane}, max_distance)
        coords_connected_ = [np.vstack(coords_connected[i]) for i in range(len(coords_connected))]
        if use_neighbouring_lanes:
            lanes_connected_, lanes_ids_connected_ = [], []
            for additional_lane_id in additional_lanes:
                _coords_connected, _lanes_connected, _lanes_ids_connected = connect_lanes_within_distance(sorted_lanes, {additional_lane_id:sorted_lanes[additional_lane_id]}, max_distance)
                for xx in [np.vstack(_coords_connected[i]) for i in range(len(_coords_connected))]:
                    if True:
                        distances = np.linalg.norm(xx[:, :, np.newaxis] - coords_connected_[0][:, :].T, axis=1)
                        min_distances = np.min(distances, axis=1)
                        non_intersecting_mask = min_distances > 0.5
                        last_non_intersecting_idx = 0
                        if np.any(non_intersecting_mask):
                            transitions = np.where((non_intersecting_mask[:-1] == False) & (non_intersecting_mask[1:] == True))[0] + 1
                            if len(transitions) > 0:
                                last_non_intersecting_idx = transitions[-1]
                        coords_connected_.append(xx[last_non_intersecting_idx:])
                lanes_connected_.append(_lanes_connected)
                lanes_ids_connected.append(_lanes_ids_connected)
        map_lanes = pad_and_stack_arrays(coords_connected_)
        center, angle = ego.copy()[-1][:2], ego.copy()[-1][2]
        map_lanes = self.normalize_map_points(deepcopy(map_lanes), center, angle)
       
        ego_lane_xyh = self.normalize_map_points(deepcopy(ego_lane_xyh[np.newaxis]), center, angle)
        lanes_copy = deepcopy(map_lanes)
        lanes_copy_norms = np.linalg.norm(lanes_copy[...,:2], axis=-1)
        zero_mask = lanes_copy_norms == 0
        lanes_copy[zero_mask] = np.inf
        lanes_copy_norms = np.linalg.norm(lanes_copy[...,:2], axis=-1)
        closest_points_in_lines = np.argmin(lanes_copy_norms, -1)
        closest_point_idx = np.argmin(lanes_copy_norms)
        closest_point_idx = np.unravel_index(closest_point_idx, lanes_copy_norms.shape)
        closest_point = deepcopy(map_lanes[closest_point_idx][...,:3])
     
                
        
        if True:
            for range_threshold in [0.5, 1, 5, 10, 20, 100]:
                direct_map_lanes_bool = [min(np.linalg.norm(lane[lane[:,0]!=0][:,:2], axis=-1)) < range_threshold for lane in map_lanes]
                if sum(direct_map_lanes_bool)>0: # if found lanes within range
                    break
            for i, lane_i in enumerate(map_lanes):
                for j, lane_j in enumerate(map_lanes):
                    if i!=j and direct_map_lanes_bool[j] and not direct_map_lanes_bool[i]:
                        to_truncate_lane = lane_i
                        reference_lane = lane_j
                        if len(to_truncate_lane[to_truncate_lane[:,0]!=0])>1 and len(reference_lane[reference_lane[:,0]!=0])>1:
                            filtered_reference_lane = reference_lane[reference_lane[:, 0] != 0]
                            filtered_to_truncate_lane = to_truncate_lane[to_truncate_lane[:, 0] != 0]
                            distances = np.linalg.norm(
                                filtered_reference_lane[:, np.newaxis, :] - filtered_to_truncate_lane[np.newaxis, :, :], 
                                axis=2
                            )
                            for range_threshold in [0.01, 0.1, 0.5]:
                                mask = np.any(distances < range_threshold, axis=0)
                                first_index = np.argmax(mask) if np.any(mask) else 0
                                if first_index>0:
                                    break
                            nonzero_indices = np.nonzero(lane_i[:, 0])[0]
                            if len(nonzero_indices) > first_index and first_index > 0:
                                lane_i[nonzero_indices[:first_index], :] = 0

        
        if True:
            indices = np.arange(map_lanes.shape[1])  # Array of indices from 0 to 299
            mask_indices = indices < closest_points_in_lines[:, np.newaxis]
            mask_angle = map_lanes[:, :, 2] < angle_threshold
            mask = mask_indices & mask_angle
            map_lanes[mask] = 0
            truncated_map_lanes = truncate_lanes_vectorized(deepcopy(map_lanes))
            map_lanes = truncated_map_lanes if len(truncated_map_lanes)>=1 else map_lanes
        for i, lane in enumerate(map_lanes):
            distances = np.linalg.norm(lane[:,:2], axis=-1)
            exceed_indices = np.where(distances > max_distance)[0]
            if len(exceed_indices) > 0:
                first_exceed_idx = exceed_indices[0]  # Get the first index where distance exceeds max_distance
                map_lanes[i, first_exceed_idx:] = 0  # Zero out all points after this index
        
        truncated_map_lanes = truncate_lanes_vectorized(deepcopy(map_lanes))
        map_lanes = truncated_map_lanes if len(truncated_map_lanes)>=1 else map_lanes
        
        if True:
            main_lanes = {}
            not_main_lanes = {}
            for i, lane in enumerate(map_lanes):
                min_distance_found = min(np.linalg.norm(lane[lane[:,0]!=0][:,:2], axis=-1))
                if min_distance_found<5:
                    main_lanes[i] = lane
            
            for i, lane in enumerate(map_lanes):
                if i not in not_main_lanes:
                    min_distance_found = min(np.linalg.norm(lane[lane[:,0]!=0][:,:2], axis=-1))
                    if min_distance_found>5:
                        if len(lane[lane[:,0]!=0])>4:
                            for j in main_lanes:
                                if i!=j:
                                    if len(main_lanes[j][main_lanes[j][:,0]!=0])>4:
                                        l1 = LineString(lane[lane[:,0]!=0])
                                        l2 = LineString(main_lanes[j][main_lanes[j][:,0]!=0])
                                        distance = l1.distance(l2)
                                    if distance<1:
                                        not_main_lanes[i] = lane
                
            main_lanes.update(not_main_lanes)
            map_lanes = pad_and_stack_arrays([v for v in main_lanes.values()])
        
        
        
        
        return map_lanes, ego_lane_xyh, max_distance, min_distance, vehicle_can_stop
    
    def plt_scene_raw(self, agent_1=None, agent_2=None, other_agents=None, figure_center=None, max_num_lanes=10, direction_lanes = None, map_lanes=None, drivable_lanes=None, center=None, angle=None):
        lane_polylines = get_polylines(self.lanes)
        lane_types = [value.type for value in self.lanes.values()]
        fig=None
        ax=None
        num_lanes = 0
       
        colors_list = ['black']*100
        if True:
            if drivable_lanes is not None:
                for i, lane in enumerate(drivable_lanes):
                    lane = lane[lane[:,0]!=0]
                    j = i%60
                    fig, ax = vizualize_background(segment=lane, segment_type=0, fig=fig, categ=0, ax=ax, force_color=colors_list[j])
                
        if True:
            for i, lane in enumerate(lane_polylines.values()):
                lane = lane[lane[:,0]!=0]
                lane = map_norm(lane, center, angle)
                if lane_types[i] in [0,1,2]:
                    fig, ax = vizualize_background(segment=lane, segment_type=lane_types[i], fig=fig, categ=0, ax=ax)
                    num_lanes+=1
                    if num_lanes>max_num_lanes:
                        break
                else:
                    continue
            if True:
                boundary_polylines = get_polylines(self.roads_boundary)
                boundary_type = [value.type for value in self.roads_boundary.values()]
                for i, lane in enumerate(boundary_polylines.values()):
                    lane = lane[lane[:,0]!=0]
                    lane = map_norm(lane, center, angle)
                    fig, ax = vizualize_background(segment=lane, segment_type=boundary_type[i], fig=fig, categ=1, ax=ax)

                edge_polylines = get_polylines(self.roads_edge)
                edge_type = [value.type for value in self.roads_edge.values()]
                for i, lane in enumerate(edge_polylines.values()):
                    lane = lane[lane[:,0]!=0]
                    lane = map_norm(lane, center, angle)
                    fig, ax = vizualize_background(segment=lane, segment_type=edge_type[i], fig=fig, categ=2, ax=ax)

                crosswalk_polylines = self.get_crosswalk_polylines()
                for i, lane in enumerate(crosswalk_polylines.values()):
                    lane = lane[lane[:,0]!=0]
                    lane = map_norm(lane, center, angle)
                    fig, ax = vizualize_background(segment=lane, segment_type=1, fig=fig, categ=4, ax=ax)

                stop_signs_points, stop_signs_lanes  = self.get_stop_signs_locations()
                for point in stop_signs_points:
                    point = map_norm(point[None], center, angle)[0]
                    fig,ax = vizualize_background(segment=point, segment_type=1, fig=fig, categ=3, ax=ax)
            

            if True:
                if other_agents is not None:
                    alphabets = [i for i in range(50)]
                    for agent_idx, agent_ in enumerate(other_agents):
                        agent_type = int((agent_[-1,-3:].argmax(-1)+1)*agent_[-1,-3:].sum())
                        if agent_type>0:
                            colors = ['','dimgrey', 'sandybrown', 'dimgrey']
                            vizualize_agent_rect(fig=fig, past=agent_, object_type=agent_type, color=colors[agent_type], agent_prefix=alphabets[agent_idx+3])
                if agent_1 is not None:            
                    agent_ = agent_1
                    agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                    colors = ['','darkviolet', 'darkviolet', 'darkviolet']
                    vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='1') # 'Ego'
                    vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
                if agent_2 is not None:            
                    agent_ = agent_2
                    agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                    colors = ['','blue', 'blue', 'blue']
                    vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='2') # 'Interactive'
                    vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
                
                traffic_light_points, traffic_light_states, _ = self.get_traffic_light_locations(unique_only=True)
                for point, state in zip(traffic_light_points, traffic_light_states):
                    fig,ax = vizualize_background(segment=point, segment_type=state, fig=fig, categ=6, ax=ax)
        center_x = 0  # Replace with your actual center x-coordinate
        center_y = 0  # Replace with your actual center y-coordinate
        range_crop = 200  # Crop range around the center
        ax.set_xlim(center_x - range_crop, center_x + range_crop)
        ax.set_ylim(center_y - range_crop, center_y + range_crop)
        

        return fig

    def process_data(self, viz=True,test=False):
        navigation_extractor = futureNavigation(num_classes=5)
        direction_classes = np.array(navigation_extractor.direction_classifier.classes)
        selected_scenario = True #comment

        if self.point_dir != '':
            self.build_points()
        for iii, data_file in enumerate(self.data_files):
            dataset = tf.data.TFRecordDataset(data_file)
            self.pbar = tqdm(total=len(list(dataset)))
            self.pbar.set_description(f"Processing {data_file.split('/')[-1]}")

            for data in dataset:
                
                parsed_data = scenario_pb2.Scenario()
                parsed_data.ParseFromString(data.numpy())
                
                scenario_id = parsed_data.scenario_id

                self.scenario_id = scenario_id
                objects_of_interest = parsed_data.objects_of_interest

                tracks_to_predict = parsed_data.tracks_to_predict
                id_list = {}
                tracks_list = []
                for ids in tracks_to_predict:
                    id_list[parsed_data.tracks[ids.track_index].id] = ids.track_index
                    tracks_list.append(ids.track_index)
                interact_list = []
                for int_id in objects_of_interest:
                    interact_list.append(id_list[int_id])

                self.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)
                

                if test:
                    if parsed_data.tracks[tracks_to_predict[0].track_index].object_type==1:
                        self.sdc_ids_list = [([tracks_list[1], tracks_list[0]],1)]
                    else:
                        self.sdc_ids_list = [(tracks_list,1)]
                else:
                    self.interactive_process(tracks_list, interact_list, parsed_data.tracks)
                
                if args.single_agent:
                    ego_ids_list = [self.sdc_ids_list[i][0][0] for i in range(len(self.sdc_ids_list))]
                    unique_ego_id_bool = []
                    unique_ego_id_list = []
                    for unique_ego_id in ego_ids_list:
                        if unique_ego_id in unique_ego_id_list:
                            unique_ego_id_bool.append(False)
                        else:
                            unique_ego_id_list.append(unique_ego_id)
                            unique_ego_id_bool.append(True)
                for pairs_idx, pairs in enumerate(self.sdc_ids_list):
                    if args.single_agent and not unique_ego_id_bool[pairs_idx]:
                        continue # if single agent data preperation, consider the ego vehicle to appear only once

                    valid_instruct = True
                    sdc_ids, interesting = pairs[0], pairs[1]
                    
                    if interesting!=1 and args.small_data: # comment for original preprocessing
                        continue
                    ego = self.ego_process(sdc_ids, parsed_data.tracks)

                    ego_type = parsed_data.tracks[sdc_ids[0]].object_type
                    if ego_type!=1:  # comment for original preprocessing
                        valid_instruct = False
                    neighbor_type = parsed_data.tracks[sdc_ids[1]].object_type
                    object_type = np.array([ego_type, neighbor_type])
                    self.object_type = object_type
                    ego_index = parsed_data.tracks[sdc_ids[0]].id
                    neighbor_index = parsed_data.tracks[sdc_ids[1]].id
                    object_index = np.array([ego_index, neighbor_index])
                    
                    neighbors, _ = self.neighbors_process(sdc_ids, parsed_data.tracks)
                    max_num_points = 500
                    
                    
                    map_lanes = np.zeros(shape=(self.drivable_lanes, max_num_points, 17), dtype=np.float32) # DONE: make 6 variable
                    additional_boundaries_ = np.zeros(shape=(self.drivable_lanes, max_num_points, 4), dtype=np.float32)
                    map_crosswalks = np.zeros(shape=(self.num_crosswalks, 100, 3), dtype=np.float32)
                    ground_truth = self.ground_truth_process(sdc_ids, parsed_data.tracks)
                    
                    inter = 'interest' if interesting==1 else 'r'
                    filename = self.save_dir + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}_{pairs_idx}.npz"
                    root_dir_jsons = "/".join(filename.split('/')[:-1])
                    if valid_instruct:
                        agent_json_filename = f"{root_dir_jsons}_agentJsons/{filename.split('/')[-1][:-4]}.json"
                        map_json_filename = f"{root_dir_jsons}_mapJsons/{filename.split('/')[-1][:-4]}.json"
                        templateLLM_filename = f"{root_dir_jsons}_templateLLM/{filename.split('/')[-1][:-4]}.txt"
                        acts_filename = f"{root_dir_jsons}_acts/{filename.split('/')[-1][:-4]}.pkl"
                        viz_dir = f"{root_dir_jsons}_fig/{filename.split('/')[-1][:-4]}.png"

                    original_map_lanes = np.zeros(shape=(2, 6, 300, 17), dtype=np.float32) # DONE: make 6 variable
                    original_map_crosswalks = np.zeros(shape=(2, 4, 100, 3), dtype=np.float32)
                    original_map_lanes[0], original_map_crosswalks[0] = self.map_process_original(ego[0])
                    original_map_lanes[1], original_map_crosswalks[1] = self.map_process_original(ego[1])
                    map_lanes, map_crosswalks, additional_boundaries_, traffic_lights, stop_signs, speed_bumps = self.map_process(ego[0])
                    
                    if valid_instruct:
                        try:
                            drivable_lanes_new, ego_lane, max_d, min_d, vehicle_can_stop = self.get_drivable_lanes(ego[0])
                        except Exception as e:
                            valid_instruct = False
                     
                    if test:
                        ground_truth = np.zeros((2, self.future_len, 5))
                    else:
                        ground_truth = self.ground_truth_process(sdc_ids, parsed_data.tracks)
                    original_ego, original_neighbors, original_map_lanes, original_map_crosswalks, original_ground_truth, original_region_dict = self.normalize_data_original(deepcopy(ego), deepcopy(neighbors), original_map_lanes, original_map_crosswalks, deepcopy(ground_truth))
                    if valid_instruct:
                        try:
                            ego, ground_truth, neighbors, not_valid_future = self.interpolate_missing_data(ego, ground_truth, neighbors)
                            if not_valid_future:
                                valid_instruct = False
                        except Exception as e:
                            valid_instruct = False
                    if valid_instruct:
                        agent_json = self.gen_instruct_caption_01(history=np.vstack((deepcopy(ego),deepcopy(neighbors))),future=deepcopy(ground_truth), navigation_extractor=navigation_extractor, vizualize=args.viz, viz_dir= 'ex.png' if not args.not_debug else viz_dir)
                        if'Agent-1' not in agent_json.keys():
                            valid_instruct = False
                        if 'Agent-2' not in agent_json.keys():
                            valid_instruct = False
                      
                    ego, neighbors, map_lanes, map_crosswalks, ground_truth, region_dict, normalization_param, speed_bumps = self.normalize_data(deepcopy(ego), deepcopy(neighbors), deepcopy(map_lanes[np.newaxis]), deepcopy(map_crosswalks[np.newaxis]), deepcopy(ground_truth), deepcopy(speed_bumps[np.newaxis]), viz=False)
                    map_lanes, map_crosswalks, speed_bumps = map_lanes[0], map_crosswalks[0], speed_bumps[0]
                    if valid_instruct and len(ego)==1:
                        valid_instruct = False
                    if valid_instruct:
                        agent_json_, map_json = self.gen_instruct_caption_02(history=np.vstack((deepcopy(ego),deepcopy(neighbors))),future=deepcopy(ground_truth), navigation_extractor=navigation_extractor, normalization_param=normalization_param)
                        if agent_json['Agent-1']['movement 0.1to8']=='INVALID':
                            valid_instruct = False
                    if valid_instruct:
                        for agent_k in agent_json.keys():
                            agent_json[agent_k].update(agent_json_[agent_k])
                    if valid_instruct and args.plausbility:

                        if drivable_lanes_new is not None:
                            plausibility, directions_lanes, _ = self.get_direction_plausibility_03(deepcopy(ego_lane), deepcopy(drivable_lanes_new), deepcopy(ego))
                        else:
                            plausibility, directions_lanes, drivable_lanes = self.get_direction_plausibility_02(deepcopy(ego), deepcopy(map_lanes), deepcopy(map_crosswalks), deepcopy(ground_truth), deepcopy(additional_boundaries_), deepcopy(traffic_lights), deepcopy(stop_signs), deepcopy(agent_json_))

                        possible_directions = plausibility
                        possible_directions_cls = [np.where(possible_direction_i == direction_classes)[0][0] for possible_direction_i in possible_directions]
                        not_possible_directions = [direction_class_i for direction_class_i in direction_classes if direction_class_i not in plausibility]
                        not_possible_directions_cls = [np.where(direction_class_i == direction_classes)[0][0] for direction_class_i in direction_classes if direction_class_i not in plausibility]


                        possible_directions_mask = [agent_json['Agent-1']['direction 0.1to8'] != dir_i for dir_i in possible_directions]
                        possible_directions = [possible_directions[i] for i, mask in enumerate(possible_directions_mask) if mask]
                        possible_directions_cls = [possible_directions_cls[i] for i, mask in enumerate(possible_directions_mask) if mask]
                        not_possible_directions_mask = [(agent_json['Agent-1']['direction 0.1to8'] != dir_i) or not (dir_i in possible_directions) for dir_i in not_possible_directions]
                        not_possible_directions = [not_possible_directions[i] for i, mask in enumerate(not_possible_directions_mask) if mask]
                        not_possible_directions_cls = [not_possible_directions_cls[i] for i, mask in enumerate(not_possible_directions_mask) if mask]
                        
                        if agent_json['Agent-1']['direction 0.1to8']!='stationary':
                            if vehicle_can_stop:
                                if 'stationary' not in possible_directions:
                                    possible_directions.append('stationary')
                                    possible_directions_cls.append(0)
                                if 'stationary' in not_possible_directions:
                                    not_possible_directions_mask = [dir_i != 'stationary' for dir_i in not_possible_directions]
                                    not_possible_directions = [not_possible_directions[i] for i, mask in enumerate(not_possible_directions_mask) if mask]
                                    not_possible_directions_cls = [not_possible_directions_cls[i] for i, mask in enumerate(not_possible_directions_mask) if mask]
                            else:
                                if 'stationary' not in not_possible_directions:
                                    not_possible_directions.append('stationary')
                                    not_possible_directions_cls.append(0)
                                if 'stationary' in possible_directions:
                                    possible_directions_mask = [dir_i != 'stationary' for dir_i in possible_directions]
                                    possible_directions = [possible_directions[i] for i, mask in enumerate(possible_directions_mask) if mask]
                                    possible_directions_cls = [possible_directions_cls[i] for i, mask in enumerate(possible_directions_mask) if mask]
                                


                        num_examples_to_validate = 100

                        history_viz=np.vstack((deepcopy(ego),deepcopy(neighbors)))
                        future_viz=deepcopy(ground_truth)
                        subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int)
                        
                        if valid_instruct:
                            llm_json = generate_template_json(agent_json, map_json, possible_directions, not_possible_directions, direction_classes)
                            with open(templateLLM_filename, 'w') as file:
                                file.write(llm_json)
                            acts = np.array([-1,-1])
                            acts[0] = agent_json['Agent-1']['direction 0.1to8_cls'] if ('Agent-1' in agent_json.keys() and object_type[0]==1)  else -1
                            acts[1] = agent_json['Agent-2']['direction 0.1to8_cls'] if ('Agent-2' in agent_json.keys() and object_type[1]==1)  else -1
                            to_store = str({'act0': acts[0], 'act1': acts[1]})
                            if not debug:
                                with open(acts_filename, 'wb') as f:
                                    pickle.dump(to_store, f)

                        n_lanes, t_samples, t_subsampling = 100, 500, 2
                        if not debug:
                            if self.point_dir == '':
                                region_dict = {6:np.zeros((6,2))}
                                original_region_dict = {6:np.zeros((6,2))}
                            np.savez_compressed(filename, ego=np.array(original_ego), neighbors=np.array(original_neighbors), map_lanes=np.array(original_map_lanes), 
                                    map_crosswalks=np.array(original_map_crosswalks),object_type=np.array(object_type),region_6=np.array(original_region_dict[6]),
                                    object_index=np.array(object_index), current_state=np.array(self.current_xyzh[0]),gt_future_states=np.array(original_ground_truth),
                                    )

                        if debug or 'validation' in args.save_path:
                            try:
                                fig3 = self.plt_scene_raw(
                                        (history_viz[0], future_viz[0, subsample_indices]),
                                        max_num_lanes=1000,
                                        map_lanes=deepcopy(map_lanes),
                                        drivable_lanes=deepcopy(drivable_lanes_new),
                                        direction_lanes=directions_lanes,
                                        center=normalization_param[0],
                                        angle=normalization_param[1]
                                    )
                                if not debug and 'validation' in args.save_path:
                                    fig3.savefig(viz_dir)
                                    plt.close()
                                elif debug:
                                    gt_cls = agent_json['Agent-1']['direction 0.1to8'] if 'straight' not in agent_json['Agent-1']['direction 0.1to8'] else 'move straight'
                                    print(f'GT: {gt_cls}')
                                    print(f'F: {possible_directions}')
                                    print(f'IF: {not_possible_directions}')
                                    fig3.savefig('ex.png')
                                    print(scenario_id)
                                    plt.close()
                                    self.get_direction_plausibility_03(deepcopy(ego_lane), deepcopy(drivable_lanes_new), deepcopy(ego))
                            except Exception as e:
                                pass
                        
                        if debug:
                            fig3 = None
                            mode = 'gt'
                            cls = agent_json['Agent-1']['direction 0.1to8'] if 'straight' not in agent_json['Agent-1']['direction 0.1to8'] else 'move straight'
                            save_dir_ = os.path.join(os.path.dirname(filename), f'{mode}', f'{cls}')
                            os.makedirs(save_dir_, exist_ok=True)
                            if len(glob.glob(os.path.join(save_dir_, '*'))) < num_examples_to_validate:
                                save_dir = os.path.splitext(os.path.basename(filename))[0] + '.png'
                                save_dir = os.path.join(save_dir_, save_dir)
                                fig3 = self.plt_scene_raw(
                                    (history_viz[0], future_viz[0, subsample_indices]),
                                    max_num_lanes=1000,
                                    map_lanes=deepcopy(map_lanes),
                                    drivable_lanes=deepcopy(drivable_lanes_new),
                                    direction_lanes=directions_lanes,
                                    center=normalization_param[0],
                                    angle=normalization_param[1]
                                )
                                fig3.savefig(save_dir)

                            mode = 'f'
                            for cls in possible_directions:
                                save_dir_ = os.path.join(os.path.dirname(filename), f'{mode}', f'{cls}')
                                os.makedirs(save_dir_, exist_ok=True)
                                if len(glob.glob(os.path.join(save_dir_, '*'))) < num_examples_to_validate:
                                    save_dir = os.path.splitext(os.path.basename(filename))[0] + '.png'
                                    save_dir = os.path.join(save_dir_, save_dir)
                                    if fig3 is None:
                                        fig3 = self.plt_scene_raw(
                                            (history_viz[0], future_viz[0, subsample_indices]),
                                            max_num_lanes=1000,
                                            map_lanes=deepcopy(map_lanes),
                                            drivable_lanes=deepcopy(drivable_lanes_new),
                                            direction_lanes=directions_lanes,
                                            center=normalization_param[0],
                                            angle=normalization_param[1]
                                        )
                                    fig3.savefig(save_dir)
                            
                            mode = 'if'
                            for cls in not_possible_directions:
                                save_dir_ = os.path.join(os.path.dirname(filename), f'{mode}', f'{cls}')
                                os.makedirs(save_dir_, exist_ok=True)
                                if len(glob.glob(os.path.join(save_dir_, '*'))) < num_examples_to_validate:
                                    save_dir = os.path.splitext(os.path.basename(filename))[0] + '.png'
                                    save_dir = os.path.join(save_dir_, save_dir)
                                    if fig3 is None:
                                        fig3 = self.plt_scene_raw(
                                            (history_viz[0], future_viz[0, subsample_indices]),
                                            max_num_lanes=1000,
                                            map_lanes=deepcopy(map_lanes),
                                            drivable_lanes=deepcopy(drivable_lanes_new),
                                            direction_lanes=directions_lanes,
                                            center=normalization_param[0],
                                            angle=normalization_param[1]
                                        )
                                    fig3.savefig(save_dir)

                            plt.close()

                    if args.small_data:
                        break

                
                self.pbar.update(1)

            self.pbar.close()

def parallel_process(root_dir):
    print(root_dir)
    processor = DataProcess(root_dir=[root_dir], point_dir=point_path, save_dir=save_path) 
    processor.process_data(viz=debug,test=test)
    print(f'{root_dir}-done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing Interaction Predictions')
    parser.add_argument('--load_path', type=str, help='path to dataset files', default='/ibex/project/c2278/felembaa/datasets/waymo/validation_interactive')
    parser.add_argument('--save_path', type=str, help='path to save processed data', default = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation_temp')
    parser.add_argument('--point_path', type=str, help='path to load K-Means Anchors (Currently not included in the pipeline)', default='')
    parser.add_argument('--processes', type=int, help='multiprocessing process num', default=8)
    parser.add_argument('--not_debug', action="store_true", help='visualize processed data', default=False)
    parser.add_argument('--test', action="store_true", help='whether to process testing set', default=False)
    parser.add_argument('--use_multiprocessing', action="store_true", help='use multiprocessing', default=False)
    parser.add_argument('--run_num', type=int, help='', default=-1)
    parser.add_argument('--small_data', help='', action="store_true", default=False)
    parser.add_argument('--delete_old', help='', action="store_true", default=False)
    parser.add_argument('--plausbility', default=True)
    parser.add_argument('--viz', help='', action="store_true", default=False)
    parser.add_argument('--single_agent', help='data preperation for single agent only', default=True)
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')
    args.not_debug = True

    if args.run_num!=-1:
        if 'validation' in args.load_path:
            numbers = [str(i).zfill(5) for i in range(150)]
            split_parts = [numbers[i:i + args.processes] for i in range(0, len(numbers), args.processes)]
            data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts[args.run_num]]
        else:
            numbers = [str(i).zfill(5) for i in range(1000)]
            split_parts = [numbers[i:i + args.processes] for i in range(0, len(numbers), args.processes)]
            data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts[args.run_num]]
    

    save_path = args.save_path
    point_path = args.point_path
    debug = not args.not_debug
    test = args.test
    if save_path != '':
        ff = save_path
        if args.delete_old:
            if os.path.exists(ff):
                shutil.rmtree(ff)
        os.makedirs(save_path, exist_ok=True)

        ff = f"{save_path}_templateLLM"
        if args.delete_old:
            if os.path.exists(ff):
                shutil.rmtree(ff)
        os.makedirs(ff, exist_ok=True)

        ff = f"{save_path}_acts"
        if args.delete_old:
            if os.path.exists(ff):
                shutil.rmtree(ff)
        os.makedirs(ff, exist_ok=True)

        if 'validation' in args.save_path:
            ff = f"{save_path}_fig"
            if args.delete_old:
                if os.path.exists(ff):
                    shutil.rmtree(ff)
            os.makedirs(ff, exist_ok=True)


        
        
        
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("*************** START Time =", current_time)

    if args.use_multiprocessing:
        with Pool(processes=args.processes) as p:
            p.map(parallel_process, data_files)
    else:
        processor = DataProcess(root_dir=data_files, point_dir=point_path, save_dir=save_path)
        processor.process_data(viz=debug,test=test)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("*************** END Time =", current_time)
    print('Done!')
  