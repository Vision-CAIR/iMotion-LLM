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
# from interaction_prediction.exctract_instruct import *
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
    """
    This function removes any entries with the word 'straight' if they appear 
    after an entry that contains the word 'turn'.
    
    Parameters:
    - entries (list of tuples): A list where each tuple contains a string (the instruction) and a number (the count).
    
    Returns:
    - list of tuples: A filtered list without 'straight' entries that appear after a 'turn'.
    """
    filtered_entries = []
    turn_found = False  # Flag to check if we've encountered a "turn"

    for entry in entries:
        instruction, _ = entry
        
        if "turn" in instruction:
            # If a "turn" is found, add the entry and set the flag
            turn_found = True
            filtered_entries.append(entry)
        elif "straight" in instruction and turn_found:
            # Skip entries that contain "straight" if "turn" has been found
            continue
        else:
            # If no "straight" after a "turn", add the entry
            filtered_entries.append(entry)
    
    return filtered_entries

def pad_and_stack_arrays(array_list, pad_value=0):
    """
    Pads each array in a list of 2D arrays with zeros (or a specified value) along the first dimension
    to match the length of the longest array, then stacks the padded arrays into a single 3D array.

    Parameters:
    - array_list (list of np.ndarray): List of 2D arrays to be padded and stacked.
    - pad_value (numeric, optional): The value to use for padding (default is 0).

    Returns:
    - np.ndarray: A 3D array where each 2D array from the list has been padded and stacked.
    """
    # Step 1: Find the maximum T (number of rows)
    max_T = max(array.shape[0] for array in array_list)

    # Step 2: Pad each array with the specified pad_value based on max_T
    padded_arrays = []
    for array in array_list:
        T = array.shape[0]  # Current number of rows (T)
        padding = ((0, max_T - T), (0, 0))  # Pad only along the first dimension (rows)
        padded_array = np.pad(array, padding, mode='constant', constant_values=pad_value)
        padded_arrays.append(padded_array)

    # Step 3: Stack the padded arrays
    return np.stack(padded_arrays)

def remove_leading_zeros(lanes):
    """
    Removes all leading rows that consist entirely of zeros from each lane.
    """
    cleaned_lanes = []
    for lane in lanes:
        # Find the first non-zero row
        non_zero_idx = np.where(np.any(lane != 0, axis=1))[0]
        
        if len(non_zero_idx) > 0:
            first_non_zero_idx = non_zero_idx[0]
            # Slice the lane from the first non-zero row onwards
            cleaned_lanes.append(lane[first_non_zero_idx:])
        else:
            # If the lane is all zeros, append an empty array
            cleaned_lanes.append(np.array([]))

    return cleaned_lanes

def truncate_lanes_vectorized(drivable_lanes_new, threshold=10.0):
    drivable_lanes_new = remove_leading_zeros(drivable_lanes_new)
    truncated_lanes = []
    
    # Loop through each lane
    for lane in drivable_lanes_new:
        if len(lane)<=1:
            continue
        # Calculate Euclidean distances for all consecutive points
        diffs = lane[1:] - lane[:-1]  # Differences between consecutive points
        distances = np.linalg.norm(diffs, axis=1)  # Euclidean distance

        # Find the index where the distance exceeds the threshold
        exceed_indices = np.where(distances > threshold)[0]
        
        # If no point exceeds the threshold, take the whole lane
        if len(exceed_indices) == 0:
            truncated_lanes.append(lane)
        else:
            # Otherwise, truncate the lane up to the first point that exceeds the threshold
            truncated_lanes.append(lane[:exceed_indices[0] + 1])
    # pad_and_stack_arrays(truncated_lanes)
    if len(truncated_lanes)==0:
        return []
    else:
        return pad_and_stack_arrays(truncated_lanes)

    # return np.array(truncated_lanes)

# # Assuming drivable_lanes_new is your (66, 652, 2) tensor
# truncated_lanes = truncate_lanes_vectorized(drivable_lanes_new)

def truncate_lanes(drivable_lanes_new, threshold=10.0):
    truncated_lanes = []
    
    # Loop through each lane
    for lane in drivable_lanes_new:
        valid_lane = []
        
        # Iterate through consecutive points in the lane
        for i in range(len(lane) - 1):
            point1 = lane[i]
            point2 = lane[i + 1]
            
            # Calculate Euclidean distance between two consecutive points
            distance = np.linalg.norm(point2 - point1)
            
            # Add the point to the valid lane if the distance is below the threshold
            valid_lane.append(point1)
            
            if distance > threshold:
                break  # Stop adding points if the distance exceeds the threshold
        
        # Add the last valid point in the lane
        valid_lane.append(lane[i + 1])
        truncated_lanes.append(np.array(valid_lane))
    
    return np.array(truncated_lanes)

def connect_lanes_within_distance(lanes, ego_lane, max_distance, max_depth=10, verbose=False):
    """
    Recursively connect lanes starting from ego_lane, following the outgoing_edges,
    and stopping when the distance between a new lane and the ego_lane exceeds max_distance
    or the recursion depth exceeds max_depth.

    :return: A tuple containing:
        - A list of lists with the discrete_path of each lane.
        - A list of lists with the midline coordinates for each lane.
        - A list of lists with the left boundary coordinates for each lane.
        - A list of lists with the right boundary coordinates for each lane.
        - A list of lane IDs for each lane.
    """
    # Store paths of connected lanes
    all_discrete_paths = []  # To store the discrete paths for each lane
    # all_mid_paths = []  # To store midline coordinates
    # all_left_paths = []  # To store left boundary coordinates
    # all_right_paths = []  # To store right boundary coordinates
    all_lane_ids = []  # To store lane IDs
    visited_lanes = set()  # Keep track of visited lanes to avoid cycles

    def traverse_lane(lanes, lane, original_lane, depth, current_discrete_path, current_lane_ids):
        # Stop if the depth exceeds max_depth
        lane_id = list(lane.keys())[0]
        # list(get_polylines(lane).values())[0][...,:2]
        # list(get_polylines(original_lane).values())[0][...,:2]
        # if depth==9:
        #     print('')
        if depth > max_depth:
            if verbose:
                print(f"Max depth reached at lane {lane_id}")
            if len(current_lane_ids)>1:
                current_discrete_path.pop()
                # current_mid_path.pop()
                # current_left_path.pop()
                # current_right_path.pop()
                current_lane_ids.pop()
                all_discrete_paths.append(list(current_discrete_path))
                # all_mid_paths.append(list(current_mid_path))
                # all_left_paths.append(list(current_left_path))
                # all_right_paths.append(list(current_right_path))
                all_lane_ids.append(list(current_lane_ids))
            return

        # Detect cycle and avoid revisiting lanes
        if lane_id in visited_lanes:
            if verbose:
                print(f"Cycle detected at lane {lane_id}, stopping traversal")
            return
        visited_lanes.add(lane_id)

        # Add the current lane's discrete_path to the current path
        if lane != original_lane or depth > 0:
            current_discrete_path.append(list(lane.values())[0])

            # # Add lane's mid, left, and right boundary to the respective lists
            # mid = [Point2D(node.x, node.y) for node in lane.baseline_path.discrete_path]
            # left = [Point2D(node.x, node.y) for node in lane.left_boundary.discrete_path]
            # right = [Point2D(node.x, node.y) for node in lane.right_boundary.discrete_path]

            # Append to current lists for mid, left, and right boundaries
            # current_mid_path.append(mid)
            # current_left_path.append(left)
            # current_right_path.append(right)

            # Add the lane ID
            current_lane_ids.append(lane_id)

        # Get the baseline path of the current lane and the original ego lane
        # list(original_lane.values())[0]
        original_polyline = list(get_polylines(original_lane).values())[0][...,:2]
        lane_polyline = list(get_polylines(lane).values())[0][...,:2]
        if len(original_polyline)>1 and len(lane_polyline)>1:
            original_linestring = LineString(original_polyline)
            lane_linestring = LineString(lane_polyline)
            # Calculate the distance between the current lane and the original ego lane
            distance = lane_linestring.distance(original_linestring)
        elif len(original_polyline)==1 and len(lane_polyline)==1:
            distance = np.linalg.norm(original_polyline[0] - lane_polyline[0])
            lane_linestring = None
            original_linestring = None
        else:
            lane_linestring = None
            original_linestring = None
            print('')

        # LineString(list(get_polylines(original_lane).values())[0])
        # original_linestring = original_lane.baseline_path.linestring
        # lane_linestring = lane.baseline_path.linestring

        
        # print(f"Traversing lane {lane.id}, distance from ego_lane: {distance}")

        # Stop the traversal if the distance is greater than the allowed max_distance without storing last lane
        if distance > max_distance:
            if verbose:
                print(f"Stopping traversal for lane {lane_id} due to distance: {distance}")
                print("******************")
            current_discrete_path.pop()
            # current_mid_path.pop()
            # current_left_path.pop()
            # current_right_path.pop()
            current_lane_ids.pop()
            all_discrete_paths.append(list(current_discrete_path))
            # all_mid_paths.append(list(current_mid_path))
            # all_left_paths.append(list(current_left_path))
            # all_right_paths.append(list(current_right_path))
            all_lane_ids.append(list(current_lane_ids))
            
            return

        # If there are no more outgoing edges, consider this a leaf and store the path
        outgoing_edges = list(lane.values())[0].exit_lanes
        if len(outgoing_edges)==0:
        # if not lane.outgoing_edges:
            if verbose:
                print(f"Reached a leaf lane {lane_id} with no outgoing edges")
                print("###################")
            all_discrete_paths.append(list(current_discrete_path))
            # all_mid_paths.append(list(current_mid_path))
            # all_left_paths.append(list(current_left_path))
            # all_right_paths.append(list(current_right_path))
            all_lane_ids.append(list(current_lane_ids))
            return

        # Recursively traverse the outgoing edges of the current lane
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
            # If the new exit lane is more than 5 meters distanced from its source lane then it is ignored
            # if new_lane_to_current_lane_distance > 5:
            #     continue
            # next_lane = outgoing_lane_connector_id  # Assuming outgoing_edges is a list of lanes
            if verbose:
                print(f"Processing outgoing lane {i} from lane {lane_id}, next lane ID: {next_lane_id}")
            traverse_lane(lanes, next_lane, original_lane, depth + 1, current_discrete_path, current_lane_ids)

        # Backtrack: remove the current lane from the path after processing all its children
        if lane != original_lane or depth > 0:
            if len(current_lane_ids)>0:
                current_discrete_path.pop()
                # current_mid_path.pop()
                # current_left_path.pop()
                # current_right_path.pop()
                current_lane_ids.pop()

    # Start the traversal from the ego_lane
    traverse_lane(lanes, ego_lane, ego_lane, 0, current_discrete_path=list(ego_lane.values()),
                  current_lane_ids=[list(ego_lane.keys())[0]])

    
    # Filtering repeated entries based on ID
    # Use a dictionary to keep track of unique lane IDs and their corresponding paths
    unique_data = {
        tuple(lane_id): (list(discrete))
        for lane_id, discrete in zip(all_lane_ids, all_discrete_paths)
    }

    if len(unique_data)==0:
        return [], [], []
    # Unzip the dictionary back into separate lists, converting keys (lane IDs) back to lists
    all_lane_ids, paths = zip(*unique_data.items())
    all_lane_ids = [list(lane_id) for lane_id in all_lane_ids]  # Convert tuples back to lists
    # all_discrete_paths = zip(*paths)
    # [item for sublist in zip(*paths) for item in sublist]
    all_discrete_paths = list(unique_data.values())
    all_coords = []
    for i in range(len(all_discrete_paths)):
        all_coords_ = []
        for j in range(len(all_discrete_paths[i])):
            all_coords_.append(list(get_polylines({0: all_discrete_paths[i][j]}).values())[0])
        all_coords.append(all_coords_)
    
    return all_coords, all_discrete_paths, all_lane_ids
    # # Flattening the lists after appending
    # flattened_discrete_paths = [item for sublist in all_discrete_paths for item in sublist]
    # # flattened_mid_paths = [item for sublist in all_mid_paths for item in sublist]
    # # flattened_left_paths = [item for sublist in all_left_paths for item in sublist]
    # # flattened_right_paths = [item for sublist in all_right_paths for item in sublist]
    # flattened_lane_ids = [item for sublist in all_lane_ids for item in sublist]
    
    # # Structured lanes with each entry as connected lanes segments (list of lists of lanes segments)
    # lane_ids_connected = all_lane_ids
    # coords_conntected = [[]]
    
    # for i in range(len(coords_conntected)):
    #     polylines_ = get_polylines({all_lane_ids[i][k]: all_discrete_paths[k] for k in range(len(all_discrete_paths))})
    #     coords_conntected[i]['lane'] = 
    #     all_mid_paths[i]).polylines)


    # # Unrolled segments, with each segment defined seperatly with its own id (list of lanes segments)
    # coords: Dict[str, MapObjectPolylines] = {}
    # coords[VectorFeatureLayer.LANE.name] = MapObjectPolylines(MapObjectPolylines(flattened_mid_paths).polylines)
    # coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(MapObjectPolylines(flattened_left_paths).polylines)
    # coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(MapObjectPolylines(flattened_right_paths).polylines)
    # lane_ids = LaneSegmentLaneIDs(flattened_lane_ids)
    
    # # ## Stucturing lanes such that each lane is a one long continous lane with no segments:
    # # for mids_, lefts_, rights_, ids_ in zip(all_mid_paths, all_left_paths, all_right_paths, all_lane_ids):
    # #     mid_ = [item for sublist in mids_ for item in sublist]
    # #     left_ = [item for sublist in lefts_ for item in sublist]
    # #     right_ = [item for sublist in rights_ for item in sublist]
    # #     break
    # if verbose:
    #     print(lane_ids_connected)
    # return coords, lane_ids, coords_conntected, lane_ids_connected#flattened_discrete_paths, flattened_mid_paths, flattened_left_paths, flattened_right_paths

def get_vel_from_traj(step1xy, step2xy, time_difference=0.1):
        x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocity = distance / time_difference
        return velocity

def get_distance_between_map_object_and_point(point, polyline):
    """
    Get distance between point and nearest point on a polyline.
    :return: Computed distance.
    """
    polyline = np.array([p for p in polyline if not np.any(np.isinf(p)) and not np.any(np.isnan(p))])
    if len(polyline)>1:
        return float(Point(point).distance(LineString(polyline)))
    elif len(polyline)==1:
        return float(Point(point).distance(Point(polyline)))
    else:
        return 10000

def get_rel_direction(angle_, correction_angle):
    # Define angle thresholds for categorization
    angle = minus_2pi(angle_ - correction_angle)
    directly_ahead_threshold = np.radians(0.5)  # Small range around 0 degrees
    directly_behind_threshold = np.radians(179.5)  # Small range around 180 degrees
    front_threshold = np.radians(10)  # Small range around 180 degrees
    behind_threshold = np.radians(110)  # Small range around 180 degrees
    # Categorize based on angle
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
        # This is not accurate, cross product is better
        # Use cross product to distinguish between left and right
        # TO use cross product you need something like: 
        # my_vector, _ = get_unit_vector(traj_history[-2,:2], traj_history[-1,:2])
        # category = "on the left" if np.cross(my_vector, neighbor_vector)>0 else "on the right"
        
    
    return category


def get_unit_vector(pointA, pointB):
    # Vector from A to B
    vector_AB = pointB - pointA
    # Normalize the vector to get the unit vector
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
    ## Guide:
    # center lanes: categ=0, type
    # boundary lanes: categ=1, type from 0 to some number
    # edge lanes: categ=2, type
    # stopsign: categ=3
    # crosswalk: categ=4
    # speedbump: categ=5
    # traffic light: categ=6, type

    ## Note, type of boundary and edge needs to be verified 

    # plt shapes per type
    center_plt_info = [('lightgray','-', 0.1),('gray','solid', 0.4), ('red','solid', 0.4), ('b','dashed', 0.2),]
    # center_plt_info = [('lightgray','-', 1.0),('lightgray','solid', 1.0), ('lightgray','solid', 1.0), ('lightgray','dashed', 1.0),]
    # center_plt_info = [('red','-', 0.1),('red','solid', 0.1), ('red','solid', 0.1), ('red','dashed', 0.1),]
    # center_plt_info = [('black','-', 1),('black','solid', 1), ('black','solid', 1), ('black','dashed', 1),]
    boundry_plt_info = [('lightgray','-', 0.1), ('w','dashed', 0.4),  ('w','solid', 0.4), ('w','solid', 0.4), ('xkcd:yellow','dashed', 0.4), ('xkcd:yellow','dashed', 0.3), ('xkcd:yellow','solid', 0.3), ('xkcd:yellow','solid', 0.3), ('xkcd:yellow','dotted', 0.4),('k', '-', 1.0), ('k', '-', 1.0),]
    edge_plt_info = [('k', '-', 1.0), ('k', '-', 1.0), ('k', '-', 1.0),]
    stopsign_plt_info = [('lightgray','-', 0.1),('r', 'solid',1)]
    crosswalk_plt_info = [('white','-', 1.0),('white', '-',1.0)]
    speedbump_plt_info = [('orange','-', 1.0),('orange', '-',1.0)]
    plt_info = [center_plt_info, boundry_plt_info, edge_plt_info, stopsign_plt_info, crosswalk_plt_info, speedbump_plt_info]
    z_orders = [2,2,2,3,1,1,3]
    if categ==4 or categ==5:
        segment_type=1
    # init figure
    if not fig is not None:
        # dpi = 300
        fig, ax = plt.subplots(dpi=300)
        # size_inches = 800 / dpi
        # fig.set_size_inches([size_inches, size_inches])
        # fig.set_dpi(dpi)
        fig.set_tight_layout(True)
        plt.gca().set_facecolor('silver')
        plt.gca().margins(0)  
        # plt.gca().axes.get_yaxis().set_visible(False)
        # plt.gca().axes.get_xaxis().set_visible(False)
        # Define the center and range for cropping
        # center_x = figure_center[0]  # Replace with your actual center x-coordinate
        # center_y = figure_center[1]  # Replace with your actual center y-coordinate
        # range_crop = 150  # Crop range around the center
        # # Set axis limits to crop around the center
        # ax.set_xlim(center_x - range_crop, center_x + range_crop)
        # ax.set_ylim(center_y - range_crop, center_y + range_crop)

    if categ!=3 and categ!=6:
        if force_color is not None:
            plt.plot(segment[:,0], segment[:,1], (force_color,'solid', 1)[0], linestyle=(force_color,'solid', 1)[1], linewidth=1, alpha=(force_color,'solid', 1)[2], zorder=z_orders[categ])
        else:
            plt.plot(segment[:,0], segment[:,1], plt_info[categ][segment_type][0], linestyle=plt_info[categ][segment_type][1], linewidth=1, alpha=plt_info[categ][segment_type][2], zorder=z_orders[categ])
    elif categ==3:
        circle = Circle((segment[0], segment[1]), 2, color='red', fill=False, zorder=z_orders[categ])
        plt.gca().add_patch(circle)
        # Add text at the center of the circle
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
        # ['unknown', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution']
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
    # object_type_str = [object_type_str_i+"\n"+str(agent_prefix) for object_type_str_i in object_type_str]
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
        # parameters
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
        # static map features
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
                # raise TypeError

        # dynamic map features
        self.traffic_signals = dynamic_map_states # TODO: This to be used to generate captions

    def plt_scene_lanes_segments(self, agent_1=None, agent_2=None, other_agents=None, figure_center=None, max_num_lanes=10):
        # to plot multiple lanes segments, each witrh a different color
        # each category [0:center lanes, 1:boundaries, 2:edges, 3:stop signs, 4:crosswalks, 5:speed bump, 6:traffic light]
        # is plotted seperatly on the same figure, use the correct categ value, refer to the function vizualize_background [in this file]
        lane_polylines = get_polylines(self.lanes)
        lane_types = [value.type for value in self.lanes.values()]
        fig=None
        ax=None
        num_lanes = 0
        # max_num_lanes = 1
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

        if False:
            boundary_polylines = get_polylines(self.roads_boundary)
            boundary_type = [value.type for value in self.roads_boundary.values()]
            for i, lane in enumerate(boundary_polylines.values()):
                fig, ax = vizualize_background(segment=lane, segment_type=boundary_type[i], fig=fig, categ=1, ax=ax)

            edge_polylines = get_polylines(self.roads_edge)
            edge_type = [value.type for value in self.roads_edge.values()]
            for i, lane in enumerate(edge_polylines.values()):
                fig, ax = vizualize_background(segment=lane, segment_type=edge_type[i], fig=fig, categ=2, ax=ax)

            crosswalk_polylines = self.get_crosswalk_polylines()
            for i, lane in enumerate(crosswalk_polylines.values()):
                fig, ax = vizualize_background(segment=lane, segment_type=1, fig=fig, categ=4, ax=ax)

            stop_signs_points, stop_signs_lanes  = self.get_stop_signs_locations()
            for point in stop_signs_points:
                fig,ax = vizualize_background(segment=point, segment_type=1, fig=fig, categ=3, ax=ax)
        

        if False:
            if other_agents is not None:
                # alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                alphabets = [i for i in range(50)]
                # alphabets = [i for i in range(len(alphabets))]            
                for agent_idx, agent_ in enumerate(other_agents):
                    agent_type = int((agent_[-1,-3:].argmax(-1)+1)*agent_[-1,-3:].sum())
                    if agent_type>0:
                        colors = ['','dimgrey', 'sandybrown', 'dimgrey']
                        vizualize_agent_rect(fig=fig, past=agent_, object_type=agent_type, color=colors[agent_type], agent_prefix=alphabets[agent_idx+3])
            if agent_1 is not None:            
                agent_ = agent_1
                agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                # colors = ['','darkviolet', 'violet', 'darkviolet']
                colors = ['','darkviolet', 'darkviolet', 'darkviolet']
                vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='1') # 'Ego'
                vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
            if agent_2 is not None:            
                agent_ = agent_2
                agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                # colors = ['','teal', 'peru', 'teal']
                colors = ['','blue', 'blue', 'blue']
                vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='2') # 'Interactive'
                vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
            
            traffic_light_points, traffic_light_states, _ = self.get_traffic_light_locations(unique_only=True)
            for point, state in zip(traffic_light_points, traffic_light_states):
                fig,ax = vizualize_background(segment=point, segment_type=state, fig=fig, categ=6, ax=ax)

        if figure_center is not None:
            center_x = figure_center[0]  # Replace with your actual center x-coordinate
            center_y = figure_center[1]  # Replace with your actual center y-coordinate
            range_crop = 75  # Crop range around the center
            # Set axis limits to crop around the center
            ax.set_xlim(center_x - range_crop, center_x + range_crop)
            ax.set_ylim(center_y - range_crop, center_y + range_crop)
        # Remove X and Y Axis Labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Optional: remove tick labels if desired
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return fig

    def plt_scene(self, agent_1=None, agent_2=None, other_agents=None, figure_center=None, max_num_lanes=10, direction_lanes = None, map_lanes=None, drivable_lanes=None):
        # each category [0:center lanes, 1:boundaries, 2:edges, 3:stop signs, 4:crosswalks, 5:speed bump, 6:traffic light]
        # is plotted seperatly on the same figure, use the correct categ value, refer to the function vizualize_background [in this file]
        fig=None
        ax=None
        num_lanes = 0
        # max_num_lanes = 1
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
            # map_lanes = map_lanes[...,:3]
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
                # lane = lane[[lane[:,10][k] in [0,1,2] for k in range(len(lane[:,10]))]]
                lane = lane[lane[:,0]!=0]
                fig, ax = vizualize_background(segment=lane, segment_type=2, fig=fig, categ=0, ax=ax, force_color = 'r')
        
        if direction_lanes is not None:
            special_lanes = direction_lanes
            unique_tuples = list({class_label: array for class_label, array in special_lanes.values()}.items())
            special_lanes = unique_tuples
            # classes_colors = ['g', 'b', 'c', 'orange', 'yellow', 'purple', 'pink', 'brown']
            classes_colors = ['brown', 'b', 'c', 'orange', 'yellow', 'purple', 'pink', 'brown', 'gray']
            classes_names = ['stationary', 'straight', 'straight-right', 'straight-left', 'right', 'left', 'right-u-turn', 'left-u-turn']
            for lane_class, lane in special_lanes:
                # Use the correct color and class name for the label
                plt.plot(lane[:, 0], lane[:, 1], color=classes_colors[lane_class], 
                        linewidth=5, alpha=0.9, label=classes_names[lane_class])

                # Show the legend
                plt.legend()
            # print('')
        if False:
            boundary_polylines = get_polylines(self.roads_boundary)
            boundary_type = [value.type for value in self.roads_boundary.values()]
            for i, lane in enumerate(boundary_polylines.values()):
                fig, ax = vizualize_background(segment=lane, segment_type=boundary_type[i], fig=fig, categ=1, ax=ax)

            edge_polylines = get_polylines(self.roads_edge)
            edge_type = [value.type for value in self.roads_edge.values()]
            for i, lane in enumerate(edge_polylines.values()):
                fig, ax = vizualize_background(segment=lane, segment_type=edge_type[i], fig=fig, categ=2, ax=ax)

            crosswalk_polylines = self.get_crosswalk_polylines()
            for i, lane in enumerate(crosswalk_polylines.values()):
                fig, ax = vizualize_background(segment=lane, segment_type=1, fig=fig, categ=4, ax=ax)

            stop_signs_points, stop_signs_lanes  = self.get_stop_signs_locations()
            for point in stop_signs_points:
                fig,ax = vizualize_background(segment=point, segment_type=1, fig=fig, categ=3, ax=ax)
        

        if True:
            if other_agents is not None:
                # alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                alphabets = [i for i in range(50)]
                # alphabets = [i for i in range(len(alphabets))]            
                for agent_idx, agent_ in enumerate(other_agents):
                    agent_type = int((agent_[-1,-3:].argmax(-1)+1)*agent_[-1,-3:].sum())
                    if agent_type>0:
                        colors = ['','dimgrey', 'sandybrown', 'dimgrey']
                        vizualize_agent_rect(fig=fig, past=agent_, object_type=agent_type, color=colors[agent_type], agent_prefix=alphabets[agent_idx+3])
            if agent_1 is not None:            
                agent_ = agent_1
                agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                # colors = ['','darkviolet', 'violet', 'darkviolet']
                colors = ['','darkviolet', 'darkviolet', 'darkviolet']
                vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='1') # 'Ego'
                vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
            if agent_2 is not None:            
                agent_ = agent_2
                agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                # colors = ['','teal', 'peru', 'teal']
                colors = ['','blue', 'blue', 'blue']
                vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='2') # 'Interactive'
                vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
            
            if False: # If map is notmalized this needs to be normalized
                traffic_light_points, traffic_light_states, _ = self.get_traffic_light_locations(unique_only=True)
                for point, state in zip(traffic_light_points, traffic_light_states):
                    fig,ax = vizualize_background(segment=point, segment_type=state, fig=fig, categ=6, ax=ax)

        if figure_center is not None:
            center_x = figure_center[0]  # Replace with your actual center x-coordinate
            center_y = figure_center[1]  # Replace with your actual center y-coordinate
            range_crop = 75  # Crop range around the center
            # Set axis limits to crop around the center
            ax.set_xlim(center_x - range_crop, center_x + range_crop)
            ax.set_ylim(center_y - range_crop, center_y + range_crop)
        # Remove X and Y Axis Labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Optional: remove tick labels if desired
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
        # TODO: Finish this, to be used to generate captions
        # current_location: current vehicle location
        # current_lane: the closest lane to the vehicle
        # to return: the state of the closest traffic light the control the current_lane, and the distance to it
        traffic_light_location = None # it could be None, if no traffic light control the lane
        traffic_light_state = None
        distance_to_traffic_light = None # by comparing the current_location
        return traffic_light_location, traffic_light_state, distance_to_traffic_light

    def get_traffic_light_locations(self, unique_only=False):
        traffic_light_lanes = {}
        # stop_sign_lanes = []

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
        # get all lane polylines
        lane_polylines = get_polylines(self.lanes) # get the xy information, center lane
        lane_types = [value.type for value in self.lanes.values()] # get the type, # 1:  
        # get all road lines and edges polylines
        road_polylines = get_polylines(self.roads)
        road_types = [value.type for value in self.roads.values()]
        edges_polylines = get_polylines(self.roads_edge)
        edges_types = [value.type for value in self.roads_edge.values()]

        current_xyh = traj[-1,:3] # our current time step xyh, where h is heading angle
        final_xyh = future_traj_[-1,:3] # the last step in the future xyh

        ## to calculate the closest distance of "lane_polylines" to the current and final xy:
        current_nearest_lane_distance, current_nearest_lane_vec, nearest_lane = search_nearest_line(lane_polylines, current_xyh[:2])
        final_nearest_lane_distance, final_nearest_lane_vec, nearest_lane = search_nearest_line(lane_polylines, final_xyh[:2])
        ## to calculate the closest distance of "road_polylines" to the current and final xy:
        # current_nearest_road_distance, current_nearest_road_vec, nearest_lane = search_nearest_line(road_polylines, current_xyh[:2])
        # final_nearest_road_distance, final_nearest_road_vec, nearest_lane = search_nearest_line(road_polylines, final_xyh[:2])
        current_nearest_road_distance, current_nearest_road_vec, nearest_lane = search_nearest_line(edges_polylines, current_xyh[:2])
        final_nearest_road_distance, final_nearest_road_vec, nearest_lane = search_nearest_line(edges_polylines, final_xyh[:2])
        ## Condition 1) distance based search:
        current_in= current_nearest_lane_distance<2
        final_in= final_nearest_lane_distance<2
        ## Condition 2) edge, lane in same directions, lane_distance<road_distance:
        if not current_in:                
            if current_nearest_road_distance>current_nearest_lane_distance:
                if np.dot(current_nearest_road_vec,current_nearest_lane_vec)>0.2:
                    current_in=True
        if not final_in:
            if final_nearest_road_distance>final_nearest_lane_distance:
                if np.dot(final_nearest_road_vec,final_nearest_lane_vec)>0.2:
                    final_in=True

        ## Condition 3) edge, lane in opposite directions:
        if not current_in:
            # if current_nearest_road_distance<current_nearest_lane_distance:
            if np.dot(current_nearest_road_vec,current_nearest_lane_vec)<-0.5:
                current_in=True
        if not final_in:
            # if final_nearest_road_distance<final_nearest_lane_distance:
            if np.dot(final_nearest_road_vec,final_nearest_lane_vec)<-0.5:
                final_in=True
        # force offroad
        if current_nearest_lane_distance>25:
            current_in=False
        if final_nearest_lane_distance>25:
            final_in=False

        if not current_in and not final_in:
            inroad_state = {'current map state': 'off-road', 'future map state':'stay off-road', 'on-off map caution': 'be careful driving off-road'}
            # print('an off-road car')
        if not current_in and final_in:
            inroad_state = {'current map state': 'off-road', 'future map state':'merge to the road'}
            # print('an off-road car merging back onto the road')
        if current_in and not final_in:
            inroad_state = {'current map state': 'on-road', 'future map state':'go off-road', 'on-off map caution': 'be careful going from on-road to off-road'}
            # print('going off-road car')
        else:
            inroad_state = {'current map state': 'on-road', 'future map state':'stay on-road'}
        
        return inroad_state
    
    def get_agent_caption(self, agent_name, history, future, navigation_extractor):
        
        # normalizing the view with respect to the agent, to cacluate directional feature. Irrespective of other agents, so using the location information with the normalized version is not correct
        history_normalized_view = history.copy()
        # history_ = history[history[:,0]!=0]
        # history[history[:,0]!=0].copy()[0,:2], history[history[:,0]!=0].copy()[0,2]
        # history.copy()[0,:2], history.copy()[0,2]
        valid_mask = history[:,0]!=0
        agent_center, agent_angle = history[valid_mask].copy()[0,:2], history[valid_mask].copy()[0,2]
        
        
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
        

    def gen_instruct_caption_02(self, history, future, navigation_extractor, vizualize=True, normalization_param=None):
        
        # # interpolate missing values        
        # for i in range(history.shape[0]):
        #     if i==0:
        #         history[i, :-1, :2] = self.interpolate_missing_traj(history[i, :-1, :2])
        #     else:
        #         if history[i].sum()!=0 or i==2:
        #             history[i, :, :2] = self.interpolate_missing_traj(history[i, :, :2])
        #     if i<2:
        #         future[i, :, :2] = self.interpolate_missing_traj(future[i, :, :2])
        
        agent_features = {}
        for i in range(history.shape[0]):
            # if history[i].sum()!=0:
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
        
        # cross_walks = self.get_crosswalk_polylines()
        # for i, cross_walk_id in enumerate(self.get_crosswalk_polylines()):
        #     print(i)
        return agent_features, map_features
        
        # caption_dict[f"Agent-{i+1}"].update({'type':
                #                                      int((history[i,-1,-3:].argmax(-1)+1)*history[i,-1,-3:].sum())
                #                                      }) # agent_type = [1:car, 2:pedestrian, 3:cylcist], if 0 then undefined}) 

    def gen_instruct_caption_01(self, history, future, navigation_extractor, vizualize=True, normalization_param=None, viz_dir=None):
        
        instruct_dict = {}
        
        for i in range(history.shape[0]):
            # if np.any(history[i][:,0] == 0) and i==0:
            #     print('')
            if sum(history[i][:,0]!=0)>=2:
                if not np.any(history[i][:,0] == 0):
                # if sum(history[i][:,0] == 0)<len(history[i][:,0])/2:
                    instruct_dict[f"Agent-{i+1}"] = self.get_agent_caption(i, history=history[i], future=future[i] if i<2 else None, navigation_extractor=navigation_extractor)
                else:
                    instruct_dict[f"Agent-{i+1}"] = self.get_agent_caption(i, history=history[i], future=future[i] if i<2 else None, navigation_extractor=navigation_extractor)
                    # if not len(np.unique(history[i][:,0]))>2:
                    #     instruct_dict[f"Agent-{i+1}"] = self.get_agent_caption(i, history=history[i], future=future[i] if i<2 else None, navigation_extractor=navigation_extractor)
                    # else:
                    #     print(history[i][:,:2])
                    #     instruct_dict[f"Agent-{i+1}"] = self.get_agent_caption(i, history=history[i], future=future[i] if i<2 else None, navigation_extractor=navigation_extractor)
            

        # # interpolate missing values        
        # for i in range(history.shape[0]):
        #     if i==0:
        #         history[i, :-1, :2] = self.interpolate_missing_traj(history[i, :-1, :2])
        #     else:
        #         if not np.any(history[i][:,0] == 0) or i==2:
        #             history[i, :, :2] = self.interpolate_missing_traj(history[i, :, :2])
        #     if i<2:
        #         future[i, :, :2] = self.interpolate_missing_traj(future[i, :, :2])
        
        # agent_names = [i+1 for i in range(history.shape[0])]
        # lane_polylines = get_polylines(self.lanes)
        
        # plotting sampe
        # if normalization_param is not None:
        #     history_global_frame = deepcopy(history)
        #     future_global_frame = deepcopy(future)
        #     for i in range(history.shape[0]):
        #         if history[i].sum()!=0:
        #             history_global_frame[i,:,:5] = reverse_agent_norm(history_global_frame[i], center=normalization_param[0], angle=normalization_param[1])
        #             if i<2:
        #                 future_global_frame[i,:,:5] = reverse_agent_norm(future_global_frame[i], center=normalization_param[0], angle=normalization_param[1])
        vizualize=False
        if vizualize:
            history_global_frame = history
            future_global_frame = future
            subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int) # subsample to sample/second
            # fig = self.plt_scene((history_global_frame[0], future_global_frame[0, subsample_indices]),(history_global_frame[1], future_global_frame[1, subsample_indices]), history_global_frame[2:])
            fig = self.plt_scene((history_global_frame[0], future_global_frame[0, subsample_indices]), max_num_lanes=50)
            # if viz_dir is not None:
            #     fig.savefig(viz_dir)
            # else:
            fig.savefig('ex.png')
            plt.close()
        
        return instruct_dict

    def gen_instruct_caption_v01(self, history_ego, future_ego, history_interactive_neighbor, future_interactive_neighbor, other_neighbors, navigation_extractor=None):
        ## TODO: This function act as a tempelate to implement any rule based captioning, in this code I am only using the ego information
        traj = self.interpolate_missing_traj(history_ego.copy())
        # future_traj = future_ego
        future_traj_ = self.interpolate_missing_traj(future_ego.copy()) # correction
        if len(traj)==0 or len(future_traj_)==0:
            return {}
        

        inter_traj = self.interpolate_missing_traj(history_interactive_neighbor.copy())
        inter_future_traj_ = self.interpolate_missing_traj(future_interactive_neighbor.copy()) # correction
        # subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int) # subsample to sample/second
        # inter_future_traj_subsampled = inter_future_traj_[subsample_indices]

        # alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        alphabets = [i+1 for i in range(50)]
        alphabets_idx = [i for i in range(len(alphabets))]
        
        lane_polylines = get_polylines(self.lanes)
        # for lane in [289, 288, 285]:
        #     plt.plot(lane_polylines[lane][:, 0], lane_polylines[lane][:, 1], 'r', linewidth=1, zorder=3)
        subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int) # subsample to sample/second
        future_traj_subsampled = future_traj_[subsample_indices]
        inter_future_traj_subsampled = inter_future_traj_[subsample_indices]
        # fig = self.plt_scene((traj, future_traj_subsampled),(inter_traj, inter_future_traj_subsampled),other_neighbors)
        # fig.savefig('ex.png')
        # plt.close()
        
        json_dict = {}
        agent_type = int((history_ego[-1,-3:].argmax(-1)+1)*history_ego[-1,-3:].sum()) # agent_type = [1:car, 2:pedestrian, 3:cylcist], if 0 then undefined
        json_dict['Agent-1'] = {'type': agent_type}
        # json_dict['Agent-1'].update{'speed before 1 seconds':, 'speed at current step':,'speed 8 seconds in the future':}
        if len(history_ego[:,:2][history_ego[:,0]!=0])>1:
            speeds = np.linalg.norm(abs_distance_to_velocity(history_ego[:,:2][history_ego[:,0]!=0]),axis=-1)*10*3.6
            speed_past, speed_current = int(speeds[1]), int(speeds[-1])
            current_instructs = navigation_extractor.get_navigation_dict(torch.tensor(future_traj_))
            history_instructs = navigation_extractor.get_navigation_dict_history(torch.tensor(history_ego[:,:5]))

            # print('')
        # if agent_type == 1: # Only if car
        agent_type_str_map = ['Unknown', 'Vehicle', 'Pedestrian', 'Cyclist']
        if agent_type != 0:
            state_features = {
                'Agent-1 speed': str(np.linalg.norm(traj[-1,3:5]).round(2)),
                'Agent-1 heading angle in radians': str(traj[-1,2].round(2)),
                'Agent-1 location': "(0.00, 0.00)",
                # 'Agent-1 type': agent_type_str_map[agent_type],
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

            # DONE: Stop signs & traffic light
            stop_sign_found = self.get_relevant_stop_sign(traj, detection_threshold=150)
            dict_stop = {f"Agent-1 stop sign {k}":v for k,v in stop_sign_found.items()}
            traffic_sign_found = self.get_relevant_traffic_sign(traj, detection_threshold=150)
            dict_traffic = {f"Agent-1 traffic sign {k}":v for k,v in traffic_sign_found.items()}
            # TODO: Close agents
            agents_within = self.get_agents_within(traj, inter_traj, other_neighbors, inter_num, 20)
            if len(agents_within)>0:
                agents_within_dict = {f"Agents within 20 meters from Agent-1": agents_within}
            else:
                agents_within_dict = {f"Agents within 20 meters from Agent-1": "No agents"}

            agent_type_ = int((history_interactive_neighbor[-1,-3:].argmax(-1)+1)*history_interactive_neighbor[-1,-3:].sum()) # agent_type = [1:car, 2:pedestrian, 3:cylcist], if 0 then undefined
            # if agent_type_==1:
            if agent_type_!=0:
                if len(inter_traj)==0 or len(inter_future_traj_)==0:
                    caption_json = {
                        **state_features,
                        **dict1,
                        # **dict2,
                        **t_sceond_rule,
                        **dict_traffic,
                        **dict_stop,
                        **agents_within_dict,
                        **rel_inter_2_ego,
                        **dict2,
                    }
                    return caption_json
                    # somethign = {}
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
                    # 'Agent-2 type': agent_type_str_map[agent_type_],
                    })
                traj, future_traj_, inter_traj = inter_traj, inter_future_traj_, traj
                onroad_state_inter = self.onoffroad(traj, future_traj_)
                instructs_inter = navigation_extractor.get_navigation_dict(torch.tensor(future_traj_))
                # instructs_inter = extract_instruct(future_traj_, traj)
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

                # if len(inter_traj)!=0:
                #     rel_inter_2_ego.update({f"The ego agent, termed Agent-A, is {distance_} away {direction_} from the interactive neighbor Agent-B.":1})
                # else:
                #     rel_inter_2_ego = {}
                
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
            
            # distances to ego
            neighbors_distance_list = {}
            for agent_i, neighbor in enumerate(other_neighbors):
                if neighbor[-1,0]!=0:
                    neighbors_distance_list[f"Agent-{agent_i+2} distance to Agent-1 at current time step"] \
                        = f"{get_unit_vector(traj[-1,:2], neighbor[-1,:2])[1]:.2f}m" if traj[-1,0] != 0 else "Unknown"
                    neighbors_distance_list[f"Agent-{agent_i+2} distance to Agent-1 at future time step (8 seconds in the future)"] \
                        = f"{get_unit_vector(future_traj_subsampled[-1,:2], neighbor[-1,:2])[1]:.2f}m" if future_traj_subsampled[-1,0] != 0 else "Unknown"
            

            # # relative motion to ego
            # for agent_i, neighbor in enumerate(other_neighbors):
            #     self.get_relative_direction(reference_traj=traj[:,:], other_traj=neighbor[:,:], navigation_extractor=navigation_extractor) # other with respect to reference, note that we will use the recorded traj and (RECORDED) heading, not the calculated heading for normalization inside this function

            caption_json = {
                **state_features,
                **dict1,
                # **dict2,
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
            
        # for k in caption_json.keys():
        #     print(k)

        return caption_json

    def get_agents_within(self, traj_history, inter_traj, other_neighbors, inter_traj_num, detection_range=20):
        # alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
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
        # idx = [i for i in range(len(signs_lanes)) if nearest_lane == signs_lanes[i] or signs_dist[i]<detection_threshold]
        idx = [i for i in range(len(signs_lanes)) if nearest_lane == signs_lanes[i]]
        if len(idx)==0:
            idx = [i for i in range(len(signs_lanes)) if signs_dist[i]<detection_threshold]
        valid_signs = {}
        if len(idx)==0:
            valid_signs = {}
        else:
            # for i in range(len(stop_signs_lanes)):
            distances = []
            for i in idx:
                # Angle check
                sign_unit_vector, distance = get_unit_vector(traj_history[-1,:2], signs_points[i])
                angle = rel_angle(traj_history[-1,:2], signs_points[i])
                category = get_rel_direction(angle, correction_angle=rel_angle(traj_history[-2,:2], traj_history[-1,:2]))
                if 'ahead' in category.lower() or (distance<10 and not 'behind' in category.lower()):
                    states_name = ['unknown', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution']
                    if states[i] == 0:
                        # print('unknown traffic light')
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
                # if not ('left' in category or 'right' in category or 'behind'):
                #     my_vector, _ = get_unit_vector(traj_history[-2,:2], traj_history[-1,:2])
                #     category = "on the left" if np.cross(my_vector, stop_sign_unit_vector)>0 else "on the right"
                # if 'behind' not in category.lower(): # if in front of me or on my right or left
                #     # check if the heading of one of its lanes is in the same heading as me
                #     distances.append(distance)
                #     valid_signs[i] = {'state':1, 'direction': category, 'distance': distance.round(2), 'relative location': 'next to the car' if distance<5 else 'Ahead'}
            
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
            # distances = np.linalg.norm(traj_history[-1,:2] - stop_signs_points, axis=-1)
            # idx = np.where(distances<150)[0]
            # for i in idx:
            #     stop_signs_lanes[i]
            #     for lane in stop_signs_lanes[i]:
            #         if LineString(lane_polylines[lane][:, :2]).distance(Point(traj_history[-1,:2])) < 4:
            #             # Angle check
            #             stop_sign_unit_vector, distance = get_unit_vector(traj_history[-1,:2], stop_signs_points[i])
            #             angle = rel_angle(traj_history[-1,:2], stop_signs_points[i])
            #             category = get_rel_direction(angle, correction_angle=rel_angle(traj_history[-2,:2], traj_history[-1,:2]))
            #             if 'left' in category or 'right' in category:
            #                 my_vector, _ = get_unit_vector(traj_history[-2,:2], traj_history[-1,:2])
            #                 category = "on the left" if np.cross(my_vector, stop_sign_unit_vector)>0 else "on the right"
            #             if 'behind' not in category.lower(): # if in front of me or on my right or left
            #                 valid_signs[i] = {'state':1, 'direction': category, 'distance': distance, 'relative location': 'next to the car' if distance<5 else 'Ahead'}
        else:
            # for i in range(len(stop_signs_lanes)):
            distances = []
            for i in idx:
                # Angle check
                stop_sign_unit_vector, distance = get_unit_vector(traj_history[-1,:2], stop_signs_points[i])
                angle = rel_angle(traj_history[-1,:2], stop_signs_points[i])
                category = get_rel_direction(angle, correction_angle=rel_angle(traj_history[-2,:2], traj_history[-1,:2]))
                if 'left' in category or 'right' in category:
                    my_vector, _ = get_unit_vector(traj_history[-2,:2], traj_history[-1,:2])
                    category = "on the left" if np.cross(my_vector, stop_sign_unit_vector)>0 else "on the right"
                if 'behind' not in category.lower(): # if in front of me or on my right or left
                    # check if the heading of one of its lanes is in the same heading as me
                    valid_signs[len(distances)] = {'state':1, 'direction': category, 'distance': distance.round(2), 'relative location': 'next to the car' if distance<5 else 'Ahead'}
                    distances.append(distance)
            if len(distances)>0:
                valid_signs = valid_signs[np.argmin(distances)]
            
        return valid_signs
        

        # stop_signs_points = stop_signs_points[idx]
        # stop_signs_dist = np.linalg.norm(traj_history[-1,:2] - stop_signs_points, axis=-1)
        # stop_signs_dist_argmin = np.argmin(stop_signs_dist)
        # if len(stop_signs_dist)>1:
        #     stop_signs_dist_min = stop_signs_dist[stop_signs_dist_argmin]

        # if stop_signs_dist_min < traffic_signs_threshold:
        #     print('')
        
        # xx = 0

    def t_seconds_rule(self, traj_future, traj_history, neighbors, seconds=3):
        # TODO: Make the 3 seconds configeratble 
        # three second rule can be applied to agents in front of us
        # can be applied to traffic lights, and stop signs. To show if the current speed is safe
        # if there are risk agents, the car should be advised to slow down (decelerate)
        # if instructed to speed up, this should be refused if it violates the 3-second rule
        hist_vel = abs_distance_to_velocity(traj_history[:,:2])[1:]*10 # *10 to convert it to m/s
        current_velocity = hist_vel[-1] # speed in m/s
        safe_relative_point = current_velocity*seconds # the relative future point compared to our reference point 
        safe_point = traj_history[-1,:2]+safe_relative_point # the absolute safe point
        safe_vector, safe_distance = get_unit_vector(traj_history[-1,:2], safe_point) # directional vector to the safe point and safe range
        ### Now check which cars are withen distance first, then withen frontal view
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
            # Distance check
            neighbor_vector, neighbor_distance = get_unit_vector(traj_history[-1,:2], neighbor[-1,:2])
            # Angle check
            angle = rel_angle(traj_history[-1,:2], neighbor[-1,:2])
            category = get_rel_direction(angle, correction_angle=rel_angle(traj_history[-2,:2], traj_history[-1,:2]))
            if 'left' in category or 'right' in category:
                my_vector, _ = get_unit_vector(traj_history[-2,:2], traj_history[-1,:2])
                category = "on the left" if np.cross(my_vector, neighbor_vector)>0 else "on the right"
            # print(category)
            # Distance check
            if neighbor_distance <= safe_distance:
                #"Directly ahead" "Directly behind" "Ahead" "Behind" "on the left" "on the right"
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

        # print(risk_agents_direction)
        # print(safe_agents_direction)
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
            # return traj
        if len(missing_indices) > 0:
            for i in range(2):  # Assuming we only need to interpolate the first two columns
                valid_indices = np.where(traj[:, i] != 0)[0]
                valid_values = traj[valid_indices, i]
                interpolator = interp1d(valid_indices, valid_values, fill_value="extrapolate")
                traj[missing_indices, i] = interpolator(missing_indices)
        return traj

    def get_relative_direction(self, reference_traj, other_traj, navigation_extractor=None, same_direction_threshold = 20/180*np.pi, opposite_direction_threshold = (180-20)/180*np.pi, crosswise_threshold_low = (90-20)/180*np.pi, crosswise_threshold_high = (90+20)/180*np.pi,
        directions = []):
        # ego_heading, other_heading, other_velocity,
        center, angle = reference_traj.copy()[-1][:2], reference_traj.copy()[-1][2]
        normalization_param = [center, angle]
        reference_traj[:,:5] = agent_norm(reference_traj[:,:], center, angle)
        other_traj[:,:5] = agent_norm(other_traj[:,:], center, angle)
        if sum(reference_traj[:,0] != 0)<=1:
            return "Unknown"
        
        # [self.get_vel_from_traj(other_traj[:,:2][ii], other_traj[:,:2][ii+1], time_difference=time_step[ii+1] - time_step[ii]) for ii in range(len(valid_states)-1)]
        other_velocity = get_vel_from_traj()
        other_velocity = abs_distance_to_velocity(other_traj[:,:2])
        max_speed = max(other_velocity)
        # if max_speed < self.kMaxSpeedForStationary and final_displacement < self.kMaxDisplacementForStationary:
        # heading direction compared to ego vehicle
        # assert ego_heading==0
        # print(f"Sample car heading: {ego_future[0,-1,2] * 180/np.pi}")
        if sum(other_velocity)==0:
            return 'Not moving'
        elif abs(other_heading) <= same_direction_threshold:
            return 'Moving in the same direction'
        elif abs(other_heading) >= opposite_direction_threshold:
            return 'Moving in the opposite direction'
        elif abs(other_heading) >= crosswise_threshold_low and abs(other_heading) <= crosswise_threshold_high:
            if other_heading<0:
                # return directions[3]
                return 'Moving crosswise from the left to the right of me'
            else:
                # return directions[4]
                return 'Moving crosswise from the right to the left of me'
        else:
            return 'Moving with an angle'
            # return directions[-1]

    def map_process_original(self, traj):
        '''
        Map point attributes
        self_point (x, y, h), left_boundary_point (x, y, h), right_boundary_pont (x, y, h), speed limit (float),
        self_type (int), left_boundary_type (int), right_boundary_type (int), traffic light (int), stop_point (bool), interpolating (bool), stop_sign (bool)
        '''
        vectorized_map = np.zeros(shape=(6, 300, 17))
        vectorized_crosswalks = np.zeros(shape=(4, 100, 3))
        agent_type = int(traj[-1][-1])

        # get all lane polylines
        lane_polylines = get_polylines(self.lanes)

        # get all road lines and edges polylines
        road_polylines = get_polylines(self.roads)

        # find current lanes for the agent
        ref_lane_ids = find_reference_lanes(agent_type, traj, lane_polylines)

        # find candidate lanes
        ref_lanes = []

        # get current lane's forward lanes
        for curr_lane, start in ref_lane_ids.items():
            candidate = depth_first_search(curr_lane, self.lanes, dist=lane_polylines[curr_lane][start:].shape[0], threshold=300)
            ref_lanes.extend(candidate)
        
        if agent_type != 2:
            # find current lanes' left and right lanes
            neighbor_lane_ids = find_neighbor_lanes(ref_lane_ids, traj, self.lanes, lane_polylines)

            # get neighbor lane's forward lanes
            for neighbor_lane, start in neighbor_lane_ids.items():
                candidate = depth_first_search(neighbor_lane, self.lanes, dist=lane_polylines[neighbor_lane][start:].shape[0], threshold=300)
                ref_lanes.extend(candidate)
            
            # update reference lane ids
            ref_lane_ids.update(neighbor_lane_ids)

        # remove overlapping lanes
        ref_lanes = remove_overlapping_lane_seq(ref_lanes)
        
        # get traffic light controlled lanes and stop sign controlled lanes
        traffic_light_lanes = {}
        stop_sign_lanes = []

        for signal in self.traffic_signals[self.hist_len-1].lane_states:
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self.lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        for i, sign in self.stop_signs.items():
            stop_sign_lanes.extend(sign.lane)
        
        # add lanes to the array
        added_lanes = 0
        for i, s_lane in enumerate(ref_lanes):
            added_points = 0
            if i > 5:
                break
            
            # create a data cache
            cache_lane = np.zeros(shape=(500, 17))

            for lane in s_lane:
                curr_index = ref_lane_ids[lane] if lane in ref_lane_ids else 0
                self_line = lane_polylines[lane][curr_index:]

                if added_points >= 500:
                    break      

                # add info to the array
                for point in self_line:
                    # self_point and type
                    cache_lane[added_points, 0:3] = point
                    cache_lane[added_points, 10] = self.lanes[lane].type

                    # left_boundary_point and type
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

                    # right_boundary_point and type
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

                    # speed limit
                    cache_lane[added_points, 9] = self.lanes[lane].speed_limit_mph / 2.237

                    # interpolating
                    cache_lane[added_points, 15] = self.lanes[lane].interpolating

                    # traffic_light
                    if lane in traffic_light_lanes.keys():
                        cache_lane[added_points, 13] = traffic_light_lanes[lane][0]
                        if np.linalg.norm(traffic_light_lanes[lane][1:] - point[:2]) < 3:
                            cache_lane[added_points, 14] = True
             
                    # add stop sign
                    if lane in stop_sign_lanes:
                        cache_lane[added_points, 16] = True

                    # count
                    added_points += 1
                    curr_index += 1

                    if added_points >= 500:
                        break             

            # scale the lane
            vectorized_map[i] = cache_lane[np.linspace(0, added_points, num=300, endpoint=False, dtype=np.int32)]
          
            # count
            added_lanes += 1

        # find surrounding crosswalks and add them to the array
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
        '''
        Map point attributes
        self_point (x, y, h), left_boundary_point (x, y, h), right_boundary_pont (x, y, h), speed limit (float),
        self_type (int), left_boundary_type (int), right_boundary_type (int), traffic light (int), stop_point (bool), interpolating (bool), stop_sign (bool)
        '''
        max_num_points = 500
        vectorized_map = np.zeros(shape=(self.drivable_lanes, max_num_points, 17))
        # additional_boundaries_L = np.zeros(shape=(self.drivable_lanes, max_num_points, 40))
        # additional_boundaries_R = np.zeros(shape=(self.drivable_lanes, max_num_points, 40))
        # additional_boundaries_ = np.zeros(shape=(50, max_num_points, 4))
        additional_boundaries_ = np.zeros(shape=(self.drivable_lanes, max_num_points, 4))
        vectorized_crosswalks = np.zeros(shape=(self.num_crosswalks, 100, 3))
        vectorized_traffic_lights = np.zeros(shape=(32, 3))
        vectorized_stop_signs = np.zeros(shape=(16, 2))
        vectorized_speed_bumps = np.zeros(shape=(self.num_crosswalks, 100, 3))
        agent_type = int(traj[-1][-1])

        # get all lane polylines
        lane_polylines = get_polylines(self.lanes)

        # get all road lines and edges polylines
        road_polylines = get_polylines(self.roads)
        
        # find current lanes for the agent
        # find_reference_lanes(agent_type, traj, lane_polylines)
        ref_lane_ids = find_all_lanes(agent_type, traj, lane_polylines)
        # find candidate lanes
        ref_lanes = ref_lane_ids
        
        if agent_type != 2:
            # find current lanes' left and right lanes
            # neighbor_lane_ids = find_neighbor_lanes(ref_lane_ids, traj, self.lanes, lane_polylines)
            neighbor_lane_ids = find_neighbor_lanes_fullmap(ref_lane_ids, traj, self.lanes, lane_polylines)

            # get neighbor lane's forward lanes
            for neighbor_lane, start in neighbor_lane_ids.items():
                candidate = depth_first_search(neighbor_lane, self.lanes, dist=lane_polylines[neighbor_lane][start:].shape[0], threshold=300)
                ref_lanes.extend(candidate)
            
            # update reference lane ids
            ref_lane_ids.update(neighbor_lane_ids)

        # remove overlapping lanes
        # ref_lanes = remove_overlapping_lane_seq(ref_lanes)
        
        # get traffic light controlled lanes and stop sign controlled lanes
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
        
        # add lanes to the array
        added_lanes = 0
        # if len(ref_lanes)>self.drivable_lanes:
        #     self.global_counter +=1
        #     print(self.global_counter)
        # found_boundaries = []
        # print(f"max num lanes = {len(ref_lanes)}")
        for i, s_lane in enumerate(ref_lanes):

            added_points = 0
            # if i > 1:
            #     break
            if i > (self.drivable_lanes-1): # DONE: make this 5 (that mean 6 drivable lanes variable)
                break
            
            # create a data cache
            cache_lane = np.zeros(shape=(max_num_points, 17))
            # cache_other_boundries_L = np.zeros(shape=(max_num_points, 40))
            # cache_other_boundries_R = np.zeros(shape=(max_num_points, 40))

            # for lane in s_lane:
            if True:
                lane = s_lane
                curr_index = ref_lane_ids[lane] if lane in ref_lane_ids else 0
                self_line = lane_polylines[lane][curr_index:]
                if added_points >= max_num_points:
                    # print("EXCEEDED MAX NUM POINTS")
                    break

                # add info to the array
                for point in self_line:
                    # self_point and type
                    cache_lane[added_points, 0:3] = point
                    cache_lane[added_points, 10] = self.lanes[lane].type

                    # if len(self.lanes[lane].left_boundaries)>10 or len(self.lanes[lane].right_boundaries)>10:
                    #     print("--"*10)
                    #     print(f"L: {len(self.lanes[lane].left_boundaries)}")
                    #     print(f"R: {len(self.lanes[lane].right_boundaries)}")
                    #     print("--"*10)
                    
                    # found_boundaries.extend([self.lanes[lane].left_boundaries[bi].boundary_feature_id for bi in range(len(self.lanes[lane].left_boundaries))])
                    # found_boundaries.extend([self.lanes[lane].right_boundaries[bi].boundary_feature_id for bi in range(len(self.lanes[lane].right_boundaries))])
                    # left_boundary_point and type
                    # for i_left_boundary, left_boundary in enumerate(self.lanes[lane].left_boundaries):
                    if len(self.lanes[lane].left_boundaries)>0:
                        i_left_boundary, left_boundary = 0, self.lanes[lane].left_boundaries[0]
                        left_boundary_id = left_boundary.boundary_feature_id
                        left_start = left_boundary.lane_start_index
                        left_end = left_boundary.lane_end_index
                        left_boundary_type = left_boundary.boundary_type # road line type
                        if left_boundary_type == 0:
                            left_boundary_type = self.roads[left_boundary_id].type + 8 # road edge type
                        
                        # if left_start <= curr_index <= left_end:
                        left_boundary_line = road_polylines[left_boundary_id]
                        nearest_point = find_neareast_point(point, left_boundary_line)
                        if i_left_boundary==0:
                            cache_lane[added_points, 3:6] = nearest_point
                            cache_lane[added_points, 11] = left_boundary_type
                        # elif np.linalg.norm(point - nearest_point) < np.linalg.norm(point - cache_lane[added_points, 3:6]):
                        #     cache_lane[added_points, 3:6] = nearest_point
                        #     cache_lane[added_points, 11] = left_boundary_type
                        
                        # else:
                        #     if i_left_boundary<=10:
                        #         cache_other_boundries_L[added_points, i_left_boundary*3-3:i_left_boundary*3] = nearest_point
                        #         cache_other_boundries_L[added_points, 30+i_left_boundary-1] = left_boundary_type

                    # right_boundary_point and type
                    # for i_right_boundary, right_boundary in enumerate(self.lanes[lane].right_boundaries):
                    if len(self.lanes[lane].right_boundaries)>0:
                        i_right_boundary, right_boundary = 0, self.lanes[lane].right_boundaries[0]
                        right_boundary_id = right_boundary.boundary_feature_id
                        right_start = right_boundary.lane_start_index
                        right_end = right_boundary.lane_end_index
                        right_boundary_type = right_boundary.boundary_type # road line type
                        if right_boundary_type == 0:
                            right_boundary_type = self.roads[right_boundary_id].type + 8 # road edge type

                        # if right_start <= curr_index <= right_end:
                        right_boundary_line = road_polylines[right_boundary_id]
                        nearest_point = find_neareast_point(point, right_boundary_line)
                        if i_right_boundary==0:
                            cache_lane[added_points, 6:9] = nearest_point
                            cache_lane[added_points, 12] = right_boundary_type
                        # elif np.linalg.norm(point - nearest_point) < np.linalg.norm(point - cache_lane[added_points, 6:9]):
                        #     cache_lane[added_points, 6:9] = nearest_point
                        #     cache_lane[added_points, 12] = right_boundary_type

                        # else:
                        #     if i_right_boundary<=10:
                        #         cache_other_boundries_R[added_points, i_right_boundary*3-3:i_right_boundary*3] = nearest_point
                        #         cache_other_boundries_R[added_points, 30+i_right_boundary-1] = right_boundary_type


                    # speed limit
                    cache_lane[added_points, 9] = self.lanes[lane].speed_limit_mph / 2.237

                    # interpolating
                    cache_lane[added_points, 15] = self.lanes[lane].interpolating

                    # traffic_light
                    if lane in traffic_light_lanes.keys():
                        cache_lane[added_points, 13] = traffic_light_lanes[lane][0]
                        if np.linalg.norm(traffic_light_lanes[lane][1:] - point[:2]) < 3:
                            cache_lane[added_points, 14] = True
             
                    # add stop sign
                    if lane in stop_sign_lanes:
                        cache_lane[added_points, 16] = True

                    # count
                    added_points += 1
                    curr_index += 1

                    if added_points >= max_num_points:
                        # print("EXCEEDED MAX NUM POINTS")
                        break             

            # scale the lane
            # vectorized_map[i] = cache_lane[np.linspace(0, added_points, num=300, endpoint=False, dtype=np.int32)]
            vectorized_map[i] = cache_lane
            # additional_boundaries_L[i] = cache_other_boundries_L
            # additional_boundaries_R[i] = cache_other_boundries_R
            
            
            
          
            # count
            added_lanes += 1
        # find surrounding crosswalks and add them to the array
        added_cross_walks = 0
        added_speed_bumps = 0
        detection = Polygon([(0, -5), (50, -20), (50, 20), (0, 5)])
        detection = affine_transform(detection, [1, 0, 0, 1, traj[-1][0], traj[-1][1]])
        detection = rotate(detection, traj[-1][2], origin=(traj[-1][0], traj[-1][1]), use_radians=True)

        for _, crosswalk in self.crosswalks.items():
            polygon = Polygon([(point.x, point.y) for point in crosswalk.polygon])
            polyline = polygon_completion(crosswalk.polygon)
            polyline = polyline[np.linspace(0, polyline.shape[0], num=100, endpoint=False, dtype=np.int32)]

            # if detection.intersects(polygon):
            if True:
                vectorized_crosswalks[added_cross_walks, :polyline.shape[0]] = polyline
                added_cross_walks += 1
            
            if added_cross_walks >= self.num_crosswalks:
                break
        
        for _, speed_bump in self.speed_bumps.items():
            polygon = Polygon([(point.x, point.y) for point in speed_bump.polygon])
            polyline = polygon_completion(speed_bump.polygon)
            polyline = polyline[np.linspace(0, polyline.shape[0], num=100, endpoint=False, dtype=np.int32)]

            # if detection.intersects(polygon):
            if True:
                vectorized_speed_bumps[added_speed_bumps, :polyline.shape[0]] = polyline
                added_speed_bumps += 1
            
            if added_speed_bumps >= len(vectorized_speed_bumps):
                break

        # missing_boundaries_ids = [bi for bi in self.roads.keys() if not bi in np.unique(found_boundaries)] # this takes ~0.1s to compute
        missing_boundaries_ids = list(self.roads.keys()) # all boundaries
        if len(missing_boundaries_ids)>0:
            # print("Missing road boundaries not captured: ", end='')
            # print(len(missing_boundaries_ids))
            for added_roads, boundary_id in enumerate(missing_boundaries_ids):
                if boundary_id in list(self.roads_boundary.keys()):
                    boundary_type = self.roads[boundary_id].type # road boundary type
                else:
                    boundary_type = self.roads[boundary_id].type + 8 # road edge type
                additional_boundaries_[added_roads, :min(additional_boundaries_.shape[1], road_polylines[boundary_id].shape[0]), :3] = road_polylines[boundary_id][:min(additional_boundaries_.shape[1], road_polylines[boundary_id].shape[0])]
                additional_boundaries_[added_roads, :min(additional_boundaries_.shape[1], road_polylines[boundary_id].shape[0]), 3] = boundary_type
        
        ## Sorting based on distance
        boundaries_min_distance = np.linalg.norm(traj[-1, :2] - additional_boundaries_[...,:2], axis=-1).min(-1)
        # additional_boundaries_[boundaries_min_distance>311] = 0.0
        additional_boundaries_ = additional_boundaries_[np.argsort(boundaries_min_distance)]
        # vectorized_map_min_distance = np.linalg.norm(traj[-1, :2] - vectorized_map[...,:2], axis=-1).min(-1)
        # vectorized_map[vectorized_map_min_distance>311] = 0.0
        # additional_boundaries_L[vectorized_map_min_distance>311] = 0.0
        # additional_boundaries_R[vectorized_map_min_distance>311] = 0.0

        # vectorized_map = vectorized_map[np.argsort(vectorized_map_min_distance)] # it is already sorted
        # additional_boundaries_L = additional_boundaries_L[np.argsort(vectorized_map_min_distance)]
        # additional_boundaries_R = additional_boundaries_R[np.argsort(vectorized_map_min_distance)]

        
        if sum(vectorized_crosswalks[...,0].sum(-1)!=0)>0:
            vectorized_crosswalks_min_distance = np.linalg.norm(traj[-1, :2] - vectorized_crosswalks[...,:2], axis=-1).min(-1)
            # vectorized_crosswalks[vectorized_crosswalks_min_distance>311] = 0.0
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
            # vectorized_stop_signs[np.array([np.linalg.norm(traj[-1, :2] - vectorized_stop_signs[i,:2]) for i in range(len(vectorized_stop_signs))]) > 311] = [0,0]
        
        if sum(vectorized_speed_bumps[...,0].sum(-1)!=0)>0:
            vectorized_speed_bumps_min_distance = np.linalg.norm(traj[-1, :2] - vectorized_speed_bumps[...,:2], axis=-1).min(-1)
            vectorized_speed_bumps[vectorized_speed_bumps_min_distance>311] = 0.0
            vectorized_speed_bumps = vectorized_speed_bumps[np.argsort(vectorized_speed_bumps_min_distance)]
            
            vectorized_speed_bumps[...,0].sum(-1)==0
            vectorized_speed_bumps[vectorized_speed_bumps[...,0].sum(-1)==0]
            vectorized_speed_bumps[vectorized_speed_bumps[...,0].sum(-1)==0]=[0,0,0]
        
        # return vectorized_map.astype(np.float32), vectorized_crosswalks.astype(np.float32), additional_boundaries_L.astype(np.float32), additional_boundaries_R.astype(np.float32), additional_boundaries_.astype(np.float32)
        return vectorized_map.astype(np.float32), vectorized_crosswalks.astype(np.float32), additional_boundaries_.astype(np.float32), vectorized_traffic_lights.astype(np.float32), vectorized_stop_signs.astype(np.float32), vectorized_speed_bumps.astype(np.float32)

    def get_direction_plausibility_vectorized(self, ego, map_lanes, map_crosswalks, ground_truth, additional_boundaries_, traffic_lights, stop_signs, agent_json_):
        # Vectorized operations for distance calculations
        ego_position = ego[0, :, :2]
        lane_positions = map_lanes[:, :, :2]
        
        bike_lanes_mask = map_lanes[:,:,10]!=3
        undefined_lanes_mask = map_lanes[:,:,10]!=0
        valid_lane_mask = bike_lanes_mask * undefined_lanes_mask
        lane_positions[~valid_lane_mask] = 10000
        
        distances = np.linalg.norm(lane_positions, axis=2)
        min_distances_indices = np.argmin(distances, axis=1)
        
        # Filtering lanes based on distance
        valid_lanes = distances[np.arange(distances.shape[0]), min_distances_indices] < 1  # 1 meter threshold
        
        # Vectorized operation for finding direction differences
        angles = np.degrees(map_lanes[np.arange(map_lanes.shape[0]), min_distances_indices, 2])
        valid_angles = np.abs(angles) < 15  # 15 degrees threshold

        valid_lanes = np.logical_and(valid_lanes, valid_angles)

        # Collecting valid lane indices
        possible_map_lanes = map_lanes[valid_lanes]
        
        # Assuming 'DirectionClassifier' is already defined
        direction_classifier = DirectionClassifier()
        possible_directions = []
        for lane in possible_map_lanes:
            start_point = lane[0]
            end_point = lane[-1]
            two_points_lane = np.vstack((start_point, end_point))
            direction_str, direction_cls, _, _ = direction_classifier.classify_lane(two_points_lane)
            possible_directions.append(direction_classifier.classes[direction_cls])
        
        return np.unique(possible_directions)

    def get_direction_plausibility(self, ego, map_lanes, map_crosswalks, ground_truth, additional_boundaries_, traffic_lights, stop_signs, agent_json_):
        # possible_map_lanes = np.zeros_like(map_lanes)
        
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
            # possible_map_lanes[0,:len(fill_in_lane)] = fill_in_lane
        
        # t_1 = time.time()
        for i in range(len(map_lanes)):
            if i==current_lane_idx:
                continue
            map_lanes[i][valid_lane_mask[i]]
            # Calculate the differences along each dimension and compute distances
            array2 = map_lanes[i][valid_lane_mask[i]][:,:3]
            if len(array2)==0:
                continue
            possible_map_lanes_cache = []
            
            for j in possible_map_lanes:
                array1 = j[:,:3]
                # Calculate the differences along each dimension and compute distances
                distances = np.sqrt(((array1[:,:2][:, np.newaxis] - array2[:,:2])**2).sum(axis=2))
                # Find the index of the minimum distance
                min_index = np.unravel_index(np.argmin(distances), distances.shape)
                # Retrieve the closest points and the minimum distance
                closest_points = (array1[min_index[0]], array2[min_index[1]])
                minimum_distance = distances[min_index]
                if minimum_distance<1: # if 1 meter between two points
                    # check angle
                    angle_difference = np.degrees(array2[min_index[1], 2] - array1[min_index[0], 2])
                    if angle_difference<15:
                        fill_in_lane = map_lanes[i, min_index[1]:]
                        fill_in_mask = valid_lane_mask[i, min_index[1]:]
                        possible_map_lanes_cache.append(fill_in_lane[fill_in_mask])
            for cache_lane in possible_map_lanes_cache:
                possible_map_lanes.append(cache_lane)
                
        # print(f'Loop1: {time.time()-t_1:.2f}s') 
        max_length = max([len(arr) for arr in possible_map_lanes])
        # Step 2: Pad each array to have the maximum length in the first dimension
        padded_arrays = [
            np.pad(arr, pad_width=((0, max_length - arr.shape[0]), (0, 0)), mode='constant', constant_values=0)
            for arr in possible_map_lanes
        ]
        possible_map_lanes = np.stack(padded_arrays)
        possible_map_lanes_mask = possible_map_lanes[...,0]!=0

        
        current_speed = np.linalg.norm(ego[-1,3:5]) # m/s
        starting_point = possible_map_lanes[0][possible_map_lanes_mask[0]][0,:3] # x,y,h
        ## Filtering lanes based on maximum reachable distance based on current speed and maximum possible acceleration (extreme acceleration of 2.5 m/s)
        # 1 m/s^2 = 3.6 km/h/s, i.e. the km/h achived in 1 second, 
        # fast cars usually have a maximum accelearion below 100 km/h/8s, 
        # aggressive_accel_threshold = 2.26 # m/s^2 # 2.26 m/s^2 -> 65.09 km/h/8s -> 8.14 km/h/s -> 8.14 km/h increase in 1 second -> 8.14/3.6 m/s -> 2.26 m/s
        max_possible_speed_increase = 2.5 # m/s maximum possible increase in speed -> 72 km/h/8s
        maximum_future_speeds = [current_speed+(i*max_possible_speed_increase) for i in range(8)] # m/s for 8 the current speed and 7 seconds in the future speeds
        max_travel_distance = sum(maximum_future_speeds)
        ## Filtering lanes based on minimum reachable distance based on current speed and maximum possible deceleration (extreme deceleration of -2.5 m/s)
        max_possible_speed_decrease = max_possible_speed_increase # m/s
        min_future_speeds = [current_speed-(i*max_possible_speed_decrease) for i in range(8) if current_speed-(i*max_possible_speed_decrease)>0] # m/s for 8 the current speed and 7 seconds in the future speeds
        min_travel_distance = sum(min_future_speeds)
        starting_point = possible_map_lanes[0][possible_map_lanes_mask[0]][0,:3] # x,y,h
        # filtering:
        map_points_distances = np.linalg.norm(starting_point[:2] - possible_map_lanes[possible_map_lanes_mask][:,:2], axis=1)
        reachable_points_mask = (map_points_distances < max_travel_distance) * (map_points_distances > min_travel_distance)
        possible_map_lanes_mask[possible_map_lanes_mask] = reachable_points_mask * possible_map_lanes_mask[possible_map_lanes_mask]

        # print(f'possible lanes shape:{possible_map_lanes_mask.shape}')
        # print(f'Num valid points:{possible_map_lanes_mask.sum()}')
        direction_classifier = DirectionClassifier(step_t=1)
        possible_directions = []
        # t_2 = time.time()
        found_lanes = 0
        for i in range(len(possible_map_lanes)):
            if len(possible_map_lanes[i][possible_map_lanes_mask[i]])>0:
                two_points_lane = np.vstack((starting_point, possible_map_lanes[i][possible_map_lanes_mask[i]][-1,:3]))
                direction_str, direction_cls,_,_ = direction_classifier.classify_lane(two_points_lane)
                possible_directions.append(direction_classifier.classes[direction_cls])
                found_lanes+=1
        # print(f"Original found lanes {found_lanes}")
        
        # print(f'Loop2: {time.time()-t_2:.2f}s')
        # for i in range(len(possible_map_lanes)):
        #     possible_map_lanes[0][possible_map_lanes_mask[0]] possible_map_lanes[i][possible_map_lanes_mask[i]]
        # get_vel_from_traj(ego[:,:2])
        possible_directions = np.unique(possible_directions)
        # print(possible_directions)
        return possible_directions

    def get_direction_plausibility_03(self, ego_lane, drivable_lanes, ego, map_lanes=None, map_crosswalks=None, ground_truth=None, additional_boundaries_=None, traffic_lights=None, stop_signs=None, agent_json_=None):
        possible_map_lanes = drivable_lanes
        possible_map_lanes[possible_map_lanes[...,0]==0] = np.inf
        # map_lanes = map_lanes[:,::2]
        ego = ego[0,:,:]
        # current_location_on_map = map_lanes
        # bike_lanes_mask = map_lanes[:,:,10]!=3
        # undefined_lanes_mask = map_lanes[:,:,10]!=0
        # valid_lane_mask = bike_lanes_mask * undefined_lanes_mask
        # map_lanes[~valid_lane_mask] = np.inf


        map_lanes_norms = np.linalg.norm(ego_lane[0,...,:2], axis=-1)
        # map_lanes_norms = np.linalg.norm(map_lanes[...,:2], axis=-1)
        # Find the index of the minimum value in the 2D array of norms
        min_index_flat = np.argmin(map_lanes_norms) # lane is already 1D
        # Convert the flat index to a tuple (row, column)
        # min_index_2d = np.unravel_index(min_index_flat, map_lanes_norms.shape)
        reference_slice = ego_lane[0][min_index_flat]
        # current_lane_idx, current_point_idx = min_index_2d
        
        starting_point = reference_slice[:3]# x,y,h

        # Mask the map_lanes array where values are np.inf
        # masked_map_lanes = np.ma.masked_where(map_lanes == np.inf, map_lanes)
        direction_classifier = DirectionClassifier(step_t=1)

        ##
        possible_directions = []
        possible_classes = []
        found_lanes = 0
        lines_to_plot_tupple = {}
        possible_colors = ['g', 'b', 'c', 'm', 'yellow', 'purple', 'pink', 'orange']
        for i in range(len(possible_map_lanes)):
            if np.any(possible_map_lanes[i][:,0]!=np.inf):
                target_lane = possible_map_lanes[i][possible_map_lanes[i][:,0]!=np.inf]
                # cropping lane based on min and max distance
                # drivable_target_lane = target_lane[(np.linalg.norm(target_lane[:,:2], axis=-1)>min_travel_distance) * (np.linalg.norm(target_lane[:,:2], axis=-1)<max_travel_distance)]
                drivable_target_lane = target_lane # The lanes already cropped based on min and max distance
                if len(drivable_target_lane)>0:
                    two_points_lane_maximum_distance = np.vstack((starting_point, drivable_target_lane[-1,:3]))
                    two_points_lane_halfway = np.vstack((starting_point, drivable_target_lane[len(drivable_target_lane)//2,:3]))
                    two_points_lane_quarterway = np.vstack((starting_point, drivable_target_lane[len(drivable_target_lane)//4,:3]))
                    two_points_lane_shortest_path = np.vstack((starting_point, drivable_target_lane[0,:3]))
                    four_splits_directions = []
                    for two_points_lane in [two_points_lane_shortest_path, two_points_lane_quarterway, two_points_lane_halfway, two_points_lane_maximum_distance]:    
                        direction_str, direction_cls,_,_ = direction_classifier.classify_lane(deepcopy(two_points_lane))
                        four_splits_directions.append((direction_str, direction_cls))
                    # print(four_splits_directions)
                    # entries with straight after turn are removed, as this require manuvering taking two turns and the current rule-based detection does not capture that
                    four_splits_directions = remove_entries_after_turn(four_splits_directions) 
                    # print(f"--> {four_splits_directions}")
                    for direction_str, direction_cls in four_splits_directions:
                    # for two_points_lane in [two_points_lane_shortest_path, two_points_lane_quarterway, two_points_lane_halfway, two_points_lane_maximum_distance]:    
                        # We need to detect direction at maximum drivable point, minimum drivable point, and half way
                        # direction_str, direction_cls,_,_ = direction_classifier.classify_lane(deepcopy(two_points_lane))
                        if direction_cls!=-1:    
                            starting_point_idx = np.linalg.norm((drivable_target_lane[:,:3] - starting_point)[:,:2], axis=-1).argmin()
                            lines_to_plot_tupple[found_lanes] = (direction_cls, drivable_target_lane[starting_point_idx:])
                            possible_directions.append(direction_classifier.classes[direction_cls])
                            possible_classes.append(direction_cls)
                            found_lanes+=1
                    # two_points_lane = np.vstack((starting_point, drivable_target_lane[-1,:3]))
                    # direction_str, direction_cls,_,_ = direction_classifier.classify_lane(two_points_lane)
                    # possible_directions.append(direction_classifier.classes[direction_cls])
                    # found_lanes+=1
        # print(found_lanes)
        # print(np.unique(possible_directions))
        # print(time.time()-t_)

        return np.unique(possible_directions), lines_to_plot_tupple, possible_map_lanes

    # def get_direction_plausibility_02(self, ego, map_lanes, map_crosswalks, ground_truth, additional_boundaries_, traffic_lights, stop_signs, agent_json_):
    #     num_hops = 3
    #     # distance_threshold = 2
    #     # angle_threshold = 10

    #     distance_threshold = 1
    #     angle_threshold = 5
    #     # possible_map_lanes = np.zeros_like(map_lanes)
    #     map_lanes = map_lanes[:,::2]
    #     ego = ego[0,:,:]
    #     current_location_on_map = map_lanes
    #     bike_lanes_mask = map_lanes[:,:,10]!=3
    #     undefined_lanes_mask = map_lanes[:,:,10]!=0
    #     valid_lane_mask = bike_lanes_mask * undefined_lanes_mask
    #     map_lanes[~valid_lane_mask] = np.inf

    #     map_lanes_norms = np.linalg.norm(map_lanes[...,:2], axis=-1)
    #     # Find the index of the minimum value in the 2D array of norms
    #     min_index_flat = np.argmin(map_lanes_norms)
    #     # Convert the flat index to a tuple (row, column)
    #     min_index_2d = np.unravel_index(min_index_flat, map_lanes_norms.shape)
    #     reference_slice = map_lanes[min_index_2d]
    #     current_lane_idx, current_point_idx = min_index_2d
    #     possible_map_lanes = np.ones_like(map_lanes)*np.inf
        
    #     ## finding max and min possible travel distances
    #     current_speed = np.linalg.norm(ego[-1,3:5]) # m/s
    #     ## Filtering lanes based on maximum reachable distance based on current speed and maximum possible acceleration (extreme acceleration of 2.5 m/s)
    #     # 1 m/s^2 = 3.6 km/h/s, i.e. the km/h achived in 1 second, 
    #     # fast cars usually have a maximum accelearion below 100 km/h/8s, 
    #     # aggressive_accel_threshold = 2.26 # m/s^2 # 2.26 m/s^2 -> 65.09 km/h/8s -> 8.14 km/h/s -> 8.14 km/h increase in 1 second -> 8.14/3.6 m/s -> 2.26 m/s
    #     # max_possible_speed_increase = 2.5 # m/s maximum possible increase in speed -> 72 km/h/8s
    #     max_possible_speed_increase = 1.5625 # 12.5 # m/s -> 45 km/h/8s -> 100 m/8s
    #     speed_limit = reference_slice[9]
    #     maximum_future_speeds = [current_speed+(i*max_possible_speed_increase) for i in range(8)] # m/s for 8 the current speed and 7 seconds in the future speeds
    #     max_travel_distance = sum(maximum_future_speeds)
    #     ## Filtering lanes based on minimum reachable distance based on current speed and maximum possible deceleration (extreme deceleration of -2.5 m/s)
    #     max_possible_speed_decrease = max_possible_speed_increase # m/s
    #     min_future_speeds = [current_speed-(i*max_possible_speed_decrease) for i in range(8) if current_speed-(i*max_possible_speed_decrease)>0] # m/s for 8 the current speed and 7 seconds in the future speeds
    #     min_travel_distance = sum(min_future_speeds)
    #     starting_point = reference_slice[:3]# x,y,h

    #     max_distance = max(min(speed_limit*8, max_travel_distance), 60)
    #     max_travel_distance = max_distance
    #     min_distance = max(0, min_travel_distance)
    #     min_travel_distance = min_distance
    #     # print('-----------------')
    #     # print(f'max distance:{max_distance}')
    #     # print(f'min distance:{min_distance}')
    #     # if speed_limit*8 < max_travel_distance:
    #     #     print('max distance exceeds speed limit!')

    #     # cropping lanes based on maximum distance (not minimum, becaus we will use that to connect lanes to current lane)
    #     map_lanes_norm_ = np.linalg.norm(map_lanes[...,:2], axis=-1)
    #     map_lanes[map_lanes_norm_>max_distance] = np.inf
        
    #     if current_lane_idx is not None and current_lane_idx is not None:
    #         map_lanes[current_lane_idx, :current_point_idx] = np.inf
    #         possible_map_lanes[current_lane_idx] = map_lanes[current_lane_idx]

    #     t_ = time.time()
    #     # Mask the map_lanes array where values are np.inf
    #     # masked_map_lanes = np.ma.masked_where(map_lanes == np.inf, map_lanes)
    #     direction_classifier = DirectionClassifier(step_t=1)

    #     lane_indices = np.arange(map_lanes.shape[0])
    #     for _ in range(num_hops):
    #         # if i==current_lane_idx:
    #         #     continue
    #         # if (map_lanes[1]!=np.inf).sum() ==0:
    #         #     continue
    #         possible_map_points = possible_map_lanes[...,:3][possible_map_lanes[...,0]!=np.inf]
            
    #         differances = map_lanes[np.newaxis, :, :, :3] - possible_map_points[:, np.newaxis, np.newaxis, :] # [number of possible points, number of map lanes, number of lane points], comparing each possible point to all map points
    #         # masked_differences = np.ma.subtract(masked_map_lanes[np.newaxis, :, :, :3], possible_map_points[:, np.newaxis, np.newaxis, :])
    #         distances = np.linalg.norm(differances[...,:2], axis=-1)
    #         # distances = np.linalg.norm(differances, axis=-1)
    #         # Find the minimum distances
            
    #         min_distances_across_points = np.min(distances, axis=0)  # Shape (500 lanes, 500)
    #         min_point_indices = np.argmin(distances, axis=0) # Shape (500 lanes, 500)
    #         min_indices_within_lane = np.argmin(min_distances_across_points, axis=1)

    #         # Get the corresponding closest point indices out of the number of possible points for each lane
    #         # closest_point_indices = min_point_indices[np.arange(possible_map_lanes.shape[0]), min_indices_within_lane]  # Shape (500,)
    #         closest_point_indices = min_point_indices[lane_indices, min_indices_within_lane]  # Shape (500,)
    #         # Output the results, assume len(possible_map_points)=55
    #         # min_indices_within_lane : Index of the closest point within each lane (500,) -> indix of lanes
    #         # closest_point_indices : Index of the closest point out of the 55 points for each lane (500,) -> corresponding index in possible_map_points
    #         # Extract the closest points from possible_map_points for each lane
    #         selected_points_from_possible = possible_map_points[closest_point_indices]
    #         # Create a lane index array to pair with min_indices_within_lane for proper indexing            
    #         # Extract the closest points in each lane from map_lanes
    #         selected_points_from_map_lanes = map_lanes[lane_indices, min_indices_within_lane, ..., :3]
    #         # (selected_points_from_map_lanes - selected_points_from_possible)
    #         np.degrees(selected_points_from_map_lanes[:, 2] - selected_points_from_possible[:, 2])
    #         angle_is_valid = np.degrees(selected_points_from_map_lanes[:, 2] - selected_points_from_possible[:, 2])<angle_threshold
    #         # distance_threshold = 1.2
    #         lane_is_within_1m = np.any(min_distances_across_points<distance_threshold, axis=-1)
    #         new_possible_lanes_mask = lane_is_within_1m * angle_is_valid
    #         new_lanes_starting_point_idx = min_indices_within_lane[new_possible_lanes_mask]
    #         new_lanes_idx = np.where(new_possible_lanes_mask)[0]
    #         for new_lane_i in range(len(new_lanes_idx)):
    #             possible_map_lanes[new_lanes_idx[new_lane_i], new_lanes_starting_point_idx[new_lane_i]:] = map_lanes[new_lanes_idx[new_lane_i], new_lanes_starting_point_idx[new_lane_i]:]
        
    #     ##
    #     possible_directions = []
    #     possible_classes = []
    #     found_lanes = 0
    #     lines_to_plot_tupple = {}
    #     possible_colors = ['g', 'b', 'c', 'm', 'yellow', 'purple', 'pink', 'orange']
    #     possible_map_lanes = map_lanes
    #     for i in range(len(possible_map_lanes)):
    #         if np.any(possible_map_lanes[i][:,0]!=np.inf):
    #             target_lane = possible_map_lanes[i][possible_map_lanes[i][:,0]!=np.inf]
    #             # cropping lane based on min and max distance
    #             drivable_target_lane = target_lane[(np.linalg.norm(target_lane[:,:2], axis=-1)>min_travel_distance) * (np.linalg.norm(target_lane[:,:2], axis=-1)<max_travel_distance)]
    #             if len(drivable_target_lane)>0:
    #                 two_points_lane_maximum_distance = np.vstack((starting_point, drivable_target_lane[-1,:3]))
    #                 two_points_lane_halfway = np.vstack((starting_point, drivable_target_lane[len(drivable_target_lane)//2,:3]))
    #                 two_points_lane_quarterway = np.vstack((starting_point, drivable_target_lane[len(drivable_target_lane)//4,:3]))
    #                 two_points_lane_shortest_path = np.vstack((starting_point, drivable_target_lane[0,:3]))
    #                 for two_points_lane in [two_points_lane_maximum_distance, two_points_lane_halfway, two_points_lane_quarterway, two_points_lane_shortest_path]:
    #                     # We need to detect direction at maximum drivable point, minimum drivable point, and half way
    #                     direction_str, direction_cls,_,_ = direction_classifier.classify_lane(two_points_lane)
    #                     if direction_cls!=-1:
    #                         starting_point_idx = np.linalg.norm((drivable_target_lane[:,:3] - starting_point)[:,:2], axis=-1).argmin()
    #                         lines_to_plot_tupple[found_lanes] = (direction_cls, drivable_target_lane[starting_point_idx:])
    #                         possible_directions.append(direction_classifier.classes[direction_cls])
    #                         possible_classes.append(direction_cls)
    #                         found_lanes+=1
    #                 # two_points_lane = np.vstack((starting_point, drivable_target_lane[-1,:3]))
    #                 # direction_str, direction_cls,_,_ = direction_classifier.classify_lane(two_points_lane)
    #                 # possible_directions.append(direction_classifier.classes[direction_cls])
    #                 # found_lanes+=1
    #     # print(found_lanes)
    #     # print(np.unique(possible_directions))
    #     # print(time.time()-t_)

    #     return np.unique(possible_directions), lines_to_plot_tupple, possible_map_lanes
        #     new_lanes_starting_point_idx = min_indices_within_lane[new_possible_lanes_mask]
        #     new_lanes_idx = np.where(new_possible_lanes_mask)[0]

        #     possible_map_lanes[new_lanes_idx]

            
        #     map_lanes[new_possible_lanes_mask]

        #     closest_point_indices

        #     np.where(new_possible_lanes_mask)
            
        #     closest_point_indices[13]



        #     possible_map_points


        #     np.unravel_index(min_point_indices, distances.shape)
        #     min_distances = np.min(distances, axis=2)  # Shape (55, 499)
        #     # Each element at (i, j) is the minimum distance from the i-th point to any point in the j-th lane
        #     # Find the indices of these minimum distances
        #     min_indices = np.argmin(distances, axis=2)  # Shape (55, 499)
            

        #     distances = possible_map_points[:, np.newaxis, np.newaxis, :] - lanes_array[np.newaxis, :, :, :]


        #     possible_map_lanes_ = possible_map_lanes.copy()
        #     possible_map_lanes_[possible_map_lanes_==np.inf] = -np.inf
        #     np.linalg.norm(possible_map_lanes_[...,np.newaxis,:2] - map_lanes[1,...,:2][np.newaxis, np.newaxis])
        #     distances = np.linalg.norm(possible_map_lanes_[...,:2] - map_lanes[1,...,:2], axis=-1) # 500 lanes each of 500 points compared to the 500 points of the target lane
        #     np.unravel_index(np.argmin(distances), distances.shape)
            
        # distances = np.linalg.norm(possible_map_lanes_[...,:2] - map_lanes[...,:2], axis=-1)
        # distances.argmin(-1)
        # distances = np.sqrt(((array1[:,:2][:, np.newaxis] - array2[:,:2])**2).sum(axis=2))
        

        # t_1 = time.time()
        # for i in range(len(map_lanes)):
        #     if i==current_lane_idx:
        #         continue
        #     map_lanes[i][valid_lane_mask[i]]
        #     # Calculate the differences along each dimension and compute distances
        #     array2 = map_lanes[i][valid_lane_mask[i]][:,:3]
        #     if len(array2)==0:
        #         continue
        #     possible_map_lanes_cache = []

        #     for j in possible_map_lanes:
        #         array1 = j[:,:3]
        #         # Calculate the differences along each dimension and compute distances
        #         distances = np.sqrt(((array1[:,:2][:, np.newaxis] - array2[:,:2])**2).sum(axis=2))
        #         # Find the index of the minimum distance
        #         min_index = np.unravel_index(np.argmin(distances), distances.shape)
        #         # Retrieve the closest points and the minimum distance
        #         closest_points = (array1[min_index[0]], array2[min_index[1]])
        #         minimum_distance = distances[min_index]
        #         if minimum_distance<1: # if 1 meter between two points
        #             # check angle
        #             angle_difference = np.degrees(array2[min_index[1], 2] - array1[min_index[0], 2])
        #             if angle_difference<15:
        #                 fill_in_lane = map_lanes[i, min_index[1]:]
        #                 fill_in_mask = valid_lane_mask[i, min_index[1]:]
        #                 possible_map_lanes_cache.append(fill_in_lane[fill_in_mask])
        #     for cache_lane in possible_map_lanes_cache:
        #         possible_map_lanes.append(cache_lane)
                
        # print(f'Loop1: {time.time()-t_1:.2f}s') 
        # max_length = max([len(arr) for arr in possible_map_lanes])
        # # Step 2: Pad each array to have the maximum length in the first dimension
        # padded_arrays = [
        #     np.pad(arr, pad_width=((0, max_length - arr.shape[0]), (0, 0)), mode='constant', constant_values=0)
        #     for arr in possible_map_lanes
        # ]
        # possible_map_lanes = np.stack(padded_arrays)
        # possible_map_lanes_mask = possible_map_lanes[...,0]!=0

        
        # current_speed = np.linalg.norm(ego[-1,3:5]) # m/s
        # starting_point = possible_map_lanes[0][possible_map_lanes_mask[0]][0,:3] # x,y,h
        # ## Filtering lanes based on maximum reachable distance based on current speed and maximum possible acceleration (extreme acceleration of 2.5 m/s)
        # # 1 m/s^2 = 3.6 km/h/s, i.e. the km/h achived in 1 second, 
        # # fast cars usually have a maximum accelearion below 100 km/h/8s, 
        # # aggressive_accel_threshold = 2.26 # m/s^2 # 2.26 m/s^2 -> 65.09 km/h/8s -> 8.14 km/h/s -> 8.14 km/h increase in 1 second -> 8.14/3.6 m/s -> 2.26 m/s
        # max_possible_speed_increase = 2.5 # m/s maximum possible increase in speed -> 72 km/h/8s
        # maximum_future_speeds = [current_speed+(i*max_possible_speed_increase) for i in range(8)] # m/s for 8 the current speed and 7 seconds in the future speeds
        # max_travel_distance = sum(maximum_future_speeds)
        # ## Filtering lanes based on minimum reachable distance based on current speed and maximum possible deceleration (extreme deceleration of -2.5 m/s)
        # max_possible_speed_decrease = max_possible_speed_increase # m/s
        # min_future_speeds = [current_speed-(i*max_possible_speed_decrease) for i in range(8) if current_speed-(i*max_possible_speed_decrease)>0] # m/s for 8 the current speed and 7 seconds in the future speeds
        # min_travel_distance = sum(min_future_speeds)
        # starting_point = possible_map_lanes[0][possible_map_lanes_mask[0]][0,:3] # x,y,h
        # # filtering:
        # map_points_distances = np.linalg.norm(starting_point[:2] - possible_map_lanes[possible_map_lanes_mask][:,:2], axis=1)
        # reachable_points_mask = (map_points_distances < max_travel_distance) * (map_points_distances > min_travel_distance)
        # possible_map_lanes_mask[possible_map_lanes_mask] = reachable_points_mask * possible_map_lanes_mask[possible_map_lanes_mask]


        # direction_classifier = DirectionClassifier(step_t=1)
        # possible_directions = []
        # t_2 = time.time()
        # for i in range(len(possible_map_lanes)):
        #     if len(possible_map_lanes[i][possible_map_lanes_mask[i]])>0:
        #         two_points_lane = np.vstack((starting_point, possible_map_lanes[i][possible_map_lanes_mask[i]][-1,:3]))
        #         direction_str, direction_cls,_,_ = direction_classifier.classify_lane(two_points_lane)
        #         possible_directions.append(direction_classifier.classes[direction_cls])
        
        # print(f'Loop2: {time.time()-t_2:.2f}s')
        # # for i in range(len(possible_map_lanes)):
        # #     possible_map_lanes[0][possible_map_lanes_mask[0]] possible_map_lanes[i][possible_map_lanes_mask[i]]
        # # get_vel_from_traj(ego[:,:2])
        # possible_directions = np.unique(possible_directions)
        # # print(possible_directions)
        # return possible_directions
    
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

            # get the sdc current state
            self.current_xyzh.append( (tracks[sdc_id].states[self.hist_len-1].center_x, tracks[sdc_id].states[self.hist_len-1].center_y, 
                                tracks[sdc_id].states[self.hist_len-1].center_z, tracks[sdc_id].states[self.hist_len-1].heading) )

            # add sdc states into the array
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

        # search for nearby agents
        for e in range(len(sdc_ids)):
            for i, track in enumerate(tracks):
                track_states = track.states[:self.hist_len]
                if i not in sdc_ids and track_states[-1].valid:
                    xy = np.stack([track_states[-1].center_x, track_states[-1].center_y], axis=-1)
                    neighbors.append((i, np.linalg.norm(xy - self.current_xyzh[e][:2]))) 

        # sort the agents by distance
        sorted_neighbors = sorted(neighbors, key=lambda item: item[1])

        # add neighbor agents into the array
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

            # only consider 'num_neihgbors' agents
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
        # Reverse normalize agent trajectories
        if ego is not None:
            ego[0, :, :5] = reverse_agent_norm(ego[0], center, angle)
            ego[1, :, :5] = reverse_agent_norm(ego[1], center, angle)
        if ground_truth is not None:
            ground_truth[0] = reverse_agent_norm(ground_truth[0], center, angle)
            ground_truth[1] = reverse_agent_norm(ground_truth[1], center, angle)

        if neighbors is not None:
            # Reverse normalize neighbors
            for i in range(neighbors.shape[0]):
                if neighbors[i, -1, 0] != 0:
                    neighbors[i, :, :5] = reverse_agent_norm(neighbors[i], center, angle)

        return ego, neighbors, ground_truth

    def interpolate_missing_data(self, ego, ground_truth, neighbors):
        # interpolate missing values
        not_valid = False
        future_start_end_with_zeros = False
        # if sum(ego[0,:,0]==0)>5 or sum(ego[1,:,0]==0)>5:
        if sum(ego[0,:,0]==0)>8 or sum(ego[1,:,0]==0)>8:
            not_valid = True 
        # elif sum(ground_truth[0,:,0]==0)>50 or sum(ground_truth[1,:,0]==0)>50:
        elif sum(ground_truth[0,:,0]==0)>70 or sum(ground_truth[1,:,0]==0)>70:
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

        # not_valid = not_valid or future_start_end_with_zeros
        return ego, ground_truth, neighbors, not_valid

    def normalize_map_points(self, lane_data, center, angle):
        for k in range(lane_data.shape[0]):
            lane_data_ = lane_data[k]
            if lane_data_[0][0] != 0:
                lane_data_[:, :3] = map_norm(lane_data_, center, angle)
        return lane_data

    def normalize_data_original(self, ego, neighbors, map_lanes, map_crosswalks, ground_truth, viz=False):
        # get the center and heading (local view)
        center, angle = ego[0].copy()[-1][:2], ego[0].copy()[-1][2]
        
        # normalize agent trajectories
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

        # normalize map points
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

        return map_lanes, map_crosswalks

    def normalize_data(self, ego, neighbors, map_lanes, map_crosswalks, ground_truth, map_speed_bumps, viz=False):
        for i in range(len(ego)):
            speeds_01 = np.linalg.norm(abs_distance_to_velocity(ego[i,:,:2][ego[i,:,0]!=0]),axis=-1)*10*3.6
            speeds_02 = np.linalg.norm(abs_distance_to_velocity(ground_truth[i,:,:2][ground_truth[i,:,0]!=0]),axis=-1)*10*3.6
            if len(speeds_01)==0 or len(speeds_02)==0:
                return [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]
            max_speed = max(max(speeds_01), max(speeds_02))
            if max_speed>130: # 130 kmh ~= 80 mph
                return [-1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]
        
        # get the center and heading (local view)
        center, angle = ego[0].copy()[-1][:2], ego[0].copy()[-1][2]
        normalization_param = [center, angle]
        
        # normalize agent trajectories
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

        # normalize map points
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


        # visulization
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
        max_num_neighboring_lanes = 8
        dist_threshold, angle_threshold = 10, 10 # 1m, 5 degrees
        # Step 1: Get ego vehicle's current state and position
        ego_state = ego[-1, :3]
        ego_state_velocity = np.linalg.norm(ego[-1, 3:5]) # in m/s
        ego_coords = ego_state[:2]
        ego_angle = ego_state[2]
        # Step 2: Get the map object (lane) at the ego vehicle's position
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
        if use_neighbouring_lanes:
            additional_lanes = {}
            for lane_id in sorted_lanes_polylines:
                if lane_id == ego_lane_id:
                    continue
                object_i_xyh = sorted_lanes_polylines[lane_id]
                diff_to_ego_lane = (ego_lane_xyh[..., np.newaxis, :] - object_i_xyh)
                object_i_dist = np.linalg.norm(diff_to_ego_lane[...,:2], axis=2)
                # Find the index of the minimum value in the flattened array
                min_index = np.argmin(object_i_dist)
                # Convert the flattened index back to row and column indices
                row_idx, col_idx = np.unravel_index(min_index, object_i_dist.shape)
                if object_i_dist[row_idx,col_idx]<=dist_threshold and np.abs(np.degrees(diff_to_ego_lane[row_idx,col_idx][-1]))<= angle_threshold:
                    additional_lanes.update({lane_id:sorted_lanes[lane_id]})
                
                if max_num_neighboring_lanes is not None and len(additional_lanes)>=max_num_neighboring_lanes:
                    break
                    # additional_lanes.append(self.lanes[lane_id])
                
        # finding max and min possible travel distances
        current_speed = ego_state_velocity # m/s
        ## Filtering lanes based on maximum reachable distance based on current speed and maximum possible acceleration (extreme acceleration of 2.5 m/s)
        # 1 m/s^2 = 3.6 km/h/s, i.e. the km/h achived in 1 second, 
        # fast cars usually have a maximum accelearion below 100 km/h/8s, 
        # aggressive_accel_threshold = 2.26 # m/s^2 # 2.26 m/s^2 -> 65.09 km/h/8s -> 8.14 km/h/s -> 8.14 km/h increase in 1 second -> 8.14/3.6 m/s -> 2.26 m/s
        # max_possible_speed_increase = 2.5 # m/s maximum possible increase in speed -> 72 km/h/8s
        # max_possible_speed_increase = 2.26 # 2.26 # m/s^2 # 2.26 m/s^2 -> 65.09 km/h/8s -> 8.14 km/h/s
        max_possible_speed_increase = 1.5625 # 12.5 # m/s -> 45 km/h/8s -> 100 m/8s
        speed_limit = ego_lane.speed_limit_mph # mph
        speed_limit = speed_limit * 0.44704 # meter-per-second
        maximum_future_speeds = [current_speed+(i*max_possible_speed_increase) for i in range(8)] # m/s for 8 the current speed and 7 seconds in the future speeds
        max_travel_distance = sum(maximum_future_speeds)
        ## Filtering lanes based on minimum reachable distance based on current speed and maximum possible deceleration (extreme deceleration of -2.5 m/s)
        max_possible_speed_decrease = max_possible_speed_increase # m/s
        min_future_speeds = [current_speed-(i*max_possible_speed_decrease) for i in range(8) if current_speed-(i*max_possible_speed_decrease)>0] # m/s for 8 the current speed and 7 seconds in the future speeds
        min_travel_distance = sum(min_future_speeds)
        max_distance = min(min(speed_limit*8, max_travel_distance), 60) # max 60 meters
        min_distance = max(0, min_travel_distance)
        if max_distance<=min_distance:
            max_distance = min_distance+20
        # print('-----------------')
        # print(f'max distance:{max_distance}')
        # print(f'min distance:{min_distance}')
        # if speed_limit*8 < max_travel_distance:
        #     print('max distance exceeds speed limit!')

        coords_connected, lanes_connected, lanes_ids_connected = connect_lanes_within_distance(sorted_lanes, {ego_lane_id:ego_lane}, max_distance)
        coords_connected_ = [np.vstack(coords_connected[i]) for i in range(len(coords_connected))]
        if use_neighbouring_lanes:
            lanes_connected_, lanes_ids_connected_ = [], []
            for additional_lane_id in additional_lanes:
                _coords_connected, _lanes_connected, _lanes_ids_connected = connect_lanes_within_distance(sorted_lanes, {additional_lane_id:sorted_lanes[additional_lane_id]}, max_distance)
                # if len(_coords_connected)>1:
                #     print('')
                for xx in [np.vstack(_coords_connected[i]) for i in range(len(_coords_connected))]:
                    coords_connected_.append(xx)
                # coords_connected_.append(_coords_connected)
                lanes_connected_.append(_lanes_connected)
                lanes_ids_connected.append(_lanes_ids_connected)

        # coords_connected_stacked = [np.vstack(coords_connected[i]) for i in range(len(coords_connected))]
        
        # # Padding Step 1: Find the maximum T (number of rows)
        # array_list = coords_connected_
        # max_T = max(array.shape[0] for array in array_list)
        # # Padding Step 2: Pad each array with zeros based on max_T
        # padded_arrays = []
        # for array in array_list:
        #     T = array.shape[0]  # Current number of rows (T)
        #     padding = ((0, max_T - T), (0, 0))  # Pad only along the first dimension (rows)
        #     padded_array = np.pad(array, padding, mode='constant', constant_values=0)
        #     padded_arrays.append(padded_array)
        
        
        # # Stack and normalize the map before post processing
        # map_lanes = np.stack(padded_arrays)
        map_lanes = pad_and_stack_arrays(coords_connected_)
        center, angle = ego.copy()[-1][:2], ego.copy()[-1][2]
        map_lanes = self.normalize_map_points(deepcopy(map_lanes), center, angle)
        ego_lane_xyh = self.normalize_map_points(deepcopy(ego_lane_xyh[np.newaxis]), center, angle)
        
        ## Post processing and zeroing out some non drivable areas
        lanes_copy = deepcopy(map_lanes)
        lanes_copy_norms = np.linalg.norm(lanes_copy[...,:2], axis=-1)
        zero_mask = lanes_copy_norms == 0
        lanes_copy[zero_mask] = np.inf
        lanes_copy_norms = np.linalg.norm(lanes_copy[...,:2], axis=-1)
        closest_points_in_lines = np.argmin(lanes_copy_norms, -1)
        closest_point_idx = np.argmin(lanes_copy_norms)
        closest_point_idx = np.unravel_index(closest_point_idx, lanes_copy_norms.shape)
        closest_point = deepcopy(map_lanes[closest_point_idx][...,:3])
        # reference_lane_point = 
        
        ### Cropping lanes based on max distance
        # ## v1: zeroing > max_distance
        # out_of_range_index = np.linalg.norm(map_lanes[:,:,:2], axis=-1)>max_distance
        # map_lanes[out_of_range_index] = 0

        ## v2: Zeroing out points where distance is greater than max_distance
        for i, lane in enumerate(map_lanes):
            distances = np.linalg.norm(lane[:,:2], axis=-1)
            exceed_indices = np.where(distances > max_distance)[0]
            
            if len(exceed_indices) > 0:
                first_exceed_idx = exceed_indices[0]  # Get the first index where distance exceeds max_distance
                map_lanes[i, first_exceed_idx:] = 0  # Zero out all points after this index
                
        ## zeroing < min_distance
        out_of_range_index = np.linalg.norm(map_lanes[:,:,:2], axis=-1)<min_distance
        map_lanes[out_of_range_index] = 0
        
        # # zeroing same lane in the back, as the vehicle is not expected to do reverse
        # Step 1: Create a mask for checking if lane indices are less than closest_points_in_lines
        # This compares each entry's index (from 0 to 299) with the corresponding closest_points_in_lines value
        indices = np.arange(map_lanes.shape[1])  # Array of indices from 0 to 299
        mask_indices = indices < closest_points_in_lines[:, np.newaxis]
        mask_angle = map_lanes[:, :, 2] < angle_threshold
        mask = mask_indices & mask_angle
        # mask = mask_indices
        map_lanes[mask] = 0
        truncated_map_lanes = truncate_lanes_vectorized(deepcopy(map_lanes))
        map_lanes = truncated_map_lanes if len(truncated_map_lanes)>=1 else map_lanes
        return map_lanes, ego_lane_xyh, max_distance, min_distance
    
    def plt_scene_raw(self, agent_1=None, agent_2=None, other_agents=None, figure_center=None, max_num_lanes=10, direction_lanes = None, map_lanes=None, drivable_lanes=None, center=None, angle=None):
        # to plot multiple lanes segments, each witrh a different color
        # each category [0:center lanes, 1:boundaries, 2:edges, 3:stop signs, 4:crosswalks, 5:speed bump, 6:traffic light]
        # is plotted seperatly on the same figure, use the correct categ value, refer to the function vizualize_background [in this file]
        lane_polylines = get_polylines(self.lanes)
        lane_types = [value.type for value in self.lanes.values()]
        fig=None
        ax=None
        num_lanes = 0
        # max_num_lanes = 1
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
            lane = lane[lane[:,0]!=0]
            lane = map_norm(lane, center, angle)
            if lane_types[i] in [0,1,2]:
                fig, ax = vizualize_background(segment=lane, segment_type=lane_types[i], fig=fig, categ=0, ax=ax)
                num_lanes+=1
                if num_lanes>max_num_lanes:
                    break
            else:
                continue

        # if False:
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
        # if True:
            if other_agents is not None:
                # alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                alphabets = [i for i in range(50)]
                # alphabets = [i for i in range(len(alphabets))]            
                for agent_idx, agent_ in enumerate(other_agents):
                    agent_type = int((agent_[-1,-3:].argmax(-1)+1)*agent_[-1,-3:].sum())
                    if agent_type>0:
                        colors = ['','dimgrey', 'sandybrown', 'dimgrey']
                        vizualize_agent_rect(fig=fig, past=agent_, object_type=agent_type, color=colors[agent_type], agent_prefix=alphabets[agent_idx+3])
            if agent_1 is not None:            
                agent_ = agent_1
                agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                # colors = ['','darkviolet', 'violet', 'darkviolet']
                colors = ['','darkviolet', 'darkviolet', 'darkviolet']
                vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='1') # 'Ego'
                vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
            if agent_2 is not None:            
                agent_ = agent_2
                agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
                # colors = ['','teal', 'peru', 'teal']
                colors = ['','blue', 'blue', 'blue']
                vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='2') # 'Interactive'
                vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=True) # single agent, single modality
            
            traffic_light_points, traffic_light_states, _ = self.get_traffic_light_locations(unique_only=True)
            for point, state in zip(traffic_light_points, traffic_light_states):
                fig,ax = vizualize_background(segment=point, segment_type=state, fig=fig, categ=6, ax=ax)

        # if False:
        # nohup python3 /home/felembaa/projects/iMotion-LLM-ICLR/gameformer/data_preprocess_v08_eval02.py > nohup/zzz.log 2>&1 & disown
        # if figure_center is not None:
        center_x = 0  # Replace with your actual center x-coordinate
        center_y = 0  # Replace with your actual center y-coordinate
        range_crop = 200  # Crop range around the center
        # Set axis limits to crop around the center
        ax.set_xlim(center_x - range_crop, center_x + range_crop)
        ax.set_ylim(center_y - range_crop, center_y + range_crop)
        # Remove X and Y Axis Labels
        # ax.set_xlabel('')
        # ax.set_ylabel('')
        # Optional: remove tick labels if desired
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])

        return fig

    def process_data(self, viz=True,test=False):
        navigation_extractor = futureNavigation()
        direction_classes = np.array(navigation_extractor.direction_classifier.classes)
        selected_scenario = True #comment
        # self.save_dir_json = f"{self.save_dir[:-1]}_json/"
        # os.makedirs(self.save_dir_json, exist_ok=True)

        if self.point_dir != '':
            self.build_points()
        # self.pbar = tqdm(total=len(list(self.data_files)))
        # self.pbar.set_description(f"Processing {len(list(self.data_files))} files, -{list(self.data_files)[0].split('/')[-1][-14:-9]}")
        for iii, data_file in enumerate(self.data_files):
            # if iii<149:
            #     continue
            dataset = tf.data.TFRecordDataset(data_file)
            self.pbar = tqdm(total=len(list(dataset)))
            self.pbar.set_description(f"Processing {data_file.split('/')[-1]}")

            for data in dataset:
                
                parsed_data = scenario_pb2.Scenario()
                parsed_data.ParseFromString(data.numpy())
                
                scenario_id = parsed_data.scenario_id
                # print(scenario_id)
                # if 'cc72b9a1ddd966b0' not in scenario_id and selected_scenario: # comment
                #     selected_scenario = False
                #     continue
                

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
                # if len(interact_list)>0:
                #     print("test")
                # for pairs in self.sdc_ids_list:
                for pairs_idx, pairs in enumerate(self.sdc_ids_list):
                    valid_instruct = True
                    sdc_ids, interesting = pairs[0], pairs[1]
                    
                    if interesting!=1 and args.small_data: # comment for original preprocessing
                        continue
                    # process data
                    ego = self.ego_process(sdc_ids, parsed_data.tracks)

                    ego_type = parsed_data.tracks[sdc_ids[0]].object_type
                    if ego_type!=1:  # comment for original preprocessing
                        valid_instruct = False
                        # continue
                    neighbor_type = parsed_data.tracks[sdc_ids[1]].object_type
                    object_type = np.array([ego_type, neighbor_type])
                    self.object_type = object_type
                    ego_index = parsed_data.tracks[sdc_ids[0]].id
                    neighbor_index = parsed_data.tracks[sdc_ids[1]].id
                    object_index = np.array([ego_index, neighbor_index])
                    
                    neighbors, _ = self.neighbors_process(sdc_ids, parsed_data.tracks)
                    max_num_points = 500
                    # # map_extra_boundaries_L, map_extra_boundaries_R = np.zeros(shape=(2, self.drivable_lanes, max_num_points, 40), dtype=np.float32), np.zeros(shape=(2, self.drivable_lanes, max_num_points, 40), dtype=np.float32)
                    # additional_boundaries_ = np.zeros(shape=(2, self.drivable_lanes, max_num_points, 4), dtype=np.float32)
                    
                    
                    map_lanes = np.zeros(shape=(self.drivable_lanes, max_num_points, 17), dtype=np.float32) # DONE: make 6 variable
                    # map_extra_boundaries_L, map_extra_boundaries_R = np.zeros(shape=(2, self.drivable_lanes, max_num_points, 40), dtype=np.float32), np.zeros(shape=(2, self.drivable_lanes, max_num_points, 40), dtype=np.float32)
                    additional_boundaries_ = np.zeros(shape=(self.drivable_lanes, max_num_points, 4), dtype=np.float32)
                    map_crosswalks = np.zeros(shape=(self.num_crosswalks, 100, 3), dtype=np.float32)
                    ground_truth = self.ground_truth_process(sdc_ids, parsed_data.tracks)
                    
                    inter = 'interest' if interesting==1 else 'r'
                    filename = self.save_dir + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}_{pairs_idx}.npz"
                    # filename = self.save_dir + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}.npz"
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
                            drivable_lanes_new, ego_lane, max_d, min_d = self.get_drivable_lanes(ego[0])
                        except Exception as e:
                            valid_instruct = False
                        # center, angle = ego[0].copy()[-1][:2], ego[0].copy()[-1][2]
                        # drivable_lanes_new = self.normalize_map_points(deepcopy(drivable_lanes_new), center, angle)
                        # drivable_lanes_new[(drivable_lanes_new<min_d) * (drivable_lanes_new>max_d)] = 0
                        # map_norm(drivable_lanes_new, center, angle)
                        # map_norm
                    if test:
                        ground_truth = np.zeros((2, self.future_len, 5))
                    else:
                        ground_truth = self.ground_truth_process(sdc_ids, parsed_data.tracks)
                    original_ego, original_neighbors, original_map_lanes, original_map_crosswalks, original_ground_truth, original_region_dict = self.normalize_data_original(deepcopy(ego), deepcopy(neighbors), original_map_lanes, original_map_crosswalks, deepcopy(ground_truth))
                    # if valid_instruct:
                    if valid_instruct:
                        try:
                            ego, ground_truth, neighbors, not_valid_future = self.interpolate_missing_data(ego, ground_truth, neighbors)
                            if not_valid_future:
                                valid_instruct = False
                        except Exception as e:
                            valid_instruct = False
                        # continue
                    if valid_instruct:
                        agent_json = self.gen_instruct_caption_01(history=np.vstack((deepcopy(ego),deepcopy(neighbors))),future=deepcopy(ground_truth), navigation_extractor=navigation_extractor, vizualize=args.viz, viz_dir= 'ex.png' if not args.not_debug else viz_dir)
                    # if not args.not_debug:
                        if'Agent-1' not in agent_json.keys():
                            valid_instruct = False
                        # if not args.not_debug:
                        if 'Agent-2' not in agent_json.keys():
                            # self.gen_instruct_caption_01(history=np.vstack((deepcopy(ego),deepcopy(neighbors))),future=deepcopy(ground_truth), navigation_extractor=navigation_extractor, vizualize=not args.not_debug)
                            valid_instruct = False
                    # print(f"caption_01: {time.time()-t_}")
                    # ego_original, neighbors_original, map_lanes_original, map_crosswalks_original, ground_truth_original = deepcopy(ego), deepcopy(neighbors), deepcopy(map_lanes), deepcopy(map_crosswalks), deepcopy(ground_truth)
                    # t_ = time.time()
                    # if valid_instruct:    
                    ego, neighbors, map_lanes, map_crosswalks, ground_truth, region_dict, normalization_param, speed_bumps = self.normalize_data(deepcopy(ego), deepcopy(neighbors), deepcopy(map_lanes[np.newaxis]), deepcopy(map_crosswalks[np.newaxis]), deepcopy(ground_truth), deepcopy(speed_bumps[np.newaxis]), viz=False)
                    map_lanes, map_crosswalks, speed_bumps = map_lanes[0], map_crosswalks[0], speed_bumps[0]
                    # print(f"normalize data: {time.time()-t_}")
                    if valid_instruct and len(ego)==1:
                        valid_instruct = False
                            # continue
                    
                    ## uncomment for additional boundaries
                    # additional_boundaries_[:,:,:3] = self.normalize_map_points(additional_boundaries_[:,:,:3], normalization_param[0], normalization_param[1])
                    # for i_point in range(len(traffic_lights)):
                    #     traffic_lights[i_point,:2] = map_norm(traffic_lights[i_point:i_point+1,:2], normalization_param[0], normalization_param[1])[0,:2]
                    # for i_point in range(len(stop_signs)):
                    #     stop_signs[i_point,:2] = map_norm(stop_signs[i_point:i_point+1,:2], normalization_param[0], normalization_param[1])[0,:2]
                    ## end of uncomment

                    # print(f"normalize boundaries (2 agents): {time.time()-t_}")

                    # t_ = time.time()
                    if valid_instruct:
                        agent_json_, map_json = self.gen_instruct_caption_02(history=np.vstack((deepcopy(ego),deepcopy(neighbors))),future=deepcopy(ground_truth), navigation_extractor=navigation_extractor, normalization_param=normalization_param)
                        if agent_json['Agent-1']['movement 0.1to8']=='INVALID':
                            valid_instruct = False
                    # print(f"caption_02: {time.time()-t_}")
                    if valid_instruct:
                        for agent_k in agent_json.keys():
                            agent_json[agent_k].update(agent_json_[agent_k])
                    # ego_normalized_back, neighbors_normalized_back, ground_truth_normalized_back = self.reverse_normalize_traj(deepcopy(ego), deepcopy(neighbors), deepcopy(ground_truth), normalization_param)
                    # to_save_json = self.gen_instruct_caption(ego[0], ground_truth[0], ego[1], ground_truth[1], neighbors, navigation_extractor)
                    # to_save_json = self.gen_instruct_caption(history=np.vstack((deepcopy(ego),deepcopy(neighbors))),future=deepcopy(ground_truth), navigation_extractor=navigation_extractor, normalization_param=normalization_param)
                    if valid_instruct and args.plausbility:

                        if drivable_lanes_new is not None:
                            plausibility, directions_lanes, _ = self.get_direction_plausibility_03(deepcopy(ego_lane), deepcopy(drivable_lanes_new), deepcopy(ego))
                        else:
                            plausibility, directions_lanes, drivable_lanes = self.get_direction_plausibility_02(deepcopy(ego), deepcopy(map_lanes), deepcopy(map_crosswalks), deepcopy(ground_truth), deepcopy(additional_boundaries_), deepcopy(traffic_lights), deepcopy(stop_signs), deepcopy(agent_json_))

                        possible_directions = plausibility
                        possible_directions_cls = [np.where(possible_direction_i == direction_classes)[0][0] for possible_direction_i in possible_directions]
                        not_possible_directions = [direction_class_i for direction_class_i in direction_classes if direction_class_i not in plausibility]
                        not_possible_directions_cls = [np.where(direction_class_i == direction_classes)[0][0] for direction_class_i in direction_classes if direction_class_i not in plausibility]

                        if 'straight' not in agent_json['Agent-1']['direction 0.1to8']:
                            possible_directions = list(set([dir_i if 'straight' not in dir_i else 'move straight' for dir_i in possible_directions]))
                        else:
                            possible_directions = list(set([dir_i for dir_i in possible_directions if 'straight' not in dir_i]))
                        possible_directions = [dir_i for dir_i in possible_directions if agent_json['Agent-1']['direction 0.1to8'] != dir_i]
                        straight_in_feasible = 'move straight' in possible_directions
                        if straight_in_feasible or 'straight' in agent_json['Agent-1']['direction 0.1to8']:
                            not_possible_directions = list(set([dir_i for dir_i in not_possible_directions if 'straight' not in dir_i]))
                        else:
                            not_possible_directions = list(set([dir_i if 'straight' not in dir_i else 'move straight' for dir_i in not_possible_directions]))
                        not_possible_directions = [dir_i for dir_i in not_possible_directions if agent_json['Agent-1']['direction 0.1to8'] != dir_i]

                        num_examples_to_validate = 20
                        if True:
                            history_viz=np.vstack((deepcopy(ego),deepcopy(neighbors)))
                            future_viz=deepcopy(ground_truth)
                            subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int)

                            fig3 = None
                            mode = 'gt'
                            cls = agent_json['Agent-1']['direction 0.1to8'] if 'straight' not in agent_json['Agent-1']['direction 0.1to8'] else 'move straight'
                            save_dir_ = os.path.join(os.path.dirname(filename), f'{mode}', f'{cls}')
                            # Check if save_dir_ exists, else create it
                            os.makedirs(save_dir_, exist_ok=True)
                            # Check if the number of existing files in save_dir_ is less than 100
                            if len(glob.glob(os.path.join(save_dir_, '*'))) < num_examples_to_validate:
                                save_dir = os.path.splitext(os.path.basename(filename))[0] + '.png'
                                save_dir = os.path.join(save_dir_, save_dir)
                                fig3 = self.plt_scene_raw(
                                    (history_viz[0], future_viz[0, subsample_indices]),
                                    # max_num_lanes=500,
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
                                # Check if save_dir_ exists, else create it
                                os.makedirs(save_dir_, exist_ok=True)
                                # Check if the number of existing files in save_dir_ is less than 100
                                if len(glob.glob(os.path.join(save_dir_, '*'))) < num_examples_to_validate:
                                    save_dir = os.path.splitext(os.path.basename(filename))[0] + '.png'
                                    save_dir = os.path.join(save_dir_, save_dir)
                                    if fig3 is None:
                                        fig3 = self.plt_scene_raw(
                                            (history_viz[0], future_viz[0, subsample_indices]),
                                            # max_num_lanes=500,
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
                                # Check if save_dir_ exists, else create it
                                os.makedirs(save_dir_, exist_ok=True)
                                # Check if the number of existing files in save_dir_ is less than 100
                                if len(glob.glob(os.path.join(save_dir_, '*'))) < num_examples_to_validate:
                                    save_dir = os.path.splitext(os.path.basename(filename))[0] + '.png'
                                    save_dir = os.path.join(save_dir_, save_dir)
                                    if fig3 is None:
                                        fig3 = self.plt_scene_raw(
                                            (history_viz[0], future_viz[0, subsample_indices]),
                                            # max_num_lanes=500,
                                            max_num_lanes=1000,
                                            map_lanes=deepcopy(map_lanes),
                                            drivable_lanes=deepcopy(drivable_lanes_new),
                                            direction_lanes=directions_lanes,
                                            center=normalization_param[0],
                                            angle=normalization_param[1]
                                        )
                                    fig3.savefig(save_dir)

                            plt.close()
                            # if agent_json['Agent-1']['direction 0.1to8_cls']==selected_cls and selected_mode == 'gt':
                                
                                
                            #     print(f"Ground truth direction: {agent_json['Agent-1']['direction 0.1to8']}")
                            #     # print(f"Other feasible directions: {possible_directions}")
                            #     # print(f"Ineafible directions: {not_possible_directions}")
                            #     # print('---')
                            #     # print('---')
                            #     mode = 'gt'
                            #     cls = agent_json['Agent-1']['direction 0.1to8_cls']
                            #     save_dir_ = '/'.join(filename.split('/')[:-1])+f'/{mode}/cls_{cls}/'
                            #     ## check if save_dir_ exist else make it
                            #     ## check if glob(save_dir_+'*') has less than 100 examples, if yes then do the following if not then do nothing
                            #     save_dir = filename.split('/')[-1].strip('.npz')+'.png'
                            #     save_dir = save_dir_+ save_dir
                            #     fig3 = self.plt_scene_raw((history_viz[0], future_viz[0, subsample_indices]), max_num_lanes=500, map_lanes = deepcopy(map_lanes), drivable_lanes=deepcopy(drivable_lanes_new), direction_lanes=directions_lanes, center=normalization_param[0], angle=normalization_param[1])
                            #     fig3.savefig(save_dir)
                                
                            #     mode = 'f'
                            #     plt.close()
                            #     print()

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
    # parser.add_argument('--save_path', type=str, help='path to save processed data', default = '/ibex/project/c2278/felembaa/dataset/waymo/gameformer/training_fullmap_small')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='/ibex/project/c2278/felembaa/datasets/waymo/validation_interactive/')
    # parser.add_argument('--save_path', type=str, help='path to save processed data', default = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/validation_16may_fullmap_wPlausbility')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='/ibex/project/c2278/felembaa/datasets/waymo/training/')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='/ibex/project/c2278/felembaa/datasets/waymo/validation_interactive')
    # parser.add_argument('--save_path', type=str, help='path to save processed data', default = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/temp')
    parser.add_argument('--load_path', type=str, help='path to dataset files', default='/ibex/project/c2278/felembaa/datasets/waymo/training')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='/ibex/project/c2278/felembaa/datasets/waymo/validation_interactive')
    parser.add_argument('--save_path', type=str, help='path to save processed data', default = '/ibex/project/c2278/verification_data/v08_04')
    # parser.add_argument('--save_path', type=str, help='path to save processed data', default = '')
    parser.add_argument('--point_path', type=str, help='path to load K-Means Anchors (Currently not included in the pipeline)', default='')
    parser.add_argument('--processes', type=int, help='multiprocessing process num', default=8)
    parser.add_argument('--not_debug', action="store_true", help='visualize processed data', default=False)
    # parser.add_argument('--not_debug', action="store_true", help='visualize processed data', default=True)
    parser.add_argument('--test', action="store_true", help='whether to process testing set', default=False)
    parser.add_argument('--use_multiprocessing', action="store_true", help='use multiprocessing', default=True)
    parser.add_argument('--run_num', type=int, help='', default=-1)
    parser.add_argument('--small_data', help='', action="store_true", default=True)
    parser.add_argument('--delete_old', help='', action="store_true", default=False)
    # parser.add_argument('--plausbility', help='', action="store_true", default=True)
    parser.add_argument('--plausbility', default=True)
    parser.add_argument('--viz', help='', action="store_true", default=False)
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')
    # args.not_debug = True
    args.not_debug = False
    # data_files[0][-14:]
    # data_files[0][:-14]
    # data_files[0][-9:]
    # Split the list into 10 parts
    
    # if args.run_num==-1:
    #     if 'validation' in args.load_path:
    #         if args.run_num==1:
    #             split_parts = [str(i).zfill(5) for i in range(100)]
    #             data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts]
    #             args.save_path = args.save_path + '_test'
    #         else:
    #             split_parts = [str(i).zfill(5) for i in range(100, 150)]
    #             data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts]
    #             args.save_path = args.save_path + '_val'


    if args.run_num!=-1:
        if 'validation' in args.load_path:
            numbers = [str(i).zfill(5) for i in range(150)]
            split_parts = [numbers[i:i + args.processes] for i in range(0, len(numbers), args.processes)]
            data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts[args.run_num]]
            # if args.run_num==1:
            #     split_parts = [str(i).zfill(5) for i in range(100)]
            #     data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts]
            #     args.save_path = args.save_path + '_test'
            # else:
            #     split_parts = [str(i).zfill(5) for i in range(100, 150)]
            #     data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts]
            #     args.save_path = args.save_path + '_val'
        else:
            numbers = [str(i).zfill(5) for i in range(1000)]
            split_parts = [numbers[i:i + args.processes] for i in range(0, len(numbers), args.processes)]
            data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts[args.run_num]]
            # if args.processes==64 or args.processes==32:
            #     numbers = [str(i).zfill(5) for i in range(1000)]
            #     split_parts = [numbers[i:i + 100] for i in range(0, len(numbers), 100)]
            #     data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts[args.run_num]]
            # elif args.processes==40:
            #     numbers = [str(i).zfill(5) for i in range(1000)]
            #     split_parts = [numbers[i:i + 40] for i in range(0, len(numbers), 40)]
            #     data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts[args.run_num]]
            # else:
            #     numbers = [str(i).zfill(5) for i in range(1000)]
            #     split_parts = [numbers[i:i + 16] for i in range(0, len(numbers), 16)]
            #     data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts[args.run_num]]
    

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

        # ff = f"{save_path}_agentJsons"
        # if args.delete_old:
        #     if os.path.exists(ff):
        #         shutil.rmtree(ff)
        # os.makedirs(ff, exist_ok=True)

        # ff = f"{save_path}_mapJsons"
        # if args.delete_old:
        #     if os.path.exists(ff):
        #         shutil.rmtree(ff)
        # os.makedirs(ff, exist_ok=True)
        
        # ff = f"{save_path}_fig"
        # if args.delete_old:
        #     if os.path.exists(ff):
        #         shutil.rmtree(ff)
        # os.makedirs(ff, exist_ok=True)

        # ff = f"{save_path}_templateLLM"
        # if args.delete_old:
        #     if os.path.exists(ff):
        #         shutil.rmtree(ff)
        # os.makedirs(ff, exist_ok=True)

        # ff = f"{save_path}_acts"
        # if args.delete_old:
        #     if os.path.exists(ff):
        #         shutil.rmtree(ff)
        # os.makedirs(ff, exist_ok=True)

        
        
        
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
  