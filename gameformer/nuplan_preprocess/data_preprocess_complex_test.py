import os
import argparse
from tqdm import tqdm
from common_utils import *
from data_utils import *
import matplotlib.pyplot as plt
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

import torch
import glob
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("/home/felembaa/projects/iMotion-LLM-ICLR")
sys.path.append("/home/felembaa/projects/iMotion-LLM-ICLR/gameformer")
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
import warnings
from scipy.signal import savgol_filter
from chatgpt_instruct_v02 import *

warnings.simplefilter("once")
# Define a custom warning class
class CustomWarning(Warning):
    pass

def get_negative_gpt_prompt(scenario_type_string, some_string, mode):
    # scenario_type_string = "####"
    # some_string = "####"
    if mode == 'as_01':
        xx = f"""For this vehicle scenario of {scenario_type_string}.
- in blue the vehicle with red dots indicating that the vehicle {some_string}.
- in pink are pedestrians, 
- in white are crosswalks, 
- in darker gray are center lanes in the road,
- in green are other vehicles. 

Please provide a json in the format: 
{{ "Instruction": ..., 
"Caption": "I can/cannot do this because ...", 
"Feasibility": "accepted/rejected", 
"scenario type": {scenario_type_string}, 
"instruction direction": "straight/right/stationart…", 
"instruction speed": "slow/medium/fast if instructed",
"Instruction length": short/detailed,
"Caption length": short/detailed}}

Guidelines: 
- You are a professional driving teacher testing how the driver will drive and respond to a given scenario.
- Provide an instruction on what the vehicle should do 
- Provide a reference caption of how the vehicle should respond transcribing how it will drive in detail and reasoning on contextual and safety aspect. 
- Use the figure to make the instruction and reasoning. 
- Avoid hypothetical reasoning like "it is important to stay vigilant and monitor for other pedestrians who might approach the crosswalk or other vehicles that could enter the turn path unexpectedly”
- Be direct and use the factual information of positions of objects in the figure and the road, and and do not hallucinate about other possibilities not clear in the figure. 
- The reasoning should be based on the vehicle location, surrounding vehicles, pedestrians, and cycles, crosswalks, and the lanes the vehicle can drive on. 
- Note that we know that the vehicle safely {some_string}. You can use this information or some of it to generate a safe instruction and in the caption reason why it can do that.
- Avoid using quickly or immediately
- The colors are illustrative, do not refer to objects by their colors.
- The instruction and caption should be natural and human like on how a professional driving teacher instruct a human and how the human driving respond.
- Provide a list of 6 jsons (labeled as accepted): 
1. two with short instruction (less than 10 words) and short caption (less than 30 words),
2. two with short instruction (less than 10 words) and detailed caption (less than 50 words), 
3. two with detailed instruction (less than 20 words) and detailed caption (less than 50 words). 

Respond with JSON only."""
    elif mode == 'neg_01':
        xx = f"""For this vehicle scenario of {scenario_type_string}. 
- in blue the vehicle with red dots indicating that the vehicle {some_string}.
- in pink are pedestrians, 
- in white are crosswalks, 
- in darker gray are center lanes in the road,
- in green are other vehicles. 

Please provide a json in the format: 
{{ "Instruction": ..., 
"Caption": "I can/cannot do this because ...", 
"Feasibility": "accepted/rejected", 
"scenario type": {scenario_type_string}, 
"instruction direction": "straight/right/stationart…", 
"instruction speed": "slow/medium/fast if instructed",
"Instruction length": short/detailed,
"Caption length": short/detailed}}

Guidelines: 
- You are a professional driving teacher testing how the driver will drive and respond to a given scenario.
- Provide an instruction on what the vehicle should do 
- Provide a reference caption of how the vehicle should respond transcribing why it can’t follow an instruction with reasoning on contextual and safety aspect. 
- Use the figure to make the instruction and reasoning. 
- The caption and instruction should be based on something infeasible for the given context. And reason why it can do that.
- Avoid hypothetical reasoning like "it is important to stay vigilant and monitor for other pedestrians who might approach the crosswalk or other vehicles that could enter the turn path unexpectedly”
- Be direct and use the factual information of positions of objects in the figure and the road, and and do not hallucinate about other possibilities not clear in the figure. 
- The reasoning should be based on the vehicle location, surrounding vehicles, pedestrians, and cycles, crosswalks, and the lanes the vehicle can drive on. 
- Note that we know that the vehicle safely {some_string}. So use something that contrast with what it can do.
- Avoid using words like quickly or immediately
- The colors are illustrative, do not refer to objects by their colors.
- The instruction and caption should be natural and human like on how a professional driving teacher instruct a human and how the human driving respond.
- Provide a list of 6 jsons (labeled as rejected): 
1. two with short instruction (less than 10 words) and short caption (less than 30 words),
2. two with short instruction (less than 10 words) and detailed caption (less than 50 words), 
3. two with detailed instruction (less than 20 words) and detailed caption (less than 50 words). 

Respond with JSON only."""
    
    return xx
    
def split_list(lst, num_splits):
    avg_length = len(lst) // num_splits  # Get the average length of each sublist
    remainder = len(lst) % num_splits  # Find out how many extra items there are
    result = []
    start = 0

    for i in range(num_splits):
        end = start + avg_length + (1 if i < remainder else 0)  # Add 1 to account for the remainder
        result.append(lst[start:end])
        start = end

    return result

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

def correct_relative_heading_angles_general(target_line):
    """
    Corrects the heading angles in the target_line array based on the relative angles between points,
    where each point is represented by (x, y, heading). The angle is calculated between each point and the previous one.
    
    Args:
    target_line: np.ndarray or torch.Tensor of shape (num_points, 3), where the first two columns are x and y coordinates,
                 and the third column is the heading angle (in radians).
    
    Returns:
    corrected_target_line: np.ndarray or torch.Tensor with corrected heading angles in the third column.
    """
    # If target_line is a tensor, convert it to numpy for ease of manipulation
    if isinstance(target_line, torch.Tensor):
        target_line = target_line.numpy()
    
    # Extract x and y coordinates
    x_coords = target_line[:, 0]
    y_coords = target_line[:, 1]
    
    # Initialize the corrected heading angles array
    corrected_headings = np.zeros_like(target_line[:, 2])
    
    # Loop through the target line to compute the relative heading based on current and previous point
    for i in range(1, len(target_line)):
        # Calculate the difference in x and y coordinates between the current point and the previous one
        dx = x_coords[i] - x_coords[i - 1]
        dy = y_coords[i] - y_coords[i - 1]
        
        # Compute the relative heading angle using arctan2 (which gives angles in radians)
        relative_angle = np.arctan2(dy, dx)
        
        # Store the calculated angle
        corrected_headings[i] = relative_angle
    
    # The heading for the first point can be set to 0 (or some other value, depending on your use case)
    corrected_headings[0] = corrected_headings[1]  # or set it to 0 or some default
    
    # Return the corrected target line with updated heading angles
    corrected_target_line = target_line.copy()
    corrected_target_line[:, 2] = corrected_headings
    
    return corrected_target_line

def connect_lanes_within_distance(ego_lane, max_distance, max_depth=5, verbose=False) -> Tuple[List[List[Point2D]], List[List[Point2D]], List[List[Point2D]], List[str]]:
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
    all_mid_paths = []  # To store midline coordinates
    all_left_paths = []  # To store left boundary coordinates
    all_right_paths = []  # To store right boundary coordinates
    all_lane_ids = []  # To store lane IDs
    visited_lanes = set()  # Keep track of visited lanes to avoid cycles

    def traverse_lane(lane, original_lane, depth, current_discrete_path, current_mid_path, current_left_path, current_right_path, current_lane_ids):
        # Stop if the depth exceeds max_depth
        if depth > max_depth:
            if verbose:
                print(f"Max depth reached at lane {lane.id}")
            if len(current_lane_ids)>1:
                current_discrete_path.pop()
                current_mid_path.pop()
                current_left_path.pop()
                current_right_path.pop()
                current_lane_ids.pop()
                all_discrete_paths.append(list(current_discrete_path))
                all_mid_paths.append(list(current_mid_path))
                all_left_paths.append(list(current_left_path))
                all_right_paths.append(list(current_right_path))
                all_lane_ids.append(list(current_lane_ids))
            return

        # Detect cycle and avoid revisiting lanes
        if lane.id in visited_lanes:
            if verbose:
                print(f"Cycle detected at lane {lane.id}, stopping traversal")
            return
        visited_lanes.add(lane.id)

        # Add the current lane's discrete_path to the current path
        if lane != original_lane or depth > 0:
            current_discrete_path.append(lane.baseline_path.discrete_path)

            # Add lane's mid, left, and right boundary to the respective lists
            mid = [Point2D(node.x, node.y) for node in lane.baseline_path.discrete_path]
            left = [Point2D(node.x, node.y) for node in lane.left_boundary.discrete_path]
            right = [Point2D(node.x, node.y) for node in lane.right_boundary.discrete_path]

            # Append to current lists for mid, left, and right boundaries
            current_mid_path.append(mid)
            current_left_path.append(left)
            current_right_path.append(right)

            # Add the lane ID
            current_lane_ids.append(lane.id)

        # Get the baseline path of the current lane and the original ego lane
        original_linestring = original_lane.baseline_path.linestring
        lane_linestring = lane.baseline_path.linestring

        # Calculate the distance between the current lane and the original ego lane
        distance = lane_linestring.distance(original_linestring)
        # print(f"Traversing lane {lane.id}, distance from ego_lane: {distance}")

        # Stop the traversal if the distance is greater than the allowed max_distance without storing last lane
        if distance > max_distance:
            if verbose:
                print(f"Stopping traversal for lane {lane.id} due to distance: {distance}")
                print("******************")
                gpt_positive_text = f"""
                
                """
                # gpt_negative_text = 
            current_discrete_path.pop()
            current_mid_path.pop()
            current_left_path.pop()
            current_right_path.pop()
            current_lane_ids.pop()
            all_discrete_paths.append(list(current_discrete_path))
            all_mid_paths.append(list(current_mid_path))
            all_left_paths.append(list(current_left_path))
            all_right_paths.append(list(current_right_path))
            all_lane_ids.append(list(current_lane_ids))
            
            return

        # If there are no more outgoing edges, consider this a leaf and store the path
        if not lane.outgoing_edges:
            if verbose:
                print(f"Reached a leaf lane {lane.id} with no outgoing edges")
                print("###################")
            all_discrete_paths.append(list(current_discrete_path))
            all_mid_paths.append(list(current_mid_path))
            all_left_paths.append(list(current_left_path))
            all_right_paths.append(list(current_right_path))
            all_lane_ids.append(list(current_lane_ids))
            return

        # Recursively traverse the outgoing edges of the current lane
        for i, outgoing_lane_connector in enumerate(lane.outgoing_edges):
            next_lane = outgoing_lane_connector  # Assuming outgoing_edges is a list of lanes
            if verbose:
                print(f"Processing outgoing lane {i} from lane {lane.id}, next lane ID: {next_lane.id}")
            traverse_lane(next_lane, original_lane, depth + 1, current_discrete_path, current_mid_path, current_left_path, current_right_path, current_lane_ids)

        # Backtrack: remove the current lane from the path after processing all its children
        if lane != original_lane or depth > 0:
            if len(current_lane_ids)>0:
                current_discrete_path.pop()
                current_mid_path.pop()
                current_left_path.pop()
                current_right_path.pop()
                current_lane_ids.pop()

    # Start the traversal from the ego_lane
    traverse_lane(ego_lane, ego_lane, 0, current_discrete_path=[ego_lane.baseline_path.discrete_path],
                  current_mid_path=[[Point2D(node.x, node.y) for node in ego_lane.baseline_path.discrete_path]],
                  current_left_path=[[Point2D(node.x, node.y) for node in ego_lane.left_boundary.discrete_path]],
                  current_right_path=[[Point2D(node.x, node.y) for node in ego_lane.right_boundary.discrete_path]],
                  current_lane_ids=[ego_lane.id])

    
    # Filtering repeated entries based on ID
    # Use a dictionary to keep track of unique lane IDs and their corresponding paths
    unique_data = {
        tuple(lane_id): (list(discrete), list(mid), list(left), list(right))
        for lane_id, discrete, mid, left, right in zip(all_lane_ids, all_discrete_paths, all_mid_paths, all_left_paths, all_right_paths)
    }

    # Unzip the dictionary back into separate lists, converting keys (lane IDs) back to lists
    all_lane_ids, paths = zip(*unique_data.items())
    all_lane_ids = [list(lane_id) for lane_id in all_lane_ids]  # Convert tuples back to lists
    all_discrete_paths, all_mid_paths, all_left_paths, all_right_paths = zip(*paths)

    # Flattening the lists after appending
    flattened_discrete_paths = [item for sublist in all_discrete_paths for item in sublist]
    flattened_mid_paths = [item for sublist in all_mid_paths for item in sublist]
    flattened_left_paths = [item for sublist in all_left_paths for item in sublist]
    flattened_right_paths = [item for sublist in all_right_paths for item in sublist]
    flattened_lane_ids = [item for sublist in all_lane_ids for item in sublist]
    
    # Structured lanes with each entry as connected lanes segments (list of lists of lanes segments)
    lane_ids_connected = [LaneSegmentLaneIDs(id_) for id_ in all_lane_ids]
    coords_conntected = [{} for _ in range(len(lane_ids_connected))]
    for i in range(len(coords_conntected)):
        coords_conntected[i][VectorFeatureLayer.LANE.name] = MapObjectPolylines(MapObjectPolylines(all_mid_paths[i]).polylines)
        coords_conntected[i][VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(MapObjectPolylines(all_left_paths[i]).polylines)
        coords_conntected[i][VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(MapObjectPolylines(all_right_paths[i]).polylines)


    # Unrolled segments, with each segment defined seperatly with its own id (list of lanes segments)
    coords: Dict[str, MapObjectPolylines] = {}
    coords[VectorFeatureLayer.LANE.name] = MapObjectPolylines(MapObjectPolylines(flattened_mid_paths).polylines)
    coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(MapObjectPolylines(flattened_left_paths).polylines)
    coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(MapObjectPolylines(flattened_right_paths).polylines)
    lane_ids = LaneSegmentLaneIDs(flattened_lane_ids)
    
    # ## Stucturing lanes such that each lane is a one long continous lane with no segments:
    # for mids_, lefts_, rights_, ids_ in zip(all_mid_paths, all_left_paths, all_right_paths, all_lane_ids):
    #     mid_ = [item for sublist in mids_ for item in sublist]
    #     left_ = [item for sublist in lefts_ for item in sublist]
    #     right_ = [item for sublist in rights_ for item in sublist]
    #     break
    if verbose:
        print(lane_ids_connected)
    return coords, lane_ids, coords_conntected, lane_ids_connected#flattened_discrete_paths, flattened_mid_paths, flattened_left_paths, flattened_right_paths

# define data processor
class DataProcessor(object):
    def __init__(self, scenarios):
        self._scenarios = scenarios

        # self.past_time_horizon = 2 # [seconds]
        self.past_time_horizon = 1 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        # self.num_agents = 20
        self.num_agents = 33
        self.num_lanes = 100
        
        self.num_crosswalks = 4
        self.num_points = 100
        self.num_points_crosswalks = 100

        # for visualization use this:
        self.num_lanes = 1000
        self.num_points = 500

        self.direction_classes = DirectionClassifier(step_t=1).classes
        
        # 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE'
        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        self._max_elements = {'LANE': self.num_lanes, 'ROUTE_LANES': self.num_lanes, 'CROSSWALK': self.num_crosswalks, 'LEFT_BOUNDARY': self.num_lanes,  'RIGHT_BOUNDARY': self.num_lanes} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': self.num_points, 'ROUTE_LANES': self.num_points, 'CROSSWALK': self.num_points_crosswalks, 'LEFT_BOUNDARY': self.num_points,  'RIGHT_BOUNDARY': self.num_points} # maximum number of points per feature to extract per featue
        # self._max_elements = {'LANE': 100, 'ROUTE_LANES': 100, 'CROSSWALK': 4, 'LEFT_BOUNDARY': 100,  'RIGHT_BOUNDARY': 100} # maximum number of elements to extract per feature layer.
        # self._max_points = {'LANE': 300, 'ROUTE_LANES': 300, 'CROSSWALK': 100, 'LEFT_BOUNDARY': 300,  'RIGHT_BOUNDARY': 300} # maximum number of points per feature to extract per featue
        # Other features: STOP_LINE
        # self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        # self._max_elements = {'LANE': 6, 'ROUTE_LANES': 6, 'CROSSWALK': 4} # maximum number of elements to extract per feature layer.
        # self._max_points = {'LANE': 300, 'ROUTE_LANES': 300, 'CROSSWALK': 100} # maximum number of points per feature to extract per featu
        
        # self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        # self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        # self._max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
        self._radius = 60 # [m] query radius scope relative to the current pose.
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.
    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states) # x,y,h,vx,vy,ax,ay

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    
    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
              sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_ego_drivable_lanes(self, ego_past=None, ego_future=None,use_neighbouring_lanes=True):
        """
        This function extracts all the drivable lanes from the map relative to the ego vehicle's position
        and classifies the possible driving directions (straight, left, right). 
        This function takes as an initial reference the ego vehicle current lane, then find all outgoing lanes from that lane (both lane connectors and lanes)
        use_neighbouring_lanes: Wither to use neighbouring lane or only ego center lane
        """
        coords: Dict[str, MapObjectPolylines] = {}
        traffic_light_data: Dict[str, LaneSegmentTrafficLightData] = {}

        dist_threshold, angle_threshold = 10, 10 # 1m, 5 degrees
        # Step 1: Get ego vehicle's current state and position
        ego_state = self.scenario.initial_ego_state  # Current ego state
        ego_state_velocity = np.linalg.norm([ego_state.agent.velocity.x, ego_state.agent.velocity.y]) # in m/s
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)  # Ego position in map coordinate
        
        # Step 2: Get the map object (lane) at the ego vehicle's position
        # Use the proper API to get the lane based on position
        # semantic_layers = [
        #     SemanticMapLayer.LANE,
        #     SemanticMapLayer.INTERSECTION,
        #     SemanticMapLayer.STOP_LINE,
        #     # SemanticMapLayer.TURN_STOP,
        #     SemanticMapLayer.CROSSWALK,
        #     # SemanticMapLayer.DRIVABLE_AREA,
        #     # SemanticMapLayer.YIELD,
        #     # SemanticMapLayer.TRAFFIC_LIGHT,
        #     # SemanticMapLayer.STOP_SIGN,
        #     # SemanticMapLayer.EXTENDED_PUDO,
        #     # SemanticMapLayer.SPEED_BUMP,
        #     SemanticMapLayer.LANE_CONNECTOR,
        #     SemanticMapLayer.BASELINE_PATHS,
        #     SemanticMapLayer.BOUNDARIES,
        #     SemanticMapLayer.WALKWAYS,
        #     SemanticMapLayer.CARPARK_AREA,
        #     # SemanticMapLayer.PUDO,
        #     SemanticMapLayer.ROADBLOCK,
        #     SemanticMapLayer.ROADBLOCK_CONNECTOR
        # ]
        layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        point = ego_coords
        radius = 10 # 4
        layers = self.map_api.get_proximal_map_objects(point, radius, layer_names)
        map_objects: List[MapObject] = []
        for layer_name in layer_names:
            map_objects += layers[layer_name]
        # sort by distance to query point
        map_objects.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))
        ego_lane = map_objects[0]
        ego_lane_xyh = np.array([[discrete_path_ego.x, discrete_path_ego.y, discrete_path_ego.heading] for discrete_path_ego in ego_lane.baseline_path.discrete_path])
        
        if use_neighbouring_lanes:
            additional_lanes = []
            for object_i in map_objects:
                if object_i.id == ego_lane.id:
                    continue
                # object_i.baseline_path.linestring
                # object_i.baseline_path.linestring.distance(ego_lane.baseline_path.linestring)
                object_i_xyh = np.array([[discrete_path_ego.x, discrete_path_ego.y, discrete_path_ego.heading] for discrete_path_ego in object_i.baseline_path.discrete_path])
                diff_to_ego_lane = (ego_lane_xyh[..., np.newaxis, :] - object_i_xyh)
                object_i_dist = np.linalg.norm(diff_to_ego_lane[...,:2], axis=2)
                # Find the index of the minimum value in the flattened array
                min_index = np.argmin(object_i_dist)
                # Convert the flattened index back to row and column indices
                row_idx, col_idx = np.unravel_index(min_index, object_i_dist.shape)
                
                if object_i_dist[row_idx,col_idx]<=dist_threshold and np.abs(np.degrees(diff_to_ego_lane[row_idx,col_idx][-1]))<= angle_threshold:
                    additional_lanes.append(object_i)
        

        ## finding max and min possible travel distances
        current_speed = ego_state_velocity # m/s
        ## Filtering lanes based on maximum reachable distance based on current speed and maximum possible acceleration (extreme acceleration of 2.5 m/s)
        # 1 m/s^2 = 3.6 km/h/s, i.e. the km/h achived in 1 second, 
        # fast cars usually have a maximum accelearion below 100 km/h/8s, 
        # aggressive_accel_threshold = 2.26 # m/s^2 # 2.26 m/s^2 -> 65.09 km/h/8s -> 8.14 km/h/s -> 8.14 km/h increase in 1 second -> 8.14/3.6 m/s -> 2.26 m/s
        # max_possible_speed_increase = 2.5 # m/s maximum possible increase in speed -> 72 km/h/8s
        # max_possible_speed_increase = 2.26 # 2.26 # m/s^2 # 2.26 m/s^2 -> 65.09 km/h/8s -> 8.14 km/h/s
        max_possible_speed_increase = 1.5625 # 12.5 # m/s -> 45 km/h/8s -> 100 m/8s
        speed_limit = ego_lane.speed_limit_mps
        maximum_future_speeds = [current_speed+(i*max_possible_speed_increase) for i in range(8)] # m/s for 8 the current speed and 7 seconds in the future speeds
        max_travel_distance = sum(maximum_future_speeds)
        ## Filtering lanes based on minimum reachable distance based on current speed and maximum possible deceleration (extreme deceleration of -2.5 m/s)
        max_possible_speed_decrease = max_possible_speed_increase # m/s
        min_future_speeds = [current_speed-(i*max_possible_speed_decrease) for i in range(8) if current_speed-(i*max_possible_speed_decrease)>0] # m/s for 8 the current speed and 7 seconds in the future speeds
        min_travel_distance = sum(min_future_speeds)
        
        # starting_point = ego_coords# (x,y only in Waymo h is also considered)
        
        if speed_limit is not None:
            max_distance = min(min(speed_limit*8, max_travel_distance), 60) # max 60 meters
        else:
            max_distance = min(max_travel_distance, 60) # max 60 meters
        min_distance = max(0, min_travel_distance)
        if max_distance<=min_distance:
            max_distance = min_distance+20
        

        coords, lane_ids, coords_conntected, lane_ids_connected = connect_lanes_within_distance(ego_lane, max_distance)
        if use_neighbouring_lanes:
            _coords, _lane_ids, _coords_conntected, _lane_ids_connected = [], [], [], []
            for additional_lane in additional_lanes:
                _connected_data = connect_lanes_within_distance(additional_lane, max_distance)
                _coords.append(_connected_data[0])
                _lane_ids.append(_connected_data[1])
                _coords_conntected.append(_connected_data[2])
                _lane_ids_connected.append(_connected_data[3])
        
        # lane traffic light data
        traffic_light_data_ = self.scenario.get_traffic_light_status_at_iteration(0)
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_data_
        )
        map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY'] # instead of self._map_features, without route_lanes or crowss_walks
        map_features_keys = ['lanes', 'left_boundary', 'right_boundary']
        
        max_elements_connected = {k: (1 if k in ['LANE', 'ROUTE_LANES', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY'] else self._max_elements[k])
            for k in self._max_elements.keys()}
        traffic_light_data_connected = [{VectorFeatureLayer.LANE.name: get_traffic_light_encoding(lane_ids_connected[i], traffic_light_data_)} for i in range(len(lane_ids_connected))]
        vector_map_conntected_ =  [map_process(ego_state.rear_axle, coords_conntected[i], traffic_light_data_connected[i], map_features,
                                 max_elements_connected, self._max_points, self._interpolation_method, join_lanes=True) for i in range(len(coords_conntected))]
        
        # Create a new dictionary where each key contains concatenated arrays
        vector_map_conntected = {key: np.concatenate([vector_map_conntected_[i][key] for i in range(len(vector_map_conntected_))], axis=0) for key in map_features_keys}

        # _vector_map_conntected is a list of additinoal maps for other found lanes that can be assigned to the ego vehicle
        _vector_map_conntected = []
        if use_neighbouring_lanes:
            for lane_ids, lane_ids_connected, coords_conntected in zip(_lane_ids, _lane_ids_connected, _coords_conntected):
                # lane traffic light data
                traffic_light_data_ = self.scenario.get_traffic_light_status_at_iteration(0)
                traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
                    lane_ids, traffic_light_data_
                )
                map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY'] # instead of self._map_features, without route_lanes or crowss_walks
                map_features_keys = ['lanes', 'left_boundary', 'right_boundary']        
                max_elements_connected = {k: (1 if k in ['LANE', 'ROUTE_LANES', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY'] else self._max_elements[k])
                    for k in self._max_elements.keys()}
                traffic_light_data_connected = [{VectorFeatureLayer.LANE.name: get_traffic_light_encoding(lane_ids_connected[i], traffic_light_data_)} for i in range(len(lane_ids_connected))]
                vector_map_conntected_ =  [map_process(ego_state.rear_axle, coords_conntected[i], traffic_light_data_connected[i], map_features,
                                        max_elements_connected, self._max_points, self._interpolation_method, join_lanes=True) for i in range(len(coords_conntected))]
                # Create a new dictionary where each key contains concatenated arrays
                _vector_map_conntected.append({key: np.concatenate([vector_map_conntected_[i][key] for i in range(len(vector_map_conntected_))], axis=0) for key in map_features_keys})
            
            warnings.warn("vector_map_conntected could include repeated map information due to concatenation of additional lines without filtering repeated ones based on lanes ids", CustomWarning)
            updated_vector_map_ = [vector_map_conntected] + _vector_map_conntected
            updated_vector_map = {key: np.concatenate([updated_vector_map_[i][key] for i in range(len(updated_vector_map_))], axis=0) for key in map_features_keys}
            # updated to include additional neighboring lanes valid to be ego assigned, with their traversed outgoing lanes. Note this could cause having repeated lanes
        else:
            updated_vector_map = vector_map_conntected
        

        lanes_copy = deepcopy(updated_vector_map['lanes'][..., :])
        lanes_copy_norms = np.linalg.norm(lanes_copy[...,:2], axis=-1)
        zero_mask = lanes_copy_norms == 0
        lanes_copy[zero_mask] = np.inf
        lanes_copy_norms = np.linalg.norm(lanes_copy[...,:2], axis=-1)
        closest_points_in_lines = np.argmin(lanes_copy_norms, -1)
        closest_point_idx = np.argmin(lanes_copy_norms)
        closest_point_idx = np.unravel_index(closest_point_idx, lanes_copy_norms.shape)
        closest_point = deepcopy(updated_vector_map['lanes'][closest_point_idx][...,:3])
        # reference_lane_point = 
        ### Cropping lanes based on max distance
        ## zeroing > max_distance
        out_of_range_index = np.linalg.norm(updated_vector_map['lanes'][:,:,:2], axis=-1)>max_distance
        updated_vector_map['lanes'][out_of_range_index] = 0
        ## zeroing < min_distance
        out_of_range_index = np.linalg.norm(updated_vector_map['lanes'][:,:,:2], axis=-1)<min_distance
        updated_vector_map['lanes'][out_of_range_index] = 0
        # # zeroing same lane in the back, as the vehicle is not expected to do reverse
        # Step 1: Create a mask for checking if lane indices are less than closest_points_in_lines
        # This compares each entry's index (from 0 to 299) with the corresponding closest_points_in_lines value
        indices = np.arange(updated_vector_map['lanes'].shape[1])  # Array of indices from 0 to 299
        mask_indices = indices < closest_points_in_lines[:, np.newaxis]
        mask_angle = updated_vector_map['lanes'][:, :, 2] < angle_threshold
        mask = mask_indices & mask_angle
        updated_vector_map['lanes'][mask] = 0

        possible_directions, directions_lanes = self.get_directions_nuplan(ego_past, updated_vector_map['lanes'], None, starting_point = closest_point)

        return updated_vector_map, possible_directions, directions_lanes
    
        # print('')
        # outgoing_edges = []
        # # ego_lane.baseline_path.linestring.distance(ego_lane.outgoing_edges[0].baseline_path.linestring) == 0 # the distance between the outgoing_edges and current line are always zero
        # # [outgoing_edges]
        # # ego_lane.baseline_path.linestring
        # if ego_lane is None:
        #     print("No lane found at the ego vehicle's position")
        #     return [], []  # Return empty if no lane is found
        
        # # Step 3: Retrieve outgoing lanes via lane connectors
        # outgoing_lane_connectors = ego_lane.outgoing_edges  # Fetch outgoing lane connectors
        
        # # Step 4: Retrieve adjacent lanes (left and right)
        # left_lane, right_lane = ego_lane.adjacent_edges  # Fetch adjacent lanes
        
        # # Step 5: Prepare to classify the direction of each connected lane
        # direction_classifier = DirectionClassifier(step_t=1)  # Initialize the direction classifier
        # possible_directions = []  # Store classified directions
        # classified_lanes = []  # Store lanes with their respective classified directions
        
        # for connector in outgoing_lane_connectors:
        #     connected_lane = self.map_api.get_map_object(connector.get_roadblock_id(), SemanticMapLayer.LANE)
        #     if connected_lane:
        #         connected_lane_heading = connected_lane.baseline_path.get_heading()
        #         print('')
        #     else:
        #         print('')
        # print('')

    def get_map(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, self._map_features, ego_coords, self._radius, route_roadblock_ids, traffic_light_data
        )

        vector_map = map_process(ego_state.rear_axle, coords, traffic_light_data, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        return vector_map
    
    def get_map_agent2(self, agent2_state=None):        
        agent_coords = Point2D(agent2_state[0], agent2_state[1])
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api, self._map_features, agent_coords, self._radius, route_roadblock_ids, traffic_light_data
        )

        vector_map = map_process(agent2_state, coords, traffic_light_data, self._map_features, 
                                 self._max_elements, self._max_points, self._interpolation_method)

        return vector_map

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
        agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, self.num_agents, agent_index)

        return agent_futures
    
    def plot_scenario(self, data, special_lanes=None, img_name='ex.png', draw_traj=True):
        num_lanes = -1
        # [1,2,4,10, 20, 30]
        # assert num_lanes!=2
        fig, ax = plt.subplots(dpi=300)
        fig.set_tight_layout(True)
        plt.gca().set_facecolor('silver')
        plt.gca().margins(0)  
        # Create map layers
        # create_map_raster(data['lanes'], data['crosswalks'], data['route_lanes'])
        # create_map_raster(data['lanes'], data['crosswalks'], data['route_lanes'], data['right_boundary'], data['left_boundary'])
        # create_map_raster(data['lanes'], data['crosswalks'], None, data['right_boundary'], data['left_boundary'])
        if special_lanes is not None:
            unique_tuples = list({class_label: array for class_label, array in special_lanes.values()}.items())
            create_map_raster(deepcopy(data['lanes']), special_lanes=unique_tuples, num_lanes=num_lanes)
        else:
            # create_map_raster(deepcopy(data['lanes']), None,None, deepcopy(data['right_boundary']), deepcopy(data['left_boundary']),special_lanes=unique_tuples, num_lanes=num_lanes)
            if 'map_lanes' in data.keys():
                create_map_raster(deepcopy(data['map_lanes']), num_lanes=num_lanes, crosswalks=deepcopy(data['crosswalks']))
            else:
                create_map_raster(deepcopy(data['lanes']), num_lanes=num_lanes, crosswalks=deepcopy(data['crosswalks']))
            # create_map_raster(deepcopy(data['lanes']), num_lanes=num_lanes)
            # create_map_raster(data['route_lanes'], data['crosswalks'], np.array([]))
            # create_map_raster(data['right_boundary'], data['crosswalks'], np.array([]))q
            # create_map_raster(data['left_boundary'], data['crosswalks'], np.array([]))
            # create_map_raster(data['lanes'], data['crosswalks'], np.array([]))

        # Create agent layers
        create_ego_raster(data['ego_agent_past'][-1])
        colors = ['yellow', 'green', 'm', 'c'] # unknown, vehicle, pedestrian, cyclist
        agents_types = (data['neighbor_agents_past'][:,0,-3:].argmax(-1)+1) * data['neighbor_agents_past'][:,0,-3:].sum(-1).astype(int)
        agents_colors = [colors[i] for i in agents_types]
        agents_colors = agents_colors
        if 'neighbor_agents_past' in data.keys():
            create_agents_raster(data['neighbor_agents_past'][:, -1], agents_colors)

        # Draw past and future trajectories
        if draw_traj:
            draw_trajectory(data['ego_agent_past'], np.array([]))
            draw_trajectory(data['ego_agent_future'], np.array([]))
        # draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'])
        # draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'])

        figure_center = [0,0]
        if figure_center is not None:
            center_x = figure_center[0]  # Replace with your actual center x-coordinate
            center_y = figure_center[1]  # Replace with your actual center y-coordinate
            range_crop = max(30,np.abs(np.linalg.norm(data['ego_agent_future'][:,:2], axis=-1)).max()+20)  # Crop with the max of ego traveling distance + 20 meters, or a minimum of 30 meters
            # # Set axis limits to crop around the center
            ax.set_xlim(center_x - range_crop, center_x + range_crop)
            ax.set_ylim(center_y - range_crop, center_y + range_crop)
        # Remove X and Y Axis Labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        # Optional: remove tick labels if desired
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        
        
        plt.show()
        # plt.savefig(img_name)
        plt.savefig(img_name, bbox_inches='tight', pad_inches=-0.05)
        plt.close()

        

        # print('')

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def work(self, save_dir, debug=False):
        # Reading the JSON file content
        with open("/home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/scenario_situation_gpt_info.json", 'r') as file:
            safety_json = json.load(file)
        self.pbar = tqdm(total=len(list(self._scenarios)))
        self.pbar.set_description(f"Processing ...")
        # for scenario in tqdm(self._scenarios):
        for scenario in self._scenarios:
            map_name = scenario._map_name
            # print(map_name)
            token = scenario.token
            scenario_type = scenario.scenario_type
            # if token == "030060ba3a165b3f":
            #     print(token)
            # else:
            #     continue
            # print(token)
            self.scenario = scenario
            self.map_api = scenario.map_api        
            # get agent past tracks
            ego_agent_past, time_stamps_past = self.get_ego_agent() # x,y,h,vx,vy,ax,ay
            neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents() # track_token, vx, vy, h, w, l, x, y

            dist_to_ego = [np.linalg.norm(neighbor_agents_past[-1][i][-2:] - ego_agent_past[-1,:2]) for i in range(len(neighbor_agents_past[-1]))]
            if len(dist_to_ego)==0:
                continue
            closest_idx = np.argmin(dist_to_ego)
            xyh_closest_neighbor = torch.tensor([neighbor_agents_past[-1][closest_idx][-2].item(), neighbor_agents_past[-1][closest_idx][-1].item(), neighbor_agents_past[-1][closest_idx][3].item()])

            ego_agent_past, neighbor_agents_past, neighbor_indices = \
                agent_past_process(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, self.num_agents) # # x,y,h,vx,vy,yaw,l,w,type_one_hot

            vector_map = self.get_map()
            vector_map2 = self.get_map_agent2(xyh_closest_neighbor)
            # get agent future tracks
            ego_agent_future = self.get_ego_agent_future()
            neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

            # gather data
            data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future}
            data.update(vector_map)

            # # visualization
            # if debug:
            #     self.plot_scenario(data)
            vector_map['lanes'] = np.concatenate((vector_map['lanes'][None],vector_map2['lanes'][None]), 0)
            vector_map['left_boundary'] = np.concatenate((vector_map['left_boundary'][None],vector_map2['left_boundary'][None]), 0)
            vector_map['right_boundary'] = np.concatenate((vector_map['right_boundary'][None],vector_map2['right_boundary'][None]), 0)
            vector_map['route_lanes'] = np.concatenate((vector_map['route_lanes'][None],vector_map2['route_lanes'][None]), 0)
            vector_map['crosswalks'] = np.concatenate((vector_map['crosswalks'][None],vector_map2['crosswalks'][None]), 0)
            data.update(vector_map)
            # Include lane feasbility and template generation here
            #Neihgbor: x,y,h,?,?,?,l,w,type_one_hot
            ego_width = get_pacifica_parameters().width
            ego_length = get_pacifica_parameters().length
            
            ego_width = np.ones(ego_agent_past.shape[0], dtype=ego_agent_past.dtype)*ego_width
            ego_length = np.ones(ego_agent_past.shape[0], dtype=ego_agent_past.dtype)*ego_length
            ego_past_ = np.zeros((ego_agent_past.shape[0], 11), dtype=ego_agent_past.dtype)
            ego_past_[:,:5] = ego_agent_past[:,:5]
            ego_past_[:,5] = ego_length
            ego_past_[:,6] = ego_width
            ego_past_[:,7] = 1.86 # some arbitraty vehicle hight
            ego_past_[:,8] = 1.0 # vehicle type

            neighbor_agents_past_ = np.zeros_like(neighbor_agents_past[:,:,:11])
            neighbor_agents_past_[:,:,5], neighbor_agents_past_[:,:,6] = neighbor_agents_past[:,:,6], neighbor_agents_past[:,:,7]
            neighbor_agents_past_[:,:,:5] = neighbor_agents_past[:,:,:5]
            neighbor_agents_past_[:,:,7] = 1.86
            neighbor_agents_past_[:,:,8:] = neighbor_agents_past[:,:,-3:]

            data["ego_agent_past"] = ego_past_
            data["neighbor_agents_past"] = neighbor_agents_past_

            
            speed_limit = 29.056772232055664 #TODO: Update
            self_type = 2.0 # surface street
            boundary_type = 0.0
            traffic_light = np.array([6, 5, 4, 0]) #green, yellow, red, unknown
            interpolating = 0.0
            stop_signs = 0.0

            # num_drivable_lanes = self.num_lanes

            map_lanes = np.zeros((2,self.num_lanes,self.num_points,17), dtype=ego_agent_past.dtype)
            map_lanes[...,:3] = vector_map['lanes'][...,:3] # center lanes
            map_lanes[...,3:6] = vector_map['left_boundary'][...,:3] # left boundary
            map_lanes[...,6:9] = vector_map['right_boundary'][...,:3] # right boundary
            map_lanes[...,9] = speed_limit
            map_lanes[...,10] = self_type
            map_lanes[...,11] = boundary_type
            map_lanes[...,12] = boundary_type
            map_lanes[...,13] = traffic_light[vector_map['lanes'][...,-4:].argmax(-1)] # traffic light
            map_lanes[...,14] = interpolating
            map_lanes[...,15] = stop_signs
            
            cross_walks = vector_map['crosswalks']

            gt = np.zeros((2,80,5), dtype=ego_agent_future.dtype)
            gt[:,:,:3] = np.concatenate((ego_agent_future[np.newaxis], neighbor_agents_future[0,np.newaxis]), 0)
            ego = np.concatenate((ego_past_[np.newaxis], neighbor_agents_past_[0,np.newaxis]), 0)
            object_type = (ego[:,-1,8:].argmax(-1)+1) * ego[:,-1,8:].sum(-1).astype(int)
            gameformer_data = {
                'ego': ego,
                'neighbors': neighbor_agents_past_[1:],
                'map_lanes': map_lanes,
                'map_crosswalks': cross_walks,
                'gt_future_states': gt,
                'object_type': object_type,
                }
            data.update(gameformer_data)
            # possible_directions, lines_to_plot_tupple = get_direction_plausibility_02_nuplan(deepcopy(ego), deepcopy(map_lanes))
            lines_to_plot_tupple = None
            # visualization
            
            # save to disk
            if sum(gt[0,:,0]!=0)<20 or sum(gt[0,:,0]!=0)<20 or sum(ego[0,:,0]!=0)<5 or sum(ego[0,:,0]!=0)<5:
                continue
            
            
            # get vector set map
            vector_map_ego_drivable, feasible_directions, directions_lanes = self.get_ego_drivable_lanes(ego[0], gt[0])
            # print(feasible_directions)
            
            direction_classes = np.array(self.direction_classes)
            possible_directions = feasible_directions
            possible_directions_cls = [np.where(possible_direction_i == direction_classes)[0][0] for possible_direction_i in possible_directions]
            not_possible_directions = [direction_class_i for direction_class_i in direction_classes if direction_class_i not in possible_directions]
            not_possible_directions_cls = [np.where(direction_class_i == direction_classes)[0][0] for direction_class_i in direction_classes if direction_class_i not in possible_directions]

            if debug:
                data_selected_lanes = {"lanes": vector_map_ego_drivable['lanes'][...,:3], "right_boundary": vector_map_ego_drivable['right_boundary'][...,:3], "left_boundary": vector_map_ego_drivable['left_boundary'][...,:3]}
                data_selected_lanes.update({"ego_agent_past": data["ego_agent_past"], "ego_agent_future": data["ego_agent_future"]})
                # self.plot_scenario(data_selected_lanes, special_lanes=directions_lanes, draw_traj=False)
            else:
                self.save_to_disk(f"{save_dir}npz", data)
            # data['lanes'][...,2]
            # self.get_ego_drivable_lanes(ego[0], gt[0])
            # print(feasible_directions)
            templateLLM_filename = f"{save_dir}templateLLM/{data['map_name']}_{data['token']}.txt"
            gpt_fig_filename = f"{save_dir}gpt_figures/{data['map_name']}_{data['token']}.png"
            gpt_positive_filename = f"{save_dir}gpt_positive_prompts/{data['map_name']}_{data['token']}.txt"
            gpt_negative_filename = f"{save_dir}gpt_negative_prompts/{data['map_name']}_{data['token']}.txt"
            gpt_meta_filename = f"{save_dir}gpt_meta_prompts/{data['map_name']}_{data['token']}.txt"
            # gpt_filename = f"{save_dir}_gpt_prompts/{data['map_name']}_{data['token']}.txt"
            
            valid_instruct = True

            if len(possible_directions)==0:
               valid_instruct = False

            instruct_dict = {}

            # if
            if valid_instruct:
                # data["ego_agent_future"][:,0]
                if sum(data["ego_agent_future"][:,0]==0) < len(data["ego_agent_future"][:,0])//4 and sum(data["ego_agent_past"][:,0] == 0) < len(data["ego_agent_past"][:,0])//2:
                    instruct_dict = {f"Agent-{1}": self.get_agent_caption(data["ego_agent_past"], data["ego_agent_future"])}
                    if sum(data['neighbor_agents_future'][0][:,0] == 0) < len(data['neighbor_agents_future'][0][:,0])//4 and sum(data['neighbor_agents_past'][0][:,0] == 0) < len(data['neighbor_agents_past'][0][:,0])//2:
                        instruct_dict.update({f"Agent-{2}": self.get_agent_caption(data['neighbor_agents_past'][0], data['neighbor_agents_future'][0])})
            agent_json = instruct_dict
            if 'Agent-1' not in agent_json.keys():
                valid_instruct = False
            
            if valid_instruct:
                llm_json = generate_template_json(agent_json, None, possible_directions, not_possible_directions, direction_classes)

                # if debug:
                #     self.plot_scenario(data, lines_to_plot_tupple, img_name='ex1.png', draw_traj=True)
                #     print(f"{scenario_type}; {agent_json['Agent-1']['direction 0.1to8']}: {agent_json['Agent-1']['direction 0.1to4']} -> {agent_json['Agent-1']['direction 4to8']}")
                #     print(json.loads(llm_json.split('\n')[0])['Reasoning'])
                if args.save_gpt_prompt:
                    # Initialize situation validity and set variables for agent behavior
                    all_acceleration = ['stationary', 'constant velocity', 'mild acceleration', 'mild deceleration', 'moderate acceleration', 'moderate deceleration', 'aggressive acceleration', 'aggressive deceleration', 'extreme acceleration', 'extreme deceleration']
                    maintain_speed_zone = ['constant velocity']
                    accel_zones = ['mild acceleration', 'moderate acceleration', 'aggressive acceleration', 'extreme acceleration']
                    decel_zones = ['mild deceleration',  'moderate deceleration',  'aggressive deceleration', 'extreme deceleration']
                    valid_situation = True
                    initial_speed = agent_json['Agent-1']['speed 0.1to4']
                    subsequent_speed = agent_json['Agent-1']['speed 4to8']
                    initial_acceleration = agent_json['Agent-1']['acceleration 0.1to4']
                    subsequent_acceleration = agent_json['Agent-1']['acceleration 4to8']

                    if initial_acceleration == 'stationary':
                        if subsequent_acceleration == 'stationary':
                            situation = 'not_moving'
                        elif subsequent_acceleration in accel_zones or subsequent_speed != 'stationary':
                            # Prepares to move after waiting
                            situation = 'wait_then_move'
                        else:
                            # Invalid movement condition encountered
                            valid_situation = False
                    elif initial_acceleration in maintain_speed_zone and subsequent_acceleration in maintain_speed_zone:
                        situation = 'maintaining_speed'
                    elif initial_acceleration in decel_zones:
                        # Decelerating initially
                        if subsequent_acceleration == 'stationary':
                            # Fully stops after decelerating
                            situation = 'stopping'
                        elif subsequent_acceleration in decel_zones + maintain_speed_zone:
                            # Slows down, preparing to stop
                            situation = 'slowing_down'
                        elif subsequent_acceleration in accel_zones:
                            situation = 'slowdown_then_speedup'
                        else:
                            # Unexpected deceleration condition
                            valid_situation = False
                    elif initial_acceleration in maintain_speed_zone:
                        if subsequent_acceleration == 'stationary':
                            # Fully stops after decelerating
                            situation = 'stopping'
                        elif subsequent_acceleration in decel_zones:
                            situation = 'slowing_down'
                        elif subsequent_acceleration in accel_zones:
                            situation = 'speeding_up'
                        else:
                            valid_situation = False
                    elif initial_acceleration in accel_zones:
                        if subsequent_acceleration in ['stationary']+decel_zones:
                            situation = 'speedup_then_slowdown'
                        elif subsequent_acceleration in accel_zones + maintain_speed_zone:
                            situation = 'speeding_up'
                        else:
                            valid_situation = False
                    else:
                        # No valid movement or deceleration detected
                        valid_situation = False

                    if situation not in safety_json[scenario_type]['safe'].keys():
                        valid_situation = False
                    if valid_situation:
                        what_safe_to_do = safety_json[scenario_type]['safe'][situation]
                        if 'unsafe' in safety_json[scenario_type].keys():
                            what_unsafe_to_do = safety_json[scenario_type]['unsafe'][situation]
                    
                    # If situation is invalid, handle accordingly
                    if not valid_situation:
                        situation = 'undefined_behavior'
                        continue
                    
                    main_driving_behaviour = agent_json['Agent-1']['direction 0.1to8']
                    if agent_json['Agent-1']['direction 0.1to4'] == 'stationary':
                        if agent_json['Agent-1']['direction 4to8'] == 'stationary':
                            driving_behaviour = f"remain stationary"
                        elif agent_json['Agent-1']['direction 4to8'] != 'stationary':
                            driving_behaviour = f"{agent_json['Agent-1']['direction 0.1to8']}, but first remain stationary waiting, then {agent_json['Agent-1']['direction 4to8']} with {agent_json['Agent-1']['speed 4to8']} speed ({agent_json['Agent-1']['speed 8']}) and {agent_json['Agent-1']['acceleration 4to8']}"
                    elif situation=='stopping':
                        if not 'turn' in agent_json['Agent-1']['direction 0.1to4'] and not 'turn' in agent_json['Agent-1']['direction 4to8']:
                            main_driving_behaviour = "stop"
                            driving_behaviour = f"stop; where it will first {agent_json['Agent-1']['direction 0.1to4']} with {agent_json['Agent-1']['speed 0.1to4']} speed ({agent_json['Agent-1']['speed 4']}) and {agent_json['Agent-1']['acceleration 0.1to4']}, then stop and remain stationary"
                        elif 'turn' in agent_json['Agent-1']['direction 0.1to4']:
                            main_driving_behaviour = f"{agent_json['Agent-1']['direction 0.1to4']} and stop"
                            driving_behaviour = f"{agent_json['Agent-1']['direction 0.1to4']} and stop; where it will first {agent_json['Agent-1']['direction 0.1to4']} with {agent_json['Agent-1']['speed 0.1to4']} speed ({agent_json['Agent-1']['speed 4']}) and {agent_json['Agent-1']['acceleration 0.1to4']}, then {agent_json['Agent-1']['direction 4to8']} with {agent_json['Agent-1']['speed 4to8']} speed ({agent_json['Agent-1']['speed 8']}) and {agent_json['Agent-1']['acceleration 4to8']}"
                        else:
                            driving_behaviour = "unknown"
                            continue
                    elif agent_json['Agent-1']['direction 4to8'] == 'stationary':
                        driving_behaviour = f"{agent_json['Agent-1']['direction 0.1to4']} with {agent_json['Agent-1']['speed 0.1to4']} speed ({agent_json['Agent-1']['speed 4']}) and {agent_json['Agent-1']['acceleration 0.1to4']}, then stop"
                    elif agent_json['Agent-1']['direction 0.1to4'] == 'move straight' and agent_json['Agent-1']['direction 4to8'] == 'move straight':
                        driving_behaviour = f"{agent_json['Agent-1']['direction 0.1to8']}; first with {agent_json['Agent-1']['speed 0.1to4']} speed ({agent_json['Agent-1']['speed 4']}) and {agent_json['Agent-1']['acceleration 0.1to4']}, then with {agent_json['Agent-1']['speed 4to8']} speed ({agent_json['Agent-1']['speed 8']}) and {agent_json['Agent-1']['acceleration 4to8']}"
                    elif 'turn' in agent_json['Agent-1']['direction 0.1to4'] and 'turn' in agent_json['Agent-1']['direction 4to8']:
                        driving_behaviour = "unknown" # we should ignore cases with two turns, at it is not expected in 8 seconds.
                        continue
                    elif 'turn' in agent_json['Agent-1']['direction 0.1to4']:
                        driving_behaviour = f"{agent_json['Agent-1']['direction 0.1to4']}; where it will first take a {agent_json['Agent-1']['direction 0.1to4']} with {agent_json['Agent-1']['speed 0.1to4']} speed ({agent_json['Agent-1']['speed 4']}) and {agent_json['Agent-1']['acceleration 0.1to4']}, then {agent_json['Agent-1']['direction 4to8']} with {agent_json['Agent-1']['speed 4to8']} speed ({agent_json['Agent-1']['speed 8']}) and {agent_json['Agent-1']['acceleration 4to8']}"
                    elif 'turn' in agent_json['Agent-1']['direction 4to8']:
                        driving_behaviour = f"{agent_json['Agent-1']['direction 4to8']}; where it will first {agent_json['Agent-1']['direction 0.1to4']} with {agent_json['Agent-1']['speed 0.1to4']} speed ({agent_json['Agent-1']['speed 4']}) and {agent_json['Agent-1']['acceleration 0.1to4']}, then {agent_json['Agent-1']['direction 4to8']} with {agent_json['Agent-1']['speed 4to8']} speed ({agent_json['Agent-1']['speed 8']}) and {agent_json['Agent-1']['acceleration 4to8']}"
                    elif 'turn' in agent_json['Agent-1']['direction 0.1to8']:
                        driving_behaviour = f"{agent_json['Agent-1']['direction 0.1to8']}; where it will first {agent_json['Agent-1']['direction 0.1to4']} with {agent_json['Agent-1']['speed 0.1to4']} speed ({agent_json['Agent-1']['speed 4']}) and {agent_json['Agent-1']['acceleration 0.1to4']}, then {agent_json['Agent-1']['direction 4to8']} with {agent_json['Agent-1']['speed 4to8']} speed ({agent_json['Agent-1']['speed 8']}) and {agent_json['Agent-1']['acceleration 4to8']}"
                    else:
                        driving_behaviour = f"{agent_json['Agent-1']['direction 0.1to8']}; where it will first {agent_json['Agent-1']['direction 0.1to4']} with {agent_json['Agent-1']['speed 0.1to4']} speed ({agent_json['Agent-1']['speed 4']}) and {agent_json['Agent-1']['acceleration 0.1to4']}, then {agent_json['Agent-1']['direction 4to8']} with {agent_json['Agent-1']['speed 4to8']} speed ({agent_json['Agent-1']['speed 8']}) and {agent_json['Agent-1']['acceleration 4to8']}"

                    
                    # safe_example = safety_json[scenario_type]['safe_example'][situation]
                    # unsafe_example = safety_json[scenario_type]['unsafe_example'][situation]
                    # safe_example_with_context = safety_json[scenario_type]['safe_example'][situation]['with_context']
                    # safe_example_without_context = safety_json[scenario_type]['safe_example'][situation]['without_context']
                    # unsafe_example_with_context = safety_json[scenario_type]['unsafe_example'][situation]['with_context']
                    # unsafe_example_without_context = safety_json[scenario_type]['unsafe_example'][situation]['without_context']
                    # safe_example, unsafe_example = safe_example_without_context, unsafe_example_without_context

                    # gpt_template = f""""""
                    gpt_meta_data_to_keep = {
                        'scenario_type': scenario_type,
                        'situation': situation,
                        'what_safe_to_do': what_safe_to_do,
                        'what_unsafe_to_do': what_unsafe_to_do if 'unsafe' in safety_json[scenario_type] else None,
                        'trajectory_description': driving_behaviour,
                    }

                    scenario_template_data_dir = f"/home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/prompts_templates/{scenario_type}.txt"
                    with open(scenario_template_data_dir, 'r') as file:
                        template_content = file.read()
                    template_content
                    template_content = template_content.replace('{scenario_type}', scenario_type.replace('_', ' '))
                    template_content = template_content.replace('{what_safe_to_do}', what_safe_to_do)
                    if '{what_unsafe_to_do}' in template_content:
                        template_content = template_content.replace('{what_unsafe_to_do}', what_unsafe_to_do)
                    template_content = template_content.replace('{driving_behaviour}', driving_behaviour)
                    template_content = template_content.replace('{main_driving_behaviour}', main_driving_behaviour)
                    
                    exec(template_content)
                    scope = {}
                    exec(template_content, scope)
                    gpt_safe_template = scope['gpt_safe_template']
                    gpt_unsafe_template = scope['gpt_unsafe_template']

                    # print(gpt_safe_template)
                    with open(gpt_positive_filename, 'w') as file:
                        file.write(gpt_safe_template)
                    with open(gpt_negative_filename, 'w') as file:
                        file.write(gpt_unsafe_template)
                    with open(gpt_meta_filename, 'w') as json_file:
                        json.dump(gpt_meta_data_to_keep, json_file, indent=4)  # indent for pretty printing
                    
                    # with open(gpt_filename, 'w') as file:
                    #     file.write(gpt_template)
                    # gpt_pos = get_negative_gpt_prompt(scenario_type_gpt, what_can_do, 'as_01')
                    # gpt_neg = get_negative_gpt_prompt(scenario_type_gpt, what_can_do, 'neg_01')
                    self.plot_scenario(data, lines_to_plot_tupple, img_name=gpt_fig_filename, draw_traj=True)
                   
                    # what_can_do = json.loads(llm_json.split('\n')[0])['Reasoning'].split('vehicle ')[1][:-1]
                    # scenario_type_gpt = ' '.join(scenario_type.split('_'))
                    # gpt_pos = get_negative_gpt_prompt(scenario_type_gpt, what_can_do, 'as_01')
                    # gpt_neg = get_negative_gpt_prompt(scenario_type_gpt, what_can_do, 'neg_01')
                    # with open(gpt_positive_filename, 'w') as file:
                    #     file.write(gpt_pos)
                    # with open(gpt_negative_filename, 'w') as file:
                    #     file.write(gpt_neg)
                    

                    # gpt_pos = f"""
                    # """
                    # gpt_neg = f"""
                    # """
                    # gpt_positive_filename
                    # gpt_negative_filename
                    

            # llm_json = self.get_agent_caption(data["ego_agent_past"], data["ego_agent_future"])
            
            # if json.loads(llm_json[0])['Direction_cls'] != -1:
            #     # print(self.direction_classes[json.loads(llm_json[0])['Direction_cls']])
            #     print(json.loads(llm_json[0])['Reasoning'])
            # else:
            #     print('Unknown')
            # llm_json = "".join(llm_json)
                if not debug:
                    with open(templateLLM_filename, 'w') as file:
                        file.write(llm_json)
            
            self.pbar.update(1)
        self.pbar.close()

    def get_vel_from_traj(self, step1xy, step2xy, time_difference=0.1):
        x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocity = distance / time_difference
        return velocity

    def get_agent_caption(self, history, future):
        navigation_extractor = futureNavigation()
        # normalizing the view with respect to the agent, to cacluate directional feature. Irrespective of other agents, so using the location information with the normalized version is not correct
        history_normalized_view = history.copy()
        # history_ = history[history[:,0]!=0]
        # history[history[:,0]!=0].copy()[0,:2], history[history[:,0]!=0].copy()[0,2]
        # history.copy()[0,:2], history.copy()[0,2]
        agent_center, agent_angle = history.copy()[0,:2], history.copy()[0,2]
        valid_mask = history[:,0]!=0
        
        # print(history[:,:2])
        history_normalized_view[valid_mask,:5] = agent_norm(history.copy()[valid_mask,:], agent_center, agent_angle, impute=True) # with respect to itself
        history_instructs = navigation_extractor.get_navigation_dict_history(torch.tensor(history_normalized_view[:,:5]))

        if future is not None:
            future_normalized_view = future.copy()
            agent_center, agent_angle = future.copy()[0,:2], future.copy()[0,2]
            valid_mask = future[:,0]!=0
            future_normalized_view[valid_mask,:] = agent_norm(future.copy()[valid_mask,:], agent_center, agent_angle, impute=False) # with respect to itself
            # future_normalized_view[valid_mask,:5] = agent_norm(future.copy()[valid_mask,:], agent_center, agent_angle, impute=True) # with respect to itself
            future_instructs = navigation_extractor.get_navigation_dict(torch.tensor(future_normalized_view))
        else:
            return history_instructs
        
        return {
            **history_instructs,
            **future_instructs
            }

        # agent = {"Agent-1": future_instructs}
        # template_instruct = "You are generating the future motion plan for the ego vehicle. Reason about and predict the future multimodal trajectory embeddings of two agents—the ego (Agent-1) and another agent it interacts with (Agent-2) —based on the observed scene embeddings and the following ego instruction: Make the ego vehicle "
        # direction = "stay stationary" if "stationary" in agent['Agent-1']['direction 0.1to8'] else agent['Agent-1']['direction 0.1to8']
        # direction1 = "stay stationary" if "stationary" in agent['Agent-1']['direction 0.1to4'] else agent['Agent-1']['direction 0.1to4']
        # direction2 = "stay stationary" if "stationary" in agent['Agent-1']['direction 4to8'] else agent['Agent-1']['direction 4to8']
        # speed1 = agent['Agent-1']['speed 0.1to4']
        # speed2 = agent['Agent-1']['speed 4to8']
        # speed1 = f" with a {speed1} speed" if ("stationary" != speed1 and "INVALID" != speed1) else ""
        # speed2 = f" with a {speed2} speed" if ("stationary" != speed2 and "INVALID" != speed2) else ""
        # accel1 = agent['Agent-1']['acceleration 0.1to4']
        # accel1 = f" and a {accel1}" if ("stationary" != accel1) else ""
        # accel2 = agent['Agent-1']['acceleration 4to8']
        # accel2 = f" and a {accel2}" if ("stationary" != accel2) else ""
        # instruct1 = template_instruct + f"{direction}."
        # instruct2 = template_instruct + f"{direction}, where it will first {direction1}{speed1}, then {direction2}{speed2}."
        # instruct3 = template_instruct + f"{direction}, where it will first {direction1}, then {direction2}."
        # action_reason = f"The ego vehicle can {direction}, where it will first {direction1}{speed1}{accel1}, then {direction2}{speed2}{accel2}."
        # # future_instructs['direction 0.1to8'], future_instructs['direction 0.1to4'], future_instructs['speed 0.1to4'], future_instructs['acceleration 0.1to4'], future_instructs['direction 4to8'], future_instructs['speed 4to8'], future_instructs['acceleration 4to8']
        # return [str({"Instruction": instruct1, "Reasoning": action_reason, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n"]*3
        # # return {
        # #     **history_instructs,
        # #     **future_instructs
        # #     }

    def get_directions_nuplan(self, ego_past, map_lanes, ground_truth, starting_point=None):
        direction_classifier = DirectionClassifier(step_t=1)
        map_lanes = map_lanes[:,::2]
        undefined_lanes_mask = np.linalg.norm(map_lanes[:,:,:2], axis=2)!=0
        # valid_lane_mask = bike_lanes_mask * undefined_lanes_mask
        valid_lane_mask = undefined_lanes_mask
        map_lanes[~valid_lane_mask] = np.inf

        if starting_point is None:
            map_lanes_norms = np.linalg.norm(map_lanes[...,:2], axis=-1)
            # Find the index of the minimum value in the 2D array of norms
            min_index_flat = np.argmin(map_lanes_norms)
            # Convert the flat index to a tuple (row, column)
            min_index_2d = np.unravel_index(min_index_flat, map_lanes_norms.shape)
            reference_slice = map_lanes[min_index_2d]
            current_lane_idx, current_point_idx = min_index_2d
            starting_point = reference_slice[:3]# x,y,h

        ##

        ##
        possible_directions = []
        possible_classes = []
        found_lanes = 0
        lines_to_plot_tupple = {}
        possible_colors = ['g', 'b', 'c', 'm', 'yellow', 'purple', 'pink', 'orange']
        possible_map_lanes = map_lanes
        for i in range(len(possible_map_lanes)):
            if np.any(possible_map_lanes[i][:,0]!=np.inf):
                target_lane = possible_map_lanes[i][possible_map_lanes[i][:,0]!=np.inf]
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
                    # entries with straight after turn are removed, as this require manuvering taking two turns and the current rule-based detection does not capture that
                    four_splits_directions = remove_entries_after_turn(four_splits_directions) 
                    for direction_str, direction_cls in four_splits_directions:
                        # We need to detect direction at maximum drivable point, minimum drivable point, and half way
                        if direction_cls!=-1:
                            starting_point_idx = np.linalg.norm((drivable_target_lane[:,:3] - starting_point)[:,:2], axis=-1).argmin()
                            lines_to_plot_tupple[found_lanes] = (direction_cls, drivable_target_lane[starting_point_idx:])
                            possible_directions.append(direction_classifier.classes[direction_cls])
                            possible_classes.append(direction_cls)
                            found_lanes+=1
        
        unique_classes = np.unique(possible_classes)
        # print('Possible directions:')
        # for i in range(len(unique_classes)):
        #     print(f"- {unique_classes[i]} ({possible_colors[unique_classes[i]]}): {direction_classifier.classes[unique_classes[i]]}")
        # print(found_lanes)
        # print(np.unique(possible_directions))
        # print(time.time()-t_)
        return np.unique(possible_directions), lines_to_plot_tupple


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/data/cache/mini')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/data/cache/train_boston')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/home/felembaa/projects/iMotion-LLM-ICLR/datasets/nuplan/data/cache/cache/train_pittsburgh')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/val/data/cache/val')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/cache/train_boston')
    parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/test/data/cache/test')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/train_pittsburgh')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/train_boston')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/train_combined')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/train_singapore')
    
    
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/train/data/cache/cache/train_vegas_2')
    # parser.add_argument('--data_path', type=str, help='path to raw data', default='/ibex/project/c2278/felembaa/datasets/nuplan/data/cache/train_pittsburgh')
    
    # parser.add_argument('--map_path', type=str, help='path to map data', default='/ibex/project/c2278/felembaa/datasets/nuplan/maps')
    parser.add_argument('--map_path', type=str, help='path to map data', default='/ibex/project/c2278/felembaa/datasets/nuplan/maps')
    
    # parser.add_argument('--save_path', type=str, help='path to save processed data', default='/ibex/project/c2278/felembaa/datasets/nuplan/data/cache/train_boston_processed_viz')
    parser.add_argument('--save_path', type=str, help='path to save processed data', default='/ibex/project/c2278/felembaa/datasets/nuplan/test_gpt_prompt_14types')
    # parser.add_argument('--save_path', type=str, help='path to save processed data', default='/ibex/project/c2278/felembaa/datasets/nuplan/data/cache/train_pittsburgh_processed_viz')
    
    # parser.add_argument('--scenarios_per_type', type=int, default=600, help='number of scenarios per type')
    parser.add_argument('--scenarios_per_type', type=int, default=10, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', default=None, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios')
    parser.add_argument('--debug', action="store_true", help='if visualize the data output')
    parser.add_argument('--run_num', type=int, help='', default=-1)
    parser.add_argument('--processes', type=int, help='multiprocessing process num', default=1)
    parser.add_argument('--map_names', help='Filter by map names (useful to filter test and validation set)', default=None)
    parser.add_argument('--save_gpt_prompt', help='Saving figures and gpt prompt to extract feasible and infeasible complex instructions', default=True)
    parser.add_argument('--scenario_type_idx', help='', default=-1, type=int)

    target_scenario_types_ = ['waiting_for_pedestrian_to_cross', 'traversing_crosswalk', 'accelerating_at_crosswalk', 'traversing_intersection', 'behind_bike', 'behind_long_vehicle', 'following_lane_with_lead', 'following_lane_with_slow_lead', 'following_lane_without_lead', 'stopping_with_lead', 'starting_protected_cross_turn', 'starting_protected_noncross_turn', 'starting_unprotected_cross_turn', 'starting_unprotected_noncross_turn']
    target_scenario_types = target_scenario_types_
    # map_names = ['us-pa-pittsburgh-hazelwood']
    # map_names = ['us-ma-boston']
    # map_names = ['sg-one-north']
    
    args = parser.parse_args()
    save_path_ = args.save_path
    # for args.scenario_type_idx in range(len(target_scenario_types_)):
    if True:
        print(f"run_num = {args.run_num}")
        # if args.run_num!=-1:
        #     target_scenario_types = split_list(target_scenario_types, 10)[args.run_num]
        # else:
        #     target_scenario_types = target_scenario_types
        if args.run_num!=-1:
            target_scenario_types = [target_scenario_types[args.run_num]]
            args.save_path = save_path_+'/'+target_scenario_types[0]+'/'
        else:
            target_scenario_types = target_scenario_types
            args.save_path = save_path_+'/all_types/'

        # print(f"***{args.scenario_type_idx}***")
        # if args.scenario_type_idx != -1:
            # target_scenario_types = [target_scenario_types_[args.scenario_type_idx]]
            # args.save_path = save_path_+'/'+target_scenario_types[0]+'/'
        # else:
            # target_scenario_types = target_scenario_types_
        if args.map_names is not None:
            args.map_names = [args.map_names]
        # args.debug = True
        # create save folder
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.save_path+"npz", exist_ok=True)
        os.makedirs(args.save_path+"templateLLM", exist_ok=True)
        if args.save_gpt_prompt:
            os.makedirs(args.save_path+"gpt_figures", exist_ok=True)
            os.makedirs(args.save_path+"gpt_positive_prompts", exist_ok=True)
            os.makedirs(args.save_path+"gpt_negative_prompts", exist_ok=True)
            os.makedirs(args.save_path+"gpt_meta_prompts", exist_ok=True)
            # os.makedirs(args.save_path+"_gpt_prompts", exist_ok=True)
            
            
        # get scenarios
        map_version = "nuplan-maps-v1.0"
        # map_version = "nuplan-maps-v1.1"
        sensor_root = None
        db_files = None
        scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
        # scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=1)
        builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version, scenario_mapping=scenario_mapping)
        # target_scenario_types = [
        #     'starting_left_turn',
        #     'starting_right_turn',
        #     'starting_straight_traffic_light_intersection_traversal',
        #     'stopping_with_lead',
        #     'high_lateral_acceleration',
        #     'high_magnitude_speed',
        #     'low_magnitude_speed',
        #     'traversing_pickup_dropoff',
        #     'waiting_for_pedestrian_to_cross',
        #     'behind_long_vehicle',
        #     'stationary_in_traffic',
        #     'near_multiple_vehicles',
        #     'changing_lane',
        #     'following_lane_with_lead',
        # ]
        # target_scenario_types = [
        #     'accelerating_at_crosswalk',
        #     'accelerating_at_stop_sign',
        #     'accelerating_at_stop_sign_no_crosswalk',
        #     'accelerating_at_traffic_light',
        #     'accelerating_at_traffic_light_with_lead',
        #     'accelerating_at_traffic_light_without_lead',
        #     'behind_bike',
        #     'behind_long_vehicle',
        #     'behind_pedestrian_on_driveable',
        #     'behind_pedestrian_on_pickup_dropoff',
        #     'changing_lane',
        #     'changing_lane_to_left',
        #     'changing_lane_to_right',
        #     'changing_lane_with_lead',
        #     'changing_lane_with_trail',
        #     'crossed_by_bike',
        #     'crossed_by_vehicle',
        #     'following_lane_with_lead',
        #     'following_lane_with_slow_lead',
        #     'following_lane_without_lead',
        #     'high_lateral_acceleration',
        #     'high_magnitude_jerk',
        #     'high_magnitude_speed',
        #     'low_magnitude_speed',
        #     'medium_magnitude_speed',
        #     'near_barrier_on_driveable',
        #     'near_construction_zone_sign',
        #     'near_high_speed_vehicle',
        #     'near_long_vehicle',
        #     'near_multiple_bikes',
        #     'near_multiple_pedestrians',
        #     'near_multiple_vehicles',
        #     'near_pedestrian_at_pickup_dropoff',
        #     'near_pedestrian_on_crosswalk',
        #     'near_pedestrian_on_crosswalk_with_ego',
        #     'near_trafficcone_on_driveable',
        #     'on_all_way_stop_intersection',
        #     'on_carpark',
        #     'on_intersection',
        #     'on_pickup_dropoff',
        #     'on_stopline_crosswalk',
        #     'on_stopline_stop_sign',
        #     'on_stopline_traffic_light',
        #     'on_traffic_light_intersection',
        #     'starting_high_speed_turn',
        #     'starting_left_turn',
        #     'starting_low_speed_turn',
        #     'starting_protected_cross_turn',
        #     'starting_protected_noncross_turn',
        #     'starting_right_turn',
        #     'starting_straight_stop_sign_intersection_traversal',
        #     'starting_straight_traffic_light_intersection_traversal',
        #     'starting_u_turn',
        #     'starting_unprotected_cross_turn',
        #     'starting_unprotected_noncross_turn',
        #     'stationary',
        #     'stationary_at_crosswalk',
        #     'stationary_at_traffic_light_with_lead',
        #     'stationary_at_traffic_light_without_lead',
        #     'stationary_in_traffic',
        #     'stopping_at_crosswalk',
        #     'stopping_at_stop_sign_no_crosswalk',
        #     'stopping_at_stop_sign_with_lead',
        #     'stopping_at_stop_sign_without_lead',
        #     'stopping_at_traffic_light_with_lead',
        #     'stopping_at_traffic_light_without_lead',
        #     'stopping_with_lead',
        #     'traversing_crosswalk',
        #     'traversing_intersection',
        #     'traversing_narrow_lane',
        #     'traversing_pickup_dropoff',
        #     'traversing_traffic_light_intersection',
        #     'waiting_for_pedestrian_to_cross'
        # ]

        # all_target_scenario_types = target_scenario_types
        # target_scenario_types = ['traversing_intersection']
        # target_scenario_types = ['traversing_intersection']
        # 

        # target_scenario_types_ = ['waiting_for_pedestrian_to_cross', 'traversing_crosswalk', 'accelerating_at_crosswalk', 'traversing_intersection', 'behind_bike', 'behind_long_vehicle', 'following_lane_with_lead', 'following_lane_with_slow_lead', 'following_lane_without_lead', 'stopping_with_lead', 'starting_protected_cross_turn', 'starting_protected_noncross_turn', 'starting_unprotected_cross_turn', 'starting_unprotected_noncross_turn']
        # target_scenario_types_ = [item for item in all_target_scenario_types if item not in target_scenario_types_]
        # for target_scenario_types in target_scenario_types_:
            # target_scenario_types = [target_scenario_types]
        # print(f"run_num = {args.run_num}")
        # # if args.run_num!=-1:
        # #     target_scenario_types = split_list(target_scenario_types, 10)[args.run_num]
        # # else:
        # #     target_scenario_types = target_scenario_types
        # if args.run_num!=-1:
        #     target_scenario_types = [target_scenario_types[args.run_num]]
        # else:
        #     target_scenario_types = target_scenario_types


            # target_scenario_types = split_list(target_scenario_types, 10)[-1]
        # target_scenario_types = ['stopping_at_stop_sign_with_lead']
        scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, scenario_types=target_scenario_types, map_names=args.map_names))
        max_workers = 1 if args.debug else None
        max_workers = args.processes
        worker = SingleMachineParallelExecutor(use_process_pool=True if not args.debug else False, max_workers=max_workers)
        scenarios = builder.get_scenarios(scenario_filter, worker)
        print(f'Scenarios: {target_scenario_types}')
        print(f"Total number of scenarios: {len(scenarios)}")
            

        # process data
        del worker, builder, scenario_filter, scenario_mapping
        processor = DataProcessor(scenarios)
        processor.work(args.save_path, debug=args.debug)



