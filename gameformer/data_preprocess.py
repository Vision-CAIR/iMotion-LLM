import torch
import glob
import sys
sys.path.append("..")
sys.path.append(".")
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

def get_vel_from_traj(step1xy, step2xy, time_difference=0.1):
        x1, y1, x2, y2 = step1xy[0], step1xy[1], step2xy[0], step2xy[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        velocity = distance / time_difference
        return velocity

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

def vizualize_background(segment, segment_type, fig, categ, ax):
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
    center_plt_info = [('lightgray','-', 0.1),('gray','solid', 0.4), ('gray','solid', 0.4), ('b','dashed', 0.2),]
    boundry_plt_info = [('lightgray','-', 0.1), ('w','dashed', 0.4),  ('w','solid', 0.4), ('w','solid', 0.4), ('xkcd:yellow','dashed', 0.4), ('xkcd:yellow','dashed', 0.4), ('xkcd:yellow','solid', 0.4), ('xkcd:yellow','solid', 0.4), ('xkcd:yellow','dotted', 0.4),('k', '-', 1.0), ('k', '-', 0.4),]
    edge_plt_info = [('k', '-', 1.0), ('k', '-', 1.0), ('k', '-', 0.4),]
    stopsign_plt_info = [('lightgray','-', 0.1),('r', 'solid',1)]
    crosswalk_plt_info = [('lightgray','-', 0.1),('honeydew', '-',0.3)]
    speedbump_plt_info = [('lightgray','-', 0.1),('purple', '-',0.1)]
    plt_info = [center_plt_info, boundry_plt_info, edge_plt_info, stopsign_plt_info, crosswalk_plt_info, speedbump_plt_info]
    if categ==4 or categ==5:
        segment_type=1
    # init figure
    if not fig is not None:
        dpi = 300
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
        plt.plot(segment[:,0], segment[:,1], plt_info[categ][segment_type][0], linestyle=plt_info[categ][segment_type][1], linewidth=2, alpha=plt_info[categ][segment_type][2])
    elif categ==3:
        circle = Circle((segment[0], segment[1]), 2, color='red', fill=False)
        plt.gca().add_patch(circle)
    elif categ==6:
        if segment_type in [1, 4, 7]:
            color = 'red'
        elif segment_type in [2, 5, 8]:
            color = 'orange'
        else:
            color = 'green'
        circle = Circle((segment[0], segment[1]), 0.7, color=color)
        plt.gca().add_patch(circle)
        # ['unknown', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution', 'green/go', 'red/stop', 'yellow/caution']
    return fig, ax

def vizualize_traj_arrow(history_traj, traj, fig, ax, color, linestyle='dotted', add_arrow=True, alpha=0.5, arrcolor='b'): # single agent, single modality
    # plt.plot(traj[:,0], traj[:,1], linestyle=linestyle, color=color, alpha=alpha, linewidth=3)
    plt.scatter(traj[:, 0], traj[:, 1], color=color, alpha=alpha, s=10, marker='o')
    if add_arrow:
        ax.annotate('', xy=traj[1,:], xytext=history_traj[:2],
                arrowprops=dict(arrowstyle="-|>", color=arrcolor, lw=1, linestyle='solid', alpha=0.5))   

def rect_(x_pos, y_pos, width, height, heading, object_type, color='r', alpha=0.8, fontsize=6):
    rect_x = x_pos - width / 2
    rect_y = y_pos - height / 2
    rect = plt.Rectangle((rect_x, rect_y), width, height, linewidth=2, color=color, alpha=alpha, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_pos, y_pos), heading) + plt.gca().transData)
    plt.gca().add_patch(rect)
    plt.text(x_pos, y_pos, object_type, color='black', fontsize=fontsize, ha='center', va='center')
    
def vizualize_agent_rect(fig, past, object_type, color, agent_prefix=''):
    agent_i = past
    object_type_str = ['Unset', 'Car', 'Pedestrian', 'Cyclic']
    object_type_str = [str(agent_prefix)+':'+object_type_str_i for object_type_str_i in object_type_str]
    rect_(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type_str[int(object_type)], color=color)


class DataProcess(object):
    def __init__(
                self, 
                root_dir=[''],
                point_dir='',
                save_dir='',
                num_neighbors=32
                ):
        # parameters
        self.num_neighbors = num_neighbors
        self.hist_len = 11
        self.future_len = 80
        self.data_files = root_dir
        self.point_dir = point_dir
        self.save_dir = save_dir
        self.drivable_lanes = 6
        self.num_crosswalks = 4
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

    def plt_scene(self, agent_1=None, agent_2=None, other_agents=None, figure_center=None):
        # each category [0:center lanes, 1:boundaries, 2:edges, 3:stop signs, 4:crosswalks, 5:speed bump, 6:traffic light]
        # is plotted seperatly on the same figure, use the correct categ value, refer to the function vizualize_background [in this file]
        lane_polylines = get_polylines(self.lanes)
        lane_types = [value.type for value in self.lanes.values()]
        fig=None
        ax=None
        for i, lane in enumerate(lane_polylines.values()):
            fig, ax = vizualize_background(segment=lane, segment_type=lane_types[i], fig=fig, categ=0, ax=ax)

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
        

        if agent_1 is not None:            
            agent_ = agent_1
            agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
            colors = ['','darkviolet', 'violet', 'darkviolet']
            vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='Ego')
            vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=False) # single agent, single modality
        if agent_2 is not None:            
            agent_ = agent_2
            agent_type = int((agent_[0][-1,-3:].argmax(-1)+1)*agent_[0][-1,-3:].sum())
            colors = ['','teal', 'peru', 'teal']
            vizualize_agent_rect(fig=fig, past=agent_[0], object_type=agent_type, color=colors[agent_type], agent_prefix='Interactive')
            vizualize_traj_arrow(agent_[0][-1,:3], agent_[1][:,:2], fig, ax, color=colors[agent_type], alpha=1, arrcolor='b', add_arrow=False) # single agent, single modality
        if other_agents is not None:
            # alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            alphabets = [i for i in range(50)]
            # alphabets = [i for i in range(len(alphabets))]            
            for agent_idx, agent_ in enumerate(other_agents):
                agent_type = int((agent_[-1,-3:].argmax(-1)+1)*agent_[-1,-3:].sum())
                if agent_type>0:
                    colors = ['','dimgrey', 'sandybrown', 'dimgrey']
                    vizualize_agent_rect(fig=fig, past=agent_, object_type=agent_type, color=colors[agent_type], agent_prefix=alphabets[agent_idx+2])
        
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

    def gen_instruct_caption(self, history_ego, future_ego, history_interactive_neighbor, future_interactive_neighbor, other_neighbors, navigation_extractor=None):
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
        
        agent_type = int((history_ego[-1,-3:].argmax(-1)+1)*history_ego[-1,-3:].sum()) # agent_type = [1:car, 2:pedestrian, 3:cylcist], if 0 then undefined
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
            

            # relative motion to ego
            for agent_i, neighbor in enumerate(other_neighbors):
                self.get_relative_direction(reference_traj=traj[:,:], other_traj=neighbor[:,:], navigation_extractor=navigation_extractor) # other with respect to reference, note that we will use the recorded traj and (RECORDED) heading, not the calculated heading for normalization inside this function

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
            # return np.array([])
            return traj
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
        
        [self.get_vel_from_traj(other_traj[:,:2][ii], other_traj[:,:2][ii+1], time_difference=time_step[ii+1] - time_step[ii]) for ii in range(len(valid_states)-1)]
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

        


    def map_process(self, traj):
        '''
        Map point attributes
        self_point (x, y, h), left_boundary_point (x, y, h), right_boundary_pont (x, y, h), speed limit (float),
        self_type (int), left_boundary_type (int), right_boundary_type (int), traffic light (int), stop_point (bool), interpolating (bool), stop_sign (bool)
        '''
        vectorized_map = np.zeros(shape=(self.drivable_lanes, 300, 17))
        vectorized_crosswalks = np.zeros(shape=(self.num_crosswalks, 100, 3))
        agent_type = int(traj[-1][-1])

        # get all lane polylines
        lane_polylines = get_polylines(self.lanes)

        # get all road lines and edges polylines
        road_polylines = get_polylines(self.roads)

        # find current lanes for the agent
        ref_lane_ids, all_lane_ids = find_reference_lanes(agent_type, traj, lane_polylines)

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
        # if len(ref_lanes)>self.drivable_lanes:
        #     self.global_counter +=1
        #     print(self.global_counter)
        for i, s_lane in enumerate(ref_lanes):
            added_points = 0
            # if i > 1:
            #     break
            if i > (self.drivable_lanes-1): # DONE: make this 5 (that mean 6 drivable lanes variable)
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
            
            if added_cross_walks >= self.num_crosswalks:
                break

        return vectorized_map.astype(np.float32), vectorized_crosswalks.astype(np.float32)

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

    def normalize_data(self, ego, neighbors, map_lanes, map_crosswalks, ground_truth, viz=False):
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

        return ego, neighbors, map_lanes, map_crosswalks, ground_truth,region_dict, normalization_param
    
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

    def process_data(self, viz=True,test=False):
        navigation_extractor = futureNavigation()
        selected_scenario = True #comment
        self.save_dir_json = f"{self.save_dir[:-1]}_json/"
        os.makedirs(self.save_dir_json, exist_ok=True)

        if self.point_dir != '':
            self.build_points()

        # self.pbar = tqdm(total=len(list(self.data_files)))
        # self.pbar.set_description(f"Processing {len(list(self.data_files))} files, -{list(self.data_files)[0].split('/')[-1][-14:-9]}")
        for data_file in self.data_files:
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

                for pairs in self.sdc_ids_list:
                    sdc_ids, interesting = pairs[0], pairs[1]      
                    
                    # if interesting!=1: # comment for original preprocessing
                    #     continue
                    # process data
                    ego = self.ego_process(sdc_ids, parsed_data.tracks)

                    ego_type = parsed_data.tracks[sdc_ids[0]].object_type
                    # if ego_type!=1:  # comment for original preprocessing
                    #     continue
                    neighbor_type = parsed_data.tracks[sdc_ids[1]].object_type
                    object_type = np.array([ego_type, neighbor_type])
                    self.object_type = object_type
                    ego_index = parsed_data.tracks[sdc_ids[0]].id
                    neighbor_index = parsed_data.tracks[sdc_ids[1]].id
                    object_index = np.array([ego_index, neighbor_index])

                    neighbors, _ = self.neighbors_process(sdc_ids, parsed_data.tracks)
                    map_lanes = np.zeros(shape=(2, self.drivable_lanes, 300, 17), dtype=np.float32) # DONE: make 6 variable
                    map_crosswalks = np.zeros(shape=(2, self.num_crosswalks, 100, 3), dtype=np.float32)
                    ground_truth = self.ground_truth_process(sdc_ids, parsed_data.tracks)
                    
                    inter = 'interest' if interesting==1 else 'r'
                    filename = self.save_dir + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}.npz"
                    filename_json = self.save_dir_json + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}.json"
                    # print('#############################')
                    # print(f'###{filename}')
                    to_save_json = self.gen_instruct_caption(ego[0], ground_truth[0], ego[1], ground_truth[1], neighbors, navigation_extractor)
                    # self.in_out_map(ego[0], ground_truth[0], ego[1], ground_truth[1], neighbors)
                    # continue
                    map_lanes[0], map_crosswalks[0] = self.map_process(ego[0])
                    map_lanes[1], map_crosswalks[1] = self.map_process(ego[1])
                    if test:
                        ground_truth = np.zeros((2, self.future_len, 5))
                    else:
                        ground_truth = self.ground_truth_process(sdc_ids, parsed_data.tracks)
                    # ego, neighbors, map_lanes, map_crosswalks, ground_truth,region_dict, normalization_param = self.normalize_data(ego, neighbors, map_lanes, map_crosswalks, ground_truth, viz=viz)
                    # self.reverse_normalize_traj(ego, neighbors, ground_truth, normalization_param)
                    ego_original, neighbors_original, map_lanes_original, map_crosswalks_original, ground_truth_original = deepcopy(ego), deepcopy(neighbors), deepcopy(map_lanes), deepcopy(map_crosswalks), deepcopy(ground_truth)
                    ego, neighbors, map_lanes, map_crosswalks, ground_truth, region_dict, normalization_param = self.normalize_data(deepcopy(ego), deepcopy(neighbors), deepcopy(map_lanes), deepcopy(map_crosswalks), deepcopy(ground_truth), viz=viz)
                    # ego_normalized_back, neighbors_normalized_back, ground_truth_normalized_back = self.reverse_normalize_traj(deepcopy(ego), deepcopy(neighbors), deepcopy(ground_truth), normalization_param)
                    

                    if self.point_dir == '':
                        region_dict = {6:np.zeros((6,2))}
                    # save data
                    if not debug:
                        if test:
                            np.savez(filename, ego=np.array(ego), neighbors=np.array(neighbors), map_lanes=np.array(map_lanes), 
                            map_crosswalks=np.array(map_crosswalks),object_type=np.array(object_type),region_6=np.array(region_dict[6]),
                            object_index=np.array(object_index),current_state=np.array(self.current_xyzh[0]))
                        else:
                            np.savez(filename, ego=np.array(ego), neighbors=np.array(neighbors), map_lanes=np.array(map_lanes), 
                            map_crosswalks=np.array(map_crosswalks),object_type=np.array(object_type),region_6=np.array(region_dict[6]),
                            object_index=np.array(object_index),current_state=np.array(self.current_xyzh[0]),gt_future_states=np.array(ground_truth))

                            if len(to_save_json)>0:
                            # Open a file in write mode
                                with open(filename_json, 'w') as file:
                                    json.dump(to_save_json, file, indent=4)  # indent=4 for pretty printing
                    else:
                        ''
                        inputs = {
                            'ego_state': torch.tensor(ego[0:1]), # [1, 11 history steps,11 features]
                            'neighbors_state': torch.cat((torch.tensor(ego[1:]),torch.tensor(neighbors)), dim=0).unsqueeze(0), # [1, 21 agents (excluding the ego), 11 history steps,11 features]
                            'map_lanes': torch.tensor(map_lanes).unsqueeze(0), # [1, 2 interactive agents (the ego and the first of the neighbors_state), 6 drivable lanes to be increased, 300 samples, 17 features]
                            'map_crosswalks': torch.tensor(map_crosswalks).unsqueeze(0), # [1, 2 agents, 4 closest crosswalks, 100 samples, 3 features]
                        }
                        ego_future = torch.tensor(ground_truth[0]).unsqueeze(0)
                        # viz_multimodal(inputs, ego_future.unsqueeze(0)[...,:2], ego_future).savefig('ex.png')
                        for k,v in to_save_json.items():
                            # if 'Agent-1' in k:
                            #     k = k.replace('Agent-1', '### Ego agent (Agent-1)')
                            # if 'Agent-2' in k:
                            #     k = k.replace('Agent-2', '$$$ Interactive agent (Agent-2)')
                            print(f"{k}: {v}")
                        
                        ground_truth
                        traj = self.interpolate_missing_traj(ego[0].copy())
                        inter_traj = self.interpolate_missing_traj(ego[1].copy())
                        future_traj_ = self.interpolate_missing_traj(ground_truth[0].copy()) # correction
                        inter_future_traj_ = self.interpolate_missing_traj(ground_truth[1].copy()) # correction
                        subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int) # subsample to sample/second
                        future_traj_subsampled = future_traj_[subsample_indices]
                        inter_future_traj_subsampled = inter_future_traj_[subsample_indices]
                        other_neighbors = neighbors[2:].copy()
                        center, angle = normalization_param
                        traj[:,:5], future_traj_subsampled[:,:5], inter_traj[:,:5], inter_future_traj_subsampled[:,:5] = \
                            reverse_agent_norm(traj, center, angle),\
                            reverse_agent_norm(future_traj_subsampled, center, angle),\
                            reverse_agent_norm(inter_traj, center, angle),\
                            reverse_agent_norm(inter_future_traj_subsampled, center, angle)
                        other_neighbors = other_neighbors[other_neighbors[:,0,0]!=0]
                        for neighbor_i in range(other_neighbors.shape[0]):
                            other_neighbors[neighbor_i,...,:5] = reverse_agent_norm(other_neighbors[neighbor_i], center, angle)
                        fig = self.plt_scene((traj, future_traj_subsampled),(inter_traj, inter_future_traj_subsampled),other_neighbors[:10], center)
                        fig.savefig('ex.png', dpi=300)
                        plt.close()
                        print('sample done')
                    # break

                
                self.pbar.update(1)

            self.pbar.close()

def parallel_process(root_dir):
    print(root_dir)
    processor = DataProcess(root_dir=[root_dir], point_dir=point_path, save_dir=save_path) 
    processor.process_data(viz=debug,test=test)
    print(f'{root_dir}-done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing Interaction Predictions')
    parser.add_argument('--load_path', type=str, help='path to dataset files', default='<internal_waymo_dataset_root>/validation_interactive')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='<internal_waymo_dataset_root>/training')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='<internal_waymo_dataset_root>/sample_data')
    # parser.add_argument('--save_path', type=str, help='path to save processed data', default = '<internal_waymo_dataset_root>/validation_interactive_processes_5mar/')
    # parser.add_argument('--save_path', type=str, help='path to save processed data', default = '<internal_waymo_dataset_root>/training_interactive_processes_5mar/')
    parser.add_argument('--save_path', type=str, help='path to save processed data', default = '')
    # parser.add_argument('--save_path', type=str, help='path to save processed data', default = '')
    parser.add_argument('--point_path', type=str, help='path to load K-Means Anchors (Currently not included in the pipeline)', default='')
    parser.add_argument('--processes', type=int, help='multiprocessing process num', default=1)
    parser.add_argument('--not_debug', action="store_true", help='visualize processed data', default=False)
    # parser.add_argument('--not_debug', action="store_true", help='visualize processed data', default=True)
    parser.add_argument('--test', action="store_true", help='whether to process testing set', default=False)
    parser.add_argument('--use_multiprocessing', action="store_true", help='use multiprocessing', default=False)
    parser.add_argument('--run_num', type=int, help='', default=-1)
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')
    
    # data_files[0][-14:]
    # data_files[0][:-14]
    # data_files[0][-9:]
    # Split the list into 10 parts
    if args.run_num!=-1:
        numbers = [str(i).zfill(5) for i in range(1000)]
        split_parts = [numbers[i:i + 100] for i in range(0, len(numbers), 100)]
        data_files = [f"{data_files[0][:-14]}{i}{data_files[0][-9:]}" for i in split_parts[args.run_num]]

    save_path = args.save_path
    point_path = args.point_path
    debug = not args.not_debug
    test = args.test
    if save_path != '':
        os.makedirs(save_path, exist_ok=True)
        
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
  