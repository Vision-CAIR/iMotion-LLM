from utils.data_utils import *
import matplotlib as mpl
from matplotlib.patches import Circle
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
# from IPython.display import HTML
import itertools
import tensorflow as tf
import torch.nn.functional as F

def vizualize_background(segments, masks, road_map_types_category, road_map_types, fig, categ_select=None):
    ## segments: lines info
    ## masks: masking empty points
    ## road_map_types_category: numerical idx of ['center', 'boundary', 'stopsign', 'crosswalk', 'speedbump']
    ## road_map_types: numerical idx of types within categories, 0 indicate non existing

    # plt shapes per type
    center_plt_info = [('lightgray','-', 0.01),('lightgray','solid', 0.01), ('lightgray','solid', 0.01), ('lightgray','dashed', 0.01),]
    boundry_plt_info = [('lightgray','-', 0.1), ('w','dashed', 0.4),  ('w','solid', 0.4), ('w','solid', 0.4), ('xkcd:yellow','dashed', 0.4), ('xkcd:yellow','dashed', 0.3), ('xkcd:yellow','solid', 0.3), ('xkcd:yellow','solid', 0.3), ('xkcd:yellow','dotted', 0.4),('k', '-', 1.0), ('k', '-', 1.0),]
    edge_plt_info = [('k', '-', 1.0), ('k', '-', 1.0), ('k', '-', 1.0),]
    stopsign_plt_info = [('lightgray','-', 0.1),('r', 'solid',1)]
    crosswalk_plt_info = [('white','-', 1.0),('white', '-',1.0)]
    speedbump_plt_info = [('cyan','-', 1.0),('cyan', '-',1.0)]

    plt_info = [center_plt_info, boundry_plt_info, stopsign_plt_info, crosswalk_plt_info, speedbump_plt_info]

    # init figure
    if not fig is not None:
        fig, ax = plt.subplots()
        dpi = 100
        size_inches = 800 / dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)
        plt.gca().set_facecolor('silver')
        # plt.gca().margins(0)  
        # plt.gca().axes.get_yaxis().set_visible(False)
        # plt.gca().axes.get_xaxis().set_visible(False)

    # plotting background
    for i in range(len(road_map_types)):
        if categ_select is not None:
            print(categ_select)
            categ = categ_select
            segment_type = road_map_types[i].numpy()
        else:
            categ = int(road_map_types_category[i].item())
            segment_type = int(road_map_types[i][categ].numpy())


        segment = segments[i][masks[i]]
        if segment_type != 0:
            if categ!=2:
                plt.plot(segment[:,0], segment[:,1], plt_info[categ][segment_type][0], linestyle=plt_info[categ][segment_type][1], linewidth=2, alpha=plt_info[categ][segment_type][2])
            else:
                circle = Circle((segment[:,0], segment[:,1]), 2, color='red', fill=False)
                plt.gca().add_patch(circle)
        # else:
        #     "Cannot plot undefined type"
        
    return fig, ax

def vizualize_traffic_light(locations, types, fig):
    ## locations, masks, types
    # states: [Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4, Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8]
    # plotting background
    for i in range(len(locations)):
        segment = locations[i]
        segment_type = int(types[i].numpy())
        if segment_type != 0:
            if segment_type in [1, 4, 7]:
                color = 'red'
            elif segment_type in [2, 5, 8]:
                color = 'orange'
            else:
                color = 'green'
            circle = Circle((segment[0], segment[1]), 0.8, color=color)
            plt.gca().add_patch(circle)


def rect_(x_pos, y_pos, width, height, heading, object_type, color='r', alpha=0.4, fontsize=5):
    rect_x = x_pos - width / 2
    rect_y = y_pos - height / 2
    rect = plt.Rectangle((rect_x, rect_y), width, height, linewidth=2, color=color, alpha=alpha, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_pos, y_pos), heading) + plt.gca().transData)
    plt.gca().add_patch(rect)
    plt.text(x_pos, y_pos, object_type, color='black', fontsize=fontsize, ha='center', va='center')
    
def vizualize_agent_rect(fig, past, object_type, color):
    agent_i = past
    object_type_str = ['Unset', 'Car', 'Ped.', 'Cyc']
    rect_(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type_str[int(object_type)], color=color)
    

def get_line_type(line_to_plt_type):
    if line_to_plt_type in [1,2,3]:
            center_type = line_to_plt_type-1 + 1 # You have to substract 1 if used to nn.embedding input, 3 unique values
            boundary_type = 0
            stopsign_type = 0
            crosswalk_type = 0
            speedbump_type = 0
    elif line_to_plt_type in [6,7,8,9,10,11,12,13]:
        center_type = 0
        boundary_type = line_to_plt_type-6 + 1# You have to substract 6 if used to nn.embedding input, 8 unique values
        stopsign_type = 0
        crosswalk_type = 0
        speedbump_type = 0
    elif line_to_plt_type in [15,16]:
        center_type = 0
        boundary_type = line_to_plt_type-15 + 1 + 8# You have to substract 15 if used to nn.embedding input, 2 unique values
        stopsign_type = 0
        crosswalk_type = 0
        speedbump_type = 0
    elif line_to_plt_type ==17:
        center_type = 0
        boundary_type = 0
        stopsign_type = 1
        crosswalk_type = 0
        speedbump_type = 0
    elif line_to_plt_type ==18:
        center_type = 0
        boundary_type = 0
        stopsign_type = 0
        crosswalk_type = 1
        speedbump_type = 0
    elif line_to_plt_type ==19:
        center_type = 0
        boundary_type = 0
        stopsign_type = 0
        crosswalk_type = 0
        speedbump_type = 1
    else:
        center_type = 0
        boundary_type = 0
        stopsign_type = 0
        crosswalk_type = 0
        speedbump_type = 0
    return center_type, boundary_type, stopsign_type, crosswalk_type, speedbump_type

def map_tfexample_preprocess(road_map, line_type):
    # max_segment_len = max([len(segment_) for segment_ in segments])
        # road_map_segments = torch.zeros(segments, max_segment_len, 3)
        # road_map_types = torch.zeros(segments,5)
    road_map_segments_=[]
    road_map_types_=[]

    line_type_ = [{"type": line_type[0].item(), "start": 0}]
    for k in range(1, line_type.shape[0]):
        next_line_type = line_type[k].item()
        if line_type_[len(line_type_)-1]["type"] != next_line_type:
            line_type_[len(line_type_)-1]["end"] = k
            if k<(line_type.shape[0]-1):
                line_type_.append({"type": next_line_type, "start": k})
        if k == (line_type.shape[0]-1):
            line_type_[len(line_type_)-1]["end"] = k+1

    for line_type__ in line_type_:
        line_to_plt = road_map[line_type__["start"]:line_type__["end"]]
        line_to_plt_type = line_type__["type"]
        
        # Step 1: Calculate distances and insert NaNs for large gaps
        line_to_plt_diff = np.sqrt(np.sum(np.diff(line_to_plt, axis=0)**2, axis=1))
        large_gaps = np.where(line_to_plt_diff > 3.5)[0] + 1 # Indexes where to insert NaNs, shifted by one to account for diff's offset
        
        if len(large_gaps)>0:
            # Determine segments to plot or store, accounting for large gaps
            segments = []
            start_idx = 0
            for gap_idx in large_gaps:
                temp_valid_mask = np.sum(line_to_plt[start_idx:gap_idx], axis=-1) != 0
                segments.append(line_to_plt[start_idx:gap_idx][temp_valid_mask])  # Add segment up to the gap
                start_idx = gap_idx  # Update start index for next segment
            segments.append(line_to_plt[start_idx:])  # Add the last segment after the final gap
        else:
            temp_valid_mask = np.sum(line_to_plt, axis=-1) != 0
            segments = [line_to_plt[temp_valid_mask]]

        center_type, boundary_type, stopsign_type, crosswalk_type, speedbump_type = get_line_type(line_to_plt_type)

        for i_seg, segment in enumerate(segments):
            road_map_segments_.append(segment)
            road_map_types_.append([center_type, boundary_type, stopsign_type, crosswalk_type, speedbump_type])
            # road_map_segments[i_seg, :len(segment)] = segment
            # road_map_types[i_seg] = [center_type, boundary_type, stopsign_type, crosswalk_type, speedbump_type]

    max_segment_len = max([len(segment_) for segment_ in road_map_segments_])
    road_map_segments = np.zeros((len(road_map_segments_), max_segment_len, 3))
    road_map_segments_masks = np.zeros((len(road_map_segments_), max_segment_len)).astype(bool)
    # road_map_segments = torch.zeros(len(road_map_segments_), max_segment_len, 3)
    # road_map_segments_masks = torch.zeros(len(road_map_segments_), max_segment_len).bool()
    for i_seg, segment_ in enumerate(road_map_segments_):
        road_map_segments[i_seg,:len(segment_)] = segment_
        road_map_segments_masks[i_seg,:len(segment_)] = True
    road_map_types = np.array(road_map_types_)

    return road_map_segments, road_map_segments_masks, road_map_types
        
def validate_movement_speed(ego, ground_truth):
    for i in range(len(ego)):
        speeds_01 = np.linalg.norm(abs_distance_to_velocity(ego[i,:,:2][ego[i,:,0]!=0]),axis=-1)*10*3.6
        speeds_02 = np.linalg.norm(abs_distance_to_velocity(ground_truth[i,:,:2][ground_truth[i,:,0]!=0]),axis=-1)*10*3.6
        max_speed = max(max(speeds_01), max(speeds_02))
        if max_speed>130: # 130 kmh ~= 80 mph
            return False
    return True

def normalize_agent_data_(ego, neighbors, ground_truth):
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
    
    normalization_param = [center, angle]
    return ego, neighbors, ground_truth, normalization_param


def normalize_map_data_(map_lanes, center, angle):
# normalize map points # if multiagent maps
# for i in range(map_lanes.shape[0]):
#     lanes = map_lanes[i]
#     crosswalks = map_crosswalks[i]

    for j in range(map_lanes.shape[0]):
        # lane = map_lanes[j][map_lanes_masks[j]]
        lane = map_lanes[j]
        if lane[0][0] != 0:
            lane[lane[:,0]!=0, :3] = map_norm(lane[lane[:,0]!=0], center, angle)
    return map_lanes


def vizualize_traj_arrow(traj, fig, ax, color, linestyle, add_arrow, alpha=0.5): # single agent, single modality
    plt.plot(traj[:,0], traj[:,1], linestyle=linestyle, color=color, alpha=alpha) 
    if add_arrow:
        ax.annotate('', xy=traj[-1], xytext=traj[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, linestyle='solid'))        


# def get_features_description_v1():
#     # If you use a custom conversion from Scenario to tf.Example, set the correct
#     # number of map samples here.
#     num_map_samples = 30000

#     # Example field definition
#     roadgraph_features = {
#         'roadgraph_samples/dir': tf.io.FixedLenFeature(
#             [num_map_samples, 3], tf.float32, default_value=None
#         ),
#         'roadgraph_samples/id': tf.io.FixedLenFeature(
#             [num_map_samples, 1], tf.int64, default_value=None
#         ),
#         'roadgraph_samples/type': tf.io.FixedLenFeature(
#             [num_map_samples, 1], tf.int64, default_value=None
#         ),
#         'roadgraph_samples/valid': tf.io.FixedLenFeature(
#             [num_map_samples, 1], tf.int64, default_value=None
#         ),
#         'roadgraph_samples/xyz': tf.io.FixedLenFeature(
#             [num_map_samples, 3], tf.float32, default_value=None
#         ),
#     }
#     # Features of other agents.
#     state_features = {
#         'state/id':
#             tf.io.FixedLenFeature([128], tf.float32, default_value=None),
#         'state/type':
#             tf.io.FixedLenFeature([128], tf.float32, default_value=None),
#         'state/is_sdc':
#             tf.io.FixedLenFeature([128], tf.int64, default_value=None),
#         'state/tracks_to_predict':
#             tf.io.FixedLenFeature([128], tf.int64, default_value=None),
#         'state/current/bbox_yaw':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/current/height':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/current/length':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/current/timestamp_micros':
#             tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
#         'state/current/valid':
#             tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
#         'state/current/vel_yaw':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/current/velocity_x':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/current/velocity_y':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/current/width':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/current/x':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/current/y':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/current/z':
#             tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
#         'state/future/bbox_yaw':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/future/height':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/future/length':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/future/timestamp_micros':
#             tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
#         'state/future/valid':
#             tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
#         'state/future/vel_yaw':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/future/velocity_x':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/future/velocity_y':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/future/width':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/future/x':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/future/y':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/future/z':
#             tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
#         'state/past/bbox_yaw':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#         'state/past/height':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#         'state/past/length':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#         'state/past/timestamp_micros':
#             tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
#         'state/past/valid':
#             tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
#         'state/past/vel_yaw':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#         'state/past/velocity_x':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#         'state/past/velocity_y':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#         'state/past/width':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#         'state/past/x':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#         'state/past/y':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#         'state/past/z':
#             tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
#     }

#     traffic_light_features = {
#         'traffic_light_state/current/state':
#             tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
#         'traffic_light_state/current/valid':
#             tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
#         'traffic_light_state/current/x':
#             tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
#         'traffic_light_state/current/y':
#             tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
#         'traffic_light_state/current/z':
#             tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
#         'traffic_light_state/past/state':
#             tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
#         'traffic_light_state/past/valid':
#             tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
#         'traffic_light_state/past/x':
#             tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
#         'traffic_light_state/past/y':
#             tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
#         'traffic_light_state/past/z':
#             tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
#     }

#     features_description = {}
#     features_description.update(roadgraph_features)
#     features_description.update(state_features)
#     features_description.update(traffic_light_features)
#     return features_description

def get_features_description():
    # If you use a custom conversion from Scenario to tf.Example, set the correct
    # number of map samples here.
    num_map_samples = 30000

    # Example field definition
    roadgraph_features = {
        'roadgraph_samples/dir': tf.io.FixedLenFeature(
            [num_map_samples, 3], tf.float32, default_value=None
        ),
        'roadgraph_samples/id': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None
        ),
        'roadgraph_samples/type': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None
        ),
        'roadgraph_samples/valid': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None
        ),
        'roadgraph_samples/xyz': tf.io.FixedLenFeature(
            [num_map_samples, 3], tf.float32, default_value=None
        ),
    }
    # Features of other agents.
    state_features = {
        'state/id':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/type':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/is_sdc':
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        'state/tracks_to_predict':
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        'state/current/bbox_yaw':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/height':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/length':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/timestamp_micros':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'state/current/valid':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'state/current/vel_yaw':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/velocity_x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/velocity_y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/width':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/z':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/future/bbox_yaw':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/height':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/length':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/timestamp_micros':
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        'state/future/valid':
            tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
        'state/future/vel_yaw':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/velocity_x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/velocity_y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/width':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/z':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/past/bbox_yaw':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/height':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/length':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/timestamp_micros':
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        'state/past/valid':
            tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
        'state/past/vel_yaw':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/velocity_x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/velocity_y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/width':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/z':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/objects_of_interest':
            tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
        'scenario/id':
            tf.io.FixedLenFeature([1], tf.string, default_value=None),
    }

    traffic_light_features = {
        'traffic_light_state/current/state':
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        'traffic_light_state/current/valid':
            tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
        'traffic_light_state/current/x':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/current/y':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/current/z':
            tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
        'traffic_light_state/past/state':
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        'traffic_light_state/past/valid':
            tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
        'traffic_light_state/past/x':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        'traffic_light_state/past/y':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
        'traffic_light_state/past/z':
            tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    }

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(state_features)
    features_description.update(traffic_light_features)
    return features_description



def visualize_scene(ego, neighbors, ground_truth, 
road_map_segments, road_map_segments_masks, road_map_types_category, road_map_types, traffic_light, categ_select=None, fig=None):
    fig, ax = vizualize_background(road_map_segments, road_map_segments_masks, road_map_types_category, road_map_types, fig, categ_select)
    traffic_light_masks = traffic_light[:, -1].bool()
    vizualize_traffic_light(traffic_light[traffic_light_masks], traffic_light[:,3], fig)

    vizualize_agent_rect(fig, ego[0], ego[0,0,8], color='purple')
    vizualize_agent_rect(fig, ego[1], ego[1,0,8], color='g')
    for i in range(neighbors.shape[0]):
        if neighbors[i][-1,0] != 0:
            if neighbors[i,0,8]==2:
                vizualize_agent_rect(fig, neighbors[i], neighbors[i,0,8], color='pink')
            else:
                vizualize_agent_rect(fig, neighbors[i], neighbors[i,0,8], color='k')
    
    colors = ['b', 'g', 'y', 'purple', 'orange', 'cyan']
    
    # for i in range(ego_multimodal.shape[1]):
    #     add_arrow = (ego_multimodal[batch_sample,i,-1]-ego_multimodal[batch_sample,i,0]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
    #     fig = vizualize_traj_arrow(ego_multimodal[batch_sample,i].cpu(), fig, ax, colors[i], 'solid', add_arrow)
    
    # GT
    object_type = int(ego[0,-1,8])
    add_arrow = (torch.tensor(ground_truth[0,-1,:2]-ground_truth[0,0,:2])).norm().item() > 2 # if more than 2 meter travel, draw an arrow
    vizualize_traj_arrow(ground_truth[0,...,:2], fig, ax, 'purple', '--', add_arrow, 0.9)
    vizualize_traj_arrow(ground_truth[1,...,:2], fig, ax, 'g', '--', add_arrow, 0.9)
    
    # plt.xlim(-100,200)
    # plt.ylim(-100,100)

    return fig
    

    # plt.show()
    # time.sleep(1)
    # plt.close()

def visualize_scene_numpy_wrapper(*args):
    """
    Wrapper function that converts all arguments to tensors before calling visualize_scene. If your data is in torch.tensor no need to use this
    """
    tensor_args = to_tensor(*args)
    return visualize_scene(*tensor_args)

def to_tensor(*args):
    """
    Convert all input arguments to torch.tensor.
    """
    return [torch.tensor(arg) if (not isinstance(arg, torch.Tensor) and arg is not None) else arg for arg in args]

def to_copy(*args):
    """
    Convert all input arguments to copied version, to avoid manipulating the original version.
    """
    return [arg.copy() for arg in args]
