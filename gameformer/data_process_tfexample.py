import torch
import glob
import sys
sys.path.append("..")
sys.path.append(".")
import argparse
from multiprocessing import Pool
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
from waymo_open_dataset.protos import scenario_pb2
from utils.data_utils import *
import pickle
import torch
import math
import uuid
import time
import glob
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
# from IPython.display import HTML
import itertools
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
from utils.data_utils import *
import matplotlib as mpl
from matplotlib.patches import Circle
from tqdm import tqdm
from utils.tfexample_utils import *
from pathlib import Path
from utils.new_utils import *
from instructions.extract_instructions import futureNavigation
from copy import deepcopy
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
tf.config.set_visible_devices([], 'GPU')

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
        self.num_crosswalks = 8
        self.global_counter = 0
        self.valid = 'valid' in save_dir
        
    def get_scene_data(self, parsed):
        
        # states = torch.zeros(128,91,9)
        states_ = np.zeros([128,91,11], dtype=np.float32)
        states = np.zeros([128,91,9], dtype=np.float32)
        time_steps_name_ = ['past', 'current', 'future']
        time_steps_ = [(0,10),(10,11),(11,91)]
        features = ['x','y','bbox_yaw','velocity_x','velocity_y','length','width','height']
        for time_steps, time_steps_name in zip(time_steps_, time_steps_name_):
            for i, feature_i in enumerate(features):
                states[:,time_steps[0]:time_steps[1],i] = parsed[f"state/{time_steps_name}/{feature_i}"].numpy()
            states[:,time_steps[0]:time_steps[1],8] = parsed[f"state/type"].numpy()[:,np.newaxis]  # Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
            

        valid_states_mask = parsed["state/current/valid"].numpy()[:,0].astype(bool)
        tracks_to_predict = parsed["state/tracks_to_predict"].numpy().astype(bool)
        
        valid_roadgraph = parsed["roadgraph_samples/valid"].numpy()[:,0].astype(bool)
        line_type = parsed["roadgraph_samples/type"].numpy()[:,0][valid_roadgraph]
        road_map = parsed["roadgraph_samples/xyz"].numpy()[:,:2][valid_roadgraph]
        heading_unit_vector = parsed["roadgraph_samples/dir"].numpy()[valid_roadgraph]
        heading_angle = np.arctan2(heading_unit_vector[:,1], heading_unit_vector[:,0])
        road_map = np.hstack((road_map, heading_angle[:, np.newaxis]))

        road_map_segments, road_map_segments_masks, road_map_types = map_tfexample_preprocess(road_map, line_type)
        type_category_list = ['center', 'boundary', 'stopsign', 'crosswalk', 'speedbump']
        road_map_types_category = road_map_types.argmax(axis=-1)

        traffic_light = np.zeros((16,4))
        for i, key in enumerate(['x','y','state','valid']):
            traffic_light[:,i] = parsed[f"traffic_light_state/current/{key}"].numpy()
       
        interesting_mask = parsed["state/objects_of_interest"].numpy().astype(bool)[:,0]
        
        is_sdc = np.floor((parsed["state/is_sdc"].numpy()+1)/2).astype(bool)

        agents_id = parsed["state/id"].numpy().astype(int)


        states_[...,:8] = states[...,:8]
        object_type = states[...,11,8]
        object_type_one_hot = np.vstack([np.eye(3)[int(i-1)] if (i>0 and i<3) else np.array([0,0,0]) for i in object_type])

        # object_type_one_hot[object_type<=0] = [0,0,0]
        # object_type_one_hot[object_type>3] = [0,0,0]
        # np.vstack([np.eye(3)[int(i)] if (i>0 and i<3) else np.array([0,0,0]) for i in object_type])
        # valid_type = list(((object_type<3) & (object_type>0)))
        # valid_type_indices = tuple([i for i in range(len(valid_type)) if valid_type[i]])
        # object_type_one_hot[valid_type_indices] = np.vstack([np.eye(3)[int(i)] for i in list(object_type[[valid_type_indices]]-1)])[:,None,:]
        
        states_[...,8:] = object_type_one_hot[:,None,:]
        
        ## to validate that it it designed correctly:
        # for i in range(len(object_type)):
        #     if object_type[i] != (states_[i, -1, 8:].argmax()+1)* int(states_[i, -1, 8:].sum()):
        #         if not (object_type[i]==-1 and (states_[i, -1, 8:].argmax()+1)* int(states_[i, -1, 8:].sum())==0):
        #             print("WRONG")
        


        return states_, valid_states_mask, tracks_to_predict, road_map_segments, road_map_segments_masks, road_map_types, road_map_types_category, traffic_light, interesting_mask, is_sdc


    def process_scene_data(self, states, valid_states_mask, sdc_ids, road_map_segments, road_map_segments_masks, road_map_types, road_map_types_category, traffic_light, navigation_extractor):
        
        tracks_to_predict = np.zeros(states.shape[0]).astype(bool)
        ego = np.zeros((2,11,11))
        ground_truth = np.zeros((2,80,5))

        for i, sdc_id in enumerate(sdc_ids):
            tracks_to_predict[sdc_id] = True
            ego[i] = states[sdc_id, :11]
            ground_truth[i] = states[sdc_id, 11:,:5]
        

        # ego = states[tracks_to_predict][:,:11]
        # ground_truth = states[tracks_to_predict][:,11:,:5]

        # if True in (np.linalg.norm((ground_truth[:,:,:2] - ego[:,-1,:2][:,np.newaxis,:]), axis=-1) > 350):
        #     valid_sample = False
        #     return None, None, None, None, None, None, valid_sample

        # Sorting based on distance to ego
        distance_to_ego = np.linalg.norm(ego[0,10,:2] - states[:,10,:2], axis=-1)
        args_sort = np.argsort(distance_to_ego)
        sorted_states = states[args_sort]
        sorted_tracks_to_predict = tracks_to_predict[args_sort]
        sorted_valid_states_mask = valid_states_mask[args_sort]

        valid_neighbors = sorted_states[~sorted_tracks_to_predict][sorted_valid_states_mask[~sorted_tracks_to_predict]]

        num_neighbors = self.num_neighbors
        neighbors = np.zeros((num_neighbors,ego.shape[1],ego.shape[-1]))
        neighbors[:min(num_neighbors,valid_neighbors.shape[0])] = valid_neighbors[:min(num_neighbors,valid_neighbors.shape[0]), :ego.shape[1], :]

        ego, ground_truth, neighbors, not_valid_future = interpolate_missing_data(ego, ground_truth, neighbors)
        
        if not_valid_future:
            valid_sample = False
            return None, None, None, None, None, None, None, None, valid_sample
        
        valid_sample = validate_movement_speed(ego, ground_truth)
        if not valid_sample:
            return None, None, None, None, None, None, None, None, valid_sample

        agent_json = gen_instruct_caption_01(history=np.vstack((deepcopy(ego),deepcopy(neighbors))),future=deepcopy(ground_truth), navigation_extractor=navigation_extractor)

        ego, neighbors, ground_truth, normalization_param = normalize_agent_data_(ego, neighbors, ground_truth)
        if len(ego)==1:
            valid_sample=False
            return None, None, None, None, None, None, None, None, valid_sample
        center, angle = normalization_param
        road_map_segments = normalize_map_data_(road_map_segments, center, angle)
        traffic_light[:,:3] = normalize_map_data_(traffic_light[:,np.newaxis,:3], center, angle)[:,0,:]

        stop_signs_indx = road_map_types_category==2

        agent_json_, map_json = gen_instruct_caption_02(history=np.vstack((deepcopy(ego),deepcopy(neighbors))),future=deepcopy(ground_truth), navigation_extractor=navigation_extractor, 
                                                         stop_signs_points=road_map_segments[stop_signs_indx][...,0,:2], traffic_light_points=traffic_light[:,:2], traffic_light_states=traffic_light[:,-1],
                                                         normalization_param=normalization_param)

        for agent_k in agent_json.keys():
            agent_json[agent_k].update(agent_json_[agent_k])
        
        if not 'Agent-1' in agent_json.keys():
            valid_sample = False
            return None, None, None, None, None, None, None, None, valid_sample
        if agent_json['Agent-1']['direction 0.1to8_cls']==-1:
            valid_sample = False
            return None, None, None, None, None, None, None, None, valid_sample

        valid_sample = True

        return ego, neighbors, ground_truth, road_map_segments, road_map_segments_masks, traffic_light, agent_json, map_json, valid_sample

    def interactive_process(self,valid_states_mask, interesting_mask, tracks_to_predict, is_sdc, states):
        
        self.sdc_ids_list = []
        list_of_valid_ids = [id_i for id_i, id_ in enumerate(valid_states_mask) if id_]
        interesting_ids = [id_i for id_i, id_ in enumerate(interesting_mask*valid_states_mask) if id_]
        tracks_list = [id_i for id_i, id_ in enumerate(tracks_to_predict*valid_states_mask) if id_] # tracks to predict
        # tracks_list = list_of_valid_ids

        sdc_ids = [id_i for id_i, id_ in enumerate(is_sdc) if id_]

        if len(interesting_ids)<2:
            interesting_ids = [-1]
        
        for ego_id in tracks_list:
            
            ego_state = states[ego_id][self.hist_len-1]
            ego_xy = ego_state[:2]

            candidate_tracks = []
            cnt = 2
            # if False:
            if len(tracks_list)==1:
                for t in tracks_list:    
                    if t!=ego_id:
                        track_states = states[t][self.hist_len-1]
                        tracks_xy = track_states[:2]
                        candidate_tracks.append((t, np.linalg.norm(tracks_xy - ego_xy)))
            else:
                for t in tracks_list:
                    if t!=ego_id:
                        if t in interesting_ids and ego_id in interesting_ids:
                            self.sdc_ids_list.append(((ego_id, t), 1))
                            cnt -= 1
                            continue
                        
                        track_states = states[t][self.hist_len-1]
                        tracks_xy = track_states[:2]
                        candidate_tracks.append((t, np.linalg.norm(tracks_xy - ego_xy)))
            
            sorted_candidate = sorted(candidate_tracks, key=lambda item: item[1])[:cnt]
            
            for can in sorted_candidate:
                self.sdc_ids_list.append(((ego_id, can[0]), 0))

    # def decompose_map_data(road_map_types_category, road_map_segments_, road_map_types):


    def process_data(self, viz=True,test=False):
        navigation_extractor = futureNavigation()
        features_description = get_features_description()
        scenario_ids = []
        # self.pbar = tqdm(total=len(list(self.data_files)))
        # self.pbar.set_description(f"Processing {len(list(self.data_files))} files, -{list(self.data_files)[0].split('/')[-1][-14:-9]}")
        for data_file in self.data_files:
            # if not "00001" in data_file.split('/')[-1]:
            #     continue
            dataset = tf.data.TFRecordDataset(data_file)
            self.pbar = tqdm(total=len(list(dataset)))
            self.pbar.set_description(f"Processing {data_file.split('/')[-1]}")

            for data_i, data in enumerate(dataset):
                parsed = tf.io.parse_single_example(data, features_description)
                scenario_id = parsed["scenario/id"].numpy()[0].decode('utf-8')
                # if scenario_id != 'f70d21a232bab7f2':
                # if scenario_id != 'ffd920fc681f7be':
                #     continue
                
                states, valid_states_mask, tracks_to_predict, road_map_segments, road_map_segments_masks, road_map_types, road_map_types_category, traffic_light, interesting_mask, is_sdc = self.get_scene_data(parsed)
                # interesting_mask
                self.interactive_process(valid_states_mask, interesting_mask, tracks_to_predict, is_sdc, states)
                for pairs_i, pairs in enumerate(self.sdc_ids_list):
                    sdc_ids, interesting = pairs[0], pairs[1] 
                    # process data
                    states_, valid_states_mask_, tracks_to_predict_, road_map_segments_, road_map_segments_masks_, road_map_types_, road_map_types_category_, traffic_light_, interesting_mask_, is_sdc_ = to_copy(states, valid_states_mask, tracks_to_predict, road_map_segments, road_map_segments_masks, road_map_types, road_map_types_category, traffic_light, interesting_mask, is_sdc)
                    ego_, neighbors_, ground_truth_, road_map_segments_, road_map_segments_masks_, traffic_light_, agent_json, map_json, valid_sample = self.process_scene_data(states_, valid_states_mask_, sdc_ids, road_map_segments_, road_map_segments_masks_, road_map_types_, road_map_types_category_, traffic_light_, navigation_extractor)
                    
                    if not valid_sample:
                        continue
                    
                    # road_map_types_category : [0:center lanes, 1:boundaries, 2: stop signs, 3: crosswalks, 4: speed bumps, ]
                    decompose = True
                    # subsample_map = True
                    # if decompose:
                    #     mask_ = (road_map_types_category==0)
                    #     map_lanes_center = road_map_segments_[mask_]
                    #     map_lanes_center_type = road_map_types[mask_][:,0]
                    #     map_lanes_center_masks = road_map_segments_masks_[mask_]
                    #     # if viz:
                    #     #     visualize_scene_numpy_wrapper(ego_, neighbors_, ground_truth_, map_lanes_center, map_lanes_center_masks, None, map_lanes_center_type, traffic_light_, np.array([0])).savefig('ex.png')
                    #     mask_ = (road_map_types_category==1)
                    #     map_lanes_boundary = road_map_segments_[mask_]
                    #     map_lanes_boundary_type = road_map_types[mask_][:,1]
                    #     map_lanes_boundary_masks = road_map_segments_masks_[mask_]

                    #     lanes_ = np.vstack((map_lanes_center, map_lanes_boundary)) # save
                    #     lanes_masks = np.vstack((map_lanes_center_masks, map_lanes_boundary_masks))
                    #     assert sum(lanes_[lanes_masks][:,:2].sum(-1)!=0) == lanes_[lanes_masks][:,:2].sum(-1).shape[0]
                    #     # continue
                    #     types0 = np.zeros((map_lanes_center_type.shape[0],2))
                    #     types0[:,0] = map_lanes_center_type
                    #     types1 = np.zeros((map_lanes_boundary_type.shape[0],2))
                    #     types1[:,0] = map_lanes_boundary_type
                    #     lanes_types = np.vstack((types0,types1)) 
                    #     lanes_categ = np.hstack((np.ones(types0.shape[0])*0, np.ones(types1.shape[0])*1)) 
                    #     lanes_types_ = np.hstack((lanes_types, lanes_categ[:,np.newaxis])) # save

                    #     # if viz:
                    #     #     visualize_scene_numpy_wrapper(ego_, neighbors_, ground_truth_, map_lanes_boundary, map_lanes_boundary_masks, None, map_lanes_boundary_type, traffic_light_, np.array([1])).savefig('ex.png')
                    #     mask_ = (road_map_types_category==2)
                    #     if sum(mask_)!=0:
                    #         stop_signs = road_map_segments_[mask_]
                    #         # stop_signs_type = road_map_types[mask_][:,2]
                    #         stop_signs_masks = road_map_segments_masks_[mask_]
                    #         max_len = max(stop_signs_masks.sum(-1))
                    #         stop_signs_ = stop_signs[:,:max_len] # save
                    #         if stop_signs_.shape[1]:
                    #             stop_signs_ = stop_signs_[:,0,:]
                    #     else:
                    #         stop_signs_ = np.array([])
                    #     # if viz:
                    #     #     visualize_scene_numpy_wrapper(ego_, neighbors_, ground_truth_, stop_signs, stop_signs_masks, None, stop_signs_type, traffic_light_, np.array([2])).savefig('ex.png')
                    #     mask_ = (road_map_types_category==3)
                    #     if sum(mask_)!=0:
                    #         crosswalks = road_map_segments_[mask_]
                    #         # crosswalks_type = road_map_types[mask_][:,3]
                    #         crosswalks_masks = road_map_segments_masks_[mask_]
                    #         max_len = max(crosswalks_masks.sum(-1))
                    #         crosswalks_ = crosswalks[:,:max_len] # save
                    #     else:
                    #         crosswalks_ = np.array([])
                    #     # if viz:
                    #     #     visualize_scene_numpy_wrapper(ego_, neighbors_, ground_truth_, crosswalks, crosswalks_masks, None, crosswalks_type, traffic_light_, np.array([3])).savefig('ex.png')
                    #     mask_ = (road_map_types_category==4)
                    #     if sum(mask_)!=0:
                    #         speed_bumps = road_map_segments_[mask_]
                    #         # speed_bumps_type = road_map_types[mask_][:,4]
                    #         speed_bumps_masks = road_map_segments_masks_[mask_]
                    #         max_len = max(speed_bumps_masks.sum(-1))
                    #         speed_bumps_ = speed_bumps[:,:max_len] # save
                    #     else:
                    #         speed_bumps_ = np.array([])

                    
                    # road_map_types_category
                    # road_map_segments_
                    # road_map_types
                    # road_map_segments_masks_

                    ## order map based on closest to ego
                    road_map_segments_ordered = np.zeros_like(road_map_segments_)
                    road_map_segments_ordered_masks = np.zeros_like(road_map_segments_masks_)
                    road_map_segments_ordered_category = np.zeros_like(road_map_types_category)
                    road_map_ordered_types = np.zeros_like(road_map_types)
                    road_idx = 0
                    for categ_i in range(5):
                        if sum(road_map_types_category==categ_i)>0:
                            sorting_idx = np.argsort(np.linalg.norm(road_map_segments_[road_map_types_category==categ_i][:,0,:2], axis=-1))
                            road_map_segments_ordered[road_idx:road_idx+len(sorting_idx)] = road_map_segments_[road_map_types_category==categ_i][sorting_idx]
                            road_map_segments_ordered_masks[road_idx:road_idx+len(sorting_idx)] = road_map_segments_masks_[road_map_types_category==categ_i][sorting_idx]
                            road_map_segments_ordered_category[road_idx:road_idx+len(sorting_idx)] = road_map_types_category[road_map_types_category==categ_i][sorting_idx]
                            road_map_ordered_types [road_idx:road_idx+len(sorting_idx)] = road_map_types[road_map_types_category==categ_i][sorting_idx]
                            road_idx+=len(sorting_idx)
                    
                    
                    for road_i in range(len(road_map_segments_ordered)):
                        max_distance_select = 288 # m, if car is moving with a speed of 130kmh, it could travel a maximum of 288m in 8 seconds
                        select_mask = np.linalg.norm(road_map_segments_ordered[road_i,:,:2], axis=-1) < max_distance_select
                        if sum(~select_mask)>0:
                            road_map_segments_ordered[road_i, ~select_mask, :] = 0
                            road_map_segments_ordered_masks[road_i, ~select_mask] = False

                    if 0 in road_map_segments_ordered_masks.sum(-1):
                        select_mask = road_map_segments_ordered_masks.sum(-1)==0
                        road_map_segments_ordered = road_map_segments_ordered[~select_mask]
                        road_map_segments_ordered_masks = road_map_segments_ordered_masks[~select_mask]
                        road_map_segments_ordered_category = road_map_segments_ordered_category[~select_mask]
                        road_map_ordered_types = road_map_ordered_types[~select_mask]

                    # road_map_types_category : [0:center lanes, 1:boundaries, 2: stop signs, 3: crosswalks, 4: speed bumps, ]
                    # select the closest set of center lanes and boundaries only
                    ## order map based on closest to ego
                    select_small_road_set = False
                    if select_small_road_set:
                        max_len_selected = 100
                        reduced_size = (sum(road_map_types_category==0)-max_len_selected) + (sum(road_map_types_category==1)-max_len_selected)
                        
                        road_map_segments_ordered_small = np.zeros_like(road_map_segments_ordered[:len(road_map_segments_ordered_category)-reduced_size])
                        road_map_segments_ordered_masks_small = np.zeros_like(road_map_segments_ordered_masks[:len(road_map_segments_ordered_category)-reduced_size])
                        road_map_segments_ordered_category_small = np.zeros_like(road_map_segments_ordered_category[:len(road_map_segments_ordered_category)-reduced_size])
                        road_map_ordered_types_small = np.zeros_like(road_map_ordered_types[:len(road_map_segments_ordered_category)-reduced_size])
                        road_idx = 0
                        for categ_i in range(5):
                            if sum(road_map_segments_ordered_category==categ_i)>max_len_selected and categ_i in [0,1]:
                                road_map_segments_ordered_small[road_idx:road_idx+max_len_selected] = road_map_segments_ordered[road_map_segments_ordered_category==categ_i][:max_len_selected]
                                road_map_segments_ordered_masks_small[road_idx:road_idx+max_len_selected] = road_map_segments_ordered_masks[road_map_segments_ordered_category==categ_i][:max_len_selected]
                                road_map_segments_ordered_category_small[road_idx:road_idx+max_len_selected] = road_map_segments_ordered_category[road_map_segments_ordered_category==categ_i][:max_len_selected]
                                road_map_ordered_types_small[road_idx:road_idx+max_len_selected] = road_map_ordered_types[road_map_segments_ordered_category==categ_i][:max_len_selected]
                                road_idx+=max_len_selected
                            else:
                                road_map_segments_ordered_small[road_idx:road_idx+len(road_map_types_category[road_map_types_category==categ_i])] = road_map_segments_[road_map_types_category==categ_i]
                                road_map_segments_ordered_masks_small[road_idx:road_idx+len(road_map_types_category[road_map_types_category==categ_i])] = road_map_segments_masks_[road_map_types_category==categ_i]
                                road_map_segments_ordered_category_small[road_idx:road_idx+len(road_map_types_category[road_map_types_category==categ_i])] = road_map_types_category[road_map_types_category==categ_i]
                                road_map_ordered_types_small[road_idx:road_idx+len(road_map_types_category[road_map_types_category==categ_i])] = road_map_types[road_map_types_category==categ_i]
                                road_idx+=len(road_map_types_category[road_map_types_category==categ_i])

                    
                    # viz=True
                    reconstruct = False
                    if reconstruct: # handling the stored data in order to visualize it
                        reconstructed_map = np.vstack((map_lanes_center, map_lanes_boundary, stop_signs, crosswalks, speed_bumps))
                        reconstructed_map_mask = np.vstack((map_lanes_center_masks, map_lanes_boundary_masks, stop_signs_masks, crosswalks_masks, speed_bumps_masks))
                        types0 = np.zeros((map_lanes_center_type.shape[0],5))
                        types0[:,0] = map_lanes_center_type
                        types1 = np.zeros((map_lanes_boundary_type.shape[0],5))
                        types1[:,1] = map_lanes_boundary_type
                        types2 = np.zeros((stop_signs_type.shape[0],5))
                        types2[:,2] = stop_signs_type
                        types3 = np.zeros((crosswalks_type.shape[0],5))
                        types3[:,3] = crosswalks_type
                        types4 = np.zeros((speed_bumps_type.shape[0],5))
                        types4[:,4] = speed_bumps_type
                        reconstructed_map_type = np.vstack((types0,types1,types2,types3,types4))
                        reconstructed_map_category = np.hstack((
                            np.ones(types0.shape[0])*0,
                            np.ones(types1.shape[0])*1,
                            np.ones(types2.shape[0])*2,
                            np.ones(types3.shape[0])*3,
                            np.ones(types4.shape[0])*4))
                        ## torch.tensor format:
                        # reconstructed_map = torch.cat((map_lanes_center, map_lanes_boundary, stop_signs, crosswalks, speed_bumps), dim=0)
                        # reconstructed_map_mask = torch.cat((map_lanes_center_masks, map_lanes_boundary_masks, stop_signs_masks, crosswalks_masks, speed_bumps_masks), dim=0)
                        # reconstructed_map_type = torch.tensor(reconstructed_map_type)
                        # reconstructed_map_category = torch.tensor(reconstructed_map_category)
                        if viz: # visualizing the reconstruction
                            visualize_scene_numpy_wrapper(ego_, neighbors_, ground_truth_, reconstructed_map, reconstructed_map_mask, reconstructed_map_category, reconstructed_map_type, traffic_light_).savefig('ex.png')
                            plt.close()
                    else:
                        if viz:
                            # visualize_scene_numpy_wrapper(deepcopy(ego_), deepcopy(neighbors_), deepcopy(ground_truth_), deepcopy(road_map_segments_), deepcopy(road_map_segments_masks_), deepcopy(road_map_types_category), deepcopy(road_map_types), deepcopy(traffic_light_)).savefig('ex.png')
                            # plt.close()
                            visualize_scene_numpy_wrapper(deepcopy(ego_), deepcopy(neighbors_), deepcopy(ground_truth_),
                                                          deepcopy(road_map_segments_ordered),
                                                          deepcopy(road_map_segments_ordered_masks),
                                                          deepcopy(road_map_segments_ordered_category),
                                                          deepcopy(road_map_ordered_types),
                                                          deepcopy(traffic_light_)).savefig('ex2.png')
                            plt.close()
                            
                            print(f"> Ego future direction: {agent_json['Agent-1']['direction 0.1to8']}")
                            print(f"> Ego future speed: {agent_json['Agent-1']['speed 0.1to8']}")
                            print(f"> Ego future acceleration: {agent_json['Agent-1']['acceleration 0.1to8']}")
                            # print(agent_json['Agent-1'])
                            print('***'*10)
                            if select_small_road_set:
                                visualize_scene_numpy_wrapper(deepcopy(ego_), deepcopy(neighbors_), deepcopy(ground_truth_), 
                                                            deepcopy(road_map_segments_ordered_small), 
                                                            deepcopy(road_map_segments_ordered_masks_small), 
                                                            deepcopy(road_map_segments_ordered_category_small), 
                                                            deepcopy(road_map_ordered_types_small), 
                                                            deepcopy(traffic_light_)).savefig('ex3.png')
                                plt.close()
                            # visualize_scene_numpy_wrapper(ego_, neighbors_, ground_truth_, 
                            #                               road_map_segments_ordered[:,::2], 
                            #                               road_map_segments_ordered_masks[:,::2], 
                            #                               road_map_segments_ordered_category, 
                            #                               road_map_ordered_types, 
                            #                               traffic_light_).savefig('ex3.png')
                            # plt.close()
                    if self.save_dir != '':
                        inter = 'interest' if interesting==1 else 'r'
                        filename = self.save_dir + f"/{scenario_id}_{sdc_ids[0]}_{sdc_ids[1]}_{inter}.npz"
                        # road_map_types_category: ['center', 'boundary', 'stopsign', 'crosswalk', 'speedbump']
                        decompose = True
                        if decompose:
                            decomposed_road = decompose_road_segments(road_segments = road_map_segments_ordered.astype(np.float32), 
                                                    road_segments_category = road_map_segments_ordered_category.astype(np.float32), 
                                                    road_segments_types = road_map_ordered_types.astype(np.float32), 
                                                    road_segments_masks = road_map_segments_ordered_masks.astype(np.float32)
                                                    )
                            center_lanes, center_lanes_masks,\
                            boundaries, boundaries_masks,\
                            stop_signs, stop_signs_masks,\
                            crosswalks, crosswalks_masks,\
                            speed_bumps, speed_bumps_masks = decomposed_road
                            np.savez(
                                filename,
                                ## state data
                                ego=ego_.astype(np.float32),
                                neighbors=neighbors_.astype(np.float32),
                                ground_truth=ground_truth_.astype(np.float32),
                                traffic_lights = traffic_light_.astype(np.float32),
                                stop_signs = stop_signs,
                                center_lanes = center_lanes,
                                center_lanes_masks = center_lanes_masks,
                                boundaries = boundaries,
                                boundaries_masks = boundaries_masks,
                                crosswalks = crosswalks,
                                crosswalks_masks = crosswalks_masks,
                                speed_bumps = speed_bumps,
                                speed_bumps_masks = speed_bumps_masks,
                            )
                            
                        else:
                            np.savez(
                                filename,
                                ## state data
                                ego=ego_.astype(np.float32),
                                neighbors=neighbors_.astype(np.float32),
                                ground_truth=ground_truth_.astype(np.float32),
                                road_segments = road_map_segments_ordered.astype(np.float32),
                                road_segments_category = road_map_segments_ordered_category.astype(np.float32),
                                road_segments_types = road_map_ordered_types.astype(np.float32),
                                road_segments_masks = road_map_segments_ordered_masks.astype(np.float32),
                                traffic_lights = traffic_light_.astype(np.float32),
                            )
                        root_dir_jsons = "/".join(filename.split('/')[:-1])
                        agent_json_filename = f"{root_dir_jsons}_agentJsons/agent_{filename.split('/')[-1][:-4]}.json"
                        with open(agent_json_filename, 'w') as file:
                            json.dump(agent_json, file, indent=4)  # indent=4 for pretty printing
                        map_json_filename = f"{root_dir_jsons}_mapJsons/map_{filename.split('/')[-1][:-4]}.json"
                        with open(map_json_filename, 'w') as file:
                            json.dump(map_json, file, indent=4)  # indent=4 for pretty printing
                        
                        # np.savez(
                        #     filename,
                        #     ## state data
                        #     ego=ego_.astype(np.float32),
                        #     neighbors=neighbors_.astype(np.float32),
                        #     ground_truth=ground_truth_.astype(np.float32),
                        #     ## map data
                        #     # lanes
                        #     map_lanes = lanes_.astype(np.float32),
                        #     map_lanes_types = lanes_types_.astype(np.int8),
                        #     # stop signs
                        #     stop_signs=stop_signs_.astype(np.float32),
                        #     # crosswalks
                        #     crosswalks=crosswalks_.astype(np.float32),
                        #     # speed bumps
                        #     speed_bumps=speed_bumps_.astype(np.float32),
                        #     # traffic lights
                        #     traffic_light_=traffic_light_.astype(np.float32),
                        # )

                        # np.savez(
                        #     filename, 
                        #     ## state data
                        #     ego=ego_, 
                        #     neighbors=neighbors_, 
                        #     ground_truth=ground_truth_, 
                        #     ## map data
                        #     # lanes
                        #     map_lanes = lanes_,
                        #     map_lanes_types = lanes_types_,
                        #     # stop signs
                        #     stop_signs=stop_signs_,
                        #     # crosswalks
                        #     crosswalks=crosswalks_,
                        #     # speed bumps
                        #     speed_bumps=speed_bumps_,
                        #     # traffic lights
                        #     traffic_light_=traffic_light_,
                        # )

                        # np.savez(
                        #     filename, 
                        #     ## state data
                        #     ego=ego_, 
                        #     neighbors=neighbors_, 
                        #     ground_truth=ground_truth_, 
                        #     ## map data
                        #     # lanes
                        #     map_lanes = lanes_,
                        #     map_lanes_types = lanes_types,
                        #     # center
                        #     map_lanes_center=map_lanes_center,
                        #     map_lanes_center_type=map_lanes_center_type,
                        #     map_lanes_center_masks=map_lanes_center_masks,
                        #     # boundaries
                        #     map_lanes_boundary=map_lanes_boundary,
                        #     map_lanes_boundary_type=map_lanes_boundary_type,
                        #     map_lanes_boundary_masks=map_lanes_boundary_masks,
                        #     # stop signs
                        #     stop_signs=stop_signs,
                        #     stop_signs_type=stop_signs_type,
                        #     stop_signs_masks=stop_signs_masks,
                        #     # crosswalks
                        #     crosswalks=crosswalks,
                        #     crosswalks_type=crosswalks_type,
                        #     crosswalks_masks=crosswalks_masks,
                        #     # speed bumps
                        #     speed_bumps=speed_bumps,
                        #     speed_bumps_type=speed_bumps_type,
                        #     speed_bumps_masks=speed_bumps_masks,
                        #     # traffic lights
                        #     traffic_light_=traffic_light_
                        # )

                self.pbar.update(1)
            self.pbar.close()

def decompose_road_segments(road_segments, road_segments_category, road_segments_types, road_segments_masks):
    center_lanes = road_segments[road_segments_category==0]
    center_lanes_type = road_segments_types[road_segments_category==0][:,0]
    center_lanes = np.concatenate((center_lanes, np.tile(center_lanes_type[:, np.newaxis], (1, center_lanes.shape[1]))[:,:,np.newaxis]), axis=-1)
    center_lanes_masks = road_segments_masks[road_segments_category==0]

    boundaries = road_segments[road_segments_category==1]
    boundaries_type = road_segments_types[road_segments_category==1][:,1]
    boundaries = np.concatenate((boundaries, np.tile(boundaries_type[:, np.newaxis], (1, boundaries.shape[1]))[:,:,np.newaxis]), axis=-1)
    boundaries_masks = road_segments_masks[road_segments_category==1]
    

    stop_signs = road_segments[road_segments_category==2]
    stop_signs_type = road_segments_types[road_segments_category==2][:,2]
    stop_signs = np.concatenate((stop_signs, np.tile(stop_signs_type[:, np.newaxis], (1, stop_signs.shape[1]))[:,:,np.newaxis]), axis=-1)
    stop_signs_masks = road_segments_masks[road_segments_category==2]
    stop_signs = stop_signs[:,0,:] # recorded data is only in stop_signs[:,0]
    

    crosswalks = road_segments[road_segments_category==3]
    crosswalks_type = road_segments_types[road_segments_category==3][:,3]
    crosswalks = np.concatenate((crosswalks, np.tile(crosswalks_type[:, np.newaxis], (1, crosswalks.shape[1]))[:,:,np.newaxis]), axis=-1)
    crosswalks_masks = road_segments_masks[road_segments_category==3]
    

    speed_bumps = road_segments[road_segments_category==4]
    speed_bumps_type = road_segments_types[road_segments_category==4][:,4]
    speed_bumps = np.concatenate((speed_bumps, np.tile(speed_bumps_type[:, np.newaxis], (1, speed_bumps.shape[1]))[:,:,np.newaxis]), axis=-1)
    speed_bumps_masks = road_segments_masks[road_segments_category==4]

    return center_lanes, center_lanes_masks,\
        boundaries, boundaries_masks,\
        stop_signs, stop_signs_masks,\
        crosswalks, crosswalks_masks,\
        speed_bumps, speed_bumps_masks


def parallel_process(root_dir, point_path, save_path):
    print(root_dir)
    processor = DataProcess(root_dir=[root_dir], point_dir=point_path, save_dir=save_path) 
    processor.process_data(viz=False,test=False)
    print(f'{root_dir}-done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing Interaction Predictions')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='<internal_waymo_dataset_root>/tfexample/validation_interactive/')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='<internal_waymo_dataset_root>/tfexample_v_1_2_1/validation_interactive/')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='<internal_waymo_dataset_root>/tfexample_v_1_2_1/training/')
    parser.add_argument('--load_path', type=str, help='path to dataset files', default='<local_dataset_root>/waymo/v_1_2/tf_example/training/')
    # parser.add_argument('--load_path', type=str, help='path to dataset files', default='<local_dataset_root>/waymo/v_1_2/tf_example/validation_interactive/')
    
    parser.add_argument('--save_path', type=str, help='path to save processed data', default = '<local_dataset_root>/gameformer/training_fullmap')
    parser.add_argument('--point_path', type=str, help='path to load K-Means Anchors (Currently not included in the pipeline)', default='', required=False)
    parser.add_argument('--processes', type=int, help='multiprocessing process num', default=1)
    parser.add_argument('--not_debug', action="store_true", help='visualize processed data', default=True)
    parser.add_argument('--test', action="store_true", help='whether to process testing set', default=False)
    parser.add_argument('--use_multiprocessing', action="store_true", help='use multiprocessing', default=False)
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')
    save_path = args.save_path
    point_path = args.point_path
    debug = not args.not_debug
    test = args.test
    if save_path != '':
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f"{save_path}_agentJsons", exist_ok=True)
        os.makedirs(f"{save_path}_mapJsons", exist_ok=True)
        

    if args.use_multiprocessing:
        with Pool(processes=args.processes) as p:
            p.starmap(parallel_process, [(root_dir, point_path, save_path) for root_dir in data_files])
    else:
        processor = DataProcess(root_dir=data_files, point_dir=point_path, save_dir=save_path)
        processor.process_data(viz=debug,test=test)
    print('Done!')
  