# Code based on GameFormer official repo, only focused on Objects of Interest (ooi) only.
# from Visualization.Visualize import *
# import io
import os
import math
import matplotlib as mpl
import matplotlib.pyplot as plt # plotting library
import numpy as np 
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import glob
# import networkx as nx
from tqdm import tqdm
import pickle
import json
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import re
# from pairing import pair, depair

def expanded_stack(input, expanded_shape):
    return torch.stack([input.clone() for _ in range(expanded_shape)])

class DrivingData(Dataset):
    def __init__(self, data_dir):
        self.data_list = glob.glob(data_dir)
        self.rand_agents_order = False 

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego'][0]
        neighbor = np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0)

        map_lanes = data['map_lanes'][:, :, :]
        map_crosswalks = data['map_crosswalks'][:, :, :]
        ego_future_states = data['gt_future_states'][0]
        neighbor_future_states = data['gt_future_states'][1]
        object_type = data['object_type']

        filename = self.data_list[idx].split('/')[-1].split('.npz')[0]
        return ego, neighbor, map_lanes, map_crosswalks, ego_future_states, neighbor_future_states, object_type, filename

class TrajectoryDataset(Dataset):
# class WaymoDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir):
        # super(TrajectoryDataset, self).__init__()
        self.data_list = glob.glob(data_dir)
        self.seq_len = 19
        self.obs_len = 3
        self.pred_len = 16

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle = True)
        traj = torch.tensor(data['traj'])
        rel = torch.tensor(data['rel'])
        disc_rel = torch.tensor(data['disc_rel'])
        disc_traj = torch.tensor(data['disc_traj'])
        anchors = torch.tensor(data['anchors'])
        object_type = torch.tensor(data['object_type'])

        return traj, rel, disc_rel, disc_traj, anchors, object_type
    
class TrajectoryDataset_old(Dataset):
# class WaymoDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir):
        # super(TrajectoryDataset, self).__init__()
        self.data_list = glob.glob(data_dir)
        self.seq_len = 19
        self.obs_len = 3
        self.pred_len = 16
        self.load_all = True

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx], allow_pickle = True)
        traj_obs = torch.tensor(data['ego'][:,:,:2])
        traj_pred = torch.tensor(data['gt_future_states'][:,:,:2])
        rel_obs = torch.tensor(data['rel_obs'])
        rel_pred = torch.tensor(data['rel_pred'])
        rel_obs_1d = torch.tensor(data['rel_obs_1d'])
        rel_pred_1d = torch.tensor(data['rel_pred_1d'])
        
        if self.load_all:
            object_type = torch.tensor(data['object_type'])
            bin_edges = torch.tensor(data['bin_edges'])
            continous_grid = torch.tensor(data['continous_grid'])
            euc_mat = torch.tensor(data['euc_mat'])
            return traj_obs, traj_pred, rel_obs, rel_pred, rel_obs_1d, rel_pred_1d, object_type, bin_edges, continous_grid, euc_mat

        return traj_obs, traj_pred, rel_obs, rel_pred, rel_obs_1d, rel_pred_1d

def viz(traj_obs, traj_pred):
    for i in range(traj_obs.shape[0]):
        plt.plot(traj_obs[i, :, 0], traj_obs[i, :, 1], 'g', linewidth=1, zorder=3)
        plt.plot(traj_pred[i, :, 0], traj_pred[i, :, 1], 'r', linewidth=1, zorder=3)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def traj2accel(traj):
    rel = torch.zeros_like(traj)
    accel = torch.zeros_like(traj)
    rel[...,1:,:] = traj[..., 1:, :]-traj[..., :-1, :]
    accel[...,2:,:] = rel[..., 2:, :]-rel[..., 1:-1, :]
    return accel, rel, traj[...,0,:], rel[...,1,:]

def accel2traj(accel, traj0, rel0):
# traj0 is an initial x,y points for all trajectories
    # if len(traj0.shape)==2:
    #     traj0 = traj0.unsqueeze(1)
    # if len(rel0.shape)==2:
    #     rel0 = rel0.unsqueeze(1)
    rel = torch.zeros(accel.shape, dtype=accel.dtype, device=accel.device)
    traj = torch.zeros(accel.shape, dtype=accel.dtype, device=accel.device)
    temporal_dim=-2
    if len(traj0.shape)<len(accel.shape):
        traj0 = traj0.unsqueeze(temporal_dim)
    if len(rel0.shape)<len(accel.shape):
        rel0 = rel0.unsqueeze(temporal_dim)
    for i in range(traj.shape[temporal_dim]): #Temporal length
        rel[..., i+1:i+2, :] = rel0 + torch.sum(accel[..., :i+2, :], dim=temporal_dim, keepdim=True)
        traj[..., i:i+1, :] = traj0 + torch.sum(rel[..., :i+1, :], dim=temporal_dim, keepdim=True)
    return traj, rel

def traj2rel(traj):
    rel = torch.zeros_like(traj)
    rel[...,1:,:] = traj[..., 1:, :]-traj[..., :-1, :]
    return rel, traj[...,0,:]

def rel2traj(rel, traj0):
# traj0 is an initial x,y points for all trajectories
    temporal_dim=-2
    if len(traj0.shape)<len(rel.shape):
        traj0 = traj0.unsqueeze(temporal_dim)
    traj = torch.zeros(rel.shape, dtype=rel.dtype, device=rel.device)
    for i in range(traj.shape[temporal_dim]): #Temporal length
        traj[..., i:i+1, :] = traj0 + torch.sum(rel[..., :i+1, :], dim=temporal_dim, keepdim=True)
    return traj

def get_euculedean_int_matrix(n, m):
    # Compute Euclidean distances and their corresponding indices
    distances = []
    for i in range(n):
        for j in range(m):
            distance = np.sqrt(i**2 + j**2)
            distances.append((distance, (i, j)))

    # Sort distances
    distances.sort(key=lambda x: x[0])

    # Assign values based on sorted distances
    matrix = np.zeros((n, m), dtype=int)
    num = 0
    for _, (i, j) in distances:
        matrix[i, j] = num
        num += 1

    return matrix

def get_discrete_from_grid_1d(grid_1d, euc_mat):
    grid_2d_where = torch.where(euc_mat == grid_1d.reshape(-1).unsqueeze(-1).unsqueeze(-1)) # where is indexed by [1] and [2] for the first and second dimension
    return torch.cat((grid_2d_where[1].reshape(grid_1d.shape).unsqueeze(-1),
    grid_2d_where[2].reshape(grid_1d.shape).unsqueeze(-1)),dim=-1)
# def get_continous_from_discrete(discrete, cont_grid):
    

def custom_bucketize(input, boundaries):
    # Calculate the absolute differences between input and boundaries
    abs_diff = torch.abs(input.unsqueeze(-1) - boundaries)
    
    # Find the index of the minimum difference for each input
    bin_indices = torch.argmin(abs_diff, dim=-1)
    
    return bin_indices

def custom_bucketize_2d(input, boundaries):
    # Calculate the absolute differences between input and boundaries
    bin_indices = torch.zeros_like(input)
    abs_diff = torch.abs(input[...,0].unsqueeze(-1) - boundaries[...,0])
    bin_indices[...,0] = torch.argmin(abs_diff, dim=-1)
    abs_diff = torch.abs(input[...,1].unsqueeze(-1) - boundaries[...,1])
    bin_indices[...,1] = torch.argmin(abs_diff, dim=-1)
    return bin_indices


def get_grid_1D(traj, grid_size=None, min_val=None, max_val=None, do_plt=False, bin_edges=None, continous_grid=None, euc_mat=None):
    if bin_edges is not None and (not continous_grid is not None):
        grid_size=len(bin_edges)
        x_indices = custom_bucketize(traj[..., 0].contiguous(), bin_edges.contiguous())
        y_indices = custom_bucketize(traj[..., 1].contiguous(), bin_edges.contiguous())
        grid = torch.stack((x_indices, y_indices), dim=-1)
        if not euc_mat is not None:
            euc_mat = torch.tensor(get_euculedean_int_matrix(grid_size+1,grid_size+1))
        grid_1d = euc_mat[grid[...,0],grid[...,1]]
        return grid, grid_1d, bin_edges, euc_mat
        
    # if bin_edges is not None:
    #     'grid anchors are alreade calculated'
    if not bin_edges is not None:
        # DONE: Center around ZERO (0.00)
        do_assert=True if min_val is None and max_val is None else False
        # Define the minimum and maximum values to be used to define the grid
        if min_val is None:
            x_min, y_min = traj[:, :, 0, :].min(), traj[:, :, 1, :].min()
            min_val = min(x_min,y_min)
        if max_val is None:
            max_val = max(x_max,y_max)
            x_max, y_max = traj[:, :, 0, :].max(), traj[:, :, 1, :].max()
        # Define the edges of the bins for each dimension
        num_bins = grid_size # -1 because there will be extra bin added for zero
        # num_bins = 19
        bin_edges_temp = torch.linspace(min_val, max_val, num_bins)
        # bin_edges_temp = torch.linspace(min_val, max_val, num_bins, dtype=torch.half)
        # shifting to have zero in the grid
        closest_to_zero_idx = torch.argmin(bin_edges_temp.abs())
        if bin_edges_temp[closest_to_zero_idx] != 0:
            assert bin_edges_temp[closest_to_zero_idx] != 0
            shifted_bin_edges_temp = bin_edges_temp - bin_edges_temp[closest_to_zero_idx]
            delta = bin_edges_temp[1]-bin_edges_temp[0]
            if bin_edges_temp[closest_to_zero_idx]>0:
                # print("1")
                bin_edges_temp = torch.cat((shifted_bin_edges_temp, torch.tensor([shifted_bin_edges_temp[-1]+delta])))
            else:
                # print("2")
                bin_edges_temp = torch.cat((torch.tensor([shifted_bin_edges_temp[0]-delta]), shifted_bin_edges_temp))
        assert bin_edges_temp[0]<=min_val and bin_edges_temp[-1]>=max_val 

        if bin_edges_temp[-1]==max_val and bin_edges_temp[0]==min_val:
            bin_edges = bin_edges_temp
            # bin_edges[0] = bin_edges[0]-torch.tensor(0.0001, dtype=torch.float16)
            # bin_edges[-1] = bin_edges[-1]+0.0001
        elif bin_edges_temp[-1]<=max_val:
            bin_edges = torch.zeros((len(bin_edges_temp)+1), dtype=bin_edges_temp.dtype)
            # bin_edges = torch.zeros((len(bin_edges_temp)+1), dtype=min_val.dtype)
            bin_edges[:len(bin_edges_temp)] = bin_edges_temp
            bin_edges[-1] = delta + max_val
            bin_edges = bin_edges-0.0000001 # for error correction
        else:
            bin_edges = bin_edges_temp

        continous_grid = bin_edges.contiguous()
    # else:
    #     'grid anchors are alreade calculated'
    x_indices = custom_bucketize(traj[:, :, 0].contiguous(), bin_edges.contiguous())
    y_indices = custom_bucketize(traj[:, :, 1].contiguous(), bin_edges.contiguous())
    # x_indices = torch.bucketize(traj[:, :, 0].contiguous(), bin_edges.contiguous())
    # y_indices = torch.bucketize(traj[:, :, 1].contiguous(), bin_edges.contiguous())
    grid = torch.stack((x_indices, y_indices), dim=2)
    if not euc_mat is not None:
        euc_mat = torch.tensor(get_euculedean_int_matrix(grid_size+1,grid_size+1))
    
    grid_1d = euc_mat[grid[:,:,0],grid[:,:,1]]

    
    # euc_mat_str = np.array2string(euc_mat, separator=', ')
    # np.fromstring(euc_mat_str.replace('[', '').replace(']', ''), sep=', ').reshape((13,13))
    return grid, grid_1d, bin_edges, continous_grid, euc_mat

def traj_2_grid_1D(traj, grid_size, min_val=None, max_val=None, do_plt=False, bin_edges=None, continous_grid=None, euc_mat=None):
    # if bin_edges is not None:
    #     'grid anchors are alreade calculated'
    if not bin_edges is not None:
        # DONE: Center around ZERO (0.00)
        do_assert=True if min_val is None and max_val is None else False
        # Define the minimum and maximum values to be used to define the grid
        if min_val is None:
            x_min, y_min = traj[:, :, 0, :].min(), traj[:, :, 1, :].min()
            min_val = min(x_min,y_min)
        if max_val is None:
            max_val = max(x_max,y_max)
            x_max, y_max = traj[:, :, 0, :].max(), traj[:, :, 1, :].max()
        # Define the edges of the bins for each dimension
        num_bins = grid_size # -1 because there will be extra bin added for zero
        # num_bins = 19
        bin_edges_temp = torch.linspace(min_val, max_val, num_bins)
        # bin_edges_temp = torch.linspace(min_val, max_val, num_bins, dtype=torch.half)
        # shifting to have zero in the grid
        closest_to_zero_idx = torch.argmin(bin_edges_temp.abs())
        if bin_edges_temp[closest_to_zero_idx] != 0:
            assert bin_edges_temp[closest_to_zero_idx] != 0
            shifted_bin_edges_temp = bin_edges_temp - bin_edges_temp[closest_to_zero_idx]
            delta = bin_edges_temp[1]-bin_edges_temp[0]
            if bin_edges_temp[closest_to_zero_idx]>0:
                # print("1")
                bin_edges_temp = torch.cat((shifted_bin_edges_temp, torch.tensor([shifted_bin_edges_temp[-1]+delta])))
            else:
                # print("2")
                bin_edges_temp = torch.cat((torch.tensor([shifted_bin_edges_temp[0]-delta]), shifted_bin_edges_temp))
        assert bin_edges_temp[0]<=min_val and bin_edges_temp[-1]>=max_val 

        if bin_edges_temp[-1]==max_val and bin_edges_temp[0]==min_val:
            bin_edges = bin_edges_temp
            # bin_edges[0] = bin_edges[0]-torch.tensor(0.0001, dtype=torch.float16)
            # bin_edges[-1] = bin_edges[-1]+0.0001
        elif bin_edges_temp[-1]<=max_val:
            bin_edges = torch.zeros((len(bin_edges_temp)+1), dtype=bin_edges_temp.dtype)
            # bin_edges = torch.zeros((len(bin_edges_temp)+1), dtype=min_val.dtype)
            bin_edges[:len(bin_edges_temp)] = bin_edges_temp
            bin_edges[-1] = delta + max_val
            bin_edges = bin_edges-0.0000001 # for error correction
        else:
            bin_edges = bin_edges_temp

        continous_grid = bin_edges.contiguous()
    # else:
    #     'grid anchors are alreade calculated'
    x_indices = custom_bucketize(traj[:, :, 0].contiguous(), bin_edges.contiguous())
    y_indices = custom_bucketize(traj[:, :, 1].contiguous(), bin_edges.contiguous())
    # x_indices = torch.bucketize(traj[:, :, 0].contiguous(), bin_edges.contiguous())
    # y_indices = torch.bucketize(traj[:, :, 1].contiguous(), bin_edges.contiguous())
    grid = torch.stack((x_indices, y_indices), dim=2)
    if not euc_mat is not None:
        euc_mat = torch.tensor(get_euculedean_int_matrix(grid_size+1,grid_size+1))
    
    grid_1d = euc_mat[grid[:,:,0],grid[:,:,1]]

    
    # euc_mat_str = np.array2string(euc_mat, separator=', ')
    # np.fromstring(euc_mat_str.replace('[', '').replace(']', ''), sep=', ').reshape((13,13))
    return grid, grid_1d, bin_edges, continous_grid, euc_mat
    
def ade_fde(pred_traj, gt_traj):
    ade = torch.norm(pred_traj - gt_traj, dim=-1)
    fde = torch.norm(pred_traj[...,-1,:] - gt_traj[...,-1,:], dim=-1)
    return ade, fde

def get_sample(samples, idx):
    return {list(samples.items())[sample_key][0]:list(samples.items())[sample_key][1][idx:idx+1] for sample_key in range(len(samples))}
        
def number_to_alpha(num):
    if num <= 26:
        return chr(64 + num)
    else:
        quotient, remainder = divmod(num - 1, 26)
        return chr(64 + quotient) + chr(65 + remainder)
    

def find_nearest(array, value):
    idx = abs(array-value).argmin()
    return idx

def linspace_anchors(values_range, N):
    anchors = torch.linspace(values_range[0], values_range[-1], int(N/2))
    anchors = torch.cat((-anchors[1:].flip(0), anchors))
    return anchors

def select_anchors_cdf(values_range, cdf, N):
    # N anchors to be selected in a cdf such that the CDF value different between two anchors is fixed, values_range is the map from cdf indexes to true values
    anchors_cdf = torch.linspace(0, max(cdf), N)
    anchors = []
    for anchor_cdf_i in anchors_cdf:
        anchors.extend([find_nearest(cdf, anchor_cdf_i.item())])
    return [values_range[anchor] for anchor in anchors]

def get_signed_grid(traj, bin_edges):
    x_indices = custom_bucketize(traj[..., 0].contiguous(), bin_edges.contiguous())
    y_indices = custom_bucketize(traj[..., 1].contiguous(), bin_edges.contiguous())
    x_indices = custom_bucketize(traj[..., 0].contiguous(), bin_edges.contiguous())
    y_indices = custom_bucketize(traj[..., 1].contiguous(), bin_edges.contiguous())
    grid = torch.stack((x_indices, y_indices), dim=-1)
    grid = grid-int((len(bin_edges)-1)/2)
    return grid


def get_sign(num):
    if num>=0:
        return '+'
    else:
        return ''

def parse_agent_data(traj, rel):
    return [f'[Step {t}, Position ({x},{y}), Move <{get_sign(x_rel)}{x_rel},{get_sign(y_rel)}{y_rel}>]' for t, x, y, x_rel, y_rel in zip(range(0,traj.shape[-2]), traj[:,0], traj[:,1], rel[:,0], rel[:,1])]

def get_waymo_parsed(disc_traj, disc_rel, steps):
    shifted_disc_rel = torch.cat((disc_rel[...,1:,:], disc_rel[...,:1,:]), dim=-2)
    agents = [f"<Agent {number_to_alpha(int(target_agent_i)+1)}>" for target_agent_i in range(disc_traj.shape[-3])]
    agents_str = [""]*len(agents)
    for i in range(len(agents)):
        agent_i = agents[i]
        agents_str[i] = '<p>'+ agent_i+': {'+ ', '.join(parse_agent_data(disc_traj[i,:steps], shifted_disc_rel[i,:steps]))+'}</p>'
    return ', '.join(agents_str)


def get_waymo_parsed_batch(disc_traj, disc_rel, steps):
    return [get_waymo_parsed(disc_traj[batch_i], disc_rel[batch_i], steps) for batch_i in range(disc_traj.shape[0])]


def parse_agent_data_short(traj, rel):
    return [f'<{t},{x},{y},{get_sign(x_rel)}{x_rel},{get_sign(y_rel)}{y_rel}>' for t, x, y, x_rel, y_rel in zip(range(0,traj.shape[-2]), traj[:,0], traj[:,1], rel[:,0], rel[:,1])]

def get_waymo_parsed_short(disc_traj, disc_rel, steps):
    shifted_disc_rel = torch.cat((disc_rel[...,1:,:], disc_rel[...,:1,:]), dim=-2)
    agents = [f"<Agent {number_to_alpha(int(target_agent_i)+1)}>" for target_agent_i in range(disc_traj.shape[-3])]
    agents_str = [""]*len(agents)
    for i in range(len(agents)):
        agent_i = agents[i]
        agents_str[i] = '<p>'+ agent_i+'{'+ ''.join(parse_agent_data_short(disc_traj[i,:steps], shifted_disc_rel[i,:steps]))+'}</p>'
    return ''.join(agents_str)


def get_waymo_parsed_short2_batch(disc_rel, steps):
    return [get_waymo_parsed_short2(disc_rel[batch_i], steps) for batch_i in range(disc_rel.shape[0])]

# def get_waymo_parsed_short2(disc_rel, steps):
#     agents = [f"<Agent {number_to_alpha(int(target_agent_i)+1)}>" for target_agent_i in range(disc_rel.shape[0])]
#     agents_str = [""]*len(agents)
#     for i in range(len(agents)):
#         agent_i = agents[i]
#         # agents_str[i] = '<p>'+ agent_i+'{'+ ''.join([f'<{disc_rel_i}>' for disc_rel_i in disc_rel[i,:steps]])+'}</p>'
#         agents_str[i] = agent_i+' {'+ ','.join([f'{disc_rel_i}' for disc_rel_i in disc_rel[i,:steps]])+'}'
#     return ', '.join(agents_str)

def get_waymo_parsed_short2(disc_rel, steps):
    agents = [f"<Agent {number_to_alpha(int(target_agent_i)+1)}>" for target_agent_i in range(disc_rel.shape[0])]
    end_agents = [f"</Agent {number_to_alpha(int(target_agent_i)+1)}>" for target_agent_i in range(disc_rel.shape[0])]
    agents_str = [""]*len(agents)
    for i in range(len(agents)):
        agents_str[i] = agents[i]+' <'+ ','.join([f'{disc_rel_i}' for disc_rel_i in disc_rel[i,:steps]])+'> '+end_agents[i]
    return ' '.join(agents_str)


def extract_tensors2(input_string):
    # Regular expression to extract values inside <>
    pattern = re.compile(r'<(\d+)>')
    # Extract values using regex
    matches = pattern.findall(input_string)
    # Initialize lists for each tensor
    nums = [int(i) for i in matches]
    nums_tensor = torch.tensor(nums)
    return nums_tensor

def extract_tensors(input_string):
    # Regular expression to extract values inside <>
    pattern = re.compile(r'<(\d+),([+-]?\d+),([+-]?\d+),([+-]\d+),([+-]\d+)>')

    # Extract values using regex
    matches = pattern.findall(input_string)

    # Initialize lists for each tensor
    t_list, x_list, y_list, x_rel_list, y_rel_list = [], [], [], [], []

    # Iterate through matches and append values to corresponding lists
    for match in matches:
        t, x, y, x_rel, y_rel = map(int, match[0:5])
        t_list.append(t)
        x_list.append(x)
        y_list.append(y)
        x_rel_list.append(x_rel)
        y_rel_list.append(y_rel)

    # Convert lists to numpy arrays
    t_tensor = torch.tensor(t_list)
    x_tensor = torch.tensor(x_list)
    y_tensor = torch.tensor(y_list)
    x_rel_tensor = torch.tensor(x_rel_list)
    y_rel_tensor = torch.tensor(y_rel_list)

    return t_tensor, x_tensor, y_tensor, x_rel_tensor, y_rel_tensor


####################################################
def abs_distance_to_velocity(abs_distance):
    return torch.cat((torch.zeros_like(abs_distance[...,0:1,:]), abs_distance[...,1:,:]-abs_distance[...,:-1,:]), dim=-2)
    # return np.concatenate((torch.zeros_like(abs_distance[...,0:1,:]), abs_distance[...,1:,:]-abs_distance[...,:-1,:]), axis = -2)

def velocity_to_accleration(velocity):
    return torch.cat((torch.zeros_like(velocity[...,0:1,:]), velocity[...,1:,:]-velocity[...,:-1,:]), dim = -2)
    # return np.concatenate((np.zeros_like(velocity[...,0:1,:]), velocity[...,1:,:]-velocity[...,:-1,:]), axis = -2)

def velocity_to_abs_distance(velocity,init_abs): 
    return torch.cumsum(velocity, dim=-2)+init_abs
    # return np.cumsum(velocity,axis=-2)+init_abs

def accleration_to_velocity(acc): 
    return torch.cumsum(acc,dim=-2)
    # return np.cumsum(acc,axis=-2)

def pair_(vel_disc):
    cantormapping = []
    for a,b in vel_disc:
        cantormapping.append(pair(a,b)) 
    return cantormapping

def depair_(cantormapping):
    vel_disc = []
    for cm in cantormapping:
        vel_disc.append(depair(cm))
    return vel_disc


def rect(x_pos, y_pos, width, height, heading, object_type, color='r', alpha=0.4, fontsize=10):
    rect_x = x_pos - width / 2
    rect_y = y_pos - height / 2
    rect = plt.Rectangle((rect_x, rect_y), width, height, linewidth=1, color=color, alpha=alpha, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_pos, y_pos), heading) + plt.gca().transData)
    plt.gca().add_patch(rect)
    plt.text(x_pos, y_pos, object_type, color='black', fontsize=fontsize, ha='center', va='center')



def vizualize(ego, ground_truth, neighbors=None, map_lanes=None, map_crosswalks=None, region_dict=None, gt=False, fig=None, background_only=False, colors_mode=1, t_colors=False):
    # visulization
    if not fig is not None:
        fig, ax = plt.subplots()
    if not background_only:
        for i in range(ego.shape[0]):
            past = ego[i]
            future = ground_truth[i]
            if gt:
                if colors_mode==1:
                    colors = ['r', 'r']
                    # colors = ['purple', 'purple']
                else:
                    colors = ['b', 'r']
                    # colors = ['r', 'b']
                # plt.scatter(past[:, 0], past[:, 1], color=colors[i], s=30, edgecolors='none', marker='_', alpha=0.5)
                # plt.scatter(future[:, 0], future[:, 1], color=colors[i], s=50, edgecolors='none', marker='_', alpha=0.5)
                plt.plot(past[:, 0], past[:, 1], color=colors[i], alpha=1.0)
                if ego.shape[0]==1:
                    if t_colors:
                        plt.scatter(future[:, 0], future[:, 1], c=np.arange(len(future)), cmap='winter', s=10, marker='o', alpha=0.7)
                    else:
                        plt.plot(future[:, 0], future[:, 1], color=colors[i+1], alpha=1.0)
                else:
                    plt.plot(future[:, 0], future[:, 1], color=colors[i], alpha=1.0)
                agent_i = past
                object_type = int((agent_i[-1, 8:].argmax(-1)+1)* agent_i[-1, 8:].sum(-1))
                rect(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type, color=colors[i])
            else:
                if colors_mode==1:
                    # cmaps = ['spring', 'cool'] # 'plasma''winter'
                    # markers = ['s', 'o']
                    cmaps = ['winter', 'winter'] # 'plasma''winter'
                    markers = ['o', 'o']
                    plt.scatter(past[:, 0], past[:, 1], c=np.arange(len(past)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1)#, edgecolors='k'
                    plt.scatter(future[:, 0], future[:, 1], c=np.arange(len(future)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1) # , edgecolors='k'
                elif colors_mode==2:
                    cmaps = ['plasma', 'plasma'] # 'plasma''winter'
                    markers = ['o', 'o']
                    plt.scatter(past[:, 0], past[:, 1], c=np.arange(len(past)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1)#, edgecolors='k'
                    plt.scatter(future[:, 0], future[:, 1], c=np.arange(len(future)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1) # , edgecolors='k'
                else:
                    # cmaps = ['Accent', 'Accent'] # 'plasma''winter'
                    markers = ['*', '*']
                    plt.scatter(past[:, 0], past[:, 1], c='r', s=5, marker=markers[i], alpha=0.3)#, edgecolors='k'
                    plt.scatter(future[:, 0], future[:, 1], c='r', s=5, marker=markers[i], alpha=0.3) # , edgecolors='k'
                    # cmaps = ['spring', 'cool'] # 'plasma''winter'
    
    if neighbors is not None:
        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                agent_i = neighbors[i]
                object_type = int((agent_i[-1, 8:].argmax(-1)+1)* agent_i[-1, 8:].sum(-1))
                rect(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type, color='y', alpha=0.5, fontsize=5)
    if map_lanes is not None:
        for i in range(map_lanes.shape[0]):
            # lanes = map_lanes[:, :, :200:2][i]
            # crosswalks = map_crosswalks[:, :, :100:2][i]

            # lanes = map_lanes[:, :, :200:2][i]
            lanes = map_lanes[i]
            # crosswalks = map_crosswalks[:, :, :100:2][i]
            

            for j in range(map_lanes.shape[1]):
                lane = lanes[j]
                if lane[0][0] != 0:
                    centerline = lane[:, 0:2]
                    centerline = centerline[centerline[:, 0] != 0]
                    left = lane[:, 3:5]
                    left = left[left[:, 0] != 0]
                    right = lane[:, 6:8]
                    right = right[right[:, 0] != 0]
                    plt.plot(centerline[:, 0], centerline[:, 1], 'k', linewidth=1, alpha=0.2) # plot centerline]
                    # plt.scatter(centerline[:, 0], centerline[:, 1], c='k', alpha=0.1, s=4)
                    plt.plot(right[:, 0], right[:, 1], 'k', linewidth=1, alpha=0.2)
                    plt.plot(left[:, 0], left[:, 1], 'k', linewidth=1, alpha=0.2)
                    # plt.scatter(right[:, 0], right[:, 1], c='b', alpha=0.1, marker='o', s=4)
                    # plt.scatter(left[:, 0], left[:, 1], c='r', alpha=0.1, marker='o', s=4)
        if map_crosswalks is not None:
            crosswalks = map_crosswalks[i]
            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk = crosswalk[crosswalk[:, 0] != 0]
                    plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'y', linewidth=1, alpha=0.2) # plot crosswalk
    
    
    # for i in range(region_dict[32].shape[0]):
    #     plt.scatter(region_dict[32][i,:,0],region_dict[32][i,:,1],marker='*',s=10)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    # plt.show()
    # plt.close()
    return fig

def vizualize01(ego, ground_truth, neighbors=None, map_lanes=None, map_crosswalks=None, region_dict=None, gt=False, fig=None, background_only=False, colors_mode=1):
    # visulization
    if not fig is not None:
        fig, ax = plt.subplots()

    for i in range(ego.shape[0]):
        past = ego[i]
        future = ground_truth[i]
        if gt:
            if colors_mode==1:
                # colors = ['r', 'b']
                colors = ['purple', 'purple']
            else:
                colors = ['r', 'b']
                # colors = ['r', 'b']
            # plt.scatter(past[:, 0], past[:, 1], color=colors[i], s=30, edgecolors='none', marker='_', alpha=0.5)
            # plt.scatter(future[:, 0], future[:, 1], color=colors[i], s=50, edgecolors='none', marker='_', alpha=0.5)
            plt.plot(past[:, 0], past[:, 1], color=colors[i], alpha=1.0)
            plt.plot(future[:, 0], future[:, 1], color=colors[i], alpha=1.0)
            agent_i = past
            object_type = int((agent_i[-1, 8:].argmax(-1)+1)* agent_i[-1, 8:].sum(-1))
            rect(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type, color=colors[i])
        else:
            if not background_only:
                if colors_mode==1:
                    # cmaps = ['spring', 'cool'] # 'plasma''winter'
                    # markers = ['s', 'o']
                    cmaps = ['winter', 'winter'] # 'plasma''winter'
                    markers = ['o', 'o']
                    plt.scatter(past[:, 0], past[:, 1], c=np.arange(len(past)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1)#, edgecolors='k'
                    plt.scatter(future[:, 0], future[:, 1], c=np.arange(len(future)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1) # , edgecolors='k'
                elif colors_mode==2:
                    cmaps = ['plasma', 'plasma'] # 'plasma''winter'
                    markers = ['o', 'o']
                    plt.scatter(past[:, 0], past[:, 1], c=np.arange(len(past)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1)#, edgecolors='k'
                    plt.scatter(future[:, 0], future[:, 1], c=np.arange(len(future)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1) # , edgecolors='k'
                else:
                    # cmaps = ['Accent', 'Accent'] # 'plasma''winter'
                    markers = ['*', '*']
                    plt.scatter(past[:, 0], past[:, 1], c='r', s=5, marker=markers[i], alpha=0.3)#, edgecolors='k'
                    plt.scatter(future[:, 0], future[:, 1], c='r', s=5, marker=markers[i], alpha=0.3) # , edgecolors='k'
                    # cmaps = ['spring', 'cool'] # 'plasma''winter'
    
    if neighbors is not None:
        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                agent_i = neighbors[i]
                object_type = int((agent_i[-1, 8:].argmax(-1)+1)* agent_i[-1, 8:].sum(-1))
                rect(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type, color='y', alpha=0.1, fontsize=5)
    if map_lanes is not None and map_crosswalks is not None:
        for i in range(map_lanes.shape[0]):
            # lanes = map_lanes[:, :, :200:2][i]
            # crosswalks = map_crosswalks[:, :, :100:2][i]

            # lanes = map_lanes[:, :, :200:2][i]
            lanes = map_lanes[i]
            # crosswalks = map_crosswalks[:, :, :100:2][i]
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
                    plt.plot(centerline[:, 0], centerline[:, 1], 'k', linewidth=1, alpha=0.1) # plot centerline
                    plt.plot(right[:, 0], right[:, 1], 'k', linewidth=1, alpha=0.1)
                    plt.plot(left[:, 0], left[:, 1], 'k', linewidth=1, alpha=0.1)
                        
        for k in range(map_crosswalks.shape[1]):
            crosswalk = crosswalks[k]
            if crosswalk[0][0] != 0:
                crosswalk = crosswalk[crosswalk[:, 0] != 0]
                plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'y', linewidth=1, alpha=0.2) # plot crosswalk
    
    
    # for i in range(region_dict[32].shape[0]):
    #     plt.scatter(region_dict[32][i,:,0],region_dict[32][i,:,1],marker='*',s=10)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    # plt.show()
    # plt.close()
    return fig
    

    # if wandb_save:
    #     wandb.log({ "Visualization": wandb.Image(image_name) })

def find_sublist(large_tensor, sublist_tensor):
        # Convert the lists to PyTorch tensors
        # large_tensor = torch.tensor(large_list)
        # sublist_tensor = torch.tensor(sublist)
        
        # Calculate the length of both lists
        len_large = len(large_tensor)
        len_sub = len(sublist_tensor)
        
        # Iterate over the large tensor with a sliding window of the sublist's length
        for i in range(len_large - len_sub + 1):
            # Extract the window from the large tensor
            window = large_tensor[i:i+len_sub]
            
            # Compare the window to the sublist tensor
            if torch.equal(window, sublist_tensor):
                return i  # Return the starting index of the first occurrence
        return -1  # Return -1 if the sublist is not found