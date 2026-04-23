# Title: Visualizing Instructions Data samples
# Description: This script visualize examples, with the paired instructions
# This script is useful for validating instructions as well visualizing output examples

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

def viz_instruction_sample(ego_state, neighbors, map_lanes, map_crosswalks,gt_ego_future):
    # ego_state = inputs['ego_state'][:,:,:2].cpu()
    # neighbors=inputs['neighbors_state'][:3].cpu()
    # map_lanes=inputs['map_lanes'][:,:, :80:2].cpu()
    # map_crosswalks=inputs['map_crosswalks'][:,:, :100:2].cpu()
    fig, ax = vizualize(
        ego = ego_state, 
        ground_truth=None, 
        neighbors=neighbors, 
        map_lanes=map_lanes, 
        map_crosswalks=map_crosswalks, 
        region_dict=None, 
        gt=False, 
        fig=None,
        ax=None,
        background_only=True)

    rect(ego_state[-1,0], ego_state[-1,1], ego_state[-1,5], ego_state[-1,6], ego_state[-1,2], 'Ego', color='r', alpha=0.4, fontsize=7)
    # add_arrow = (gt_ego_future[-1,:2]-gt_ego_future[0,:2]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
    # fig = vizualize_traj_arrow(gt_ego_future[...,:2].cpu(), fig, ax, 'r', '--', add_arrow, 0.9)
    add_arrow = False
    fig = scatter_traj(gt_ego_future[...,:2].cpu(), fig, ax, 'r', '--', add_arrow, 0.9)

    return fig

def rect(x_pos, y_pos, width, height, heading, object_type, color='r', alpha=0.4, fontsize=10):
    rect_x = x_pos - width / 2
    rect_y = y_pos - height / 2
    rect = plt.Rectangle((rect_x, rect_y), width, height, linewidth=1, color=color, alpha=alpha, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_pos, y_pos), heading) + plt.gca().transData)
    plt.gca().add_patch(rect)
    plt.text(x_pos, y_pos, object_type, color='black', fontsize=fontsize, ha='center', va='center')

def scatter_traj(traj, fig, ax, color, linestyle, add_arrow, alpha=0.5): # single agent, single modality
    # Simulating a time array (sequential index)
    time = np.arange(traj.shape[0])
    # Create a color array based on the 'viridis' colormap
    normalized_time = time / max(time)
    colors = plt.cm.tab10(normalized_time) #plt.cm.cool(normalized_time)  # Normalize time values for colormap
    # Creating the scatter plot
    scatter = plt.scatter(traj[:,0], traj[:,1], c=colors, alpha=0.95)
    # Creating a color bar
    # Adding color bar correctly matching the 'cool' colormap
    sm = plt.cm.ScalarMappable(cmap='tab10', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Future Time')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0.1s', '8s'])
    # colorbar = plt.colorbar(scatter, label='Future Time (seconds)')
    # colorbar.set_ticks([0, 1])  # Optional: set ticks to match start/end colors if using a subset of the colormap
    # colorbar.set_ticklabels(['0.1s', '8s'])
    # Set additional tick marks for the colorbar if needed
    # additional_ticks = np.linspace(0, 1, num=10)  # Adjust number as needed for finer gradation
    # colorbar.set_ticks(np.concatenate(([0, 1], additional_ticks)))  # Merge arrays
    # colorbar.set_ticklabels(['Start'] + ['']*(len(additional_ticks)-2) + ['End'])  # Label only start/end
    # plt.plot(traj[:,0], traj[:,1], linestyle=linestyle, color=color, alpha=alpha)
    return fig

def vizualize_traj_arrow(traj, fig, ax, color, linestyle, add_arrow, alpha=0.5): # single agent, single modality
    plt.plot(traj[:,0], traj[:,1], linestyle=linestyle, color=color, alpha=alpha) 
    if add_arrow:
        ax.annotate('', xy=traj[-1], xytext=traj[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, linestyle='solid'))        
    return fig

def vizualize(ego, ground_truth, neighbors=None, map_lanes=None, map_crosswalks=None, region_dict=None, gt=False, fig=None, background_only=False, ax=None):
    # visulization
    if not fig is not None:
        fig, ax = plt.subplots(dpi=300)
    if not background_only:
        for i in range(ego.shape[0]):
            past = ego[i]
            future = ground_truth[i]
            if gt:
                colors = ['r', 'r']
                plt.plot(past[:, 0], past[:, 1], color=colors[i], alpha=1.0)
                if ego.shape[0]==1:
                    plt.plot(future[:, 0], future[:, 1], color=colors[i+1], alpha=1.0)
                else:
                    plt.plot(future[:, 0], future[:, 1], color=colors[i], alpha=1.0)
                agent_i = past
                object_type = int((agent_i[-1, 8:].argmax(-1)+1)* agent_i[-1, 8:].sum(-1))
                rect(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type, color=colors[i])
            else:
                cmaps = ['winter', 'winter'] # 'plasma''winter'
                markers = ['o', 'o']
                plt.scatter(past[:, 0], past[:, 1], c=np.arange(len(past)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1)#, edgecolors='k'
                plt.scatter(future[:, 0], future[:, 1], c=np.arange(len(future)), cmap=cmaps[i], s=10, marker=markers[i], alpha=1) # , edgecolors='k'
    
    if neighbors is not None:
        for i in range(neighbors.shape[0]):
            if neighbors[i, -1, 0] != 0:
                agent_i = neighbors[i]
                # object_type = int((agent_i[-1, 8:].argmax(-1)+1)* agent_i[-1, 8:].sum(-1))
                rect(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type='', color='lightgray', alpha=0.5, fontsize=5)
    if map_lanes is not None:
        for i in range(map_lanes.shape[0]):
            lanes = map_lanes[i]
            
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
                    plt.plot(right[:, 0], right[:, 1], 'k', linewidth=1, alpha=0.2)
                    plt.plot(left[:, 0], left[:, 1], 'k', linewidth=1, alpha=0.2)
        if map_crosswalks is not None:
            crosswalks = map_crosswalks[i]
            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk = crosswalk[crosswalk[:, 0] != 0]
                    plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'y', linewidth=1, alpha=0.2) # plot crosswalk
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    return fig, ax