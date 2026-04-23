from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
import torch
# Visualize
from scipy.interpolate import splprep, splev
import numpy as np


def smooth_line(points, num =5):
    # print(len(points))

    num = 30
    # print("mean = ", points.mean(axis = 0))
    if len(points) < num:
        return points
    
    if len(points) < num:
        return points
    smoothed = np.zeros_like(points)
    for i in range(len(points)):
        start = max(0, i - num // 2)
        end = min(len(points), i + num // 2 + 1)
        smoothed[i] = points[start:end].mean(axis=0)
    return smoothed


# def smooth_line(points, num=5):
#     """
#     Replace every `chunk_size` points with their mean.
#     Returns a shorter and smoother polyline.
    
#     Args:
#         points (np.ndarray): (N, 2) array of 2D points.
#         chunk_size (int): Number of points per chunk.

#     Returns:
#         np.ndarray: (N // chunk_size, 2) averaged points.
#     """
#     num = 10
#     if points is None or len(points) < num:
#         return points

#     num_chunks = len(points) // num
#     remaining = len(points) % num

#     # Average full chunks
#     main_points = points[:num_chunks * num]
#     reshaped = main_points.reshape(num_chunks, num, 2)
#     averaged = reshaped.mean(axis=1)

#     # Repeat each average `num` times
#     repeated = np.repeat(averaged, num, axis=0)

#     # Handle leftover: repeat last averaged point `remaining` times
#     if remaining > 0:
#         last_avg = averaged[-1:]
#         padding = np.repeat(last_avg, remaining, axis=0)
#         repeated = np.concatenate([repeated, padding], axis=0)
#     print(repeated)
#     return repeated

# def smooth_line(points, num=5):
#     if points is None or len(points) < num:
#         return points
#     kernel = np.ones(num) / num
#     smoothed = np.copy(points)
#     smoothed[:, 0] = np.convolve(points[:, 0], kernel, mode='same')
#     smoothed[:, 1] = np.convolve(points[:, 1], kernel, mode='same')
#     return smoothed


# def smooth_line(points, smoothing=0.0, num=100):
#     """Smooth a polyline with cubic spline. Expects (N,2) array."""
#     if points is None or len(points) < 4:  
#         # Not enough points to smooth → just return original
#         return points

#     try:
#         print(f"Smoothing called on {len(points)} points")
#         x, y = points[:,0], points[:,1]
#         tck, u = splprep([x, y], s=smoothing)
#         unew = np.linspace(0, 1, num)
#         x_new, y_new = splev(unew, tck)
#         return np.vstack([x_new, y_new]).T
#     except Exception as e:
#         # fallback: return raw points if smoothing fails
#         # print(f"[smooth_line] Warning: spline failed ({e}), using raw points")
#         return points


AGENT_TYPE_COLORS = {
    1: 'blue',       # Vehicle / Car
    2: 'green',      # Pedestrian
    3: 'orange',     # Cyclist
    8: 'red',        # Ego (force red)
}



def viz_multimodal_18may(inputs, ego_multimodal, gt_ego_future, fixed_color=None, fig=None, ax=None):
   
    if fig is None:
        fig, ax = vizualize_(
            inputs['ego_state'][:,:2].cpu(), 
            None, 
            neighbors=inputs['neighbors_state'][:].cpu(), 
            # map_lanes=inputs['map_lanes'][batch_sample][:,:, :80:2].cpu(),
            map_lanes=inputs['map_lanes'].cpu(), 
            map_crosswalks=inputs['map_crosswalks'].cpu(), 
            region_dict=None, 
            gt=False, 
            fig=None,
            ax=None,
            background_only=True,
            map_of_ego_only=True,
            boundaries_L=None,
            boundaries_R=None,
            more_boundaries= inputs['additional_boundaries'].cpu() if inputs['additional_boundaries'] is not None else None,
            )

        # GT
        agent_select=0
        object_type = int((inputs['ego_state'][agent_select,-1, 8:].argmax(-1)+1)* inputs['ego_state'][agent_select,-1, 8:].sum(-1))
        inputs['ego_state'] = inputs['ego_state'].cpu()
        object_type = 'Ego'
        rect_(inputs['ego_state'][agent_select,-1,0], inputs['ego_state'][agent_select,-1,1], inputs['ego_state'][agent_select,-1,5], inputs['ego_state'][agent_select,-1,6], inputs['ego_state'][agent_select,-1,2], object_type, color='k', alpha=0.8, fontsize=10)
        # add_arrow = (gt_ego_future[agent_select,-1,:2]-gt_ego_future[agent_select,0,:2]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
        # add_arrow = False
        # fig = vizualize_traj_arrow(gt_ego_future[agent_select,...,:2].cpu(), fig, ax, 'k', '--', add_arrow, 0.9)

    agent_select = 0
    if fixed_color is not None:
        colors = [fixed_color,fixed_color,fixed_color,fixed_color,fixed_color,fixed_color]
    else:
        colors = ['b', 'g', 'y', 'purple', 'orange', 'cyan']
        
    for i in range(ego_multimodal.shape[1]):
        add_arrow = (ego_multimodal[agent_select,i,-1]-ego_multimodal[agent_select,i,0]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
        add_arrow = False
        fig = vizualize_traj_arrow(ego_multimodal[agent_select,i].cpu(), fig, ax, colors[i], 'solid', add_arrow, alpha=0.2)
    
    
    

    


    plt.gca().set_facecolor('silver')
    plt.gca().margins(0)  
    plt.gca().set_aspect('equal')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axis([-60 + inputs['ego_state'][0,-1,0], 60 + inputs['ego_state'][0,-1,0], -60 + inputs['ego_state'][0,-1,1], 60 + inputs['ego_state'][0,-1,1]])

    # plt.savefig('ex.png')
    # plt.xlim(-70,70)
    # plt.ylim(-70,70)
    # plt.close()
    return fig, ax

def viz_multimodal_nuplan(inputs, ego_multimodal, gt_ego_future, two_agents=True):
    fig, ax = vizualize_(
        inputs['ego_state'][:,:2].cpu(), 
        None, 
        neighbors=inputs['neighbors_state'][:].cpu(), 
        map_lanes=inputs['map_lanes'].cpu(), 
        map_crosswalks=inputs['map_crosswalks'].cpu(), 
        region_dict=None, 
        gt=False, 
        fig=None,
        ax=None,
        background_only=True,
        map_of_ego_only=True,
        boundaries_L=None,
        boundaries_R=None,
        more_boundaries=None,
    )

    # colors = ['darkred', 'green', 'orange', 'purple', 'cyan', 'magenta']
    colors = ['teal', 'green', 'magenta', 'purple', 'orange', 'cyan']
    # === Predicted agent (if any)
    if two_agents and inputs['ego_state'].shape[0] > 1:
        agent_select = 1
        rect_(inputs['ego_state'][agent_select,-1,0], inputs['ego_state'][agent_select,-1,1], 
              inputs['ego_state'][agent_select,-1,5], inputs['ego_state'][agent_select,-1,6], 
              inputs['ego_state'][agent_select,-1,2], 'Agent 2', color='darkred', alpha=0.5, fontsize=6)
        
        for i in range(ego_multimodal.shape[1]):
            traj = ego_multimodal[agent_select,i].cpu()
            ax.plot(traj[:,0], traj[:,1], linestyle='-', color='gray', linewidth=1.0, alpha=0.4)

    # === Ground Truth agent (agent 0)
    agent_select = 0
    rect_(inputs['ego_state'][agent_select,-1,0], inputs['ego_state'][agent_select,-1,1], 
          inputs['ego_state'][agent_select,-1,5] * 0.5, inputs['ego_state'][agent_select,-1,6] * 0.5, 
          inputs['ego_state'][agent_select,-1,2], 'Ego', color='red', alpha=0.8, fontsize=8)

    # Ground truth trajectory
    traj_gt = gt_ego_future[agent_select, :, :2].cpu()
    ax.plot(traj_gt[:, 0], traj_gt[:, 1], linestyle='--', color='red', linewidth=4, label='GT', zorder=1)

    # Multimodal predictions
    # import pdb; pdb.set_trace()
    # for i in range(ego_multimodal.shape[1]):
    #     traj = ego_multimodal[agent_select, 5 - i].cpu()
    #     ax.plot(traj[:, 0], traj[:, 1], linestyle='-', color=colors[i % len(colors)], linewidth=4, alpha=0.8, label=f'Modal {i}')
    # Compute lengths of each trajectory
    traj_lengths = []
    for i in range(ego_multimodal.shape[1]):
        traj = ego_multimodal[agent_select, i]
        # Compute total L2 distance along the trajectory
        length = torch.norm(traj[1:] - traj[:-1], dim=1).sum().item()
        traj_lengths.append((i, length))

    # Sort indices by length (longest first, shortest last)
    sorted_indices = sorted(traj_lengths, key=lambda x: -x[1])  # negative for descending order

    # Plot in sorted order (shortest last = on top)
    for plot_i, (i, _) in enumerate(sorted_indices):
        traj = ego_multimodal[agent_select, i].cpu()
        ax.plot(traj[:, 0], traj[:, 1], 
                linestyle='-', 
                color=colors[plot_i % len(colors)], 
                linewidth=4, 
                alpha=0.8, 
                label=f'Modal {i}')

        # === Format the plot ===
        ax.set_facecolor('whitesmoke')
        ax.set_aspect('equal')
        ax.axis('off')

    # Automatically zoom to ROI
    # all_x = torch.cat([traj_gt[:,0]] + [ego_multimodal[agent_select,i,:,0].cpu() for i in range(ego_multimodal.shape[1])])
    # all_y = torch.cat([traj_gt[:,1]] + [ego_multimodal[agent_select,i,:,1].cpu() for i in range(ego_multimodal.shape[1])])
    # margin = 10.0
    # ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    # ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # ax.plot([all_x.min(), all_x.max(), all_x.max(), all_x.min(), all_x.min()],
    #     [all_y.min(), all_y.min(), all_y.max(), all_y.max(), all_y.min()],
    #     'r--', label='Zoom Bounds')

    # import pdb; pdb.set_trace()
    # ax.set_xlim(-10, 50)
    # ax.set_ylim(-50, 50)

    # === Merge previous legend (modal, GT) + agent type legend
    existing_handles, existing_labels = ax.get_legend_handles_labels()

    agent_type_legend = [
        mpatches.Patch(color=color, label=label)
        for label, color in {
            'Ego': 'red',
            'Vehicle': 'blue',
            'Pedestrian': 'green',
            'Cyclist': 'orange',
        }.items()
    ]

    # Combine both
    # ax.legend(handles=existing_handles + agent_type_legend, loc='lower right', fontsize=6, title="Legend", title_fontsize=7)


    fig.tight_layout()
    return fig


# Visualize
def viz_multimodal(inputs, ego_multimodal, gt_ego_future, batch_sample = 0):
    fig, ax = vizualize_(
        inputs['ego_state'][batch_sample,:,:2].cpu(), 
        None, 
        neighbors=inputs['neighbors_state'][batch_sample,:5].cpu(), 
        # map_lanes=inputs['map_lanes'][batch_sample][:,:, :80:2].cpu(),
        map_lanes=inputs['map_lanes'][batch_sample].cpu(), 
        map_crosswalks=inputs['map_crosswalks'][batch_sample].cpu(), 
        region_dict=None, 
        gt=False, 
        fig=None,
        ax=None,
        background_only=True,
        map_of_ego_only=True,
        boundaries_L=inputs['boundaries_L'][batch_sample][:,:,:,:].cpu() if inputs['boundaries_L'] is not None else inputs['boundaries_L'],
        boundaries_R=inputs['boundaries_R'][batch_sample][:,:,:,:].cpu() if inputs['boundaries_R'] is not None else None,
        more_boundaries= inputs['additional_boundaries'][batch_sample][:,:,:].cpu() if inputs['additional_boundaries'] is not None else None,
        )

    
    colors = ['b', 'g', 'y', 'purple', 'orange', 'cyan']
    for i in range(ego_multimodal.shape[1]):
        add_arrow = (ego_multimodal[batch_sample,i,-1]-ego_multimodal[batch_sample,i,0]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
        fig = vizualize_traj_arrow(ego_multimodal[batch_sample,i].cpu(), fig, ax, colors[i], 'solid', add_arrow)
    
    # GT
    object_type = int((inputs['ego_state'][batch_sample,-1, 8:].argmax(-1)+1)* inputs['ego_state'][batch_sample,-1, 8:].sum(-1))
    inputs['ego_state'] = inputs['ego_state'].cpu()
    rect_(inputs['ego_state'][batch_sample,-1,0], inputs['ego_state'][batch_sample,-1,1], inputs['ego_state'][batch_sample,-1,5], inputs['ego_state'][batch_sample,-1,6], inputs['ego_state'][batch_sample,-1,2], object_type, color='r', alpha=0.4, fontsize=10)
    add_arrow = (gt_ego_future[batch_sample,-1,:2]-gt_ego_future[batch_sample,0,:2]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
    fig = vizualize_traj_arrow(gt_ego_future[batch_sample,...,:2].cpu(), fig, ax, 'r', '--', add_arrow, 0.9)
    


    plt.gca().set_facecolor('silver')
    plt.gca().margins(0)  
    plt.gca().set_aspect('equal')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axis([-60 + inputs['ego_state'][0,-1,0], 60 + inputs['ego_state'][0,-1,0], -60 + inputs['ego_state'][0,-1,1], 60 + inputs['ego_state'][0,-1,1]])

    # plt.savefig('ex.png')
    # plt.xlim(-70,70)
    # plt.ylim(-70,70)
    plt.close()
    return fig
    

def vizualize_traj_arrow(traj, fig, ax, color, linestyle, add_arrow, alpha=0.5): # single agent, single modality
    plt.plot(traj[:,0], traj[:,1], linestyle=linestyle, color=color, alpha=alpha, zorder=10) 
    if add_arrow:
        ax.annotate('', xy=traj[-1], xytext=traj[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1, linestyle='solid'))        
    return fig

def rect_(x_pos, y_pos, width, height, heading, object_type, color=None, alpha=0.4, fontsize=10):
    if isinstance(object_type, int):  # numerical ID
        color = AGENT_TYPE_COLORS.get(object_type, 'gray') if color is None else color
    else:  # fallback for labels like "Ego"
        color = 'red' if object_type == 'Ego' else (color or 'gray')

    rect_x = x_pos - width / 2
    rect_y = y_pos - height / 2
    rect = plt.Rectangle(
        (rect_x, rect_y), width, height,
        linewidth=1, color=color, alpha=alpha, zorder=3,
        transform=mpl.transforms.Affine2D().rotate_around(*(x_pos, y_pos), heading) + plt.gca().transData
    )
    plt.gca().add_patch(rect)

    # === Draw arrow indicating heading ===
    arrow_length = height * 0.7  # length proportional to vehicle height
    dx = arrow_length * np.cos(heading)
    dy = arrow_length * np.sin(heading)
    plt.arrow(
        x_pos, y_pos, dx, dy,
        head_width=width * 0.3, head_length=width * 0.3,
        fc='black', ec='black', linewidth=1.0, zorder=5
    )
    # plt.text(x_pos, y_pos, str(object_type), color='black', fontsize=5, ha='center', va='center', alpha=0.8)





def vizualize_(ego, ground_truth, neighbors=None, map_lanes=None, map_crosswalks=None, region_dict=None, gt=False, fig=None, background_only=False, colors_mode=1, ax=None, map_of_ego_only = True, boundaries_L=None, boundaries_R=None, more_boundaries=None):
    # visulization
    if not fig is not None:
        fig, ax = plt.subplots(dpi=600)
        fig.set_tight_layout(True)
        # plt.gca().set_facecolor('silver')
        plt.gca().set_facecolor('whitesmoke')
        
        plt.gca().margins(0)  
        
        # fig, ax = plt.subplots()
        # dpi = 100
        # size_inches = 800 / dpi
        # fig.set_size_inches([size_inches, size_inches])
        # fig.set_dpi(dpi)
        # fig.set_tight_layout(True)
    if not background_only:
        for i in range(ego.shape[0]):
            past = ego[i]
            future = ground_truth[i]
            if gt:
                if colors_mode==1:
                    colors = ['r', 'r']
                    # colors = ['purple', 'purple']
                else:
                    colors = ['r', 'b']
                    # colors = ['r', 'b']
                # plt.scatter(past[:, 0], past[:, 1], color=colors[i], s=30, edgecolors='none', marker='_', alpha=0.5)
                # plt.scatter(future[:, 0], future[:, 1], color=colors[i], s=50, edgecolors='none', marker='_', alpha=0.5)
                plt.plot(past[:, 0], past[:, 1], color=colors[i], alpha=1.0)
                if ego.shape[0]==1:
                    plt.plot(future[:, 0], future[:, 1], color=colors[i+1], alpha=1.0)
                else:
                    plt.plot(future[:, 0], future[:, 1], color=colors[i], alpha=1.0)
                agent_i = past
                object_type = int((agent_i[-1, 8:].argmax(-1)+1)* agent_i[-1, 8:].sum(-1))

                scale_factor = 2.0 if object_type == 2 else 1.0  # Pedestrian = 2
                rect_(
                    x_pos=agent_i[-1, 0], 
                    y_pos=agent_i[-1, 1],
                    width=agent_i[-1, 5] * scale_factor, 
                    height=agent_i[-1, 6] * scale_factor,
                    heading=agent_i[-1, 2], 
                    object_type=object_type,
                    alpha=0.5, 
                    fontsize=5
                )
                # rect_(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type, color=colors[i])
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
                obj_type_id = int((agent_i[-1, 8:].argmax(-1)+1) * agent_i[-1, 8:].sum(-1))
                scale_factor = 0.5
                rect_(
                    x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1],
                    width=agent_i[-1, 5] * scale_factor, height=agent_i[-1, 6] * scale_factor,
                    heading=agent_i[-1, 2], object_type=obj_type_id,
                    alpha=0.5, fontsize=5
                )

    if map_lanes is not None:
        agent_maps = range(map_lanes.shape[0]) if not map_of_ego_only else [0]
        for i in agent_maps:
            # lanes = map_lanes[:, :, :200:2][i]
            # crosswalks = map_crosswalks[:, :, :100:2][i]

            # lanes = map_lanes[:, :, :200:2][i]
            map_lanes = map_lanes.unsqueeze(0)
            lanes = map_lanes[i]
            # crosswalks = map_crosswalks[:, :, :100:2][i]
            

            
            centerlines = []
            roadlines = []  # (x, y, color, linestyle, linewidth, alpha)

            ego_xy = torch.tensor([0., -4.6566e-10])

            for j in range(map_lanes.shape[1]):
                lane = lanes[j]
                if lane[0][0] == 0:
                    continue

                # --- Centerline ---
                centerline = lane[:, 0:2]
                centerline = centerline[centerline[:, 0] != 0]
                centerline = smooth_line(centerline, num=50)
                if len(centerline) > 1:
                    centerlines.append(centerline)

                # --- Road boundaries ---
                if lane.shape[-1] > 2:
                    color_linestyle_per_type = [
                        ('k', None),  # 0
                        ('w', 'dashed'),  # 1
                        ('w', 'solid'),   # 2
                        ('w', 'solid'),   # 3
                        ('xkcd:yellow', 'dashed'),  # 4
                        ('xkcd:yellow', 'dashed'),  # 5
                        ('xkcd:yellow', 'solid'),   # 6
                        ('xkcd:yellow', 'solid'),   # 7
                        ('xkcd:yellow', 'dotted'),  # 8
                        ('k-', None),  # 9
                        ('k-', None),  # 10
                        ('k-', None),  # 11
                    ]

                    for lr_idx in [11, 12]:  # left and right boundaries
                        type_feature_idx = lr_idx
                        lane_type = [{"type": lane[0, type_feature_idx].int().item(), "start": 0}]
                        for k in range(1, lane.shape[0]):
                            point_type = lane[k, type_feature_idx].int().item()
                            if lane_type[-1]["type"] != point_type:
                                lane_type[-1]["end"] = k
                                if k < (lane.shape[0] - 1):
                                    lane_type.append({"type": point_type, "start": k})
                            if k == (lane.shape[0] - 1):
                                lane_type[-1]["end"] = k + 1

                        for lane_type_ in lane_type:
                            lane_boundary = lane[:, 3:5] if lr_idx == 11 else lane[:, 6:8]
                            lane_boundary = lane_boundary[lane_type_["start"]:lane_type_["end"]]
                            lane_boundary = lane_boundary[lane_boundary[:, 0] != 0]
                            lane_boundary = smooth_line(lane_boundary, num=50)

                            if len(lane_boundary) < 2:
                                continue

                            lane_mean = lane_boundary.mean(axis=0)
                            dist = torch.norm(torch.tensor(lane_mean, dtype=torch.float32) - ego_xy)
                            if dist > 3000:
                                continue

                            lane_type_id = lane_type_["type"]
                            if lane_type_id >= len(color_linestyle_per_type):
                                continue

                            color, linestyle = color_linestyle_per_type[lane_type_id]
                            if lane_type_id in [1, 2, 3, 4, 5, 6, 7, 8]:
                                roadlines.append((lane_boundary[:, 0], lane_boundary[:, 1], color, linestyle, 1, 0.3, lane_boundary))
                            elif lane_type_id in [9, 10, 11]:
                                roadlines.append((lane_boundary[:, 0], lane_boundary[:, 1], color, None, 1, 1.0, lane_boundary))
                            else:
                                roadlines.append((lane_boundary[:, 0], lane_boundary[:, 1], color, None, 0.3, 1.0, lane_boundary))

            # --- PLOTTING PHASE ---
            # Plot all centerlines
            # for cl in centerlines:
            #     plt.plot(cl[:, 0], cl[:, 1], 'gray', linestyle="dashed", linewidth=0.3, alpha=0.2)

            for x, y, color, linestyle, lw, alpha, lane_boundary in roadlines:
                pass
                # plt.plot(x, y, color=color, linestyle=linestyle, linewidth=lw, alpha=alpha)
        
        # color_linestyle_per_type = [('purple',None), ('purple','dashed'), ('purple','solid'), ('purple','solid'), ('purple','dashed'), ('purple','dashed'), ('purple','solid'), ('purple','solid'), ('purple','dotted'), ('purple', None), ('purple', None), ('purple', None)]
        if boundaries_L is not None and boundaries_R is not None:
            agent_i = 0
            for boundaries in [boundaries_L[0,boundaries_L[0].sum(-1).sum(-1)!=0], boundaries_R[0,boundaries_R[0].sum(-1).sum(-1)!=0]]:
                for j in range(boundaries.shape[0]):
                    lane = boundaries[j]
                    for i_lr_idx, lr_idx in enumerate(range(30,40)):
                        type_feature_idx = lr_idx
                        lane_type = [{"type": lane[0,type_feature_idx].int().item(), "start": 0}]
                        for k in range(1, lane.shape[0]):
                            point_type = lane[k,type_feature_idx].int().item()
                            if lane_type[len(lane_type)-1]["type"] != point_type:
                                lane_type[len(lane_type)-1]["end"] = k
                                if k<(lane.shape[0]-1):
                                    lane_type.append({"type": point_type, "start": k})
                            if k == (lane.shape[0]-1):
                                lane_type[len(lane_type)-1]["end"] = k+1

                        for lane_type_ in lane_type:
                            lane_boundary = lane[:, i_lr_idx*3 : (i_lr_idx+1)*3]
                            lane_boundary = lane_boundary[lane_type_["start"]:lane_type_["end"]]
                            lane_boundary = lane_boundary[lane_boundary[:, 0] != 0]
                            if lane_type_["type"] in [1,2,3,4,5,6,7,8]:
                                plt.plot(lane_boundary[:, 0], lane_boundary[:, 1], color_linestyle_per_type[lane_type_["type"]][0], linestyle=color_linestyle_per_type[lane_type_["type"]][1], linewidth=2)
                            elif lane_type_["type"] in [9,10,11]:
                                plt.plot(lane_boundary[:, 0], lane_boundary[:, 1], color_linestyle_per_type[lane_type_["type"]][0], linewidth=2)
                            else:
                                plt.plot(lane_boundary[:, 0], lane_boundary[:, 1], color_linestyle_per_type[lane_type_["type"]][0], linewidth=2)
        
        if more_boundaries is not None:
            more_boundaries = more_boundaries[more_boundaries.sum(-1).sum(-1)!=0]
            agent_i = 0
            for j in range(more_boundaries.shape[0]):
                lane = more_boundaries[j]
                lane_type = [{"type": lane[0, 3].int().item(), "start": 0}]
                for k in range(1, lane.shape[0]):
                    point_type = lane[k,3].int().item()
                    if lane_type[len(lane_type)-1]["type"] != point_type:
                        lane_type[len(lane_type)-1]["end"] = k
                        if k<(lane.shape[0]-1):
                            lane_type.append({"type": point_type, "start": k})
                    if k == (lane.shape[0]-1):
                        lane_type[len(lane_type)-1]["end"] = k+1
                
                for lane_type_ in lane_type:
                    lane_boundary = lane[:, :3]
                    lane_boundary = lane_boundary[lane_type_["start"]:lane_type_["end"]]
                    lane_boundary = lane_boundary[lane_boundary[:, 0] != 0]
                    if lane_type_["type"] in [1,2,3,4,5,6,7,8]:
                        plt.plot(lane_boundary[:, 0], lane_boundary[:, 1], color_linestyle_per_type[lane_type_["type"]][0], linestyle=color_linestyle_per_type[lane_type_["type"]][1], linewidth=1, alpha=0.3)
                    elif lane_type_["type"] in [9,10,11]:
                        plt.plot(lane_boundary[:, 0], lane_boundary[:, 1], color_linestyle_per_type[lane_type_["type"]][0], linewidth=1)
                    else:
                        plt.plot(lane_boundary[:, 0], lane_boundary[:, 1], color_linestyle_per_type[lane_type_["type"]][0], linewidth=0.3)


        
        if map_crosswalks is not None:
            map_crosswalks = map_crosswalks.unsqueeze(0)
            crosswalks = map_crosswalks[i]
            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    # crosswalk = smooth_line(crosswalk[crosswalk[:, 0] != 0])
                    plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'white',linestyle='--', linewidth=2, alpha=1) # plot crosswalk
                

        


    
    
    # for i in range(region_dict[32].shape[0]):
    #     plt.scatter(region_dict[32][i,:,0],region_dict[32][i,:,1],marker='*',s=10)
    # plt.gca().set_aspect('equal')
    # plt.tight_layout()
    # plt.save('ex.png')
    # plt.show()
    # plt.close()
    return fig, ax