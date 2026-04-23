##
##

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np



# Visualize
def viz_multimodal2(inputs, ego_multimodal, gt_ego_future, inter_future, batch_sample = 0):
    fig, ax = vizualize(
        inputs['ego_state'][:,:,:2].cpu(), 
        None, 
        neighbors=inputs['neighbors_state'][batch_sample,:3].cpu(), 
        map_lanes=inputs['map_lanes'][batch_sample][:,:, :80:2].cpu(), 
        map_crosswalks=inputs['map_crosswalks'][batch_sample][:,:, :100:2].cpu(), 
        region_dict=None, 
        gt=False, 
        fig=None,
        ax=None,
        background_only=True)

    
    # colors = ['b', 'g', 'y', 'purple', 'orange', 'cyan']
    # for i in range(ego_multimodal.shape[2]):
    #     for agent_i in range(ego_multimodal.shape[1]):
    #         add_arrow = (ego_multimodal[batch_sample,agent_i,i,-1]-ego_multimodal[batch_sample,agent_i,i,0]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
    #         fig = vizualize_traj_arrow(ego_multimodal[batch_sample,agent_i,i].cpu(), fig, ax, colors[i], 'solid', add_arrow)
    
    # GT EGO
    # object_type = int((inputs['ego_state'][batch_sample,-1, 8:].argmax(-1)+1)* inputs['ego_state'][batch_sample,-1, 8:].sum(-1))
    object_type = ""
    inputs['ego_state'] = inputs['ego_state'].cpu()
    rect(inputs['ego_state'][batch_sample,-1,0], inputs['ego_state'][batch_sample,-1,1], inputs['ego_state'][batch_sample,-1,5], inputs['ego_state'][batch_sample,-1,6], inputs['ego_state'][batch_sample,-1,2], object_type, color='purple', alpha=0.4, fontsize=10)
    add_arrow = (gt_ego_future[batch_sample,-1,:2]-gt_ego_future[batch_sample,0,:2]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
    fig = vizualize_traj_arrow(gt_ego_future[batch_sample,...,:2].cpu(), fig, ax, 'r', '--', add_arrow, 0.9)

    # GT Neighbor
    # object_type = int((inputs['ego_state'][batch_sample,-1, 8:].argmax(-1)+1)* inputs['ego_state'][batch_sample,-1, 8:].sum(-1))
    object_type = "Inter"
    inputs['neighbors_state'] = inputs['neighbors_state'].cpu()
    rect(inputs['neighbors_state'][:,0][batch_sample,-1,0], inputs['neighbors_state'][:,0][batch_sample,-1,1], inputs['neighbors_state'][:,0][batch_sample,-1,5], inputs['neighbors_state'][:,0][batch_sample,-1,6], inputs['neighbors_state'][:,0][batch_sample,-1,2], object_type, color='teal', alpha=0.4, fontsize=10)
    add_arrow = (inter_future[batch_sample,-1,:2]-inter_future[batch_sample,0,:2]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
    fig = vizualize_traj_arrow(inter_future[batch_sample,...,:2].cpu(), fig, ax, 'teal', '--', add_arrow, 0.9)
    
    colors = [['slateblue', 'g', 'y', 'violet', 'orange', 'indigo'], ['teal', 'teal', 'teal', 'teal', 'teal', 'teal']]
    for i in range(ego_multimodal.shape[2]):
        for agent_i in range(ego_multimodal.shape[1]):
            add_arrow = (ego_multimodal[batch_sample,agent_i,i,-1]-ego_multimodal[batch_sample,agent_i,i,0]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
            fig = vizualize_traj_arrow(ego_multimodal[batch_sample,agent_i,i].cpu(), fig, ax, colors[agent_i][i], 'solid', add_arrow)
    # plt.savefig('ex.png')
    plt.close()
    return fig

# Visualize
def viz_multimodal(inputs, ego_multimodal, gt_ego_future, batch_sample = 0):
    fig, ax = vizualize(
        inputs['ego_state'][:,:,:2].cpu(), 
        None, 
        neighbors=inputs['neighbors_state'][batch_sample,:3].cpu(), 
        map_lanes=inputs['map_lanes'][batch_sample][:,:, :80:2].cpu(), 
        map_crosswalks=inputs['map_crosswalks'][batch_sample][:,:, :100:2].cpu(), 
        region_dict=None, 
        gt=False, 
        fig=None,
        ax=None,
        background_only=True)

    if ego_multimodal is not None:
        colors = ['b', 'g', 'y', 'purple', 'orange', 'cyan']
        for i in range(ego_multimodal.shape[1]):
            add_arrow = (ego_multimodal[batch_sample,i,-1]-ego_multimodal[batch_sample,i,0]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
            fig = vizualize_traj_arrow(ego_multimodal[batch_sample,i].cpu(), fig, ax, colors[i], 'solid', add_arrow)
    
    # GT
    object_type = int((inputs['ego_state'][batch_sample,-1, 8:].argmax(-1)+1)* inputs['ego_state'][batch_sample,-1, 8:].sum(-1))
    inputs['ego_state'] = inputs['ego_state'].cpu()
    rect(inputs['ego_state'][batch_sample,-1,0], inputs['ego_state'][batch_sample,-1,1], inputs['ego_state'][batch_sample,-1,5], inputs['ego_state'][batch_sample,-1,6], inputs['ego_state'][batch_sample,-1,2], 'Ego', color='r', alpha=0.4, fontsize=7)
    add_arrow = (gt_ego_future[batch_sample,-1,:2]-gt_ego_future[batch_sample,0,:2]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
    fig = vizualize_traj_arrow(gt_ego_future[batch_sample,...,:2].cpu(), fig, ax, 'r', '--', add_arrow, 0.9)
    
    # plt.savefig('ex.png')
    # plt.close()
    return fig
    

def vizualize_traj_arrow(traj, fig, ax, color, linestyle, add_arrow, alpha=0.5): # single agent, single modality
    plt.plot(traj[:,0], traj[:,1], linestyle=linestyle, color=color, alpha=alpha) 
    if add_arrow:
        ax.annotate('', xy=traj[-1], xytext=traj[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, linestyle='solid'))        
    return fig

def rect(x_pos, y_pos, width, height, heading, object_type, color='r', alpha=0.4, fontsize=10):
    rect_x = x_pos - width / 2
    rect_y = y_pos - height / 2
    rect = plt.Rectangle((rect_x, rect_y), width, height, linewidth=1, color=color, alpha=alpha, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_pos, y_pos), heading) + plt.gca().transData)
    plt.gca().add_patch(rect)
    plt.text(x_pos, y_pos, object_type, color='black', fontsize=fontsize, ha='center', va='center')




def vizualize(ego, ground_truth, neighbors=None, map_lanes=None, map_crosswalks=None, region_dict=None, gt=False, fig=None, background_only=False, colors_mode=1, ax=None):
    # visulization
    if not fig is not None:
        fig, ax = plt.subplots(dpi=300)
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
                rect(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type='', color='lightgray', alpha=0.5, fontsize=5)
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
    # plt.save('ex.png')
    # plt.show()
    # plt.close()
    return fig, ax