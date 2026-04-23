from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np

# Visualize
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
        more_boundaries= None,
        )


    if two_agents:
        agent_select=1
        object_type = int((inputs['ego_state'][agent_select,-1, 8:].argmax(-1)+1)* inputs['ego_state'][agent_select,-1, 8:].sum(-1))
        inputs['ego_state'] = inputs['ego_state'].cpu()
        object_type = '2'
        rect_(inputs['ego_state'][agent_select,-1,0], inputs['ego_state'][agent_select,-1,1], inputs['ego_state'][agent_select,-1,5], inputs['ego_state'][agent_select,-1,6], inputs['ego_state'][agent_select,-1,2], object_type, color='r', alpha=0.4, fontsize=10)
        # add_arrow = (gt_ego_future[agent_select,-1,:2]-gt_ego_future[agent_select,0,:2]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
        # add_arrow = False
        # fig = vizualize_traj_arrow(gt_ego_future[agent_select,...,:2].cpu(), fig, ax, 'k', '--', add_arrow, 0.9)

        agent_select=1
        colors = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray']
        for i in range(ego_multimodal.shape[1]):
            add_arrow = (ego_multimodal[agent_select,i,-1]-ego_multimodal[agent_select,i,0]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
            add_arrow = False
            fig = vizualize_traj_arrow(ego_multimodal[agent_select,i].cpu(), fig, ax, colors[i], 'solid', add_arrow)

     # GT
    agent_select=0
    object_type = int((inputs['ego_state'][agent_select,-1, 8:].argmax(-1)+1)* inputs['ego_state'][agent_select,-1, 8:].sum(-1))
    inputs['ego_state'] = inputs['ego_state'].cpu()
    object_type = 'Ego'
    rect_(inputs['ego_state'][agent_select,-1,0], inputs['ego_state'][agent_select,-1,1], inputs['ego_state'][agent_select,-1,5], inputs['ego_state'][agent_select,-1,6], inputs['ego_state'][agent_select,-1,2], object_type, color='r', alpha=0.4, fontsize=10)
    # add_arrow = (gt_ego_future[agent_select,-1,:2]-gt_ego_future[agent_select,0,:2]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
    # add_arrow = False
    # fig = vizualize_traj_arrow(gt_ego_future[agent_select,...,:2].cpu(), fig, ax, 'k', '--', add_arrow, 0.9)

    agent_select = 0
    colors = ['b', 'g', 'y', 'purple', 'orange', 'cyan']
    for i in range(ego_multimodal.shape[1]):
        add_arrow = (ego_multimodal[agent_select,i,-1]-ego_multimodal[agent_select,i,0]).norm().item() > 2 # if more than 2 meter travel, draw an arrow
        add_arrow = False
        fig = vizualize_traj_arrow(ego_multimodal[agent_select,i].cpu(), fig, ax, colors[i], 'solid', add_arrow)
    
    
    
   
    


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

def rect_(x_pos, y_pos, width, height, heading, object_type, color='r', alpha=0.4, fontsize=10):
    rect_x = x_pos - width / 2
    rect_y = y_pos - height / 2
    rect = plt.Rectangle((rect_x, rect_y), width, height, linewidth=1, color=color, alpha=alpha, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_pos, y_pos), heading) + plt.gca().transData)
    plt.gca().add_patch(rect)
    plt.text(x_pos, y_pos, object_type, color='black', fontsize=5, ha='center', va='center', alpha=0.8)




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
                rect_(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type, color=colors[i])
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
                # object_type = int((agent_i[-1, 8:].argmax(-1)+1)* agent_i[-1, 8:].sum(-1))
                object_type = ''
                rect_(x_pos=agent_i[-1, 0], y_pos=agent_i[-1, 1], width=agent_i[-1, 5], height=agent_i[-1, 6], heading=agent_i[-1, 2], object_type=object_type, color='gray', alpha=0.5, fontsize=5)
    if map_lanes is not None:
        agent_maps = range(map_lanes.shape[0]) if not map_of_ego_only else [0]
        for i in agent_maps:
            # lanes = map_lanes[:, :, :200:2][i]
            # crosswalks = map_crosswalks[:, :, :100:2][i]

            # lanes = map_lanes[:, :, :200:2][i]
            map_lanes = map_lanes.unsqueeze(0)
            lanes = map_lanes[i]
            # crosswalks = map_crosswalks[:, :, :100:2][i]
            

            
            for j in range(map_lanes.shape[1]):
            # for j in range([0]):
                lane = lanes[j]
                if lane[0][0] != 0:
                    centerline = lane[:, 0:2]
                    centerline = centerline[centerline[:, 0] != 0]
                    plt.plot(centerline[:, 0], centerline[:, 1], 'gray', linewidth=0.3, alpha=0.2) # plot centerline]
                    if lane.shape[-1]>2:
                        # roadlines
                        color_linestyle_per_type = [('k',None), ('w','dashed'), ('w','solid'), ('w','solid'), ('xkcd:yellow','dashed'), ('xkcd:yellow','dashed'), ('xkcd:yellow','solid'), ('xkcd:yellow','solid'), ('xkcd:yellow','dotted'), ('k-', None), ('k-', None), ('k-', None)]
                        
                        for lr_idx in [11,12]:
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
                                lane_boundary = lane[:, 3:5] if lr_idx==11 else lane[:, 6:8]
                                lane_boundary = lane_boundary[lane_type_["start"]:lane_type_["end"]]
                                lane_boundary = lane_boundary[lane_boundary[:, 0] != 0]
                                if lane_type_["type"] in [1,2,3,4,5,6,7,8]:
                                    plt.plot(lane_boundary[:, 0], lane_boundary[:, 1], color_linestyle_per_type[lane_type_["type"]][0], linestyle=color_linestyle_per_type[lane_type_["type"]][1], linewidth=1, alpha=0.3)
                                elif lane_type_["type"] in [9,10,11]:
                                    plt.plot(lane_boundary[:, 0], lane_boundary[:, 1], color_linestyle_per_type[lane_type_["type"]][0], linewidth=1)
                                else:
                                    plt.plot(lane_boundary[:, 0], lane_boundary[:, 1], color_linestyle_per_type[lane_type_["type"]][0], linewidth=0.3)

                        # left_lane_types = lane[:,11].unique().int()
                        # for k in range(lane.shape[0]):
                        #     left = lane[:, 3:5]
                        #     left = left[left[:, 0] != 0]
                        #     left_type = lane[:,11]
                        #     right = lane[:, 6:8]
                        #     right = right[right[:, 0] != 0]
                        #     right_type = lane[:,12]
                        #     # plt.scatter(centerline[:, 0], centerline[:, 1], c='k', alpha=0.1, s=4)
                        #     plt.plot(right[:, 0], right[:, 1], 'c', linewidth=1, alpha=0.6)
                        #     plt.plot(left[:, 0], left[:, 1], 'c', linewidth=1, alpha=0.6)
                            # plt.scatter(right[:, 0], right[:, 1], c='b', alpha=0.1, marker='o', s=4)
                            # plt.scatter(left[:, 0], left[:, 1], c='r', alpha=0.1, marker='o', s=4)
        
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


        map_crosswalks = map_crosswalks.unsqueeze(0)
        if map_crosswalks is not None:
            crosswalks = map_crosswalks[i]
            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk = crosswalk[crosswalk[:, 0] != 0]
                    plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'w', linewidth=1, alpha=0.4) # plot crosswalk
        
        


    
    
    # for i in range(region_dict[32].shape[0]):
    #     plt.scatter(region_dict[32][i,:,0],region_dict[32][i,:,1],marker='*',s=10)
    # plt.gca().set_aspect('equal')
    # plt.tight_layout()
    # plt.save('ex.png')
    # plt.show()
    # plt.close()
    return fig, ax