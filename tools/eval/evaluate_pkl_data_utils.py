import torch
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from instructions.extract_instructions import futureNavigation
from gameformer.utils.data_utils import *

class get_act_util:
    def get_act(self, history, future, skip_future_start_end_with_zeros=True, not_skip_invalid_history=True, agent_angle=None):
        """ Extracts navigation actions based on history and future trajectories. """
        navigation_extractor = futureNavigation(num_classes=5)
        
        if len(history.shape)==2:
            history = history[None][None]
            future = future[None][None]
        # Ensure ego, ground_truth, and neighbors are valid and interpolated
        ego, ground_truth, neighbors, not_valid_future = self.interpolate_missing_data(history[0], future[0], history[1:], skip_future_start_end_with_zeros=skip_future_start_end_with_zeros, not_skip_invalid_history=not_skip_invalid_history)
        
        if not_valid_future:
            return None  # Return None if the data is invalid

        agent_json = self.gen_instruct_caption_01(
            history=np.vstack((deepcopy(ego), deepcopy(neighbors))) if neighbors.shape[0]!=0 else deepcopy(ego),
            future=deepcopy(ground_truth),
            navigation_extractor=navigation_extractor,
            vizualize=False,  # Set visualization flag as needed
            viz_dir='ex.png',
            not_skip_invalid_history=not_skip_invalid_history,
            agent_angle=agent_angle
        )
        return agent_json

    def get_agent_caption(self, agent_name, history, future, navigation_extractor, agent_angle=None):
        """ Generates a structured instruction set for an agent based on its trajectory history and future. """
        history_normalized_view = history.copy()
        valid_mask = history[:, 0] != 0
        if not True in valid_mask:
            valid_mask = ~valid_mask
        agent_center, agent_angle = history[valid_mask].copy()[0, :2], history[valid_mask].copy()[0, 2]

        try:
            # Normalize history
            history_normalized_view[valid_mask, :5] = agent_norm(history.copy()[valid_mask, :], agent_center, agent_angle, impute=True)
            history_instructs = navigation_extractor.get_navigation_dict_history(torch.tensor(history_normalized_view[:, :5]))
        except:
            history_instructs = {}

        if future is not None:
            future_normalized_view = future.copy()
            if future.shape[1]>2:
                agent_center, agent_angle = future.copy()[0, :2], future.copy()[0, 2]
                valid_mask = future[:, 0] != 0
                future_normalized_view[valid_mask, :5] = agent_norm(future.copy()[valid_mask, :], agent_center, agent_angle, impute=True)
                future_instructs = navigation_extractor.get_navigation_dict(torch.tensor(future_normalized_view))
            else:
                agent_center = future.copy()[0, :2]
                agent_angle = agent_angle if agent_angle is not None else 0
                valid_mask = future[:, 0] != 0
                future_normalized_view[valid_mask, :5] = agent_norm(future.copy()[valid_mask, :], agent_center, agent_angle, impute=False)
                future_instructs = navigation_extractor.get_navigation_dict(torch.tensor(future_normalized_view))
            
        else:
            return history_instructs

        return {**history_instructs, **future_instructs}

    def gen_instruct_caption_01(self, history, future, navigation_extractor, vizualize=True, normalization_param=None, viz_dir=None, not_skip_invalid_history=True, agent_angle=None):
        """ Generates structured instructions for multiple agents based on trajectory data. """
        instruct_dict = {}

        for i in range(history.shape[0]):
            if (sum(history[i][:, 0] != 0) >= 2 and not_skip_invalid_history) or not not_skip_invalid_history:
                instruct_dict[f"Agent-{i+1}"] = self.get_agent_caption(
                    i, history=history[i], future=future[i] if i < 2 else None, navigation_extractor=navigation_extractor, agent_angle=agent_angle
                )

        # Visualization (optional)
        if vizualize:
            history_global_frame = history
            future_global_frame = future
            subsample_indices = np.linspace(0, 80 - 1, 8, dtype=int)
            fig = self.plt_scene((history_global_frame[0], future_global_frame[0, subsample_indices]), max_num_lanes=50)
            fig.savefig('ex.png')
            plt.close()

        return instruct_dict

    def interpolate_missing_data(self, ego, ground_truth, neighbors, test_data=False, skip_future_start_end_with_zeros=True, not_skip_invalid_history=True):
        """ Interpolates missing trajectory data and validates its completeness. """
        not_valid = False
        future_start_end_with_zeros = False
        threshold_1 = 5 if not test_data else 0
        threshold_2 = 50 if not test_data else 0

        # if sum(ego[0, :, 0] == 0) > threshold_1 or sum(ego[1, :, 0] == 0) > threshold_1:
        if sum(ego[0, :, 0] == 0) > threshold_1 and not_skip_invalid_history:
            not_valid = True
        elif sum(ground_truth[0, :, 0] == 0) > threshold_2:
        # elif sum(ground_truth[0, :, 0] == 0) > threshold_2 or sum(ground_truth[1, :, 0] == 0) > threshold_2:
            not_valid = True
        else:
            for i in range(len(ego)):
                if np.any(ego[i, :, 0] == 0):
                    try:
                        start_non_zero = np.where(ego[i][:, 0] != 0)[0][0]
                        end_non_zero = np.where(ego[i][:, 0] != 0)[0][-1]
                        ego[i, start_non_zero:end_non_zero] = self.interpolate_missing_traj(ego[i, start_non_zero:end_non_zero])
                    except:
                        pass

                if np.any(ground_truth[i, :, 0] == 0):
                    start_non_zero = np.where(ground_truth[i][:, 0] != 0)[0][0]
                    end_non_zero = np.where(ground_truth[i][:, 0] != 0)[0][-1]
                    if start_non_zero > 0 or end_non_zero < len(ground_truth[i]) - 1:
                        future_start_end_with_zeros = True
                    ground_truth[i, start_non_zero:end_non_zero] = self.interpolate_missing_traj(ground_truth[i, start_non_zero:end_non_zero])

            for i in range(len(neighbors)):
                if np.any(neighbors[i, :, 0] == 0) and sum(neighbors[i, :, 0] != 0) > 2:
                    start_non_zero = np.where(neighbors[i][:, 0] != 0)[0][0]
                    end_non_zero = np.where(neighbors[i][:, 0] != 0)[0][-1]
                    neighbors_interpolated = self.interpolate_missing_traj(neighbors[i, start_non_zero:end_non_zero])
                    if len(neighbors_interpolated) == 0:
                        continue
                    neighbors[i, start_non_zero:end_non_zero] = neighbors_interpolated

        if future_start_end_with_zeros and skip_future_start_end_with_zeros:
            not_valid = True
        return ego, ground_truth, neighbors, not_valid

    def interpolate_missing_traj(self, traj):
        """ Interpolates missing trajectory points using linear interpolation. """
        missing_indices = np.where(traj[:, :2].sum(axis=-1) == 0)[0]
        if len(missing_indices) > traj.shape[0] / 2:
            return np.array([])

        if len(missing_indices) > 0:
            for i in range(2):  # Interpolate only x and y coordinates
                valid_indices = np.where(traj[:, i] != 0)[0]
                if len(valid_indices) == traj[:, i].shape[0]:
                    continue  # Skip if no missing values
                interpolator = interp1d(valid_indices, traj[valid_indices, i], fill_value="extrapolate")
                traj[missing_indices, i] = interpolator(missing_indices)

        return traj
