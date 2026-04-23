"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
import torch

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[],
        obs_pred_split_type = None, obs_len = None, pred_len = None,
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        annotation is data in the context of trajectories
        """
        ##TODO: Clean the following
        # data_info_path = ann_paths[0][:ann_paths[0].find(ann_paths[0].split('/')[-1])] + "data_info.pt" # This gets the storage path (removes train.json or val.json)
        # data_info = torch.load(data_info_path)
        # self.rel_min= data_info['rel_min']
        # self.traj_max= data_info['rel_cont_grid']
        # self.rel_min= data_info['rel_min']
        # self.rel_max= data_info['rel_max']
        # self.seq_len= data_info['seq_len']
        # self.grid_size= data_info['grid_size']
        # self.traj_cont_grid= data_info['traj_cont_grid']
        # self.rel_contr_grid= data_info['rel_cont_grid']
        ##end: Clean the following
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))['data'])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.obs_pred_split_type = obs_pred_split_type
        self.obs_len = obs_len
        self.pred_len = pred_len

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)