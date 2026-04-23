import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
# from minigpt4.datasets.datasets.traj_dataset import TrajDataset, TrajAlignData0set
from minigpt4.datasets.datasets.traj_dataset import TrajAlignDataset




# @registry.register_builder("traj")
# class TrajBuilder(BaseDatasetBuilder):
#     train_dataset_cls = TrajDataset

#     DATASET_CONFIG_DICT = {"default": "configs/datasets/traj/defaults.yaml"}

#     def _download_ann(self):
#         pass

#     def _download_vis(self):
#         pass

#     def build(self):
#         self.build_processors()

#         build_info = self.config.build_info

#         datasets = dict()
#         split = "train"

#         # create datasets
#         # [NOTE] return inner_datasets (wds.DataPipeline)
#         dataset_cls = self.train_dataset_cls
#         datasets[split] = dataset_cls(
#             vis_processor=self.vis_processors[split],
#             text_processor=self.text_processors[split],
#             location=build_info.storage,
#         ).inner_dataset

#         return datasets

@registry.register_builder("traj_align")
class TrajBuilder(BaseDatasetBuilder):
    train_dataset_cls = TrajAlignDataset
    # eval_dataset_cls = TrajAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/traj/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        # storage_path = build_info.storage
        storage_path = self.config.dataset_path

        datasets = dict()

        # if not os.path.exists(storage_path):
        #     warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        if self.config.get("mtr", False):
            from minigpt4.datasets.datasets.traj_dataset import TrajAlignDataset_mtr
            dataset_cls = TrajAlignDataset_mtr
        datasets['train'] = dataset_cls(
            data_dir = storage_path,
            act = self.config['processor'][list(self.config['processor'].keys())[0]].get("act", False),
            act_json = self.config['processor'][list(self.config['processor'].keys())[0]].get("act_json", True),
            template_select = self.config['processor'][list(self.config['processor'].keys())[0]].get("template_select", 4),
            contrastive = self.config['processor'][list(self.config['processor'].keys())[0]].get("contrastive", False),
            train = self.config['processor'][list(self.config['processor'].keys())[0]].get("train", True),
            random_select=self.config['processor'][list(self.config['processor'].keys())[0]].get("random_select", False),
            two_agent=self.config['processor'][list(self.config['processor'].keys())[0]].get("two_agent", False),
            two_agent_mode=self.config['processor'][list(self.config['processor'].keys())[0]].get("two_agent_mode", False),
            stage_2=self.config['processor'][list(self.config['processor'].keys())[0]].get('stage_2', ''),
            random_drop=self.config['processor'][list(self.config['processor'].keys())[0]].get('random_drop', True),
            of_samples=self.config['processor'][list(self.config['processor'].keys())[0]].get('of_samples', False),
            nuplan_complex=self.config['processor'][list(self.config['processor'].keys())[0]].get('nuplan_complex', False),
            num_classes=self.config['processor'][list(self.config['processor'].keys())[0]].get('num_classes', 8),
            weighted_sampling=self.config['processor'][list(self.config['processor'].keys())[0]].get('weighted_sampling', False),
            debug=self.config['processor'][list(self.config['processor'].keys())[0]].get('debug', False),
            nuplan_direction=self.config['processor'][list(self.config['processor'].keys())[0]].get('nuplan_direction', False),
        )

        


        return datasets

@registry.register_builder("traj_align_valid")
class TrajBuilderValid(BaseDatasetBuilder):
    # train_dataset_cls = TrajAlignDataset
    eval_dataset_cls = TrajAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/traj/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        # storage_path = build_info.storage
        storage_path = self.config.dataset_path

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        dataset_cls = self.eval_dataset_cls
        if self.config.get("mtr", False):
            from minigpt4.datasets.datasets.traj_dataset import TrajAlignDataset_mtr
            dataset_cls = TrajAlignDataset_mtr
        datasets['valid'] = dataset_cls(
            data_dir = storage_path,
            act = self.config['processor'][list(self.config['processor'].keys())[0]].get("act", False),
            act_json = self.config['processor'][list(self.config['processor'].keys())[0]].get("act_json", True),
            template_select = self.config['processor'][list(self.config['processor'].keys())[0]].get("template_select", 4),
            contrastive = self.config['processor'][list(self.config['processor'].keys())[0]].get("contrastive", False),
            train = self.config['processor'][list(self.config['processor'].keys())[0]].get("train", False),
            random_select=self.config['processor'][list(self.config['processor'].keys())[0]].get("random_select", False),
            two_agent=self.config['processor'][list(self.config['processor'].keys())[0]].get("two_agent", False),
            positive_notgt=self.config['processor'][list(self.config['processor'].keys())[0]].get("positive_notgt", False),
            new_eval=self.config['processor'][list(self.config['processor'].keys())[0]].get("new_eval", False),
            new_eval_mode=self.config['processor'][list(self.config['processor'].keys())[0]].get("new_eval_mode", 'traj_pred'),
            agents_instructed=self.config['processor'][list(self.config['processor'].keys())[0]].get('agents_instructed', ''),
            nuplan_complex=self.config['processor'][list(self.config['processor'].keys())[0]].get('nuplan_complex', False),
            num_classes=self.config['processor'][list(self.config['processor'].keys())[0]].get('num_classes', 8),
            new_twoAgent_eval_instruct_select=self.config['processor'][list(self.config['processor'].keys())[0]].get('new_twoAgent_eval_instruct_select', None),
            # stage_2=self.config['processor'][list(self.config['processor'].keys())[0]].get('stage_2', ''),
        )

        # print(f"TEMPELATE SELECT: {self.config['processor'][list(self.config['processor'].keys())[0]].get("template_select", 4)}")


        return datasets