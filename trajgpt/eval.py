"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# torchrun --nproc_per_node 2 <legacy_repo_root>/trajgpt/train.py
# torchrun --nproc-per-node 4 <legacy_trajgpt_repo>/train.py --cfg-path 
#torchrun --nproc-per-node 2 <legacy_trajgpt_repo>/train.py --cfg-path <legacy_trajgpt_repo>/train_configs/ibex.yaml
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imotion_paths import env_or_default

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for debugging
os.environ.setdefault("HF_HOME", str(env_or_default("HF_HOME", REPO_ROOT / ".cache" / "huggingface")))
sys.path.append(".")
# from huggingface_hub import login
# Set HUGGINGFACE_HUB_TOKEN in the environment if gated model access is required.
import argparse
import random
import wandb

import numpy as np
import torch
# Clear GPU memory cache
torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
# from transformers import TrainingArguments, Trainer
from transformers import Trainer
import transformers

from torch.utils.data.dataloader import default_collate
from typing import Dict, List, Optional
from torch.utils.data import DataLoader, Sampler
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, has_length
from typing import Dict, List, Sequence
from dataclasses import dataclass, field
import bitsandbytes
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from minigpt4 import imotion_sft_trainer
from minigpt4.models.imotion_llm import IMotionLLMModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.nn.parallel.scatter_gather import Scatter, _is_namedtuple
try:
    from mtr.utils import common_utils
    from minigpt4.models.imotion_llm import IMotionLLMMTRModel
    print("✅ mtr module loaded successfully.")
except ImportError:
    print("❌ mtr model is not supported (module not found).")

def imotion_mtr_collator(batch_list, convert_to_bfloat16=False):
    """
    Combines general_data_collator and collate_batch_mtr into a unified function.
    
    Args:
        batch_list (list of dict): List of samples, where each sample is a dictionary.
        convert_to_bfloat16 (bool): Whether to convert float32 tensors to bfloat16.

    Returns:
        dict: A batch with all keys collated.
    """
    batch_size = len(batch_list)
    key_to_list = {}

    # Step 1: Extract all keys from batch_list
    for key in batch_list[0].keys():
        key_to_list[key] = [batch_list[i][key] for i in range(batch_size)]

    # Step 2: Initialize final batch dictionary
    input_dict = {}

    # Step 3: Process each key in key_to_list
    for key, values in key_to_list.items():
            
        # Case 1: Keys requiring padding (MTR-specific)
        if key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask', 'map_polylines_center',
                'obj_trajs_pos', 'obj_trajs_last_pos', 'obj_trajs_future_state', 'obj_trajs_future_mask']:
            values = [torch.from_numpy(x) for x in values]
            input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(values)

        # Case 2: Concatenable Numpy arrays (MTR-specific)
        elif key in ['scenario_id', 'obj_types', 'obj_ids', 'center_objects_type', 'center_objects_id']:
            input_dict[key] = np.concatenate(values, axis=0)

        elif isinstance(values[0], (np.ndarray)):
            values = [torch.from_numpy(x) for x in values]
            input_dict[key] = torch.cat(values, dim=0)
        # Case 3: General tensor and array collation
        elif isinstance(values[0], (torch.Tensor, np.ndarray)):
            input_dict[key] = default_collate(values)
            
            # Convert tensors to bfloat16 if required
            if convert_to_bfloat16 and isinstance(input_dict[key], torch.Tensor):
                if input_dict[key].dtype == torch.float32:
                    input_dict[key] = input_dict[key].to(torch.bfloat16)

        # Case 4: Scalars (int, float)
        elif isinstance(values[0], (int, float)):
            input_dict[key] = torch.tensor(values)

        # Case 5: Strings (Keep as a list)
        elif isinstance(values[0], str):
            input_dict[key] = values

        # Case 6: Default fallback (Keep as list)
        else:
            input_dict[key] = values

    # Step 4: Add batch_size and batch_sample_count
    batch_sample_count = [len(x['track_index_to_predict']) for x in batch_list]

    # Step 5: Add scatter index if "caption" exists
    if "caption" in input_dict:
        input_dict["scatter_idx"] = torch.arange(len(input_dict["caption"]))
    
    batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_sample_count}
    return batch_dict


# Backward-compatible alias for preserved legacy imports.
mtr_imotion_collator = imotion_mtr_collator

def collate_batch_mtr(batch_list):
        """
        Args:
        batch_list:
            scenario_id: (num_center_objects)
            track_index_to_predict (num_center_objects):

            obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            obj_trajs_mask (num_center_objects, num_objects, num_timestamps):
            map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_polylines, num_points_each_polyline)

            obj_trajs_pos: (num_center_objects, num_objects, num_timestamps, 3)
            obj_trajs_last_pos: (num_center_objects, num_objects, 3)
            obj_types: (num_objects)
            obj_ids: (num_objects)

            center_objects_world: (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            center_objects_type: (num_center_objects)
            center_objects_id: (num_center_objects)

            obj_trajs_future_state (num_center_objects, num_objects, num_future_timestamps, 4): [x, y, vx, vy]
            obj_trajs_future_mask (num_center_objects, num_objects, num_future_timestamps):
            center_gt_trajs (num_center_objects, num_future_timestamps, 4): [x, y, vx, vy]
            center_gt_trajs_mask (num_center_objects, num_future_timestamps):
            center_gt_final_valid_idx (num_center_objects): the final valid timestamp in num_future_timestamps
        """
        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():

            if key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask', 'map_polylines_center',
                'obj_trajs_pos', 'obj_trajs_last_pos', 'obj_trajs_future_state', 'obj_trajs_future_mask']:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list)
            elif key in ['scenario_id', 'obj_types', 'obj_ids', 'center_objects_type', 'center_objects_id']:
                input_dict[key] = np.concatenate(val_list, axis=0)
            else:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = torch.cat(val_list, dim=0)

        batch_sample_count = [len(x['track_index_to_predict']) for x in batch_list]
        batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_sample_count}
        return batch_dict


def convert_to_bfloat16_exclude(model, exclude_name=['model', 'lm_head']):
    for name, module in model.named_children():
        if name not in exclude_name:
            module.to(torch.bfloat16)  # Convert this submodule and all its submodules
    return model

# def upcast_all_except_llm(model, torch_dtype):
#     for name, module in model.named_children():
#         if name != 'model':
#             print(f"Module: {name}, Type: {type(module)}")
#     for name, module in model.named_modules():
#         # if module.weight.dtype!=torch_dtype:
#         #     module.to(torch_dtype)
#         # if isinstance(module, LoraLayer) and module.weight.dtype!=torch_dtype:
#         #     module.to(torch_dtype)
#         # if "norm" in name and module.weight.dtype!=torch_dtype:
#         #     module.to(torch_dtype)
#         # # if "lm_head" in name or "embed_tokens" in name:
#         if hasattr(module, "weight") and module.weight.dtype!=torch_dtype:
#             # print(f'>>> {name}')
#             module.to(torch_dtype)
#         else:
#             print('')
#             print(f'xxx {name}')
#             print(module)

#     return model

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    tune_text_decoder: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_tower_lr: Optional[float] = None
    unfreeze_mm_image_decoder: bool = field(default=False)

    mm_vision_sampler_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    model_max_length: Optional[int] = field(default=8192)

    lora_enable: bool = True
    # lora_r: int = 16
    # lora_alpha: int = 4
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = 4
        self.world_size = 0
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)

def general_data_collator(batch, convert_to_bfloat16=False):
    """
    Collate a batch of samples with mixed data types dynamically.
    Handles tensors, arrays, lists, and strings. Optionally converts float32 tensors to bfloat16.

    Args:
        batch (list of dict): List of samples, where each sample is a dictionary.
        convert_to_bfloat16 (bool): Whether to convert float32 tensors to bfloat16.

    Returns:
        dict: A batch with all keys collated.
    """
    collated_batch = {}

    # Iterate over all keys in the first sample
    for key in batch[0]:
        # Extract all values corresponding to the key
        values = [sample[key] for sample in batch]

        # Collate based on the type of the values
        if isinstance(values[0], (torch.Tensor, np.ndarray)):
            # Use default collate for tensors and arrays
            collated_batch[key] = default_collate(values)
            
            # Convert tensors to bfloat16 if required
            if convert_to_bfloat16 and isinstance(collated_batch[key], torch.Tensor):
                if collated_batch[key].dtype == torch.float32:
                    collated_batch[key] = collated_batch[key].to(torch.bfloat16)

        elif isinstance(values[0], (int, float)):
            # Collate scalars into a tensor
            collated_batch[key] = torch.tensor(values)
        elif isinstance(values[0], str):
            # Keep strings as a list
            collated_batch[key] = values
        else:
            # Fallback: Keep as a list
            collated_batch[key] = values

    # torch.nn.parallel.scatter_gather.Scatter scatter tensors across gpus, but replicate non-tensors across gpus, the following can help retriving order of non-tensors in the model forward call.
    # useful for filename, caption, and instruct, as we pass them as string and we tokanize them in the forward pass.
    collated_batch["scatter_idx"] = torch.arange(len(collated_batch["caption"]), device=collated_batch['traj'].device)
    return collated_batch


class CustomTrainer(Trainer):
    def __init__(
        self,
        train_dataloader: Optional[DataLoader] = None,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        self.train_dataloader = train_dataloader
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataloader is not None:
            print("Using sonic dataloader")
            # pyre-fixme[16]: `LLaVATrainer` has no attribute `accelerator`.
            return self.accelerator.prepare(self.train_dataloader)
        # pyre-fixme[16]: `Trainer` has no attribute `get_train_dataloader`.
        return super().get_train_dataloader()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return None
        # pyre-fixme[16]: `LLaVATrainer` has no attribute `train_dataset`.
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # pyre-fixme[16]: `LLaVATrainer` has no attribute `args`.
        # if self.args.group_by_modality_length:
        if True:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            # pyre-fixme[16]: `Trainer` has no attribute `_get_train_sampler`.
            return super()._get_train_sampler()

    def training_step(self, model, inputs, *args, **kwargs):
        # Debugging: Print the inputs to inspect the structure
        # print("Inputs to model:", inputs)
        
        # Forward pass; the model's forward method should return the loss
        loss = model(inputs)["loss"]
        return loss

# class CustomTrainer(Trainer):
#     def training_step(self, model, inputs):
#         return model(inputs)

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_new/eval_jul/e03_act_contrastive.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_new/eval_jul/e01gt1.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_new/eval_jul/e05/e05_0gt1.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_new/rebuttal/eval_m07_gt_fulldata_noInstruct.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_new/rebuttal/train_finetune_m07_nuplan_e2e.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_new/rebuttal/eval_m07_nuplan.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_new/rebuttal/train_small_LLM.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_new/rebuttal/eval_small_LLM.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_new/train_jul/t01_act_contrastive_2agent.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_aug/train_configs_24aug/train_small_gf_predOnly.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_aug/train_configs_24aug/train_small_cgf_predOnly_fewTokens4_cimotion.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_sep/train/sep09_gf_noDecoderHead_4tokens_smallLLM.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_sep/eval_22sep/imotion_2tokens/imotion_2tokens_pred.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_sep/sep30/imotion_2tokens_waymo/train_stage_2/imotion_2tokens_stage2.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_nov/train/train_lisa_like.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_dec/train/train_nuplan_complex_temp.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_dec/train/train_nuplan_complex_26dec.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_dec/eval/eval_base_27dec.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_cvpr_rebuttal/train_imotion08_complex_r8.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_dec/eval/eval_base_27dec.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_iccv_feb_2025/train_imotion_mtr.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_cvpr_rebuttal/train_imotion08_complex_r8.yaml', help="path to configuration file.")
    parser.add_argument(
        "--cfg-path",
        required=False,
        default=str(REPO_ROOT / "configs" / "release" / "imotion_waymo_eval.yaml"),
        help="path to configuration file.",
    )
    
    
    
    
    
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_dec/train/temp.yaml', help="path to configuration file.")
    
    
    
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_sep/eval/sep09_gf_clsHeadFinetune_4tokens_smallLLM.yaml', help="path to configuration file.")
    
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_sep/train/sep09_gf_encFinetune_4tokens_smallLLM.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls

def convert_cfg_to_wandb_cfg(input_dict):
    converted_dict = {}
    for key, value in input_dict.items():
        converted_dict[key] = {'value': value}
    return converted_dict


def parse_args_closed_loop(yaml_path):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=False, default=yaml_path, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def initialize_model_for_closed_loop(yaml_path, checkpoint_path):
    cfg = Config(parse_args_closed_loop(yaml_path))
    
    cfg.run_cfg.evaluate=True
    cfg.run_cfg.evaluate_only=True
    cfg.run_cfg.eval_dir = checkpoint_path
    cfg.run_cfg.output_dir = cfg.run_cfg.eval_dir
    cfg.run_cfg.evaluate = True
    cfg.run_cfg.evaluate_only = True
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    if cfg.model_cfg.get("mtr", False):
        model = IMotionLLMMTRModel(cfg)
    else:
        model = IMotionLLMModel(cfg)
    
    model.from_config(cfg.model_cfg)
    print(model.device)
    target_modules = [
                    'q_proj', 
                    'k_proj', 
                    'v_proj', 
                    'o_proj', 
                    'gate_proj', 
                    'up_proj', 
                    'down_proj', 
                    ] 
    loraconfig = LoraConfig(
                    r=cfg.run_cfg.r,
                    lora_alpha=cfg.run_cfg.lora_alpha,
                    # r=32,
                    # lora_alpha=64,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=target_modules,
                    # lora_dropout=0.05,
                    # lora_dropout=0.01,
                    lora_dropout=cfg.run_cfg.lora_dropout,
                    modules_to_save=['lm_head','embed_tokens'],
                )
    sft_training_args = SFTConfig(output_dir=cfg.run_cfg.output_dir)
    sft_training_args.remove_unused_columns = False
    sft_training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    # sft_training_args.fp16=False
    sft_training_args.bf16 = True
    # sft_training_args.tf32=False
    sft_training_args.learning_rate = cfg.run_cfg.init_lr # note that projection and lora modules will have this lr, other modules will have lr*0.1
    # sft_training_args.learning_rate = 1e-5 # note that projection and lora modules will have this lr, other modules will have lr*0.1
    # sft_training_args.num_train_epochs=1
    sft_training_args.num_train_epochs=cfg.run_cfg.epochs
    
    sft_training_args.do_train=True
    sft_training_args.gradient_accumulation_steps=cfg.run_cfg.gradient_accumulation_steps
    sft_training_args.lr_scheduler_type="cosine"
    sft_training_args.warmup_ratio = cfg.run_cfg.warmup_ratio
    sft_training_args.max_grad_norm = cfg.run_cfg.max_grad_norm
    sft_training_args.logging_steps = 1
    sft_training_args.eval_strategy = "no"
    sft_training_args.save_strategy = "steps"
    sft_training_args.log_on_each_node = False
    sft_training_args.per_device_train_batch_size = cfg.run_cfg.batch_size_train
    sft_training_args.gradient_checkpointing = cfg.run_cfg.gradient_checkpointing
    if sft_training_args.gradient_checkpointing:
        sft_training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    sft_training_args.save_steps=cfg.run_cfg.save_every
    sft_training_args.resume_from_checkpoint= False
    sft_training_args.weight_decay=cfg.run_cfg.weight_decay
    sft_training_args.resume_from_checkpoint= cfg.run_cfg.output_dir if 'checkpoint' in cfg.run_cfg.output_dir else True
    dataset_100 = datasets['traj_align_valid']['valid']

    trainer = imotion_sft_trainer.CustomSFTTrainer(
        model,
        train_dataset=dataset_100,
        args=sft_training_args,
        peft_config=loraconfig,
        data_collator=general_data_collator if not cfg.model_cfg.get("mtr", False) else imotion_mtr_collator,
        donot_prepare_model_for_kbit_training=True,
    )
    # Nussair 
    # import pdb; pdb.set_trace()
    # model = trainer.return_merged_model(resume_from_checkpoint=sft_training_args.resume_from_checkpoint)
    model = trainer.return_merged_model(resume_from_checkpoint=sft_training_args.resume_from_checkpoint)

    print('model loading done')
    return model


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())
    cfg.run_cfg.output_dir = cfg.run_cfg.eval_dir
    cfg.run_cfg.evaluate=True
    cfg.run_cfg.evaluate_only=True
    init_distributed_mode(cfg.run_cfg)

    # setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    # cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    
    if cfg.model_cfg.get("mtr", False):
        model = IMotionLLMMTRModel(cfg)
    else:
        model = IMotionLLMModel(cfg)
    model.from_config(cfg.model_cfg)
    print(model.device)
    # model = convert_to_bfloat16_exclude(model)

    
    # model = task.build_model(cfg)
    
    # model = Model.from_pretrained(
    #     cfg.model_cfg.llama_model,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     # device_map="auto",
    #     # quantization_config=bnb_config,
    #     use_cache=False,
    #     load_in_8bit=True, # results in 21.26% trainable parameters instead of 100% for Llama3-1B
    #     # attn_implementation="flash_attention_2", # use_flash_attention_2=True
    # )

    

    target_modules = [
                    'q_proj', 
                    'k_proj', 
                    'v_proj', 
                    'o_proj', 
                    'gate_proj', 
                    'up_proj', 
                    'down_proj', 
                    ] 
    loraconfig = LoraConfig(
                    r=cfg.run_cfg.r,
                    lora_alpha=cfg.run_cfg.lora_alpha,
                    # r=32,
                    # lora_alpha=64,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=target_modules,
                    # lora_dropout=0.05,
                    # lora_dropout=0.01,
                    lora_dropout=cfg.run_cfg.lora_dropout,
                    modules_to_save=['lm_head','embed_tokens'],
                )
    sft_training_args = SFTConfig(output_dir=cfg.run_cfg.output_dir)
    sft_training_args.remove_unused_columns = False
    sft_training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    # sft_training_args.fp16=False
    sft_training_args.bf16 = True
    # sft_training_args.tf32=False
    sft_training_args.learning_rate = cfg.run_cfg.init_lr # note that projection and lora modules will have this lr, other modules will have lr*0.1
    # sft_training_args.learning_rate = 1e-5 # note that projection and lora modules will have this lr, other modules will have lr*0.1
    # sft_training_args.num_train_epochs=1
    sft_training_args.num_train_epochs=cfg.run_cfg.epochs
    
    sft_training_args.do_train=True
    sft_training_args.gradient_accumulation_steps=cfg.run_cfg.gradient_accumulation_steps
    sft_training_args.lr_scheduler_type="cosine"
    sft_training_args.warmup_ratio = cfg.run_cfg.warmup_ratio
    sft_training_args.max_grad_norm = cfg.run_cfg.max_grad_norm
    sft_training_args.logging_steps = 1
    sft_training_args.eval_strategy = "no"
    sft_training_args.save_strategy = "steps"
    sft_training_args.log_on_each_node = False
    sft_training_args.per_device_train_batch_size = cfg.run_cfg.batch_size_train
    sft_training_args.gradient_checkpointing = cfg.run_cfg.gradient_checkpointing
    if sft_training_args.gradient_checkpointing:
        sft_training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    sft_training_args.save_steps=cfg.run_cfg.save_every
    sft_training_args.resume_from_checkpoint= False
    sft_training_args.weight_decay=cfg.run_cfg.weight_decay
    sft_training_args.resume_from_checkpoint= cfg.run_cfg.output_dir if 'checkpoint' in cfg.run_cfg.output_dir else True
   # Nussair 
    dataset_100 = datasets['traj_align_valid']['valid']
    dataset_100.data_list = dataset_100.data_list[:100]
   # import pdb; pdb.set_trace()

    trainer = imotion_sft_trainer.CustomSFTTrainer(
        model,
        train_dataset=dataset_100,
        args=sft_training_args,
        peft_config=loraconfig,
        data_collator=general_data_collator if not cfg.model_cfg.get("mtr", False) else imotion_mtr_collator,
        donot_prepare_model_for_kbit_training=True,
    )
    # Nussair 
    # import pdb; pdb.set_trace()
    model = trainer.return_merged_model(resume_from_checkpoint=sft_training_args.resume_from_checkpoint)
    # model = model._load_from_checkpoint(resume_from_checkpoint)
    # trainer.custom_eval(resume_from_checkpoint=sft_training_args.resume_from_checkpoint)
    # trainer.train(resume_from_checkpoint=sft_training_args.resume_from_checkpoint) 
    
    # trainer.train()
    
    import time
    start_time = time.time()
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds\n")

    print('')
    # trainer = SFTTrainer(
    # model,
    # train_dataset=datasets['traj_align']['train'],
    # args=SFTConfig(output_dir="<internal_temp_root>"),
    # peft_config=loraconfig,
    # )
    # peft_config = LoraConfig(
    # r=16,
    # lora_alpha=32,
    # lora_dropout=0.05,
    # target_modules="all-linear",
    # modules_to_save=["lm_head", "embed_token"],
    # task_type="CAUSAL_LM",
    # )
if __name__ == "__main__":
    main()
