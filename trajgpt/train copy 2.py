"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# torchrun --nproc_per_node 2 /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py
# torchrun --nproc-per-node 4 /home/felembaa/projects/trajgpt/train.py --cfg-path 
#torchrun --nproc-per-node 2 /home/felembaa/projects/trajgpt/train.py --cfg-path /home/felembaa/projects/trajgpt/train_configs/ibex.yaml
import os
os.environ['HF_HOME'] = '/ibex/project/c2253/cache/'

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for debugging
import sys
sys.path.append(".")
sys.path.append("/home/felembaa/projects/iMotion-LLM-ICLR/")
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint


try:
    from mtr.utils import common_utils
    from minigpt4.models.mini_gpt4_mtr_dev import MiniGPT4 as Model_mtr
    print("✅ mtr module loaded successfully.")
except ImportError:
    print("❌ mtr model is not supported (module not found).")



def mtr_imotion_collator(batch_list, convert_to_bfloat16=False):
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

# Monkey patching
def custom_on_pre_optimizer_step(self, args, state, control, **kwargs):
    # print("Custom logic before the optimizer step.")
    # Add your custom functionality here
    # return
    if kwargs['model'].use_embed_mask:
        with torch.no_grad():
            for n, p in kwargs['model'].llm.named_parameters():
                if p.grad is None:
                    continue
                if "lm_head" in n:
                    # break
                    # print(n)
                    # print(p.grad)
                    # try:
                    p.grad = p.grad*kwargs['model'].output_embed_grad_mask
                    # except:
                    #     p.grad = p.grad*self.output_embed_grad_mask.to(self.TRAINABLE_PRECISION_LLM)
                    assert sum(p.grad[-1]!=0)>0
                elif "embed_tokens" in n:
                    # print(n)
                    # print(p.grad)
                    # try:
                    p.grad = p.grad*kwargs['model'].input_embed_grad_mask
                    # except:
                    # p.grad = p.grad*self.input_embed_grad_mask.to(self.TRAINABLE_PRECISION_LLM)
                    assert sum(p.grad[-1]!=0)>0
    else:
        pass
    # kwargs['model'].llm.get_input_embeddings().weight.grad = kwargs['model'].llm.get_input_embeddings().weight.grad*kwargs['model'].input_embed_grad_mask
    # kwargs['model'].llm.get_output_embeddings().weight.grad = kwargs['model'].llm.get_output_embeddings().weight.grad*kwargs['model'].output_embed_grad_mask
    # kwargs['model'].get_input_embeddings().weight.grad = kwargs['model'].get_input_embeddings().weight.grad*kwargs['model'].input_embed_grad_mask
    # kwargs['model'].get_output_embeddings().weight.grad = kwargs['model'].get_output_embeddings().weight.grad*kwargs['model'].output_embed_grad_mask

# Monkey patch the method
TrainerCallback.on_pre_optimizer_step = custom_on_pre_optimizer_step


# from huggingface_hub import login
# login(token="<huggingface_token>")
# login(token="IBEX")
import argparse
import random
import wandb

import numpy as np
import torch
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
from minigpt4.models.mini_gpt4 import MiniGPT4 as Model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.nn.parallel.scatter_gather import Scatter, _is_namedtuple





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

    # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_iccv_feb_2025/train_imotion_02_q_complex.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_iccv_feb_2025/train_imotion_mtr.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_iccv_feb_2025/train_imotion_e2e_mtr.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train_configs_mar_2025/ablation_llms/llama_2_7b_debug.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/ablation_llms/nussair_llama_2_7b_contrastive.yaml', help="path to configuration file.")
    
    # Query Only
    # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/ablation_llms/1_nussair_llama_2_7b_contrastive_query_only.yaml', help="path to configuration file.")
    
    # 114 KVs and Query
    # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/ablation_llms/2_nussair_llama_2_7b_contrastive_114_kvs_query.yaml', help="path to configuration file.")

    # 34 KVs and Query
    # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/ablation_llms/3_nussair_llama_2_7b_contrastive_34_actors_query.yaml', help="path to configuration file.")

    # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/ablation_llms/4_nussair_llama_2_7b_contrastive_60_lanes_query.yaml', help="path to configuration file.")

   # parser.add_argument("--cfg-path", required=False, default='/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/ablation_llms/5_nussair_llama_2_7b_contrastive_20_crosswalks_query.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/ablation_llms/llama_2_7b_04_1_epoch_1e5_r32_a16.yaml", help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/ablation_llms/04_llama_2_7b_2_epoch_1e4_01_r32_a16_2_tokens_unfrozengf.yaml", help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/2_iccv_rebuttal_may_2025/complex/01_c_llama_7b.yaml", help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/1_nussair_train_configs_april_2025/may_experiments/56_Llama_2_7b_hf_1e5_1e4_1e4_none_r32_a16_2r_25_shared_linear.yaml", help="path to configuration file.")
    # 

    parser.add_argument("--cfg-path", required=False, default="/home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/3_wacv_rebuttal_15_sep_2025/01_c_llama_7b.yaml", help="path to configuration file.")
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


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())


    # cfg.pretty_print()

    task = tasks.setup_task(cfg)
    #import pdb; pdb.set_trace()

    datasets = task.build_datasets(cfg)

    if cfg.model_cfg.get("mtr", False):
        model = Model_mtr(cfg)
    else:
        model = Model(cfg)
    

    
    # model = Model.custom_from_pretrained(
    #     pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     # use_cache=True,
    #     # load_in_8bit=True,
    #     quantization_config=bnb_config,
    #     # attn_implementation="flash_attention_2",
    #     device_map={'':torch.cuda.current_device()},
    # )
    model.from_config(cfg.model_cfg)
    
    # model.generate_dummy()
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

    
    
    if cfg.run_cfg.wandb_log:
        wandb_cfg = {**convert_cfg_to_wandb_cfg(cfg.model_cfg), **convert_cfg_to_wandb_cfg(cfg.run_cfg)}
        wandb.login()
        wandb.init(project="trajgpt", name=cfg.run_cfg.job_name, config=wandb_cfg)
        wandb.watch(model)

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
    sft_training_args.logging_steps = 10
    sft_training_args.eval_strategy = "no"
    sft_training_args.save_strategy = "steps"
    sft_training_args.log_on_each_node = False
    sft_training_args.per_device_train_batch_size = cfg.run_cfg.batch_size_train
    sft_training_args.gradient_checkpointing = cfg.run_cfg.gradient_checkpointing
    if sft_training_args.gradient_checkpointing:
        sft_training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}
    sft_training_args.save_steps=cfg.run_cfg.save_every
    sft_training_args.resume_from_checkpoint= cfg.run_cfg.get("resume_from_checkpoint", False)
    try:
        resume_from_checkpoint = get_last_checkpoint(cfg.run_cfg.output_dir)
    except:
        resume_from_checkpoint = None
    if resume_from_checkpoint is None:
        sft_training_args.resume_from_checkpoint=False
    sft_training_args.weight_decay=cfg.run_cfg.weight_decay

    # init_distributed_mode(cfg.run_cfg)

    # # setup_seeds(cfg)

    # # set after init_distributed_mode() to only log on master.
    # setup_logger()
    trainer = imotion_sft_trainer.CustomSFTTrainer(
        model,
        train_dataset=datasets['traj_align']['train'],
        # train_dataset=None,
        args=sft_training_args,
        peft_config=loraconfig,
        data_collator=general_data_collator if not cfg.model_cfg.get("mtr", False) else mtr_imotion_collator,
        # pretraining=True,
        # non_llm_trainable_modules=non_llm_trainable_modules,
        donot_prepare_model_for_kbit_training= cfg.run_cfg.get("donot_prepare_model_for_kbit_training", True),
        scale_projection=cfg.run_cfg.get("scale_projection", False),
        scale_trajectory_modules_lr=cfg.run_cfg.get("scale_trajectory_modules_lr", 1.0),
    )
    
    if cfg.run_cfg.get("pretrained_llm_dir", None):
        trainer._load_from_checkpoint(cfg.run_cfg.get("pretrained_llm_dir", None)) # loading the model only, not
    

    
    trainer.train(resume_from_checkpoint=sft_training_args.resume_from_checkpoint)
    print('')

if __name__ == "__main__":
    main()
