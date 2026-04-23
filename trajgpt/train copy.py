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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for debugging
import sys
sys.path.append(".")
sys.path.append("<legacy_repo_root>/")
from transformers.trainer_callback import TrainerCallback

# Monkey patching
def custom_on_pre_optimizer_step(self, args, state, control, **kwargs):
    # print("Custom logic before the optimizer step.")
    # Add your custom functionality here
    kwargs['model'].get_input_embeddings().weight.grad = kwargs['model'].get_input_embeddings().weight.grad*kwargs['model'].input_embed_grad_mask
    kwargs['model'].get_output_embeddings().weight.grad = kwargs['model'].get_output_embeddings().weight.grad*kwargs['model'].output_embed_grad_mask

# Monkey patch the method
TrainerCallback.on_pre_optimizer_step = custom_on_pre_optimizer_step


from huggingface_hub import login
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
if hf_token:
    login(token=hf_token)
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
    parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_dec/train/train_kv_caption_28dec.yaml', help="path to configuration file.")
    # parser.add_argument("--cfg-path", required=False, default='<legacy_repo_root>/trajgpt/train_configs_dec/eval/eval_base_27dec.yaml', help="path to configuration file.")
    
    
    
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


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    # setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    # Load the tokenizer and model
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # use_cache=True,
        # load_in_8bit=True,
        quantization_config=bnb_config,
        # attn_implementation="flash_attention_2",
        device_map={'':torch.cuda.current_device()},
    )
    instruction = "### Instruction:\nYou are an AI assistant. Please provide a detailed and accurate answer."
    question = "### Question:\nWhat is the time"
    answer_prompt = "### Answer:\n"
    input_text = f"{instruction}\n\n{question}\n\n{answer_prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to('cuda')
    output_ids = model.generate(
        input_ids.input_ids,
        max_length=100,  # Set maximum length of output
    )
    # Decode and print the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)

    llama_model = "meta-llama/Llama-2-7b-hf"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        # torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        # use_cache=True,
        # load_in_8bit=True,
        # quantization_config=bnb_config,
        # attn_implementation="flash_attention_2",
        device_map={'':torch.cuda.current_device()},
    )
    your_system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information."
    user_message_1 = "Hi, how are you?"
    dummy_input = f"<s>[INST] <<SYS>>\n{your_system_message}\n<</SYS>>\n\n{user_message_1} [/INST]"
    dummy_input =f"""<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    There's a llama in my garden 😱 What should I do? [/INST]"""
    # dummy_input = "<s>[INST] «SYS»\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.\n«/SYS»\n\nHi, how are you? [/INST]"
    # dummy_input = "Input: Hi, how are you? Output: "
    dummy_input = llama_tokenizer(dummy_input, return_tensors="pt", add_special_tokens=False)
    model_input_emb = model.get_input_embeddings()
    output = model.generate(dummy_input.input_ids.to('cuda'), max_new_tokens=70)
    print(llama_tokenizer.decode(output[0]))
    # dummy_input_embds = model.embed_tokens(torch.tensor(dummy_input.input_ids, device=self.model.embed_tokens.weight.device))
    # dummy_input_atts = dummy_input.attention_mask.to(dummy_input_embds.device)
    # model = Model(cfg)
    model = Model.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # use_cache=True,
        # load_in_8bit=True,
        quantization_config=bnb_config,
        # attn_implementation="flash_attention_2",
        device_map={'':torch.cuda.current_device()},
    )
    model.from_config(cfg.model_cfg)
    
    model.generate_dummy()
    # model = convert_to_bfloat16_exclude(model)

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
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
        # wandb.watch(model, log_freq=int(6*10))
        wandb.watch(model)
    # runner = get_runner_class(cfg)(
    #     cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    # )
    # runner.train()
    
    # training_args = TrainingArguments(
    #     output_dir="<internal_temp_root>/results",
    #     num_train_epochs=1,  # Equivalent to `epoch` loop
    #     per_device_train_batch_size=4,  # Match batch size
    #     learning_rate=1e-5,
    #     logging_dir="<internal_temp_root>/logs",
    #     logging_steps=10,  # Replace `log_freq`
    #     gradient_accumulation_steps=1,  # Replace `accum_grad_iters`
    #     bf16=True,  # Replace `scaler` for mixed precision
    #     # save_strategy="epoch",  # Save checkpoint per epoch
    #     # evaluation_strategy="epoch",  # Optional: evaluate per epoch
    #     report_to="none",  # or "wandb"/"tensorboard" for logging
    #     remove_unused_columns=False,
    #     fp16=False,
    #     log_on_each_node=False,
    #     evaluation_strategy="no",
    #     save_strategy="steps",
    #     warmup_ratio=0.03,
    #     lr_scheduler_type="cosine",
    #     tf32=False,
    #     dataloader_num_workers=0,
    #     # freeze_backbone=False,
    #     # fsdp="full_shard auto_wrap",
    #     # fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer',
    #     # gradient_checkpointing=True,
    # )
    # trainer = CustomTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=datasets['traj_align']['train'],
    #     data_collator= general_data_collator,
    #     # eval_dataset=eval_dataset,
    #     # tokenizer=tokenizer,
    # )
    # trainer.train()

    target_modules = [
                    # 'embed_tokens', 
                    'q_proj', 
                    'k_proj', 
                    'v_proj', 
                    'o_proj', 
                    # 'rotatry_emb', 
                    'gate_proj', 
                    'up_proj', 
                    'down_proj', 
                    # 'lm_head'
                    ] 
    loraconfig = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=target_modules,
                    lora_dropout=0.05,
                    modules_to_save=['lm_head','embed_tokens'],
                )
    # include non-llm modules:
    # loraconfig.modules_to_save = list(set(loraconfig.modules_to_save).union({name for name, child in model.named_children() if name != "model"}))
    non_llm_trainable_modules = [
    '.'.join(name.split('.')[:-1])
    for name, param in model.named_parameters()
    if param.requires_grad and name.split('.')[0] not in ['lm_head', 'model', 'embed_tokens']
    ]
    
    loraconfig.modules_to_save = loraconfig.modules_to_save + non_llm_trainable_modules
    sft_training_args = SFTConfig(output_dir="<internal_temp_root>3/")
    sft_training_args.remove_unused_columns = False
    sft_training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    sft_training_args.fp16=False
    sft_training_args.bf16 = True
    sft_training_args.tf32=False
    # sft_training_args.learning_rate = 1e-4 # note that projection and lora modules will have this lr, other modules will have lr*0.1
    # sft_training_args.learning_rate = 1e-5 # note that projection and lora modules will have this lr, other modules will have lr*0.1
    sft_training_args.learning_rate = 1e-3 # note that projection and lora modules will have this lr, other modules will have lr*0.1
    # sft_training_args.num_train_epochs=1
    sft_training_args.num_train_epochs=1000
    
    sft_training_args.do_train=True
    sft_training_args.gradient_accumulation_steps=4
    sft_training_args.lr_scheduler_type="cosine"
    # sft_training_args.warmup_ratio = 0.03
    sft_training_args.warmup_ratio = 0.0
    # sft_training_args.logging_steps = 10
    sft_training_args.logging_steps = 1
    # sft_training_args.save_steps = 1000
    sft_training_args.eval_strategy = "no"
    sft_training_args.save_strategy = "steps"
    sft_training_args.log_on_each_node = False
    # sft_training_args.save_total_limit = 1
    # sft_training_args.per_device_train_batch_size = 8
    # sft_training_args.per_device_train_batch_size = 16
    sft_training_args.per_device_train_batch_size = 1
    # sft_training_args.per_device_train_batch_size = 20
    # for k,v in sft_training_args.__dict__.items():
    #     print(f"{k}: {v}")
    
    # sft_training_args.save_steps=500
    sft_training_args.save_steps=100
    # sft_training_args.resume_from_checkpoint= "<internal_temp_root>/checkpoint-3000"
    sft_training_args.resume_from_checkpoint= False


    trainer = imotion_sft_trainer.CustomSFTTrainer(
        model,
        train_dataset=datasets['traj_align']['train'],
        # train_dataset=None,
        args=sft_training_args,
        peft_config=loraconfig,
        data_collator=general_data_collator,
        non_llm_trainable_modules=non_llm_trainable_modules,
    )
    
    trainer.train(resume_from_checkpoint=sft_training_args.resume_from_checkpoint) 
    # trainer.train()
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
