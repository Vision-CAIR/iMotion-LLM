import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
# from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import BitsAndBytesConfig
from transformers import LlamaForCausalLM
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer
# from minigpt4.models.modeling_llama import LlamaForCausalLM
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
# from transformers import BitsAndBytesConfig

import re
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from transformers import StoppingCriteria, StoppingCriteriaList
from minigpt4.tasks.image_text_pretrain import StoppingCriteriaSub

from traj_utils import *

from sklearn.preprocessing import KBinsDiscretizer
import pickle

@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        grid_size=100,
        num_agents=2,
        data_info_path="",
        freeze_embed_tokens=False,
        lora=False,
        space_token_id = 259,
        xy_identifier = False,
        point_identifier = False,
        step_identifier = False,
        no_new_vocab=True,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        # discretizer_model_path = '/home/felembaa/projects/trajgpt/KBinsDiscretizer_76.pkl',
        discretizer_model_path = './KBinsDiscretizer_76.pkl',
        lora_target_modules = 'all',
    ):
        super().__init__()

        with open(discretizer_model_path, 'rb') as file:
            self.discretizer = pickle.load(file)
         
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
       
        print('Loading LLAMA tokenizer')
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, trust_remote_code=True)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        self.lora = lora
        if self.lora:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            # bnb_config = BitsAndBytesConfig(
            #         load_in_4bit=True,
            #         bnb_4bit_quant_type="nf4",
            #         bnb_4bit_compute_dtype=torch.float16,
            #     )
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                quantization_config=bnb_config,
                # load_in_8bit=True,
                # torch_dtype=torch.float16,
                # trust_remote_code=True,
                # use_auth_token=True,
                # device_map="auto",
                # low_cpu_mem_usage=True
                device_map={'':torch.cuda.current_device()}
            )
            self.llama_model.config.use_cache = False
            self.llama_model.config.pretraining_tp = 1
            # self.llama_model = LlamaForCausalLM.from_pretrained(
            #     llama_model,
            #     load_in_8bit=True,
            #     low_cpu_mem_usage=True,
            #     quantization_config=bnb_config,
            #     # device_map="auto",
            #     # low_cpu_mem_usage=True
            #     device_map={'':torch.cuda.current_device()}
            # )
        elif self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                # torch_dtype=torch.float16,
                load_in_8bit=True,
                # device_map={'': device_8bit}
                # device_map="auto",
                low_cpu_mem_usage=True,
                device_map={'':torch.cuda.current_device()}
            )
            # self.llama_model = LlamaForCausalLM.from_pretrained(
            #     llama_model,
            #     torch_dtype=torch.float16,
            #     load_in_8bit=True,
            #     # device_map={'': device_8bit}
            #     # device_map="auto"
            #     device_map={'':torch.cuda.current_device()}
            # )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                device_map={'':torch.cuda.current_device()}
            )

        # self.llama_model.config.use_cache = True
        # self.llama_model.config.pretraining_tp = 1
        
        # for name, param in self.llama_model.named_parameters():
        #         param.requires_grad = False                
        #         if param.ndim == 1:
        #             param.data = param.data.to(torch.float32)


        self.llama_model.gradient_checkpointing_enable()
        # self.llama_model.enable_input_require_grads()
        self.llama_model = prepare_model_for_kbit_training(self.llama_model)

        # for name, param in self.llama_model.named_parameters():
        #     param.requires_grad = False
        print('Loading LLAMA Done')


        if self.lora:
                ### PEFT SETUP
                # self.llama_model.gradient_checkpointing_enable()
                target_modules = [
                    'embed_tokens', 
                    'q_proj', 
                    'k_proj', 
                    'v_proj', 
                    'o_proj', 
                    'rotatry_emb', 
                    'gate_proj', 
                    'up_proj', 
                    'down_proj', 
                    'lm_head'] if lora_target_modules=='all' else ["q_proj",
                    "up_proj",
                    "o_proj",
                    "k_proj",
                    "down_proj",
                    "gate_proj",
                    "v_proj"]
                # target_modules=["q_proj",
                #     "up_proj",
                #     "o_proj",
                #     "k_proj",
                #     "down_proj",
                #     "gate_proj",
                #     "v_proj"] if lora_target_modules=='all' else ["q_proj", "v_proj"] # old
                lconfig = LoraConfig(
                    # inference_mode=False,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    # target_modules=["q_proj", "v_proj"],
                    lora_dropout=lora_dropout,
                    bias="lora_only",
                    task_type="CAUSAL_LM",
                    )
                self.llama_model = get_peft_model(self.llama_model, lconfig)
                
                self.print_trainable_parameters(self.llama_model)

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id
        self.llama_model.generation_config.pad_token_id = self.llama_model.config.pad_token_id

        self.print_trainable_parameters(self.llama_model)
        
        # if 'Llama-2' in llama_model:
        self.llama2=True
        # with open('/home/felembaa/projects/TrajGPT-1/prompts/input_8.txt', 'r') as f:
        #     input_ = f.read()
        # prompt_template = '[INST] <<SYS>>\nYou are a helpful motion prediction assistant. Your answers should be abstract and accurate. Please ensure that your answer is based on all of the provided initial positions and motion tokens of all the interactive agents.\n<</SYS>>\n\n Predict the next 16 motion tokens of both agents <in_data> [/INST] '
        # prompt_template = '[INST] <<SYS>>\nYou are a helpful motion prediction assistant. Your answers should be abstract and accurate. Please ensure that your answer is based on all of the provided initial positions and motion tokens of all the interactive agents.\n<</SYS>>\n\n Predict the steps from 0 to 18 for both agents <Agent A> and <Agent B>'# You have to add '[/INST] 'to the end of this or what is after it
        # prompt_template = '[INST] <<SYS>>\nYou are a helpful motion prediction assistant. Your answers should be abstract and accurate. Please ensure that your answer is based on all of the provided initial positions and motion tokens of both agents. The motion will be encoded as <t,x,y,move x,move y>, where t is the step, x is the horizontal position, y is the vertical position, move x is the shift of the horizontal position in the next step, and move y is the shift of the vertical position in the next step. All numbers will be represented as integers.\n<</SYS>>\n\n Predict the next 16 steps motion for both interactive agents <Agent A> and <Agent B> based on the motion of the first 3 steps'
        # prompt_template = '[INST] <<SYS>>\nYou are a motion prediction assistant. Provide abstract and accurate answers based on initial positions and motion tokens for both agents. Motion is encoded as <t,x,y,move x,move y>, with t as the step, x and y as positions, and move x and move y as position shifts for the next step. All numbers are integers.\n<</SYS>>\n\n Predict the next 16 steps for both interactive agents <Agent A> and <Agent B> based on the motion of the first 3 steps.'
        # prompt_template = '[INST] <<SYS>>\nYou are a motion prediction assistant. Provide abstract and accurate answers based on the observed motion tokens of both agents. \n<</SYS>>\n\n Predict the next 16 tokens for both interactive agents <Agent A> and <Agent B> based on the motion tokens of the first 3 steps.'
        
        # prompt_template = '<s>[INST] Predict the next sixteen tokens for both interactive agents based on the provided three initial motion tokens.'
        # self.input_end_text = '[/INST]' # old

        prompt_template = '<s>[INST] Predict the next sixteen tokens for both interactive agents based on the provided three initial motion tokens.'
        self.input_end_text = ' [/INST] '


        # self.prompt_list = [prompt_template.format(input_)]
        self.prompt = prompt_template
        # else:
        #     self.prompt_list = []
        self.llama_tokenizer(self.end_sym, add_special_tokens=False).input_ids

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def forward(self, samples, _training=True):
        
        self.llama_tokenizer.padding_side = "right"
        
        device = samples['traj'].device
        batch_size = samples['traj'].shape[0]

        rel = abs_distance_to_velocity(abs_distance=samples['traj']).cpu()
        rel_disc = self.discretizer.transform(rel.reshape(-1,2))
        rel_disc = torch.tensor(pair_(rel_disc)).reshape(rel[...,0].shape)
        obs = get_waymo_parsed_short2_batch(rel_disc, 3)
        # obs = get_waymo_parsed_batch(samples['disc_traj'], samples['disc_rel'], 3)
        # obs = get_waymo_parsed_batch_short(samples['disc_traj'], samples['disc_rel'], 3)
        # obs = get_waymo_parsed_short2(samples['disc_rel'], 3)

        obs = [' ' + obs_+self.input_end_text for obs_ in obs]
        # obs = [obs_+self.input_end_text for obs_ in obs] # old
        prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
        obs_tokens = self.llama_tokenizer(obs, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
        
        if _training:
            # FOR TRAINING AND EVALUATION
            if not self.lora:
                prompt_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids)
                obs_embeds = self.llama_model.model.embed_tokens(obs_tokens.input_ids)    
            else:
                prompt_embeds = self.llama_model.model.model.embed_tokens(prompt_tokens.input_ids)
                obs_embeds = self.llama_model.model.model.embed_tokens(obs_tokens.input_ids)
            input_tokens = torch.cat((prompt_tokens.input_ids.expand(batch_size, -1), obs_tokens.input_ids), dim=1)
            input_atts = torch.cat((prompt_tokens.attention_mask.expand(batch_size, -1), obs_tokens.attention_mask), dim=1)
            input_embeds = torch.cat((prompt_embeds.expand(batch_size, -1, -1), obs_embeds), dim=1)
            # FOR TRAINING ONLY:
            # pred = get_waymo_parsed_batch(samples['disc_traj'], samples['disc_rel'], 19)
            # pred = get_waymo_parsed_batch_short(samples['disc_traj'], samples['disc_rel'], 19)
            pred = get_waymo_parsed_short2_batch(rel_disc, 19)
            pred = [pred_+' '+self.end_sym for pred_ in pred]
            # pred = [pred_+self.end_sym for pred_ in pred] # old
            pred_tokens = self.llama_tokenizer(pred, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
            if not self.lora:
                pred_embeds = self.llama_model.model.embed_tokens(pred_tokens.input_ids)
            else:
                pred_embeds = self.llama_model.model.model.embed_tokens(pred_tokens.input_ids)
            targets = pred_tokens.input_ids.masked_fill(
                pred_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            empty_targets = (
                torch.ones([input_atts.shape[0], input_atts.shape[1]],
                        dtype=torch.long).to(device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            
            embs = torch.cat([input_embeds, pred_embeds], dim=1)
            attention_mask = torch.cat([input_atts, pred_tokens.attention_mask], dim=1)
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=embs,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            logits = outputs.logits
            return {"loss": loss, "logits": logits}
        else:
            with torch.no_grad():
                # FOR TRAINING AND EVALUATION
                if not self.lora:
                    prompt_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids)
                    obs_embeds = self.llama_model.model.embed_tokens(obs_tokens.input_ids)    
                else:
                    prompt_embeds = self.llama_model.model.model.embed_tokens(prompt_tokens.input_ids)
                    obs_embeds = self.llama_model.model.model.embed_tokens(obs_tokens.input_ids)
                input_tokens = torch.cat((prompt_tokens.input_ids.expand(batch_size, -1), obs_tokens.input_ids), dim=1)
                input_atts = torch.cat((prompt_tokens.attention_mask.expand(batch_size, -1), obs_tokens.attention_mask), dim=1)
                input_embeds = torch.cat((prompt_embeds.expand(batch_size, -1, -1), obs_embeds), dim=1)
                embs = input_embeds
                attention_mask = input_atts
                stop_words_ids = [torch.tensor([self.llama_tokenizer.eos_token_id]).to(input_tokens.device)]
                # stop_words_ids = [torch.tensor([self.llama_tokenizer.eos_token_id]).to(input_tokens.device), 
                # torch.tensor([29966]).to(input_tokens.device)]
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

                max_new_tokens = self.max_txt_len
                with self.maybe_autocast():
                    outputs = self.llama_model.model.generate(
                        inputs_embeds=embs,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        stopping_criteria=stopping_criteria, #</motion> and </s>
                        # num_beams=num_beams,
                        # do_sample=do_sample,
                        # min_length=150,
                        # early_stopping=True,
                        # pad_token_id = self.llama_tokenizer.pad_token_id,
                        # bos_token_id = 529, #<
                        # force_words_ids= [[self.motion_start_token], [self.motion_end_token]], #<motion> </motion>
                        # force_words_ids = [[529, 29885, 8194, 29958], [1533, 29885, 8194, 29958]],
                        # use_cache = True,
                        # bad_words_ids = [[i] for i in list(self.original_vocab.values()) if i not in self.llama_tokenizer.all_special_ids + [self.space_token_id]], #Check if useful to use or not
                    )
                
                rel_reconstructed = torch.zeros((len(samples['traj']), 2, 19, 2))
                valid_rel = [False]*len(samples['traj'])
                # Post processing
                for j in range(len(samples['traj'])):
                    output_token = outputs[j]
                    if output_token[0] == 0 or output_token[0] == 1:  # the model might output a unknow token <unk> at the beginning. remove it. or <s>
                        output_token = output_token[1:]
                    if output_token[-1] == 2:
                        output_token = output_token[:-1]
                    if self.llama_model.config.eos_token_id in output_token:
                        output_token = output_token[:(output_token == self.llama_model.config.eos_token_id).nonzero()[0].item()]
                    # if 13 in output_token: # old
                    #     output_token = output_token[:(output_token == 13).nonzero()[0].item()]

                    start_token_1 = torch.tensor(self.llama_tokenizer('<Agent A>', add_special_tokens=False).input_ids).to(output_token.device)
                    start_token_2 = torch.tensor(self.llama_tokenizer('<Agent B>', add_special_tokens=False).input_ids).to(output_token.device)
                    end_token_1 = torch.tensor(self.llama_tokenizer('</Agent A>', add_special_tokens=False).input_ids).to(output_token.device)
                    end_token_2 = torch.tensor(self.llama_tokenizer('</Agent B>', add_special_tokens=False).input_ids).to(output_token.device)
                    start_token_i_1 = find_sublist(output_token, start_token_1) + len(start_token_1)
                    end_token_i_1 = find_sublist(output_token, end_token_1)
                    start_token_i_2 = find_sublist(output_token, start_token_2) + len(start_token_2)
                    end_token_i_2 = find_sublist(output_token, end_token_2)
                    if start_token_i_1 != -1 and end_token_i_1 != -1 and start_token_i_2 != -1 and end_token_i_2 != -1 and end_token_i_1>start_token_i_1 and end_token_i_2>start_token_i_2:
                        output_text_1 = self.llama_tokenizer.decode(output_token[start_token_i_1:end_token_i_1])
                        output_text_2 = self.llama_tokenizer.decode(output_token[start_token_i_2:end_token_i_2])
                        rel_disc_1 = [int(motion_i) for motion_i in output_text_1.strip('<>').split(',') if motion_i.isdigit()]
                        rel_disc_2 = [int(motion_i) for motion_i in output_text_2.strip('<>').split(',') if motion_i.isdigit()]
                    
                    # start_token_1 = torch.tensor(self.llama_tokenizer('{', add_special_tokens=False).input_ids).to(output_token.device) # old
                    # start_token_i_1 = find_sublist(output_token, start_token_1) + len(start_token_1)
                    # end_token_1 = torch.tensor([1118]).to(output_token.device)
                    # end_token_i_1 = find_sublist(output_token, end_token_1)
                    # output_token_2 = output_token[end_token_i_1+len(end_token_1):]
                    # start_token_2 = start_token_1
                    # start_token_i_2 = find_sublist(output_token_2, start_token_2) + len(start_token_2)
                    # end_token_2 = torch.tensor([29913]).to(output_token_2.device)
                    # end_token_i_2 = find_sublist(output_token_2, end_token_2)
                    # if start_token_i_1 != -1 and end_token_i_1 != -1 and start_token_i_2 != -1 and end_token_i_2 != -1 and end_token_i_1>start_token_i_1 and end_token_i_2>start_token_i_2:
                    #     output_token_ = output_token[start_token_i_1: end_token_i_1]
                    #     output_token_1 = output_token_
                    #     output_text_1 = self.llama_tokenizer.decode(output_token_1)
                    #     output_token_2_ = output_token_2[start_token_i_2:end_token_i_2]
                    #     output_text_2 = self.llama_tokenizer.decode(output_token_2_)
                    #     rel_disc_1 = [int(motion_i) for motion_i in output_text_1.split(',') if motion_i.isdigit()]
                    #     rel_disc_2 = [int(motion_i) for motion_i in output_text_2.split(',') if motion_i.isdigit()] # old

                        try:
                            rel_disc_1 = depair_(rel_disc_1)
                            rel_disc_2 = depair_(rel_disc_2)
                            rel_reconstructed_1 = torch.tensor(self.discretizer.inverse_transform(rel_disc_1))
                            rel_reconstructed_2 = torch.tensor(self.discretizer.inverse_transform(rel_disc_2))
                            if rel_reconstructed_1.shape == rel_reconstructed_2.shape:
                                rel_reconstructed[j] = torch.cat((rel_reconstructed_1.unsqueeze(0),rel_reconstructed_2.unsqueeze(0)), dim=0)
                                valid_rel[j] = True
                        except Exception as e:
                            continue
                return rel_reconstructed, valid_rel
                # return outputs



    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        lora = cfg.get("lora", False)
        grid_size = cfg.get("grid_size", 100)
        num_agents = cfg.get("num_agents", 2)
        data_info_path = cfg.get("data_info_path", "")
        freeze_embed_tokens = cfg.get("freeze_embed_tokens", True)

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        lora_dropout = cfg.get("lora_dropout", 0.05)

        lora_target_modules = cfg.get("lora_target_modules", 'all')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            grid_size=grid_size,
            num_agents=num_agents,
            data_info_path=data_info_path,
            freeze_embed_tokens=freeze_embed_tokens,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules = lora_target_modules,
            discretizer_model_path = cfg.get("discretizer", './KBinsDiscretizer_76.pkl')
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model


