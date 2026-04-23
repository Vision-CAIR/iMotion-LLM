import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from transformers import LlamaTokenizer
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import BitsAndBytesConfig
# from transformers import LlamaForCausalLM
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
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

from minigpt4.models.gameformer_enc import * # we will use Encoder() from here
from minigpt4.models import GameFormer
from minigpt4.models.storygpt_layers import TextFcLayer
from gf_utils import *
# from GameFormer_modules import CrossTransformer

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
        # discretizer_model_path = '<legacy_trajgpt_repo>/KBinsDiscretizer_76.pkl',
        discretizer_model_path = './KBinsDiscretizer_76.pkl',
        lora_target_modules = 'all',
        lora_bias=False,
        quant = 8,
        gameformer_enc=False,
        ignore_motion=False,
        rollouts=32,
        gf_encoder_path = None,
        gf_encoder_layers = None,
        freeze_gf=True,
    ):
        super().__init__()
        
        self.rollouts = rollouts
        with open(discretizer_model_path, 'rb') as file:
            self.discretizer = pickle.load(file)
         
        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
       
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, trust_remote_code=True)
        # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.pad_token = '$$'
        
        self.lora = lora
        if low_resource:
            # bnb_config_8bit = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     llm_int8_has_fp16_weight=False,
            # )
            if quant==8:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                    # torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map={'':torch.cuda.current_device()},
                )
            elif quant==4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    # bnb_4bit_compute_dtype=torch.float16,
                )
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                    # torch_dtype=torch.float16,
                    device_map={'':torch.cuda.current_device()},
                    quantization_config=bnb_config,
                )
            else:
                raise("Not implemented")
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
        print('> Loading LLAMA Done')
        if self.lora:
            print('Loading LoRA')
            # self.llama_model = prepare_model_for_int8_training(self.llama_model)
            self.llama_model = prepare_model_for_kbit_training(self.llama_model)
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
            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                bias="lora_only" if lora_bias else "none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)
            print('> Loading LoRA Done')
            self.print_trainable_parameters(self.llama_model)
            
        
        # self.llama_model.enable_input_require_grads()
        self.llama_model.gradient_checkpointing_enable()
        self.llama_model.config.use_cache = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id
        self.llama_model.generation_config.pad_token_id = self.llama_model.config.pad_token_id
        self.llama2=True
        
        # self.llama_tokenizer(self.end_sym, add_special_tokens=False).input_ids

        if gameformer_enc:
            self.gameformer_enc = True
            print('loading GameFormer encoder')
            self.gameformer_encoder = GameformerEncoder(1, gf_encoder_layers)
            if gf_encoder_path is not None:
                print("Load gameformer Checkpoint: {}".format(gf_encoder_path))
                ckpt = torch.load(gf_encoder_path, map_location="cpu")
                msg = self.gameformer_encoder.load_state_dict(ckpt['model_states'], strict=False)
                # print(msg)
            #1 neighbors to predict, we might need 2 not 1 lets check later
            print('> loading GameFormer encoder Done')
            # self.gameformer_adapter = nn.Linear(256, 4096, dtype=torch.float32)
            # define the gameformer encoder layers here
            prompt_template = '<s>[INST] Predict the next sixteen tokens for both interactive agents based on the scene information and the provided three initial motion tokens. '
            self.input_end_text = ' [/INST] '
            self.prompt = prompt_template
            self.scene_start_txt = ['<Scene A>', '<Scene B>']
            self.scene_end_txt = ['</Scene A>', '</Scene B>']
            if ignore_motion:
                raise('not implemented')
            self.ignore_motion=ignore_motion

            print('gameformer trainable parameters:')

            
            if freeze_gf:
                for param_name, param in self.gameformer_encoder.named_parameters():
                    if 'adapter' in param_name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            self.print_trainable_parameters(self.gameformer_encoder)

        else:
            self.gameformer_enc = False
            prompt_template = '<s>[INST] Predict the next sixteen tokens for both interactive agents based on the provided three initial motion tokens.'
            self.input_end_text = ' [/INST] '
            self.prompt = prompt_template
            

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
            if self.gameformer_enc:
                start_scene_token = self.llama_tokenizer(self.scene_start_txt, return_tensors="pt", add_special_tokens=False).to(device)
                end_scene_token = self.llama_tokenizer(self.scene_end_txt, return_tensors="pt", add_special_tokens=False).to(device)
                with self.maybe_autocast(dtype=torch.float32):
                    gameformer_input = {key:value for key,value in samples.items() if key in ['ego_state', 'neighbors_state', 'ego_state', 'map_lanes', 'map_crosswalks']}
                    # with self.maybe_autocast():
                    context = self.gameformer_encoder(gameformer_input)
                    context_embs = context['encodings']
                    # context_embs = self.gameformer_adapter(context['encodings']) # [batch, 2 agents, 114 features, 4096 projected embeds dim]
                    context_atts = (~context['masks']).int() # assuming False in context['masks'] means no masking (1s)
                if not self.lora:
                    start_scene_embed = self.llama_model.model.embed_tokens(start_scene_token.input_ids)
                    end_scene_embed = self.llama_model.model.embed_tokens(end_scene_token.input_ids)
                    pad_embed = self.llama_model.model.embed_tokens(torch.tensor([self.llama_model.config.pad_token_id], device=device))
                else:
                    start_scene_embed = self.llama_model.model.model.embed_tokens(start_scene_token.input_ids)
                    end_scene_embed = self.llama_model.model.model.embed_tokens(end_scene_token.input_ids)
                    pad_embed = self.llama_model.model.model.embed_tokens(torch.tensor(self.llama_model.config.pad_token_id, device=device))
                context_embs[context['masks']] = pad_embed

            if not self.lora:
                prompt_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids)
                obs_embeds = self.llama_model.model.embed_tokens(obs_tokens.input_ids)    
            else:
                prompt_embeds = self.llama_model.model.model.embed_tokens(prompt_tokens.input_ids)
                obs_embeds = self.llama_model.model.model.embed_tokens(obs_tokens.input_ids)
            
            input_tokens = torch.cat((prompt_tokens.input_ids.expand(batch_size, -1), obs_tokens.input_ids), dim=1) # This is not used

            if self.gameformer_enc:
                input_atts = torch.cat((
                    prompt_tokens.attention_mask.expand(batch_size, -1), 
                    start_scene_token.attention_mask[0].expand(batch_size, -1),
                    context_atts[:,0],
                    end_scene_token.attention_mask[0].expand(batch_size, -1),
                    start_scene_token.attention_mask[1].expand(batch_size, -1),
                    context_atts[:,1],
                    end_scene_token.attention_mask[1].expand(batch_size, -1),
                    obs_tokens.attention_mask
                    ), dim=1) 

                input_embeds = torch.cat((
                    prompt_embeds.expand(batch_size, -1, -1),
                    start_scene_embed[0].expand(batch_size, -1, -1),
                    context_embs[:,0],
                    end_scene_embed[0].expand(batch_size, -1, -1),
                    start_scene_embed[1].expand(batch_size, -1, -1),
                    context_embs[:,1],
                    end_scene_embed[1].expand(batch_size, -1, -1),
                    obs_embeds
                    ), dim=1)
            else:
                input_atts = torch.cat((prompt_tokens.attention_mask.expand(batch_size, -1), obs_tokens.attention_mask), dim=1)
                input_embeds = torch.cat((prompt_embeds.expand(batch_size, -1, -1), obs_embeds), dim=1)
            # End FOR TRAINING AND EVALUATION
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
                if self.gameformer_enc:
                    start_scene_token = self.llama_tokenizer(self.scene_start_txt, return_tensors="pt", add_special_tokens=False).to(device)
                    end_scene_token = self.llama_tokenizer(self.scene_end_txt, return_tensors="pt", add_special_tokens=False).to(device)
                    with self.maybe_autocast(dtype=torch.float32):
                        gameformer_input = {key:value for key,value in samples.items() if key in ['ego_state', 'neighbors_state', 'ego_state', 'map_lanes', 'map_crosswalks']}
                        # with self.maybe_autocast():
                        context = self.gameformer_encoder(gameformer_input)
                        context_embs = context['encodings']
                        # context_embs = self.gameformer_adapter(context['encodings']) # [batch, 2 agents, 114 features, 4096 projected embeds dim]
                        context_atts = (~context['masks']).int() # assuming False in context['masks'] means no masking (1s)
                    if not self.lora:
                        start_scene_embed = self.llama_model.model.embed_tokens(start_scene_token.input_ids)
                        end_scene_embed = self.llama_model.model.embed_tokens(end_scene_token.input_ids)
                        pad_embed = self.llama_model.model.embed_tokens(torch.tensor([self.llama_model.config.pad_token_id], device=device))
                    else:
                        start_scene_embed = self.llama_model.model.model.embed_tokens(start_scene_token.input_ids)
                        end_scene_embed = self.llama_model.model.model.embed_tokens(end_scene_token.input_ids)
                        pad_embed = self.llama_model.model.model.embed_tokens(torch.tensor(self.llama_model.config.pad_token_id, device=device))
                    context_embs[context['masks']] = pad_embed

                if not self.lora:
                    prompt_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids)
                    obs_embeds = self.llama_model.model.embed_tokens(obs_tokens.input_ids)    
                else:
                    prompt_embeds = self.llama_model.model.model.embed_tokens(prompt_tokens.input_ids)
                    obs_embeds = self.llama_model.model.model.embed_tokens(obs_tokens.input_ids)
                
                input_tokens = torch.cat((prompt_tokens.input_ids.expand(batch_size, -1), obs_tokens.input_ids), dim=1) # This is not used

                if self.gameformer_enc:
                    input_atts = torch.cat((
                        prompt_tokens.attention_mask.expand(batch_size, -1), 
                        start_scene_token.attention_mask[0].expand(batch_size, -1),
                        context_atts[:,0],
                        end_scene_token.attention_mask[0].expand(batch_size, -1),
                        start_scene_token.attention_mask[1].expand(batch_size, -1),
                        context_atts[:,1],
                        end_scene_token.attention_mask[1].expand(batch_size, -1),
                        obs_tokens.attention_mask
                        ), dim=1) 

                    input_embeds = torch.cat((
                        prompt_embeds.expand(batch_size, -1, -1),
                        start_scene_embed[0].expand(batch_size, -1, -1),
                        context_embs[:,0],
                        end_scene_embed[0].expand(batch_size, -1, -1),
                        start_scene_embed[1].expand(batch_size, -1, -1),
                        context_embs[:,1],
                        end_scene_embed[1].expand(batch_size, -1, -1),
                        obs_embeds
                        ), dim=1)
                else:
                    input_atts = torch.cat((prompt_tokens.attention_mask.expand(batch_size, -1), obs_tokens.attention_mask), dim=1)
                    input_embeds = torch.cat((prompt_embeds.expand(batch_size, -1, -1), obs_embeds), dim=1)
                # End FOR TRAINING AND EVALUATION
                embs = input_embeds
                attention_mask = input_atts
                stop_words_ids = [torch.tensor([self.llama_tokenizer.eos_token_id]).to(prompt_tokens.input_ids.device)]
                # stop_words_ids = [torch.tensor([self.llama_tokenizer.eos_token_id]).to(input_tokens.device), 
                # torch.tensor([29966]).to(input_tokens.device)]
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

                max_new_tokens = self.max_txt_len

                with self.maybe_autocast():
                    # outputs = self.llama_model.model.generate(
                    #     inputs_embeds=embs,
                    #     attention_mask=attention_mask,
                    #     max_new_tokens=max_new_tokens,
                    #     stopping_criteria=stopping_criteria, #</motion> and </s>
                    #     use_cache=True,
                    #     # num_beams=num_beams,
                    #     # do_sample=do_sample,
                    #     # min_length=150,
                    #     # early_stopping=True,
                    #     # pad_token_id = self.llama_tokenizer.pad_token_id,
                    #     # bos_token_id = 529, #<
                    #     # force_words_ids= [[self.motion_start_token], [self.motion_end_token]], #<motion> </motion>
                    #     # force_words_ids = [[529, 29885, 8194, 29958], [1533, 29885, 8194, 29958]],
                    #     # bad_words_ids = [[i] for i in list(self.original_vocab.values()) if i not in self.llama_tokenizer.all_special_ids + [self.space_token_id]], #Check if useful to use or not
                    # )
                    # outputs_ = self.llama_model.model.generate(
                    #     inputs_embeds=embs,
                    #     attention_mask=attention_mask,
                    #     max_new_tokens=max_new_tokens,
                    #     stopping_criteria=stopping_criteria, #</motion> and </s>
                    #     use_cache=True,
                    # )
                    # outputs = self.llama_model.model.generate(
                    #     inputs_embeds=embs,
                    #     attention_mask=attention_mask,
                    #     max_new_tokens=max_new_tokens,
                    #     stopping_criteria=stopping_criteria, #</motion> and </s>
                    #     use_cache=True,
                    #     num_beams=self.rollouts,
                    #     num_return_sequences=self.rollouts,
                    #     early_stopping=True,
                    #     temperature=0.7 # temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
                    #     # top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
                    #     # 
                    # )
                    outputs = self.llama_model.model.generate(
                        inputs_embeds=embs,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        stopping_criteria=stopping_criteria, #</motion> and </s>
                        top_p= 0.95,
                        use_cache=True,
                        num_return_sequences=self.rollouts,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    transition_scores = self.llama_model.model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=True
                    )
                    scores = transition_scores.exp().sum(axis=1)/transition_scores.shape[1]
                    outputs = outputs.sequences

                    # outputs = self.llama_model.model.generate(
                    #     inputs_embeds=embs,
                    #     attention_mask=attention_mask,
                    #     max_new_tokens=max_new_tokens,
                    #     stopping_criteria=stopping_criteria, #</motion> and </s>
                    #     top_p= 0.95,
                    #     use_cache=True,
                    #     # num_beams=self.rollouts,
                    #     num_return_sequences=self.rollouts,
                    #     # early_stopping=True,
                    #     # temperature=0.7 # temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
                    #     # top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
                    # )

                if outputs.shape[0]>batch_size:
                    outputs = outputs.reshape(batch_size,-1, outputs.shape[-1]) # the middle dimension represent the rollouts
                    scores = scores.reshape(batch_size, -1)
                    # outputs = torch.cat((outputs_.unsqueeze(1), outputs), dim=1)

                # rel_reconstructed = torch.zeros((len(samples['traj']), 2, 19, 2))
                rel_reconstructed = torch.zeros((batch_size, self.rollouts, 2, 19, 2))
                valid_rel = [False]*batch_size
                valid_rel_rollout = {i:[] for i in range(batch_size)}
                # valid_rel = torch.Tensor([False]).expand(batch_size,self.rollouts)
                # Post processing
                for j in range(batch_size):
                    for k in range(self.rollouts):
                        output_token = outputs[j,k]
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
                            if len(rel_disc_1)==len(rel_disc_2) and len(rel_disc_1)==samples['traj'].shape[2]:
                                try:
                                    rel_disc_1 = depair_(rel_disc_1)
                                    rel_disc_2 = depair_(rel_disc_2)
                                    rel_reconstructed_1 = torch.tensor(self.discretizer.inverse_transform(rel_disc_1))
                                    rel_reconstructed_2 = torch.tensor(self.discretizer.inverse_transform(rel_disc_2))
                                    if rel_reconstructed_1.shape == rel_reconstructed_2.shape:
                                        # rel_reconstructed[j] = torch.cat((rel_reconstructed_1.unsqueeze(0),rel_reconstructed_2.unsqueeze(0)), dim=0)
                                        # valid_rel[j] = True
                                        rel_reconstructed[j,k] = torch.cat((rel_reconstructed_1.unsqueeze(0),rel_reconstructed_2.unsqueeze(0)), dim=0)
                                        valid_rel[j] = True
                                        valid_rel_rollout[j].append(True)
                                    else:
                                        valid_rel_rollout[j].append(False)
                                        # valid_rel[j] = False
                                        # break #ignore all rollouts if one of them in invalid
                                except Exception as e:
                                    continue
                            else:
                                valid_rel_rollout[j].append(False)
                        else:
                            valid_rel_rollout[j].append(False)
                return rel_reconstructed, valid_rel, valid_rel_rollout, scores
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
            discretizer_model_path = cfg.get("discretizer", './KBinsDiscretizer_76.pkl'),
            quant = cfg.get("quant", 4),
            lora_bias= cfg.get("lora_bias", False),
            ignore_motion = cfg.get("ignore_motion", False),
            rollouts = cfg.get("rollouts", 6),
            gameformer_enc = cfg.get("gameformer_enc", False),
            gf_encoder_path = cfg.get("gf_encoder_path", None),
            gf_encoder_layers = cfg.get("gf_encoder_layers", None),
            freeze_gf = cfg.get("freeze_gf", True),
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model



##############################
##############################
##############################
@registry.register_model("gameformer_gpt")
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
        # discretizer_model_path = '<legacy_trajgpt_repo>/KBinsDiscretizer_76.pkl',
        discretizer_model_path = './KBinsDiscretizer_76.pkl',
        lora_target_modules = 'all',
        lora_bias=False,
        quant = 8,
        gameformer_enc=False,
        ignore_motion=False,
        rollouts=32,
        gf_encoder_path = None,
        gf_encoder_layers = None,
        freeze_gf=True,
        in_adapter_per_feature = True,
        out_adapter_per_embed = True,
        in_adapter_mlp=False,
        out_adapter_mlp=True,
        act=False,
        recon_loss_adapter = False,
        recon_loss_llm = False,
        freeze_gf_enc = True,
        motion_act = False,
        img_token_num=204,
        output_crossattn=True,
        gf_token_num=204,
        regress_act=False,
    ):
        super().__init__()
        self.gf_token_num = gf_token_num
        self.output_crossattn = output_crossattn
        self.motion_act = motion_act
        self.recon_loss_adapter = recon_loss_adapter
        self.recon_loss_llm = recon_loss_llm
        self.act = act
        self.rollouts = rollouts
        with open(discretizer_model_path, 'rb') as file:
            self.discretizer = pickle.load(file)
         
        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
       
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, trust_remote_code=True)
        # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.pad_token = '$$'
        if low_resource:
            # bnb_config_8bit = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     llm_int8_has_fp16_weight=False,
            # )
            if quant==8:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                    # torch_dtype=torch.float16,
                    load_in_8bit=True,
                    device_map={'':torch.cuda.current_device()},
                )
            elif quant==4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    # bnb_4bit_compute_dtype=torch.float16,
                )
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                    # torch_dtype=torch.float16,
                    device_map={'':torch.cuda.current_device()},
                    quantization_config=bnb_config,
                )
            else:
                raise("Not implemented")
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
        print('> Loading LLAMA Done')
        
        
        if gameformer_enc:
            TRAINABLE_PRECISION = torch.float32
            self.gameformer_enc = True
            print('loading GameFormer')
            self.gameformer_model = GameFormer.GameFormer(
                modalities=6,
                encoder_layers=gf_encoder_layers,
                decoder_levels=6,
                # future_len=16,
                future_len=80, 
                neighbors_to_predict=1
                )
            if gf_encoder_path is not None:
                print("Load gameformer Checkpoint: {}".format(gf_encoder_path))
                ckpt = torch.load(gf_encoder_path, map_location="cpu")
                # for k,v in self.gameformer_model.named_parameters():
                #     print(k)
                # for k in ckpt['model_states'].keys():
                #     print(k)
                msg = self.gameformer_model.load_state_dict(ckpt['model_states'])
                print(msg)
            self.gameformer_model.to(TRAINABLE_PRECISION)
            if freeze_gf:
                self.gameformer_model.requires_grad_(False)
            elif freeze_gf_enc:
                self.gameformer_model.decoder.requires_grad_(False)

            print('> loading GameFormer encoder Done')
            # prompt_template = '<s>[INST] Predict the next sixteen tokens for both interactive agents based on the scene information and the provided three initial motion tokens. '
            prompt_template = '<s>[INST] Predict the future 8 seconds trajectory scene embeddings based on the 1 second observed trajectory scene embeddings of two interactive agents. '
            if self.motion_act:
                prompt_template_act = '<s>[INST] Predict the future 8 seconds trajectory scene embeddings based on the 1 second observed trajectory scene embeddings of two interactive agents. Make the ego vehicle (the first agent) [turn], [move]. Please make sure to follow the provided turning and motion instruction of the ego. '
            if self.act:
                prompt_template_act = '<s>[INST] Predict the future 8 seconds trajectory scene embeddings based on the 1 second observed trajectory scene embeddings of two interactive agents. Make the ego vehicle (the first agent) [turn]. Please make sure to follow the provided instruction. '
            
            self.regress_act = regress_act
            if regress_act:
                self.output_prompt_act = 'The motion embeddings of the two interactive agents, with the ego agent following the instruction ([turn]) are: '
            self.output_prompt = 'The motion embeddings of the two interactive agents are: '


            self.input_end_text = ' [/INST] '
            self.prompt = prompt_template
            self.prompt_act = prompt_template_act

            self.scene_start_txt = ['<Scene A>', '<Scene B>']
            self.scene_end_txt = ['</Scene A>', '</Scene B>']


            ## Adapted code from MiniGPT-5
            # IMG_TOKEN_NUM = 228
            IMG_TOKEN_NUM = img_token_num #204
            
            self.IMG_TOKEN_NUM = IMG_TOKEN_NUM
            self.img_token_num = IMG_TOKEN_NUM

            lm_hidden_size = self.llama_model.config.hidden_size
            hidden_size = lm_hidden_size
            
            self.in_adapter_per_feature = in_adapter_per_feature
            self.out_adapter_per_embed = out_adapter_per_embed
            if self.in_adapter_per_feature:
                if in_adapter_mlp:
                    self.llm_input_adapter =  nn.ModuleList([
                        nn.Sequential(
                        nn.Linear(256, 256),
                        nn.GELU(),
                        nn.Linear(256, lm_hidden_size),
                    ).to(TRAINABLE_PRECISION) for _ in range(self.gf_token_num)])
                else:
                    self.llm_input_adapter = nn.ModuleList([nn.Linear(256, lm_hidden_size).to(TRAINABLE_PRECISION) for _ in range(self.gf_token_num)]) # could be replaced with a transformer

            else:
                self.llm_input_adapter = nn.Linear(256, lm_hidden_size).to(TRAINABLE_PRECISION) # could be replaced with a transformer
            
            if self.output_crossattn:
                self.gen_text_hidden_fcs = TextFcLayer(lm_hidden_size, 256, img_token_num, num_output_tokens=self.gf_token_num,mode='gill_mapper').to(TRAINABLE_PRECISION)
            else:
                if self.out_adapter_per_embed:
                    # self.llm_output_adapter = nn.ModuleList([nn.Linear(lm_hidden_size, 256).to(TRAINABLE_PRECISION) for _ in range(IMG_TOKEN_NUM)]) # could be replaced with a transformer
                    if out_adapter_mlp:
                        self.llm_output_adapter = nn.ModuleList([
                            nn.Sequential(
                            nn.Linear(lm_hidden_size, 256),
                            nn.GELU(),
                            nn.Linear(256, 256),
                        ).to(TRAINABLE_PRECISION) for _ in range(IMG_TOKEN_NUM)])
                    else:   
                        self.llm_output_adapter = nn.ModuleList([nn.Linear(lm_hidden_size, 256).to(TRAINABLE_PRECISION) for _ in range(IMG_TOKEN_NUM)])
                else:
                    self.llm_output_adapter = nn.Linear(lm_hidden_size, 256).to(TRAINABLE_PRECISION) # could be replaced with a transformer

            
                # self.ret_text_hidden_fcs.append(layers.TextFcLayer(self.lm.config.hidden_size, self.args.ret_emb_dim,
                #                                                    num_input_tokens=self.args.num_tokens,
                #                                                    num_output_tokens=1,
                #                                                    mode=self.args.ret_text_fc_mode))

            ALL_IMG_TOKENS = [f'[IMG{i}]' for i in range(IMG_TOKEN_NUM)]
            self.ALL_IMG_TOKENS = ALL_IMG_TOKENS
            ALL_IMG_TOKENS_STR = "".join(ALL_IMG_TOKENS)
            self.targets = f'{ALL_IMG_TOKENS_STR} </s>'
            all_img_tokens = ALL_IMG_TOKENS
            self.num_new_tokens = self.llama_tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": all_img_tokens
                }
            )
            if self.num_new_tokens > 0:
                self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
                input_embeddings = self.llama_model.get_input_embeddings().weight.data
                output_embeddings = self.llama_model.get_output_embeddings().weight.data
                input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
                input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
                output_embeddings[-self.num_new_tokens:] = output_embeddings_avg
                self.input_embeddings = self.llama_model.get_input_embeddings()
                
            if len(ALL_IMG_TOKENS):
                self.output_img_id = self.llama_tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
                self.gen_token_idx = self.llama_tokenizer.convert_tokens_to_ids(self.ALL_IMG_TOKENS)

            
            self.llama_model.get_input_embeddings().to(TRAINABLE_PRECISION)
            self.llama_model.get_output_embeddings().to(TRAINABLE_PRECISION)
            # self.llama_model.get_input_embeddings().requires_grad_(True)
            # self.llama_model.get_output_embeddings().requires_grad_(True)
            input_embed_grad_mask = torch.ones_like(self.llama_model.get_input_embeddings().weight.data)
            output_embed_grad_mask = torch.ones_like(self.llama_model.get_output_embeddings().weight.data)
            input_embed_grad_mask[:-self.num_new_tokens] = 0
            output_embed_grad_mask[:-self.num_new_tokens] = 0
            self.register_buffer("input_embed_grad_mask", input_embed_grad_mask, persistent=False)
            self.register_buffer("output_embed_grad_mask", output_embed_grad_mask, persistent=False)
            # self.base_model_prepare_inputs_for_generation = self.llama_model.prepare_inputs_for_generation
            self.base_model_torch_dtype = self.llama_model.dtype
            self.llama_model.output_img_id = self.output_img_id
            # self.llama_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

            # IS_STAGE2=False
            # if IS_STAGE2:
            #     for n, p in self.llm_output_adapter.named_parameters():
            #         p.requires_grad = False


        self.lora = lora
        if self.lora:
            print('Loading LoRA')
            # self.llama_model = prepare_model_for_int8_training(self.llama_model)
            self.llama_model = prepare_model_for_kbit_training(self.llama_model)
            target_modules = [
                # 'embed_tokens', 
                'q_proj', 
                'k_proj', 
                'v_proj', 
                'o_proj', 
                'rotatry_emb', 
                'gate_proj', 
                'up_proj', 
                'down_proj', 
                # 'lm_head'
                ] 
            # target_modules = [ 
            #         'q_proj', 
            #         'v_proj',
            #         # 'lm_head','embed_tokens',
            # ]
            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                bias="lora_only" if lora_bias else "none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                modules_to_save=['lm_head','embed_tokens'],
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)
            self.llama_model.base_model.model.model.embed_tokens.original_module.weight.requires_grad = False
            self.llama_model.base_model.model.lm_head.original_module.weight.requires_grad = False
        else:
            for name, param in self.llama_model.named_parameters():
                if "lm_head" in name or "embed_tokens" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        


            # self.llama_model = get_peft_model(self.llama_model, loraconfig)
            # print('> Loading LoRA Done')
            # self.print_trainable_parameters(self.llama_model)
            
        
        # self.llama_model.enable_input_require_grads()
        # self.llama_model.gradient_checkpointing_enable()
        # self.llama_model.config.use_cache = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id
        self.llama_model.generation_config.pad_token_id = self.llama_model.config.pad_token_id
        self.llama2=True
        
        print('> Loading LoRA Done')
        print('llama:')
        self.print_trainable_parameters(self.llama_model)
        print('gameformer:')
        self.print_trainable_parameters(self.gameformer_model)
        # print('input_mapping:')
        # self.print_trainable_parameters(self.llm_input_adapter)
        # print('output_mapping:')
        # self.print_trainable_parameters(self.llm_output_adapter)
        print('')
            

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

    def gf_post_processing(self, outputs, context, input_adapted_context_embs, ego_future, neighbor_future, level=6, subsample=False):
        # text_loss = outputs["loss"]
        last_hidden_state = outputs['hidden_states'][-1]
        special_token_index = outputs['output_token_index']
        t2i_input_embedding = []
        for i in range(len(special_token_index)):
            bs_id, seq_id = special_token_index[i]
            t2i_input_embedding.append(last_hidden_state[bs_id:bs_id+1, seq_id:seq_id+self.img_token_num, :]) # get the embedding of the selected tokens
        
        if len(t2i_input_embedding) == 0:
            # loss = 0.01 * text_loss
            return -1 #{'loss': loss, 'text_loss': text_loss, 'image_loss': 0.0}
        else:
            t2i_input_embedding = torch.cat(t2i_input_embedding, dim=0)
            img_token_bs = t2i_input_embedding.shape[0]
            
            # recon loss
            recon_losses = [0,0]
            if self.recon_loss_llm: # for the llm dimension data
                if self.output_crossattn:
                    raise 'not implemented'
                llm_out_emb = t2i_input_embedding.reshape(t2i_input_embedding.shape[0],2,int(t2i_input_embedding.shape[1]/2),t2i_input_embedding.shape[-1])
                if llm_out_emb.shape != input_adapted_context_embs.shape:
                    raise 'not implemented'
                lmm_recon_loss = F.mse_loss(llm_out_emb, input_adapted_context_embs, reduction='mean')
                # (input_adapted_context_embs, llm_out_emb)
                recon_losses[0] = lmm_recon_loss
                
            ## here we might need an fc per token for our usage:
            if self.output_crossattn:
                # with self.maybe_autocast(dtype=torch.float16):
                t2i_input_embedding = self.gen_text_hidden_fcs(t2i_input_embedding, t2i_input_embedding)
            else:
                if self.out_adapter_per_embed:
                    t2i_input_embedding_ = []
                    for adapter_i, adapter_module_i in enumerate(self.llm_output_adapter):
                        t2i_input_embedding_.append(adapter_module_i(t2i_input_embedding[:,adapter_i]))
                    t2i_input_embedding = torch.stack(t2i_input_embedding_, dim=1)
                else:
                    t2i_input_embedding = self.llm_output_adapter(t2i_input_embedding)
                
                if int(t2i_input_embedding.shape[1]/2) != int(self.gf_token_num/2):
                    raise 'not implemented'
                # t2i_input_embedding = self.llm_output_mapping_fc(t2i_input_embedding) # shape: [batch, self.img_token_num, 256]
                ## transformer mapping that is used in minigpt5, we will skip in this implementation:
                # mapping_feature = self.llm_to_t2i_mapping(src=t2i_input_embedding, tgt=self.t2i_decoder_prompt.repeat(img_token_bs, 1, 1)) 
                # model.gameformer
                ## We need to use context['actors'], context['masks'], and map t2i_input_embedding to the same shape as context['encodings'] to be used by the decoder
                ## changing t2i_input_embedding.shape = [batch, self.img_token_num, 256] to context['encodings'].shape = [batch, 2, 114, 256]
            t2i_input_embedding = t2i_input_embedding.reshape(t2i_input_embedding.shape[0],2,int(t2i_input_embedding.shape[1]/2),t2i_input_embedding.shape[-1])
            predicted_context_encodings = t2i_input_embedding
            # we need to change img_num_tokens to 114*2 in the begenning.
            
                # raise 'not impelemnted'
                # predicted_context_encodings = self.fc_to_114(predicted_context_encodings.transpose(-1,-2)).transpose(-1,-2)
            
            # recon loss
            if self.recon_loss_adapter: # for the gameformer dimension data
                adapter_recon_loss = F.mse_loss(predicted_context_encodings, context['encodings'], reduction='mean')
                recon_losses[1] = adapter_recon_loss
            
            ## We possibly can add predicted_context_encodings to original encodings, as risidual connection?, or possible also passing original encoding with another linear layer
            gameformer_decoder_input = {'encodings': predicted_context_encodings,'actors': context['actors'], 'masks': context['masks']}
            decoded_output = self.gameformer_model.decoder(gameformer_decoder_input)
            loss, future, _ = level_k_loss(decoded_output, ego_future, neighbor_future, level, subsample=subsample)
            return loss, future, recon_losses

    def gf_post_processing_eval(self, outputs, context, ego_future, neighbor_future, level=6, subsample=False):
        t2i_input_embedding = outputs
        img_token_bs = t2i_input_embedding.shape[0]
        ## here we might need an fc per token for our usage:
        if self.output_crossattn:
            # with self.maybe_autocast(dtype=torch.float16):
            t2i_input_embedding = self.gen_text_hidden_fcs(t2i_input_embedding, t2i_input_embedding)
        else:
            if self.out_adapter_per_embed:
                t2i_input_embedding_ = []
                for adapter_i, adapter_module_i in enumerate(self.llm_output_adapter):
                    t2i_input_embedding_.append(adapter_module_i(t2i_input_embedding[:,adapter_i]))
                t2i_input_embedding = torch.stack(t2i_input_embedding_, dim=1)
            else:
                t2i_input_embedding = self.llm_output_adapter(t2i_input_embedding)
        
        
            # we need to change img_num_tokens to 114*2 in the begenning.
            # if predicted_context_encodings.shape[2] != int(self.num_new_tokens/2):
            if int(t2i_input_embedding.shape[1]/2) != int(self.gf_token_num/2):
                raise 'not impelemnted'

        # t2i_input_embedding = self.llm_output_mapping_fc(t2i_input_embedding) # shape: [batch, self.img_token_num, 256]
        ## transformer mapping that is used in minigpt5, we will skip in this implementation:
        # mapping_feature = self.llm_to_t2i_mapping(src=t2i_input_embedding, tgt=self.t2i_decoder_prompt.repeat(img_token_bs, 1, 1)) 
        # model.gameformer
        ## We need to use context['actors'], context['masks'], and map t2i_input_embedding to the same shape as context['encodings'] to be used by the decoder
        ## changing t2i_input_embedding.shape = [batch, self.img_token_num, 256] to context['encodings'].shape = [batch, 2, 114, 256]
        t2i_input_embedding = t2i_input_embedding.reshape(t2i_input_embedding.shape[0],2,int(t2i_input_embedding.shape[1]/2),t2i_input_embedding.shape[-1])
        predicted_context_encodings = t2i_input_embedding
        
            # predicted_context_encodings = self.fc_to_114(predicted_context_encodings.transpose(-1,-2)).transpose(-1,-2)
        ## We possibly can add predicted_context_encodings to original encodings, as risidual connection?, or possible also passing original encoding with another linear layer
        gameformer_decoder_input = {'encodings': predicted_context_encodings,'actors': context['actors'], 'masks': context['masks']}
        decoded_output = self.gameformer_model.decoder(gameformer_decoder_input)
        loss, future, trajectories = level_k_loss(decoded_output, ego_future, neighbor_future, level, subsample=subsample)
        scores = decoded_output[f'level_{level}_scores']
        return loss, future, trajectories, scores

            # print('')

        
    def forward(self, samples, _training=True):
        
        self.llama_tokenizer.padding_side = "right"
        device = samples['traj'].device
        batch_size = samples['traj'].shape[0]

        rel = abs_distance_to_velocity(abs_distance=samples['traj']).cpu()
        # rel_disc = self.discretizer.transform(rel.reshape(-1,2))
        # rel_disc = torch.tensor(pair_(rel_disc)).reshape(rel[...,0].shape)
        # obs = get_waymo_parsed_short2_batch(rel_disc, 3)
        # obs = get_waymo_parsed_batch(samples['disc_traj'], samples['disc_rel'], 3)
        # obs = get_waymo_parsed_batch_short(samples['disc_traj'], samples['disc_rel'], 3)
        # obs = get_waymo_parsed_short2(samples['disc_rel'], 3)

        # obs = [' ' + obs_+self.input_end_text for obs_ in obs]
        # obs = [obs_+self.input_end_text for obs_ in obs] # old
        # obs_tokens = self.llama_tokenizer(obs, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
        if self.act:
            prompts_act = []
            prompts_act_out = []
            for i, valid_action in enumerate(samples['valid_action']):
                if valid_action:
                    if self.act:
                        sample_prompt = self.prompt_act.split('[turn]')[0] + samples['turn'][i] + self.prompt_act.split('[turn]')[1]
                    if self.motion_act:
                        if samples['move1'][i] == samples['move2'][i]:
                            move_prompt = 'it should '+ samples['move1'][i]
                        else:
                            move_prompt = 'it should first '+ samples['move1'][i] +' then '+ samples['move2'][i]
                        sample_prompt = sample_prompt.split('[move]')[0] + move_prompt+ sample_prompt.split('[move]')[1]
                    if self.regress_act:
                        sample_prompt_out = self.output_prompt_act.split('[turn]')[0] + samples['turn'][i] + self.output_prompt_act.split('[turn]')[1]
                    else:
                        sample_prompt_out = self.output_prompt
                else:
                    sample_prompt = self.prompt
                    sample_prompt_out = self.output_prompt
                prompts_act.append(sample_prompt)
                prompts_act_out.append(sample_prompt_out)
            prompts_to_tokenize = prompts_act
            output_prompts_to_tokenize = prompts_act_out
            output_prompts_to_tokenize = [prompt_ + self.targets for prompt_ in output_prompts_to_tokenize]
        else:
            prompts_to_tokenize = self.prompt # the old, with no act instruction
            # raise 'prompts_to_tokenize and output_prompts_to_tokenize needs to be redefined here'

        prompt_tokens = self.llama_tokenizer(prompts_to_tokenize, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
        prompt_tokens_attention_mask = prompt_tokens.attention_mask

        if not self.lora:
            prompt_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids)
            # obs_embeds = self.llama_model.model.embed_tokens(obs_tokens.input_ids)    
        else:
            prompt_embeds = self.llama_model.model.model.embed_tokens(prompt_tokens.input_ids)
            # obs_embeds = self.llama_model.model.model.embed_tokens(obs_tokens.input_ids)
        
        if not self.act:
            prompt_embeds = expanded_stack(prompt_embeds[0], batch_size)
            prompt_tokens_attention_mask = expanded_stack(prompt_tokens_attention_mask[0], batch_size)

        if _training:
            ### MiniGPT-5 codes
            target_ids = self.llama_tokenizer(self.targets, add_special_tokens=False).input_ids
            target_ids = torch.tensor(target_ids)
            target_ids_ = target_ids
            output_img_id = self.output_img_id #self.llama_tokenizer.convert_tokens_to_ids(self.ALL_IMG_TOKENS[0])
            # label = torch.ones_like(input_dict["input_ids"])*-100
            # label = torch.cat((label, target_ids), dim=1)
            index = torch.nonzero(target_ids == output_img_id)
            if len(index):
                # index = index[0,1]
                index = index[0][0]
                # target_ids_[:, index+1:index+self.IMG_TOKEN_NUM-1] = -100
                target_ids_[index+1:] = -100
            target_ids_atts = torch.ones_like(target_ids)
            # torch.ones_like(target_ids)
            

            ### End MiniGPT-5 codes
            # FOR TRAINING AND EVALUATION
            if self.gameformer_enc:
                start_scene_token = self.llama_tokenizer(self.scene_start_txt, return_tensors="pt", add_special_tokens=False).to(device)
                end_scene_token = self.llama_tokenizer(self.scene_end_txt, return_tensors="pt", add_special_tokens=False).to(device)
                # with self.maybe_autocast(dtype=torch.float32):
                # with self.maybe_autocast():
                with self.maybe_autocast(dtype=torch.bfloat16):
                    gameformer_input = {key:value for key,value in samples.items() if key in ['ego_state', 'neighbors_state', 'ego_state', 'map_lanes', 'map_crosswalks']}
                    # with self.maybe_autocast():
                    gameformer_input['map_lanes'] = gameformer_input['map_lanes'][...,:200:2,:]
                    gameformer_input['map_crosswalks'] = gameformer_input['map_crosswalks'][...,:100:2,:]
                    context = self.gameformer_model.encoder(gameformer_input)
                    # context = self.gameformer_encoder(gameformer_input)
                    context_embs = context['encodings'] # [b,N:2, tokens:114, dim=256]
                    # context_embs = context_embs.masked_fill(context['masks'].unsqueeze(-1), 0.0) # mask it
                    context_embs_shape = context_embs.shape
                    context_embs = context_embs.reshape(context_embs.shape[0], -1, context_embs.shape[-1])
                    if self.in_adapter_per_feature:
                        context_embs_llm_maped = []
                        for adapter_i in range(len(self.llm_input_adapter)):
                            context_embs_llm_maped.append(self.llm_input_adapter[adapter_i](context_embs[:,adapter_i,:]))
                        context_embs = torch.stack(context_embs_llm_maped, dim=1)
                    else:
                        context_embs = self.llm_input_adapter(context_embs)
                    
                    context_embs = context_embs.reshape(context_embs_shape[0], context_embs_shape[1], context_embs_shape[2], -1)
                    # context_embs = self.gameformer_adapter(context['encodings']) # [batch, 2 agents, 114 features, 4096 projected embeds dim]
                    context_atts = (~context['masks']).int() # assuming False in context['masks'] means no masking (1s)
                if not self.lora:
                    start_scene_embed = self.llama_model.model.embed_tokens(start_scene_token.input_ids)
                    end_scene_embed = self.llama_model.model.embed_tokens(end_scene_token.input_ids)
                    pad_embed = self.llama_model.model.embed_tokens(torch.tensor([self.llama_model.config.pad_token_id], device=device))
                else:
                    start_scene_embed = self.llama_model.model.model.embed_tokens(start_scene_token.input_ids)
                    end_scene_embed = self.llama_model.model.model.embed_tokens(end_scene_token.input_ids)
                    pad_embed = self.llama_model.model.model.embed_tokens(torch.tensor(self.llama_model.config.pad_token_id, device=device))
                    
                    # if self.pad_masked_context: # missing context, not necessarily have zero values here.
                    context_embs[context['masks']] = pad_embed.to(torch.bfloat16) # We can ad random embeddings masking here
                    # context_embs[context['masks']] = pad_embed


            
            
            # input_tokens = torch.cat((expanded_stack(prompt_tokens.input_ids[0], batch_size), obs_tokens.input_ids), dim=1) # This is not used

            if self.gameformer_enc:
                input_embeds = torch.cat((
                    prompt_embeds,
                    expanded_stack(start_scene_embed[0], batch_size),
                    context_embs[:,0],
                    expanded_stack(end_scene_embed[0],batch_size),
                    expanded_stack(start_scene_embed[1], batch_size),
                    context_embs[:,1],
                    expanded_stack(end_scene_embed[1],batch_size),
                    # obs_embeds
                    ), dim=1)
                    
                input_atts = torch.cat((
                    prompt_tokens_attention_mask,
                    expanded_stack(start_scene_token.attention_mask[0],batch_size),
                    context_atts[:,0],
                    expanded_stack(end_scene_token.attention_mask[0],batch_size),
                    expanded_stack(start_scene_token.attention_mask[1],batch_size),
                    context_atts[:,1],
                    expanded_stack(end_scene_token.attention_mask[1],batch_size),
                    # obs_tokens.attention_mask
                    ), dim=1) 

                
            else:
                input_atts = torch.cat((prompt_tokens.attention_mask.expand(batch_size, -1), obs_tokens.attention_mask), dim=1)
                input_embeds = torch.cat((prompt_embeds.expand(batch_size, -1, -1), obs_embeds), dim=1)
            
            # targets_ = self.llama_tokenizer(self.targets, return_tensors="pt", add_special_tokens=False).to(device)
            # if self.regress_act:
            if False:
                targets_ = self.llama_tokenizer(output_prompts_to_tokenize, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
                targets_tokens = targets_.input_ids
                ontext_atts_reshaped = context_atts.reshape(context_atts.shape[0], -1)
                targets_atts = targets_.attention_mask
            else:
                targets_ = self.llama_tokenizer(self.targets, return_tensors="pt", add_special_tokens=False).to(device)
                targets_tokens = expanded_stack(targets_.input_ids[0], batch_size)
                context_atts_reshaped = context_atts.reshape(context_atts.shape[0], -1)
                # targets_atts = torch.zeros(expanded_stack(targets_.attention_mask[0], batch_size).shape, device=device)
                targets_atts = torch.ones(expanded_stack(targets_.attention_mask[0], batch_size).shape, device=device)
                # targets_atts[:, :context_atts_reshaped.shape[-1]] = context_atts_reshaped # using the same input masking
                # targets_atts[:,-1:] = expanded_stack(targets_.attention_mask[0][-1:], batch_size) # EOS
                # targets_atts = expanded_stack(targets_.attention_mask[0], batch_size)
                # targets = expanded_stack(targets[0], batch_size)
            empty_targets = (
                torch.ones([input_atts.shape[0], input_atts.shape[1]],
                        dtype=torch.long).to(device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets_tokens], dim=1)
            # targets = targets.masked_fill(targets == self.llama_tokenizer.pad_token_id, -100) # eos masking
            # targets = targets.masked_fill(targets == 835, -100)# eos masking
            # targets = targets.masked_fill(targets == 835, self.llama_tokenizer.pad_token_id)# changing the eos
            # self.llama_tokenizer.pad_token_id
            # targets[:,-1] = -100 
            if not self.lora:
                targets_embeds = self.llama_model.model.embed_tokens(targets_tokens)
            else:
                targets_embeds = self.llama_model.model.model.embed_tokens(targets_tokens)
            embs = torch.cat((input_embeds, targets_embeds),dim=1)
            attention_mask = torch.cat([input_atts, targets_atts], dim=1)
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=embs,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                    output_hidden_states=True,
                )
            loss = outputs.loss
            logits = outputs.logits
            hidden_states = outputs.hidden_states
            output_token_index = (targets == self.output_img_id).nonzero()
            if len(output_token_index):
                addon_index = torch.ones_like(output_token_index)*(-1)
                addon_index[:, 0] = 0
                output_token_index += addon_index
            
            outputs_dict = {"lm_loss": loss, "logits": logits, "hidden_states": hidden_states, "output_token_index": output_token_index}
            # self.post_processing(outputs_dict, context)
            gf_loss, gf_future, recon_losses = self.gf_post_processing(outputs_dict, context, context_embs, ego_future=samples['ground_truth'][:,0,:,:], neighbor_future=samples['ground_truth'][:,1,:,:], level=6, subsample=False)
            total_loss = loss + gf_loss
            if recon_losses[0]>0:
                total_loss+=recon_losses[0]
            if recon_losses[1]>0:
                total_loss+=recon_losses[1]

            outputs_dict.update({"gf_loss": gf_loss, "gf_future": gf_future, "loss": total_loss, "llm_recon_loss":recon_losses[0], "gf_recon_loss": recon_losses[1]})
            # outputs_dict.update({"gf_loss": gf_loss, "gf_future": gf_future, "loss": loss})
            self.reset_embeddings()
            return outputs_dict
        else:
            if self.act:
                # "from_"++"_to_"+
                outputs1 = self.predict(samples)
                batch_data_names = []
                turn_acts = ['not move', 'move straight', 'take a soft left turn', 'take a soft right turn', 'take a left turn', 'take a right turn', 'take a sharp left turn', 'take a sharp right turn', 'take a left U-turn', 'take a right U-turn']
                motion_acts = ['not move', 'start to move', 'stop', 'move with a constant velocity', 'accelerate', 'decelerate']
                for batch_i in range(len(samples['turn'])):
                    data_name = "Original: "+ samples['turn'][batch_i]
                    if samples['valid_action'][batch_i]:
                        # print(f"Original: {samples['turn'][batch_i]}")
                        turn_idx = [i_ for i_ in range(len(turn_acts)) if turn_acts[i_]== samples['turn'][batch_i]][0]
                        if turn_acts[turn_idx] == 'move straight':
                            samples['turn'][batch_i] = 'take a right turn'
                        elif turn_acts[turn_idx] == 'U-turn':
                            samples['turn'][batch_i] = 'move straight'
                        elif 'left' in turn_acts[turn_idx]:
                            samples['turn'][batch_i] = turn_acts[turn_idx+1]
                        elif 'right' in turn_acts[turn_idx]:
                            samples['turn'][batch_i] = turn_acts[turn_idx-1]
                        # print(f"Updated: {samples['turn'][batch_i]}")
                        data_name += "\nUsed: "+samples['turn'][batch_i]
                    
                    batch_data_names.append(data_name)
                outputs2 = self.predict(samples)
                # outputs2 = outputs1
                outputs1.update({k+'_act':v for k,v in outputs2.items()})
                outputs1.update({'batch_data_names':batch_data_names})
                return outputs1
            else:
                return self.predict(samples)
    
    def generate(self, embeddings=torch.FloatTensor, max_len: int = 288,
                temperature: float = 0.0, top_p: float = 1.0, min_word_tokens: int = 0,
                filter_value: float = -float('Inf')):
        """Runs greedy decoding and returns generated captions.

        Args:
            min_word_tokens: Minimum number of words to generate before allowing a [IMG] output.
            filter_value: Value to assign to tokens that should never be generated.
        Outputs:
            out: (N, T) int32 sequence of output tokens.
            output_embeddings: (N, T, 256) sequence of text output embeddings.
        """
        self.eval()
        # self.lm.eval()

        with torch.no_grad():  # no tracking history
            # init output with image tokens
            out = None
            output_embeddings = []
            output_logits = []
            max_len = 2
            for i in range(max_len):
                output = self.llama_model(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)

                text_emb_layers = [-1]
                for idx in text_emb_layers:
                    output_embeddings.append(output.hidden_states[idx])

                logits = output.logits[:, -1, :]  # (N, vocab_size)
                if top_p == 1.0:
                    logits = logits.cpu()
                output_logits.append(logits)


                self.retrieval_token_idx = self.gen_token_idx
                # Prevent the model from generating the [IMG1..n] tokens.
                # logits[:, self.retrieval_token_idx[1:]] = filter_value
                logits[:, self.gen_token_idx[1:]] = filter_value

                if (self.retrieval_token_idx or self.gen_token_idx) and self.retrieval_token_idx[0] != -1 and \
                        self.gen_token_idx[0] != -1:
                        if i < min_word_tokens:
                            # Eliminate probability of generating [IMG] if this is earlier than min_word_tokens.
                            # logits[:, self.retrieval_token_idx] = filter_value
                            logits[:, self.gen_token_idx] = filter_value
                        else:
                            mask = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
                            mask[:, self.retrieval_token_idx] = False
                            logits[mask] = filter_value

                if temperature == 0.0:
                    if top_p != 1.0:
                        raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                else:
                    logits = logits / temperature

                    # Apply top-p filtering.
                    if top_p < 1.0:
                        assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # (N, D)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for j in range(sorted_indices.shape[0]):
                            indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                            logits[j, indices_to_remove] = filter_value

                    token_weights = logits.exp()  # (N, vocab_size)
                    next_token = torch.multinomial(token_weights, 1)  # (N, 1)

                # Force generation of the remaining [IMG] tokens if [IMG0] is generated.
                if next_token.shape[0] == 1 and next_token.item() == self.retrieval_token_idx[0]:
                    assert self.retrieval_token_idx == self.gen_token_idx, (
                    self.retrieval_token_idx, self.gen_token_idx)
                    next_token = torch.tensor(self.retrieval_token_idx)[None, :].long().to(
                        embeddings.device)  # (1, num_tokens)
                else:
                    next_token = next_token.long().to(embeddings.device)

                if out is not None:
                    out = torch.cat([out, next_token], dim=-1)
                else:
                    out = next_token

                next_embedding = self.input_embeddings(next_token)
                embeddings = torch.cat([embeddings, next_embedding], dim=1)
                embeddings = embeddings[:, max(0, embeddings.shape[1] - 2048):]

        return out, output_embeddings, output_logits
    
    def predict(self, samples):
        with torch.no_grad():
            self.llama_tokenizer.padding_side = "right"
            device = samples['traj'].device
            batch_size = samples['traj'].shape[0]

            if self.act:
                prompts_act = []
                for i, valid_action in enumerate(samples['valid_action']):
                    if valid_action:
                        sample_prompt = self.prompt_act.split('[turn]')[0] + samples['turn'][i] + self.prompt_act.split('[turn]')[1]
                        if self.motion_act:
                            if samples['move1'][i] == samples['move2'][i]:
                                move_prompt = 'it should '+ samples['move1'][i]
                            else:
                                move_prompt = 'it should first '+ samples['move1'][i] +' then '+ samples['move2'][i]
                            sample_prompt = sample_prompt.split('[move]')[0] + move_prompt+ sample_prompt.split('[move]')[1]
                    else:
                        sample_prompt = self.prompt
                    prompts_act.append(sample_prompt)
                prompts_to_tokenize = prompts_act
            else:
                prompts_to_tokenize = self.prompt # the old, with no act instruction

            prompt_tokens = self.llama_tokenizer(prompts_to_tokenize, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
            prompt_tokens_attention_mask = prompt_tokens.attention_mask
            if not self.lora:
                prompt_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids)
            else:
                prompt_embeds = self.llama_model.model.model.embed_tokens(prompt_tokens.input_ids)
            if not self.act:
                prompt_embeds = expanded_stack(prompt_embeds[0], batch_size)
                prompt_tokens_attention_mask = expanded_stack(prompt_tokens_attention_mask[0], batch_size)
            # prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)       
            ### MiniGPT-5 codes
            target_ids = self.llama_tokenizer(self.targets, add_special_tokens=False).input_ids
            target_ids = torch.tensor(target_ids)
            target_ids_ = target_ids
            output_img_id = self.output_img_id
            index = torch.nonzero(target_ids == output_img_id)
            if len(index):
                index = index[0][0]
                target_ids_[index+1:] = -100
            target_ids_atts = torch.ones_like(target_ids)
            # FOR TRAINING AND EVALUATION
            if self.gameformer_enc:
                start_scene_token = self.llama_tokenizer(self.scene_start_txt, return_tensors="pt", add_special_tokens=False).to(device)
                end_scene_token = self.llama_tokenizer(self.scene_end_txt, return_tensors="pt", add_special_tokens=False).to(device)
                with self.maybe_autocast(dtype=torch.bfloat16):
                    gameformer_input = {key:value for key,value in samples.items() if key in ['ego_state', 'neighbors_state', 'ego_state', 'map_lanes', 'map_crosswalks']}
                    # with self.maybe_autocast():
                    gameformer_input['map_lanes'] = gameformer_input['map_lanes'][...,:200:2,:]
                    gameformer_input['map_crosswalks'] = gameformer_input['map_crosswalks'][...,:100:2,:]
                    context = self.gameformer_model.encoder(gameformer_input)
                    # context = self.gameformer_encoder(gameformer_input)
                    context_embs = context['encodings'] # [b,N:2, tokens:114, dim=256]
                    context_embs_shape = context_embs.shape
                    context_embs = context_embs.reshape(context_embs.shape[0], -1, context_embs.shape[-1])
                    context_embs_llm_maped = []
                    if self.in_adapter_per_feature:
                        context_embs_llm_maped = []
                        for adapter_i in range(len(self.llm_input_adapter)):
                            context_embs_llm_maped.append(self.llm_input_adapter[adapter_i](context_embs[:,adapter_i,:]))
                        context_embs = torch.stack(context_embs_llm_maped, dim=1)
                    else:
                        context_embs = self.llm_input_adapter(context_embs)
                    # context_embs = torch.stack(context_embs_llm_maped, dim=1)
                    context_embs = context_embs.reshape(context_embs_shape[0], context_embs_shape[1], context_embs_shape[2], -1)
                    # context_embs = self.gameformer_adapter(context['encodings']) # [batch, 2 agents, 114 features, 4096 projected embeds dim]
                    context_atts = (~context['masks']).int() # assuming False in context['masks'] means no masking (1s)
                if not self.lora:
                    start_scene_embed = self.llama_model.model.embed_tokens(start_scene_token.input_ids)
                    end_scene_embed = self.llama_model.model.embed_tokens(end_scene_token.input_ids)
                    pad_embed = self.llama_model.model.embed_tokens(torch.tensor([self.llama_model.config.pad_token_id], device=device))
                else:
                    start_scene_embed = self.llama_model.model.model.embed_tokens(start_scene_token.input_ids)
                    end_scene_embed = self.llama_model.model.model.embed_tokens(end_scene_token.input_ids)
                    pad_embed = self.llama_model.model.model.embed_tokens(torch.tensor(self.llama_model.config.pad_token_id, device=device))
                    
                    context_embs[context['masks']] = pad_embed.to(torch.bfloat16)
                    # context_embs[context['masks']] = pad_embed
            
            # input_tokens = torch.cat((expanded_stack(prompt_tokens.input_ids[0], batch_size), obs_tokens.input_ids), dim=1) # This is not used

            if self.gameformer_enc:
                input_embeds = torch.cat((
                    prompt_embeds,
                    expanded_stack(start_scene_embed[0], batch_size),
                    context_embs[:,0],
                    expanded_stack(end_scene_embed[0],batch_size),
                    expanded_stack(start_scene_embed[1], batch_size),
                    context_embs[:,1],
                    expanded_stack(end_scene_embed[1],batch_size),
                    # obs_embeds
                    ), dim=1)
                    
                input_atts = torch.cat((
                    prompt_tokens_attention_mask,
                    expanded_stack(start_scene_token.attention_mask[0],batch_size),
                    context_atts[:,0],
                    expanded_stack(end_scene_token.attention_mask[0],batch_size),
                    expanded_stack(start_scene_token.attention_mask[1],batch_size),
                    context_atts[:,1],
                    expanded_stack(end_scene_token.attention_mask[1],batch_size),
                    # obs_tokens.attention_mask
                    ), dim=1) 

            else:
                input_atts = torch.cat((prompt_tokens.attention_mask.expand(batch_size, -1), obs_tokens.attention_mask), dim=1)
                input_embeds = torch.cat((prompt_embeds.expand(batch_size, -1, -1), obs_embeds), dim=1)
            
            stop_words_ids = [torch.tensor([self.llama_tokenizer.eos_token_id]).to(prompt_tokens.input_ids.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
            num_words = 2 # to get 1 scene generation

            with torch.no_grad():
                with self.maybe_autocast():
                    output_embeddings = []
                    for i in range(input_embeds.shape[0]):
                        generated_ids, generated_embeddings, _ = self.generate(
                                embeddings=input_embeds[i:i+1],
                                max_len=num_words,
                                min_word_tokens=0, # no text output
                        )
                        output_embeddings.append(generated_embeddings[-1][:, input_embeds.shape[1]:][0])
            output_embeddings = torch.stack(output_embeddings)
            gf_loss, gf_future, trajectories, gf_scores = self.gf_post_processing_eval(output_embeddings, context, ego_future=samples['ground_truth'][:,0,:,:], neighbor_future=samples['ground_truth'][:,1,:,:], level=6, subsample=False)
            return {'gf_loss': gf_loss,'output_traj': gf_future[0], 'modalities': trajectories, 'scores': gf_scores}

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, force_generation=None, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(input_ids, inputs_embeds=inputs_embeds, *args, **kwargs)
        bs = inputs_embeds.shape[0]
        device = inputs_embeds.device
        if 'input_ids' in model_kwargs:
            new_token_ids = model_kwargs['input_ids'][:, -1:]
            if new_token_ids == self.output_img_id:
                #Generated the image token, force add all the image tokens
                current_position_ids = model_kwargs['position_ids'][0, -1] #TODO: Only support batch size 1
                all_img_tokens = torch.tensor(self.llama_tokenizer.convert_tokens_to_ids(self.ALL_IMG_TOKENS)).unsqueeze(0).to(device)
                all_img_tokens_mask = torch.ones_like(all_img_tokens)[:, :self.IMG_TOKEN_NUM-1].to(device)
                all_img_position_ids = torch.arange(current_position_ids, current_position_ids + self.IMG_TOKEN_NUM).unsqueeze(0).to(device)
                
                model_kwargs['attention_mask'] = torch.cat([model_kwargs['attention_mask'], all_img_tokens_mask], dim=1)
                model_kwargs['position_ids'] = all_img_position_ids
                inputs_embeds = self.get_input_embeddings(all_img_tokens)
                model_kwargs['input_ids'] = None
                model_kwargs['inputs_embeds'] = inputs_embeds

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
            discretizer_model_path = cfg.get("discretizer", './KBinsDiscretizer_76.pkl'),
            quant = cfg.get("quant", 4),
            lora_bias= cfg.get("lora_bias", False),
            ignore_motion = cfg.get("ignore_motion", False),
            rollouts = cfg.get("rollouts", 6),
            gameformer_enc = cfg.get("gameformer_enc", False),
            gf_encoder_path = cfg.get("gf_encoder_path", None),
            gf_encoder_layers = cfg.get("gf_encoder_layers", None),
            freeze_gf = cfg.get("freeze_gf", True),
            in_adapter_per_feature = cfg.get("in_adapter_per_feature", True),
            out_adapter_per_embed = cfg.get("out_adapter_per_embed", True),
            in_adapter_mlp=cfg.get("in_adapter_mlp", False),
            out_adapter_mlp=cfg.get("out_adapter_mlp", True),
            act=cfg.get("act", False),
            recon_loss_adapter = cfg.get("recon_loss_adapter", False),
            recon_loss_llm = cfg.get("recon_loss_llm", False),
            freeze_gf_enc = cfg.get("freeze_gf_enc", True),
            motion_act = cfg.get("motion_act", False),
            img_token_num = cfg.get("img_token_num", 204),
            gf_token_num =  cfg.get("gf_token_num", 204),
            output_crossattn = cfg.get("output_crossattn", False),
            regress_act = cfg.get("regress_act", False),
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
        
    def reset_embeddings(self):
        with torch.no_grad():
            if self.lora:
                for n, p in self.llama_model.named_parameters():
                    if p.grad is None:
                        continue
                    if "lm_head" in n:
                        p.grad = p.grad*self.output_embed_grad_mask
                    elif "embed_tokens" in n:
                        p.grad = p.grad*self.input_embed_grad_mask
            else:
                self.llama_model.get_input_embeddings().weight.grad = self.llama_model.get_input_embeddings().weight.grad*self.input_embed_grad_mask
                self.llama_model.get_output_embeddings().weight.grad = self.llama_model.get_output_embeddings().weight.grad*self.output_embed_grad_mask
    
    def on_before_optimizer_step(self, optimizer) -> None:
        self.reset_embeddings()

    