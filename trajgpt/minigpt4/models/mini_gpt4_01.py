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
# from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
# from transformers import BitsAndBytesConfig

import re
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from transformers import StoppingCriteria, StoppingCriteriaList
from minigpt4.tasks.image_text_pretrain import StoppingCriteriaSub


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
        step_identifier = True,
        no_new_vocab=True,
        lora_r=64,
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()
        self.no_new_vocab=no_new_vocab
        self.lora = lora
        self.step_identifier = step_identifier
        self.xy_identifier = xy_identifier 
        self.point_identifier = point_identifier

        self.space_token_id= space_token_id
        self.freeze_embed_tokens = freeze_embed_tokens
        if data_info_path:
            data_info = torch.load(data_info_path)
            self.rel_min= data_info['rel_min']
            self.traj_max= data_info['rel_cont_grid']
            self.rel_min= data_info['rel_min']
            self.rel_max= data_info['rel_max']
            self.seq_len= data_info['seq_len']
            self.traj_cont_grid= data_info['traj_cont_grid']
            self.rel_cont_grid= data_info['rel_cont_grid']
            self.grid_size= data_info['grid_size']
        else:
            self.grid_size = grid_size

        jls_extract_var = self
        jls_extract_var.num_agents = num_agents

        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
       
        print('Loading LLAMA tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        ### --------------------- Wahab-Start: Adding new tokens ---------------------
        # print('NEW: Adding new trajectory domain specific tokens')
        ## Alternative solution is to use add_special_tokens(), instead of add_tokens
    
        # special_tokens_dict = {'additional_special_tokens':new_tokens_list}
        # self.llama_tokenizer.add_special_tokens(special_tokens_dict)
        
        ## grid size tokens <g-512x512>: 
        # grid_size=512
        # grid_sizes=[grid_size]
        # new_grid_tokens_list = [f"<g-{g}x{g}>" for g in grid_sizes]
         ## agent specific tokens:
        # num_agents = 2
        self.motion_start_symbol = "<motion>"
        self.motion_end_symbol = "</motion>" 
        new_data_holders_list = [self.motion_start_symbol, self.motion_end_symbol]
        
        # self.llama_tokenizer.add_special_tokens({"pad_token":"<pad>"})
        
        new_agent_tokens_list = [f"<a-{i}>" for i in range(self.num_agents)]
        ## spatial tokens:
        s_range = [i for i in range(0, self.grid_size+1)]
        new_coor_tokens_list = [f"<s-{i}>" for i in s_range]
        ## Concatinating all new tokens
        # new_tokens_list = new_grid_tokens_list + new_agent_tokens_list + new_coor_tokens_list
        new_tokens_list = new_agent_tokens_list + new_coor_tokens_list + new_data_holders_list
        if self.xy_identifier:
            self.xy_identifier_symbols = ['<x>','</x>','<y>','</y>']
            new_tokens_list = new_tokens_list + self.xy_identifier_symbols
        if self.point_identifier:
            self.point_identifier_symbols = ['<p>','</p>']
            new_tokens_list = new_tokens_list + self.point_identifier_symbols
        if self.step_identifier:
            self.step_identifier_symbols = [f'<step-{i+1}>' for i in range(self.seq_len)]
            new_tokens_list = new_tokens_list + self.step_identifier_symbols
        ## Checking tokens does not exist in pretrained vocab:
        pretrained_vocab = self.llama_tokenizer.get_vocab()
        self.original_vocab = pretrained_vocab
        if any(new_token_i in pretrained_vocab for new_token_i in new_tokens_list):
            raise 'Added token is/are already in the tokenizer vocabulary'
        else:
            ## Adding new vocab
            if not self.no_new_vocab:
                self.llama_tokenizer.add_tokens(new_tokens_list)
                print(f"* New {len(self.llama_tokenizer.get_vocab()) - len(pretrained_vocab)} tokens added to the tokenizer vocab")
                self.motion_start_token = self.llama_tokenizer(self.motion_start_symbol, add_special_tokens=False).input_ids[0]
                self.motion_end_token = self.llama_tokenizer(self.motion_end_symbol, add_special_tokens=False).input_ids[0]


        ### --------------------- Wahab-End: Adding new tokens ---------------------
        if self.lora:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                quantization_config=bnb_config,
                # device_map="auto",
                # low_cpu_mem_usage=True
                device_map={'':torch.cuda.current_device()}
            )
        elif self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                # torch_dtype=torch.float16,
                load_in_8bit=True,
                # device_map={'': device_8bit}
                device_map="auto",
                low_cpu_mem_usage=True
                # device_map={'':torch.cuda.current_device()}
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
        self.llama_model.config.pretraining_tp = 1
        
        for name, param in self.llama_model.named_parameters():
                param.requires_grad = False                
                if param.ndim == 1:
                    param.data = param.data.to(torch.float32)
        # if use_grad_checkpoint:

        self.llama_model.gradient_checkpointing_enable()
        self.llama_model.enable_input_require_grads()
        self.llama_model = prepare_model_for_kbit_training(self.llama_model)

        # for name, param in self.llama_model.named_parameters():
        #     param.requires_grad = False
        print('Loading LLAMA Done')

        new_vocab_size = len(self.llama_tokenizer)
        if not self.no_new_vocab:
            self.llama_model.model.resize_token_embeddings(new_vocab_size) # Do we need to set requires_grad to True? ,  # TODO: Needs to be finetuned
            # self.llama_model.lm_head = nn.Linear(self.llama_model.config.hidden_size, new_vocab_size, bias=True, device=self.llama_model.device, dtype=torch.float32)
            self.llama_model.lm_head = nn.Linear(self.llama_model.config.hidden_size, new_vocab_size, bias=True, dtype=torch.float32)
        # if not freeze_embed_tokens:
            self.llama_model.model.embed_tokens = self.llama_model.model.embed_tokens.to(torch.float32)

        if self.lora:
                ### PEFT SETUP
                # self.llama_model.gradient_checkpointing_enable()
                lconfig = LoraConfig(
                    # inference_mode=False,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    )
                self.llama_model = get_peft_model(self.llama_model, lconfig)
                
                self.print_trainable_parameters(self.llama_model)

        if not self.no_new_vocab:
            for name, param in self.llama_model.named_parameters():
                if 'lm_head' in name or (not freeze_embed_tokens and 'embed_tokens' in name):
                # if 'lm_head' in name:
                    print("Finetuning:")
                    print(name)
                    param.requires_grad = True
                elif not self.lora:
                    param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id
        self.llama_model.generation_config.pad_token_id = self.llama_model.config.pad_token_id

        self.print_trainable_parameters(self.llama_model)
        
        # if 'Llama-2' in llama_model:
        self.llama2=True
        with open('<legacy_trajgpt_repo>/prompts/input_8.txt', 'r') as f:
            input_ = f.read()
        prompt_template = '[INST] <<SYS>>\nYou are a helpful motion tokens prediction assistant. Your answers should be abstract and accurate without explanation. Please ensure that your answer is based on all of the provided initial motion tokens of all of the agents.\n<</SYS>>\n\n{}[/INST] '
        self.prompt_list = [prompt_template.format(input_)]
        # else:
        #     self.prompt_list = []

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
    
    
    def parse_coordinates(self, data_, txt_seperator=" ", num_agents=None, obs_len=None, pred_len=None, obs_data=True):
        batch_size=data_.shape[0]
        x = data_[:,0,:]
        y = data_[:,1,:]
        # txt_seperator=", "     
        if self.no_new_vocab:
            xy = [
                self.motion_start_symbol +' '+ txt_seperator.join([f"{x[i][j]} {y[i][j]}" for j in range(len(x[i]))]).strip()+' '+ self.motion_end_symbol
                for i in range(batch_size)
                ]
        elif self.xy_identifier and not self.point_identifier:
            xy = [
                self.motion_start_symbol +' '+ txt_seperator.join([f"<x> <s-{x[i][j]}> </x> <y> <s-{y[i][j]}> </y>" for j in range(len(x[i]))]).strip()+' '+ self.motion_end_symbol
                for i in range(batch_size)
                ]
        elif self.point_identifier and not self.xy_identifier:
            xy = [
                self.motion_start_symbol +' '+ txt_seperator.join([f"<p> <s-{x[i][j]}> <s-{y[i][j]}> </p>" for j in range(len(x[i]))]).strip()+' '+ self.motion_end_symbol
                for i in range(batch_size)
                ]
        elif self.point_identifier and self.xy_identifier:
            xy = [
                self.motion_start_symbol +' '+ txt_seperator.join([f"<p> <x> <s-{x[i][j]}> </x> <y> <s-{y[i][j]}> </y> </p>" for j in range(len(x[i]))]).strip()+' '+ self.motion_end_symbol
                for i in range(batch_size)
                ]
        elif self.step_identifier:
            if obs_data:
                xy = [
                self.motion_start_symbol +' '+ txt_seperator.join([f"{self.step_identifier_symbols[j]} <s-{x[i][j]}> <s-{y[i][j]}>" for j in range(len(x[i]))]).strip()+' '+ self.motion_end_symbol
                for i in range(batch_size)
                ]
            else:
                xy = [
                self.motion_start_symbol +' '+ txt_seperator.join([f"{self.step_identifier_symbols[j+obs_len]} <s-{x[i][j]}> <s-{y[i][j]}>" for j in range(len(x[i]))]).strip()+' '+ self.motion_end_symbol
                for i in range(batch_size)
                ]
        else:
            xy = [
            self.motion_start_symbol +' '+ txt_seperator.join([f"<s-{x[i][j]}> <s-{y[i][j]}>" for j in range(len(x[i]))]).strip()+' '+ self.motion_end_symbol
            for i in range(batch_size)
            ]

        return xy
    
    def parse_traj_data(self, obs, pred, target_agent, min_grid, max_grid, num_agents=2):
        batch_size = obs.shape[0]
        
        target_agent_symbol = [f"<a-{target_agent_i}>" for target_agent_i in target_agent]
        # neighbour_agent = 1 if target_agent==0 else 0
        # neighbour_agent_symbol = f"<a-{neighbour_agent}>"
        min_grid_symbol = f"<s-{min_grid}>"
        max_grid_symbol = f"<s-{max_grid}>"
        
        # obs_data_list = [self.parse_coordinates(obs[:,0,:,:]),
        #     self.parse_coordinates(obs[:,1,:,:])]
        
        # obs_data_list = [self.parse_coordinates(obs[:,i,:,:], num_agents=num_agents) for i in range(obs.shape[1])]
        obs_data_list = [self.parse_coordinates(obs[batch_i,:num_agents[batch_i],:,:], obs_len=obs.shape[-1],  pred_len=pred.shape[-1]) for batch_i in range(batch_size)]
        
        # obs_data_list = []
        pred_data_list = [self.parse_coordinates(pred[batch_i,target_agent[batch_i],:,:].unsqueeze(0), obs_len=obs.shape[-1], pred_len=pred.shape[-1], obs_data=False) for batch_i in range(batch_size)]
            #TODO: Include init_traj
        return {
            '<obs_len>':obs.shape[-1], 
            '<pred_len>':pred.shape[-1], 
            '<target_agent_symbol>':target_agent_symbol, 
            # '<neighbour_agent_symbol>':neighbour_agent_symbol, 
            '<min_grid_symbol>':min_grid_symbol, 
            '<max_grid_symbol>':max_grid_symbol, 
            '<obs_data_list>':obs_data_list,
            '<pred_data_list>':pred_data_list,
            '<num_agents>': num_agents,
        }
    
    def _from_rel(self, rel, traj0):
    # traj0 is an initial x,y points for all trajectories
        traj = torch.zeros(rel.shape, dtype=rel.dtype, device=rel.device)
        # for i in range(traj.shape[-1]): #Temporal length
        #     traj[:,:,:,i:i+1] = traj0 + torch.sum(rel[:,:,:,:i+1], dim=-1, keepdim=True)
        for i in range(traj.shape[-1]): #Temporal length
            traj[:,:,:,i:i+1] = traj0 + torch.sum(rel[:,:,:,:i+1], dim=-1, keepdim=True)
        return traj

    def input_prompt_wrap(self, data_, prompt, device,
    place_holders = ['<obs_len>', '<pred_len>', '<target_agent_symbol>', '<min_grid_symbol>', '<max_grid_symbol>', '<obs_data_list>']):
        prompt = prompt.replace('<pred_len>', str(data_['<pred_len>']*2))
        batch_size = len(data_['<obs_data_list>'])

        p1 = prompt.split('<target_agent_symbol>')[0].strip(' ')
        p2 = prompt.split('<target_agent_symbol>')[1].split('<input_data>')[0].strip(' ')
        p3 = prompt.split('<target_agent_symbol>')[1].split('<input_data>')[1]

        obs_traj_data_list = []
        for batch_i in range(batch_size):
            obs_traj_data_list.append(''.join([f'<a-{agent_i}>:' +' '+ data_['<obs_data_list>'][batch_i][agent_i] + '\n' for agent_i in range(len(data_['<obs_data_list>'][batch_i]))]).strip('\n')+' ' + p3)

        traj_tokens = self.llama_tokenizer(obs_traj_data_list, return_tensors="pt", add_special_tokens=False, padding="longest").to(device) # tokens.input_ids.shape = torch.Size([batch_size, num_tokens=35]), num_tokens=num of spatial tokens + number of ',' symbols. Should we remove ',' ?
        # traj_tokens = self.llama_tokenizer(obs_traj_data_list, return_tensors="pt", add_special_tokens=False, max_length= self.max_txt_len, padding=True, truncation=True).to(device)
        if not self.lora:
            traj_embeds = self.llama_model.model.embed_tokens(traj_tokens.input_ids)
        else:
            traj_embeds = self.llama_model.model.model.embed_tokens(traj_tokens.input_ids)
        traj_atts=traj_tokens.attention_mask

     
        target_agent_num = [int(target_i.strip('<a->')) for target_i in data_['<target_agent_symbol>']]

        p1t = self.llama_tokenizer(p1, return_tensors="pt", add_special_tokens=False).to(traj_embeds.device)
        p1t_atts = p1t.attention_mask
        
        if self.no_new_vocab:
            target_agent_tokens = self.llama_tokenizer(data_['<target_agent_symbol>'], return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
        else:
            target_agent_tokens = self.llama_tokenizer(data_['<target_agent_symbol>'], return_tensors="pt", add_special_tokens=False).to(device) # tokens.input_ids.shape = torch.Size([batch_size, num_tokens=35]), num_tokens=num of spatial tokens + number of ',' symbols. Should we remove ',' ?
        
        target_agent_atts=target_agent_tokens.attention_mask

        p2t = self.llama_tokenizer(p2, return_tensors="pt", add_special_tokens=False).to(traj_embeds.device)
        p2t_atts = p2t.attention_mask



        if not self.lora:
            target_agent_embeds = self.llama_model.model.embed_tokens(target_agent_tokens.input_ids)
            p1e = self.llama_model.model.embed_tokens(p1t.input_ids).expand(batch_size, -1, -1)
            p2e = self.llama_model.model.embed_tokens(p2t.input_ids).expand(batch_size, -1, -1)

        else:
            target_agent_embeds = self.llama_model.model.model.embed_tokens(target_agent_tokens.input_ids)
            p1e = self.llama_model.model.model.embed_tokens(p1t.input_ids).expand(batch_size, -1, -1)
            p2e = self.llama_model.model.model.embed_tokens(p2t.input_ids).expand(batch_size, -1, -1)

        
        p1e = p1e.expand(batch_size, -1, -1)
        p2e = p2e.expand(batch_size, -1, -1)

        

        wrapped_embeds = torch.cat((p1e, target_agent_embeds, p2e, traj_embeds),dim=1)
        wrapped_atts =  torch.cat((p1t_atts.expand(batch_size,-1), target_agent_atts, p2t_atts.expand(batch_size,-1), traj_atts),dim=1)  
        tokens = torch.cat((p1t.input_ids.expand(batch_size,-1), target_agent_tokens.input_ids, p2t.input_ids.expand(batch_size,-1), traj_tokens.input_ids),dim=1)
        
        return wrapped_embeds, wrapped_atts, tokens

    def output_prompt_wrap(self, data_, prompt=None,
    place_holders = ['<target_agent_symbol>', '<pred_data_list>']):
        to_regress_text = [sample_i[0] + self.end_sym for sample_i in data_['<pred_data_list>']]
        return to_regress_text       

    def get_sample(self, samples, idx):
        return {list(samples.items())[sample_key][0]:list(samples.items())[sample_key][1][idx:idx+1] for sample_key in range(len(samples))}
        
    def forward(self, samples, _training=True):
        
        self.llama_tokenizer.padding_side = "right"

        obs = samples['rel_obs']
        pred = samples['rel_pred']
        obs_len = obs.shape[-1]
        pred_len = pred.shape[-1]
        # target_agent = np.random.binomial(1,0.5)
        target_agent = samples['target_agent']
        num_agents = samples['num_agents']
        min_grid=0
        # max_grid=512
        max_grid = self.grid_size
        parsed_data = self.parse_traj_data(obs, pred, target_agent, min_grid, max_grid, num_agents)
        prompt = random.choice(self.prompt_list)
        # to_device = self.llama_model.device
        to_device = obs.device
        input_embeds, input_atts, input_tokens = self.input_prompt_wrap(parsed_data, prompt, device = to_device)
        batch_size = input_embeds.shape[0]
        # print(f"GPU[{str(to_device)}]")
        if _training:
            text = self.output_prompt_wrap(parsed_data)

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(to_device)
            
            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            
            empty_targets = (
                torch.ones([input_atts.shape[0], input_atts.shape[1]+1],
                        dtype=torch.long).to(to_device).fill_(-100)  # plus one for bos
            )
            # empty_targets = (
            #     torch.ones([input_atts.shape[0], self.max_txt_len+1],
            #             dtype=torch.long).to(to_device).fill_(-100)  # plus one for bos
            # )
            targets = torch.cat([empty_targets, targets], dim=1)
            
            bos = torch.ones([batch_size, 1],
                            dtype=to_regress_tokens.input_ids.dtype,
                            device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id

            if self.lora:
                bos_embeds = self.llama_model.model.model.embed_tokens(bos)
                to_regress_embeds = self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
            else:
                bos_embeds = self.llama_model.model.embed_tokens(bos)
                to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

            atts_bos = input_atts[:, :1]
            embs = torch.cat([bos_embeds, input_embeds, to_regress_embeds], dim=1)
            # embs = torch.cat([bos_embeds, input_embeds[:,:int(self.max_txt_len)], to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, input_atts, to_regress_tokens.attention_mask], dim=1)
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
                bos = torch.ones([batch_size, 1],
                        dtype=input_tokens.dtype,
                        device=input_tokens.device) * self.llama_tokenizer.bos_token_id 
                if self.lora:
                    bos_embeds = self.llama_model.model.model.embed_tokens(bos)
                else:
                    bos_embeds = self.llama_model.model.embed_tokens(bos)
                embs = torch.cat([bos_embeds, input_embeds], dim=1)
                atts_bos = input_atts[:, :1]
                attention_mask = torch.cat([atts_bos, input_atts], dim=1)
                stop_words_ids = [torch.tensor([self.llama_tokenizer.eos_token_id]).to(input_tokens.device)]
                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
                num_beams = 2
                do_sample = False
                min_length = pred_len*2
                max_new_tokens = self.max_txt_len
                with self.maybe_autocast():
                    outputs = self.llama_model.generate(
                        inputs_embeds=embs,
                        attention_mask=attention_mask,
                        # max_new_tokens=max_new_tokens,
                        max_new_tokens=100,
                        stopping_criteria=stopping_criteria, #</motion> and </s>
                        # num_beams=num_beams,
                        # do_sample=do_sample,
                        # min_length=min_length,
                        early_stopping=True,
                        # pad_token_id = self.llama_tokenizer.pad_token_id,
                        # bos_token_id = 529, #<
                        # force_words_ids= [[self.motion_start_token], [self.motion_end_token]], #<motion> </motion>
                        # force_words_ids = [[529, 29885, 8194, 29958], [1533, 29885, 8194, 29958]],
                        use_cache = True,
                        # bad_words_ids = [[i] for i in list(self.original_vocab.values()) if i not in self.llama_tokenizer.all_special_ids + [self.space_token_id]], #Check if useful to use or not
                    )
                return outputs

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
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
