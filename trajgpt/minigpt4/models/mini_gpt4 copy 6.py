import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from transformers import LlamaTokenizer
from transformers import AutoTokenizer
# from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
from transformers import LlamaForCausalLM
# from minigpt4.models.custom_modeling_llama import LlamaForCausalLM
# <legacy_repo_root>/trajgpt/minigpt4/models/custom_modeling_llama.py
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModel
# from transformers.models.llama.modeling_llama import LlamaForCausalLM
# from transformers import BitsAndBytesConfig

import re
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from transformers import StoppingCriteria, StoppingCriteriaList
from minigpt4.tasks.image_text_pretrain import StoppingCriteriaSub

from traj_utils import *

# from sklearn.preprocessing import KBinsDiscretizer
import pickle

from minigpt4.models.gameformer_enc import * # we will use Encoder() from here
# from minigpt4.models import GameFormer
# from minigpt4.models import GameFormer_act as GameFormer
# from minigpt4.models.GameFormer_act_25mar import GameFormer_ as GameFormer
from gameformer.model.GameFormer import GameFormer_ as GameFormer
from minigpt4.models.storygpt_layers import TextFcLayer
from gf_utils import *
try:
    from transformers.trainer import _is_peft_model
except ImportError:
    _is_peft_model = None  # or define an alternative behavior
from accelerate import PartialState

# Helper function to modify parameters
def modify_parameters(model, parameter_name, add_value):
    for name, param in model.named_parameters():
        if parameter_name in name:
            print(f"Modifying {name}")  # Output which parameter is being modified
            original_value = param.clone()  # Store original value for comparison
            with torch.no_grad():
                param.add_(add_value)  # Add a value in-place
            return original_value, param.clone()  # Return both original and new values for comparison
    return None, None  # Return None if no parameter was modified

# Helper function to print parameter values
def print_parameter_values(model, parameter_name):
    found = False
    for name, param in model.named_parameters():
        if parameter_name in name:
            found = True
            print(f"{name}: {param.data}")
    if not found:
        print(f"Parameter {parameter_name} not found in the model.")

# from minigpt4.models.llama_patch import upcast_layer_for_flash_attention
# unplace_flash_attn_with_attn()
# from GameFormer_modules import CrossTransformer
class BinaryClassificationHead(nn.Module):
    def __init__(self, hidden_size):
        """
        A simple binary classification head for the CLS token.
        
        Args:
            hidden_size (int): The size of the hidden state of the language model.
        """
        super(BinaryClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # First layer with hidden size reduction
            nn.ReLU(),                                # Activation
            nn.Dropout(0.1),                          # Dropout for regularization
            nn.Linear(hidden_size // 2, 1),           # Final output layer
            # nn.Sigmoid()                              # Sigmoid for binary classification
        )
    
    def forward(self, cls_token):
        """
        Forward pass for the classification head.
        
        Args:
            cls_token (torch.Tensor): The CLS token embedding of shape (batch_size, hidden_size).
        
        Returns:
            torch.Tensor: Predicted probabilities of shape (batch_size, 1).
        """
        return self.classifier(cls_token)

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

    def __init__(self, cfg):
        super().__init__()
        # bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True,
        # #     bnb_4bit_quant_type="nf4",
        # #     bnb_4bit_compute_dtype=torch.bfloat16,
        # #     bnb_4bit_use_double_quant=True
        # )
        # self.llm = LlamaForCausalLM.from_pretrained(
        #             pretrained_model_name_or_path="meta-llama/Llama-2-7b-hf",
        #             torch_dtype=torch.bfloat16,
        #             # load_in_8bit=True,
        #             quantization_config=bnb_config,
        #             device_map={'':torch.cuda.current_device()},
        #             # device_map="auto",
        #             attn_implementation="flash_attention_2",
        #         )

        if cfg.run_cfg.get("quant_new", 4)==4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            # device_string = PartialState().process_index
            self.llm = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=cfg.model_cfg.llama_model,
                torch_dtype=torch.bfloat16,
                # low_cpu_mem_usage=True,
                # use_cache=True,
                # load_in_8bit=True,
                quantization_config=bnb_config,
                # attn_implementation="flash_attention_2",
                device_map={'':torch.cuda.current_device()},
                # device_map={'':device_string},
            )
        else:
            # bnb_model_from_pretrained_args = {}
            # bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # load_in_4bit=False,
            # load_in_8bit=True,
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=False,
            #     load_in_8bit=True,
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            # )
            # ))
            # bnb_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            # )
            self.llm = LlamaForCausalLM.from_pretrained(
                        pretrained_model_name_or_path=cfg.model_cfg.llama_model,
                        torch_dtype=torch.bfloat16,
                        # quantization_config=bnb_config,
                        device_map={'':torch.cuda.current_device()},
                        load_in_8bit=True,
                    )
        
        

    
    def __init_additional_modules__(
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
        freeze_gf_dec = True,
        motion_act = False,
        img_token_num=204,
        output_crossattn=True,
        gf_token_num=204,
        regress_act=False,
        instruct_act=True,
        caption=False,
        smpl_instruct=False,
        very_smpl_turn = False,
        gf_act_enc=False,
        gf_act_dec=False,
        full_map=False,
        eval_only=False,
        gf_level=3,
        two_agent=False,
        gmm_head_only=False,
        use_llama2_template=False,
        neighbors_to_predict=1,
        act_kv=False,
        late_fusion=False,
        use_flash_attention=False,
        upcast_to_bfloat16=False,
        resume_ckpt_path="",
        always_calculate_gt_traj=False,
        use_float32_embd=False,
        add_decision_cls=False,
        use_embed_mask=True,
        num_direction_classes=8,
        no_sys_prompt=False,
        auto_txt_formatter=False,
    ):
        self.use_mtr = False
        self.use_embed_mask=use_embed_mask
        self.add_decision_cls = add_decision_cls
        self.neighbors_to_predict = neighbors_to_predict
        # self.print_trainable_parameters(self.model)
        # super().__init__(AutoModelForCausalLM.from_pretrained().config)
        self.always_calculate_gt_traj = always_calculate_gt_traj
        # if resume_ckpt_path != "":
        #     llama_model = resume_ckpt_path.replace('.pth', '_llama_model')
        self.late_fusion = late_fusion
        self.use_llama2_template = use_llama2_template
        self.full_map = full_map
        self.two_agent = two_agent
        if gameformer_enc:
            TRAINABLE_PRECISION = torch.float32
            # TRAINABLE_PRECISION = torch.bfloat16
            # TRAINABLE_PRECISION = torch.bfloat16
            TRAINABLE_PRECISION_LLM = torch.bfloat16
            self.TRAINABLE_PRECISION_LLM = TRAINABLE_PRECISION_LLM
            # self.TRAINABLE_PRECISION_LLM = None
            # TRAINABLE_PRECISION = torch.float32
            # TRAINABLE_PRECISION_LLM = torch.float32
            # if eval_only:
            #     TRAINABLE_PRECISION = torch.bfloat16
            
            self.gameformer_enc = True
            print('loading GameFormer')
            self.gf_level = gf_level
            self.gameformer_model = GameFormer(
                modalities=6,
                encoder_layers=gf_encoder_layers,
                decoder_levels=gf_level,
                # future_len=16,
                future_len=80, 
                neighbors_to_predict=neighbors_to_predict,
                act=gf_act_enc,
                act_dec=gf_act_dec,
                num_act_classes=num_direction_classes,
                full_map=full_map,
                ego_act_only= not two_agent,
                act_kv=act_kv
            )
            if gf_encoder_path is not None:
                print("Load gameformer Checkpoint: {}".format(gf_encoder_path))
                ckpt = torch.load(gf_encoder_path, map_location="cpu")
                # for k,v in self.gameformer_model.named_parameters():
                #     print(k)
                # for k in ckpt['model_states'].keys():
                #     print(k)
                print('Loading GameFormer pretrained in the mini_gpt4 class init')
                msg = self.gameformer_model.load_state_dict(ckpt['model_states'])
                print(msg)
                self.act_kv = act_kv
                if act_kv:
                    self.gameformer_model.encoder_01.act_embedding = nn.Identity()
                    self.gameformer_model.encoder_01.act_kv = False # To avoid concatenating the act embedding in the encoder_01 forward call, it needs to be concatenated before applying the fusion in encoder_02
                    self.gameformer_model.decoder.act = False # As we will not apply conditioning query using the decoder, it will be applied through concatenating kv conditioning signal
                if gf_act_enc:
                    self.gameformer_model.encoder_02.act_encoder = nn.Identity()
                if gf_act_dec:
                    ## Used to test if error raise when having unused parameters without require grad
                    # print(self.gameformer_model.decoder.initial_stage.act_embedding)
                    # for _,weights_act_emb in self.gameformer_model.decoder.initial_stage.act_embedding.named_parameters():
                    #     weights_act_emb.requires_grad = False
                    self.gameformer_model.decoder.initial_stage.act_embedding = nn.Identity() # We will not need the condition embedding layer as we will feed that using our LLM
            self.gmm_head_only = gmm_head_only
            if gmm_head_only:
                self.gmm_head = self.gameformer_model.decoder.interaction_stage[-1].decoder
                # self.gameformer_model.decoder = nn.ModuleList()
        # heads_, dim_, dropout_ = 8, 256, 0.1
        # attention_layer = nn.TransformerEncoderLayer(d_model=dim_, nhead=heads_, dim_feedforward=dim_*4,
        #                                              activation=F.gelu, dropout=dropout_, batch_first=True)
        # self.fusion_encoder_temp = nn.TransformerEncoder(attention_layer, 1, enable_nested_tensor=False)
        
        self.very_smpl_turn = very_smpl_turn
        self.smpl_instruct = smpl_instruct
        self.caption = caption
        self.instruct_act = instruct_act
        self.gf_token_num = gf_token_num
        self.output_crossattn = output_crossattn
        self.motion_act = motion_act
        self.recon_loss_adapter = recon_loss_adapter
        self.recon_loss_llm = recon_loss_llm
        self.act = act
        if self.act and not gf_act_enc and not gf_act_dec and (img_token_num==6 or img_token_num==1) and not self.act_kv:
            self.gameformer_model.decoder.act = True
        self.rollouts = rollouts
        # with open(discretizer_model_path, 'rb') as file:
        #     self.discretizer = pickle.load(file)
         
        # self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
       
        print('Loading LLAMA')
        print(f"... Loading LLAMA tokenizer: {llama_model}")
        if auto_txt_formatter:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        elif 'Llama-3' in llama_model:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        elif 'Lite' in llama_model:
            self.llama_tokenizer = tokenizer = AutoTokenizer.from_pretrained(llama_model)
        else:
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, trust_remote_code=True)
        # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        self.llama_tokenizer.pad_token = '$$'
        # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        if False and low_resource:
            print('**1')
            # bnb_config_8bit = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     llm_int8_has_fp16_weight=False,
            # )
            if quant==8:
                # bnb_config = BitsAndBytesConfig(
                #     load_in_8bit=True)
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_model,
                    # torch_dtype=torch.bfloat16,
                    torch_dtype=TRAINABLE_PRECISION_LLM,
                    load_in_8bit=True,
                    # quantization_config = bnb_config,
                    # device_map={'':torch.cuda.current_device()},
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )
            elif quant==4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    # bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_compute_dtype=TRAINABLE_PRECISION_LLM,
                    # bnb_4bit_compute_dtype=torch.float16,
                )
                # if eval_only:
                #     self.llama_model = AutoModelForCausalLM.from_pretrained(
                #         llama_model,
                #         torch_dtype=torch.bfloat16,
                #         # torch_dtype=torch.float16,
                #         device_map={'':torch.cuda.current_device()},
                #         quantization_config=bnb_config,
                #         # attn_implementation="flash_attention_2" # use_flash_attention_2=True
                #     )
                # else:
                if use_flash_attention:
                    self.llama_model = AutoModelForCausalLM.from_pretrained(
                        llama_model,
                        # torch_dtype=torch.bfloat16,
                        torch_dtype=TRAINABLE_PRECISION_LLM,
                        # torch_dtype=torch.float16,
                        device_map={'':torch.cuda.current_device()},
                        # quantization_config=bnb_config,
                        use_cache=True,
                        attn_implementation="flash_attention_2", # use_flash_attention_2=True
                    )
                else:
                    
                    # resume_ckpt_path.replace('.pth', '_llama_model')
                    self.llama_model = AutoModelForCausalLM.from_pretrained(
                        llama_model,
                        torch_dtype=TRAINABLE_PRECISION_LLM,
                        device_map={'':torch.cuda.current_device()},
                        # quantization_config=bnb_config,
                        # use_cache=True,
                    )
            else:
                raise("Not implemented")
        else:
            if False:
                print('**1')
                print(f"... Loading LLAMA model: {llama_model}")
                if use_flash_attention:
                    print("> Using flash attention")
                    self.llama_model = AutoModelForCausalLM.from_pretrained(
                        llama_model,
                        # torch_dtype=torch.bfloat16,
                        torch_dtype=TRAINABLE_PRECISION_LLM,
                        device_map={'':torch.cuda.current_device()},
                        attn_implementation="flash_attention_2"
                    )
                else:
                    # config=AutoConfig.from_pretrained('meta-llama/Llama-2-7b-hf', vocab_size=32001)
                    # self.llama_model = AutoModelForCausalLM.from_pretrained(
                    #     llama_model,
                    #     torch_dtype=TRAINABLE_PRECISION_LLM,
                    #     device_map={'':torch.cuda.current_device()},
                    # )
                    self.llama_model = AutoModelForCausalLM.from_pretrained(
                        "meta-llama/Llama-2-7b-hf",
                        torch_dtype=TRAINABLE_PRECISION_LLM,
                        device_map={'':torch.cuda.current_device()},
                    )
                    # self.llama_model = AutoModel.from_pretrained(
                    #     llama_model,
                    #     torch_dtype=TRAINABLE_PRECISION_LLM,
                    #     device_map={'':torch.cuda.current_device()},
                    #     # is_trainable=True
                    # )
                    # self.llama_model_x = AutoModel.from_pretrained(
                    #     "meta-llama/Llama-2-7b-hf",
                    #     torch_dtype=TRAINABLE_PRECISION_LLM,
                    #     device_map={'':torch.cuda.current_device()},
                    # )
            print('> Loading LLAMA Done')
        
        
        if gameformer_enc:

            if freeze_gf or eval_only:
                self.gameformer_model.requires_grad_(False)
            else:
                # raise 'not implemented'
                if freeze_gf_enc:
                    self.gameformer_model.encoder_01.requires_grad_(False)
                    self.gameformer_model.encoder_02.requires_grad_(False)
                if freeze_gf_dec:
                    self.gameformer_model.decoder.requires_grad_(False)
                # if freeze_gf_dec:
                #     self.gameformer_model.decoder.requires_grad_(False)

            print('> loading GameFormer encoder Done')
            self.no_sys_prompt = no_sys_prompt
            # prompt_template = '<s>[INST] Predict the next sixteen tokens for both interactive agents based on the scene information and the provided three initial motion tokens. '
            prompt_template = '<s>[INST] Predict the future 8 seconds trajectory scene embeddings based on the 1 second observed trajectory scene embeddings of two interactive agents. '
            prompt_template_act = None
            if self.motion_act:
                prompt_template_act = '<s>[INST] Predict the future 8 seconds trajectory scene embeddings based on the 1 second observed trajectory scene embeddings of two interactive agents. Make the ego vehicle (the first agent) [turn], [move]. Please make sure to follow the provided turning and motion instruction of the ego. '
            if self.act:
                # prompt_template_act = '<s>[INST] Predict the future 8 seconds trajectory scene embeddings based on the 1 second observed trajectory scene embeddings of two interactive agents. Make the ego vehicle (the first agent) [turn]. Please make sure to follow the provided instruction. '
                prompt_template_act = '<s>[INST] Predict the future 8 seconds trajectory scene embeddings based on the 1 second observed trajectory scene embeddings of two interactive agents. '
            if self.smpl_instruct:
                prompt_template_act = '<s>[INST] Predict the future multimodal trajectory embeddings of two agents based on the observed scene embeddings and the ego instruction. Ego instruction: Make the ego vehicle '
                # prompt_template_act = '<s>[INST] Predict 8 seconds trajectory embeddings based on 1 second observed trajectory embeddings of two interactive agents. Make the egi vehicle '
            self.regress_act = regress_act
            if regress_act:
                self.output_prompt_act = 'The motion embeddings of the two interactive agents, with the ego agent following the instruction ([turn]) are: '
            self.output_prompt = 'The motion embeddings of the two interactive agents are: '


            self.input_end_text = ' [/INST] '
            self.prompt = prompt_template
            self.prompt_act = prompt_template_act

            self.scene_start_txt = '<scene_embeddings>'
            self.scene_end_txt = '</scene_embeddings>'
            # self.scene_start_txt = ['<Scene A>', '<Scene B>']
            # self.scene_end_txt = ['</Scene A>', '</Scene B>']


            ## Adapted code from MiniGPT-5
            # IMG_TOKEN_NUM = 228
            IMG_TOKEN_NUM = img_token_num #204
            
            self.IMG_TOKEN_NUM = IMG_TOKEN_NUM
            self.img_token_num = IMG_TOKEN_NUM

            # lm_hidden_size = self.llama_model.config.hidden_size
            lm_hidden_size = self.llm.config.hidden_size
            # lm_hidden_size = self.config.hidden_size
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
            if self.add_decision_cls:
                ALL_IMG_TOKENS = ALL_IMG_TOKENS + [f'[IMG{IMG_TOKEN_NUM}]'] # additional token
            self.ALL_IMG_TOKENS = ALL_IMG_TOKENS
            ALL_IMG_TOKENS_STR = "".join(ALL_IMG_TOKENS)
            # self.targets = f'{ALL_IMG_TOKENS_STR} </s>'
            self.targets = f"{ALL_IMG_TOKENS_STR} {self.llama_tokenizer.decode(self.llama_tokenizer.eos_token_id)}"
            
            all_img_tokens = ALL_IMG_TOKENS
            if all_img_tokens != self.llama_tokenizer.additional_special_tokens:
                self.num_new_tokens = self.llama_tokenizer.add_special_tokens(
                    {
                        "additional_special_tokens": all_img_tokens
                    }
                )
                # if self.num_new_tokens > 0:
            self.num_new_tokens = len(all_img_tokens)
            # resize, and use the average embedding to init the new embeddings
            # self.resize_token_embeddings(len(self.llama_tokenizer))
            self.llm.resize_token_embeddings(len(self.llama_tokenizer))
            # self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            

        self.lora = lora
        if self.lora:
            if resume_ckpt_path:
                self.llama_model = PeftModel.from_pretrained(
                    self.llama_model, resume_ckpt_path.replace('.pth', '_llama_model'),
                    is_trainable=True # 👈 here
                )
                # self.llama_model.load_adapter(resume_ckpt_path.replace('.pth', '_llama_model'), is_trainable=True)
                # self.llama_model = get_peft_model(self.llama_model)
            else:
                print('Loading LoRA')
                # self.llama_model = prepare_model_for_int8_training(self.llama_model)
                if low_resource:
                    self.llama_model.gradient_checkpointing_enable()
                    self.llama_model.config.use_cache=False
                    self.llama_model = prepare_model_for_kbit_training(self.llama_model)
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

                # # Set lm_head and embed_tokens modules requires_grad to True, no need if you will use modules_to_save in the LoRA config
                # for name, param in self.llama_model.named_parameters():
                #     if "lm_head" in name or "embed_tokens" in name:
                #         param.requires_grad = True
                #         print(f"{name} gradient: True")
                # self.llama_model.base_model.model.model.embed_tokens.original_module.weight.requires_grad = False
                # self.llama_model.base_model.model.lm_head.original_module.weight.requires_grad = False
                # self.llama_model.base_model.model.model.embed_tokens.original_module.weight.requires_grad = True
                # self.llama_model.base_model.model.lm_head.original_module.weight.requires_grad = True
        # else:
            
        print('Warning: When updating the modules to save (embed_tokens and lm_head) the peft wrapped modules_to_save parameters are only updated not the base model (original model), to use the orginial model with udpated weights you need to use model.merge_and_unload()')
        # input_embeddings = self.llama_model.get_input_embeddings().weight.data
        # output_embeddings = self.llama_model.get_output_embeddings().weight.data
        # input_embeddings = self.get_input_embeddings().weight.data
        # output_embeddings = self.get_output_embeddings().weight.data
        input_embeddings = self.llm.get_input_embeddings().weight.data
        output_embeddings = self.llm.get_output_embeddings().weight.data
        if not resume_ckpt_path: # if no checkpoint, initialize with the average embeddings
            input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
            # input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
            # output_embeddings[-self.num_new_tokens:] = output_embeddings_avg
        else:
            for name, param in self.llama_model.named_parameters():
                if "modules_to_save" in name:
                    print(name)
                    param.requires_grad = True
        

        # self.input_embeddings = self.llama_model.get_input_embeddings()
        # self.input_embeddings = self.get_input_embeddings()
        # self.input_embeddings = self.llm.get_input_embeddings()
            
        if len(ALL_IMG_TOKENS):
            self.output_img_id = self.llama_tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
            self.gen_token_idx = self.llama_tokenizer.convert_tokens_to_ids(self.ALL_IMG_TOKENS)

        ### lets avoid changing the precision for now
        # if use_float32_embd:
        #     self.llama_model.get_input_embeddings().to(TRAINABLE_PRECISION)
        #     self.llama_model.get_output_embeddings().to(TRAINABLE_PRECISION)
        # else:
        #     self.llama_model.get_input_embeddings().to(TRAINABLE_PRECISION_LLM)
        #     self.llama_model.get_output_embeddings().to(TRAINABLE_PRECISION_LLM)
        # # self.llama_model.get_input_embeddings().requires_grad_(True)
        # # self.llama_model.get_output_embeddings().requires_grad_(True)
        # if use_float32_embd:
        #     input_embed_grad_mask = torch.ones_like(self.llama_model.get_input_embeddings().weight.data).to(TRAINABLE_PRECISION)
        #     output_embed_grad_mask = torch.ones_like(self.llama_model.get_output_embeddings().weight.data).to(TRAINABLE_PRECISION)
        # else:
        #     input_embed_grad_mask = torch.ones_like(self.llama_model.get_input_embeddings().weight.data).to(TRAINABLE_PRECISION_LLM)
        #     output_embed_grad_mask = torch.ones_like(self.llama_model.get_output_embeddings().weight.data).to(TRAINABLE_PRECISION_LLM)
        
        # input_embed_grad_mask = torch.ones_like(self.get_input_embeddings().weight.data)
        # output_embed_grad_mask = torch.ones_like(self.get_output_embeddings().weight.data)
        input_embed_grad_mask = torch.ones_like(self.llm.get_input_embeddings().weight.data)
        output_embed_grad_mask = torch.ones_like(self.llm.get_output_embeddings().weight.data)

        # input_embed_grad_mask = torch.ones_like(self.llama_model.get_input_embeddings().weight.data)
        # output_embed_grad_mask = torch.ones_like(self.llama_model.get_output_embeddings().weight.data)
        
        input_embed_grad_mask[:-self.num_new_tokens] = 0
        output_embed_grad_mask[:-self.num_new_tokens] = 0
        self.register_buffer("input_embed_grad_mask", input_embed_grad_mask, persistent=False)
        self.register_buffer("output_embed_grad_mask", output_embed_grad_mask, persistent=False)
        # self.base_model_prepare_inputs_for_generation = super().prepare_inputs_for_generation
        # self.base_model_torch_dtype = self.llama_model.dtype
        # self.llama_model.output_img_id = self.output_img_id
        # self.llama_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

        # IS_STAGE2=False
        # if IS_STAGE2:
        #     for n, p in self.llm_output_adapter.named_parameters():
        #         p.requires_grad = False
        if self.add_decision_cls:
            self.decision_head = BinaryClassificationHead(lm_hidden_size)
        if eval_only:
            self.llama_model = self.llama_model.merge_and_unload()
            self.lora = False
            # if False:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            for name, param in self.llm_input_adapter.named_parameters():
                param.requires_grad = False
            for name, param in self.llm_output_adapter.named_parameters():
                param.requires_grad = False
        # else:
        # for name, param in self.llm.named_parameters():
        #     if "lm_head" in name or "embed_tokens" in name:
        #     # if "embed_tokens" in name:
        #         # print(name)
        #         param.requires_grad = False
                # print(f"{name}")
                # print(f"{param}")
        
        


                # else:
                #     param.requires_grad = False



            # self.llama_model = get_peft_model(self.llama_model, loraconfig)
            # print('> Loading LoRA Done')
            # self.print_trainable_parameters(self.llama_model)
            
        
        # self.llama_model.enable_input_require_grads()
        # self.llama_model.gradient_checkpointing_enable()
        # self.llama_model.config.use_cache = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.llm.config.pad_token_id = self.llama_tokenizer.pad_token_id
        # self.config.pad_token_id = self.llama_tokenizer.pad_token_id
        self.llm.generation_config.pad_token_id = self.llama_tokenizer.pad_token_id
        # self.generation_config.pad_token_id = self.llama_tokenizer.pad_token_id
        # self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id
        # self.llama_model.generation_config.pad_token_id = self.llama_model.config.pad_token_id
        # self.config.pad_token_id = self.llama_tokenizer.pad_token_id
        # self.generation_config.pad_token_id = self.config.pad_token_id
        self.llama2=True
        
        # if upcast_to_bfloat16: # no need for this, as things should be in bfloat16 already
        #     self.llama_model = upcast_layer_for_flash_attention(self.llama_model, torch.bfloat16)
        # if use_flash_attention:
            # self.llama_model = upcast_layer_for_flash_attention(self.llama_model, TRAINABLE_PRECISION_LLM)
        for name, param in self.llm.named_parameters():
            if "lm_head" in name or "embed_tokens" in name:
                param.requires_grad = False

        print('> Loading LoRA Done')
        print('llama:')
        # self.print_trainable_parameters(self.llama_model)
        # self.print_trainable_parameters(self.model)
        self.print_trainable_parameters(self.llm)
        print('gameformer:')
        # self.gameformer_model.to(self.model.embed_tokens.weight.device)
        self.gameformer_model.to(self.llm.model.embed_tokens.weight.device)
        self.print_trainable_parameters(self.gameformer_model)
        print('input_mapping:')
        # self.llm_input_adapter.to(self.model.embed_tokens.weight.device)
        self.llm_input_adapter.to(self.llm.model.embed_tokens.weight.device)
        self.print_trainable_parameters(self.llm_input_adapter)
        print('output_mapping:')
        # self.llm_output_adapter.to(self.model.embed_tokens.weight.device)
        self.llm_output_adapter.to(self.llm.lm_head.weight.device)
        self.print_trainable_parameters(self.llm_output_adapter)
        # self.model.layers[15].input_layernorm.weight.device
        # print('input fusion layer:')
        # self.print_trainable_parameters(self.fusion_encoder_temp)
        print('')
            

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for n, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def classification_processing(self, outputs, targets=None):
        # text_loss = outputs["loss"]
        last_hidden_state = outputs['hidden_states'][-1]
        special_token_index = outputs['output_token_index']
        t2i_input_embedding = []
        for i in range(len(special_token_index)):
            bs_id, seq_id = special_token_index[i]
            # similar to gf_post_processing, but selecting the next token (after the generation tokens) as the classifcation token
            t2i_input_embedding.append(last_hidden_state[bs_id:bs_id+1, seq_id+self.img_token_num:seq_id+self.img_token_num+1, :]) # get the embedding of the selected tokens
        if len(t2i_input_embedding) == 0:
            # loss = 0.01 * text_loss
            return -1 #{'loss': loss, 'text_loss': text_loss, 'image_loss': 0.0}
        else:
            t2i_input_embedding = torch.cat(t2i_input_embedding, dim=0)
            img_token_bs = t2i_input_embedding.shape[0]
            predictions = self.decision_head(t2i_input_embedding)
            if targets is not None:
                # Define the loss function
                criterion = nn.BCEWithLogitsLoss()
                # Compute binary cross-entropy loss directly
                loss = criterion(predictions.squeeze(-1).squeeze(-1), targets.float())
                return loss
            else:
                # Get probabilities
                probabilities = torch.sigmoid(predictions.squeeze(-1).squeeze(-1))  # Shape: [16]
                # Convert probabilities to class labels
                predicted_classes = (probabilities > 0.5).long()  # Shape: [16]
                # print(predicted_classes)
                return predicted_classes
        

    def gf_post_processing(self, outputs, context, input_adapted_context_embs, ego_future, neighbor_future, level=6, subsample=False, reduction='mean'):
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
                
                # if int(t2i_input_embedding.shape[1]/2) != int(self.gf_token_num/2):
                if False: # TODO: Check if needed, or the later part that is based on num_condition_tokens will take care of this.
                    query_act = t2i_input_embedding[:,0]
                    scene_act = t2i_input_embedding[:,1]
                    t2i_input_embedding_ = t2i_input_embedding[:,2:]
                    t2i_input_embedding_ = t2i_input_embedding_.reshape(t2i_input_embedding_.shape[0], 2, int(t2i_input_embedding_.shape[1]/2), t2i_input_embedding_.shape[-1])
                    gameformer_decoder_input = {'encodings': t2i_input_embedding_,'actors': context['actors'], 'masks': context['masks'], 'act': scene_act, 'query_act': query_act}
                    gameformer_decoder_input = self.gameformer_model.encoder_02(gameformer_decoder_input)
                    decoded_output = self.gameformer_model.decoder(gameformer_decoder_input)
                    loss, future, _ = level_k_loss(decoded_output, ego_future, neighbor_future, level, subsample=subsample)
                    # raise 'not implemented'
                # t2i_input_embedding = self.llm_output_mapping_fc(t2i_input_embedding) # shape: [batch, self.img_token_num, 256]
                ## transformer mapping that is used in minigpt5, we will skip in this implementation:
                # mapping_feature = self.llm_to_t2i_mapping(src=t2i_input_embedding, tgt=self.t2i_decoder_prompt.repeat(img_token_bs, 1, 1)) 
                # model.gameformer
                ## We need to use context['actors'], context['masks'], and map t2i_input_embedding to the same shape as context['encodings'] to be used by the decoder
            
                ## changing t2i_input_embedding.shape = [batch, self.img_token_num, 256] to context['encodings'].shape = [batch, 2, 114, 256]
                else:
                    # 25 march
                    num_condition_tokens = self.img_token_num - self.gf_token_num
                    if num_condition_tokens>0: # number of img tokens is larger the same as number of gf tokens, if we are generating all tokens
                        condition = t2i_input_embedding[:,:num_condition_tokens,:]
                    else:
                        condition= None
                    if self.gf_token_num==356:
                        predicted_context_encodings = t2i_input_embedding[:,num_condition_tokens:num_condition_tokens+228].reshape(t2i_input_embedding.shape[0],2,int(228/2),t2i_input_embedding.shape[-1])
                        predicted_fullmap_encodings = t2i_input_embedding[:,num_condition_tokens+228:]
                    elif self.neighbors_to_predict==0 and self.img_token_num==2:
                        num_condition_tokens = 1
                        condition = t2i_input_embedding[:,:num_condition_tokens,:]
                        t2i_input_embedding = t2i_input_embedding[:,num_condition_tokens:].unsqueeze(2)
                        predicted_context_encodings = t2i_input_embedding
                        predicted_fullmap_encodings = None
                    elif self.img_token_num==2 or self.img_token_num==4:
                        num_condition_tokens = 0
                        t2i_input_embedding = t2i_input_embedding[:,num_condition_tokens:].reshape(t2i_input_embedding.shape[0], 2, int(self.img_token_num/2), t2i_input_embedding.shape[-1])
                        predicted_context_encodings = t2i_input_embedding
                        predicted_fullmap_encodings = None
                    elif self.img_token_num==3 or self.img_token_num==5:
                        num_condition_tokens = 1
                        condition = t2i_input_embedding[:,:num_condition_tokens,:]
                        t2i_input_embedding = t2i_input_embedding[:,num_condition_tokens:].reshape(t2i_input_embedding.shape[0], 2, int((self.img_token_num-1)/2), t2i_input_embedding.shape[-1])
                        predicted_context_encodings = t2i_input_embedding
                        predicted_fullmap_encodings = None
                    elif self.img_token_num==6:
                        num_condition_tokens = 6
                        condition = t2i_input_embedding[:,:num_condition_tokens,:]
                        predicted_context_encodings = None
                        predicted_fullmap_encodings = None
                    elif self.img_token_num==1: # only single query generated (condition)
                            num_condition_tokens = 1
                            condition = t2i_input_embedding[:,:num_condition_tokens,:]
                            predicted_context_encodings = None
                            predicted_fullmap_encodings = None
                    elif  self.img_token_num==12:
                        raise 'Not implemented'
                    else:
                        t2i_input_embedding = t2i_input_embedding[:,num_condition_tokens:].reshape(t2i_input_embedding.shape[0],2,int(self.gf_token_num/2),t2i_input_embedding.shape[-1])
                        predicted_context_encodings = t2i_input_embedding
                        predicted_fullmap_encodings = None
                    # 5 march
                    # t2i_input_embedding = t2i_input_embedding.reshape(t2i_input_embedding.shape[0],2,int(t2i_input_embedding.shape[1]/2),t2i_input_embedding.shape[-1])
                    # predicted_context_encodings = t2i_input_embedding
                    # we need to change img_num_tokens to 114*2 in the begenning.
                    
                        # raise 'not impelemnted'
                        # predicted_context_encodings = self.fc_to_114(predicted_context_encodings.transpose(-1,-2)).transpose(-1,-2)
                    
                    # recon loss
                    if self.recon_loss_adapter: # for the gameformer dimension data
                        adapter_recon_loss = F.mse_loss(predicted_context_encodings, context['encodings'], reduction='mean')
                        recon_losses[1] = adapter_recon_loss
                    
                    if self.late_fusion and not self.act_kv:
                        context = self.gameformer_model.encoder_02(context) # late fusion
                    
                    if self.act_kv:
                        # act_kv_mask = inputs['act'][:,i]==-1
                        context['masks'] = torch.cat((context['masks'], torch.zeros_like(context['masks'][:,0:1,0:1])), dim=-1)
                        context['encodings'] = torch.cat((context['encodings'], condition.unsqueeze(2)), dim=-2)
                        context = self.gameformer_model.encoder_02(context) # late fusion
                        gameformer_decoder_input = {'encodings': context['encodings'],'actors': context['actors'], 'masks': context['masks'], 'act': None}
                    elif predicted_context_encodings is None and (condition.shape[1] == 6 or condition.shape[1] == 1):
                        gameformer_decoder_input = {'encodings': context['encodings'],'actors': context['actors'], 'masks': context['masks'], 'act': condition}
                    elif predicted_context_encodings.shape[2]==1:
                        # only changing the first index out of 114 features (encodings shape is [batch, 2, 114, 256]), corresponding to ego encodings for both agents views
                        predicted_context_encodings_ = torch.zeros_like(context['encodings'])
                        predicted_context_encodings_[:,:,1:,:] = context['encodings'][:,:,1:,:]
                        predicted_context_encodings_[:,0,0,:] = predicted_context_encodings[:,0,0]
                        if predicted_context_encodings_.shape[1]>1: # two agent
                            predicted_context_encodings_[:,1,0,:] = predicted_context_encodings[:,1,0]
                        gameformer_decoder_input = {'encodings': predicted_context_encodings_,'actors': context['actors'], 'masks': context['masks'], 'act': condition}
                    elif predicted_context_encodings.shape[2]==2:
                        predicted_context_encodings_ = torch.zeros_like(context['encodings'])
                        predicted_context_encodings_[:,:,2:,:] = context['encodings'][:,:,2:,:]
                        # Update ego view - ego and interactive agent encodings
                        predicted_context_encodings_[:,0,:2,:] = predicted_context_encodings[:,0,:2]
                        # Update interactive view - ego and interactive agent encodings
                        predicted_context_encodings_[:,1,:2,:] = predicted_context_encodings[:,1,:2]
                        gameformer_decoder_input = {'encodings': predicted_context_encodings_,'actors': context['actors'], 'masks': context['masks'], 'act': condition}
                    else:
                        ## We possibly can add predicted_context_encodings to original encodings, as risidual connection?, or possible also passing original encoding with another linear layer
                        gameformer_decoder_input = {'encodings': predicted_context_encodings,'actors': context['actors'], 'masks': context['masks'], 'act': condition}
                    if predicted_fullmap_encodings is not None:
                        gameformer_decoder_input.update({'full_map_encodings': predicted_fullmap_encodings, 'full_map_masks': context['full_map_masks']})
                    decoded_output = self.gameformer_model.decoder(gameformer_decoder_input)
                    loss, future, _ = level_k_loss(decoded_output, ego_future, neighbor_future, level, subsample=subsample, reduction=reduction, two_agent_reduction='none' if self.neighbors_to_predict>0 else 'yes')
            
            return loss, future, recon_losses
        
    def gf_post_processing_eval_new(self, t2i_input_embedding, context, input_adapted_context_embs, ego_future, neighbor_future, level=3, subsample=False):
        
        img_token_bs = t2i_input_embedding.shape[0]
        # recon loss
        recon_losses = [0,0]
                
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
            
            # 25 march
            num_condition_tokens = self.img_token_num - self.gf_token_num
            if num_condition_tokens>0:
                condition = t2i_input_embedding[:,:num_condition_tokens,:]
            else:
                condition= None
            if self.neighbors_to_predict==0 and self.img_token_num==2:
                num_condition_tokens = 1
                condition = t2i_input_embedding[:,:num_condition_tokens,:]
                t2i_input_embedding = t2i_input_embedding[:,num_condition_tokens:].unsqueeze(2)
                predicted_context_encodings = t2i_input_embedding
                predicted_fullmap_encodings = None
            elif self.img_token_num==2 or self.img_token_num==4:
                num_condition_tokens = 0
                t2i_input_embedding = t2i_input_embedding[:,num_condition_tokens:].reshape(t2i_input_embedding.shape[0], 2, int(self.img_token_num/2), t2i_input_embedding.shape[-1])
                predicted_context_encodings = t2i_input_embedding
                predicted_fullmap_encodings = None
            elif self.img_token_num==3 or self.img_token_num==5:
                num_condition_tokens = 1
                condition = t2i_input_embedding[:,:num_condition_tokens,:]
                t2i_input_embedding = t2i_input_embedding[:,num_condition_tokens:].reshape(t2i_input_embedding.shape[0], 2, int((self.img_token_num-1)/2), t2i_input_embedding.shape[-1])
                predicted_context_encodings = t2i_input_embedding
                predicted_fullmap_encodings = None
            elif self.img_token_num==6:
                num_condition_tokens = 6
                condition = t2i_input_embedding[:,:num_condition_tokens,:]
                predicted_context_encodings = None
                predicted_fullmap_encodings = None
            elif self.img_token_num==1: # only single query generated (condition)
                num_condition_tokens = 1
                condition = t2i_input_embedding[:,:num_condition_tokens,:]
                predicted_context_encodings = None
                predicted_fullmap_encodings = None
            elif  self.img_token_num==12:
                raise 'Not implemented'
            elif self.gf_token_num==356:
                predicted_context_encodings = t2i_input_embedding[:,num_condition_tokens:num_condition_tokens+228].reshape(t2i_input_embedding.shape[0],2,int(228/2),t2i_input_embedding.shape[-1])
                predicted_fullmap_encodings = t2i_input_embedding[:,num_condition_tokens+228:]
            else:
                t2i_input_embedding = t2i_input_embedding[:,num_condition_tokens:].reshape(t2i_input_embedding.shape[0],2,int(self.gf_token_num/2),t2i_input_embedding.shape[-1])
                predicted_context_encodings = t2i_input_embedding
                predicted_fullmap_encodings = None
            # 5 march
            # t2i_input_embedding = t2i_input_embedding.reshape(t2i_input_embedding.shape[0],2,int(t2i_input_embedding.shape[1]/2),t2i_input_embedding.shape[-1])
            # predicted_context_encodings = t2i_input_embedding
            # we need to change img_num_tokens to 114*2 in the begenning.
            
                # raise 'not impelemnted'
                # predicted_context_encodings = self.fc_to_114(predicted_context_encodings.transpose(-1,-2)).transpose(-1,-2)
            
            # recon loss
            if self.recon_loss_adapter: # for the gameformer dimension data
                adapter_recon_loss = F.mse_loss(predicted_context_encodings, context['encodings'], reduction='mean')
                recon_losses[1] = adapter_recon_loss
            
            if self.late_fusion and not self.act_kv:
                context = self.gameformer_model.encoder_02(context) # late fusion
            
            if self.act_kv:
                # act_kv_mask = inputs['act'][:,i]==-1
                context['masks'] = torch.cat((context['masks'], torch.zeros_like(context['masks'][:,0:1,0:1])), dim=-1)
                context['encodings'] = torch.cat((context['encodings'], condition.unsqueeze(2)), dim=-2)
                context = self.gameformer_model.encoder_02(context) # late fusion
                gameformer_decoder_input = {'encodings': context['encodings'],'actors': context['actors'], 'masks': context['masks'], 'act': None}
            elif predicted_context_encodings is None and (condition.shape[1] == 6 or condition.shape[1] == 1):
                gameformer_decoder_input = {'encodings': context['encodings'],'actors': context['actors'], 'masks': context['masks'], 'act': condition}
            elif predicted_context_encodings.shape[2]==1:
                # only changing the first index out of 114 features (encodings shape is [batch, 2, 114, 256]), corresponding to ego encodings for both agents views
                predicted_context_encodings_ = torch.zeros_like(context['encodings'])
                predicted_context_encodings_[:,:,1:,:] = context['encodings'][:,:,1:,:]
                predicted_context_encodings_[:,0,0,:] = predicted_context_encodings[:,0,0]
                if predicted_context_encodings_.shape[1]>1: # two agent
                    predicted_context_encodings_[:,1,0,:] = predicted_context_encodings[:,1,0]
                gameformer_decoder_input = {'encodings': predicted_context_encodings_,'actors': context['actors'], 'masks': context['masks'], 'act': condition}
            elif predicted_context_encodings.shape[2]==2:
                predicted_context_encodings_ = torch.zeros_like(context['encodings'])
                predicted_context_encodings_[:,:,2:,:] = context['encodings'][:,:,2:,:]
                # Update ego view - ego and interactive agent encodings
                predicted_context_encodings_[:,0,:2,:] = predicted_context_encodings[:,0,:2]
                # Update interactive view - ego and interactive agent encodings
                predicted_context_encodings_[:,1,:2,:] = predicted_context_encodings[:,1,:2]
                gameformer_decoder_input = {'encodings': predicted_context_encodings_,'actors': context['actors'], 'masks': context['masks'], 'act': condition}
            else:
                ## We possibly can add predicted_context_encodings to original encodings, as risidual connection?, or possible also passing original encoding with another linear layer
                gameformer_decoder_input = {'encodings': predicted_context_encodings,'actors': context['actors'], 'masks': context['masks'], 'act': condition}
            if predicted_fullmap_encodings is not None:
                gameformer_decoder_input.update({'full_map_encodings': predicted_fullmap_encodings, 'full_map_masks': context['full_map_masks']})
            decoded_output = self.gameformer_model.decoder(gameformer_decoder_input)
            loss, future, trajectories = level_k_loss(decoded_output, ego_future, neighbor_future, level, subsample=subsample)
            scores = decoded_output[f'level_{level}_scores']
        return loss, future, trajectories, scores

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
        
        # 25 march
        num_condition_tokens = self.img_token_num - self.gf_token_num
        if num_condition_tokens>0:
            condition = t2i_input_embedding[:,:num_condition_tokens,:]
        else:
            condition= None
        t2i_input_embedding = t2i_input_embedding[:,num_condition_tokens:].reshape(t2i_input_embedding.shape[0],2,int(self.gf_token_num/2),t2i_input_embedding.shape[-1])
        predicted_context_encodings = t2i_input_embedding
        gameformer_decoder_input = {'encodings': predicted_context_encodings,'actors': context['actors'], 'masks': context['masks'], 'act': condition}
        decoded_output = self.gameformer_model.decoder(gameformer_decoder_input)
        loss, future, trajectories = level_k_loss(decoded_output, ego_future, neighbor_future, level, subsample=subsample)
        scores = decoded_output[f'level_{level}_scores']
        return loss, future, trajectories, scores

            # print('')
    
#     def llama2_template_input(self,q):
#         sys_prompt = "You are a helpful driving assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
#         template = f"""<s>[INST] <<SYS>>
# {sys_prompt}
# <</SYS>>
    def llama2_template_input(self,q):
        sys_instruction = "### Instruction:\nYou are a safe AI driver."
        input_instruction = f"### Input:\n{q}"
        # answer_prompt = "### Answer:\n"
        # return f"{sys_instruction}\n\n{input_instruction}\n\n{answer_prompt}"
        return f"{sys_instruction}\n\n{input_instruction}\n\n### Scene embeddings:\n"
    
    def simple_llama2_template(self,q):
        return f"<s>[INST] {q} "
        # sys_instruction = "<s>[INST] "
        # input_instruction = f"### Input:\n{q}"
        # # answer_prompt = "### Answer:\n"
        # # return f"{sys_instruction}\n\n{input_instruction}\n\n{answer_prompt}"
        # return f"{sys_instruction}\n\n{input_instruction}\n\n### Scene embeddings:\n"

    def llama2_template_output(self,q,a,gen_tokens):
        sys_prompt = "You are a helpful driving assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
        template = f"""<s>[INST] <<SYS>>
{sys_prompt}
<</SYS>>

{q} [/INST] {a} {gen_tokens} </s>"""
        # template.replace("{{ system_prompt }}", "You are a helpful driving assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.")
        # template.replace("{{ user_msg_1 }}", q)
        # template.replace("{{ model_answer_1 }}", a)
        return template

    def llama2_template(self,q,a,gen_tokens):
        sys_prompt = "You are a helpful driving assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
        template = f"""<s>[INST] <<SYS>>
{sys_prompt}
<</SYS>>

{q} [/INST] {a} {gen_tokens} </s>"""
        # template.replace("{{ system_prompt }}", "You are a helpful driving assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.")
        # template.replace("{{ user_msg_1 }}", q)
        # template.replace("{{ model_answer_1 }}", a)
        return template

    # def forward(self, samples, _training=True):
    # def forward(self, samples, _training=True):
    def forward(self, input_ids, _training=True, **kwargs):

        embed_tokens = self.llm.model.model.embed_tokens if _is_peft_model(self.llm) else self.llm.model.embed_tokens
        embed_device = embed_tokens.weight.device

        if False: # dummy example to check model generation
            inference_llm = self.llm.merge_and_unload()
            
            sys_instruction = "### Instruction:\nYou are an AI driving assistant."
            input_instruction = "### Question: Is it safe to turn right at a speed of 140 km/h? and explain how he can do it"
            answer_prompt = "### Answer:\n"
            # input_text = f"{sys_instruction} {input_instruction}\n\n{answer_prompt}"
            input_text = f"{sys_instruction}\n\n{input_instruction}\n\n{answer_prompt}"
            # instruction = "### Instruction:\nYou are a helpful and knowledgeable AI assistant. Provide clear, concise, and accurate answers to the questions."
            # question = "### Question:\nDo you love me? (say no :))"
            # answer_prompt = "### Answer:\n"
            # input_text = f"{instruction}\n\n{question}\n\n{answer_prompt}"
            dummy_input = input_text
            # dummy_input = "[INST] «SYS»\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.\n«/SYS»\n\nHi, how are you? [/INST]"
            dummy_input = self.llama_tokenizer(dummy_input, return_tensors="pt", add_special_tokens=False)
            dummy_input_embds = embed_tokens(torch.tensor(dummy_input.input_ids, device=embed_tokens.weight.device))
            dummy_input_atts = dummy_input.attention_mask.to(dummy_input_embds.device)
            output_ids = inference_llm.generate(
            inputs_embeds= dummy_input_embds,
            max_new_tokens=100,  # Set maximum length of output
            )
            print(self.llama_tokenizer.decode(output_ids[0]))

        samples = input_ids
        local_batch_size = samples['traj'].shape[0]  # Size of the local batch
        # local_device = samples['traj'].device       # Device for the local batch

        # in case of data parallel, sometimes non-tensors of the global batch are not splitted into local batches
        # Ensure non-tensor fields are correctly split into local batches
        if local_batch_size < len(samples['instruct']):
            # Use 'scatter_idx' to select the appropriate subset
            for key, value in samples.items():
                if not isinstance(value, torch.Tensor) and key != "scatter_idx":
                    # Select subset for non-tensor fields using scatter_idx
                    samples[key] = [value[i] for i in samples['scatter_idx']]

        # if not _training:
        #     self.eval()
            
        self.llama_tokenizer.padding_side = "right"
        # device = samples['traj'].device
        batch_size = samples['traj'].shape[0]

        # print(samples['act'])
        # if self.act:
        if self.instruct_act:
            prompts_act = []
            prompts_act_out = []
            # template_data = []
            for i, instruct in enumerate(samples['instruct']):
                # if '<s>[INST] ' in instruct:
                if self.use_llama2_template:
                    llama2_template_data_q = instruct.strip().split('<s>[INST] ')[1]
                    prompts_act.append(self.llama2_template_input(llama2_template_data_q))
                    # llama2_template_data_a = samples['caption'][i].strip()
                    if not self.caption:
                        prompts_act_out.append("Generated embeddings: ")
                    else:
                        prompts_act_out.append(samples['caption'][i])
                    # llama2_template_data = self.llama2_template(q=llama2_template_data_q, a=llama2_template_data_a, gen_tokens=''.join(self.ALL_IMG_TOKENS))
                    # template_data.append(llama2_template_data)
                else:
                    llama2_template_data_q = instruct.strip().split('<s>[INST] ')[1]
                    if self.no_sys_prompt:
                        llama2_template_data_q = llama2_template_data_q.split('. ')[-1]
                    prompts_act.append(self.simple_llama2_template(llama2_template_data_q))
                    # llama2_template_data_a = samples['caption'][i].strip()
                    if not self.caption:
                        prompts_act_out.append("Generated embeddings: ")
                    else:
                        prompts_act_out.append(samples['caption'][i])
                    
                    # prompts_act.append(instruct)
                    # if self.smpl_instruct:
                    #     if self.motion_act:
                    #         sample_prompt = self.prompt_act+samples['all_captions'][3][i].split(',')[-1][1:]
                    #     else:
                    #         if self.very_smpl_turn:
                    #             sample_prompt = self.prompt_act+instruct+'.'
                    #             # turns = ['left', 'right', 'straight','not move','unknown movement']
                    #             # if 'not move' in samples['all_captions'][1][i].split(',')[-1][1:]:
                    #             #     sample_prompt = 'not move.'
                    #             #     print(sample_prompt)
                    #             # elif 'left' in samples['all_captions'][1][i].split(',')[-1][1:]:
                    #             #     sample_prompt = 'take a left turn.'
                    #             # elif 'right' in samples['all_captions'][1][i].split(',')[-1][1:]:
                    #             #     sample_prompt = 'take a right turn.'
                    #             # elif 'straight' in samples['all_captions'][1][i].split(',')[-1][1:]:
                    #             #     sample_prompt = 'move straight.'
                    #             # else:
                    #             #     sample_prompt = 'unknown movement'
                    #         else:
                    #             sample_prompt = self.prompt_act+samples['all_captions'][1][i].split(',')[-1][1:]
                    #     prompts_act.append(sample_prompt)
                    # else:
                    #     sample_prompt = self.prompt_act+instruct+' '
                    #     prompts_act.append(sample_prompt)
                    
                    # if self.caption:
                    #     prompts_act_out.append(samples['caption'][i])
                        # sample_prompt_out = samples['caption'][i] + '\n'+self.output_prompt
                        # prompts_act_out.append(sample_prompt_out)


            prompts_to_tokenize = prompts_act
            if self.caption:
                output_prompts_to_tokenize = prompts_act_out
                output_prompts_to_tokenize = [prompt_ + self.targets for prompt_ in output_prompts_to_tokenize]
            else:
                output_prompts_to_tokenize = [self.targets for _ in range(len(prompts_act))]
        else:
            prompts_act = []
            prompts_act_out = []
            # for i, instruct in enumerate(samples['instruct']):
            for i in range(len(samples['instruct'])):
                sample_prompt = self.prompt
                if self.caption:
                    sample_prompt_out = samples['caption'][i] + '\n'+self.output_prompt
                else:
                    sample_prompt_out = self.output_prompt
                prompts_act.append(sample_prompt)
                prompts_act_out.append(sample_prompt_out)

            prompts_to_tokenize = prompts_act
            output_prompts_to_tokenize = prompts_act_out
            output_prompts_to_tokenize = [prompt_ + self.targets for prompt_ in output_prompts_to_tokenize]
            # raise 'not implemented'
            if self.use_llama2_template:
                raise 'not implemented'
        # if self.use_llama2_template:
        #     prompts_to_tokenize = [template_data[i][:template_data[0].index('[/INST] ')+len('[/INST] ')] for i in range(len(template_data))]

        prompt_tokens = self.llama_tokenizer(prompts_to_tokenize, return_tensors="pt", add_special_tokens=False, padding="longest").to(embed_device)
        # prompt_tokens = self.llama_tokenizer(prompts_to_tokenize, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
        prompt_tokens_attention_mask = prompt_tokens.attention_mask
        
        prompt_embeds = embed_tokens(prompt_tokens.input_ids)

        if True:
            # target_ids_atts = torch.ones_like(target_ids)
            if self.gameformer_enc:
                start_scene_token = self.llama_tokenizer(self.scene_start_txt, return_tensors="pt", add_special_tokens=False).to(embed_device)
                if self.use_llama2_template:
                    end_scene_token = self.llama_tokenizer(self.scene_end_txt+'\n\n### Output:\n', return_tensors="pt", add_special_tokens=False).to(embed_device)
                else:
                    end_scene_token = self.llama_tokenizer(self.scene_end_txt+' [/INST]', return_tensors="pt", add_special_tokens=False).to(embed_device)
                    
                # start_scene_token = self.llama_tokenizer(self.scene_start_txt, return_tensors="pt", add_special_tokens=False).to(device)
                # end_scene_token = self.llama_tokenizer(self.scene_end_txt, return_tensors="pt", add_special_tokens=False).to(device)
                # with self.maybe_autocast(dtype=torch.bfloat16):
                    # with self.maybe_autocast():
                # with self.maybe_autocast(dtype=torch.bfloat16):
                if True:
                    gameformer_input = {key:value for key,value in samples.items() if key in ['ego_state', 'neighbors_state', 'map_lanes', 'map_crosswalks']}
                    if self.full_map:
                        gameformer_input.update({key:value for key,value in samples.items() if key in ['additional_map_lanes', 'additional_map_crosswalks', 'additional_boundaries', 'traffic_lights', 'stop_signs', 'speed_bumps']})
                    # gameformer_input['map_lanes'] = gameformer_input['map_lanes'][...,:200:2,:]
                    # gameformer_input['map_crosswalks'] = gameformer_input['map_crosswalks'][...,:100:2,:]
                    # context = self.gameformer_model.encoder_01(gameformer_input) # self.gameformer_model.encoder(gameformer_input)
                    if self.act_kv or self.late_fusion: # no encodings fusion
                        context = self.gameformer_model.encoder_01(gameformer_input)
                    else:
                        context = self.gameformer_model.encoder_02(self.gameformer_model.encoder_01(gameformer_input))
                    # if True: # TODO: Fix this later, if we need extra fusion layer, else ignore it
                    #     encodings_ = []
                    #     for e_i in range(2):
                    #         encodings_.append(self.fusion_encoder_temp(context['encodings'][:,e_i], src_key_padding_mask=context['masks'][:,e_i]))
                    #     encodings_ = torch.stack(encodings_, dim=1)
                    #     context['encodings'] = encodings_
                    
                    context_embs = context['encodings'] # [b,N:2, tokens:114, dim=256]
                    # context_embs = context_embs.masked_fill(context['masks'].unsqueeze(-1), 0.0) # mask it
                    context_embs_shape = context_embs.shape
                    context_embs = context_embs.reshape(context_embs.shape[0], -1, context_embs.shape[-1])
                    # if self.full_map:
                    #     additional_context_embs = context['full_map_encodings']
                    #     context_embs = torch.cat((context_embs, additional_context_embs), 1)
                    if self.in_adapter_per_feature:
                        context_embs_llm_maped = []
                        if (int(context_embs.shape[1]/2) == len(self.llm_input_adapter)):
                            # this checks if defined number of input adapters is twice the number of encodings, thus it process both agents views using same layers of a single agent view. If not then every single embedding is mapped seperatly. Here we follow the GameFormer design of having shared layers    
                            for adapter_i in range(len(self.llm_input_adapter)):
                                context_embs_llm_maped.append(self.llm_input_adapter[adapter_i](context_embs[:,adapter_i,:]))
                            for adapter_i in range(len(self.llm_input_adapter)):
                                context_embs_llm_maped.append(self.llm_input_adapter[adapter_i](context_embs[:,adapter_i+int(context_embs.shape[1]/2),:]))
                        else:
                            for adapter_i in range(len(self.llm_input_adapter)):
                                context_embs_llm_maped.append(self.llm_input_adapter[adapter_i](context_embs[:,adapter_i,:]))
                        context_embs = torch.stack(context_embs_llm_maped, dim=1)
                    else:
                        context_embs = self.llm_input_adapter(context_embs)
                    
                    # if not self.full_map:

                    combined_masks = context['masks'].reshape(context_embs.shape[0], -1)
                    context_atts = (~combined_masks).int()

                    
                    start_scene_embed = embed_tokens(start_scene_token.input_ids)
                    end_scene_embed = embed_tokens(end_scene_token.input_ids)
                    pad_embed = embed_tokens(torch.tensor([self.llm.config.pad_token_id], device=embed_device))

                    # context_embs = context_embs.to(prompt_embeds.dtype)
                    # try:
                    if pad_embed.dtype != context_embs.dtype:
                        pad_embed = pad_embed.to(context_embs.dtype)
                        context_embs = context_embs.to(pad_embed.dtype)
                    context_embs[combined_masks] = pad_embed # We can ad random embeddings masking here, or pad missing embeddings
                    # except:
                        # context_embs[combined_masks] = pad_embed.to(self.TRAINABLE_PRECISION_LLM)
                            # context_embs[combined_masks] = pad_embed.to(torch.float32)
                        # context_embs[combined_masks] = pad_embed.to(torch.bfloat16) # We can ad random embeddings masking here
                        # context_embs[context['masks'].reshape(context_embs.shape[0], -1)] = pad_embed.to(torch.bfloat16) # We can ad random embeddings masking here
                        # context_embs[context['full_map_masks']] = pad_embed.to(torch.bfloat16)
                    # context_embs[context['masks']] = pad_embed.to(torch.half)
                    # context_embs[context['masks']] = pad_embed
                    
            
            if self.gameformer_enc:
                input_embeds = torch.cat((
                    prompt_embeds,
                    expanded_stack(start_scene_embed[0], batch_size),
                    context_embs,
                    expanded_stack(end_scene_embed[0], batch_size),
                    # expanded_stack(start_scene_embed[1], batch_size),
                    # context_embs[:,1],
                    # expanded_stack(end_scene_embed[1],batch_size),
                    # obs_embeds
                    ), dim=1)
                    
                input_atts = torch.cat((
                    prompt_tokens_attention_mask,
                    expanded_stack(start_scene_token.attention_mask[0],batch_size),
                    context_atts,
                    expanded_stack(end_scene_token.attention_mask[0],batch_size),
                    # expanded_stack(start_scene_token.attention_mask[1],batch_size),
                    # context_atts[:,1],
                    # expanded_stack(end_scene_token.attention_mask[1],batch_size),
                    # obs_tokens.attention_mask
                    ), dim=1) 

                # if self.use_llama2_template:
                #     eos_tokens = self.llama_tokenizer(" [/INST] ", return_tensors="pt").to(self.model.embed_tokens.weight.device)
                #     # eos_tokens = self.llama_tokenizer(" [/INST] ", return_tensors="pt").to(device)
                    
                #     eos_embs = self.model.embed_tokens(eos_tokens.input_ids)
                #     # if not self.lora:
                #     #     eos_embs = self.llama_model.model.embed_tokens(eos_tokens.input_ids)
                #     # else:
                #     #     eos_embs = self.llama_model.model.model.embed_tokens(eos_tokens.input_ids)
                #     input_embeds = torch.cat((input_embeds, expanded_stack(eos_embs[0], batch_size)), dim=1)
                #     input_atts = torch.cat((input_atts, expanded_stack(eos_tokens.attention_mask[0], batch_size)), dim=1)

            # if _training:
            if True:
                ### MiniGPT-5 codes
                targets_ = self.llama_tokenizer(output_prompts_to_tokenize, return_tensors="pt", add_special_tokens=False, padding="longest").to(embed_device)
                # targets_ = self.llama_tokenizer(output_prompts_to_tokenize, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
                target_ids = targets_.input_ids
                targets_atts = targets_.attention_mask
                # target_ids = self.llama_tokenizer(self.targets, add_special_tokens=False).input_ids
                # target_ids = torch.tensor(target_ids)
                target_ids_ = target_ids
                output_img_id = self.output_img_id
                index = torch.nonzero(target_ids == output_img_id)
                if False: # enable this to set the generation tokens other than the first one to -100, attention needs to be changed to reflect this as well
                    if len(index):
                        for i in range(len(index)):
                            index_ = index[i][1]
                            target_ids_[i, index_+1:index_+self.IMG_TOKEN_NUM] = -100
                            targets_atts[i, index_+1:index_+self.IMG_TOKEN_NUM] = 0
                        # index = index[0][0]
                        # target_ids_[index+1:] = -100

                targets_tokens = target_ids
                
                # if True:
                #     targets_ = self.llama_tokenizer(output_prompts_to_tokenize, return_tensors="pt", add_special_tokens=False, padding="longest").to(device)
                #     targets_tokens = targets_.input_ids
                #     ontext_atts_reshaped = context_atts.reshape(context_atts.shape[0], -1)
                #     targets_atts = targets_.attention_mask
                # else:
                #     targets_ = self.llama_tokenizer(self.targets, return_tensors="pt", add_special_tokens=False).to(device)
                #     targets_tokens = expanded_stack(targets_.input_ids[0], batch_size)
                #     context_atts_reshaped = context_atts.reshape(context_atts.shape[0], -1)
                #     # targets_atts = torch.zeros(expanded_stack(targets_.attention_mask[0], batch_size).shape, device=device)
                #     targets_atts = torch.ones(expanded_stack(targets_.attention_mask[0], batch_size).shape, device=device)

                # empty_targets = (
                #     torch.ones([input_atts.shape[0], input_atts.shape[1]],
                #             dtype=torch.long).fill_(-100)
                # )

                empty_targets = (
                    torch.ones([input_atts.shape[0], input_atts.shape[1]],
                            dtype=torch.long).to(embed_device).fill_(-100)
                )
                targets = torch.cat([empty_targets, targets_tokens], dim=1)

                targets_embeds = embed_tokens(targets_tokens)
                # if not self.lora:
                #     targets_embeds = self.llama_model.model.embed_tokens(targets_tokens)
                # else:
                #     targets_embeds = self.llama_model.model.model.embed_tokens(targets_tokens)
                
                embs = torch.cat((input_embeds, targets_embeds),dim=1)
                attention_mask = torch.cat([input_atts, targets_atts], dim=1)
                # with self.maybe_autocast(dtype=torch.bfloat16):
                # with self.maybe_autocast():
                # with torch.backends.cuda.sdp_kernel(enable_flash=True):
                # if True:
                xxx = False
                if xxx:
                    xx = self.generate(input_embeds, attention_mask=input_atts, min_word_tokens=100, max_len=100)
                    # inference_llm = self.llm.merge_and_unload()
                    # output_ids = self.llm.model.generate(
                    # # output_ids = inference_llm.generate(
                    #     inputs_embeds= input_embeds,
                    #     attention_mask=input_atts,
                    #     max_new_tokens=100,  # Set maximum length of output
                    #     # temperature=1.0,
                    #     # top_p=0.9
                    # )
                    # print(self.llama_tokenizer.decode(output_ids[0]))
                if _training:
                    outputs = self.llm.forward(
                        inputs_embeds=embs,
                        attention_mask=attention_mask,
                        return_dict=True,
                        # labels=targets,
                        labels=targets,
                        output_hidden_states=True,
                    )
                    # outputs = self.llm.forward(
                    #     inputs_embeds=embs,
                    #     attention_mask=attention_mask,
                    #     return_dict=True,
                    #     # labels=targets,
                    #     labels=targets,
                    #     output_hidden_states=True,
                    # )
                    # outputs = super().forward(
                    #     inputs_embeds=embs,
                    #     attention_mask=attention_mask,
                    #     return_dict=True,
                    #     labels=targets,
                    #     output_hidden_states=True,
                    # )
                    # outputs = self.model(
                    #     inputs_embeds=embs,
                    #     attention_mask=attention_mask,
                    #     return_dict=True,
                    #     labels=targets,
                    #     output_hidden_states=True,
                    # )
                    # outputs = self.llama_model(
                    #     inputs_embeds=embs,
                    #     attention_mask=attention_mask,
                    #     return_dict=True,
                    #     labels=targets,
                    #     output_hidden_states=True,
                    # )
                    loss = outputs.loss
                    logits = outputs.logits
                    hidden_states = outputs.hidden_states
                    # num_words=500
                    # self.llm_generate(input_embeds, input_atts, num_words, stopping_criteria=None)
                    output_token_index = (targets == self.output_img_id).nonzero()
                    # if len(output_token_index):
                    #     addon_index = torch.ones_like(output_token_index)*(-1)
                    #     addon_index[:, 0] = 0
                    #     output_token_index += addon_index
                    ## targets[0][output_token_index[0][1]:] # this how the token index is used
                    outputs_dict = {"lm_loss": loss, "logits": logits, "hidden_states": hidden_states, "output_token_index": output_token_index}

                    reduction = 'none' if len(samples['contrastive_sample'])>0 else 'mean'
                    gf_loss, gf_future, recon_losses = self.gf_post_processing(outputs_dict, context, context_embs, ego_future=samples['ground_truth'][:,0,:,:], neighbor_future=samples['ground_truth'][:,1,:,:], level=self.gf_level, subsample=False, reduction=reduction)
                    if reduction=='none':
                        if not self.always_calculate_gt_traj: # mask out trajectories that are of infeasible/negative samples, with no ground truth trajectory reference matching the instruction.
                            if len(gf_loss.shape)>1:
                                gf_loss[:,0] = gf_loss[:,0]*~samples['contrastive_sample']
                                gf_loss = gf_loss.mean(-1)
                            else:
                                gf_loss = gf_loss*~samples['contrastive_sample']
                        if self.two_agent:
                            gf_loss.mean(-1)
                        gf_loss = gf_loss.mean()
        ########## end of cleaned code
                    # scaling_factor = gf_loss / (loss + 1e-8)  # Avoid division by zero
                    # scaling_factor = min(scaling_factor, 30)
                    if self.add_decision_cls:
                        cls_loss = self.classification_processing(outputs_dict, ~samples['contrastive_sample'])
                        total_loss = loss*10 + gf_loss + cls_loss*10
                        # total_loss = loss + gf_loss + cls_loss
                    else:
                        total_loss = loss*10 + gf_loss
                        # total_loss = loss + gf_loss
                    # total_loss = loss + gf_loss
                    # total_loss = loss + gf_loss
                    # total_loss = loss + gf_loss*0

                    # total_loss = loss*30 + gf_loss # scale up the llm loss by 30, to make it comparable to gf_loss
                    if recon_losses[0]>0:
                        total_loss+=recon_losses[0]
                    if recon_losses[1]>0:
                        total_loss+=recon_losses[1]

                    outputs_dict.update({"gf_loss": gf_loss, "gf_future": gf_future, "loss": total_loss, "llm_recon_loss":recon_losses[0], "gf_recon_loss": recon_losses[1]})
                    # outputs_dict.update({"gf_loss": gf_loss, "gf_future": gf_future, "loss": loss})
                    # raise 'Fake error'
                    # self.reset_embeddings()
                    outputs_dict = {"loss": total_loss}
                    return outputs_dict
            # else: # not training
                # with self.maybe_autocast():
                # if True:
                else:

                    text_out, output_embeddings = self.old_generate(input_embeds, input_atts, 100)
                    gf_loss, gf_future, trajectories, gf_scores = self.gf_post_processing_eval_new(output_embeddings, context, context_embs, ego_future=samples['ground_truth'][:,0,:,:], neighbor_future=samples['ground_truth'][:,1,:,:], level=self.gf_level)
                    # tokens_out, output_embeddings, output_logits, special_tokens_embeds, text_out  = self.generate(input_embeds, attention_mask=input_atts, min_word_tokens=100, max_len=100)
                    # gf_loss, gf_future, trajectories, gf_scores = self.gf_post_processing_eval_new(special_tokens_embeds, context, context_embs, ego_future=samples['ground_truth'][:,0,:,:], neighbor_future=samples['ground_truth'][:,1,:,:])  
                    # print(gf_loss)
                    if False:
                        for ii in range(len(tokens_out)):
                            print('---')
                            print(prompts_act[ii])
                            # print(self.llama_tokenizer.batch_decode(tokens_out)[ii])
                            print(text_out[ii])
                            print(f">>> GT: {prompts_act_out[ii]}")

                    return {'gf_loss': gf_loss,'output_traj': gf_future[0], 'modalities': trajectories, 'scores': gf_scores, 'text': text_out}
            
        # else:
        #     return self.predict(samples)
    def generate_dummy(self):
        dummy_input = "[INST] «SYS»\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.\n«/SYS»\n\nHi, how are you? [/INST]"
        dummy_input = self.llama_tokenizer(dummy_input, return_tensors="pt", add_special_tokens=False)
        dummy_input_embds = self.model.embed_tokens(torch.tensor(dummy_input.input_ids, device=self.model.embed_tokens.weight.device))
        dummy_input_atts = dummy_input.attention_mask.to(dummy_input_embds.device)
        # self.llm_generate(dummy_input_embds, dummy_input_atts, num_words=None, stopping_criteria=None)
        self.llm_generate(dummy_input_embds, None, num_words=None, stopping_criteria=None)
        print()

    def generate(self, embeddings=torch.FloatTensor, max_len: int = 15,
                temperature: float = 0.0, top_p: float = 1.0, min_word_tokens: int = 5,
                filter_value: float = -float('Inf'), attention_mask=None):
        """Runs greedy decoding and returns generated captions.

        Args:
            min_word_tokens: Minimum number of words to generate before allowing a [IMG] output.
            filter_value: Value to assign to tokens that should never be generated.
        Outputs:
            out: (N, T) int32 sequence of output tokens.
            output_embeddings: (N, T, 256) sequence of text output embeddings.
        """
        input_embeds_len = embeddings.shape[1]
        self.eval()
        self.llm.eval()
        generation_output_embeddings = torch.zeros_like(embeddings[:,:1,:])
        with torch.no_grad():  # no tracking history
            # init output with image tokens
            out = None
            output_embeddings = []
            output_logits = []
            # max_len = min_word_tokens+2
            idx_first_observed_generation_token = [100000]*embeddings.shape[0]
            found_all_generation_tokens = [False]*embeddings.shape[0]
            if True:
                output = self.llm.generate(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    max_new_tokens=max_len,
                    # stopping_criteria=stopping_criteria, #</motion> and </s>
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
                found_generation_token = [32000 in output.sequences[i] for i in range(len(output.sequences))]
                if False in found_generation_token:
                    for i in range(4): # retry four times
                        found_generation_token = [32000 in output.sequences[i] for i in range(len(output.sequences))]
                        if False in found_generation_token:
                        # if self.targets not in self.llama_tokenizer.decode(output.sequences[0]):
                            # retry
                            output = self.llm.generate(
                                inputs_embeds=embeddings,
                                attention_mask=attention_mask,
                                max_new_tokens=max_len,
                                # stopping_criteria=stopping_criteria, #</motion> and </s>
                                return_dict_in_generate=True,
                                output_hidden_states=True,
                            )
                        else:
                            break
                
                selected_out_text = self.llama_tokenizer.batch_decode(output.sequences)
                # selected_out_text = self.llama_tokenizer.decode(output.sequences[0])
                found_generation_token = [32000 in output.sequences[i] for i in range(len(output.sequences))]
                # Get the index of 32000 within each sequence if found
                idx_found_generation_token_dict = {
                    i: (seq == 32000).nonzero(as_tuple=True)[0].item() if (seq == 32000).any() else False
                    for i, seq in enumerate(output.sequences)
                }
                
                input_embeddings = self.llm.model.model.embed_tokens if _is_peft_model(self.llm) else self.llm.model.embed_tokens
                for k,v in idx_found_generation_token_dict.items():
                    if v:
                        generation_output_embeddings[k] = output.hidden_states[v][-1][k]
                    else: # if not found, force generation
                        new_embeds = torch.cat([output.hidden_states[i][-1] for i in range(len(output.hidden_states))], 1)
                        new_embeds = new_embeds[k].unsqueeze(0) # for one example
                        new_atts = torch.ones((attention_mask.shape[0], new_embeds.shape[1]), device=new_embeds.device)
                        new_atts[:,:attention_mask.shape[1]] = attention_mask
                        new_atts = new_atts[k].unsqueeze(0)
                        output = self.llm.forward(
                            inputs_embeds=new_embeds, 
                            attention_mask=new_atts, 
                            use_cache=False, 
                            output_hidden_states=True)
                        output.hidden_states[-1]
                        logits = output.logits[:, -1, :]  # (N, vocab_size)
                        if top_p == 1.0:
                            logits = logits.cpu()
                        # Force the generation of self.retrieval_token_idx
                        mask = torch.ones_like(logits, dtype=torch.bool)
                        mask[:, self.gen_token_idx] = False
                        logits[mask] = filter_value
                        next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                        next_token = next_token.long().to(embeddings.device)
                        next_embedding = input_embeddings(next_token)
                        generation_output_embeddings[k] = next_embedding # the forced embedding, please check
                
                out, output_embeddings, output_logits = None, None, None
                return out, output_embeddings, output_logits, generation_output_embeddings, selected_out_text
                if False:
                # if False in found_generation_token:
                # if self.targets not in self.llama_tokenizer.decode(output.sequences[0]):
                    new_embeds = torch.cat([output.hidden_states[i][-1] for i in range(len(output.hidden_states))], 1)
                    new_atts = torch.ones((attention_mask.shape[0], new_embeds.shape[1]), device=new_embeds.device)
                    new_atts[:,:attention_mask.shape[1]] = attention_mask
                    output = self.llm.forward(
                        inputs_embeds=new_embeds, 
                        attention_mask=new_atts, 
                        use_cache=False, 
                        output_hidden_states=True)
                    output.hidden_states[-1]
                    logits = output.logits[:, -1, :]  # (N, vocab_size)
                    if top_p == 1.0:
                        logits = logits.cpu()
                    # Force the generation of self.retrieval_token_idx
                    mask = torch.ones_like(logits, dtype=torch.bool)
                    mask[:, self.gen_token_idx] = False
                    logits[mask] = filter_value
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                    next_token = next_token.long().to(embeddings.device)
                    input_embeddings = self.llm.model.model.embed_tokens if _is_peft_model(self.llm) else self.llm.model.embed_tokens
                    next_embedding = input_embeddings(next_token)
                    generation_output_embeddings = next_embedding # the forced embedding
                else:
                    ## Only support one token generation
                    generation_output_embeddings = output.hidden_states[int(np.where(output.sequences[0].to('cpu') == 32000)[0][0])][-1]
                out, output_embeddings, output_logits = None, None, None
                return out, output_embeddings, output_logits, generation_output_embeddings, selected_out_text
            force_generation_ids = None
            for i in range(max_len):
                if attention_mask is not None:
                    if embeddings.shape[1]!=attention_mask.shape[1]:
                        # extend the mask with ones to count for newly generated and added toknes
                        new_att = torch.ones((embeddings.shape[0], embeddings.shape[1] - attention_mask.shape[1]), device=embeddings.device)
                        attention_mask = torch.cat((attention_mask, new_att), dim=1)
                    output = self.llm.forward(
                        inputs_embeds=embeddings, 
                        attention_mask=attention_mask, 
                        use_cache=False, 
                        output_hidden_states=True)
                    # output = super().forward(
                    #     inputs_embeds=embeddings, 
                    #     attention_mask=attention_mask, 
                    #     use_cache=False, 
                    #     output_hidden_states=True)
                    # output = self.model(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False, output_hidden_states=True)
                    # output = self.llama_model(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False, output_hidden_states=True)
                else:
                    output = self.llm.forward(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                    # output = super().forward(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                    # output = self.model(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                    # output = self.llama_model(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)

                text_emb_layers = [-1]
                for idx in text_emb_layers:
                    output_embeddings.append(output.hidden_states[idx])

                logits = output.logits[:, -1, :]  # (N, vocab_size)
                if top_p == 1.0:
                    logits = logits.cpu()
                output_logits.append(logits)


                
                # Prevent the model from generating the [IMG1..n] tokens.
                # logits[:, self.retrieval_token_idx[1:]] = filter_value
                logits[:, self.gen_token_idx[1:]] = filter_value
                self.retrieval_token_idx = self.gen_token_idx

                if (self.retrieval_token_idx or self.gen_token_idx) and self.retrieval_token_idx[0] != -1 and \
                        self.gen_token_idx[0] != -1:
                        if i < min_word_tokens:
                            # Eliminate probability of generating [IMG] if this is earlier than min_word_tokens.
                            # logits[:, self.retrieval_token_idx] = filter_value
                            # logits[:, self.gen_token_idx] = filter_value
                            pass
                        else:
                            # Force the generation of self.retrieval_token_idx
                            mask = torch.ones_like(logits, dtype=torch.bool)
                            # mask = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
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
                # elif next_token.shape[0] > 1:
                #     next_token_batch = []
                #     for batch_i in range(next_token.shape[0]):
                #         if next_token[batch_i].item() == self.retrieval_token_idx[0]:
                #             assert self.retrieval_token_idx == self.gen_token_idx, (
                #             self.retrieval_token_idx, self.gen_token_idx)
                #             next_token_batch.append(torch.tensor(self.retrieval_token_idx)[None, :].long().to(
                #                 embeddings.device))  # (1, num_tokens)
                #         else:
                #             next_token = next_token.long().to(embeddings.device)
                #     assert len(next_token_batch) == logits.shape[0] # if not, need special handling where not all batch examples we generated tokens for
                #     next_token = torch.stack(next_token_batch, dim=0)[:,0] # (N, num_tokens)
                else:
                    next_token = next_token.long().to(embeddings.device)

                if out is not None:
                    out = torch.cat([out, next_token], dim=-1)
                else:
                    out = next_token

                # input_embeddings = self.llm.get_input_embeddings()
                input_embeddings = self.llm.model.model.embed_tokens if _is_peft_model(self.llm) else self.llm.model.embed_tokens
                next_embedding = input_embeddings(next_token)
                embeddings = torch.cat([embeddings, next_embedding], dim=1)
                embeddings = embeddings[:, max(0, embeddings.shape[1] - 2048):] # Is this needed?
                
                ## Detecting the first found generation token [IMG0]
                found_generation_token = [last_found_txt in ['[IMG0]'] for last_found_txt in self.llama_tokenizer.batch_decode(next_token)]
                if True in found_generation_token:
                    for found_i, its_found in enumerate(found_generation_token):
                        if its_found and i<idx_first_observed_generation_token[found_i]:
                            idx_first_observed_generation_token[found_i] = i
                            found_all_generation_tokens[found_i] = True
                ## If end of scentence token of padding found in all examples then stop
                end_critria = [last_found_txt in ['</s>', '$$'] for last_found_txt in self.llama_tokenizer.batch_decode(next_token)]
                
                if False not in end_critria:
                    if not (False in found_all_generation_tokens):
                        break
                    else:
                        ## in case no generation token found and stopping criteria found, force generation before stopping
                        ## Note that here we only force the model to generate 
                        ## Need fix: Ensure the model outputs "[IMG0] </s>" Not only "[IMG0]", to match the model traing data format, as the hidden state of [IMG0] will be different
                        force_generation_ids = [force_id for force_id in range(len(found_all_generation_tokens)) if not found_all_generation_tokens[force_id]]
                        if attention_mask is not None:
                            if embeddings.shape[1]!=attention_mask.shape[1]:
                                # extend the mask with ones to count for newly generated and added toknes
                                new_att = torch.ones((embeddings.shape[0], embeddings.shape[1] - attention_mask.shape[1]), device=embeddings.device)
                                attention_mask = torch.cat((attention_mask, new_att), dim=1)
                            output = self.llm.forward(
                                inputs_embeds=embeddings, 
                                attention_mask=attention_mask, 
                                use_cache=False, 
                                output_hidden_states=True)
                        else:
                            output = self.llm.forward(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                        text_emb_layers = [-1]
                        for idx in text_emb_layers:
                            output_embeddings.append(output.hidden_states[idx])
                        logits = output.logits[:, -1, :]  # (N, vocab_size)
                        assert top_p == 1.0 and temperature == 0.0
                        if top_p == 1.0:
                            logits = logits.cpu()
                        output_logits.append(logits)

                        # logits[force_generation_ids, self.gen_token_idx[1:]] = filter_value
                        self.retrieval_token_idx = self.gen_token_idx
                        mask = torch.ones_like(logits, dtype=torch.bool)
                        ## Only unmask examples with no found generation token
                        mask[force_generation_ids, self.retrieval_token_idx] = False
                        ## Unmask all logits for examples with already found generation token, so the model most probably will output EOS or PAD token
                        mask[[k for k in range(mask.shape[0]) if k not in force_generation_ids], :] = False 
                        logits[mask] = filter_value
                        next_token = torch.argmax(logits, keepdim=True, dim=-1)
                        next_token = next_token.long().to(embeddings.device)
                        out = torch.cat([out, next_token], dim=-1)

                        input_embeddings = self.llm.model.model.embed_tokens if _is_peft_model(self.llm) else self.llm.model.embed_tokens
                        next_embedding = input_embeddings(next_token)
                        embeddings = torch.cat([embeddings, next_embedding], dim=1)
                        embeddings = embeddings[:, max(0, embeddings.shape[1] - 2048):] # Is this needed?

                        found_generation_token = [last_found_txt in ['[IMG0]'] for last_found_txt in self.llama_tokenizer.batch_decode(next_token)]
                        if True in found_generation_token:
                            for found_i, its_found in enumerate(found_generation_token):
                                if its_found and (i+1)<idx_first_observed_generation_token[found_i]:
                                    idx_first_observed_generation_token[found_i] = i+1
                                    found_all_generation_tokens[found_i] = True
                    break
        
        # generation_output_embeddings = generation_output_embeddings
        generation_output_embeddings = torch.zeros((embeddings.shape[0], 1, embeddings.shape[2]), device=embeddings.device) # consider increasing the 1 to the number of tokens if more than 1 token
        selected_out_text = []
        for i in range(len(out)):
            ## The index correspond to first observing "[IMG0] </s>" --> idx_first_observed_generation_token[i]+2
            selected_embd_idx = idx_first_observed_generation_token[i]+2 
            selected_embd_idx = min(selected_embd_idx ,len(output_embeddings)-1)
            target_embed = output_embeddings[selected_embd_idx][i]
            ## shifting by the length of input embeddings
            special_token_idx = idx_first_observed_generation_token[i] + input_embeds_len-1
            generation_output_embeddings[i] = target_embed[special_token_idx]
            # print('---'*10)
            # print('---'*10)
            # print(self.llama_tokenizer.decode(out[i]))
            ## NOTE check the following printing to see how we select output_embeddings, based on when we first observe "[IMG0] <s>", becsause we select [IMG0] in that given scenario
            selected_out_text.append(self.llama_tokenizer.decode(self.llm.lm_head(output_embeddings[idx_first_observed_generation_token[i]+2])[:,input_embeds_len-1:].argmax(-1)[i]))
            # print(self.llama_tokenizer.decode(self.llm.lm_head(output_embeddings[idx_first_observed_generation_token[i]+2])[:,input_embeds_len-1:].argmax(-1)[i]))
            ## NOTE check the following to see the selected tokens (they should correspond to the special tokens most of the time, if no forced generation)
            # print(self.llama_tokenizer.decode(self.llm.lm_head(generation_output_embeddings.to('cuda')).argmax(-1)))
        return out, output_embeddings, output_logits, generation_output_embeddings, selected_out_text

    def continue_generate(self, tokens_to_force_generating, embeddings=torch.FloatTensor, #max_len: int = 288,
                temperature: float = 0.0, top_p: float = 1.0, #min_word_tokens: int = 0,
                filter_value: float = -float('Inf'), attention_mask=None):
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
            for i in range(2): # We generate one token in the first iteration, then all in the second
                output = self.llm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                text_emb_layers = [-1]
                for idx in text_emb_layers:
                    output_embeddings.append(output.hidden_states[idx])

                logits = output.logits[:, -1, :]  # (N, vocab_size)
                if top_p == 1.0:
                    logits = logits.cpu()
                output_logits.append(logits)

                # Prevent the model from generating other than the first [IMG] token in tokens_to_force_generating.
                logits[:, tokens_to_force_generating[1:]] = filter_value
                logits[:, tokens_to_force_generating[1:]] = filter_value

                mask = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
                mask[:, tokens_to_force_generating] = False # we might change this, to a selected value
                logits[mask] = filter_value

                if temperature == 0.0:
                    if top_p != 1.0:
                        raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                else:
                    raise 'Not implemented'
                
                if next_token.shape[0] == 1 and next_token.item() == tokens_to_force_generating[0]:
                    next_token = torch.tensor(tokens_to_force_generating)[None, :].long().to(
                        embeddings.device)  # (1, num_tokens)

                if out is not None:
                    out = torch.cat([out, next_token], dim=-1)
                else:
                    out = next_token
                input_embeddings = self.llm.model.model.embed_tokens if _is_peft_model(self.llm) else self.llm.model.embed_tokens
                next_embedding = input_embeddings(next_token)
                embeddings = torch.cat([embeddings, next_embedding], dim=1)
                embeddings = embeddings[:, max(0, embeddings.shape[1] - 2048):]

        return out, output_embeddings, output_logits

    # def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, force_generation=None, attention_mask=None, position_ids=None, **kwargs):
    #     # Call the base model's `prepare_inputs_for_generation` and extract necessary model_kwargs
    #     model_kwargs = self.base_model_prepare_inputs_for_generation(
    #         input_ids,
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         **kwargs
    #     )
        
    #     bs = inputs_embeds.shape[0]
    #     device = inputs_embeds.device
        
    #     if 'input_ids' in model_kwargs and model_kwargs['input_ids'] is not None:
    #         new_token_ids = model_kwargs['input_ids'][:, -1:]
    #         if new_token_ids == self.output_img_id:
    #             # Generated the image token, force add all the image tokens
    #             current_position_ids = model_kwargs['position_ids'][0, -1]  # TODO: Only support batch size 1
    #             all_img_tokens = torch.tensor(
    #                 self.llama_tokenizer.convert_tokens_to_ids(self.ALL_IMG_TOKENS)
    #             ).unsqueeze(0).to(device)
                
    #             all_img_tokens_mask = torch.ones_like(all_img_tokens)[:, :self.IMG_TOKEN_NUM - 1].to(device)
    #             all_img_position_ids = torch.arange(
    #                 current_position_ids, current_position_ids + self.IMG_TOKEN_NUM
    #             ).unsqueeze(0).to(device)
                
    #             # Update `attention_mask`
    #             if model_kwargs.get('attention_mask') is not None:
    #                 model_kwargs['attention_mask'] = torch.cat([model_kwargs['attention_mask'], all_img_tokens_mask], dim=1)
    #             else:
    #                 model_kwargs['attention_mask'] = all_img_tokens_mask
                
    #             # Update `position_ids`
    #             model_kwargs['position_ids'] = all_img_position_ids
                
    #             # Update `inputs_embeds`
    #             inputs_embeds = self.get_input_embeddings(all_img_tokens)
    #             model_kwargs['input_ids'] = None
    #             model_kwargs['inputs_embeds'] = inputs_embeds
    
    # return model_kwargs

    # def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, force_generation=None, *args, **kwargs):
    #     model_kwargs = self.base_model_prepare_inputs_for_generation(input_ids, inputs_embeds=inputs_embeds, *args, **kwargs)
    #     bs = inputs_embeds.shape[0]
    #     device = inputs_embeds.device
    #     if 'input_ids' in model_kwargs:
    #         new_token_ids = model_kwargs['input_ids'][:, -1:]
    #         if new_token_ids == self.output_img_id:
    #             #Generated the image token, force add all the image tokens
    #             current_position_ids = model_kwargs['position_ids'][0, -1] #TODO: Only support batch size 1
    #             all_img_tokens = torch.tensor(self.llama_tokenizer.convert_tokens_to_ids(self.ALL_IMG_TOKENS)).unsqueeze(0).to(device)
    #             all_img_tokens_mask = torch.ones_like(all_img_tokens)[:, :self.IMG_TOKEN_NUM-1].to(device)
    #             all_img_position_ids = torch.arange(current_position_ids, current_position_ids + self.IMG_TOKEN_NUM).unsqueeze(0).to(device)
                
    #             model_kwargs['attention_mask'] = torch.cat([model_kwargs['attention_mask'], all_img_tokens_mask], dim=1)
    #             model_kwargs['position_ids'] = all_img_position_ids
    #             inputs_embeds = self.get_input_embeddings(all_img_tokens)
    #             model_kwargs['input_ids'] = None
    #             model_kwargs['inputs_embeds'] = inputs_embeds
    def load_mtr(self):
        return
    # @classmethod
    def from_config(self, cfg):
    # def from_config(self, cfg):
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
        
        # model = self.from_pretrained(
        #     llama_model,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        #     # quantization_config=bnb_config,
        #     use_cache=False,
        #     load_in_8bit=True,
        #     attn_implementation="flash_attention_2", # use_flash_attention_2=True
        # )
        self.__init_additional_modules__(
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
            instruct_act = cfg.get("instruct_act", True),
            caption = cfg.get("caption", False),
            smpl_instruct = cfg.get("smpl_instruct", False),
            very_smpl_turn = cfg.get("very_smpl_turn", False),
            gf_act_enc=cfg.get("gf_act_enc", False),
            gf_act_dec=cfg.get("gf_act_dec", False),
            full_map=cfg.get("full_map", False),
            eval_only=cfg.get("eval_only", False),
            gf_level=cfg.get("gf_level", 3),
            two_agent=cfg.get("two_agent", False),
            gmm_head_only=cfg.get("gmm_head_only", False),
            freeze_gf_dec = cfg.get("freeze_gf_dec", False),
            use_llama2_template = cfg.get("use_llama2_template", False),
            neighbors_to_predict = cfg.get("neighbors_to_predict", 1),
            act_kv = cfg.get("act_kv", False),
            late_fusion = cfg.get("late_fusion", False),
            use_flash_attention = cfg.get("use_flash_attention", False),
            upcast_to_bfloat16 = cfg.get("upcast_to_bfloat16", False),
            resume_ckpt_path = cfg.get("resume_ckpt_path", ""),
            always_calculate_gt_traj = cfg.get("always_calculate_gt_traj", ""),
            use_float32_embd = cfg.get("use_float32_embd", False),
            add_decision_cls = cfg.get("add_decision_cls", False),
            use_embed_mask = cfg.get("use_embed_mask", True),
            num_direction_classes = cfg.get("num_direction_classes", 8),
            no_sys_prompt = cfg.get("no_sys_prompt", False),
            auto_txt_formatter = cfg.get("auto_txt_formatter", False)
        )
        # model = cls(
        #     vit_model=vit_model,
        #     q_former_model=q_former_model,
        #     img_size=img_size,
        #     drop_path_rate=drop_path_rate,
        #     use_grad_checkpoint=use_grad_checkpoint,
        #     vit_precision=vit_precision,
        #     freeze_vit=freeze_vit,
        #     freeze_qformer=freeze_qformer,
        #     num_query_token=num_query_token,
        #     llama_model=llama_model,
        #     prompt_path=prompt_path,
        #     prompt_template=prompt_template,
        #     max_txt_len=max_txt_len,
        #     end_sym=end_sym,
        #     low_resource=low_resource,
        #     device_8bit=device_8bit,
        #     grid_size=grid_size,
        #     num_agents=num_agents,
        #     data_info_path=data_info_path,
        #     freeze_embed_tokens=freeze_embed_tokens,
        #     lora=lora,
        #     lora_r=lora_r,
        #     lora_alpha=lora_alpha,
        #     lora_dropout=lora_dropout,
        #     lora_target_modules = lora_target_modules,
        #     discretizer_model_path = cfg.get("discretizer", './KBinsDiscretizer_76.pkl'),
        #     quant = cfg.get("quant", 4),
        #     lora_bias= cfg.get("lora_bias", False),
        #     ignore_motion = cfg.get("ignore_motion", False),
        #     rollouts = cfg.get("rollouts", 6),
        #     gameformer_enc = cfg.get("gameformer_enc", False),
        #     gf_encoder_path = cfg.get("gf_encoder_path", None),
        #     gf_encoder_layers = cfg.get("gf_encoder_layers", None),
        #     freeze_gf = cfg.get("freeze_gf", True),
        #     in_adapter_per_feature = cfg.get("in_adapter_per_feature", True),
        #     out_adapter_per_embed = cfg.get("out_adapter_per_embed", True),
        #     in_adapter_mlp=cfg.get("in_adapter_mlp", False),
        #     out_adapter_mlp=cfg.get("out_adapter_mlp", True),
        #     act=cfg.get("act", False),
        #     recon_loss_adapter = cfg.get("recon_loss_adapter", False),
        #     recon_loss_llm = cfg.get("recon_loss_llm", False),
        #     freeze_gf_enc = cfg.get("freeze_gf_enc", True),
        #     motion_act = cfg.get("motion_act", False),
        #     img_token_num = cfg.get("img_token_num", 204),
        #     gf_token_num =  cfg.get("gf_token_num", 204),
        #     output_crossattn = cfg.get("output_crossattn", False),
        #     regress_act = cfg.get("regress_act", False),
        #     instruct_act = cfg.get("instruct_act", True),
        #     caption = cfg.get("caption", False),
        #     smpl_instruct = cfg.get("smpl_instruct", False),
        #     very_smpl_turn = cfg.get("very_smpl_turn", False),
        #     gf_act_enc=cfg.get("gf_act_enc", False),
        #     gf_act_dec=cfg.get("gf_act_dec", False),
        #     full_map=cfg.get("full_map", False),
        #     eval_only=cfg.get("eval_only", False),
        #     gf_level=cfg.get("gf_level", 3),
        #     two_agent=cfg.get("two_agent", False),
        #     gmm_head_only=cfg.get("gmm_head_only", False),
        #     freeze_gf_dec = cfg.get("freeze_gf_dec", False),
        #     use_llama2_template = cfg.get("use_llama2_template", False),
        #     neighbors_to_predict = cfg.get("neighbors_to_predict", 1),
        #     act_kv = cfg.get("act_kv", False),
        #     late_fusion = cfg.get("late_fusion", False),
        #     use_flash_attention = cfg.get("use_flash_attention", False),
        #     upcast_to_bfloat16 = cfg.get("upcast_to_bfloat16", False),
        #     resume_ckpt_path = cfg.get("resume_ckpt_path", ""),
        #     always_calculate_gt_traj = cfg.get("always_calculate_gt_traj", ""),
        #     use_float32_embd = cfg.get("use_float32_embd", False),
        # )

        # ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        # if ckpt_path:
        #     print("Load Checkpoint: {}".format(ckpt_path))
        #     ckpt = torch.load(ckpt_path, map_location="cpu")
        #     msg = model.load_state_dict(ckpt['model'], strict=False)

        # return model
        
    def reset_embeddings(self):
        with torch.no_grad():
        # if True:
            if self.lora:
                for n, p in self.llama_model.named_parameters():
                    if p.grad is None:
                    #     print(n)
                        continue
                    if "lm_head" in n:
                        # break
                        # print(n)
                        # print(p.grad)
                        # try:
                        p.grad = p.grad*self.output_embed_grad_mask
                        # except:
                        #     p.grad = p.grad*self.output_embed_grad_mask.to(self.TRAINABLE_PRECISION_LLM)
                        assert sum(p.grad[-1]!=0)>0
                    elif "embed_tokens" in n:
                        # print(n)
                        # print(p.grad)
                        # try:
                        p.grad = p.grad*self.input_embed_grad_mask
                        # except:
                        # p.grad = p.grad*self.input_embed_grad_mask.to(self.TRAINABLE_PRECISION_LLM)
                        assert sum(p.grad[-1]!=0)>0
            else:
                self.llama_model.get_input_embeddings().weight.grad = self.llama_model.get_input_embeddings().weight.grad*self.input_embed_grad_mask
                self.llama_model.get_output_embeddings().weight.grad = self.llama_model.get_output_embeddings().weight.grad*self.output_embed_grad_mask
    
    # def on_before_optimizer_step(self, optimizer) -> None:
    #     self.reset_embeddings()
    
    def on_before_optimizer_step(self) -> None:
        self.reset_embeddings()


    @torch.no_grad()
    def llm_generate(self, input_embeds, input_atts, num_words, stopping_criteria):#, do_sample = True, temperature = 1.0, top_k = 50, max_length = 2048, top_p = 0.95):
        self.eval()
        output = self.llm.generate(
                            inputs_embeds=input_embeds,
                            attention_mask=input_atts,
                            max_new_tokens=num_words,
                            stopping_criteria=stopping_criteria, #</motion> and </s>
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                        )
        output_ids = output.sequences
        hidden_states = output.hidden_states # tuple(tuple) -> ((layer1, layer2, ..., layern), ..., (layer1, layer2, ..., layern)) -> sequential_dimension((layer dimension))
        return output_ids, hidden_states


    def gradient_checkpointing_enable(self):
        self.model.llama_model.gradient_checkpointing_enable()

    def merge_and_unload(self):
        # merge and unload the llm peft module
        self.llm = self.llm.merge_and_unload()

    def old_generate(self, input_embeds, input_atts, num_words=100):
        batch_wise_processing = True
        output_sequences, output_hidden_states = self.llm_generate(input_embeds, input_atts, num_words, stopping_criteria=None)

        
        output_embeddings = []
        text_out = []
        valid_examples = []
        input_embeddings = self.llm.model.model.embed_tokens if _is_peft_model(self.llm) else self.llm.model.embed_tokens
        for i in range(output_sequences.shape[0]):
            k = len(self.gen_token_idx)
            # Create a sliding window using unfold
            if len(output_sequences[i])>k:
                unfolded = output_sequences[i].unfold(dimension=0, size=min(k, len(output_sequences[i])), step=1)
                # Check if any window matches the target sequence
                matches = (unfolded == torch.tensor(self.gen_token_idx, device=unfolded.device)).all(dim=1)    
            else:
                matches = [False]
            first_token_idx = -1
            if True in matches:
                # the model generated all IMG tokens
                # Find indices where the match occurs
                indices = matches.nonzero(as_tuple=False).squeeze()
                if len(indices.shape)>=1:
                    indices = indices[0]
                output_embeddings.append(torch.cat([token_i_hidden_states[-1][i] for token_i_hidden_states in output_hidden_states[indices:indices+len(self.gen_token_idx)]], 0))
                first_token_idx = indices
            else:
                # find how many tokens were generated
                found_tokens = 0
                for j in range(len(output_sequences[i])):
                    if output_sequences[i][j] == self.gen_token_idx[0]:
                        last_token_found=output_sequences[i][j]
                        found_tokens = 1
                        first_token_idx = j
                        for k in range(j+1, len(output_sequences[i])):
                            if output_sequences[i][k]-last_token_found == 1: # the next token in dictionary
                                last_token_found = output_sequences[i][k]
                                found_tokens+=1
                            else:
                                break
                # Force generation of the remaining [IMG] tokens, assuming some were already generated (based on the found_tokens above)
                if found_tokens==0:
                    # No generated tokens found
                    already_generated_embeddings = input_embeddings(output_sequences[i]) # input embedding of all generated data up to the last found generation token
                else:
                    already_generated_embeddings = input_embeddings(output_sequences[i][:first_token_idx+found_tokens]) # input embedding of all generated data up to the last found generation token
                all_previous_embeddings = torch.cat((input_embeds[i], already_generated_embeddings),0)
                tokens_to_force_generating = self.gen_token_idx[found_tokens:]
                generated_ids_, generated_embeddings_, _ = self.continue_generate(
                        embeddings=all_previous_embeddings.unsqueeze(0),
                        tokens_to_force_generating=torch.tensor(tokens_to_force_generating, device=already_generated_embeddings.device),
                )
                if found_tokens==0:
                    output_embeddings.append(generated_embeddings_[-1][0,all_previous_embeddings.shape[0]:])
                else:
                    output_embeddings_pt1 = torch.cat([token_i_hidden_states[-1][i] for token_i_hidden_states in output_hidden_states[first_token_idx:first_token_idx+found_tokens]], 0)
                    output_embeddings_pt2 = generated_embeddings_[-1][0,all_previous_embeddings.shape[0]:]
                    # output_embeddings_pt2 = generated_embeddings_[-1][0, input_embeds[i].shape[0]+first_token_idx+found_tokens:]
                    output_embeddings.append(torch.cat((output_embeddings_pt1, output_embeddings_pt2), 0))
                # if output_embeddings[1].shape[0]>357:
                #     print('')
            text_out.append(self.llama_tokenizer.decode(output_sequences[i][:first_token_idx]))
        output_embeddings = torch.stack(output_embeddings)
        return text_out, output_embeddings