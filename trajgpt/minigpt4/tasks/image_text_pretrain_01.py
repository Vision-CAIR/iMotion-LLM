"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.common.registry import registry
from minigpt4.tasks.base_task import BaseTask

from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.common.logger import MetricLogger, SmoothedValue
import logging
import torch
import random
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList
from matplotlib import pyplot as plt
from tqdm import tqdm
from minigpt4.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def find_sublist(self, large_tensor, sublist_tensor):
        # Convert the lists to PyTorch tensors
        # large_tensor = torch.tensor(large_list)
        # sublist_tensor = torch.tensor(sublist)
        
        # Calculate the length of both lists
        len_large = len(large_tensor)
        len_sub = len(sublist_tensor)
        
        # Iterate over the large tensor with a sliding window of the sublist's length
        for i in range(len_large - len_sub + 1):
            # Extract the window from the large tensor
            window = large_tensor[i:i+len_sub]
            
            # Compare the window to the sublist tensor
            if torch.equal(window, sublist_tensor):
                return i  # Return the starting index of the first occurrence
        return -1  # Return -1 if the sublist is not found

    @torch.no_grad()
    def evaluation(self, model, data_loader, epoch, cuda_enabled=True, num_eval_batches=None):
        # model._training=False
        # model.update_model_require_grad()
        # model.llama_model.eval()
        # if model.lora:
        #     model.lora_llama_model.eval()
        num_figs = 1
        if num_eval_batches and num_eval_batches!=-1:
            num_eval_examples=num_eval_batches
        else:
            num_eval_examples=len(data_loader)
        
        num_eval_examples=int(num_eval_examples/get_world_size())
        figs = []
        #TODO: fix next line, do this in the model class
        # data_info_path = '<internal_user_root>/gpt_datasets/minigpt4_datasets/traj/traj_29june/data_info.pt'
        # data_info = torch.load(data_info_path)
        # rel_cont_grid = data_info['rel_cont_grid'].to(model.device)
        #TODO: fix next line, do this in the model class
        use_amp = True

        # if model.llama2:
        #     print('')
        # elif model.prompt_list:
        #     prompt = random.choice(model.prompt_list)

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        # if iter-based runner, schedule lr based on inner epoch.
        # logging.info(
        #     "Start validation"
        # )
        # header = "Valid"
        # log_freq=1
        evaluation_samples_count=0
        _rel_err=0.0                
        _rel_err_c=0.0
        _traj_err=0.0
        _traj_err_c=0.0
        _final_traj_err=0.0
        _final_traj_err_c=0.0
        _len_err=0.0
        _len_err_c=0.0
        _tokens_abs_err=0.0
        _tokens_abs_err_c=0.0
        # with torch.no_grad():
        motion_start_token = model.llama_tokenizer(model.motion_start_symbol, add_special_tokens=False, return_tensors="pt").input_ids[0]
        motion_end_token = model.llama_tokenizer(model.motion_end_symbol, add_special_tokens=False, return_tensors="pt").input_ids[0]
        # if num_eval_examples==-1 or num_eval_examples > len(data_loader.loaders[0]):
        #         num_eval_examples=len(data_loader.loaders[0])

        # with torch.no_grad():
        # for i in metric_logger.log_every(range(len(data_loader.loaders[0])), log_freq, header):
        # for i in metric_logger.log_every(range(num_eval_examples), log_freq, header):
        # with torch.no_grad():
        header = "Val: data epoch: [{}]".format(epoch)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("rel_err", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("traj_err", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("final_traj_err", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        for i in metric_logger.log_every(range(num_eval_examples), 50, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= num_eval_examples:
                break
        # for i in tqdm(range(num_eval_examples)):
            # if i>=111:
            #     print('')
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
############## Evaluation
            target_agent = samples['target_agent']
            with torch.cuda.amp.autocast(enabled=use_amp):
                # outputs, target_agent = self.valid_step(model, samples)
                outputs = self.valid_step(model, samples)
            
            # output_token = 
            

            for j in range(len(samples['sample_id'])):
                didnt_work_counter = 0
                output_motion=[]
                output_token = outputs[j]
                while len(output_motion)<2:
                    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                        output_token = output_token[1:]
                    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                        output_token = output_token[1:]
                    if output_token[-1] == 2:
                        output_token = output_token[:-1]
                        # <motion>: 529, 29885, 8194, 29958
                        # </motion>: 1533, 29885, 8194, 29958
                    if len(motion_start_token)==1:
                        if motion_start_token in output_token and motion_end_token in output_token:
                            start_token_i = max([token_i for token_i in range(len(output_token)) if output_token[token_i] == motion_start_token])+1
                            end_token_i = min([token_i for token_i in range(len(output_token)) if output_token[token_i] == motion_end_token])
                            if end_token_i>start_token_i:
                                output_token = output_token[start_token_i:end_token_i]
                        else:
                            didnt_work_counter+=1
                            if didnt_work_counter>=5:
                                break
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                output_token = self.valid_step(model, model.get_sample(samples, j))
                                output_token = output_token[0]
                    else:
                        start_token_i = self.find_sublist(output_token, motion_start_token.to(output_token.device))+len(motion_start_token)
                        end_token_i = self.find_sublist(output_token, motion_end_token.to(output_token.device))
                        if start_token_i != -1 and end_token_i != -1 and end_token_i>start_token_i:
                            output_token = output_token[start_token_i: end_token_i]
                        elif start_token_i != -1:
                            output_token = output_token[start_token_i:]
                        else:
                            output_token = output_token

                    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
                    # output_text = output_text.split('###')[0]  # remove the stop sign '###'
                    # output_text = output_text.split('</s>')[0]  # remove the stop sign '###'
                    # output_text = output_text.split('Assistant:')[-1].strip()
                    # output_text = output_text.split(model.motion_start_symbol)[1].split(model.motion_end_symbol)[0].strip()
                    # output_text = output_text.replace(model.motion_start_symbol , '').replace(model.motion_end_symbol, '').strip(' ')
                    output_text = output_text.split(' ')
                    if model.no_new_vocab:
                        output_motion = [motion_i for motion_i in output_text if motion_i.isdigit()]
                    else:
                        output_motion = [motion_i for motion_i in output_text if '<s-' in motion_i and '>' in motion_i]
                    if len(output_motion)<2:
                        didnt_work_counter+=1
                        if didnt_work_counter>=5:
                            break
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            output_token = self.valid_step(model, model.get_sample(samples, j))
                            output_token = output_token[0]
                    # print(output_motion)
                if didnt_work_counter>=5:
                    break
                
                output_motion_idx = [int(motion_i.strip('<s->'+model.motion_start_symbol+model.motion_end_symbol)) for motion_i in output_motion if motion_i.strip('<s->').isdigit()]
                output_motion_idx = output_motion_idx[:len(output_motion_idx) - len(output_motion_idx)%2]
                output_rel = model.rel_cont_grid[output_motion_idx].reshape(-1,2).transpose(1,0).float()

                pred_rel_gt = model.rel_cont_grid[samples['rel_pred'][j, target_agent[j], :, :].cpu()].float()
                obs_rel_gt = model.rel_cont_grid[samples['rel_obs'][j, target_agent[j], :, :].cpu()].float() # first elements should always represent zero

                zero_motion = obs_rel_gt[:,0] # [-0.0002, -0.0002], supposed to be [0,0] # this is not being used in other than the assertion and getting the zero idx
                assert True in (model.rel_cont_grid == 0.0)
                assert True in (model.rel_cont_grid == zero_motion[0])
                zero_idx = (model.rel_cont_grid == zero_motion[0]).nonzero().item()
                zero_token_txt = '<s-'+str(zero_idx)+'>'
                init_traj = samples['init_traj'][j][target_agent[j]].cpu()

                obs_traj_gt = model.get_traj_from_rel(obs_rel_gt.unsqueeze(0).unsqueeze(0), init_traj.unsqueeze(-1))
                pred_traj_gt = model.get_traj_from_rel(pred_rel_gt.unsqueeze(0).unsqueeze(0), obs_traj_gt[:,:,:,-1].squeeze(0).squeeze(0).unsqueeze(-1))
                output_traj = model.get_traj_from_rel(output_rel.unsqueeze(0).unsqueeze(0), obs_traj_gt[:,:,:,-1].squeeze(0).squeeze(0).unsqueeze(-1))

                # output_motion_tokens = torch.tensor(model.llama_tokenizer(output_motion, add_special_tokens=False).input_ids).float()
                gt_motion = model.parse_coordinates(samples['rel_pred'][j:j+1, target_agent[j], :, :])[0].replace(model.motion_start_symbol, '').replace(model.motion_end_symbol, '').strip().split(" ")
                # gt_motion_tokens = torch.tensor(model.llama_tokenizer(gt_motion, add_special_tokens=False).input_ids)

                output_len = output_rel.shape[-1]
                # output_motion_tokens_len = output_motion_tokens.shape[0]
                # gt_motion_tokens_len = gt_motion_tokens.shape[0]
                gt_pred_len = pred_rel_gt.shape[-1]
                #DONE: Study if the following is correct, including the main if statement
                # if output_motion_tokens_len >= gt_motion_tokens_len:
                if output_len >= gt_pred_len:
                    # If we predict more motion, its fine just use what you need and ignore the rest
                    # tokens_abs_err = abs(output_motion_tokens[:gt_motion_tokens_len] - gt_motion_tokens).float().mean()
                    rel_err = (((output_rel[:, :gt_pred_len] - pred_rel_gt)**2).sum(dim=0)).sqrt().mean()
                    traj_err_list = (((output_traj[0, 0, :, :gt_pred_len] - pred_traj_gt[0, 0, :, :])**2).sum(dim=0)).sqrt()
                    traj_err = traj_err_list.mean()
                    final_traj_err = traj_err_list[-1]
                else: #If we fall short on number of predictions, the output is padded with the zero motion token
                    #DONE above: Change based on the true zero
                    # zero_token_txt = '<s-279>'# represents -0.0002
                    zero_token = model.llama_tokenizer(zero_token_txt, add_special_tokens=False).input_ids[0]
                    # tokens_abs_err = torch.cat( (abs(output_motion_tokens - gt_motion_tokens[:output_motion_tokens_len]), abs(zero_token - gt_motion_tokens[output_motion_tokens_len:]))).mean()
                    # rel_err = (((output_rel - pred_rel_gt[:, :output_len])**2).sum(dim=0)).sqrt().mean() #DONE: Fix to include zero motion for missing values, through padding with zero idx token
                    rel_err = torch.cat((((output_rel - pred_rel_gt[:, :output_len])**2).sum(dim=0),
                                        ((zero_motion.unsqueeze(-1) - pred_rel_gt[:, output_len:])**2).sum(dim=0))).sqrt().mean()
                    # traj_err = (((output_traj[0, 0, :, :] - pred_traj_gt[0, 0, :, :output_len])**2).sum(dim=0)).sqrt().mean() #DONE: Fix to include zero motion for missing values, through repeating the last predicted position
                    traj_err_list = torch.cat((((output_traj[0, 0, :, :] - pred_traj_gt[0, 0, :, :output_len])**2).sum(dim=0),
                                            ((output_traj[0, 0, :, -1:] - pred_traj_gt[0, 0, :, output_len:])**2).sum(dim=0))).sqrt()
                    traj_err = traj_err_list.mean()
                    final_traj_err = traj_err_list[-1]
                # len_err = abs(len(output_motion_tokens) - len(gt_motion_tokens))

                metric_logger.update(rel_err=rel_err.item())
                metric_logger.update(traj_err=traj_err.item())
                metric_logger.update(final_traj_err = final_traj_err.item())
                metric_logger.update(evaluation_samples_count = 1)
                evaluation_samples_count+=1    

                # _rel_err+=rel_err.item()
                # _rel_err_c +=1
                # _traj_err+=traj_err.item()
                # _traj_err_c+=1
                # _final_traj_err += final_traj_err.item()
                # _final_traj_err_c +=1
                # _len_err+=float(len_err)
                # _len_err_c +=1
                # _tokens_abs_err += tokens_abs_err.item()
                # _tokens_abs_err_c +=1

                # if evaluation_samples_count < num_figs:
                #     _ax, _fig = model.plt_grid_and_original_02(grid=obs_traj_gt.cpu(), grid_shape='.',ax1=None,fig1=None, colors_=1, color_alpha=1)
                #     _ax, _fig = model.plt_grid_and_original_02(grid=pred_traj_gt.cpu(), grid_shape='s',ax1=_ax,fig1=_fig, colors_=1, color_alpha=0.5)
                #     _, _fig = model.plt_grid_and_original_02(grid=output_traj.cpu(), grid_shape='o',ax1=_ax,fig1=_fig, colors_=2, color_alpha=0.8)
                #     figs.append(model.fig2img(_fig))
                
                   
            
            # if _rel_err/_rel_err_c < 0.1:
            #     print("YO")
                
        # if _rel_err_c!=0 and _rel_err and _traj_err and _len_err:
        # if _rel_err_c!=0 and _rel_err and _traj_err:
        #     rel_err=_rel_err/_rel_err_c
        #     traj_err=_traj_err/_traj_err_c
        #     final_traj_err=_final_traj_err/_final_traj_err_c
            
            # len_err=_len_err/_len_err_c
            # tokens_abs_err=_tokens_abs_err/_tokens_abs_err_c
            # tokens_abs_err=0

        # print('num evaluation samples='+str(evaluation_samples_count))
            # print("rel err:")
            # print(rel_err)
            # print("####")
            # print("traj err:")
            # print(traj_err)
            # print("####")
            # print("final traj err:")
            # print(final_traj_err)
            # print("####")

            # print("len err:")
            # print(len_err)
            # print("####")
            # print("token dist err:")
            # print(tokens_abs_err)
        # print("#######################")
        # ax1,fig1 = model.plt_grid_and_original_02(grid=pred_traj_gt.cpu(), grid_shape='o',ax1=None,fig1=None, colors_=1, color_alpha=0.5)
        # ax1,fig1 = model.plt_grid_and_original_02(grid=output_traj.cpu(), grid_shape='s',ax1=ax1,fig1=fig1, colors_=2)
        # model._training=True
        # # model.update_model_require_grad()
        # # model.print_trainable_parameters(model.llama_model)
        # model.llama_model.train()
        # if model.lora:
        #     model.lora_llama_model.train()
        figs=0

        metric_logger.synchronize_between_processes()
        val_log = {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items() if k!='evaluation_samples_count'
        }
        val_log['evaluation_samples_count'] = str(metric_logger.meters['evaluation_samples_count'].count)
        return val_log
        # return evaluation_samples_count, rel_err, traj_err, final_traj_err
    
    
        # fig1.save('<internal_user_root>/tst.png')
        # print("Validation example:")
        #         # print("")
        # print(output_text)

#         # after train_epoch()
#         # gather the stats from all processes
#         metric_logger.synchronize_between_processes()
#         logging.info("Averaged stats: " + str(metric_logger.global_avg()))
#         # return {
#         #     k: "{:.3f}".format(meter.global_avg)
#         #     for k, meter in metric_logger.meters.items()
#         # }
#         model._training=True
#         print("DAH")
#         if not hasattr(data_loader, "__next__"):
#             # convert to iterator if not already
#             data_loader = iter(data_loader)
#         data_info = torch.load('<internal_user_root>/gpt_datasets/minigpt4_datasets/traj/traj_29june/data_info.pt')
#         rel_cont_grid = data_info['rel_cont_grid'].to(model.device)
        
#         _rel_err=0.0
#         _rel_err_c=0.0
#         _traj_err=0.0
#         _traj_err_c=0.0
#         _len_err=0.0
#         _len_err_c=0.0
#         _tokens_abs_err=0.0
#         _tokens_abs_err_c=0.0
        
#         with torch.no_grad():
#             for i in tqdm(range(len(data_loader.loaders[0]))):  
#                 samples = next(data_loader)
#                 samples = prepare_sample(samples)
#                 samples.update(
#                     {
#                         "epoch": epoch,
#                         "iters": i,
#                     }
#                 )

#                 if model.prompt_list:
#                     prompt = random.choice(model.prompt_list)
                
#                 obs = samples['rel_obs']
#                 pred = samples['rel_pred']
#                 # obs_traj = samples['traj_obs']
#                 # pred_traj = samples['Traj_pred']

#                 obs_len = obs.shape[-1]
#                 pred_len = pred.shape[-1]
#                 target_agent = np.random.binomial(1,0.5)
#                 min_grid = 0
#                 max_grid = 512
#                 parsed_data = model.parse_traj_data(obs, pred, target_agent, min_grid, max_grid, prompt)
                
#                 batch_size = obs.shape[0]

#                 # With torch.cuda.amp.autocast(enabled=use_amp): # This is used in training, do we need it here?
#                 input_embeds, input_atts, input_tokens = model.input_prompt_wrap(parsed_data, prompt, device = samples['sample_id'].device)
                
#                 bos = torch.ones([batch_size, 1],
#                         dtype=input_tokens.dtype,
#                         device=input_tokens.device) * model.llama_tokenizer.bos_token_id
#                 bos_embeds = model.llama_model.model.embed_tokens(bos)

#                 # tokens = torch.cat([bos, input_tokens], dim=1)
#                 embs = torch.cat([bos_embeds, input_embeds], dim=1)

#                 max_new_tokens = model.max_txt_len
#                 stop_words_ids = [torch.tensor([835]).to(input_tokens.device),
#                         torch.tensor([2277, 29937]).to(input_tokens.device)]  # '###' can be encoded in two different ways.
#                 stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
#                 num_beams = 1
#                 min_length = 1
#                 top_p = 0.9
#                 repetition_penalty = 1.0
#                 length_penalty = 1
#                 temperature = 1.0
                
                
#                 with model.maybe_autocast():
#                     outputs = model.llama_model.generate(
#                         inputs_embeds=embs,
#                         max_new_tokens=max_new_tokens,
#                         stopping_criteria=stopping_criteria,
#                         num_beams=num_beams,
#                         do_sample=True,
#                         min_length=min_length,
#                         top_p=top_p,
#                         repetition_penalty=repetition_penalty,
#                         length_penalty=length_penalty,
#                         temperature=temperature,
#                     )
#                 # Output sample (1 sample of the batch)
#                 # TODO: include batch evaluation metric
#                 # TODO: Change max_new_tokens based on expected length based on pred_len
#                 # TODO: Hyperparameter search of above hardcoded parameters (To optemize evaluation metric on 1 good finetuned (freezed) model)
#                 # obs_traj_target_agent = obs_traj[:,target_agent]
#                 # pred_traj_target_agent = pred_traj[:,target_agent]
#                 for i in range(outputs.shape[0]):
#                     output_token = outputs[i]
#                     if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
#                         output_token = output_token[1:]
#                     if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
#                         output_token = output_token[1:]
#                     output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
#                     output_text = output_text.split('###')[0]  # remove the stop sign '###'
#                     output_text = output_text.split('Assistant:')[-1].strip()
#                     output_text = output_text.split(' ')
#                     output_motion = [motion_i for motion_i in output_text if '<s-' in motion_i and '>' in motion_i]

#                     output_motion_idx = [int(motion_i.strip('<s->/img')) for motion_i in output_motion if motion_i.strip('<s->').isdigit()]
#                     output_motion_idx = output_motion_idx[:len(output_motion_idx) - len(output_motion_idx)%2]
#                     output_rel = rel_cont_grid[output_motion_idx].reshape(-1,2).transpose(1,0)

#                     pred_rel_gt = rel_cont_grid[samples['rel_pred'][i, target_agent, :, :]]
#                     obs_rel_gt = rel_cont_grid[samples['rel_obs'][i, target_agent, :, :]] # first elements should always represent zero

#                     zero_idx = obs_rel_gt[:,0] # [-0.0002, -0.0002], supposed to be [0,0] #
                    
#                     obs_traj_gt = model.get_traj_from_rel(obs_rel_gt.unsqueeze(0).unsqueeze(0), zero_idx.unsqueeze(-1)) # We are supposed to feed the true init traj, not zeros as the initial traj, we will assume zero for now
#                     pred_traj_gt = model.get_traj_from_rel(pred_rel_gt.unsqueeze(0).unsqueeze(0), obs_traj_gt[:,:,:,-1].squeeze(0).squeeze(0).unsqueeze(-1))
#                     output_traj = model.get_traj_from_rel(output_rel.unsqueeze(0).unsqueeze(0), obs_traj_gt[:,:,:,-1].squeeze(0).squeeze(0).unsqueeze(-1))

#                     output_len = output_rel.shape[-1]

#                     output_motion_tokens = torch.tensor(model.llama_tokenizer(output_motion, add_special_tokens=False).input_ids)
#                     gt_motion_tokens = torch.tensor(model.llama_tokenizer(model.parse_coordinates(samples['rel_pred'][i:i+1, target_agent, :, :])[0].split(" "), add_special_tokens=False).input_ids)
#                     if output_motion_tokens.shape[0] >= gt_motion_tokens.shape[0]:
#                         tokens_abs_err = abs(output_motion_tokens[:gt_motion_tokens.shape[0]] - gt_motion_tokens).float().mean()
#                         rel_err = (((output_rel[:, :pred_rel_gt.shape[-1]] - pred_rel_gt)**2).sum(dim=0)).sqrt().mean()
#                         traj_err = (((output_traj[0, 0, :, :pred_traj_gt.shape[-1]] - pred_traj_gt)**2).sum(dim=0)).sqrt().mean()
#                     else: #short output is padded with the zero motion token
#                         zero_token_txt = '<s-279>'# represents -0.0002
#                         zero_token = model.llama_tokenizer(zero_token_txt, add_special_tokens=False).input_ids[0]
#                         tokens_abs_err = torch.cat( (abs(output_motion_tokens - gt_motion_tokens[:output_motion_tokens.shape[0]]).float(), abs(zero_token - gt_motion_tokens[output_motion_tokens.shape[0]:]).float())).mean()
#                         rel_err = (((output_rel - pred_rel_gt[:, :output_rel.shape[-1]])**2).sum(dim=0)).sqrt().mean()
#                         traj_err = (((output_traj[0, 0, :, :] - pred_traj_gt[0, 0, :, :output_traj.shape[-1]])**2).sum(dim=0)).sqrt().mean()

#                     len_err = abs(len(output_motion_tokens) - len(gt_motion_tokens))




#                     # ax1,fig1 = model.plt_grid_and_original_02(grid=pred_traj_gt.cpu(), grid_shape='o',ax1=None,fig1=None, colors_=1, color_alpha=0.5)
#                     # ax1,fig1 = model.plt_grid_and_original_02(grid=output_traj.cpu(), grid_shape='s',ax1=ax1,fig1=fig1, colors_=2)
#                     # fig1.save('<internal_user_root>/tst.png')
#                     # ''
#                     # Figure ready to be saved, now lets calculate the metrics

#                     # traj_init = obs_traj_target_agent[i, :, -1]
#                     # data_info['traj_cont_grid']
#                     # rel_2_traj =
# # The resolution on which rel and traj where descretized were different, thus we need to be careful in using them, 
# # I will only use the descretized rel in this experiment (that will be converted back to continous and then to trajectories. this will be done for booth predections and ground truth)
                    

#                     _rel_err+=rel_err
#                     _rel_err_c +=1
#                     _traj_err+=traj_err
#                     _traj_err_c+=1
#                     _len_err+=len_err
#                     _len_err_c +=1
#                     _tokens_abs_err += tokens_abs_err
#                     _tokens_abs_err_c +=1
#                     # break
                
                
#                 # print("Validation example:")
#                 # print("")
#                 # print(output_text)
#                     # output_tokens = model.llama_model.generate(torch.cat([bos, input_tokens], dim=1), max_length=model.max_txt_len)
#                     # output_data = model.llama_tokenizer.batch_decode(output_tokens)
#                     # # model_out = model(samples)
#                     # # loss = model_out["loss"].item()
#                     # # logits = model_out["logits"].it
#                     # print("Validation example:")
#                     # print(output_data[0])
#                     # return output_text
#                 # return None

#         print("#######################")
#         valid_rel_err = _rel_err/_rel_err_c
#         valid_traj_err = _traj_err/_traj_err_c
#         valid_len_err = _len_err/_len_err_c
#         valid_tokens_abs_err = _tokens_abs_err/_tokens_abs_err_c

        # print("rel err:")
        # print(valid_rel_err)
        # print("####")
        # print("traj err:")
        # print(valid_traj_err)
        # print("####")
        # print(print("len err:"))
        # print(valid_len_err)
        # print("####")
        # print("token dist err:")
        # print(valid_tokens_abs_err)
        # print("#######################")
        # ax1,fig1 = model.plt_grid_and_original_02(grid=pred_traj_gt.cpu(), grid_shape='o',ax1=None,fig1=None, colors_=1, color_alpha=0.5)
        # ax1,fig1 = model.plt_grid_and_original_02(grid=output_traj.cpu(), grid_shape='s',ax1=ax1,fig1=fig1, colors_=2)
        # fig1.save('<internal_user_root>/tst.png')
        # print("Validation example:")
        #         # print("")
        # print(output_text)

    # TODO: traj & rel abs avg err:
    ## 1. tokens to discerete numerics
    ## 2. discerete numrics to continous 
    ## 3. rel avg error
    
    ## 4. traj avg error
    
    #### 4.1. converting rel to traj first

    # TODO: visualizing a sample at each evaluation step (traj only)
    
    # TODO: token abs avg error
    
    # TODO: token len avg error (difference between true and predicted length), if len not equal to true len 0 padding is used for calculating the other metrics



# """
# An inner training loop compatible with both epoch-based and iter-based training.

# When using epoch-based, training stops after one epoch; when using iter-based,
# training stops after #iters_per_epoch iterations.
# """
# # use_amp = scaler is not None