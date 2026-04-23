"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import time
import warnings
import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample

from traj_utils import *
import wandb
import sys
from matplotlib import pyplot as plt
import numpy as np
from mm_viz import *

from inter_pred_utils import *

from extract_instruct_v3 import *

from torch.nn.parallel import DistributedDataParallel as DDP

from gameformer.utils.data_utils import *

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        # data_info_path = cfg.datasets_cfg.traj_align.build_info.storage + "data_info.pt"
        # model_config.data_info_path = data_info_path
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            if dataset_config.split == 'valid' and not cfg.run_cfg.evaluate:
                continue
            if dataset_config.split == 'train' and cfg.run_cfg.evaluate_only:
                continue
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            if 'train' in dataset.keys():
                dataset['train'].name = name
                if 'sample_ratio' in dataset_config:
                    dataset['train'].sample_ratio = dataset_config.sample_ratio
            elif 'valid' in dataset.keys():
                dataset['valid'].name = name

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        return model(samples)
        # loss = model(samples)["loss"]
        # return loss
    
    def train_step_v02(self, model, samples):
        outputs = model(samples)
        loss = outputs["loss"]
        return loss


    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError
    
    def valid_step(self, model, samples):
        return model(samples, _training=False)
        # outputs = model(samples, _training=False)
        # return outputs

    @torch.no_grad()
    def evaluation_gf(
        self, 
        model, 
        data_loader, 
        epoch, 
        iters_per_epoch, 
        cuda_enabled=True,
        start_iters=None,
        scaler=None,
        log_freq=10,
        didnt_work_repeat=2,
        num_eval_figs=0,
        rollouts=32,
        save_samples=False,
        output_dir='',
        ):
        # import pdb; pdb.set_trace()
        subsample = False
        metric_time = [5, 10, 15] if subsample else [29, 49, 79]

        if save_samples:
            os.makedirs(str(output_dir) + "/data", exist_ok=True)
        if num_eval_figs>0:
            log_images=True
            images_to_log = {}
        else:
            log_images=False
        use_amp = scaler is not None
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        # metric_logger.add_meter("worked", SmoothedValue(window_size=1, fmt="{value:.2f}"))
        # metric_logger.add_meter("counter", SmoothedValue(window_size=1, fmt="{value:.2f}"))
        metric_logger.add_meter("minADE", SmoothedValue(window_size=1, fmt="{value:.6f}"))

        # metric_logger.add_meter("minFDE", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        # metric_logger.add_meter("timemetric_minADE", SmoothedValue(window_size=1, fmt="{value:.6f}"))

        ## uncomment for eval only
        epoch_metrics = MotionMetrics() 
        act_epoch_metrics = MotionMetrics() 
        metric_names_ = config_util.get_breakdown_names_from_motion_config(epoch_metrics._metrics_config)
        metric_names = []
        for i, m in enumerate(['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']):
            for j, n in enumerate(metric_names_):
                metric_names.append(f'{m}_{n}')
                metric_names.append(f'act_{m}_{n}')
        for metric_name in metric_names:
            metric_logger.add_meter(metric_name, SmoothedValue(window_size=1, fmt="{value:.6f}"))
        ## end of uncomment

        logging.info(
            "Start validation epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Val: data epoch: [{}]".format(epoch)
        inner_epoch = epoch
        # for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
        for i in range(iters_per_epoch):
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                valid_output = self.valid_step(model, samples)
                pred_traj = valid_output['output_traj']
                

                
            batch_ade, batch_fde = ade_fde(pred_traj, samples['ground_truth'][...,:2])
            for b_i in range(samples['ground_truth'][...,:2].shape[0]):
                j = b_i
                ade, fde = batch_ade[b_i], batch_fde[b_i]
                
                # metric_logger.update(counter=1)
                # metric_logger.update(worked=1)
                metric_logger.update(minADE=ade.mean().item())
                # metric_logger.update(minFDE=fde.mean().item())
                # metric_logger.update(timemetric_minADE=ade[...,metric_time].mean().item())
                # waymo metrics:



                if 'modalities_act' in list(valid_output.keys()):
                    img_name = str(i) + 'ACT_'+valid_output['batch_data_names'][b_i]
                    images_to_log[img_name] = viz_multimodal(samples, valid_output['modalities_act'][:,0], samples['ground_truth'][:,0,:,:2], batch_sample=b_i)
                    img_name = str(i) + '**_'+valid_output['batch_data_names'][b_i]
                    images_to_log[img_name] = viz_multimodal(samples, valid_output['modalities'][:,0], samples['ground_truth'][:,0,:,:2], batch_sample=b_i)
                else:
                    if log_images and len(images_to_log)<=num_eval_figs:
                        # img_name = f'{inner_epoch}-{i}-{j} ADE_FDE: {np.mean(ade.flatten().tolist()):.2f} _ {fde.mean().item():.2f}'
                        img_name = f'{inner_epoch}-{i}-{b_i}- [mADE: {ade.mean().item():.2f}, mFDE: {fde.mean().item():.2f}]'
                        
                        # background
                        images_to_log[img_name] = vizualize(
                            samples['ego'][j].cpu(), 
                            samples['ground_truth'][j].cpu(), 
                            neighbors=samples['neighbors'][j].cpu(), 
                            map_lanes=samples['map_lanes'][j].cpu(), 
                            map_crosswalks=samples['map_crosswalks'][j].cpu(), 
                            region_dict=None, 
                            gt=False, 
                            fig=None,
                            background_only=True)
                        
                        # # all other modes than the best mode
                        # for roll_i in range(rollouts):
                        #     if roll_i== best_ade_mode:
                        #         continue
                        #     images_to_log[img_name] = vizualize(
                        #         pred_traj[roll_i,:,:3], 
                        #         pred_traj[roll_i,:,3:], 
                        #         neighbors=None, 
                        #         map_lanes=None, 
                        #         map_crosswalks=None, 
                        #         region_dict=None, 
                        #         gt=False, 
                        #         fig=images_to_log[img_name],
                        #         colors_mode=3)
                        
                        #best mode
                        images_to_log[img_name] = vizualize( 
                            pred_traj[b_i,:,:11].cpu(), 
                            pred_traj[b_i,:,11:].cpu(), 
                            neighbors=None, 
                            map_lanes=None, 
                            map_crosswalks=None, 
                            region_dict=None, 
                            gt=False, 
                            fig=images_to_log[img_name],
                            colors_mode=1)

                        #gt
                        images_to_log[img_name] = vizualize(
                            samples['ego'][j].cpu(), 
                            samples['ground_truth'][j].cpu(), 
                            neighbors=None, 
                            map_lanes=None, 
                            map_crosswalks=None, 
                            region_dict=None, 
                            gt=True, 
                            fig=images_to_log[img_name])

                        plt.title(img_name)
                        plt.close(images_to_log[img_name]) 

            if save_samples:
                samples_output_file = str(output_dir) + f"/data/{samples['file_name'][j]}_pred_roll{len(valid_rollouts_range)}.npz"
                np.savez(samples_output_file, 
                ego=np.array(samples['ego'][j].cpu()), 
                neighbors=np.array(samples['neighbors'][j].cpu()), 
                map_lanes=np.array(samples['map_lanes'][j].cpu()), 
                map_crosswalks=np.array(np.array(samples['map_crosswalks'][j].cpu())),
                gt_future_states=np.array(np.array(samples['ground_truth'][j].cpu())),
                pred_traj=np.array(pred_traj.cpu()),
                )

            
            ## uncomment for eval only
            ## waymo results:
            egos = valid_output['modalities'].transpose(1,2)
            actors = samples['ego']
            actors_future = samples['ground_truth']
            ego_ground_truth = torch.cat([actors[:, :, :, :5], actors_future], dim=2)
            ego_ground_truth = torch.cat([
                ego_ground_truth[:, :, :, :2], 
                actors[:,:, -1, 5:7].unsqueeze(2).expand(-1,-1, ego_ground_truth.shape[2], -1), 
                ego_ground_truth[:, :, :, 2:]
                ], dim=-1)
            scores = valid_output['scores'].sum(1)
            scores = F.softmax(scores,dim=-1)
            object_type = samples['data_object_type']
            epoch_metrics.update_state(
                    egos.float(), scores.float(), 
                    ego_ground_truth, torch.ne(ego_ground_truth, 0).bool(), 
                    object_type.long()
                    )
            # epoch_metrics_results = epoch_metrics.result()
            # epoch_metrics_results = {k:float(v) for k,v in epoch_metrics_results.items() if v!=-1}
            # metric_logger.update_(epoch_metrics_results)
            
            if 'modalities_act' in list(valid_output.keys()):
                egos = valid_output['modalities_act'].transpose(1,2) #the only difference
                act_epoch_metrics.update_state(
                    egos.float(), scores.float(), 
                    ego_ground_truth, torch.ne(ego_ground_truth, 0).bool(), 
                    object_type.long()
                    )
                # act_epoch_metrics_results = act_epoch_metrics.result()
                # act_epoch_metrics_results = {'act_'+k:float(v) for k,v in act_epoch_metrics_results.items() if v!=-1}
                # metric_logger.update_(act_epoch_metrics_results)



            # if wandb.run is not None and (i+1)%20:
            if wandb.run and (i+1)%20 == 0:
                epoch_metrics_results = epoch_metrics.result()
                epoch_metrics_results = {k:float(v) for k,v in epoch_metrics_results.items() if v!=-1}
                act_epoch_metrics_results = act_epoch_metrics.result()
                act_epoch_metrics_results = {'act_'+k:float(v) for k,v in act_epoch_metrics_results.items() if v!=-1}
                avg_metrics = epoch_metrics_results
                avg_metrics.update(act_epoch_metrics_results)
                wandb.log(avg_metrics)

            # if True:
                # metric_logger.synchronize_between_processes()
                # avg_metrics = {}
                for metric_j in ['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']:
                    avg_metrics['avg_'+metric_j] = np.mean([metric_logger.meters[name].global_avg for name in metric_names if metric_j in name and name in epoch_metrics_results.keys()])
                    avg_metrics['act_avg_'+metric_j] = np.mean([metric_logger.meters[name].global_avg for name in metric_names if metric_j in name and name in act_epoch_metrics_results.keys()])
                    for object_j in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
                        avg_metrics['avg_'+metric_j+'_'+object_j] = np.mean([metric_logger.meters[name].global_avg for name in metric_names if metric_j in name and object_j in name and name in epoch_metrics_results.keys()])
                        avg_metrics['act_avg_'+metric_j+'_'+object_j] = np.mean([metric_logger.meters[name].global_avg for name in metric_names if metric_j in name and object_j in name and name in act_epoch_metrics_results.keys()])
                wandb.log(avg_metrics)

                

            # if wandb.run is not None:
            #     metric_logger.synchronize_between_processes()
            #     running_avg_logs = {'minADE_runningAvg': metric_logger.meters['minADE'].global_avg, 'minFDE_runningAvg': metric_logger.meters['minFDE'].global_avg}
            #     running_avg_logs.update({'timemetric_minADE_runningAvg': metric_logger.meters['timemetric_minADE'].global_avg})
            #     wandb.log(running_avg_logs)
                
        
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        # logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return_dict = {}
        for k, meter in metric_logger.meters.items():
            if k=='worked' or k=='counter':
                continue
            else:
                return_dict[k] = float("{:.8f}".format(meter.global_avg))
        return_dict['figures'] = images_to_log
        # return_dict['worked_count'] = metric_logger.meters['worked'].count
        # return_dict['failed_count'] = metric_logger.meters['counter'].count - metric_logger.meters['worked'].count
        # return_dict['failed_perc'] = (metric_logger.meters['counter'].count - metric_logger.meters['worked'].count)/metric_logger.meters['counter'].count
        
        # avg_metrics = {}
        # for metric_j in ['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']:
        #     avg_metrics['waymo_avg_'+metric_j] = np.mean([metric_logger.meters[name].global_avg for name in metric_names if metric_j in name])
        #     for object_j in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
        #         avg_metrics['waymo_avg_'+metric_j+'_'+object_j] = np.mean([metric_logger.meters[name].global_avg for name in metric_names if metric_j in name and object_j in name])
        # return_dict.update(avg_metrics)
        print('finished evaluation')
        return return_dict

    def var_copy(self, input):
        if torch.is_tensor(input):
            return input.detach().clone()
        elif isinstance(input, list):
            return input.copy()
        else:
            return input

    def gen_contrastive_samples(self, samples):
        
        contrastive_instruct = np.array([samples['contrastive_instruct'][i].split(';') for i in range(len(samples['contrastive_instruct']))])
        num_contrastive = len(contrastive_instruct[0])
        samples__ = {k: self.var_copy(samples[k]) for k in samples}
        contrastive_flag = torch.ones(num_contrastive*(len(samples['instruct'])+1)).bool().to(samples['traj'].device)
        contrastive_flag[:len(samples['instruct'])] = False
        for contrastive_i in range(num_contrastive):
            samples_ = {k: self.var_copy(samples__[k]) for k in samples__}
            samples_['instruct'] = list(contrastive_instruct[:,contrastive_i])
            for k in samples.keys():
                if torch.is_tensor(samples[k]):
                    samples[k] = torch.cat((samples[k], samples_[k]), dim=0)
                else:
                    samples[k] = samples[k] + samples_[k]
        samples['contrastive_flag'] = contrastive_flag

    @torch.no_grad()
    def evaluation_only_gf(
        self, 
        model, 
        data_loader, 
        epoch, 
        iters_per_epoch, 
        cuda_enabled=True,
        start_iters=None,
        scaler=None,
        log_freq=10,
        didnt_work_repeat=2,
        num_eval_figs=0,
        rollouts=32,
        save_samples=False,
        output_dir='',
        contrastive_act=False,
        new_eval_mode='',
        ):
        # import pdb; pdb.set_trace()
        inference_time = []
        pred_dicts = []

        figures_uploaded = False
        subsample = False
        metric_time = [5, 10, 15] if subsample else [29, 49, 79]
        
        if save_samples:
            output_dir_ = output_dir / f'result/{new_eval_mode}_eval{epoch}' / 'data'
            os.makedirs(output_dir_, exist_ok=True)
            # os.makedirs(str(output_dir) + "/data", exist_ok=True)
        if num_eval_figs>0:
            log_images=True
            images_to_log = {}
        else:
            log_images=False
        use_amp = scaler is not None
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        # Nussair_WACV
        # ---- timing/memory: eval scope start ----
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        eval_start = time.perf_counter()
        model_ms_list = []
        iter_ms_list = []
        # -----------------------------------------


        ## uncomment for eval only
        # epoch_metrics = MotionMetrics() 
        # act_epoch_metrics = MotionMetrics() 
        # metric_names_ = config_util.get_breakdown_names_from_motion_config(epoch_metrics._metrics_config)
        # metric_names = []
        # for i, m in enumerate(['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']):
        #     for j, n in enumerate(metric_names_):
        #         metric_names.append(f'{m}_{n}')
        #         metric_names.append(f'act_{m}_{n}')
        # ## end of uncomment

        inner_epoch = epoch
        # for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
        ADE_ = []
       
        for i in tqdm(range(iters_per_epoch)):
            if i >= iters_per_epoch:
                break
            
             # Nussair_WACV
            # ---- iteration timer start (prep + forward) ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_iter_start = time.perf_counter()
            # -----------------------------------------------

            samples = next(data_loader)
            
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            # model.to(torch.bfloat16)
            # samples = {k:v.to(torch.bfloat16) if torch.is_tensor(v) else v for k,v in samples.items()}
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )
            if contrastive_act:
                self.gen_contrastive_samples(samples)
            
            
            
            # start_time = time.time()
            # Nussair_WACV
            # ---- model latency timer (forward only) ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_model_start = time.perf_counter()
            # ---- cont 

            if use_amp:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    valid_output = self.valid_step(model, samples)
                    pred_traj = valid_output['output_traj']
            else:
                # with torch.cuda.amp.autocast(enabled=True):
                    # with maybe_autocast(torch.bfloat16):
                valid_output = self.valid_step(model, samples)

            # Nussair_WACV
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_ms_list.append((time.perf_counter() - t_model_start) * 1000.0)
            # --------------------------------------------

            # Nussair_WACV
            # ---- iteration timer stop (prep + forward; excludes disk I/O) ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iter_ms_list.append((time.perf_counter() - t_iter_start) * 1000.0)
            # ------------------------------------------------------------------

            # inference_time_ = time.time() - start_time
            # inference_time.append(inference_time_)
            # print(inference_time_)

            if save_samples:
                if 'input_dict' in samples and 'mtr_id' in samples['input_dict']:
                    batch_pred_dicts = valid_output
                    # filename = samples['input_dict']['filename'] 
                    final_output_dir = output_dir_
                    final_pred_dicts = data_loader._dataloader.dataset.generate_prediction_dicts(batch_pred_dicts, output_path=final_output_dir)
                    pred_dicts += final_pred_dicts
                    batch = samples
                    for file_i, filename in enumerate(batch['input_dict']['filename']):
                        to_save_dict = {}
                        save_dir = final_output_dir / (filename+'.pkl')
                        # get the center and heading (local view)
                        #[np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading, x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
                        to_save_dict['gt_trajs'] = final_pred_dicts[file_i][0]['gt_trajs'][:, [0,1,6,7,8]].copy()
                        center, angle = to_save_dict['gt_trajs'][10][:2].copy(), to_save_dict['gt_trajs'][10][2].copy()
                        to_save_dict['gt_trajs'] = agent_norm(to_save_dict['gt_trajs'], center, angle, impute=True).astype(np.float16)
                        to_save_dict['modalities'] = final_pred_dicts[file_i][0]['pred_trajs'].astype(np.float16)
                        for m_i, m in enumerate(to_save_dict['modalities']):
                            to_save_dict['modalities'][m_i] = agent_norm(m.copy(), center, angle)
                        to_save_dict['scores'] = final_pred_dicts[file_i][0]['pred_scores'].astype(np.float16)
                        to_save_dict['act'] = batch['input_dict']['act'][file_i]
                        
                        to_save_dict['modalities'] = to_save_dict['modalities'][None]
                        to_save_dict['scores'] = to_save_dict['scores'][None]
                        # Save the dictionary as a .pkl file
                        with open(save_dir, 'wb') as f:
                            pickle.dump(to_save_dict, f)
                else:
                    # Save dictionary using pickle
                    for i_save in range(len(samples['file_name'])):
                        sample_valid_output = {k:v[i_save].to(torch.float16).cpu().numpy() for k,v in valid_output.items() if k!='gf_loss' and k!= 'text'}
                        sample_valid_output.update({'loss': valid_output['gf_loss'].cpu()})
                        sample_valid_output.update({'text': valid_output['text'][i_save]})
                        if 'act' in samples:
                            sample_valid_output.update({'act': samples['act'][i_save].cpu()})
                            if 'act_2' in samples:
                                sample_valid_output.update({'act_2': samples['act_2'][i_save].cpu()}) 
                        # if 'complex' in str(output_dir_):
                            sample_valid_output.update({'instruct': '.'.join(samples['instruct'][i_save].split('<s>[INST] ')[1].split('.')[1:]).strip(' ')})
                            sample_valid_output.update({'caption': samples['caption'][i_save].split(' Decision')[0]})
                            sample_valid_output.update({'answer': valid_output['text'][i_save].split(' Decision')[0]})

                        with open(f"{output_dir_}/{samples['file_name'][i_save]}.pkl", 'wb') as f:
                            pickle.dump(sample_valid_output, f)
                    
                
        if save_samples and 'input_dict' in samples and 'mtr_id' in samples['input_dict']:
                result_dir = '/'.join(str(final_output_dir).split('/')[:-1]) + '/result.pkl'
                with open(result_dir, 'wb') as f:
                    pickle.dump(pred_dicts, f)

                # result_str, result_dict = data_loader._dataloader.dataset.evaluation(
                #     pred_dicts,
                #     output_path=final_output_dir, 
                # )
                    
                    # with open(f"{output_dir_}/{samples['file_name'][i_save]}.pkl", 'rb') as f:
                    #     data_loaded = pickle.load(f)
                # pred_traj = valid_output['output_traj']
                

                
            # batch_ade, batch_fde = ade_fde(pred_traj, samples['ground_truth'][...,:2])
            # for b_i in range(samples['ground_truth'][...,:2].shape[0]):
            #     j = b_i
            #     ade, fde = batch_ade[b_i], batch_fde[b_i]
                
            #     ADE_.append(ade.mean().item())

            #     if not figures_uploaded:
            #         if 'modalities_act' in list(valid_output.keys()):
            #             img_name = str(i) + 'ACT_'+valid_output['batch_data_names'][b_i]
            #             images_to_log[img_name] = viz_multimodal(samples, valid_output['modalities_act'][:,0], samples['ground_truth'][:,0,:,:2], batch_sample=b_i)
            #             img_name = str(i) + '**_'+valid_output['batch_data_names'][b_i]
            #             images_to_log[img_name] = viz_multimodal(samples, valid_output['modalities'][:,0], samples['ground_truth'][:,0,:,:2], batch_sample=b_i)
            
            ## uncomment for eval only
            ## waymo results:
            # egos = valid_output['modalities'].transpose(1,2)
            # actors = samples['ego']
            # actors_future = samples['ground_truth']
            # ego_ground_truth = torch.cat([actors[:, :, :, :5], actors_future], dim=2)
            # ego_ground_truth = torch.cat([
            #     ego_ground_truth[:, :, :, :2], 
            #     actors[:,:, -1, 5:7].unsqueeze(2).expand(-1,-1, ego_ground_truth.shape[2], -1), 
            #     ego_ground_truth[:, :, :, 2:]
            #     ], dim=-1)
            # scores = valid_output['scores'].sum(1)
            # scores = F.softmax(scores,dim=-1)
            # object_type = samples['data_object_type']
            # epoch_metrics.update_state(
            #         egos.float(), scores.float(), 
            #         ego_ground_truth, torch.ne(ego_ground_truth, 0).bool(), 
            #         object_type.long()
            #         )
            
            # if 'modalities_act' in list(valid_output.keys()):
            #     egos = valid_output['modalities_act'].transpose(1,2) #the only difference
            #     act_epoch_metrics.update_state(
            #         egos.float(), scores.float(), 
            #         ego_ground_truth, torch.ne(ego_ground_truth, 0).bool(), 
            #         object_type.long()
            #         )
            
            # unknown_instruct_flag = [instruct_i == 'unknown' for instruct_i in samples['instruct']]
            
            
            
            # pred_instruct_int = get_batch_instruct(egos[:,0,0].cpu(), None,num_classes=8)[:,0]

    
            # # if save_samples:
            # if False:
            #     filename = str(output_dir_)+f'/iter{i}.npz'
            #     np.savez(filename, 
            #     ## Waymo
            #     egos=egos.float().cpu().numpy(),
            #     scores=scores.float().cpu().numpy(),
            #     ego_ground_truth=ego_ground_truth.cpu().numpy(),
            #     ego_ground_truth_mask=torch.ne(ego_ground_truth, 0).bool().cpu().numpy(),
            #     object_type = object_type.long().cpu().numpy(),
            #     ## Other metrics data
            #     ego_state = samples['ego_state'].cpu().numpy(), # history ego
            #     neighbor_state = samples['neighbors_state'][:,0].cpu().numpy(), # history interactive
            #     instruct = samples['instruct'],
            #     contrastive_flag = samples['contrastive_flag'].cpu().numpy() if 'contrastive_flag' in samples.keys() else None,
            #     file_name = samples['file_name'],
            #     unknown_instruct_flag = unknown_instruct_flag,
            #     gt = ego_ground_truth[:,0,11:,:2].cpu().numpy(),
            #     pred = egos[:,:,0].cpu().numpy(),
            #     scenario_id = [file_name_.split('_')[0] for file_name_ in samples['file_name']],
            #     # pred_instruct = pred_instruct_int,
            #     )
            # else:
            #     filename = '/'.join(str(output_dir_).split('/')[:-1])+f'/figures/'
            #     batch_sample = 1+8*4
            #     # batch_sample = batch_sample*8
            #     image_name = samples['file_name'][batch_sample]+'png'
            #     fig = viz_multimodal_2agent(samples, valid_output['modalities'], samples['ground_truth'], batch_sample=batch_sample)
            #     plt.xlim(-5,90)
            #     plt.ylim(-15,15)
            #     fig.savefig('ex1.png', dpi=300)

            #     batch_sample = 1+8*5
            #     # batch_sample = batch_sample*8
            #     image_name = samples['file_name'][batch_sample]+'png'
            #     fig = viz_multimodal_2agent(samples, valid_output['modalities'], samples['ground_truth'], batch_sample=batch_sample)
            #     plt.xlim(-5,90)
            #     plt.ylim(-15,15)
            #     fig.savefig('ex2.png', dpi=300)
            #     print(samples['instruct'][batch_sample])
               


            # # if wandb.run is not None and (i+1)%20:
            # if wandb.run and (i+1)%10 == 0:
            #     epoch_metrics_results = epoch_metrics.result()
            #     epoch_metrics_results = {k:float(v) for k,v in epoch_metrics_results.items() if v!=-1}
            #     act_epoch_metrics_results = act_epoch_metrics.result()
            #     act_epoch_metrics_results = {'act_'+k:float(v) for k,v in act_epoch_metrics_results.items() if v!=-1}
            #     avg_metrics = epoch_metrics_results
            #     avg_metrics.update(act_epoch_metrics_results)

            #     for metric_j in ['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']:
            #         avg_metrics['avg_'+metric_j] = np.mean([epoch_metrics_results[name] for name in metric_names if metric_j in name and 'act' not in name])
            #         avg_metrics['act_avg_'+metric_j] = np.mean([act_epoch_metrics_results[name] for name in metric_names if metric_j in name and 'act' in name])
            #         for object_j in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
            #             avg_metrics['avg_'+metric_j+'_TYPE_'+object_j] = np.mean([epoch_metrics_results[name] for name in metric_names if metric_j in name and object_j in name and 'act' not in name])
            #             avg_metrics['act_avg_'+metric_j+'_TYPE_'+object_j] = np.mean([act_epoch_metrics_results[name] for name in metric_names if metric_j in name and object_j in name and 'act' in name])
                
            #     avg_metrics.update({'ADE': np.mean(ADE_)})

            #     if i>8 and not figures_uploaded:
            #         figures_uploaded=True
            #         wandb_images = [wandb.Image(value, caption=key) for key, value in images_to_log.items()]
            #         if len(wandb_images)>100:
            #             wandb_images = wandb_images[:100]
            #         avg_metrics.update({"images": wandb_images})


            #     wandb.log(avg_metrics)

                

        # return_dict = avg_metrics
        # return_dict['figures'] = images_to_log
        return_dict = {}

        # return_dict['worked_count'] = metric_logger.meters['worked'].count
        # return_dict['failed_count'] = metric_logger.meters['counter'].count - metric_logger.meters['worked'].count
        # return_dict['failed_perc'] = (metric_logger.meters['counter'].count - metric_logger.meters['worked'].count)/metric_logger.meters['counter'].count
        
        # avg_metrics = {}
        # for metric_j in ['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']:
        #     avg_metrics['waymo_avg_'+metric_j] = np.mean([metric_logger.meters[name].global_avg for name in metric_names if metric_j in name])
        #     for object_j in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
        #         avg_metrics['waymo_avg_'+metric_j+'_'+object_j] = np.mean([metric_logger.meters[name].global_avg for name in metric_names if metric_j in name and object_j in name])
        # return_dict.update(avg_metrics)
        print('finished evaluation')
        # ---- finalize timing/memory ----
        eval_total_s = time.perf_counter() - eval_start

        avg_model_ms = float(np.mean(model_ms_list)) if len(model_ms_list) else float('nan')
        avg_iter_ms  = float(np.mean(iter_ms_list))  if len(iter_ms_list)  else float('nan')

        return_dict['avg_model_ms'] = avg_model_ms
        return_dict['avg_iter_ms']  = avg_iter_ms
        return_dict['eval_total_s'] = eval_total_s

        if torch.cuda.is_available():
            peak_alloc_mb    = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved_mb = torch.cuda.max_memory_reserved()  / (1024**2)
            return_dict['peak_mem_alloc_MB']    = float(peak_alloc_mb)
            return_dict['peak_mem_reserved_MB'] = float(peak_reserved_mb)

        # Optional: print to SLURM log
        logging.info(f"[EVAL] avg_model_ms={avg_model_ms:.2f} | avg_iter_ms={avg_iter_ms:.2f} | total_s={eval_total_s:.2f}")
        if torch.cuda.is_available():
            logging.info(f"[EVAL] peak_mem_alloc={peak_alloc_mb:.2f} MB | peak_mem_reserved={peak_reserved_mb:.2f} MB")
        # --------------------------------

        return return_dict

    @torch.no_grad()
    def evaluation(
        self, 
        model, 
        data_loader, 
        epoch, 
        iters_per_epoch, 
        cuda_enabled=True,
        start_iters=None,
        scaler=None,
        log_freq=1,
        didnt_work_repeat=2,
        num_eval_figs=0,
        rollouts=32,
        save_samples=False,
        output_dir='',
        ):
        # import pdb; pdb.set_trace()
        if save_samples:
            os.makedirs(str(output_dir) + "/data", exist_ok=True)
        if num_eval_figs>0:
            log_images=True
            images_to_log = {}
        else:
            log_images=False
        use_amp = scaler is not None
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("ADE_mean", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("FDE_mean", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("recon_ADE_mean", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("recon_FDE_mean", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("worked", SmoothedValue(window_size=1, fmt="{value:.2f}"))
        metric_logger.add_meter("counter", SmoothedValue(window_size=1, fmt="{value:.2f}"))


        if rollouts>1:
            metric_logger.add_meter("minADE", SmoothedValue(window_size=1, fmt="{value:.6f}"))
            metric_logger.add_meter("minFDE", SmoothedValue(window_size=1, fmt="{value:.6f}"))

        
        logging.info(
            "Start validation epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Val: data epoch: [{}]".format(epoch)
        inner_epoch = epoch
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break
            # -------- end-to-end timer start --------
            # Nussair_WACV
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_iter_start = time.perf_counter()

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            # Nussair_WACV
            # -------- model-only timer --------
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()

            with torch.cuda.amp.autocast(enabled=use_amp):
                rel_reconstructed, valid_rel, valid_rel_rollout, scores = self.valid_step(model, samples)

            # Nussair_WACV
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                model_ms = start_event.elapsed_time(end_event)  # milliseconds
            else:
                # CPU fallback (rough timing if no CUDA)
                model_ms = (time.perf_counter() - t_iter_start) * 1000.0


            for _ in range(samples['traj'].shape[0]):
                metric_logger.update(counter=1)
            # rel_reconstructed = rel_reconstructed[valid_rel]
            # gt_traj = samples['traj'][valid_rel]
            # if sum(valid_rel)==0:
                # metric_logger.synchronize_between_processes()
                # return {'worked_count':0}
                # Nothing worked in a batch
            
            # for j in range(sum(valid_rel)):
            valid_samples_range = [valid_i for valid_i in range(len(valid_rel)) if valid_rel[valid_i]]
            for j in valid_samples_range:
                metric_logger.update(worked=1)
                
                if rollouts==1:
                    pred_traj = velocity_to_abs_distance(rel_reconstructed[j], gt_traj[j][:,:1,:].cpu())
                    ade, fde = ade_fde(pred_traj[:,3:,:], gt_traj[j,:, 3:,:].cpu())
                    recon_ade, recon_fde =  ade_fde(pred_traj[:,:3,:], gt_traj[j][:, :3,:].cpu())

                    if log_images and len(images_to_log)<=num_eval_figs:
                        img_name = f'{inner_epoch}-{i}-{j} ADE_FDE: {np.mean(ade.flatten().tolist()):.2f} _ {fde.mean().item():.2f}'
                        images_to_log[img_name] = vizualize(
                            samples['ego'][j].cpu(), 
                            samples['ground_truth'][j].cpu(), 
                            neighbors=samples['neighbors'][j].cpu(), 
                            map_lanes=samples['map_lanes'][j].cpu(), 
                            map_crosswalks=samples['map_crosswalks'][j].cpu(), 
                            region_dict=None, 
                            gt=True, 
                            fig=None)
                        images_to_log[img_name] = vizualize(
                            pred_traj[:,:3], 
                            pred_traj[:,3:], 
                            neighbors=None, 
                            map_lanes=None, 
                            map_crosswalks=None, 
                            region_dict=None, 
                            gt=False, 
                            fig=images_to_log[img_name])
                        plt.title(img_name)
                        plt.close(images_to_log[img_name]) 
                else:
                    metric_time = [5, 10, 15]
                    # pred_traj = velocity_to_abs_distance(rel_reconstructed[j,0], gt_traj[j][:,:1,:].cpu())
                    # ade, fde = ade_fde(pred_traj[:,3:,:], gt_traj[j,:, 3:,:].cpu())
                    # recon_ade, recon_fde =  ade_fde(pred_traj[:,:3,:], gt_traj[j][:, :3,:].cpu())

                    valid_rollouts_range = [rollout_i for rollout_i in range(rollouts) if valid_rel_rollout[j][rollout_i]]
                    pred_rel = torch.stack([rel_reconstructed[j,roll_k] for roll_k in valid_rollouts_range])
                    pred_scores = torch.stack([scores[j,roll_k] for roll_k in valid_rollouts_range])
                    gt_traj = samples['traj']
                    pred_traj = torch.stack([
                        velocity_to_abs_distance(rel_reconstructed[j,roll_k], gt_traj[j][:,:1,:].cpu()) for roll_k in valid_rollouts_range
                    ])

                    # pred_traj = torch.stack([
                    #     velocity_to_abs_distance(rel_reconstructed[j,roll_k], gt_traj[j][:,:1,:].cpu()) for roll_k in range(rollouts)
                    # ])
                    gt_to_compare = gt_traj[j,:, 3:,:].cpu().unsqueeze(0).repeat(pred_traj.shape[0],1,1,1)
                    gt_to_compare = gt_to_compare[:,:,metric_time,:]
                    pred_traj_to_compare = pred_traj[:,:,3:,:]
                    pred_traj_to_compare = pred_traj_to_compare[:,:,metric_time,:]
                    ade_roll, fde_roll = ade_fde(pred_traj_to_compare, gt_to_compare)
                    best_ade_mode = ade_roll.mean(-1).mean(-1).argmin()
                    min_ade = ade_roll[best_ade_mode]
                    min_ade = ade_roll[best_ade_mode].mean(0).mean(0).item() # the first mean calculate across agent, per metric_time. This line uses the best mode to calculate the minADE
                    mean_min_ade = min_ade
                    # min_ade = ade_roll.transpose(0,1).transpose(1,2).min(-1).values # this finds the closest possible points, not using the best mode only
                    # mean_min_ade = min_ade.flatten().mean().item()
                    best_fde_mode = fde_roll.mean(-1).argmin()
                    min_fde = fde_roll[best_fde_mode]
                    # min_fde = fde_roll.min(0).values
                    mean_min_fde = min_fde.mean().item()
                    metric_logger.update(minADE=mean_min_ade)
                    metric_logger.update(minFDE=mean_min_fde)

                    best_pred_traj = pred_traj[best_ade_mode]
                    ade, fde = ade_fde(best_pred_traj[:,3:,:], gt_traj[j,:, 3:,:].cpu())
                    recon_ade, recon_fde =  ade_fde(best_pred_traj[:,:3,:], gt_traj[j][:, :3,:].cpu())

                    # code to save prediction and input features
                    if save_samples:
                        samples_output_file = str(output_dir) + f"/data/{samples['file_name'][j]}_pred_roll{len(valid_rollouts_range)}.npz"
                        np.savez(samples_output_file, 
                        ego=np.array(samples['ego'][j].cpu()), 
                        neighbors=np.array(samples['neighbors'][j].cpu()), 
                        map_lanes=np.array(samples['map_lanes'][j].cpu()), 
                        map_crosswalks=np.array(np.array(samples['map_crosswalks'][j].cpu())),
                        gt_future_states=np.array(np.array(samples['ground_truth'][j].cpu())),
                        pred_traj=np.array(pred_traj.cpu()),
                        pred_rel=np.array(pred_rel.cpu()),
                        pred_scores=np.array(pred_scores.cpu())
                        )
                        

                    if log_images and len(images_to_log)<=num_eval_figs:
                        # img_name = f'{inner_epoch}-{i}-{j} ADE_FDE: {np.mean(ade.flatten().tolist()):.2f} _ {fde.mean().item():.2f}'
                        img_name = f'{inner_epoch}-{i}-{j}- [mADE: {mean_min_ade:.2f}]'
                        
                        # background
                        images_to_log[img_name] = vizualize(
                            samples['ego'][j].cpu(), 
                            samples['ground_truth'][j].cpu(), 
                            neighbors=samples['neighbors'][j].cpu(), 
                            map_lanes=samples['map_lanes'][j].cpu(), 
                            map_crosswalks=samples['map_crosswalks'][j].cpu(), 
                            region_dict=None, 
                            gt=False, 
                            fig=None,
                            background_only=True)
                        
                        # all other modes than the best mode
                        for roll_i in range(rollouts):
                            if roll_i== best_ade_mode:
                                continue
                            images_to_log[img_name] = vizualize(
                                pred_traj[roll_i,:,:3], 
                                pred_traj[roll_i,:,3:], 
                                neighbors=None, 
                                map_lanes=None, 
                                map_crosswalks=None, 
                                region_dict=None, 
                                gt=False, 
                                fig=images_to_log[img_name],
                                colors_mode=3)
                        
                        #best mode
                        images_to_log[img_name] = vizualize( 
                            pred_traj[best_ade_mode,:,:3], 
                            pred_traj[best_ade_mode,:,3:], 
                            neighbors=None, 
                            map_lanes=None, 
                            map_crosswalks=None, 
                            region_dict=None, 
                            gt=False, 
                            fig=images_to_log[img_name],
                            colors_mode=1)

                        #gt
                        images_to_log[img_name] = vizualize(
                            samples['ego'][j].cpu(), 
                            samples['ground_truth'][j].cpu(), 
                            neighbors=None, 
                            map_lanes=None, 
                            map_crosswalks=None, 
                            region_dict=None, 
                            gt=True, 
                            fig=images_to_log[img_name])

                        plt.title(img_name)
                        plt.close(images_to_log[img_name]) 

                        # wandb.log({'image': wandb.Image(images_to_log[img_name], caption=img_name)})
                    # pred_traj = velocity_to_abs_distance(rel_reconstructed[j,0], gt_traj[j][:,:1,:].cpu()) #Only one used for vizualization
                    

                metric_logger.update(ADE_mean=np.mean(ade.flatten().tolist()))
                metric_logger.update(FDE_mean=fde.mean().item())
                metric_logger.update(recon_ADE_mean=np.mean(recon_ade.flatten().tolist()))
                metric_logger.update(recon_FDE_mean=recon_fde.mean().item())

                # Nussair_WACV
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                iter_ms = (time.perf_counter() - t_iter_start) * 1000.0
                metric_logger.update(model_ms=model_ms, iter_ms=iter_ms)
                # if log_images and len(images_to_log)<=num_eval_figs:
                #     img_name = f'{inner_epoch}-{i}-{j} ADE_FDE: {np.mean(ade.flatten().tolist()):.2f} _ {fde.mean().item():.2f}'
                #     images_to_log[img_name] = vizualize(
                #         samples['ego'][j].cpu(), 
                #         samples['ground_truth'][j].cpu(), 
                #         neighbors=samples['neighbors'][j].cpu(), 
                #         map_lanes=samples['map_lanes'][j].cpu(), 
                #         map_crosswalks=samples['map_crosswalks'][j].cpu(), 
                #         region_dict=None, 
                #         gt=True, 
                #         fig=None)
                #     images_to_log[img_name] = vizualize(
                #         pred_traj[:,:3], 
                #         pred_traj[:,3:], 
                #         neighbors=None, 
                #         map_lanes=None, 
                #         map_crosswalks=None, 
                #         region_dict=None, 
                #         gt=False, 
                #         fig=images_to_log[img_name])
                #     plt.close(images_to_log[img_name]) 
        
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        # logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return_dict = {}
        for k, meter in metric_logger.meters.items():
            if k=='worked' or k=='counter':
                continue
            else:
                return_dict[k] = float("{:.8f}".format(meter.global_avg))
        return_dict['worked_count'] = metric_logger.meters['worked'].count
        return_dict['failed_count'] = metric_logger.meters['counter'].count - metric_logger.meters['worked'].count
        return_dict['failed_perc'] = (metric_logger.meters['counter'].count - metric_logger.meters['worked'].count)/metric_logger.meters['counter'].count
        return_dict['figures'] = images_to_log

        # Nussair_WACV
        logging.info(f"Average model forward latency: {return_dict['model_ms']:.2f} ms")
        logging.info(f"Average iteration latency (end-to-end): {return_dict['iter_ms']:.2f} ms")

        return return_dict
                        # images_to_log[img_name] = vizualize(pred_traj[:,:3], pred_traj[:,3:], neighbors=None, map_lanes=None, map_crosswalks=None, region_dict=None, gt=False, fig=images_to_log[img_name])
        # print_freq = 10
        # if num_figs>0:
        #     images_to_log = {}
        # results = []

        # for samples in metric_logger.log_every(data_loader, print_freq, header):
        #     samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

        #     eval_output = self.valid_step(model=model, samples=samples)
        #     results.extend(eval_output)

        # if is_dist_avail_and_initialized():
        #     dist.barrier()

        # return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.7f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.7f}"))
        metric_logger.add_meter("loss_2", SmoothedValue(window_size=1, fmt="{value:.7f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)
        
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            # with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            # with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                train_outputs = self.train_step(model=model, samples=samples)
                loss = train_outputs['loss']

            # after_train_step()
            # if not torch.isnan(loss):
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # # if you want to play with the gradient and masking some, it should be done here after the loss.backward
            # model.on_before_optimizer_step()
            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                max_grad_norm = 10.0
                
                # if not torch.isnan(loss):
                # if model.gameformer_enc:
                # model.on_before_optimizer_step(optimizer)
                if use_amp:
                    if isinstance(model, DDP):
                        model.module.on_before_optimizer_step()
                    else:
                        model.on_before_optimizer_step()
                    scaler.unscale_(optimizer)
                    if max_grad_norm>0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    # if you want to play with the gradient and masking some, it should be done here after the loss.backward
                    
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    if isinstance(model, DDP):
                    # if True:
                        # print("*****************************************************")
                        # print("*****************************************************")
                        # print("*****************************************************")
                        model.module.on_before_optimizer_step()
                    else:
                        # print("#####################################################")
                        # print("#####################################################")
                        # print("#####################################################")
                        model.on_before_optimizer_step()
                    if max_grad_norm>0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
            # if not torch.isnan(loss):
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if 'gf_loss' in train_outputs.keys():
                metric_logger.update(loss_2=train_outputs['gf_loss'].item())
            else:
                metric_logger.update(loss_2=loss.item())

            
            # else:
            #     warnings.warn("########### !!!! Nan loss observed and skipped !!!! ###########")

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.8f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
