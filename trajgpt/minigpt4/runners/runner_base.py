"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path
import math
import torch
import torch.distributed as dist
import webdataset as wds
from minigpt4.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn import DataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import copy

@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.eval_only=False
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0

        # self.setup_seeds()
        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed and not self.eval_only:
                if self._wrapped_model is None:
                    # print("<><><><><><><><><><><><><> DDPPPPP")
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu]
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # frozen weights
                # print(n)
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run_cfg.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            
            if self.config.model_cfg.gameformer_enc:
                lr = self.config.run_cfg.init_lr
                # self.model.gameformer_model is not used since it is not trainable, it is included in self.model
                optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if p.requires_grad if 'llm_output_adapter' not in n and 'llm_input_adapter' not in n],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if p.requires_grad if  'llm_output_adapter' in n or 'llm_input_adapter' in n],
                    # "lr": lr * 10,
                },
                # {
                #     "params": [p for n, p in self.model.named_parameters() if p.requires_grad if  'llm_output_adapter' in n or 'llm_input_adapter' in n],
                #     "lr": lr * 10,
                # },
                
                # {
                #     "params": [p for n, p in self.model.llm_output_adapter.named_parameters() if p.requires_grad],
                #     "lr": lr * 10,
                # },
                # {
                #     "params": [p for n, p in self.model.llm_input_adapter.named_parameters() if p.requires_grad],
                #     "lr": lr * 10,
                # },
                ]

                self._optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=lr,
                    eps=1e-8,
                )
            else:
                self._optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=float(self.config.run_cfg.init_lr),
                    weight_decay=float(self.config.run_cfg.weight_decay),
                    betas=(0.9, beta2),
                )

        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            # max_epoch = self.config.run_cfg.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.run_cfg.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run_cfg.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)
            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)

            if iters_per_epoch is None:
                # try:
                #     iters_per_epoch = len(self.dataloaders['train'])
                # except (AttributeError, TypeError):
                #     iters_per_epoch = 10000
                if self.config.datasets_cfg.traj_align.filelabel == 'z0':
                    iters_per_epoch = 3178
                elif self.config.datasets_cfg.traj_align.filelabel == 'z1':
                    iters_per_epoch = int(624/self.config.run_cfg.batch_size_train)
                    # iters_per_epoch = 3994
                elif self.config.datasets_cfg.traj_align.filelabel == 'all':
                    iters_per_epoch = 7172
                elif self.config.datasets_cfg.traj_align.filelabel == 'waymo_interactive':
                    # iters_per_epoch = int(368454/self.config.run_cfg.batch_size_train)
                    iters_per_epoch = int(184227/self.config.run_cfg.batch_size_train) #/8 to save checkpoints quicker
                else:
                    raise 'Something wrong, cant define iters_per_epoch'
            else:
                iters_per_epoch = int(int(iters_per_epoch)/self.config.run_cfg.batch_size_train)

            if get_world_size()>1:
                iters_per_epoch = int(iters_per_epoch/get_world_size())

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            datasets = reorg_datasets_by_split(self.datasets)
            self.datasets = datasets
            # self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            if getattr(self.config.model_cfg, "mtr", False):
                from train import imotion_mtr_collator
                collate_fns[0] = imotion_mtr_collator
                dataloaders = self.create_loaders(
                    datasets=datasets,
                    num_workers=self.config.run_cfg.num_workers,
                    batch_sizes=batch_sizes,
                    is_trains=is_trains,
                    collate_fns=collate_fns,
                )
            else:
                dataloaders = self.create_loaders(
                    datasets=datasets,
                    num_workers=self.config.run_cfg.num_workers,
                    batch_sizes=batch_sizes,
                    is_trains=is_trains,
                    collate_fns=collate_fns[0],
                )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    # @property
    # def evaluate_only(self):
    #     """
    #     Set to True to skip training.
    #     """
    #     return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", False)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]

        return train_dataloader
    
    @property
    def valid_loader(self):
        valid_loader = self.dataloaders["valid"]

        return valid_loader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))
        # output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        output_dir = Path(self.config.run_cfg.output_dir)
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):

        start_time = time.time()
        best_agg_metric = 1000
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        if self.resume_ckpt_path is not None:
            # print(self.resume_ckpt_path)
            self._load_checkpoint(self.resume_ckpt_path, eval_only=self.config.model_cfg.get("eval_only", False), new_optimizer = self.config.model_cfg.get("new_optimizer", False))
            best_agg_metric = self.config.run_cfg.get("best_agg_metric", best_agg_metric)
            # best_epoch = self.config.run_cfg.get("best_epoch", 0)
            resume_epoch = self.start_epoch
            best_epoch = resume_epoch
            # resume_epoch = self.resume_ckpt_path.split('/')[-1].split('_')[1].split('.pth')[0]
            # if isinstance(resume_epoch, int):
            #     best_epoch = int(resume_epoch)
            # else:
            #     best_epoch = 90
            # if not self.config.run_cfg.evaluate_only:
            #     best_epoch = 0
            #     self.start_epoch = 0
        # if self.config.run_cfg.evaluate_only and self.resume_ckpt_path is not None:
        #     self._load_checkpoint(self.resume_ckpt_path)
        # if not self.config.run_cfg.evaluate_only and self.resume_ckpt_path is not None:
        #     self._load_checkpoint(self.resume_ckpt_path)
        if self.config.run_cfg.evaluate_only:
            self.model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    self.eval_epoch(split_name="valid", cur_epoch=best_epoch, save_samples=True, output_dir=self.output_dir)
        else:
            for cur_epoch in range(self.start_epoch, self.max_epoch):
                if self.config.run_cfg.evaluate_only:
                    val_log = self.eval_epoch(split_name="valid", cur_epoch=cur_epoch, save_samples=self.config.run_cfg.get('save_samples', False), output_dir=self.output_dir)
                    break
                # training phase
                if not self.config.run_cfg.evaluate_only:
                    logging.info("Start training")
                    train_stats = self.train_epoch(cur_epoch)
                    self.log_stats(split_name="train", stats=train_stats)
                    if is_main_process():
                        train_stats.update({"epoch": cur_epoch})
                        for k_train_stats, v_train_stats in train_stats.items():
                            train_stats[k_train_stats] = float(v_train_stats)
                        if wandb.run is not None:
                            wandb.log(train_stats)
                        if math.isnan(train_stats['loss']):
                            raise ValueError("Loss is NaN!")
                # self._save_checkpoint(cur_epoch, is_best=True)
                # evaluation phase
                if self.config.run_cfg.get('save_samples', False):
                    self.eval_epoch(split_name="valid", cur_epoch=cur_epoch, save_samples=True, output_dir=self.output_dir)
                if len(self.valid_splits) > 0 and self.config.run_cfg.evaluate:
                    if cur_epoch<self.config.run_cfg.get('evaluate_min_epoch', 0):
                    # if False:
                        if self.config.run_cfg.distributed:
                            dist.barrier()
                        continue
                    else:
                        if self.config.run_cfg.distributed:
                            dist.barrier()
                    logging.info("Evaluating on {}.".format("valid"))
                    val_log = self.eval_epoch(split_name="valid", cur_epoch=cur_epoch)
                    if is_main_process():
                        # if val_log['worked_count']==0:
                        if False:
                            if self.config.run_cfg.evaluate_only:
                                break
                            else:
                                print(">>> VALIDATION DIDNT WORK! <<<")
                                if self.config.run_cfg.distributed:
                                    dist.barrier()
                                continue
                        if "waymo_avg_minADE" in val_log.keys():
                            agg_metrics = val_log['waymo_avg_minADE'] # val_log['ADE_mean']
                        else:
                            agg_metrics = val_log['minADE'] # val_log['ADE_mean']
                        if float(agg_metrics) < float(best_agg_metric):
                            best_epoch, best_agg_metric = cur_epoch, agg_metrics
                            if self.config.run_cfg.save_model:
                                self._save_checkpoint(cur_epoch, is_best=True)
                        else:
                            if self.config.run_cfg.save_model:
                                self._save_checkpoint(cur_epoch, is_best=False)

                        if wandb.run is not None:
                            figures_ = val_log['figures']
                            wandb_images = [wandb.Image(value, caption=key) for key, value in val_log['figures'].items()]
                            val_log.pop('figures')
                            val_log.update({
                                "best_epoch": best_epoch,
                                "best_epoch_minADE": best_agg_metric,
                                "images":wandb_images,
                                "epoch":cur_epoch,
                            })
                            wandb.log(val_log)
                            # wandb.log({
                            #     "ADE_mean": val_log['ADE_mean'], 
                            #     "FDE_mean":val_log['FDE_mean'], 
                            #     "recon_ADE_mean":val_log['recon_ADE_mean'], 
                            #     "recon_FDE_mean":val_log['recon_FDE_mean'], 
                            #     "worked_count":val_log['worked_count'], 
                            #     "failed_count":val_log['failed_count'],
                            #     "failed_perc": val_log['failed_perc'],
                            #     "best_epoch": best_epoch,
                            #     "best_epoch_mean_ADE": best_agg_metric,
                            #     "images":wandb_images,
                            #     "epoch":cur_epoch,
                            #     "minADE":val_log['minADE'],
                            #     "minFDE":val_log['minFDE'],
                            #     })
                else:
                    # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                    if not self.config.run_cfg.evaluate_only:
                        if self.config.run_cfg.save_model:
                            self._save_checkpoint(cur_epoch, is_best=False)

                if self.config.run_cfg.evaluate_only:
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()

        # testing phase
        # test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        # self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False, save_samples=False, output_dir=''):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        self.model.eval()
        if self.config.run_cfg.get("eval_iters_per_epoch", -1) != -1:
            eval_iters_per_epoch = int(self.config.run_cfg.eval_iters_per_epoch/get_world_size()/self.config.run_cfg.batch_size_eval)
        else:
            # eval_iters_per_epoch = int(len(self.valid_loader)/get_world_size())
            eval_iters_per_epoch = int(len(self.valid_loader))
            # print(f">>>>>>>> ITERS PER GPU: {eval_iters_per_epoch}")
        if self.config.model_cfg.gameformer_enc:
            if self.config.run_cfg.evaluate_only:
                return self.task.evaluation_only_gf(
                    model=self.model, 
                    data_loader=self.valid_loader, 
                    epoch=cur_epoch, 
                    iters_per_epoch= eval_iters_per_epoch, 
                    cuda_enabled=self.cuda_enabled,
                    scaler=self.scaler,
                    log_freq=1,
                    # didnt_work_repeat=self.config.run_cfg.didnt_work_repeat,
                    didnt_work_repeat=0,
                    num_eval_figs=self.config.run_cfg.num_eval_figs,
                    rollouts=self.config.model_cfg.rollouts,
                    save_samples=save_samples,
                    output_dir=output_dir if save_samples else '',
                    contrastive_act=self.config.run_cfg.get('contrastive_act', False),
                    new_eval_mode = self.config.datasets_cfg.traj_align_valid.processor.valid.get('new_eval_mode', '')
                    )
            else:
                return self.task.evaluation_gf(
                    model=self.model, 
                    data_loader=self.valid_loader, 
                    epoch=cur_epoch, 
                    iters_per_epoch= eval_iters_per_epoch, 
                    cuda_enabled=self.cuda_enabled,
                    scaler=self.scaler,
                    log_freq=1,
                    # didnt_work_repeat=self.config.run_cfg.didnt_work_repeat,
                    didnt_work_repeat=0,
                    num_eval_figs=self.config.run_cfg.num_eval_figs,
                    rollouts=self.config.model_cfg.rollouts,
                    save_samples=save_samples,
                    output_dir=output_dir if save_samples else '',
                    )
        else:
            return self.task.evaluation(
                model=self.model, 
                data_loader=self.valid_loader, 
                epoch=cur_epoch, 
                iters_per_epoch= eval_iters_per_epoch, 
                cuda_enabled=self.cuda_enabled,
                scaler=self.scaler,
                log_freq=1,
                # didnt_work_repeat=self.config.run_cfg.didnt_work_repeat,
                didnt_work_repeat=0,
                num_eval_figs=self.config.run_cfg.num_eval_figs,
                rollouts=self.config.model_cfg.rollouts,
                save_samples=save_samples,
                output_dir=output_dir if save_samples else '',
                )
    
    
        # data_loader = self.dataloaders.get(split_name, None)        
        # assert data_loader, "data_loader for split {} is None.".format(split_name)

        # model = self.unwrap_dist_model(self.model)
        # if not skip_reload and cur_epoch == "best":
            # model = self._reload_best_model(model)
        # num_eval_batches = len(data_loader) # need to divide by number of gpus
        # num_eval_batches = int(self.config.run_cfg.num_eval_batches) if self.config.run_cfg.num_eval_batches else None
        # val_log = self.task.evaluation(self.model, data_loader, cur_epoch, num_eval_batches, num_figs=10)
        # return val_log
        # results = self.task.evaluation(model, data_loader, cur_epoch)

        # if results is not None:
        #     return self.task.after_evaluation(
        #         val_result=results,
        #         split_name=split_name,
        #         epoch=cur_epoch,
        #     )

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train, # not only if train
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    # if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        # sampler = sampler if is_train else None
                    sampler = sampler
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train, # not only if train
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                # loader = PrefetchLoader(loader)
                loader = IterLoader(loader, use_distributed=self.use_distributed)
                # if is_train:
                #     loader = IterLoader(loader, use_distributed=self.use_distributed)
                # else:
                #     loader = IterLoader(loader, use_distributed=False) # same, without distributed

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list):
                if len(dataset)==1:
                    dataset=dataset[0]
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = copy.deepcopy(model_no_ddp.state_dict())
        
        # Separate out llama_model's state_dict
        llama_model = model_no_ddp.llama_model
        # llama_model_clone = copy.deepcopy(llama_model).merge_and_unload()
        llama_model_clone = copy.deepcopy(llama_model)
        # llama_model_clone = llama_model.merge_and_unload()
        # llama_model_clone = llama_model
        llama_tokenizer = copy.deepcopy(model_no_ddp.llama_tokenizer)
        # Remove llama_model's parameters from the main state_dict
        for key in list(state_dict.keys()):
            if key.startswith("llama_model."):
                del state_dict[key]
            

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }

        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else "last"),
        )
        llama_save_dir = os.path.join(
            self.output_dir,
            "checkpoint_{}_llama_model".format("best" if is_best else "last"),
        )

        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
        # if cur_epoch % 5 == 0:
        save_to = os.path.join(
            self.output_dir,
            f"checkpoint_{cur_epoch}.pth"
        )
        torch.save(save_obj, save_to)
        
        llama_model_clone.save_pretrained(llama_save_dir, save_embedding_layers=True, is_main_process=True)
        llama_tokenizer.save_pretrained(llama_save_dir, is_main_process=True)
        llama_save_dir = os.path.join(
            self.output_dir,
            f"checkpoint_{cur_epoch}_llama_model"
        )
        llama_model_clone.save_pretrained(llama_save_dir, save_embedding_layers=True, is_main_process=True)
        llama_tokenizer.save_pretrained(llama_save_dir, is_main_process=True)
        

    # @main_process
    # def _save_checkpoint(self, cur_epoch, is_best=False):
    #     """
    #     Save the checkpoint at the current epoch.
    #     """
    #     model_no_ddp = self.unwrap_dist_model(self.model)
    #     param_grad_dic = {
    #         k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
    #     }
    #     state_dict = model_no_ddp.state_dict()
    #     # for k in list(state_dict.keys()):
    #     #     if k in param_grad_dic.keys() and not param_grad_dic[k]:
    #     #         # delete parameters that do not require gradient
    #     #         del state_dict[k]
    #     save_obj = {
    #         "model": state_dict,
    #         "optimizer": self.optimizer.state_dict(),
    #         "config": self.config.to_dict(),
    #         "scaler": self.scaler.state_dict() if self.scaler else None,
    #         "epoch": cur_epoch,
    #     }
    #     # print("##########################################")
    #     # print("##########################################")
    #     # print("##########################################")
    #     # print("##########################################")
    #     # print("OUTPUT DIR")
    #     # print(self.output_dir)
    #     # # print(str(os.path.join(wandb.run.dir,"checkpoint_"+str(wandb.run.id)+
    #     # #         "_{}.pth".format("best" if is_best else cur_epoch))))
    #     # print("##########################################")
    #     # print("##########################################")
    #     # print("##########################################")
    #     # print("##########################################")
    #     if self.config.run_cfg.wandb_log:
    #         save_to = os.path.join(
    #             self.output_dir,
    #             "checkpoint_{}.pth".format("best" if is_best else "last"),
    #         )
    #         # save_to = os.path.join(wandb.run.dir,"checkpoint_"+str(wandb.run.id)+
    #         #     "_{}.pth".format("best" if is_best else cur_epoch),
    #         # )
    #     else:
    #         save_to = os.path.join(
    #             self.output_dir,
    #             "checkpoint_{}.pth".format("best" if is_best else "last"),
    #         )
    #     logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    #     torch.save(save_obj, save_to)
    #     # if cur_epoch % 5 == 0:
    #     save_to = os.path.join(
    #         self.output_dir,
    #         f"checkpoint_{cur_epoch}.pth"
    #     )
    #     torch.save(save_obj, save_to)
    #     # if cur_epoch%

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename, eval_only=False, new_optimizer=False):
        """
        Resume from a checkpoint.
        """
        self.eval_only=eval_only
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError(f"checkpoint url or path is invalid {url_or_filename}")

        state_dict = checkpoint["model"]
        if not eval_only:
            msg = self.unwrap_dist_model(self.model).load_state_dict(state_dict,strict=False) # gameformer weights not included here, at least in the experiment where gameformer is freezed
            if not new_optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            msg = self.model.load_state_dict(state_dict,strict=False) # gameformer weights not included here, at least in the experiment where gameformer is freezed
            # msg = self.model.load_state_dict(state_dict,strict=True)
        # print(msg)
        print(f"> Number of missing keys: {len(msg.missing_keys)}")
        print('> NOTE: llama model could be already loaded in the model initialization, and no need to load here')
        print(f"> NOTE: If out of {len(msg.missing_keys)} missing keys == {len([k for k in msg.missing_keys if 'llama_model' in k])} llama_model missing keys then you are ok if the model was already loaded")
        print(f"> Number of unexpected keys: {len(msg.unexpected_keys)}")
        # for name, module in self.model.named_modules():
        #     if 'base' in name:
        #         print(f"Module Name: {name}")
            # print(f"Module Object: {module}\n")

        # # To get all modules without names
        # for module in self.model.modules():
        #     print(f"Module Object: {module}\n")
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        
        self.start_epoch = checkpoint["epoch"] + 1 if not new_optimizer else 0
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir,"log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
