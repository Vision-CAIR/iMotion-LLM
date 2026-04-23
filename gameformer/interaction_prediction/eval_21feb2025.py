# torchrun --nproc-per-node 4 ~/project/gameformer_p/interaction_prediction/train.py --wandb --distributed --name gameformer_temp1
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --wandb --distributed --name gameformer_temp1 --batch_size 8
# torchrun --nproc-per-node 2 train.py --distributed --workers 8 --batch_size 64 --train_set <internal_user_root>/waymo_dataset/training_interactive_gameformer_10hz_agentorder --valid_set <internal_user_root>/waymo_dataset/validation_interactive_gameformer_10hz --subsample False --future_len 80
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --act --act_dec --save_model <internal_user_root>/gameformer_models/gf_4mar_lvl0_act/ --name gf_4mar_lvl0_act --load_dir '' --wandb --distributed --workers 8 --batch_size 64 --level 0 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20nvidia-smi
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --save_model <internal_user_root>/gameformer_models/gf_4mar_lvl6_valid/ --name gf_4mar_lvl6_valid --load_dir '' --wandb --distributed --workers 8 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20nvidia-smi
import torch
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('<legacy_repo_root>')
sys.path.append('...')
import csv
import argparse
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from gameformer.interaction_prediction.eval import validation_epoch, validation_epoch_pkl

import time
from gameformer.model.GameFormer import GameFormer, GameFormer_
# from model.GameFormer import GameFormer_2
from gameformer.utils.inter_pred_utils import *
import wandb

from gameformer.interaction_prediction.dist_utils import get_rank, init_distributed_mode
from gameformer.interaction_prediction.logger import setup_logger
from gameformer.interaction_prediction.logger import MetricLogger, SmoothedValue
from tqdm import tqdm
# from multimodal_viz import *

# from exctract_instruct import *

def unwrap_dist_model(model, distributed=False):
        if distributed:
            return model.module
        else:
            return model

def convert_cfg_to_wandb_cfg(input_dict):
    converted_dict = {}
    for key, value in input_dict.items():
        converted_dict[key] = {'value': value}
    return converted_dict

# define model training epoch
def training_epoch(train_data, model, optimizer, epoch, act=False, two_agent_act=False):
    epoch_loss = []
    model.train()
    current = 0
    start_time = time.time()
    size = len(train_data)
    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0
    # args.local_rank = get_rank()
    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter("ADE", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter("FDE", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter("loss_m", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if not hasattr(train_data, "__next__"):
        train_data = iter(train_data)
    # for i in metric_logger.log_every(range(len(train_data)), 100, "Train: data epoch: [{}]".format(epoch)):
    max_center_lanes_length, max_num_center_lanes, max_boundaries_length, max_num_boundaries, max_crosswalks_length, max_num_crosswalks, max_speed_bumps_length, max_num_speed_bumps, max_traffic_lights_length, max_stop_signs_length = [],[],[],[],[],[],[],[],[],[]
    for i in tqdm(range(len(train_data))):
        # prepare data
        
        batch = next(train_data)
        if args.full_map:
            inputs = {
                'ego_state': batch[0].to(args.local_rank),
                'neighbors_state': batch[1].to(args.local_rank),
                'map_lanes': batch[2].to(args.local_rank),
                'map_crosswalks': batch[3].to(args.local_rank),
                'act': batch[7].to(args.local_rank),
                'additional_map_lanes': batch[8].to(args.local_rank),
                'additional_map_crosswalks': batch[9].to(args.local_rank),
                'additional_boundaries':batch[10].to(args.local_rank),
                'traffic_lights':batch[11].to(args.local_rank),
                'stop_signs':batch[12].to(args.local_rank),
                'speed_bumps':batch[13].to(args.local_rank),
            }   
            ego_future = batch[4].to(args.local_rank)
            neighbor_future = batch[5].to(args.local_rank)
        else:
            inputs = {
                'ego_state': batch[0].to(args.local_rank),
                'neighbors_state': batch[1].to(args.local_rank),
                'map_lanes': batch[2].to(args.local_rank),
                'map_crosswalks': batch[3].to(args.local_rank),
                'act':batch[7].to(args.local_rank),
            } if torch.cuda.is_available() else {
                'ego_state': batch[0],
                'neighbors_state': batch[1],
                'map_lanes': batch[2],
                'map_crosswalks': batch[3],
                'act':batch[7],
            }
            ego_future = batch[4].to(args.local_rank) if torch.cuda.is_available() else batch[4]
            neighbor_future = batch[5].to(args.local_rank) if torch.cuda.is_available() else batch[5]
        # zero gradients for every batch
        optimizer.zero_grad()
        # query the model
        outputs = model(inputs)
        loss,future,_ = level_k_loss(outputs, ego_future, neighbor_future, args.level, gmm=args.gmm, subsample=args.subsample)
        # back-propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # compute metrics
        current += args.batch_size
        epoch_loss.append(loss.item())
        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    logging.info("Averaged stats: " + str(metric_logger.global_avg()))
    de_metrics = {}
    # for k, meter in metric_logger.meters.items():
    #     de_metrics['train_'+k] = float("{:.8f}".format(meter.global_avg))
    
    return epoch_loss, de_metrics

# Define model training process
def main():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.distributed:
        init_distributed_mode(args)
    setup_logger()

    wandb_cfg = {**convert_cfg_to_wandb_cfg(args.__dict__)}
    if args.wandb and get_rank()==0:
        wandb.login()
        wandb.init(project="trajgpt", name=args.name, config=wandb_cfg)

    # if args.save_model and get_rank()==0:
    if get_rank()==0:
        # log_path = f"{args.save_model}/training_log/{args.name}/"
        log_path = args.save_path if args.save_path[-1] == '/' else f"{args.save_path}/"
    # os.makedirs(log_path, exist_ok=True)
    # if args.distributed:
        os.makedirs(log_path, exist_ok=True)
        if args.new_eval or (args.new_eval_mode is not None and args.new_eval_mode!=''):
            os.makedirs(log_path+args.new_eval_mode+'results/', exist_ok=True)
        else:
            os.makedirs(log_path+'results/', exist_ok=True)
        # initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    

    set_seed(args.seed)
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
    # local_rank = args.local_rank
    # torch.cuda.set_device(local_rank)
    # if args.distributed:
        # dist.init_process_group(backend='nccl')
        # logging.info("Use device: {}".format(local_rank))

    model = GameFormer_(
                modalities=args.modalities,
                encoder_layers=args.encoder_layers,
                decoder_levels=args.level,
                future_len=args.future_len, 
                neighbors_to_predict=args.neighbors_to_predict,
                act = args.act,
                act_dec = args.act_dec,
                full_map=args.full_map,
                ego_act_only=args.ego_act_only,
                simple_model=args.simple_model,
                act_kv=args.act_kv,
                num_act_classes=args.num_act_classes
                )
    # if args.decomposed_gf:
    #     model = GameFormer_(
    #             modalities=args.modalities,
    #             encoder_layers=args.encoder_layers,
    #             decoder_levels=args.level,
    #             future_len=args.future_len, 
    #             neighbors_to_predict=args.neighbors_to_predict,
    #             act = args.act,
    #             act_dec = args.act_dec,
    #             shared_act = args.shared_act,
    #             ego_act_only = args.ego_act_only,
    #             no_fuse_act = args.no_fuse_act
    #             )
    # else:
    #     model = GameFormer(
    #                 modalities=args.modalities,
    #                 encoder_layers=args.encoder_layers,
    #                 decoder_levels=args.level,
    #                 future_len=args.future_len, 
    #                 neighbors_to_predict=args.neighbors_to_predict,
    #                 act = args.act,
    #                 act_dec = args.act_dec
    #                 )

    if args.load_dir != '':
        model_path = args.load_dir
        model_ckpts = torch.load(model_path, map_location='cpu')
        msg = model.load_state_dict(model_ckpts['model_states'])
        print(msg)
    # model = model.to(local_rank)
    
    if args.distributed:
        model = model.to(args.gpu)
        # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model = DDP(model, device_ids=[args.gpu])
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

    # define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
                                            optimizer, 
                                            milestones=[int(item) for item in args.lr_steps.split(',')],  #[20, 22, 24, 26, 28]
                                            gamma=0.5,
                                            verbose=True)
    
    
    # load ckpts:
    curr_ep = 0
    if args.load_dir != '':
        optimizer.load_state_dict(model_ckpts['optim_states'])
        curr_ep = model_ckpts['current_ep']
        scheduler.step(curr_ep)
    # print("DONE")
    
    # datasets:
    if not args.eval_only:
        if args.act:
            raise 'not implemented!'
        train_dataset = DrivingData(args.train_set+'/*', act=args.act_dec or args.act_kv, full_map=args.full_map, ego_act_only=args.ego_act_only, nuplan=args.nuplan, random_drop_act=not args.no_random_drop_act, files_with_act_only=args.files_with_act_only, num_classes=args.num_act_classes)
        print(f'found {len(train_dataset)} training data')
        training_size = len(train_dataset)
    valid_dataset = DrivingData(args.valid_set+'/*', act=args.act_dec or args.act_kv, full_map=args.full_map, contrastive=args.contrastive, positive_notgt=args.positive, new_eval=args.new_eval, new_eval_mode=args.new_eval_mode, nuplan=args.nuplan, num_classes=args.num_act_classes)
    print(f'found {len(valid_dataset)} validation data')
    valid_size = len(valid_dataset)

    if args.distributed:
        if dist.get_rank() == 0:
            logging.info(f'Length train: {training_size if not args.eval_only else 0}; Valid: {valid_size}')
    else:
        logging.info(f'Length train: {training_size if not args.eval_only else 0}; Valid: {valid_size}')
        # print(f'Length train: {training_size}; Valid: {valid_size}')
    if args.distributed:
         if not args.eval_only:
            train_sampler = DistributedSampler(train_dataset)
        # valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    if not args.eval_only:
        train_data = DataLoader(
            train_dataset, batch_size=args.batch_size, 
            sampler=train_sampler if args.distributed else None, num_workers=args.workers,
            )
    
    if get_rank()==0:
        valid_data = DataLoader(
            valid_dataset, batch_size= args.batch_size,
            sampler= None, num_workers=args.workers, shuffle=False,
            )

    #start training:
    epochs = args.training_epochs

    for epoch in range(epochs):
        if epoch<=curr_ep and not args.eval_only:
            continue
        if args.eval_only:
            epoch = curr_ep
        # if args.distributed and epoch!=0:
        #         model = model.to(args.gpu)
        #         model = DDP(model, device_ids=[args.gpu])
        if args.distributed:
            if dist.get_rank() == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}")
        else:
            # print((f"Epoch {epoch+1}/{epochs}"))
            logging.info(f"Epoch {epoch+1}/{epochs}")
        # if epoch<=curr_ep and epoch!=0:
        #     continue
        if args.distributed:
            train_data.sampler.set_epoch(epoch)
            # valid_data.sampler.set_epoch(epoch)

        if not args.eval_only:
            train_loss, train_de_metrics = training_epoch(train_data, model, optimizer, epoch, args.act or args.act_dec)
            # adjust learning rate
            scheduler.step()
            if get_rank()==0:
                # if args.distributed:
                #     model = model.to(torch.device("cuda:0"))
                # with torch.no_grad():
                #     results_dict = validation_epoch(valid_data, model, epoch, act= args.act or args.act_dec, two_agent_act=False, save_dir=log_path+'results/', num_classes=8, args=args)
                # Synchronize all processes
                
            # save to training log
             #   
                log = {
                    'epoch': epoch+1, 
                    'train_loss': np.mean(train_loss), 
                    # 'val_loss': np.mean(val_loss),
                    'lr': optimizer.param_groups[0]['lr'],
                    # 'val_minADE': results_dict['minADE'],
                    # 'val_minFDE': results_dict['minFDE'],
                    # 'val_rgif': results_dict['RGIF class weighted Avg'],
                    }
                log.update(train_de_metrics)
            # dist.barrier()
        else:
            model.eval()
            # valid_metrics, val_loss, valid_de_metrics, output_figs = validation_epoch(valid_data, model, epoch, act= args.act or args.act_dec, two_agent_act=False, save_dir=log_path+'results/', num_classes=8, args=args)
            with torch.no_grad():
                eval_save_dir = log_path+args.new_eval_mode+'results' if args.new_eval_mode!='none' else log_path+'results'
                results_dict = validation_epoch_pkl(valid_data, model, epoch, act= args.act or args.act_dec, two_agent_act=False, save_dir=eval_save_dir
                , num_classes=args.num_act_classes, args=args)
                # results_dict = validation_epoch(valid_data, model, epoch, act= args.act or args.act_dec, two_agent_act=False, save_dir=log_path+'results/', num_classes=8, args=args)
            # for k,v in results_dict.items():
            #     print(f"{k}: {v:.4f}")
            # save to training log
                if len(results_dict)!=0:
                    log = {
                        'epoch': epoch+1,
                        'val_minADE': results_dict['minADE'],
                        'val_minFDE': results_dict['minFDE'],
                        'val_rgif': results_dict['RGIF class weighted Avg'],
                        }
                else:
                    log = {}
            break
        if get_rank()==0:
            print(log)
        # log.update(valid_metrics)
        # log.update(valid_de_metrics)
        # print(log)
        # if epoch == epochs-1:
        #     wandb_images = [wandb.Image(value, caption=key) for key, value in output_figs.items()]        
        #     log.update({"images":wandb_images})
        if args.distributed:
            if dist.get_rank() == 0:
                if wandb.run is not None:
                    wandb.log(log)
                if args.save_model:
                    # log & save
                    if epoch == 0:
                        with open(log_path + f'train_log.csv', 'w') as csv_file: 
                            writer = csv.writer(csv_file) 
                            writer.writerow(log.keys())
                            writer.writerow(log.values())
                    else:
                        with open(log_path + f'train_log.csv', 'a') as csv_file: 
                            writer = csv.writer(csv_file)
                            writer.writerow(log.values())
                    
                    model_no_ddp = unwrap_dist_model(model, args.distributed)
                    state_dict = model_no_ddp.state_dict()
                    # param_grad_dic = {
                    #     k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
                    # }
                    # for k in list(state_dict.keys()):
                    #      if k in param_grad_dic.keys() and not param_grad_dic[k]:
                    #         # delete parameters that do not require gradient
                    #         del state_dict[k]
                    save_state = {
                        'optim_states' : optimizer.state_dict(),
                        'model_states' :state_dict,
                        'current_ep': epoch
                    }

                    # save_state = {
                    #     'optim_states' : optimizer.state_dict(),
                    #     'model_states' :model.state_dict(),
                    #     'current_ep': epoch
                    # }
                    torch.save(save_state, log_path + f'epochs_{epoch}.pth')
                    # torch.save(save_state, log_path + f'epochs_last.pth')
        else:
            if wandb.run is not None:
                wandb.log(log)
            if args.save_model:
                save_state = {
                        'optim_states' : optimizer.state_dict(),
                        'model_states' :model.state_dict(),
                        'current_ep': epoch
                    }
                torch.save(save_state, log_path + f'epochs_{epoch}.pth')
                # torch.save(save_state, log_path + f'epochs_last.pth')

        # # adjust learning rate
        # scheduler.step()
    if not args.eval_only:
        if get_rank()==0:
            with torch.no_grad():
                results_dict = validation_epoch(valid_data, model, epoch, act= args.act or args.act_dec, two_agent_act=False, save_dir=log_path+'results/', num_classes=args.num_act_classes, args=args)
                log = {
                        'epoch': epoch+1, 
                        'train_loss': np.mean(train_loss), 
                        # 'val_loss': np.mean(val_loss),
                        'lr': optimizer.param_groups[0]['lr'],
                        'val_minADE': results_dict['minADE'],
                        'val_minFDE': results_dict['minFDE'],
                        'val_rgif': results_dict['RGIF class weighted Avg'],
                        }
                if wandb.run is not None:
                    wandb.log(log)


if __name__ == "__main__":
    # default train parameters: 
    # GPUs = 4
    # batch_size=16 \
    # workers=8 \
    parser = argparse.ArgumentParser(description='Interaction Prediction Training')
    parser.add_argument("--model", default='gameformer')
    # parser.add_argument("--local_rank", type=int, default=0)
    # training
    parser.add_argument("--batch_size", type=int, help='training batch sizes', default=64)
    # parser.add_argument("--batch_size", type=int, help='training batch sizes', default=128)
    parser.add_argument("--training_epochs", type=int, help='training epochs', default=30)
    parser.add_argument("--learning_rate", type=float, help='training learning rates', default=1e-4)
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    # data & loggings
    parser.add_argument("--workers", type=int, default=4, help="number of workers used for dataloader")
    # model
    parser.add_argument("--level", type=int, help='decoder reasoning levels (K)', default=3) # default: 6
    # parser.add_argument("--neighbors_to_predict", type=int, help='neighbors to predict, 1 for Waymo Joint Prediction', default=1)
    parser.add_argument("--neighbors_to_predict", type=int, help='neighbors to predict, 1 for Waymo Joint Prediction', default=0)
    parser.add_argument("--modalities", type=int, help='joint num of modalities', default=6)
    parser.add_argument("--future_len", type=int, help='prediction horizons', default=80) # 16 or 80
    parser.add_argument("--encoder_layers", type=int, help='encoder layers', default=6)
    parser.add_argument("--gmm", action="store_true", help='gmm or imitation loss, default gmm=True', default=True)
    parser.add_argument("--distributed", action="store_true", help='distributed mode', default=False)
    parser.add_argument("--wandb", action="store_true", help='distributed mode', default=False)
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="gameformer_temp")
    parser.add_argument('--subsample', type=bool, help='2Hz if True, 10Hz if false', default=False)
    parser.add_argument('--lr_steps', type=str, help='', default='15,18,21,24,27')
    parser.add_argument("--act", action="store_true", help='act', default=False)
    parser.add_argument("--viz", action="store_true", help='visualize', default=False)
    # parser.add_argument('--train_set', type=str, help='path to train data', default='<internal_dataset_root>/waymo/gameformer/training_15may_fullmap_fulldata')
    # parser.add_argument('--train_set', type=str, help='path to train data', default='<internal_dataset_root>/waymo/gameformer/validation_3jul') # training_small_1jul
    # parser.add_argument('--train_set', type=str, help='path to train data', default='<internal_dataset_root>/waymo/gameformer/training_21aug')
    parser.add_argument('--train_set', type=str, help='path to train data', default='<internal_dataset_root>/waymo/gameformer/training_28nov')
    # parser.add_argument('--train_set', type=str, help='path to train data', default='<internal_dataset_root>/waymo/gameformer/training_24nov')
    
    # parser.add_argument('--train_set', type=str, help='path to train data', default='<internal_dataset_root>/waymo/gameformer/validation_30sep')
    
    # parser.add_argument('--train_set', type=str, help='path to train data', default='<internal_dataset_root>/waymo/gameformer/validation_1jul')
    # parser.add_argument('--valid_set', type=str, help='path to train data', default='<internal_dataset_root>/waymo/gameformer/validation_30sep')
    parser.add_argument('--valid_set', type=str, help='path to train data', default='<internal_dataset_root>/waymo/gameformer/feb16_2025/validation')
    
    
    parser.add_argument("--eval_only", action="store_true", help='', default=False)
    # parser.add_argument("--eval_only", help='', default=True)
    parser.add_argument("--act_dec", action="store_true", help='act', default=False)
    parser.add_argument("--act_kv", action="store_true", help='act', default=False)
    # parser.add_argument("--act_dec", action="store_true", help='act', default=True)
    
    parser.add_argument("--full_map", action="store_true", help='', default=False)
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/cgf_2jul_smalldata/epochs_29.pth')
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/cgf_7jul_fulldata/epochs_17.pth')
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_user_root>/gameformer_models/gf_7may_act_smalldata/epochs_last.pth')
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/gf_23aug/epochs_29.pth')
    # parser.add_argument('--save_path', type=str, help='', default='<internal_model_root>/gameformer/gf_23aug/validation_23aug')
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/cgf_23aug/epochs_29.pth')
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/gf_23aug/epochs_29.pth')
    # parser.add_argument('--save_path', type=str, help='', default='<internal_model_root>/gameformer/gf_23aug/results30sep')
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/cgf_23aug/epochs_29.pth')
    # parser.add_argument('--save_path', type=str, help='', default='<internal_model_root>/gameformer/cgf_23aug/results_fullData_gt')
    
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/gf_23aug/epochs_29.pth')
    # parser.add_argument('--save_path', type=str, help='', default='<internal_model_root>/gameformer/gf_23aug/results_fullData_gt')
    parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='')
    parser.add_argument("--save_model", type=str, help='save model directory, not saved if not provided', default=False)
    # <internal_model_root>/gameformer/temp
    parser.add_argument('--save_path', type=str, help='', default='')
    
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/gf_7may_act_smalldata/epochs_last.pth')
    
    # parser.add_argument('--save_path', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/gf_7may_fullmap_smalldata/wPlausbility_val_gt_s1')
    # parser.add_argument("--contrastive", action="store_true", help='', default=False)
    # parser.add_argument("--positive", action="store_true", help='', default=False)
    
    
    # parser.add_argument('--save_path', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/gf_7may_act_smalldata/wPlausbility_val_c_s1')
    parser.add_argument("--contrastive", action="store_true", help='', default=False)
    parser.add_argument("--positive", action="store_true", help='', default=False)
    
    parser.add_argument("--two_agents", action="store_true", help='', default=False)
    parser.add_argument("--new_eval", action="store_true", default=False)
    # parser.add_argument("--new_eval", default=True)
    parser.add_argument("--new_eval_mode", default='gt1')
    # parser.add_argument("--new_eval_mode", default='')
    parser.add_argument("--simple_model", action="store_true", default=False)
    parser.add_argument("--nuplan", action="store_true", default=False) # WOMD or nuplan
    parser.add_argument("--no_random_drop_act", action="store_true", default=False)
    parser.add_argument("--files_with_act_only", action="store_true", default=False)
    parser.add_argument("--num_act_classes", type=int, default=5)
    
    

    # parser.add_argument("--two_agents", default=False)

    

    # parser.add_argument('--save_path', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_model_root>/gameformer/gf_7may_fullmap_act_smalldata/wPlausbility_val_p_s1')
    # parser.add_argument("--contrastive", action="store_true", help='', default=False)
    # parser.add_argument("--positive", action="store_true", help='', default=True)
    
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_user_root>/gameformer_models/gf_act_5mar_final_300k_train_03/epochs_last.pth')
    # parser.add_argument('--load_dir', type=str, help='name to load ckpts from log path (e.g. epochs_0.pth)', default='<internal_user_root>/gameformer_models/gf_7may_fullmap_smalldata/epochs_last.pth')
    # parser.add_argument("--act_dec", action="store_true", help='act', default=True)
    # parser.add_argument("--eval_only", action="store_true", help='', default=True)
    
    
    
    args = parser.parse_args()
    args.ego_act_only = not args.two_agents
    print(f"Controlling {1 if args.ego_act_only else 2} Agents")
    if args.save_path=='':
        if args.save_model:
            args.save_path = args.save_model
        else:
            args.save_path = '<legacy_repo_root>'
        
        if args.eval_only:
            args.save_path = '/'.join(args.load_dir.split('/')[:-1])+'/'+args.valid_set.split('/')[-1]
            if 'split' in args.save_path:
                args.save_path = '/'.join(args.save_path.split('/')[:-1])+'/'+args.valid_set.split('/')[-2]
                args.save_path = args.save_path[:-1]+args.valid_set.split('/')[-1][-1]
    # print(args.gmm)
    # return
    print(">>> SAVE PATH:")
    print(args.save_path)

    main()


# torchrun --nproc-per-node 4 --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --wandb --distributed --workers 8 --batch_size 64 --level 6 --modalities 6 --encoder_layers 6 --future_len 80 --learning_rate 1e-4 --gmm False --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --name temp --load_dir '' --train_set <internal_user_root>/waymo_dataset/validation_interactive_gameformer_10hz --valid_set <internal_user_root>/waymo_dataset/validation_interactive_gameformer_10hz --save_model <internal_user_root>/gameformer_models/temp/

# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --act_dec --save_model <internal_user_root>/gameformer_models/gf_4mar_lvl6_actdec/ --name gf_4mar_lvl6_actdec --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '10,15,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/training_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --name temp --load_dir '' --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '10,15,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/training_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20


# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --shared_act --act --act_dec --decomposed_gf --save_model <internal_user_root>/gameformer_models/gf_5mar_shared_act/ --name gf_5mar_shared_act --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20


# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --two_agent_act --act --act_dec --decomposed_gf --save_model <internal_user_root>/gameformer_models/gf_5mar_2agentact/ --name gf_5mar_2agentact --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20

# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --two_agent_act --act --act_dec --decomposed_gf --load_dir '' --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20

# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --two_agent_act --act --decomposed_gf --save_model <internal_user_root>/gameformer_models/gf_5mar_2a_noDec/ --name gf_5mar_2a_noDec --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --two_agent_act --act --act_dec --shared_act --decomposed_gf --save_model <internal_user_root>/gameformer_models/gf_5mar_2a_shared/ --name gf_5mar_2a_shared --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --ego_act_only --act --act_dec --decomposed_gf --save_model <internal_user_root>/gameformer_models/gf_5mar_egonly/ --name gf_5mar_egonly --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --ego_act_only --act --decomposed_gf --save_model <internal_user_root>/gameformer_models/gf_5mar_egonly_nodec/ --name gf_5mar_egonly_nodec --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --ego_act_only --act --act_dec --decomposed_gf --shared_act --save_model <internal_user_root>/gameformer_models/gf_5mar_egonly_sh/ --name gf_5mar_egonly_sh --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20
# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --act --decomposed_gf --no_fuse_act --save_model <internal_user_root>/gameformer_models/gf_5mar_nofuse/ --name gf_5mar_nofuse --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/validation_interactive_original_20 --valid_set <internal_waymo_dataset_root>/validation_interactive_original_20

# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train.py --act_dec --act --decomposed_gf --save_model <internal_user_root>/gameformer_models/gf_act_5mar_final_80k/ --name gf_act_5mar_final_80k --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/training_interactive_5mar_80k --valid_set <internal_waymo_dataset_root>/validation_interactive_5mar_5k


# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train_03.py --act_dec --act --save_model <internal_user_root>/gameformer_models/gf_act_5mar_final_80k_train_03/ --name gf_act_5mar_final_80k_train_03 --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/training_interactive_5mar_80k --valid_set <internal_waymo_dataset_root>/validation_interactive_5mar_5k

# torchrun --nproc-per-node 4 <legacy_gameformer_repo>/interaction_prediction/train_03.py --act_dec --act --save_model <internal_user_root>/gameformer_models/gf_act_5mar_final_300k_train_03/ --name gf_act_5mar_final_300k_train_03 --load_dir '' --wandb --distributed --workers 4 --batch_size 64 --level 6 --modalities 6 --future_len 80 --learning_rate 1e-4 --subsample False --lr_steps '18,21,24,27,30' --training_epochs 30 --train_set <internal_waymo_dataset_root>/training_interactive_5mar --valid_set <internal_waymo_dataset_root>/validation_interactive_5mar_5k
# python <legacy_repo_root>/gameformer/interaction_prediction/train.py --workers 2 --batch_size 4 --level 6 --modalities 6 --future_len 80 --learning_rate 5e-5 --lr_steps '15,18,21,24,27,30' --training_epochs 30 --train_set <internal_dataset_root>/waymo/gameformer/training_full_3jul --save_model <internal_model_root>/gameformer/cgf_7jul_fulldata/ --name cgf_7jul_fulldata_res1 --load_dir <internal_model_root>/gameformer/cgf_7jul_fulldata/epochs_5.pth --wandb --act_dec