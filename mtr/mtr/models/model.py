# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .context_encoder import build_context_encoder
from .motion_decoder import build_motion_decoder

def print_device(name, tensor):
    if isinstance(tensor, torch.Tensor):
        print(f"{name}.device = {tensor.device}")
    elif isinstance(tensor, dict):
        for key, value in tensor.items():
            print_device(f"{name}[{key}]", value)
    elif isinstance(tensor, list) or isinstance(tensor, tuple):
        for idx, value in enumerate(tensor):
            print_device(f"{name}[{idx}]", value)

class MotionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER)
        self.motion_decoder = build_motion_decoder(
            in_channels=self.context_encoder.num_out_channels,
            config=self.model_cfg.MOTION_DECODER
        )
    
    def imotion_forward_step1(self, batch_dict):
        batch_dict = self.context_encoder(batch_dict)
        batch_dict, other_features = self.motion_decoder.imotion_forward_step1_(batch_dict)
        # other_features: (center_objects_feature, obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos)
        return batch_dict, other_features

    
    def imotion_forward_step2(self, batch_dict, other_features, act=None, reduction='mean'):
        # center_objects_feature, obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos = other_features[0], other_features[1], other_features[2], other_features[3], other_features[4], other_features[5], other_features[6]
        center_objects_feature, obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos = other_features['center_objects_feature'], other_features['obj_feature'], other_features['obj_mask'], other_features['obj_pos'], other_features['map_feature'], other_features['map_mask'], other_features['map_pos']
        center_objects_feature = center_objects_feature.to(obj_feature.dtype)
        act = act.to(obj_feature.dtype)
        batch_dict = self.motion_decoder.imotion_forward_step2_(batch_dict, center_objects_feature, obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos, act=act)
        if self.training:
            loss, tb_dict, disp_dict = self.get_loss(reduction=reduction)
            if reduction!='none':
                tb_dict.update({'loss': loss.item()})
                disp_dict.update({'loss': loss.item()})
            else:
                tb_dict.update({'loss': loss})
                disp_dict.update({'loss': loss})
            return loss, tb_dict, disp_dict

        return batch_dict

    def forward(self, batch_dict):
        batch_dict = self.context_encoder(batch_dict)
        batch_dict = self.motion_decoder(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_loss()

            tb_dict.update({'loss': loss.item()})
            disp_dict.update({'loss': loss.item()})
            return loss, tb_dict, disp_dict

        return batch_dict

    def get_loss(self, reduction='mean'):
        loss, tb_dict, disp_dict = self.motion_decoder.get_loss(reduction=reduction)

        return loss, tb_dict, disp_dict

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        # logger.info('==> Done')
        logger.info('==> Done (loaded %d/%d)' % (len(checkpoint['model_state']), len(checkpoint['model_state'])))

        return it, epoch

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        
        if logger is not None:
            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        else:
            print('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            if logger is not None:
                logger.info('==> Checkpoint trained from version: %s' % version)
            else:
                print('==> Checkpoint trained from version: %s' % version)

        if logger is not None:
            logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        else:
            print('==> Checkpoint trained from version: %s' % version)
        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)
        if logger is not None:
            logger.info(f'Missing keys: {missing_keys}')
            logger.info(f'The number of missing keys: {len(missing_keys)}')
            logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
            logger.info('==> Done (total keys %d)' % (len(model_state)))
        else:
            print(f'Missing keys: {missing_keys}')
            print(f'The number of missing keys: {len(missing_keys)}')
            print(f'The number of unexpected keys: {len(unexpected_keys)}')
            print('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch


