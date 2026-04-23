import torch
from .modules import *
import time

class Encoder_01(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6, act=False, full_map=False, simple_model=False, act_kv=False):
        super(Encoder_01, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self._neighbors = neighbors_to_predict
        self.act_kv = act_kv
        if self.act_kv:
            num_act_classes = 8
            self.act_embedding = nn.Embedding(num_act_classes, 256)
        self.simple_model = simple_model
        if not simple_model:
            self.agent_encoder = AgentEncoder()
            self.ego_encoder = AgentEncoder()
            self.lane_encoder = LaneEncoder()
            self.crosswalk_encoder = CrosswalkEncoder()
        else:
            self.agent_encoder = SimpleAgentEncoder()
            self.ego_encoder = SimpleAgentEncoder()
            # self.agent_encoder = AgentEncoder()
            # self.ego_encoder = AgentEncoder()
            self.lane_encoder = CenterLaneEncoder()

        self.full_map = full_map
        if full_map:
            self.additional_lane_encoder = AdditionalLaneEncoder()
            # self.speedbump_encoder = SpeedBumpEncoder()
            # self.trafficlight_encoder = TrafficLightEncoder()
            # self.stopsign_encoder = StopSignEncoder()
            
    def segment_map(self, map, map_encoding, stride=10):
        B, N_e, N_p, D = map_encoding.shape

        # segment map
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, stride))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        # segment mask
        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//stride, N_p//(N_p//stride))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, inputs):
        # agent encoding
        ego = inputs['ego_state']
        neighbors = inputs['neighbors_state']
        if self.simple_model:
            if ego.shape[-1]==11:
                ego = ego[:,:,[0,1,2,3,4,5,6,8,9,10]]
                neighbors = neighbors[:,:,:,[0,1,2,3,4,5,6,8,9,10]]
            # elif ego.shape[-1]==10:
            #     ego = ego[:,:,[0,1,2,3,4,5,6,8,9,10]]

        actors = torch.cat([ego.unsqueeze(1), neighbors], dim=1)
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # map encoding
        map_lanes = inputs['map_lanes']
        encoded_map_lanes = self.lane_encoder(map_lanes)
        if not self.simple_model:
            map_crosswalks = inputs['map_crosswalks']
            encoded_map_crosswalks = self.crosswalk_encoder(map_crosswalks)

        # attention fusion
        encodings = []
        masks = []
        if self.full_map:
            fm_encodings = []
            fm_masks = []
        N = self._neighbors + 1
        assert actors.shape[1] >= N, 'Too many neighbors to predict'
        for i in range(N):
            lanes, lanes_mask = self.segment_map(map_lanes[:, i], encoded_map_lanes[:, i])
            if not self.simple_model:
                crosswalks, crosswalks_mask = self.segment_map(map_crosswalks[:, i], encoded_map_crosswalks[:, i])
                fusion_input = torch.cat([encoded_actors, lanes, crosswalks], dim=1)
                mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
            else:
                fusion_input = torch.cat([encoded_actors, lanes], dim=1)
                mask = torch.cat([actors_mask, lanes_mask], dim=1)
            
            if self.act_kv:
                act_kv_mask = inputs['act'][:,i]==-1
                act_kv_enc = torch.zeros_like(fusion_input[:,:1,:])
                if sum(~act_kv_mask)>0:
                    act_kv_enc[~act_kv_mask] = self.act_embedding(inputs['act'][:,i][~act_kv_mask]).unsqueeze(1)
                fusion_input = torch.cat([fusion_input, act_kv_enc], dim=1)
                mask = torch.cat([mask, act_kv_mask.unsqueeze(-1)], dim=1)

                # print()

            masks.append(mask)
            encoding = fusion_input
            encodings.append(encoding)
        
            if self.full_map:
                # map encoding
                _map_lanes = inputs['additional_map_lanes'].unsqueeze(1)
                # map_lanes = map_lanes[:,:,:,::5]
                # additional features
                _boundaries = inputs['additional_boundaries'].unsqueeze(1)
                
                # _map_crosswalks = inputs['additional_map_crosswalks'].unsqueeze(1)
                # _map_crosswalks = _map_crosswalks[:,:,:8]
                # _encoded_map_crosswalks = self.crosswalk_encoder(_map_crosswalks)
                # # additional_boundaries = additional_boundaries[:,:,:,::5]
                # map_traffic_lights, map_stop_signs, map_speed_bumps = inputs['traffic_lights'].unsqueeze(1), inputs['stop_signs'].unsqueeze(1), inputs['speed_bumps'].unsqueeze(1)
                # map_traffic_lights, map_stop_signs, map_speed_bumps = map_traffic_lights[:,:,:20], map_stop_signs[:,:,:8], map_speed_bumps[:,:,:6]
                _encoded_map_lanes = self.additional_lane_encoder(_map_lanes, _boundaries)
                # encoded_speed_bumps = self.speedbump_encoder(map_speed_bumps)
                # encoded_traffic_lights = self.trafficlight_encoder(map_traffic_lights)
                # encoded_stop_signs = self.stopsign_encoder(map_stop_signs)

                # attention fusion
                # _encoding = []
                # _mask = []
                # i=0
                _lanes, _lanes_mask = self.segment_map(_map_lanes[:, 0], _encoded_map_lanes[:, 0]) # we use 0 because we use the exact same map for all N agents
                # _crosswalks, _crosswalks_mask = self.segment_map(_map_crosswalks[:, i], _encoded_map_crosswalks[:, i], 5)
                # speed_bumps, speed_bumps_mask = self.segment_map(map_speed_bumps[:, i], encoded_speed_bumps[:, i], 5)
                # traffic_lights, traffic_lights_mask = encoded_traffic_lights[:, i], torch.eq(map_traffic_lights[:,i,:,-1], 0)
                # stop_signs, stop_signs_mask = encoded_stop_signs[:, i], torch.eq(map_stop_signs[:,i,:,0], 0)
                
                enc_lanes_fm = _lanes
                mask_lanes_fm = _lanes_mask
                fm_encodings.append(enc_lanes_fm)
                fm_masks.append(mask_lanes_fm)
                # enc_crosswalks_fm = _crosswalks
                # mask_crosswalks_fm = _crosswalks_mask
                # enc_speed_bumps_fm = speed_bumps
                # mask_speed_bumps_fm = speed_bumps_mask
                # enc_stop_signs_fm = stop_signs
                # mask_stop_signs_fm = stop_signs_mask
                # enc_traffic_lights_fm = traffic_lights
                # mask_traffic_lights_fm = traffic_lights_mask
                # masks.append(mask)
                # masks.append(mask)
                # encodings.append(encoding)
                # encodings.append(encoding)

        # outputs
        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)
        encoder_outputs = {
            'actors': actors,
            'encodings': encodings,
            'masks': masks
        }
        if self.full_map:
            fm_encodings = torch.stack(fm_encodings, dim=1)
            fm_masks = torch.stack(fm_masks, dim=1)
            encoder_outputs.update({
                'enc_lanes_fm': fm_encodings,
                'mask_lanes_fm': fm_masks,
                # 'enc_lanes_fm': enc_lanes_fm,
                # 'mask_lanes_fm': mask_lanes_fm,
                # 'enc_crosswalks_fm': enc_crosswalks_fm,
                # 'mask_crosswalks_fm': mask_crosswalks_fm,
                # 'enc_speed_bumps_fm': enc_speed_bumps_fm,
                # 'mask_speed_bumps_fm': mask_speed_bumps_fm,
                # 'enc_stop_signs_fm': enc_stop_signs_fm,
                # 'mask_stop_signs_fm': mask_stop_signs_fm,
                # 'enc_traffic_lights_fm': enc_traffic_lights_fm,
                # 'mask_traffic_lights_fm': mask_traffic_lights_fm,
                })
        
        if 'act' in inputs.keys():
            encoder_outputs['act'] = inputs['act']

        return encoder_outputs

class Encoder_02(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6, act=False, shared_act=False, ego_act_only=False, no_fuse_act=False, num_act_classes=8, full_map=False):
        super(Encoder_02, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self._neighbors = neighbors_to_predict
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)
        self.act = act
        self.shared_act = shared_act
        self.ego_act_only=ego_act_only
        self.no_fuse_act = no_fuse_act
        self.intent_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(128, 5)
            nn.Linear(128, num_act_classes)
        )
        if act:
            print("ACT ENCODER ACTIVATED")
            self.act = act
            self.act_encoder = nn.Embedding(num_act_classes, 256, padding_idx=0)
        
        self.full_map = full_map
        if full_map:
            self.lane_mapping_crossatt = CrossTransformer()
            num_latent_tokens = 16
            self.lane_query_embed = nn.Embedding(num_latent_tokens, 256)
            self.agent_embedding = nn.Embedding(2, 256)
            self.register_buffer('lane_projection_queries', torch.arange(num_latent_tokens).long())


    def forward(self, inputs):
        masks_ = inputs['masks']
        encodings_ = inputs['encodings']
        actors = inputs['actors']
        # attention fusion
        encodings = []
        masks = []
        if self.shared_act:
            shared_act = []
        N = self._neighbors + 1
        assert actors.shape[1] >= N, 'Too many neighbors to predict'
        intent_losses = []
        for i in range(N):
            if self.act and not self.no_fuse_act:
                if self.ego_act_only:
                    if i!=0:
                        encoded_act = torch.zeros((inputs['act'].shape[0], 256), device=inputs['act'].device)
                        encoded_act_mask = torch.ones_like(inputs['act'][:,i]).bool()
                    else:
                        encoded_act = torch.zeros((inputs['act'].shape[0], 256), device=inputs['act'].device)
                        encoded_act_mask = torch.ones_like(inputs['act'][:,i]).bool()
                        valid_act_mask = inputs['act'][:,i].int() != -1
                        if sum(valid_act_mask)>0:
                            encoded_act[valid_act_mask] = self.act_encoder(inputs['act'][valid_act_mask,i].int())
                            encoded_act_mask[valid_act_mask] = torch.zeros_like(inputs['act'][valid_act_mask,i]).bool()
                else:
                    encoded_act = torch.zeros((inputs['act'].shape[0], 256), device=inputs['act'].device)
                    encoded_act_mask = torch.ones_like(inputs['act'][:,i]).bool()
                    valid_act_mask = inputs['act'][:,i].int() != -1
                    if sum(valid_act_mask)>0:
                        encoded_act[valid_act_mask] = self.act_encoder(inputs['act'][valid_act_mask,i].int())
                        encoded_act_mask[valid_act_mask] = torch.zeros_like(inputs['act'][valid_act_mask,i]).bool()
                    # encoded_act = self.act_encoder(inputs['act'][:,i].int())
                #     encoded_act_mask = torch.zeros_like(inputs['act'][:,i]).bool() #torch.zeros(inputs['act'][:,i].shape, device=encoded_act.device).bool()
                if self.shared_act:
                    shared_act.append(encoded_act)
                # fusion_input = torch.cat([encodings_[:,i], encoded_act[:,None]], dim=1)
                fusion_input = torch.cat([encodings_[:,i], encoded_act[:,None]], dim=1)
                mask = torch.cat([masks_[:,i], encoded_act_mask[:,None]], dim=1)
            else:
                fusion_input = encodings_[:,i]
                mask = masks_[:,i]
            
            if self.full_map:
                # lane_q = self.lane_query_embed(self.lane_projection_queries.repeat(inputs['enc_lanes_fm'].shape[0],1))
                lane_q = self.lane_query_embed(self.lane_projection_queries)
                agent_embed = self.agent_embedding(torch.tensor(i, device=self.lane_projection_queries.device))
                lane_q = lane_q + agent_embed
                lane_q = torch.stack([lane_q for _ in range(inputs['enc_lanes_fm'].shape[0])])
                enc_lanes_fm = self.lane_mapping_crossatt(lane_q, inputs['enc_lanes_fm'][:,i], inputs['enc_lanes_fm'][:,i], inputs['mask_lanes_fm'][:,i])
                mask_lanes_fm = torch.zeros_like(enc_lanes_fm[:,...,0], dtype=torch.bool)
                # full_map_encodings = [enc_lanes_fm] + [inputs[fm_enc_key] for fm_enc_key in inputs.keys() if 'fm' in fm_enc_key and 'enc' in fm_enc_key and 'lanes' not in fm_enc_key]
                # full_map_encodings = torch.cat(full_map_encodings, 1)
                # full_map_masks = [mask_lanes_fm] + [inputs[fm_enc_key] for fm_enc_key in inputs.keys() if 'fm' in fm_enc_key and 'mask' in fm_enc_key and 'lanes' not in fm_enc_key]
                # full_map_masks = torch.cat(full_map_masks, 1)
                fusion_input = torch.cat((fusion_input, enc_lanes_fm), dim=1)
                mask = torch.cat((mask, mask_lanes_fm), dim=1)

            # (Pdb) fusion_input.shape
            # torch.Size([64, 114, 256]) (batch size, number of features, feature dimension)
            
            encoding = self.fusion_encoder(fusion_input, src_key_padding_mask=mask)
            ################################################################################################################################################################################
            pooled_fusion = encoding.mean(dim=1)  # [B, 256]
            intent_logits = self.intent_mlp(pooled_fusion)  # [B, num_act_classes]
            intent_pred = intent_logits.argmax(dim=-1)     # [B]
            neighbor_dummy = torch.full_like(intent_pred, fill_value=-1)  # [B]
            intent_full = torch.stack([intent_pred, neighbor_dummy], dim=1)  # [B, 2]

            if i == 0 and 'act' in inputs:
                gt_act = inputs['act'][:, 0].long()  # ground truth for ego
                # print(gt_act)
                valid_mask = gt_act != -1
                if valid_mask.any():
                    ce_loss = F.cross_entropy(intent_logits[valid_mask], gt_act[valid_mask])
                    intent_losses.append(ce_loss)

            # print(intent_full)
            # print(inputs['act'])
            # with open("evaluation_acts_mlp_classifier.txt", "a") as f:
            #     for i in range(len(intent_full)):
            #         f.write(f"({int(intent_full[0][0])},{inputs['act'][0][0]})\n")

            preds = intent_full[:, 0]
            gts = inputs['act'][:, 0]
            valid = gts != -1

            preds = preds[valid]
            gts = gts[valid]

            classes = torch.unique(gts)

            total_correct = 0
            total_samples = 0

            for cls in classes:
                cls_mask = gts == cls
                cls_total = cls_mask.sum().item()
                cls_correct = (preds[cls_mask] == gts[cls_mask]).sum().item()
                
                accuracy = 100.0 * cls_correct / cls_total if cls_total > 0 else 0.0
                # print(f"Class {cls.item()}: {cls_correct} correct out of {cls_total} "
                    # f"({accuracy:.2f}%)")
                
                total_correct += cls_correct
                total_samples += cls_total

            overall_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
            # print(f"\nOverall accuracy: {total_correct} correct out of {total_samples} "
                # f"({overall_accuracy:.2f}%)")


            ################################################################################################################################################################################
            # import pdb; pdb.set_trace()
            # this code is not being used.
            if self.act and self.no_fuse_act:
                if i==0:
                    encoding = torch.cat((encoding, self.act_encoder(inputs['act'][:,i].int())[:,None]), dim=1)
                    mask = torch.cat((mask, torch.zeros_like(inputs['act'][:,i]).bool()[:,None]), dim=1)
                else:
                    encoding = torch.cat((encoding, torch.zeros((inputs['act'].shape[0], 256), device=inputs['act'].device)[:,None]), dim=1)
                    mask = torch.cat((mask, torch.ones_like(inputs['act'][:,i]).bool()[:,None]), dim=1)

            encodings.append(encoding)
            masks.append(mask)
        
        # if self.full_map:
        #     # lane_q = self.lane_query_embed(self.lane_projection_queries.repeat(inputs['enc_lanes_fm'].shape[0],1))
        #     lane_q = self.lane_query_embed(self.lane_projection_queries)
        #     lane_q = torch.stack([lane_q for i in range(inputs['enc_lanes_fm'].shape[0])])
        #     enc_lanes_fm = self.lane_mapping_crossatt(lane_q, inputs['enc_lanes_fm'], inputs['enc_lanes_fm'], inputs['mask_lanes_fm'])
        #     mask_lanes_fm = torch.zeros_like(enc_lanes_fm[...,0], dtype=torch.bool)
        #     full_map_encodings = [enc_lanes_fm] + [inputs[fm_enc_key] for fm_enc_key in inputs.keys() if 'fm' in fm_enc_key and 'enc' in fm_enc_key and 'lanes' not in fm_enc_key]
        #     full_map_encodings = torch.cat(full_map_encodings, 1)
        #     full_map_masks = [mask_lanes_fm] + [inputs[fm_enc_key] for fm_enc_key in inputs.keys() if 'fm' in fm_enc_key and 'mask' in fm_enc_key and 'lanes' not in fm_enc_key]
        #     full_map_masks = torch.cat(full_map_masks, 1)
            # for i in range(len(encodings)):
            #     encodings[i] = torch.cat((encodings[i], full_map_encodings),1)
            #     masks[i] =  torch.cat((masks[i], full_map_masks),1)


        # outputs
        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)
        encoder_outputs = {
            'actors': actors,
            'encodings': encodings,
            'masks': masks
        }
        
        # if self.full_map:
        #     encoder_outputs.update({
        #         'full_map_encodings': full_map_encodings,
        #         'full_map_masks': full_map_masks
        #     })

        # if 'act' in inputs.keys():
        #     # encoder_outputs['act'] = inputs['act']
        #     encoder_outputs['act'] = intent_full.detach()
            
        # if intent_losses:
        #     encoder_outputs['intent_loss'] = torch.stack(intent_losses).mean()

        if intent_losses:
            loss = torch.stack(intent_losses).mean()
            encoder_outputs['intent_loss'] = loss  
            encoder_outputs['intent_loss_scalar'] = loss.detach().item()

        if self.shared_act:
            shared_act = torch.stack(shared_act, dim=1)
            encoder_outputs['shared_act'] = shared_act
        # import pdb; pdb.set_trace()
        # encoder_outputs keys = actors encodings masks act
        # print(encoder_outputs['act'].shape) = (64, 2)
        return encoder_outputs


class GameFormer_(nn.Module):
    def __init__(self, modalities, neighbors_to_predict, future_len, encoder_layers=6, decoder_levels=4, act=False, act_dec=False, shared_act=False, ego_act_only=True, no_fuse_act=False, num_act_classes =8, full_map = False, simple_model=False, act_kv=False):
        super(GameFormer_, self).__init__()
        if full_map:
            print("FULL MAP")
        self.encoder_01 = Encoder_01(neighbors_to_predict, encoder_layers, act, full_map, simple_model, act_kv)
        self.encoder_02 = Encoder_02(neighbors_to_predict, encoder_layers, act, shared_act, ego_act_only, no_fuse_act, num_act_classes, full_map)

    def forward(self, inputs):
        encoder_outputs = self.encoder_01(inputs)
        encoder_outputs = self.encoder_02(encoder_outputs)
        outputs = encoder_outputs

        # if 'intent_loss' in encoder_outputs:
        #     outputs['intent_loss'] = encoder_outputs['intent_loss']

        return outputs
