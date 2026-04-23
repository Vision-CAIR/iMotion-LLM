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


            encoding = self.fusion_encoder(fusion_input, src_key_padding_mask=mask)

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

        if 'act' in inputs.keys():
            encoder_outputs['act'] = inputs['act']

        if self.shared_act:
            shared_act = torch.stack(shared_act, dim=1)
            encoder_outputs['shared_act'] = shared_act

        return encoder_outputs

class Encoder(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6, act=False):
        super(Encoder, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self._neighbors = neighbors_to_predict
        self.agent_encoder = AgentEncoder()
        self.ego_encoder = AgentEncoder()
        self.lane_encoder = LaneEncoder()
        self.crosswalk_encoder = CrosswalkEncoder()
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)
        self.act = act
        self.act_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(128, num_intent_classes)
            nn.Linear(128, 5)
        )
        if act:
            print("ACT ENCODER ACTIVATED")
            self.act = act
            self.act_encoder = nn.Embedding(5, 256, padding_idx=0)
            
    def segment_map(self, map, map_encoding):
        stride = 10
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
        actors = torch.cat([inputs['ego_state'].unsqueeze(1), neighbors], dim=1)
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        # map encoding
        map_lanes = inputs['map_lanes']
        map_crosswalks = inputs['map_crosswalks']
        encoded_map_lanes = self.lane_encoder(map_lanes)
        encoded_map_crosswalks = self.crosswalk_encoder(map_crosswalks)

        # attention fusion
        encodings = []
        masks = []
        N = self._neighbors + 1
        assert actors.shape[1] >= N, 'Too many neighbors to predict'

        for i in range(N):
            lanes, lanes_mask = self.segment_map(map_lanes[:, i], encoded_map_lanes[:, i])
            crosswalks, crosswalks_mask = self.segment_map(map_crosswalks[:, i], encoded_map_crosswalks[:, i])
            if self.act:
                encoded_act = self.act_encoder(inputs['act'][:,i].int())
                encoded_act_mask = torch.zeros(inputs['act'][:,i].shape, device=encoded_act.device).bool()
                fusion_input = torch.cat([encoded_actors, lanes, crosswalks, encoded_act[:,None]], dim=1)
                mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask, encoded_act_mask[:,None]], dim=1)
            else:
                fusion_input = torch.cat([encoded_actors, lanes, crosswalks], dim=1)
                mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
            masks.append(mask)
            encoding = self.fusion_encoder(fusion_input, src_key_padding_mask=mask)
            encodings.append(encoding)

        # outputs
        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)
        encoder_outputs = {
            'actors': actors,
            'encodings': encodings,
            'masks': masks
        }
        
        if 'act' in inputs.keys():
            encoder_outputs['act'] = inputs['act']
        # if self.act:
        #     encoder_outputs['act'] = inputs['act']

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, modalities, future_len, neighbors_to_predict, levels=3, act=False, shared_act=False, ego_act_only=False, num_act_classes=8, full_map=False):
        super(Decoder, self).__init__()
        self._levels = levels
        self._neighbors = neighbors_to_predict
        future_encoder = FutureEncoder()
        self.initial_stage = InitialDecoder(modalities, neighbors_to_predict, future_len, act=act, shared_act=shared_act, num_act_classes=num_act_classes)
        self.interaction_stage = nn.ModuleList([InteractionDecoder(future_encoder, future_len) for _ in range(levels)])
        self.ego_act_only = ego_act_only
        self.act = act
        # self.full_map = full_map

    def forward(self, encoder_inputs):
        decoder_outputs = {}
        N = self._neighbors + 1
        assert encoder_inputs['actors'].shape[1] >= N, 'Too many neighbors to predict'

        current_states = encoder_inputs['actors'][:, :, -1]
        encodings, masks = encoder_inputs['encodings'], encoder_inputs['masks']

        # if self.full_map:
        #     full_map_encodings = torch.stack((encoder_inputs['full_map_encodings'], encoder_inputs['full_map_encodings']),1)
        #     full_map_masks = torch.stack((encoder_inputs['full_map_masks'], encoder_inputs['full_map_masks']),1)
        #     encodings = torch.cat((encodings, full_map_encodings), 2)
        #     masks = torch.cat((masks, full_map_masks), 2)
        # level 0
        if self.act:
            if 'shared_act' in encoder_inputs.keys():
                results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i], encoder_inputs['shared_act'][:, i]) for i in range(N)]
            else:
                if self.ego_act_only:
                    results = []
                    i=0
                    if encoder_inputs['act'].shape[1]==6:
                        results.append(self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i], encoder_inputs['act'][:]))
                    else:
                        results.append(self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i], encoder_inputs['act'][:, i]))
                    if N==2:
                        i=1
                        results.append(self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i], None))
                else:
                    results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i], encoder_inputs['act'][:, i]) for i in range(N)]
        else:
            results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
        last_content = torch.stack([result[0] for result in results], dim=1) # query content [batch, N=2 agents, M modalities, dim=256]
        last_level = torch.stack([result[1] for result in results], dim=1) # GMM [batch, N=2, M, future T steps, 4 represent GMM parameters]
        last_scores = torch.stack([result[2] for result in results], dim=1) # [batch, 2, N, M]
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_scores
        
        # level k reasoning
        for k in range(1, self._levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            results = [interaction_decoder(i, current_states[:, :N], last_level, last_scores, \
                       last_content[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
            last_content = torch.stack([result[0] for result in results], dim=1)
            last_level = torch.stack([result[1] for result in results], dim=1)
            last_scores = torch.stack([result[2] for result in results], dim=1)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_scores

        return decoder_outputs


class GameFormer(nn.Module):
    def __init__(self, modalities, neighbors_to_predict, future_len, encoder_layers=6, decoder_levels=4, act=False, act_dec=False):
        super(GameFormer, self).__init__()
        self.encoder = Encoder(neighbors_to_predict, encoder_layers, act)
        self.decoder = Decoder(modalities, future_len, neighbors_to_predict, decoder_levels, act_dec)

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        outputs = self.decoder(encoder_outputs)

        return outputs

class GameFormer_(nn.Module):
    def __init__(self, modalities, neighbors_to_predict, future_len, encoder_layers=6, decoder_levels=4, act=False, act_dec=False, shared_act=False, ego_act_only=True, no_fuse_act=False, num_act_classes =8, full_map = False, simple_model=False, act_kv=False):
        super(GameFormer_, self).__init__()
        if full_map:
            print("FULL MAP")
        self.encoder_01 = Encoder_01(neighbors_to_predict, encoder_layers, act, full_map, simple_model, act_kv)
        self.encoder_02 = Encoder_02(neighbors_to_predict, encoder_layers, act, shared_act, ego_act_only, no_fuse_act, num_act_classes, full_map)
        self.decoder = Decoder(modalities, future_len, neighbors_to_predict, decoder_levels, act_dec, shared_act, ego_act_only, num_act_classes, full_map=full_map)

    def forward(self, inputs):
        encoder_outputs = self.encoder_01(inputs)
        encoder_outputs = self.encoder_02(encoder_outputs)
        outputs = self.decoder(encoder_outputs)

        return outputs



class Encoder_01_2(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6, act=False, full_map=True):
        super(Encoder_01_2, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self._neighbors = neighbors_to_predict
        self.agent_encoder = AgentEncoder()
        self.ego_encoder = AgentEncoder()
        self.full_map = full_map
        # if not full_map:
        self.lane_encoder = LaneEncoder(full_map)
        self.crosswalk_encoder = CrosswalkEncoder()
        self.speedbump_encoder = SpeedBumpEncoder()
        self.trafficlight_encoder = TrafficLightEncoder()
        self.stopsign_encoder = StopSignEncoder()
        # else:
            # self.full_map_encoder = LaneEncoderFullMap()
            # self.crosswalk_encoder = CrosswalkEncoder()

            
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
        actors = torch.cat([inputs['ego_state'].unsqueeze(1), neighbors], dim=1)
        encoded_ego = self.ego_encoder(ego)
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        
        if self.full_map:
            # map encoding
            map_lanes = inputs['map_lanes'].unsqueeze(1)
            # map_lanes = map_lanes[:,:,:,::5]
            map_crosswalks = inputs['map_crosswalks'].unsqueeze(1)
            map_crosswalks = map_crosswalks[:,:,:8]
            encoded_map_crosswalks = self.crosswalk_encoder(map_crosswalks)
            # additional features
            additional_boundaries = inputs['additional_boundaries'].unsqueeze(1)
            # additional_boundaries = additional_boundaries[:,:,:,::5]
            map_traffic_lights, map_stop_signs, map_speed_bumps = inputs['traffic_lights'].unsqueeze(1), inputs['stop_signs'].unsqueeze(1), inputs['speed_bumps'].unsqueeze(1)
            map_traffic_lights, map_stop_signs, map_speed_bumps = map_traffic_lights[:,:,:20], map_stop_signs[:,:,:8], map_speed_bumps[:,:,:6]
            encoded_map_lanes = self.lane_encoder(map_lanes, additional_boundaries)
            encoded_speed_bumps = self.speedbump_encoder(map_speed_bumps)
            encoded_traffic_lights = self.trafficlight_encoder(map_traffic_lights)
            encoded_stop_signs = self.stopsign_encoder(map_stop_signs)

            # attention fusion
            encodings = []
            masks = []
            N = self._neighbors + 1
            assert actors.shape[1] >= N, 'Too many neighbors to predict'
            # for i in range(N):
            i=0
            lanes, lanes_mask = self.segment_map(map_lanes[:, i], encoded_map_lanes[:, i])
            crosswalks, crosswalks_mask = self.segment_map(map_crosswalks[:, i], encoded_map_crosswalks[:, i])
            speed_bumps, speed_bumps_mask = self.segment_map(map_speed_bumps[:, i], encoded_speed_bumps[:, i])
            traffic_lights, traffic_lights_mask = encoded_traffic_lights[:, i], torch.eq(map_traffic_lights[:,i,:,-1], 0)
            stop_signs, stop_signs_mask = encoded_stop_signs[:, i], torch.eq(map_stop_signs[:,i,:,0], 0)
            fusion_input = torch.cat([encoded_actors, lanes, crosswalks, speed_bumps, stop_signs, traffic_lights], dim=1)
            mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask, speed_bumps_mask, stop_signs_mask, traffic_lights_mask], dim=1)
            encoding = fusion_input
            # masks.append(mask)
            masks.append(mask)
            # encodings.append(encoding)
            encodings.append(encoding)
        
        else:
            # map encoding
            map_lanes = inputs['map_lanes']
            map_crosswalks = inputs['map_crosswalks']
            encoded_map_crosswalks = self.crosswalk_encoder(map_crosswalks)
            encoded_map_lanes = self.lane_encoder(map_lanes)
            # attention fusion
            encodings = []
            masks = []
            N = self._neighbors + 1
            assert actors.shape[1] >= N, 'Too many neighbors to predict'

            for i in range(N):
                lanes, lanes_mask = self.segment_map(map_lanes[:, i], encoded_map_lanes[:, i])
                crosswalks, crosswalks_mask = self.segment_map(map_crosswalks[:, i], encoded_map_crosswalks[:, i])
                fusion_input = torch.cat([encoded_actors, lanes, crosswalks], dim=1)
                mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
                masks.append(mask)
                encoding = fusion_input
                encodings.append(encoding)

        # outputs
        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)
        encoder_outputs = {
            'actors': actors,
            'encodings': encodings,
            'masks': masks,
            'lane_encodings_idx': [encoded_actors.shape[1], encoded_actors.shape[1] + lanes.shape[1]]
        }
        
        if 'act' in inputs.keys():
            encoder_outputs['act'] = inputs['act']

        return encoder_outputs

class Encoder_02_2(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6, act=False):
        super(Encoder_02_2, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self._neighbors = neighbors_to_predict
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)
        self.act = act
        if act:
            print("ACT ENCODER ACTIVATED")
            self.act = act
        wf.act_encoder = nn.Embedding(5, 256, padding_idx=0)

        self.mapping_crossatt = CrossTransformer(dropout=0.0)
        num_latent_tokens = 128
        self.query_embed = nn.Embedding(num_latent_tokens, 256)
        self.register_buffer('projection_queries', torch.arange(num_latent_tokens).long())
            
    def forward(self, inputs):
        masks_ = inputs['masks']
        encodings_ = inputs['encodings']
        actors = inputs['actors']
        # attention fusion
        encodings = []
        masks = []
        N = self._neighbors + 1
        assert actors.shape[1] >= N, 'Too many neighbors to predict'
        # for i in range(N):
        for i in range(encodings_.shape[1]):
            if self.act:
                if i!=0:
                    encoded_act = torch.zeros((inputs['act'].shape[0], 256), device=inputs['act'].device)
                    encoded_act_mask = torch.ones_like(inputs['act'][:,i]).bool()
                else:
                    encoded_act = self.act_encoder(inputs['act'][:,i].int())
                    encoded_act_mask = torch.zeros_like(inputs['act'][:,i]).bool() #torch.zeros(inputs['act'][:,i].shape, device=encoded_act.device).bool()
                
                fusion_input = torch.cat([encodings_[:,i], encoded_act[:,None]], dim=1)
                mask = torch.cat([masks_[:,i], encoded_act_mask[:,None]], dim=1)
            else:
                fusion_input = encodings_[:,i]
                mask = masks_[:,i]
            # ## cross attention mapping to smaller number of tokens, applied on lanes information only
            q = self.query_embed(self.projection_queries.repeat(fusion_input.shape[0],1))
            lane_fusion_input = self.mapping_crossatt(q, fusion_input[:, inputs['lane_encodings_idx'][0]:inputs['lane_encodings_idx'][1]], fusion_input[:, inputs['lane_encodings_idx'][0]:inputs['lane_encodings_idx'][1]], mask[:, inputs['lane_encodings_idx'][0]:inputs['lane_encodings_idx'][1]])
            lane_mask = torch.zeros_like(lane_fusion_input[...,0], dtype=torch.bool)
            fusion_input = torch.cat((fusion_input[:,:inputs['lane_encodings_idx'][0]], lane_fusion_input, fusion_input[:,inputs['lane_encodings_idx'][1]:]), dim=1)
            mask = torch.cat((mask[:,:inputs['lane_encodings_idx'][0]], lane_mask, mask[:,inputs['lane_encodings_idx'][1]:]), dim=1)
            # ## end of cross attention mapping to smaller number of tokens
            masks.append(mask)
            encoding = self.fusion_encoder(fusion_input, src_key_padding_mask=mask)
            encodings.append(encoding)

        # outputs
        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)
        encoder_outputs = {
            'actors': actors,
            'encodings': encodings,
            'masks': masks
        }
        
        if 'act' in inputs.keys():
            encoder_outputs['act'] = inputs['act']

        return encoder_outputs

class Decoder_2(nn.Module):
    def __init__(self, modalities, future_len, neighbors_to_predict, levels=3, act=False):
        super(Decoder_2, self).__init__()
        self._levels = levels
        self._neighbors = neighbors_to_predict
        future_encoder = FutureEncoder()
        self.initial_stage = InitialDecoder_2(modalities, neighbors_to_predict, future_len, act=act)
        self.interaction_stage = nn.ModuleList([InteractionDecoder(future_encoder, future_len) for _ in range(levels)])
        self.act = act

    def forward(self, encoder_inputs):
        decoder_outputs = {}
        N = self._neighbors + 1
        assert encoder_inputs['actors'].shape[1] >= N, 'Too many neighbors to predict'

        current_states = encoder_inputs['actors'][:, :, -1]
        encodings, masks = encoder_inputs['encodings'], encoder_inputs['masks']

        # level 0
        if self.act:
            results = []
            i=0
            results.append(self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i], encoder_inputs['act'][:, i]))
            i=1
            if encodings.shape[1] == N:
                i_enc = i
            else:
                i_enc = 0
            results.append(self.initial_stage(i, current_states[:, i], encodings[:, i_enc], masks[:, i_enc], None))
            # results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i], encoder_inputs['act'][:, i]) for i in range(N)]
        else:
            results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
        last_content = torch.stack([result[0] for result in results], dim=1) # query content [batch, N=2 agents, M modalities, dim=256]
        last_level = torch.stack([result[1] for result in results], dim=1) # GMM [batch, N=2, M, future T steps, 4 represent GMM parameters]
        last_scores = torch.stack([result[2] for result in results], dim=1) # [batch, 2, N, M]
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_scores
        
        # level k reasoning
        if encodings.shape[1]!=N:
            encodings = torch.cat((encodings,encodings), dim=1)
            masks = torch.cat((masks,masks), dim=1)
        
        for k in range(1, self._levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            results = [interaction_decoder(i, current_states[:, :N], last_level, last_scores, \
                       last_content[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
            last_content = torch.stack([result[0] for result in results], dim=1)
            last_level = torch.stack([result[1] for result in results], dim=1)
            last_scores = torch.stack([result[2] for result in results], dim=1)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_scores

        return decoder_outputs

class GameFormer_2(nn.Module):
    def __init__(self, modalities, neighbors_to_predict, future_len, encoder_layers=6, decoder_levels=4, act=False, act_dec=False, full_map = False):
        super(GameFormer_2, self).__init__()
        self.encoder_01 = Encoder_01_2(neighbors_to_predict, encoder_layers, act, full_map)
        self.encoder_02 = Encoder_02_2(neighbors_to_predict, encoder_layers, act)
        self.decoder = Decoder_2(modalities, future_len, neighbors_to_predict, decoder_levels, act_dec)
        
        self.mapping_crossatt = CrossTransformer(dropout=0.0)
        num_latent_tokens = 256
        self.query_embed = nn.Embedding(num_latent_tokens, 256)
        self.register_buffer('projection_queries', torch.arange(num_latent_tokens).long())

    def forward(self, inputs):
        encoder_outputs = self.encoder_01(inputs)
        encoder_outputs = self.encoder_02(encoder_outputs)
        
        q = self.query_embed(self.projection_queries.repeat(encoder_outputs['encodings'].shape[0],1))
        encoder_outputs['encodings'] = self.mapping_crossatt(q, encoder_outputs['encodings'][:,0], encoder_outputs['encodings'][:,0], encoder_outputs['masks'][:,0]).unsqueeze(1)
        encoder_outputs['masks'] = torch.zeros_like(encoder_outputs['encodings'][...,0], dtype=torch.bool)
        ## encoder_outputs['masks'] = torch.zeros((encoder_outputs['encodings'].shape[0], encoder_outputs['encodings'].shape[1], encoder_outputs['encodings'].shape[2]), dtype=torch.bool)
        
        outputs = self.decoder(encoder_outputs)
        # print(f"decoding: {time.time()-t_}s")

        return outputs
