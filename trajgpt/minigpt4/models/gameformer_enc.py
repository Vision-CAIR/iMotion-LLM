import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, max_len=100):
        super(PositionalEncoding, self).__init__()
        d_model = 256
        dropout = 0.1
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)


class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]
        # type = self.type_emb(inputs[:, -1, 8].int()) #This needs correction object type is encoded as one hot vecotor of length 3, thus using the 8th element only is not the object type
        assert inputs[:, -1, 8].max()<2 # Remove this after validating that it is correct
        object_type = (inputs[:, -1, 8:].argmax(dim=-1)+1)* inputs[:, -1, 8:].sum(dim=-1).int() # This fix the problem above
        type = self.type_emb(object_type)
        output = output + type

        return output

class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256))
        self.position_encode = PositionalEncoding(max_len=100)

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[...,  6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].int())
        left_type = self.left_type(inputs[..., 11].int())
        right_type = self.right_type(inputs[..., 12].int()) 
        traffic_light = self.traffic_light_type(inputs[..., 13].int())
        interpolating = self.interpolating(inputs[..., 14].int()) 
        stop_sign = self.stop_sign(inputs[..., 15].int())

        lane_attr = self_type + left_type + right_type + traffic_light + interpolating + stop_sign
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
    
        # process
        output = self.position_encode(self.pointnet(lane_embedding))

        return output

class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
    
    def forward(self, inputs):
        output = self.point_net(inputs)

        return output


class GameformerEncoder(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6):
        super(GameformerEncoder, self).__init__()
        heads, dim, dropout = 8, 256, 0.01
        self._neighbors = neighbors_to_predict
        self.agent_encoder = AgentEncoder()
        self.ego_encoder = AgentEncoder()
        self.lane_encoder = LaneEncoder()
        self.crosswalk_encoder = CrosswalkEncoder()
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)
        self.llm_adapter = nn.Linear(256, 4096)

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
            fusion_input = torch.cat([encoded_actors, lanes, crosswalks], dim=1)
            mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)
            masks.append(mask)
            encoding = self.fusion_encoder(fusion_input, src_key_padding_mask=mask)
            encoding = self.llm_adapter(encoding)
            encodings.append(encoding)

        # outputs
        encodings = torch.stack(encodings, dim=1)
        # encodings = self.llm_adapter(encodings)
        masks = torch.stack(masks, dim=1)
        
        # traj data
        encoder_outputs = {
            'actors': actors,
            'encodings': encodings,
            'masks': masks,
        }
        
        return encoder_outputs