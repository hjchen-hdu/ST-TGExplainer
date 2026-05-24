import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder, MergeLayer
from utils.utils import NeighborSampler
from models.modules import MLP


class ST_TGExplainer(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, num_neighbors: int, num_layers: int = 2, dropout: float = 0.1, device: str = 'cpu', max_time_shift: float = 0.0, version: str = 'ALL'):
        super(ST_TGExplainer, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        
        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = 172
        self.edge_feat_dim = 172
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.dropout_prob = dropout
        self.device = device
        self.num_tokens = num_neighbors
        self.num_channels = self.edge_feat_dim
        self.hidden_size = self.edge_feat_dim
        self.token_dim_expansion_factor = 0.5 
        self.channel_dim_expansion_factor = 4.0

        self.max_time_shift = max_time_shift

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, parameter_requires_grad=False)
        self.time_projection = MLP(num_layers=1, input_dim=1, hidden_dim=self.hidden_size, output_dim=self.hidden_size, dropout=self.dropout_prob, use_act=True)

        self.node_embedding = nn.Embedding(
            node_raw_features.shape[0]+1, self.hidden_size, padding_idx=0
        )

        self.sampl_encoder = SampleEncoder_speed(self.node_feat_dim, self.device)

        self.edge_projection_layer = nn.Linear(self.edge_feat_dim + time_feat_dim, self.num_channels)

        self.edge_mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_tokens * 2, num_channels=self.num_channels,
                     token_dim_expansion_factor=self.token_dim_expansion_factor,
                     channel_dim_expansion_factor=self.channel_dim_expansion_factor, dropout=self.dropout_prob)
            for _ in range(self.num_layers)
        ])

        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.affinity_score = MergeLayer(self.node_feat_dim, self.node_feat_dim, self.node_feat_dim, 1)

        self.discriminator = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + 1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        self.fix_r = False
        self.init_r = 0.7
        self.decay_interval = 5
        self.decay_r = 0.1
        self.final_r = 0.5 # 0.1

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20, epoch: int=1, training: bool = True, weight = None):
        node_features, neighbor_features, sample_ratio = self.exact_subgraph(src_node_ids, dst_node_ids, node_interact_times, num_neighbors, weight)

        src_subgraph_embedding, dst_subgraph_embedding = self.Mixer(node_features, neighbor_features)

        score_logits = self.affinity_score(src_subgraph_embedding, dst_subgraph_embedding)
        score = self.sampling(score_logits, training).squeeze(dim=2)

        loss = {}
        r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (score * torch.log(score/r + 1e-6) + (1-score) * torch.log((1-score)/(1-r+1e-6) + 1e-6)).mean()
        loss['info_loss'] = info_loss

        rep_ratio = self.sampling(sample_ratio, training).squeeze()
        exp_ratio = 1-rep_ratio
        
        rep_ratio = rep_ratio*score
        exp_ratio = exp_ratio*score
        
        rep_src_embedding, rep_dst_embedding = self.Mixer(node_features, neighbor_features, rep_ratio)
        exp_src_embedding, exp_dst_embedding = self.Mixer(node_features, neighbor_features, exp_ratio)

        h_S = torch.mean(torch.cat([rep_src_embedding, rep_dst_embedding], dim=1), dim=1)
        h_T = torch.mean(torch.cat([exp_src_embedding, exp_dst_embedding], dim=1), dim=1)
        loss['h_S'] = h_S
        loss['h_T'] = h_T

        src_embedding = torch.mean(rep_src_embedding+exp_src_embedding, dim=1)
        dst_embedding = torch.mean(rep_dst_embedding+exp_dst_embedding, dim=1)

        return loss, score, src_embedding, dst_embedding
    
    def Mixer(self, node_features, neighbor_features, ratio=None):
        bs = node_features.shape[0]//2
        for mlp_mixer in self.edge_mlp_mixers:
            neighbor_features = mlp_mixer(neighbor_features, ratio)
        src_subgraph_embedding = neighbor_features[:, :self.num_neighbors, :]
        dst_subgraph_embedding = neighbor_features[:, self.num_neighbors:, :]
        src_subgraph_embedding = node_features[bs:,:].expand(-1, self.num_neighbors, -1) + src_subgraph_embedding
        dst_subgraph_embedding = node_features[:bs,:].expand(-1, self.num_neighbors, -1) + dst_subgraph_embedding

        # layernormal
        src_subgraph_embedding = self.LayerNorm(src_subgraph_embedding)
        dst_subgraph_embedding = self.LayerNorm(dst_subgraph_embedding)
        # dropout
        src_subgraph_embedding = self.dropout(src_subgraph_embedding)
        dst_subgraph_embedding = self.dropout(dst_subgraph_embedding)

        return src_subgraph_embedding, dst_subgraph_embedding
    
    def exact_subgraph(self, node_src_ids: np.ndarray, node_dst_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20, weight=None):

        node_src_ids = torch.from_numpy(node_src_ids).to(self.device)
        node_dst_ids = torch.from_numpy(node_dst_ids).to(self.device)
        src_node_raw_features = self.node_embedding(node_src_ids).unsqueeze(1)
        dst_node_raw_features = self.node_embedding(node_dst_ids).unsqueeze(1)
        
        neighbor_src_node_ids, neighbor_src_edge_ids, neighbor_src_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_src_ids,
                                                            node_interact_times=node_interact_times,
                                                            num_neighbors=num_neighbors)

        neighbor_dst_node_ids, neighbor_dst_edge_ids, neighbor_dst_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_dst_ids,
                                                            node_interact_times=node_interact_times,
                                                            num_neighbors=num_neighbors)
        
        
        # mask                                                                
        if weight is not None:
            neighbor_src_node_ids = neighbor_src_node_ids * weight[:, :num_neighbors].cpu().numpy().astype(np.int32)
            neighbor_dst_node_ids = neighbor_dst_node_ids * weight[:, num_neighbors:].cpu().numpy().astype(np.int32)
        
        # Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        neighbor_src_node_ids = torch.from_numpy(neighbor_src_node_ids).to(self.device)
        neighbor_dst_node_ids = torch.from_numpy(neighbor_dst_node_ids).to(self.device)
        src_nodes_raw_features = self.node_embedding(neighbor_src_node_ids)
        dst_nodes_raw_features = self.node_embedding(neighbor_dst_node_ids)
        src_delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_src_times).float().to(self.device)
        dst_delta_times = torch.from_numpy(node_interact_times[:, np.newaxis] - neighbor_dst_times).float().to(self.device)

        # Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        src_neighbor_time_features = self.time_encoder(timestamps=src_delta_times)
        dst_neighbor_time_features = self.time_encoder(timestamps=dst_delta_times)

        # ndarray, set the time features to all zeros for the padded timestamp
        src_neighbor_time_features[neighbor_src_node_ids == 0] = 0.0
        dst_neighbor_time_features[neighbor_dst_node_ids == 0] = 0.0

        src_nodes_raw_features[neighbor_src_node_ids == 0] = 0.0
        dst_nodes_raw_features[neighbor_dst_node_ids == 0] = 0.0

        src_nodes_neighbor_latest_time_interval = torch.from_numpy(node_interact_times - np.max(neighbor_src_times, axis=1)).float().to(self.device)
        dst_nodes_neighbor_latest_time_interval = torch.from_numpy(node_interact_times - np.max(neighbor_dst_times, axis=1)).float().to(self.device)
        src_nodes_neighbor_latest_time_interval = torch.exp(- src_nodes_neighbor_latest_time_interval / self.max_time_shift)
        dst_nodes_neighbor_latest_time_interval = torch.exp(- dst_nodes_neighbor_latest_time_interval / self.max_time_shift)

        dst_nodes_neighbor_latest_time_interval = torch.mean(dst_nodes_neighbor_latest_time_interval)
        dst_nodes_neighbor_latest_time_interval = dst_nodes_neighbor_latest_time_interval.repeat(neighbor_dst_node_ids.shape[0])

        # Tensor, shape (batch_size, time_feat_dim)，node self information
        src_node_time_intervals_feat = self.time_projection(src_nodes_neighbor_latest_time_interval.unsqueeze(dim=-1)).unsqueeze(dim=1)
        dst_node_time_intervals_feat = self.time_projection(dst_nodes_neighbor_latest_time_interval.unsqueeze(dim=-1)).unsqueeze(dim=1)
        
        # neighbor feature
        src_neighbor_feature = torch.cat([src_nodes_raw_features, src_neighbor_time_features], dim=-1)
        dst_neighbor_feature = torch.cat([dst_nodes_raw_features, dst_neighbor_time_features], dim=-1)

        src_neighbor_feature = self.edge_projection_layer(src_neighbor_feature)
        dst_neighbor_feature = self.edge_projection_layer(dst_neighbor_feature)

        neighbor_feature = torch.cat([src_neighbor_feature, dst_neighbor_feature], dim=1)
        node_features = torch.cat([src_node_raw_features+src_node_time_intervals_feat, dst_node_raw_features+dst_node_time_intervals_feat], dim=0)
        sample_ratio = self.sampl_encoder(node_src_ids, node_dst_ids, neighbor_src_node_ids, neighbor_dst_node_ids)

        
        return node_features, neighbor_feature, sample_ratio
    
    @staticmethod
    def sampling(att_log_logit, training):
        temp = 1
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
    

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)


class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels,
                                                  dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)

    def forward(self, input_tensor: torch.Tensor, explain_weights=None):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)

        if explain_weights is not None:
            # Tensor, shape (batch_size, num_tokens, num_channels)
            input_tensor = input_tensor * explain_weights.unsqueeze(dim=-1)

        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)

        if explain_weights is not None:
            hidden_tensor = hidden_tensor * explain_weights.unsqueeze(dim=-1)

        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)

        if explain_weights is not None:
            hidden_tensor = hidden_tensor * explain_weights.unsqueeze(dim=-1)

        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor


class SampleEncoder_speed(nn.Module):

    def __init__(self, hidden_dim: int, device: str = 'cpu'):
       
        super(SampleEncoder_speed, self).__init__()

        self.hidden_dim = hidden_dim
        self.device = device

        self.nif_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=1))

        
    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_nodes_neighbor_ids and dst_nodes_neighbor_ids
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param src_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_nodes_neighbor_ids: ndarray, shape (batch_size, dst_max_seq_length)
        :return: sample_score: Tensor, shape (batch_size, max_seq_length, 1)
        """
        nodes_neighbor_ids = torch.cat([src_nodes_neighbor_ids, dst_nodes_neighbor_ids], dim=1)
        
        # Get batch size and sequence length
        batch_size, seq_length = nodes_neighbor_ids.shape
        valid_mask = (nodes_neighbor_ids != 0)

        # Compute maximum node ID for count tensor size
        max_node_id = nodes_neighbor_ids.max().item()
        
        # Compute node counts per batch using scatter_add_
        counts = torch.zeros(batch_size, int(max_node_id) + 1, device=self.device)
        counts.scatter_add_(1, nodes_neighbor_ids, torch.ones_like(nodes_neighbor_ids, dtype=counts.dtype))
        
        # Gather appearance counts for each position
        nodes_appearances = counts.gather(1, nodes_neighbor_ids)
        nodes_appearances = nodes_appearances * valid_mask.to(counts.dtype)
        
        sample_ratio = self.nif_encode_layer(nodes_appearances.unsqueeze(dim=-1)).sum(dim=2).squeeze()
        
        return sample_ratio
    

