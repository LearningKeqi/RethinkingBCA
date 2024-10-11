import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing
from torch.nn import Parameter
import numpy as np
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot, zeros
from typing import Tuple
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch import nn
import torch_geometric
import math


class MPGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, edge_emb_dim: int, gcn_mp_type: str, bucket_sz: float, 
                 normalize: bool = True, bias: bool = True):
        super(MPGCNConv, self).__init__(in_channels=in_channels, out_channels=out_channels, aggr='add')

        self.edge_emb_dim = edge_emb_dim
        self.gcn_mp_type = gcn_mp_type
        self.bucket_sz = bucket_sz
        self.bucket_num = math.ceil(2.0/self.bucket_sz)
        if gcn_mp_type == "bin_concate":
            self.edge2vec = nn.Embedding(self.bucket_num, edge_emb_dim)

        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        
        input_dim = out_channels
        if gcn_mp_type == "bin_concate" or gcn_mp_type == "edge_weight_concate":
            input_dim = out_channels + edge_emb_dim
        elif gcn_mp_type == "edge_node_concate":
            input_dim = out_channels*2 + 1
        elif gcn_mp_type == "node_concate":
            input_dim = out_channels*2
        self.edge_lin = torch.nn.Linear(input_dim, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def message(self, x_i, x_j, edge_weight):
        # x_j: [E, in_channels]
        if self.gcn_mp_type == "weighted_sum": 
            # use edge_weight as multiplier
            msg = edge_weight.view(-1, 1) * x_j
        elif self.gcn_mp_type == "bin_concate":
            # concat xj and learned bin embedding
            bucket = torch.div(edge_weight + 1, self.bucket_sz, rounding_mode='trunc').int()
            msg = torch.cat([x_j, self.edge2vec(bucket)], dim=1)
            msg = self.edge_lin(msg) 
        elif self.gcn_mp_type == "edge_weight_concate":
            # concat xj and tiled edge attr
            msg = torch.cat([x_j, edge_weight.view(-1, 1).repeat(1, self.edge_emb_dim)], dim=1)
            msg = self.edge_lin(msg) 
        elif self.gcn_mp_type == "edge_node_concate": 
            # concat xi, xj and edge_weight
            msg = torch.cat([x_i, x_j, edge_weight.view(-1, 1)], dim=1)
            msg = self.edge_lin(msg)
        elif self.gcn_mp_type == "node_concate":
            # concat xi and xj
            msg = torch.cat([x_i, x_j], dim=1)
            msg = self.edge_lin(msg)
        else:
            raise ValueError(f'Invalid message passing variant {self.gcn_mp_type}')
        return msg

        
class BrainGB(torch.nn.Module):
    def __init__(self, config):
        super(BrainGB, self).__init__()

        print(f'Using BrainGB model..')
        self.cfg = config
        self.activation = torch.nn.ReLU()
        self.convs = torch.nn.ModuleList()
        self.pooling = self.cfg.model.pooling
        self.num_nodes = self.cfg.dataset.node_sz

        hidden_dim = self.cfg.model.hidden_dim
        num_layers = self.cfg.model.n_GNN_layers
        edge_emb_dim = self.cfg.model.edge_emb_dim
        gcn_mp_type = self.cfg.model.gcn_mp_type
        bucket_sz = self.cfg.model.bucket_sz
        gcn_input_dim = self.cfg.dataset.node_feature_dim

        for i in range(num_layers-1):
            conv = torch_geometric.nn.Sequential('x, edge_index, edge_attr', [
                (MPGCNConv(gcn_input_dim, hidden_dim, edge_emb_dim, gcn_mp_type, bucket_sz, normalize=True, bias=True),'x, edge_index, edge_attr -> x'),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            gcn_input_dim = hidden_dim
            self.convs.append(conv)

        input_dim = 0

        if self.pooling == "concat":
            node_dim = 8
            conv = torch_geometric.nn.Sequential('x, edge_index, edge_attr', [
                (MPGCNConv(hidden_dim, hidden_dim, edge_emb_dim, gcn_mp_type, bucket_sz, normalize=True, bias=True),'x, edge_index, edge_attr -> x'),
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(64, node_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(node_dim)
            ])
            input_dim = node_dim*self.num_nodes

        elif self.pooling == 'sum' or self.pooling == 'mean':
            node_dim = 256
            # input_dim = node_dim  # modify previous one may not correct?
            input_dim = hidden_dim
            conv = torch_geometric.nn.Sequential('x, edge_index, edge_attr', [
                (MPGCNConv(hidden_dim, hidden_dim, edge_emb_dim, gcn_mp_type, bucket_sz, normalize=True, bias=True),'x, edge_index, edge_attr -> x'),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.2),
                # nn.BatchNorm1d(node_dim)
                nn.BatchNorm1d(hidden_dim)  # modify previous one may not correct?
            ])

        self.convs.append(conv)

        fc_output_dim = 2 if self.cfg.dataset.task == 'classification' else 1

        self.fcn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            # nn.Linear(32, 2),
            nn.Linear(32, fc_output_dim)        ### modify for cognition score
        )


    def forward(self, m, node_feature):
        x, edge_index, batch, edge_attr, _ = self.transform_data(m, node_feature)

        z = x
        edge_attr = torch.abs(edge_attr)
   
        for i, conv in enumerate(self.convs):
            z = conv(z, edge_index, edge_attr)
            
        if self.pooling == "concat":
            z = z.reshape((z.shape[0]//self.num_nodes, -1))
        elif self.pooling == 'sum':
            z = global_add_pool(z,  batch)  # [N, F]
        elif self.pooling == 'mean':
            z = global_mean_pool(z, batch)  # [N, F]

        out = self.fcn(z)  
        return out

        

    def transform_data(self, m, node_feature):
        '''
        Transform input data 'm' and 'node_feature' into the forms needed by Braingnn model.
        Input: 
                'm' - adjacency matrix generated before. [batch_size, num_nodes, num_nodes]   
                'node_feature' - node feature generated before. [batch_size, num_nodes, node_feature]
        Output: 
                'x' - node feature. [batch_num_nodes, node_feature]
                'edge_index' - each column represents an edge. [2, batch_num_edges]
                'batch' - a column vector which maps each node to its respective graph in the batch. [batch_num_nodes, 1]
                'edge_attr' - edge weights. [batch_num_edges, 1]
                'pos' - one-hot regional information. Its ROI representation ri is a N-dimensional vector with 1 in the i th entry and 0 for the other entries. [batch_num_nodes, num_nodes]

        '''

        ## handling x
        x = node_feature.view(-1, node_feature.size(2))

        ## handling edge_index and edge_attr
        bz = m.shape[0]
        num_nodes = m.shape[1]
        all_edge_indices = []
        all_edge_weights = []
        
        for b in range(bz):
            row, col = torch.where(m[b] != 0)            
            row += b * num_nodes
            col += b * num_nodes

            all_edge_indices.append(torch.stack([row, col], dim=0))
            all_edge_weights.append(m[b, m[b] != 0])

        edge_index = torch.cat(all_edge_indices, dim=1)
        edge_attr = torch.cat(all_edge_weights).float()

        ## handling batch 
        batch = torch.arange(bz).repeat_interleave(num_nodes).view(-1, 1).squeeze()

        ## handling pos
        pos = torch.eye(num_nodes).repeat(bz, 1)

        return x.cuda(), edge_index.cuda(), batch.cuda(), edge_attr.cuda(), pos.cuda()
