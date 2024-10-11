import torch
from torch.nn import Linear
from torch import nn
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, SGConv, GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import random
import numpy as np
softmax = torch.nn.LogSoftmax(dim=1)



class Neurograph(torch.nn.Module):
    def __init__(self, config):
        super(Neurograph, self).__init__()
        self.cfg = config

        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        hidden_channels = self.cfg.model.hidden_channels
        num_features = self.cfg.dataset.node_sz
        num_layers = self.cfg.model.num_layers
        hidden = self.cfg.model.hidden

        if num_layers>0:
            self.convs.append(GCNConv(num_features, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_features * num_features)/2)- (num_features/2)+(hidden_channels*num_layers))
        input_dim = int(((num_features * num_features)/2)- (num_features/2))
        self.bn = nn.BatchNorm1d(input_dim)

        if self.cfg.model.has_mid_layer:
            self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        else:
            self.bnh = nn.BatchNorm1d(hidden_channels)

        output_dim = 2 if self.cfg.dataset.task == 'classification' else 1

        if not self.cfg.model.has_residual:
            if self.cfg.model.has_mid_layer:
                input_dim1 = int(hidden_channels*num_layers)
            else:
                input_dim1 = hidden_channels

            
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), output_dim),
        )

    def forward(self, m, node_feature):
        num_graphs, num_nodes, _ = m.shape

        x, edge_index, batch = self.transform_data(m, node_feature)

        # print(f'edge_index.len={edge_index.shape}')

        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(num_graphs, x.shape[1],-1)
                # x = torch.stack([t.triu(diagonal=1).flatten()[t.triu(diagonal=1).flatten().nonzero(as_tuple=True)] for t in xx])
                offset = 0 if self.cfg.has_self_loop else 1
                rows, cols = torch.triu_indices(row=num_nodes, col=num_nodes, offset=offset)  # offset 0 including diag
                x = xx[:, rows, cols]

                x = self.bn(x)
            else:
                xx = self.aggr(xx,batch)
                h.append(xx)
        
        if self.cfg.model.has_mid_layer:
            h = torch.cat(h,dim=1)   
        else:
            h = h[-1]
        
        h = self.bnh(h)


        if self.cfg.model.has_residual:
            x = torch.cat((x,h),dim=1)
        else:
            x = h
            
        x = self.mlp(x)
        
        return x


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
        # all_edge_weights = []
        
        for b in range(bz):
            row, col = torch.where(m[b] != 0)
            row += b * num_nodes
            col += b * num_nodes

            all_edge_indices.append(torch.stack([row, col], dim=0))
            # all_edge_weights.append(m[b, m[b] != 0])

        edge_index = torch.cat(all_edge_indices, dim=1)
        # edge_attr = torch.cat(all_edge_weights)

        ## handling batch 
        batch = torch.arange(bz).repeat_interleave(num_nodes).view(-1, 1).squeeze()

        return x.cuda(), edge_index.cuda(), batch.cuda()
