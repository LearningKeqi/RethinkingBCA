import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Learnable_Nodefeature import Centrality_Nodefeature, SAN_Nodefeature, GCN_Nodefeature, DegreeBin_Nodefeature, TimeSeriesEncoder
from omegaconf import DictConfig, open_dict


class MLP_Graph(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(MLP_Graph, self).__init__()
        print(f'Using MLP_graph model')
        
        self.cfg = config

        if self.cfg.dataset.task == 'classification':
            last_output_dim = 2
        elif self.cfg.dataset.task == 'regression':
            last_output_dim = 1

        num_nodes = self.cfg.dataset.node_sz
        if self.cfg.has_self_loop:
            fc_input_dim = int((num_nodes*(num_nodes+1))/2) 
        else:
            fc_input_dim = int((num_nodes*(num_nodes-1))/2) # do not take diagonal into consideration

        if self.cfg.model.num_mlp_layer != 1:
            self.fc = nn.Sequential(
                    nn.Linear(fc_input_dim, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 32),
                    nn.LeakyReLU(), 
                    nn.Linear(32, last_output_dim)
                )
        else:
            self.fc = nn.Sequential(nn.Linear(fc_input_dim, last_output_dim))


    def forward(self, time_series, node_feature):
        bz, num_nodes, _ = node_feature.shape

        offset = 0 if self.cfg.has_self_loop else 1
        rows, cols = torch.triu_indices(row=num_nodes, col=num_nodes, offset=offset)  # offset 0 including diag
        mlp_feature = node_feature[:, rows, cols]

        return self.fc(mlp_feature)
    