import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Learnable_Nodefeature import Centrality_Nodefeature, SAN_Nodefeature, GCN_Nodefeature, DegreeBin_Nodefeature, TimeSeriesEncoder
from omegaconf import DictConfig, open_dict



class MLP_Node(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super(MLP_Node, self).__init__()
        print(f'Using MLP Node model')
        
        self.cfg = config

        self.mlp_list = nn.ModuleList()
        # self.norms = nn.ModuleList()
        self.dropout= nn.Dropout(0.2)
        for i in range(self.cfg.model.num_mlp_layer):
            self.mlp_list.append(nn.Linear(self.cfg.dataset.node_sz, self.cfg.dataset.node_sz))
            # self.norms.append(nn.LayerNorm(self.cfg.dataset.node_sz))


        if self.cfg.dataset.task == 'classification':
            last_output_dim = 2
        elif self.cfg.dataset.task == 'regression':
            last_output_dim = 1

        fc_input_dim = self.cfg.dataset.node_sz * self.cfg.dataset.node_sz

        self.fc = nn.Sequential(nn.Linear(fc_input_dim, last_output_dim, bias=True))


    def forward(self, time_series, node_feature):
        mlp_feature = node_feature.clone()
        for idx, mlp_layer in enumerate(self.mlp_list):
            mlp_feature = mlp_layer(mlp_feature)
            # mlp_feature = self.norms[idx](mlp_feature)
            mlp_feature = F.relu(mlp_feature)
            mlp_feature = self.dropout(mlp_feature)

        # flatten
        bz = mlp_feature.shape[0]
        mlp_feature = mlp_feature.reshape((bz, -1))

        return self.fc(mlp_feature)
