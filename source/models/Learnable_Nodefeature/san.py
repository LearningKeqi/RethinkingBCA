import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SAN_Nodefeature(nn.Module):

    def __init__(self, m, k):
        super(SAN_Nodefeature, self).__init__()
        self.m = m   # sequence length
        self.k = k   # pe_dim
        
        self.linear_A = nn.Linear(2, self.k)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.k, nhead=2)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        

    def forward(self, eigen_info):
        EigVecs = eigen_info[:,:,:self.m]  # bat_sz x num_nodes x m
        EigVals = eigen_info[:,:,-1] # bat_sz x num_nodes
        EigVals = EigVals[:,:self.m] # bat_sz x m
        
        num_nodes = EigVecs.shape[1]
        EigVals_expanded = EigVals.unsqueeze(1).expand(-1, num_nodes, -1)  # [bat_sz, num_nodes, m]
        EigVals_expanded = EigVals_expanded.unsqueeze(-1)  # [bat_sz, num_nodes, m, 1]
        EigVecs_expanded = EigVecs.unsqueeze(-1)  # [bat_sz, num_nodes, m, 1]

        # combine vec with val
        PosEnc = torch.cat((EigVecs_expanded, EigVals_expanded), dim=-1)  # [bat_sz, num_nodes, m, 2]
        
        # PosEnc = torch.transpose(PosEnc, 1 ,2).float() # (bat_sz) x (Num Eigenvectors) x (Num nodes) x 2
        PosEnc = self.linear_A(PosEnc) # [bat_sz, num_nodes, m, k]
        
        PosEnc = PosEnc.permute(2, 0, 1, 3)  # [m, bat_sz, num_nodes, k]
        
        m, bat_sz, num_nodes, k = PosEnc.shape

        PosEnc = PosEnc.reshape(m, bat_sz * num_nodes, k)  

        #1st Transformer: Learned PE
        PosEnc = self.PE_Transformer(src=PosEnc)  # [m, bat_sz * num_nodes, k]

        PosEnc = PosEnc.reshape(m, bat_sz, num_nodes, k)

        #Sum pooling
        PosEnc = PosEnc.sum(dim=0)  #[bat_sz, num_nodes, k]
        
        return PosEnc