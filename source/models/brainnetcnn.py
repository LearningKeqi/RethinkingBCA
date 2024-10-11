import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from .base import BaseModel


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.in_planes = 1
        self.d = config.dataset.node_sz
        self.config = config

        self.e2econv1 = E2EBlock(1, config.model.dim1, config.dataset.node_sz, bias=True)
        self.e2econv2 = E2EBlock(config.model.dim1, config.model.dim2, config.dataset.node_sz, bias=True)
        self.E2N = torch.nn.Conv2d(config.model.dim2, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)

        output_dim = 2 if config.dataset.task == 'classification' else 1
        self.dense3 = torch.nn.Linear(30, output_dim) 

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):
        node_feature = node_feature.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(node_feature), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=self.config.model.dropout_rate)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=self.config.model.dropout_rate)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=self.config.model.dropout_rate)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out
