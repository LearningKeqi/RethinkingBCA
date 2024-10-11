import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm
from omegaconf import DictConfig


import sys
import inspect
# from torch_geometric.utils import scatter_
from torch_scatter import scatter,scatter_add

from torch.nn import Parameter
from torch_geometric.utils import add_remaining_self_loops,softmax

from torch_geometric.typing import (OptTensor)

import math


"""
Extracted model of BrainGNN

BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis

"""

class BrainGNN_orig(torch.nn.Module):
    def __init__(self, config: DictConfig):
    # def __init__(self, config: DictConfig, indim, nclass, ratio=0.5, k=8, R=116):
        '''
        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(BrainGNN_orig, self).__init__()

        self.cfg = config
        self.indim = self.cfg.dataset.node_feature_dim 
        self.dim1 = self.cfg.model.dim1
        self.dim2 = self.cfg.model.dim2
        self.dim3 = self.cfg.model.dim3
        self.dim4 = self.cfg.model.dim4
        self.dim5 = self.cfg.model.dim5
        self.k = self.cfg.model.k
        self.R = self.cfg.dataset.node_feature_dim
        

        self.n1 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim1 * self.indim))
        self.conv1 = MyNNConv(self.indim, self.dim1, self.n1, normalize=False)
        self.pool1 = TopKPooling(self.dim1, ratio=self.cfg.model.ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.n2 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim2 * self.dim1))
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
        self.pool2 = TopKPooling(self.dim2, ratio=self.cfg.model.ratio, multiplier=1, nonlinearity=torch.sigmoid)

        #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
        self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)
        self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn2 = torch.nn.BatchNorm1d(self.dim3)

        output_dim = 2 if self.cfg.dataset.task == 'classification' else 1

        self.fc3 = torch.nn.Linear(self.dim3, output_dim)



    def forward(self, m, node_feature):
        '''
        m: adjacency matrix   (bz, num_nodes, num_nodes)
        node_feature: node feature  (bz, num_nodes, feature_dim)
        '''

        x, edge_index, batch, edge_attr, pos = self.transform_data(m, node_feature)

        # logging.info('here 1')
        # print(f'x.dtype={x.dtype}, edge_index.dtype={edge_index.dtype}, edge_attr.dtype={edge_attr.dtype}, pos.dtype={pos.dtype}')
        x = self.conv1(x, edge_index, edge_attr, pos)

        # logging.info('here 2')

        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)

        # logging.info('here 3')

        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # logging.info('here 10')

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        # logging.info('here 4')
        
        # print(f'x.dtype={x.dtype}, edge_index.dtype={edge_index.dtype}, edge_attr.dtype={edge_attr.dtype}, pos.dtype={pos.dtype}')
        x = x.float()
        x = self.conv2(x, edge_index, edge_attr, pos)
        # logging.info('here 5')

        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)
        # logging.info('here 6')

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # logging.info('here 7')

        x = torch.cat([x1,x2], dim=1).float()
        
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x= F.dropout(x, p=0.5, training=self.training)
        # x = F.log_softmax(self.fc3(x), dim=-1)
        x = self.fc3(x)
        # logging.info('here 8')

        return x,self.pool1.select.weight,self.pool2.select.weight, torch.sigmoid(score1).view(x.size(0),-1), torch.sigmoid(score2).view(x.size(0),-1)


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        # edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
        #                                          num_nodes=num_nodes)
        # logging.info(f'augment 0')

        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        # logging.info(f'augment 1')
    
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        # logging.info(f'augment 2')
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # logging.info(f'augment 3')
        return edge_index, edge_weight
    


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
        edge_attr = torch.cat(all_edge_weights)

        ## handling batch 
        batch = torch.arange(bz).repeat_interleave(num_nodes).view(-1, 1).squeeze()

        ## handling pos
        pos = torch.eye(num_nodes).repeat(bz, 1)

        return x.cuda(), edge_index.cuda(), batch.cuda(), edge_attr.cuda(), pos.cuda()
    

    def braingnn_loss(self, output, w1, w2, s1, s2, label):
        # print(f'output.dtype={output.dtype},w1.dtype={w1.dtype}, w2.dtype={w2.dtype}, s1.dtype={s1.dtype}, s2.dtype={s2.dtype}, label.dtype={label.dtype}')
        s1, s2 = s1.float(), s2.float()

        if self.cfg.dataset.task == 'classification':
            major_loss = torch.nn.CrossEntropyLoss()(output,label)
        else:
            major_loss = torch.nn.MSELoss()(output, label)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,self.cfg.model.ratio)
        loss_tpk2 = topk_loss(s2,self.cfg.model.ratio)
        loss_consist = 0

        if self.cfg.dataset.task == 'classification':
            for c in range(2):
                loss_consist += consist_loss(s1[label[:,1] == c])

        loss = self.cfg.model.lamb0*major_loss + self.cfg.model.lamb1 * loss_p1 + self.cfg.model.lamb2 * loss_p2 \
                    + self.cfg.model.lamb3 * loss_tpk1 + self.cfg.model.lamb4 *loss_tpk2 + self.cfg.model.lamb5* loss_consist
        
        return loss
    




# ============================ Training Loss ============================
'''
output, w1, w2, s1, s2 = model(m, node_feature)
loss_c = F.nll_loss(output, data.y)

loss_p1 = (torch.norm(w1, p=2)-1) ** 2
loss_p2 = (torch.norm(w2, p=2)-1) ** 2
loss_tpk1 = topk_loss(s1,opt.ratio)
loss_tpk2 = topk_loss(s2,opt.ratio)
loss_consist = 0
for c in range(opt.nclass):
    loss_consist += consist_loss(s1[data.y == c])
loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
            + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist

loss.backward()
optimizer.step()
'''

def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.cuda()
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res


# ============================ Dependencies ============================

special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec


# ---------------------------- MyMessagePassing ----------------------------

class MyMessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers
    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),
    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.
    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`0`)
    """
    def __init__(self, aggr='add', flow='source_to_target', node_dim=0):
        super(MyMessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__message_args__ = getargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = getargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.
        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferred and assumed to be symmetric.
                (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """

        dim = self.node_dim
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(dim)
                            if size[1 - idx] != tmp[1 - idx].size(dim):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if tmp is None:
                        message_args.append(tmp)
                    else:
                        if size[idx] is None:
                            size[idx] = tmp.size(dim)
                        if size[idx] != tmp.size(dim):
                            raise ValueError(__size_error_msg__)

                        tmp = torch.index_select(tmp, dim, edge_index[idx])
                        message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        out = self.message(*message_args)
        # out = scatter_(self.aggr, out, edge_index[i], dim, dim_size=size[i])
        out = scatter_add(out, edge_index[i], dim, dim_size=size[i])
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages to node :math:`i` in analogy to
        :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and
        :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out



# ---------------------------- MyNNConv ----------------------------

class MyNNConv(MyMessagePassing):
    def __init__(self, in_channels, out_channels, nn, normalize=False, bias=True,
                 **kwargs):
        super(MyNNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.nn = nn
        #self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
#        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_weight=None, pseudo= None, size=None):
        """"""
        edge_weight = edge_weight.squeeze()
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(0))

        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        if torch.is_tensor(x):
            x = torch.matmul(x.unsqueeze(1), weight).squeeze(1)
        else:
            x = (None if x[0] is None else torch.matmul(x[0].unsqueeze(1), weight).squeeze(1),
                 None if x[1] is None else torch.matmul(x[1].unsqueeze(1), weight).squeeze(1))

        # weight = self.nn(pseudo).view(-1, self.out_channels,self.in_channels)
        # if torch.is_tensor(x):
        #     x = torch.matmul(x.unsqueeze(1), weight.permute(0,2,1)).squeeze(1)
        # else:
        #     x = (None if x[0] is None else torch.matmul(x[0].unsqueeze(1), weight).squeeze(1),
        #          None if x[1] is None else torch.matmul(x[1].unsqueeze(1), weight).squeeze(1))

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight)

    def message(self, edge_index_i, size_i, x_j, edge_weight, ptr: OptTensor):
        edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)




''''
Train and Evaluate
'''

import numpy as np
import torch
import pandas as pd
import os


import numpy as np
import matplotlib.pyplot as plt 
import torch.utils.data as utils
from torch.utils.data import Subset
import torch.optim as optim
import logging
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import wandb
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def save_indices(dataset_name, train_index, valid_index, test_index):
    len_train = len(train_index)
    len_valid = len(valid_index)
    len_test = len(test_index)

    max_len = max(len_train, len_valid, len_test)

    if len_train < max_len:
        train_index = np.append(train_index, [None] * (max_len - len_train))
    
    if len_valid < max_len:
        valid_index = np.append(valid_index, [None] * (max_len - len_valid))

    if len_test < max_len:
        test_index = np.append(test_index, [None] * (max_len - len_test))

    ### save split indices 
    df = pd.DataFrame({
        'Train Index': train_index,
        'Valid Index': valid_index,
        'Test Index': test_index
    })

    if dataset_name == 'adhd_e2e':
        file_path = '/local/scratch3/khan58/BrainGB/BrainGNN_e2e/adhd_e2e_indices.csv'
    elif dataset_name == 'abide_e2e':
        file_path = '/local/scratch3/khan58/BrainGB/BrainGNN_e2e/abide_e2e_indices.csv'

    print(f'file_path={file_path}')

    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        print(f"File {file_path} already exists, skipping saving.")


def load_data(train_data_path, val_data_path, test_data_path, dataset_name):
    train_data = np.load(train_data_path)
    val_data = np.load(val_data_path)
    test_data = np.load(test_data_path)

    train_len, val_len, test_len = len(train_data), len(val_data), len(test_data)

    labels = []
    adjs = []

    train_index = np.arange(train_len)
    val_index = np.arange(val_len) + train_len
    test_index = np.arange(test_len) + train_len + val_len
    save_indices(dataset_name, train_index, val_index, test_index)

    for cur_data in train_data:
        id,label,ext_img,reg_img,roi_means,adjacency_matrix = cur_data

        labels.append(label)
        adjs.append(adjacency_matrix)

    for cur_data in val_data:
        id,label,ext_img,reg_img,roi_means,adjacency_matrix = cur_data

        labels.append(label)
        adjs.append(adjacency_matrix)

    for cur_data in test_data:
        id,label,ext_img,reg_img,roi_means,adjacency_matrix = cur_data

        labels.append(label)
        adjs.append(adjacency_matrix)

    labels = np.array(labels)
    adjs = np.array(adjs)

    adjs, labels = [torch.from_numpy(data).float() for data in (adjs, labels)]

    print(f'labels.shape={labels.shape}')
    print(f'adjs.shape={adjs.shape}')

    return adjs, labels


def init_dataloader(final_pearson,
                    labels, train_batch_size, val_batch_size, test_batch_size, drop_last, dataset_name):
   
    labels = F.one_hot(labels.to(torch.int64))    # label is 2-D


    print(f'final_pearson.shape = {final_pearson.shape}, labels.shape = {labels.shape}')
    dataset = utils.TensorDataset(
        final_pearson,
        labels
    )

    if dataset_name == 'adhd_e2e':
        load_path = '/local/scratch3/khan58/BrainGB/BrainGNN_e2e/adhd_e2e_indices.csv'
    elif dataset_name == 'abide_e2e':
        load_path = '/local/scratch3/khan58/BrainGB/BrainGNN_e2e/abide_e2e_indices.csv'
    
    indices = pd.read_csv(load_path)
    
    print(f'load_path={load_path}')

    train_indices = indices['Train Index'].dropna().to_numpy().astype(int)
    valid_indices = indices['Valid Index'].dropna().to_numpy().astype(int)
    test_indices = indices['Test Index'].dropna().to_numpy().astype(int)

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f'len dataset = {len(dataset)}')
    print(f"len train_dataset={len(train_dataset)},len valid_dataset={len(valid_dataset)},len test_dataset={len(test_dataset)}")

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=drop_last)

    val_dataloader = utils.DataLoader(
        valid_dataset, batch_size=val_batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def continus_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    # y = lam * y + (1-lam) * y[index]
    return *new_xs, y, y[index], lam

from sklearn.metrics import roc_auc_score, accuracy_score

def test_per_epoch(model, dataloader):
    model.eval()

    labels = []
    results = []

    total_time = 0.0
    num_samples = 0

    for node_feature, label in dataloader:
        node_feature, label = node_feature.cuda(), label.cuda()
        m = node_feature

        start_time = time.time() 

        output, w1, w2, s1, s2 = model(m, node_feature)

        end_time = time.time()

        total_time += (end_time - start_time)
        num_samples += node_feature.shape[0]

        label = label.float()
                
        results += F.softmax(output, dim=1)[:, 1].tolist()
        labels += label[:, 1].tolist()
    
    # logging.info(f'total_time={total_time}, num_samples={num_samples}, avg_inference_time={total_time/num_samples}')

    auc = roc_auc_score(labels, results)
    results, labels = np.array(results), np.array(labels)
    results[results > 0.5] = 1
    results[results <= 0.5] = 0
    
    acc = accuracy_score(labels, results)

    return auc, acc


EPS = 1e-10











    