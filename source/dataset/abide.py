import numpy as np
import torch
from .preprocess import StandardScaler, threshold_adjacency_matrices, threshold_adjacency_matrices_knn
from omegaconf import DictConfig, open_dict
from .process_node_feature import preprocess_nodefeature



def load_abide_data(cfg: DictConfig):

    data = np.load(cfg.dataset.path, allow_pickle=True).item()

    final_timeseires = data["timeseires"]
    print(f'final_timeseires.shape={final_timeseires.shape}')

    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']

    # convert fisher transformation back to pearson correlation matrix
    for i in range(final_pearson.shape[0]): 
        np.fill_diagonal(final_pearson[i], np.inf)
    
    final_pearson = np.tanh(final_pearson)

    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)
    
    orig_connection = final_pearson.copy()  # orig_connection record input pearson connectivity

    if cfg.fc_type != 'pearson':
        print(f'loading new fc type: {cfg.fc_type}')
        orig_connection = load_other_fc(cfg)



    final_timeseires, final_pearson, labels, orig_connection = [np.array(
        data) for data in (final_timeseires, final_pearson, labels, orig_connection)]
    
    
    print(f'labels = {labels}')

    print(f'datasize = {labels.shape[0]}')
    
    # construct sparse graph
    if cfg.fc_type != 'pearson':
        sparse_connection = threshold_adjacency_matrices_newfc(torch.from_numpy(orig_connection), cfg.dataset.sparse_ratio)
    else:
        sparse_connection = threshold_adjacency_matrices(torch.from_numpy(orig_connection), cfg.dataset.sparse_ratio, cfg.dataset.only_positive_corr)

    
    # preprocess pearson matrix for different node feature
    final_pearson = preprocess_nodefeature(cfg, orig_connection, sparse_connection.numpy(), final_timeseires)   # final pearson records the preprocessed node feature


    final_timeseires, final_pearson, labels, orig_connection = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels, orig_connection)]


    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
        cfg.dataset.time_series_input_size = final_timeseires.shape[2]

    
    if cfg.dataset.binary_sparse:
        sparse_connection[sparse_connection!=0] = 1
        assert torch.all((sparse_connection == 0) | (sparse_connection == 1)), "Tensor should only contain 0 and 1."

    num_samples = final_pearson.shape[0]
    used_subjectids = torch.tensor([i for i in range(num_samples)])

    saved_eigenvectors = torch.zeros(orig_connection.shape[0],orig_connection.shape[1],1) # placeholder, useless


    return final_timeseires, final_pearson, labels, site, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids

