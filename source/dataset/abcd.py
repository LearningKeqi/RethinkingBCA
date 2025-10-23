import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler, threshold_adjacency_matrices, threshold_adjacency_matrices_knn
from omegaconf import DictConfig, open_dict
from .process_node_feature import preprocess_nodefeature
from ..utils import draw_single_connectivity
from sklearn.preprocessing import LabelEncoder
import os



def load_abcd_data(cfg: DictConfig):
    ts_data = np.load(cfg.dataset.time_seires, allow_pickle=True)
    print(f'ts_data.shape={ts_data.shape}')

    pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)

    predicted_labels = pd.read_csv(cfg.dataset.comb_measures_path)


    with open(cfg.dataset.node_id, 'r') as f:
        lines = f.readlines()
        pearson_id = [line[:-1] for line in lines]

    with open(cfg.dataset.seires_id, 'r') as f:
        lines = f.readlines()
        ts_id = [line[:-1] for line in lines]

    id2pearson = dict(zip(pearson_id, pearson_data))

    id2label = dict(zip(predicted_labels['subjectkey'], predicted_labels[cfg.dataset.measure]))

     
    final_timeseires, final_label, final_pearson, orig_connection = [], [], [], []

    used_subjectids = []

    for ts, l in zip(ts_data, ts_id):
        if l in id2label and l in id2pearson:
            if cfg.dataset.measure == 'sex':
                if not np.any(np.isnan(id2pearson[l])):
                    final_timeseires.append(ts)
                    final_label.append(id2label[l])
                    final_pearson.append(id2pearson[l])
                    used_subjectids.append(l)
            else:
                if not np.any(np.isnan(id2pearson[l])) and not np.any(np.isnan(id2label[l])):
                    final_timeseires.append(ts)
                    final_label.append(id2label[l])
                    final_pearson.append(id2pearson[l])
                    used_subjectids.append(l)
    

    id_mapping_file = f'./exp_results/double_route/id_mapping/{cfg.dataset.name}_id_mapping.csv'
    
    label_encoder = LabelEncoder()
    encoded_sample_id = label_encoder.fit_transform(used_subjectids)
    used_subjectids = torch.tensor(encoded_sample_id)

    id_mapping = pd.DataFrame({
        'Original ID': label_encoder.classes_,
        'Encoded ID': range(len(label_encoder.classes_))
    })

    id_mapping.to_csv(id_mapping_file, index=False)

    print(f'used_subjectids.shape={used_subjectids.shape}')
    

    labels = final_label
    orig_connection = final_pearson.copy()  # orig_connection record input pearson connectivity

    # updated for new fc type
    if cfg.fc_type != 'pearson':
        orig_connection = load_other_fc(cfg)
        
    orig_connection = final_pearson.copy()  # orig_connection record input pearson connectivity


    if cfg.dataset.measure == 'sex':
        encoder = preprocessing.LabelEncoder()
        encoder.fit(predicted_labels["sex"])
        labels = encoder.transform(final_label)  


    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))


    final_timeseires = scaler.transform(final_timeseires)
    

    
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

        cfg.dataset.node_sz = final_pearson.shape[1]

        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
        cfg.dataset.time_series_input_size = final_timeseires.shape[2]


    if cfg.dataset.binary_sparse:
        sparse_connection[sparse_connection!=0] = 1
        assert torch.all((sparse_connection == 0) | (sparse_connection == 1)), "Tensor should only contain 0 and 1."


    saved_eigenvectors = torch.zeros(orig_connection.shape[0],orig_connection.shape[1],1) # placeholder, useless


    if not cfg.dataset.stratified:
        return final_timeseires, final_pearson, labels, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids
    else:
        return final_timeseires, final_pearson, labels, labels, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids
