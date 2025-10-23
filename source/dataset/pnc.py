import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler, threshold_adjacency_matrices, threshold_adjacency_matrices_knn
from omegaconf import DictConfig, open_dict
from .process_node_feature import preprocess_nodefeature
from sklearn.preprocessing import LabelEncoder



def load_pnc_data(cfg: DictConfig):

    ts_data = np.load(cfg.dataset.time_seires, allow_pickle=True)
    pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)

    label_df = pd.read_csv(cfg.dataset.label)

    pearson_data, timeseries_data = pearson_data.item(), ts_data.item()

    pearson_id = pearson_data['id']
    pearson_data = pearson_data['data']


    for i in range(pearson_data.shape[0]): 
        np.fill_diagonal(pearson_data[i], 1)

    id2pearson = dict(zip(pearson_id, pearson_data))
    

    ts_id = timeseries_data['id']
    timeseries_data = timeseries_data['data']

    id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

    final_timeseires, final_label, final_pearson = [], [], []

    used_subjectids = []

    for fc, l in zip(timeseries_data, ts_id):
        if l in id2gender and l in id2pearson:
            final_timeseires.append(fc)
            final_label.append(id2gender[l])
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



    final_pearson = np.array(final_pearson)

    final_timeseires = np.array(final_timeseires).transpose(0, 2, 1)

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df["sex"])

    labels = encoder.transform(final_label)

    scaler = StandardScaler(mean=np.mean(
        final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson, labels = [np.array(
        data) for data in (final_timeseires, final_pearson, labels)]

    print(f'final timeseries.shape={final_timeseires.shape}')

    orig_connection = final_pearson.copy()
    
    if cfg.fc_type != 'pearson':
        orig_connection = load_other_fc(cfg)

    
    print(f'labels = {labels}')


    print(f'datasize = {labels.shape[0]}')
    
    # construct sparse graph
    if cfg.fc_type != 'pearson':
        sparse_connection = threshold_adjacency_matrices_newfc(torch.from_numpy(orig_connection), cfg.dataset.sparse_ratio)
    else:
        sparse_connection = threshold_adjacency_matrices(torch.from_numpy(orig_connection), cfg.dataset.sparse_ratio, cfg.dataset.only_positive_corr)


    # preprocess pearson matrix for different node feature
    final_pearson = preprocess_nodefeature(cfg, orig_connection, sparse_connection.numpy(), final_timeseires)   # final pearson records the preprocessed node feature


    final_timeseires, final_pearson, labels, orig_connection= [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels, orig_connection)]


    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
        cfg.dataset.num_classes = labels.unique().shape[0]
        cfg.dataset.time_series_input_size = final_timeseires.shape[2]


    if cfg.dataset.binary_sparse:
        sparse_connection[sparse_connection!=0] = 1
        assert torch.all((sparse_connection == 0) | (sparse_connection == 1)), "Tensor should only contain 0 and 1."

    saved_eigenvectors = torch.zeros(orig_connection.shape[0],orig_connection.shape[1],1) # placeholder, useless


    if not cfg.dataset.stratified:
        return final_timeseires, final_pearson, labels, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids
    else:
        return final_timeseires, final_pearson, labels, labels, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids

