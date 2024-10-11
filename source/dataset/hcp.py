import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler, threshold_adjacency_matrices, threshold_adjacency_matrices_knn
from omegaconf import DictConfig, open_dict
from .process_node_feature import preprocess_nodefeature
from sklearn.preprocessing import LabelEncoder


def load_hcp_data(cfg: DictConfig):
    data = np.load(cfg.dataset.pearson_path, allow_pickle=True).item()

    pearson_data = data["corr"]    # original diagonal values are nan

    ts_data = np.load(cfg.dataset.time_series_path, allow_pickle=True).item()
    ts_id = ts_data['id']
    ts_ts = ts_data['time_series']
    id2ts = dict(zip(ts_id, ts_ts))

    # convert fisher transformation back to pearson correlation matrix
    for i in range(pearson_data.shape[0]): 
        np.fill_diagonal(pearson_data[i], np.inf)
    
    pearson_data = np.tanh(pearson_data)
    
    print(f'pearson_data.shape[0]={pearson_data.shape[0]}')

    id = data['id']
 
    id2pearson = dict(zip(id, pearson_data))

    predicted_labels = pd.read_csv(cfg.dataset.comb_measures_path)

    id2label = dict(zip(predicted_labels['Subject'], predicted_labels[cfg.dataset.measure]))

    final_timeseires, final_label, final_pearson, orig_connection = [], [], [], []

    used_subjectids = []

    age_class_dict = {'22-25':0, '26-30':1, '31-35':2}

    temp_cnt = 0

    for l in id:
        if l in id2label and l in id2pearson and l in id2ts:
            temp_cnt+=1 
            if cfg.dataset.measure == 'Gender':
                if not np.any(np.isnan(id2pearson[l])):
                    final_label.append(id2label[l])
                    final_pearson.append(id2pearson[l])
                    final_timeseires.append(id2ts[l])
                    used_subjectids.append(l)
            elif cfg.dataset.measure == 'Age':
                final_label.append(age_class_dict[id2label[l]])
                final_pearson.append(id2pearson[l])
                final_timeseires.append(id2ts[l])
                used_subjectids.append(l)
            else:
                if not np.any(np.isnan(id2pearson[l])) and not np.any(np.isnan(id2label[l])):
                    final_label.append(id2label[l])
                    final_pearson.append(id2pearson[l])
                    final_timeseires.append(id2ts[l])
                    used_subjectids.append(l)
    
    print(f'temp_cnt = {temp_cnt}')

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
    
    if cfg.dataset.measure == 'Gender':
        encoder = preprocessing.LabelEncoder()
        encoder.fit(predicted_labels["Gender"])
        labels = encoder.transform(final_label)  
    
    
    scaler = StandardScaler(mean=np.mean(
    final_timeseires), std=np.std(final_timeseires))

    final_timeseires = scaler.transform(final_timeseires)

    final_pearson, final_timeseires,labels, orig_connection = [np.array(
        data) for data in (final_pearson, final_timeseires, labels, orig_connection)]
    

    print(f'labels = {labels}')

    print(f'datasize = {labels.shape[0]}')


    # construct sparse graph
    sparse_connection = threshold_adjacency_matrices(torch.from_numpy(orig_connection), cfg.dataset.sparse_ratio, cfg.dataset.only_positive_corr)


    if cfg.dataset.timeseries_used_length != -1:
        final_timeseires = final_timeseires[:,:,:cfg.dataset.timeseries_used_length]

    print(f'final_time_series.shape={final_timeseires.shape}')

    # preprocess pearson matrix for different node feature
    final_pearson = preprocess_nodefeature(cfg, orig_connection, sparse_connection.numpy(), final_timeseires)   # final pearson records the preprocessed node feature


    final_pearson, final_timeseires, labels, orig_connection = [torch.from_numpy(
        data).float() for data in (final_pearson, final_timeseires, labels, orig_connection)]


    with open_dict(cfg):

        cfg.dataset.node_sz = final_pearson.shape[1]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
        cfg.dataset.time_series_input_size = final_timeseires.shape[2]
    

    if cfg.dataset.binary_sparse:
        sparse_connection[sparse_connection!=0] = 1
        assert torch.all((sparse_connection == 0) | (sparse_connection == 1)), "Tensor should only contain 0 and 1."

    saved_eigenvectors = torch.zeros(orig_connection.shape[0],orig_connection.shape[1],1) # placeholder, useless


    if cfg.dataset.stratified:
        return final_timeseires, final_pearson, labels, labels, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids
    else:
        return final_timeseires, final_pearson, labels, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids


