import numpy as np
import pandas as pd




def load_abide_data():
    dataset_path = '/local/scratch3/khan58/BrainGB/examples/datasets/ABIDE/abide.npy'
    data = np.load(dataset_path, allow_pickle=True).item()

    final_timeseries = data["timeseires"]
    print(f'final_timeseries.shape={final_timeseries.shape}')

    eps=1e-8

    mean = final_timeseries.mean(axis=(1, 2), keepdims=True)     # (N,1,1)

    std  = final_timeseries.std(axis=(1, 2), ddof=0, keepdims=True)


    normed = (final_timeseries - mean) / (std + eps)

    print(f'normed.shape={normed.shape}')

    return normed



def load_hcp_data():

    data = np.load('/local/scratch3/khan58/Datasets/HCP_data_release/HCP_correlation_matrix.npy', allow_pickle=True).item()

    pearson_data = data["corr"]    # original diagonal values are nan

    ts_data = np.load('/local/scratch3/khan58/Datasets/HCP_data_release/hcp_timeseries_data.npy', allow_pickle=True).item()
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

    predicted_labels = pd.read_csv('/local/scratch3/khan58/Datasets/HCP_data_release/HCP_comb_measures.csv')

    id2label = dict(zip(predicted_labels['Subject'], predicted_labels['PMAT24_A_CR']))

    final_timeseries, final_label, final_pearson, orig_connection = [], [], [], []

    used_subjectids = []

    temp_cnt = 0

    for l in id:
        if l in id2label and l in id2pearson and l in id2ts:
            temp_cnt+=1 
            if not np.any(np.isnan(id2pearson[l])) and not np.any(np.isnan(id2label[l])):
                final_label.append(id2label[l])
                final_pearson.append(id2pearson[l])
                final_timeseries.append(id2ts[l])
                used_subjectids.append(l)
    
    final_timeseries = np.array(final_timeseries)

    final_timeseries = final_timeseries[:,:,:1024]

    eps=1e-8

    mean = final_timeseries.mean(axis=(1, 2), keepdims=True)     # (N,1,1)
    std  = final_timeseries.std(axis=(1, 2), ddof=0, keepdims=True)

    normed = (final_timeseries - mean) / (std + eps)

    print(f'normed.shape={normed.shape}')

    return normed



def load_pnc_data():
    ts_data = np.load('/local/scratch3/khan58/BrainNetworkTransformer/backup_dataset/PNC/514_timeseries.npy', allow_pickle=True)
    pearson_data = np.load('/local/scratch3/khan58/BrainNetworkTransformer/backup_dataset/PNC/514_pearson.npy', allow_pickle=True)

    label_df = pd.read_csv('/local/scratch3/khan58/BrainNetworkTransformer/backup_dataset/PNC/PNC_Gender_Age.csv')

    pearson_data, timeseries_data = pearson_data.item(), ts_data.item()

    pearson_id = pearson_data['id']
    pearson_data = pearson_data['data']

    for i in range(pearson_data.shape[0]): 
        np.fill_diagonal(pearson_data[i], 1)

    id2pearson = dict(zip(pearson_id, pearson_data))
    

    ts_id = timeseries_data['id']
    timeseries_data = timeseries_data['data']

    id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

    final_timeseries, final_label, final_pearson = [], [], []

    used_subjectids = []

    for fc, l in zip(timeseries_data, ts_id):
        if l in id2gender and l in id2pearson:
            final_timeseries.append(fc)
            final_label.append(id2gender[l])
            final_pearson.append(id2pearson[l])
            used_subjectids.append(l)


    final_timeseries = np.array(final_timeseries).transpose(0, 2, 1)

    mean = final_timeseries.mean(axis=(1, 2), keepdims=True)     # (N,1,1)
    std  = final_timeseries.std(axis=(1, 2), ddof=0, keepdims=True)

    eps=1e-8

    normed = (final_timeseries - mean) / (std + eps)

    print(f'normed.shape={normed.shape}')

    return normed



def load_abcd_data():
    ts_data = np.load('/local/scratch3/khan58/BrainNetworkTransformer/backup_dataset/ABCD/abcd_rest-timeseires-HCP2016.npy', allow_pickle=True)
    print(f'ts_data.shape={ts_data.shape}')

    pearson_data = np.load('/local/scratch3/khan58/BrainNetworkTransformer/backup_dataset/ABCD/abcd_rest-pearson-HCP2016.npy', allow_pickle=True)

    predicted_labels = pd.read_csv('/local/scratch3/khan58/Datasets/ABCD/labels37_final_comb_measures.csv')


    with open('/local/scratch3/khan58/BrainNetworkTransformer/backup_dataset/ABCD/ids_HCP2016.txt', 'r') as f:
        lines = f.readlines()
        pearson_id = [line[:-1] for line in lines]

    with open('/local/scratch3/khan58/BrainNetworkTransformer/backup_dataset/ABCD/ids_HCP2016_timeseires.txt', 'r') as f:
        lines = f.readlines()
        ts_id = [line[:-1] for line in lines]

    id2pearson = dict(zip(pearson_id, pearson_data))

    id2label = dict(zip(predicted_labels['subjectkey'], predicted_labels['pea_wiscv_trs']))

    final_timeseries, final_label, final_pearson, orig_connection = [], [], [], []

    used_subjectids = []

    for ts, l in zip(ts_data, ts_id):
        if l in id2label and l in id2pearson:
            if not np.any(np.isnan(id2pearson[l])) and not np.any(np.isnan(id2label[l])):
                final_timeseries.append(ts)
                final_label.append(id2label[l])
                final_pearson.append(id2pearson[l])
                used_subjectids.append(l)
    
    final_timeseries = np.array(final_timeseries)

    mean = final_timeseries.mean(axis=(1, 2), keepdims=True)     # (N,1,1)
    std  = final_timeseries.std(axis=(1, 2), ddof=0, keepdims=True)

    eps=1e-8

    normed = (final_timeseries - mean) / (std + eps)

    print(f'normed.shape={normed.shape}')

    return normed


    
