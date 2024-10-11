import numpy as np
import torch
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing


def load_abcd_data(args):

    ts_data = np.load('/local/scratch/xkan/ABCD/ABCD/abcd_rest-timeseires-HCP2016.npy', allow_pickle=True)

    pearson_data = np.load('/local/scratch/xkan/ABCD/abcd_rest-pearson-HCP2016.npy', allow_pickle=True)

    predicted_labels = pd.read_csv('/local/scratch3/khan58/Datasets/ABCD/labels37_final_comb_measures.csv')

    with open('/local/scratch/xkan/ABCD/ids_HCP2016.txt', 'r') as f:
        lines = f.readlines()
        pearson_id = [line[:-1] for line in lines]

    with open('/local/scratch/xkan/ABCD/ABCD/ids_HCP2016_timeseires.txt', 'r') as f:
        lines = f.readlines()
        ts_id = [line[:-1] for line in lines]

    id2pearson = dict(zip(pearson_id, pearson_data))

    id2label = dict(zip(predicted_labels['subjectkey'], predicted_labels[args.measure]))
     
    final_label, final_pearson = [], []

    for ts, l in zip(ts_data, ts_id):
        if l in id2label and l in id2pearson:
            if args.measure == 'sex':
                if not np.any(np.isnan(id2pearson[l])):
                    final_label.append(id2label[l])
                    final_pearson.append(id2pearson[l])
            else:
                if not np.any(np.isnan(id2pearson[l])) and not np.any(np.isnan(id2label[l])):
                    final_label.append(id2label[l])
                    final_pearson.append(id2pearson[l])
        
    labels = final_label
    
    if args.measure == 'sex':
        for i in range(len(labels)):
            labels[i] = 0 if labels[i] == 'F' else 1
    
    final_pearson, labels = [np.array(
        data) for data in (final_pearson, labels)]

    print(f'labels = {labels}')
    print(f'datasize = {labels.shape[0]}')

    return final_pearson, labels



def load_abide_data(args):

    data = np.load('/local/scratch3/khan58/BrainGB/examples/datasets/ABIDE/abide.npy', allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']

    # convert fisher transformation back to pearson correlation matrix
    for i in range(final_pearson.shape[0]): 
        np.fill_diagonal(final_pearson[i], np.inf)
    
    final_pearson = np.tanh(final_pearson)
    
    final_timeseires, final_pearson, labels = [np.array(
        data) for data in (final_timeseires, final_pearson, labels)]
    
    print(f'labels = {labels}')
    print(f'labels.shape={labels.shape}')
    print(f'datasize = {labels.shape[0]}')

    return final_pearson, labels




def load_hcp_data(args):
    data = np.load('/local/scratch3/khan58/Datasets/HCP_data_release/HCP_correlation_matrix.npy', allow_pickle=True).item()
    final_timeseires = data["corr"]    # we do not use timeseries here, it's just a placeholder

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

    id2label = dict(zip(predicted_labels['Subject'], predicted_labels[args.measure]))

    final_label, final_pearson = [], []

    age_class_dict = {'22-25':0, '26-30':1, '31-35':2}

    for l in id:
        if l in id2label and l in id2pearson and l in id2ts:
            if args.measure == 'Gender':
                if not np.any(np.isnan(id2pearson[l])):
                    final_label.append(id2label[l])
                    final_pearson.append(id2pearson[l])
            elif args.measure == 'Age':
                final_label.append(age_class_dict[id2label[l]])
                final_pearson.append(id2pearson[l])
            else:
                if not np.any(np.isnan(id2pearson[l])) and not np.any(np.isnan(id2label[l])):
                    final_label.append(id2label[l])
                    final_pearson.append(id2pearson[l])    

    labels = final_label
    
    if args.measure == 'Gender':
        for i in range(len(labels)):
            labels[i] = 0 if labels[i] == 'F' else 1
    
    final_pearson, labels = [np.array(
        data) for data in (final_pearson, labels)]

    print(f'labels = {labels}')
    print(f'labels.shape={labels.shape}')
    print(f'datasize = {labels.shape[0]}')

    return final_pearson, labels



def load_pnc_data(args):

    ts_data = np.load('/local/scratch/xkan/PNC_data/514_timeseries.npy', allow_pickle=True)
    pearson_data = np.load('/local/scratch/xkan/PNC_data/514_pearson.npy', allow_pickle=True)

    label_df = pd.read_csv('/local/scratch/xkan/PNC_data/PNC_Gender_Age.csv')

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

    for fc, l in zip(timeseries_data, ts_id):
        if l in id2gender and l in id2pearson:
            final_timeseires.append(fc)
            final_label.append(id2gender[l])
            final_pearson.append(id2pearson[l])

    final_pearson = np.array(final_pearson)

    final_timeseires = np.array(final_timeseires).transpose(0, 2, 1)

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df["sex"])

    labels = encoder.transform(final_label)

    final_timeseires, final_pearson, labels = [np.array(
        data) for data in (final_timeseires, final_pearson, labels)]
    
    print(f'labels = {labels}')
    print(f'labels.shape={labels.shape}')
    print(f'datasize = {labels.shape[0]}')

    return final_pearson, labels




