import torch
import torch.utils.data as utils
from torch.utils.data import Subset
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt
from ..utils import draw_single_connectivity, draw_multiple_connectivity, draw_single_attn



def split_dataset(dataset):
    num_graphs = len(dataset)
    train_length = int(num_graphs*0.7)
    val_length = int(num_graphs*0.1)
    test_length = num_graphs-train_length-val_length
    data_index = np.arange(num_graphs)
    np.random.shuffle(data_index)

    train_index = data_index[:train_length]
    valid_index = data_index[train_length: train_length + val_length]
    test_index = data_index[train_length + val_length:]

    return train_index, valid_index, test_index

def save_indices(train_index, valid_index, test_index, cur_repeat, args):
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
    
    file_path = f'./exp_results/split_with_valid/{str(args.dataset.name)}_repeat{str(cur_repeat)}_indices.csv'

    print(f'Index file_path={file_path}')

    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        print(f"Index File {file_path} already exists, skipping saving.")

    

def save_input_conn(cfg, orig_connection, sparse_connection, used_subjectids, only_positive_corr, data_type):
    first8_subject_id_path =  f'./exp_results/double_route/first8_subject_ids/{cfg.common_save}_{data_type}_repeat{cfg.dataset.cur_repeat}.npy'
    if not os.path.exists(first8_subject_id_path):
        num_subjects = orig_connection.shape[0]
        id_index = np.random.choice(num_subjects, 8, replace=False)
        first8_subject_id = used_subjectids[id_index].numpy()
        np.save(first8_subject_id_path, first8_subject_id)
    else:
        first8_subject_id = np.load(first8_subject_id_path)

    sparse_mask = (sparse_connection!=0).float()

    # save weighted
    base_folder = './exp_results/double_route/input_conn'
    weighted_sparse = sparse_mask * orig_connection
        
    positions = [np.where(used_subjectids.cpu().detach().numpy() == id)[0][0] for id in first8_subject_id]
    # print(f'positions={positions}')

    first8_weighted_conn = weighted_sparse[positions]

    first_8_save_file = f'{base_folder}/first8_weighted_{cfg.common_save}_{data_type}_repeat{cfg.dataset.cur_repeat}.npy'
    np.save(first_8_save_file, first8_weighted_conn)
    first_8_list = [first8_weighted_conn[i] for i in range(8)]
    draw_path = first_8_save_file.replace('.npy','')

    if cfg.draw_heatmap:
        draw_multiple_connectivity(first_8_list, draw_path, f'{str(only_positive_corr)}_weighted_connectivity')

    average_across_subjects = torch.mean(weighted_sparse, dim=0)
    average_save_file = f'{base_folder}/average_weighted_{cfg.common_save}_{data_type}_repeat{cfg.dataset.cur_repeat}.npy'
    np.save(average_save_file, average_across_subjects)
    draw_average_save_file = average_save_file.replace('.npy','')

    if cfg.draw_heatmap:
        draw_single_connectivity(average_across_subjects, draw_average_save_file, f'{str(only_positive_corr)}_weighted_connectivity')
    
    # save binary (probably)
    first8_sparse_conn = sparse_connection[positions]
    first_8_save_file = f'{base_folder}/first8_sparse_{cfg.common_save}_{data_type}_repeat{cfg.dataset.cur_repeat}.npy'
    np.save(first_8_save_file, first8_sparse_conn)
    first_8_list = [first8_sparse_conn[i] for i in range(8)]
    draw_path = first_8_save_file.replace('.npy','')

    if cfg.draw_heatmap:
        draw_multiple_connectivity(first_8_list, draw_path, f'binary_connectivity')

    average_across_subjects = torch.mean(sparse_connection, dim=0)
    average_save_file = f'{base_folder}/average_sparse_{cfg.common_save}_{data_type}_repeat{cfg.dataset.cur_repeat}.npy'
    np.save(average_save_file, average_across_subjects)
    draw_average_save_file = average_save_file.replace('.npy','')

    if cfg.draw_heatmap:
        draw_single_connectivity(average_across_subjects, draw_average_save_file, f'binary_connectivity')
    


def save_input_connectivity(cfg, orig_connection, sparse_connection,
                            train_indices, valid_indices, test_indices, used_subjectids):
    
    # save_input_conn(cfg, orig_connection[train_indices], sparse_connection[train_indices], used_subjectids[train_indices], cfg.dataset.only_positive_corr, 'train')
    # save_input_conn(cfg, orig_connection[valid_indices], sparse_connection[valid_indices], used_subjectids[valid_indices], cfg.dataset.only_positive_corr, 'valid')
    save_input_conn(cfg, orig_connection[test_indices], sparse_connection[test_indices], used_subjectids[test_indices], cfg.dataset.only_positive_corr, 'test')






def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor,
                    orig_connection: torch.tensor,
                    saved_eigenvectors: torch.tensor,
                    sparse_connection: torch.tensor,
                    used_subjectids: torch.tensor) -> List[utils.DataLoader]:
    
    if cfg.dataset.task == 'classification':
        labels = F.one_hot(labels.to(torch.int64))    # label dimention is num_class

    print(f'labels.shape={labels.shape}')

    print(f'final_timeseries.shape = {final_timeseires.shape}, final_pearson.shape = {final_pearson.shape}, labels.shape = {labels.shape}')
    
    print(f'orig_connection.shape = {orig_connection.shape}, saved_eigenvectors.shape = {saved_eigenvectors.shape}, sparse_connection.shape = {sparse_connection.shape}')

    dataset = utils.TensorDataset(
        final_timeseires,
        final_pearson,
        labels,
        orig_connection,
        saved_eigenvectors,
        sparse_connection,
        used_subjectids
    )
    
    train_index, valid_index, test_index = split_dataset(dataset)

    save_indices(train_index, valid_index, test_index, cfg.dataset.cur_repeat, cfg)  # save indices
    load_path = f'./exp_results/split_with_valid/{str(cfg.dataset.name)}_repeat{str(cfg.dataset.cur_repeat)}_indices.csv'


    indices = pd.read_csv(load_path)
    
    print(f'Index load_path={load_path}')

    train_indices = indices['Train Index'].dropna().to_numpy().astype(int)
    valid_indices = indices['Valid Index'].dropna().to_numpy().astype(int)
    test_indices = indices['Test Index'].dropna().to_numpy().astype(int)


    if cfg.dataset.plot_figures:
        save_input_connectivity(cfg, orig_connection, sparse_connection,
                                train_indices, valid_indices, test_indices, used_subjectids)



    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            len(train_indices) - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
    


    print(f'len dataset = {len(dataset)}')
    print(f"len train_dataset={len(train_dataset)},len valid_dataset={len(valid_dataset)},len test_dataset={len(test_dataset)}")
    # train_dataset, val_dataset, test_dataset = utils.random_split(
    #     dataset, [train_length, val_length, test_length])


    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    print(f'train_dataloader.size beginning = {len(train_dataloader)}')

    val_dataloader = utils.DataLoader(
        valid_dataset, batch_size=cfg.dataset.val_batch_size, shuffle=True, drop_last=False)
    
    print(f'valid_dataloader.size beginning = {len(val_dataloader)}')

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.test_batch_size, shuffle=True, drop_last=False)
    print(f'test_dataloader.size beginning = {len(test_dataloader)}')

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array,
                               orig_connection: torch.tensor,
                               saved_eigenvectors: torch.tensor,
                               sparse_connection: torch.tensor,
                               used_subjectids: torch.tensor) -> List[utils.DataLoader]:
    
    if cfg.dataset.task == 'classification':
        labels = F.one_hot(labels.to(torch.int64))

    
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs


    if cfg.dataset.task == 'classification':
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=val_length+test_length, train_size=train_length)
        
        train_index = None
        test_valid_index = None
        for train_index_inner, test_valid_index_inner in split.split(final_timeseires, stratified):
            train_index = train_index_inner
            test_valid_index = test_valid_index_inner

            final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
                test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
            stratified = stratified[test_valid_index_inner]

        split2 = StratifiedShuffleSplit(
            n_splits=1, train_size=test_length)

        test_index, valid_index = None, None

        for test_index_inner, valid_index_inner in split2.split(final_timeseires_val_test, stratified):
            test_index = test_valid_index[test_index_inner]
            valid_index = test_valid_index[valid_index_inner]

        save_indices(train_index, valid_index, test_index, cfg.dataset.cur_repeat, cfg)  # save indices
    
    load_path = f'./exp_results/split_with_valid/{str(cfg.dataset.name)}_repeat{str(cfg.dataset.cur_repeat)}_indices.csv'

    indices = pd.read_csv(load_path)

    train_indices = indices['Train Index'].dropna().to_numpy().astype(int)
    valid_indices = indices['Valid Index'].dropna().to_numpy().astype(int)
    test_indices = indices['Test Index'].dropna().to_numpy().astype(int)


    if cfg.dataset.plot_figures:
        save_input_connectivity(cfg, orig_connection, sparse_connection,
                                train_indices, valid_indices, test_indices, used_subjectids)



    if cfg.dataset.task == 'classification':
        print(f'train_one_num = {torch.sum(labels[train_indices][:,1])}, train_zero_num = {torch.sum(1 - labels[train_indices][:,1])}')
        print(f'valid_one_num = {torch.sum(labels[valid_indices][:,1])}, valid_zero_num = {torch.sum(1 - labels[valid_indices][:,1])}')
        print(f'test_one_num = {torch.sum(labels[test_indices][:,1])}, test_zero_num = {torch.sum(1 - labels[test_indices][:,1])}')


    print(load_path)

    final_timeseires_train, final_pearson_train, labels_train, \
        orig_connection_train, saved_eigenvectors_train, sparse_connection_train, used_subjectids_train = final_timeseires[train_indices], final_pearson[train_indices], labels[train_indices], orig_connection[train_indices], saved_eigenvectors[train_indices], sparse_connection[train_indices], used_subjectids[train_indices]
    final_timeseires_val, final_pearson_val, labels_val, \
        orig_connection_val, saved_eigenvectors_val, sparse_connection_val, used_subjectids_val = final_timeseires[valid_indices], final_pearson[valid_indices], labels[valid_indices], orig_connection[valid_indices], saved_eigenvectors[valid_indices], sparse_connection[valid_indices], used_subjectids[valid_indices]
    final_timeseires_test, final_pearson_test, labels_test, \
        orig_connection_test, saved_eigenvectors_test, sparse_connection_test, used_subjectids_test = final_timeseires[test_indices], final_pearson[test_indices], labels[test_indices], orig_connection[test_indices], saved_eigenvectors[test_indices], sparse_connection[test_indices], used_subjectids[test_indices]

    print(f'final_timeseries.shape = {final_timeseires_train.shape}, final_pearson.shape = {final_pearson_train.shape}, labels.shape = {labels_train.shape}')
    
    print(f'orig_connection.shape = {orig_connection_train.shape}, saved_eigenvectors.shape = {saved_eigenvectors_train.shape}, sparse_connection.shape = {sparse_connection_train.shape}')


    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train,
        orig_connection_train,
        saved_eigenvectors_train,
        sparse_connection_train,
        used_subjectids_train
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val, orig_connection_val, saved_eigenvectors_val, sparse_connection_val, used_subjectids_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test, orig_connection_test, saved_eigenvectors_test, sparse_connection_test, used_subjectids_test
    )


    print(f'len dataset = {final_timeseires.shape[0]}')
    print(f"len train_dataset={len(train_dataset)},len valid_dataset={len(val_dataset)},len test_dataset={len(test_dataset)}")


    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]
