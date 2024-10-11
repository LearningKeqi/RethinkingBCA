import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import seaborn as sns
import numpy as np
import torch


def draw_single_connectivity(matrix, save_file, draw_type):
    # print(f'matix={matrix}')

    if isinstance(matrix, torch.Tensor):
        if matrix.is_cuda:
            matrix = matrix.cpu() 
        matrix = matrix.numpy()


    map_name = {'Somatomotor': "SM", 'DMN': "DMN",
            'VentralSalience': "VS", 'CentralExecutive': "CE",
            'DorsalSalience': "DS", 'Visual': "Vis"}

    network_labels = ['SM', 'DMN', 'VS', 'CE', 'DS', 'Vis']

    new_order, sorted_aac6_mapped = get_reorder()
    sorted_aac6_mapped = np.array(sorted_aac6_mapped)

    # Reorder to get new matrix
    reorder_matrix = matrix[new_order][:, new_order]

    # Use new network_labels
    plot_heatmap(reorder_matrix,save_file, network_labels, sorted_aac6_mapped, draw_type=draw_type)




def draw_single_x(matrix, save_file, draw_type):

    if isinstance(matrix, torch.Tensor):
        if matrix.is_cuda:
            matrix = matrix.cpu() 
        matrix = matrix.numpy()

    map_name = {'Somatomotor': "SM", 'DMN': "DMN",
            'VentralSalience': "VS", 'CentralExecutive': "CE",
            'DorsalSalience': "DS", 'Visual': "Vis"}

    network_labels = ['SM', 'DMN', 'VS', 'CE', 'DS', 'Vis']

    new_order, sorted_aac6_mapped = get_reorder()
    sorted_aac6_mapped = np.array(sorted_aac6_mapped)

    # Reorder to get new matrix
    reorder_matrix = matrix[new_order][:, new_order]

    # Use new network_labels
    plot_heatmap(reorder_matrix,save_file, network_labels, sorted_aac6_mapped, draw_type = draw_type)



def draw_single_attn(matrix, save_file):
    # print(f'matix={matrix}')

    if isinstance(matrix, torch.Tensor):
        if matrix.is_cuda:
            matrix = matrix.cpu() 
        matrix = matrix.numpy()

    map_name = {'Somatomotor': "SM", 'DMN': "DMN",
            'VentralSalience': "VS", 'CentralExecutive': "CE",
            'DorsalSalience': "DS", 'Visual': "Vis"}

    network_labels = ['SM', 'DMN', 'VS', 'CE', 'DS', 'Vis']

    new_order, sorted_aac6_mapped = get_reorder()
    sorted_aac6_mapped = np.array(sorted_aac6_mapped)

    # Reorder to get new matrix
    matrix = map2zero_one(matrix)
    reorder_matrix = matrix[new_order][:, new_order]

    # Use new network_labels
    plot_heatmap(reorder_matrix,save_file, network_labels, sorted_aac6_mapped, draw_type = 'attention_map')


def add_division_lines(ax, functional_ids):

    boundaries = np.where(np.diff(functional_ids))[0] + 1
    
    for boundary in boundaries:
        ax.axhline(boundary, color='white', lw=2, linestyle='--')
        ax.axvline(boundary, color='white', lw=2, linestyle='--')


def plot_heatmap(matrix,file_name,network_labels,sorted_aac6_mapped, draw_type):

    fig, ax = plt.subplots(figsize=(10, 8))

    # sns.heatmap(matrix, ax=ax, cmap='seismic', square=True, vmin=-1, vmax=1)
    if draw_type == 'True_weighted_connectivity':
        sns.heatmap(matrix, ax=ax, cmap='hot', square=True, vmin=0, vmax=1)
    elif draw_type == 'False_weighted_connectivity':
        sns.heatmap(matrix, ax=ax, cmap='coolwarm', square=True, vmin=-1, vmax=1)
    elif draw_type == 'binary_connectivity':
        sns.heatmap(matrix, ax=ax, cmap='hot', square=True, vmin=0, vmax=1)
    elif draw_type == 'attention_map':
        sns.heatmap(matrix, ax=ax, cmap='hot', square=True, vmin=0, vmax=1)
    elif draw_type == 'integrated_gradient':
        max_value = np.max(np.abs(matrix))
        matrix = matrix/max_value
        sns.heatmap(matrix, ax=ax, cmap='coolwarm', square=True, vmin=-1, vmax=1)
    elif draw_type == 'x_no_norm':
        sns.heatmap(matrix, ax=ax, cmap='coolwarm', square=True)
    elif draw_type == 'zero_one':
        max_value = np.max(matrix)
        matrix = matrix/max_value
        sns.heatmap(matrix, ax=ax, cmap='plasma', square=True, vmin=0, vmax=1)



    add_division_lines(ax, sorted_aac6_mapped)


    ax.set_xticks([np.mean(np.where(sorted_aac6_mapped == i)[0]) + 0.5 for i in range(6)])
    ax.set_yticks([np.mean(np.where(sorted_aac6_mapped == i)[0]) + 0.5 for i in range(6)])
    ax.set_xticklabels(network_labels, rotation=90, fontsize=10)
    ax.set_yticklabels(network_labels, rotation=0, fontsize=10)
    
    
    plt.savefig(file_name, dpi=300)

    plt.close()
    # print(f'saved file_name = {file_name}')



def get_reorder():

    df = pd.read_csv('/local/scratch3/khan58/Datasets/ABCD/HCP2016_Node_Information_withAAC6.csv')

    sort_order = {
        'Somatomotor': 0,
        'DMN': 1,
        'VentralSalience': 2,
        'CentralExecutive': 3,
        'DorsalSalience': 4,
        'Visual': 5
    }


    df['sort_key'] = df['AAc-6'].map(sort_order)

    sorted_df = df.sort_values(by='sort_key', kind='mergesort')


    sorted_index_list = sorted_df.index.tolist()
    sorted_aac6_list = sorted_df['AAc-6'].tolist()

    sorted_aac6_mapped = [sort_order[value] for value in sorted_aac6_list]


    return sorted_index_list, sorted_aac6_mapped



def map2zero_one(input_matrix):
    min_val = np.min(input_matrix)
    max_val = np.max(input_matrix)


    normalized_matrix = (input_matrix - min_val) / (max_val - min_val)

    return normalized_matrix