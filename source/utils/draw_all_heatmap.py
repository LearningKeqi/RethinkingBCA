import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import seaborn as sns
import numpy as np
import torch
from PIL import Image
import re

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


def draw_single_connectivity(matrix, save_file):
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
    plot_heatmap(reorder_matrix,save_file, network_labels, sorted_aac6_mapped, draw_type='connectivity')



def plot_heatmap(matrix,file_name,network_labels,sorted_aac6_mapped, draw_type):

    fig, ax = plt.subplots(figsize=(10, 8))

    # sns.heatmap(matrix, ax=ax, cmap='seismic', square=True, vmin=-1, vmax=1)
    if draw_type == 'connectivity':
        sns.heatmap(matrix, ax=ax, cmap='seismic', square=True, vmin=-1, vmax=1)
    elif draw_type == 'attention_map':
        sns.heatmap(matrix, ax=ax, cmap='seismic', square=True, vmin=0, vmax=1)


    add_division_lines(ax, sorted_aac6_mapped)


    ax.set_xticks([np.mean(np.where(sorted_aac6_mapped == i)[0]) + 0.5 for i in range(6)])
    ax.set_yticks([np.mean(np.where(sorted_aac6_mapped == i)[0]) + 0.5 for i in range(6)])
    ax.set_xticklabels(network_labels, rotation=90, fontsize=10)
    ax.set_yticklabels(network_labels, rotation=0, fontsize=10)
    
    
    plt.savefig(file_name, dpi=300)

    plt.close()


def add_division_lines(ax, functional_ids):

    boundaries = np.where(np.diff(functional_ids))[0] + 1

    
    for boundary in boundaries:
        ax.axhline(boundary, color='white', lw=2, linestyle='--')
        ax.axvline(boundary, color='white', lw=2, linestyle='--')



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

##### 

def draw_multiple_attn(matrix_list, draw_labels, colors, save_file):
    if not matrix_list:
        print("Matrix list is empty.")
        return


    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(12, 18))
    # fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(24, 20), sharex=True, sharey=True)
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axes = axes.flatten() 

    new_order, sorted_aac6_mapped = get_reorder()
    sorted_aac6_mapped = np.array(sorted_aac6_mapped)
    network_labels = ['SM', 'DMN', 'VS', 'CE', 'DS', 'Vis']
    
    # matrix_list = [matrix_list[i] for i in range(4)]


    for idx, matrix in enumerate(matrix_list):
        if isinstance(matrix, torch.Tensor):
            if matrix.is_cuda:
                matrix = matrix.cpu()
            matrix = matrix.numpy()


        # matrix = map2zero_one(matrix)
        reorder_matrix = matrix[new_order][:, new_order]

        # sns.heatmap(reorder_matrix, ax=axes[idx], cmap='seismic', vmin=0, vmax=1, cbar=False, square=True)
        sns.heatmap(reorder_matrix, ax=axes[idx], cmap='seismic', cbar=False, square=True)


        add_division_lines(axes[idx], sorted_aac6_mapped)


        axes[idx].set_xticks([np.mean(np.where(sorted_aac6_mapped == i)[0]) + 0.5 for i in range(6)])
        axes[idx].set_yticks([np.mean(np.where(sorted_aac6_mapped == i)[0]) + 0.5 for i in range(6)])
        axes[idx].set_xticklabels(network_labels, rotation=90, fontsize=10)
        axes[idx].set_yticklabels(network_labels, rotation=0, fontsize=10)

        axes[idx].set_title(draw_labels[idx], fontsize=18, color=colors[idx])

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()



def draw_multiple_connectivity(matrix_list, save_file, norm_type):
    if not matrix_list:
        print("Matrix list is empty.")
        return

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 18))
    # fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(24, 20), sharex=True, sharey=True)
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axes = axes.flatten() 

    new_order, sorted_aac6_mapped = get_reorder()
    sorted_aac6_mapped = np.array(sorted_aac6_mapped)
    network_labels = ['SM', 'DMN', 'VS', 'CE', 'DS', 'Vis']
    
    # matrix_list = [matrix_list[i] for i in range(4)]

    for idx, matrix in enumerate(matrix_list):
        if isinstance(matrix, torch.Tensor):
            if matrix.is_cuda:
                matrix = matrix.cpu()
            matrix = matrix.numpy()


        if norm_type == 'integrated_gradient':
            max_value = np.max(np.abs(matrix))
            matrix = matrix/max_value
            reorder_matrix = matrix[new_order][:, new_order]
            sns.heatmap(reorder_matrix, ax=axes[idx], cmap='coolwarm', vmin=-1, vmax=1, cbar=False, square=True)
        elif norm_type == 'zero_one':
            max_value = np.max(matrix)
            matrix = matrix/max_value
            reorder_matrix = matrix[new_order][:, new_order]
            sns.heatmap(reorder_matrix, ax=axes[idx], cmap='plasma', vmin=0, vmax=1, cbar=False, square=True)
        elif norm_type == 'attention':
            min_val = np.min(matrix)
            max_val = np.max(matrix)
            matrix = (matrix - min_val) / (max_val - min_val)
            reorder_matrix = matrix[new_order][:, new_order]
            sns.heatmap(reorder_matrix, ax=axes[idx], cmap='hot', vmin=0, vmax=1, cbar=False, square=True)
        elif norm_type == 'True_weighted_connectivity':
            reorder_matrix = matrix[new_order][:, new_order]
            sns.heatmap(reorder_matrix, ax=axes[idx], cmap='hot', vmin=0, vmax=1, cbar=False, square=True)
        elif norm_type == 'False_weighted_connectivity':
            reorder_matrix = matrix[new_order][:, new_order]
            sns.heatmap(reorder_matrix, ax=axes[idx], cmap='coolwarm', vmin=-1, vmax=1, cbar=False, square=True)
        elif norm_type == 'binary_connectivity':
            reorder_matrix = matrix[new_order][:, new_order]
            sns.heatmap(reorder_matrix, ax=axes[idx], cmap='hot', vmin=0, vmax=1, cbar=False, square=True)



        add_division_lines(axes[idx], sorted_aac6_mapped)

        axes[idx].set_xticks([np.mean(np.where(sorted_aac6_mapped == i)[0]) + 0.5 for i in range(6)])
        axes[idx].set_yticks([np.mean(np.where(sorted_aac6_mapped == i)[0]) + 0.5 for i in range(6)])
        axes[idx].set_xticklabels(network_labels, rotation=90, fontsize=10)
        axes[idx].set_yticklabels(network_labels, rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

