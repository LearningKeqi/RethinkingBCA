import numpy as np
from omegaconf import DictConfig
import torch


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def transform(self, data: np.array):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.array):
        return (data * self.std) + self.mean


def reduce_sample_size(config: DictConfig, *args):
    sz = args[0].shape[0]
    used_sz = int(sz * config.datasz.percentage)
    return [d[:used_sz] for d in args]


def threshold_adjacency_matrices(orig_connection, ratio, only_positive):
    """
    Function to threshold adjacency matrices based on the given ratio.
    It keeps the edges with absolute weights in the top specified ratio and zeros out the rest.
    
    Parameters:
    - X (torch.Tensor): The input batch of weighted adjacency matrices of shape (batch_size, num_nodes, num_nodes).
    - ratio (float): The ratio of edges to keep based on their absolute weight.
    - only_positive (boolean): consider only positive correlation?
    Returns:
    - torch.Tensor: The thresholded adjacency matrices with the same shape as X.
    """

    X = orig_connection.clone()

    if only_positive:
        print('only positive correlations')
        X[X<0]=0

    batch_size, num_nodes, _ = X.size()

    if ratio == 1:
        for i in range(batch_size):
            X[i].fill_diagonal_(1)
        return X


    if ratio == 0:    # keep self-loop
        thresholded_X = torch.zeros_like(X)
        for i in range(batch_size):
            thresholded_X[i].fill_diagonal_(1)
        
        return thresholded_X
    
    # Create a tensor to store the thresholded adjacency matrices
    thresholded_X = torch.zeros_like(X)
    
    for i in range(batch_size):
        if only_positive:
            # Flatten the upper triangular part of the matrix to avoid duplicating symmetric edges
            upper_triangular_flat = X[i].triu(diagonal=1).flatten()   # diagnoal = 1, no diagonal
            upper_num_positive = torch.sum(upper_triangular_flat>0)

            # Number of edges to keep per adjacency matrix
            num_edges_to_keep = int(ratio * upper_num_positive)
        else:
            # upper_triangular_flat = X[i].triu().flatten()
            # num_edges_to_keep = int(ratio * num_nodes * (num_nodes + 1) / 2)  # Divide by 2 because the matrices are symmetric
            upper_triangular_flat = X[i].triu(diagonal=1).flatten()
            num_edges_to_keep = int(ratio * num_nodes * (num_nodes - 1) / 2)  # Divide by 2 because the matrices are symmetric



        # Get the absolute values and sort them to find the threshold
        values, indices = torch.abs(upper_triangular_flat).sort(descending=True)
        threshold = values[num_edges_to_keep]
        
        # Apply thresholding
        mask = torch.abs(X[i]) >= threshold
        
        # Apply the symmetrical mask and update the thresholded adjacency matrix
        thresholded_X[i] = X[i] * mask
    

    for i in range(batch_size):
        thresholded_X[i].fill_diagonal_(1) # keep self-loop
        
    return thresholded_X



def threshold_adjacency_matrices_knn(orig_connection, k, only_positive):
    X = orig_connection.clone()
    
    if only_positive:
        print('only positive correlations')
        X[X < 0] = 0

    batch_size, num_nodes, _ = X.size()

    # Create a tensor to store the thresholded adjacency matrices
    thresholded_X = torch.zeros_like(X)

    for i in range(batch_size):
        X[i].fill_diagonal_(0)  # keep self-loop

    for i in range(batch_size):
        for node in range(num_nodes):
            # Get the connection strengths for the current node
            connections = X[i, node, :]
            if only_positive:
                # Get the top k positive connections
                top_k_values, top_k_indices = torch.topk(connections, k)
            else:
                # Get the top k absolute connections
                top_k_values, top_k_indices = torch.topk(torch.abs(connections), k)
                        
            # Set the corresponding values in the thresholded matrix
            for idx in top_k_indices:
                thresholded_X[i, node, idx] = X[i, node, idx]
                thresholded_X[i, idx, node] = X[i, node, idx]  # Ensure symmetry

    for i in range(batch_size):
        thresholded_X[i].fill_diagonal_(1)  # keep self-loop
        
    return thresholded_X