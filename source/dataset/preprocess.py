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



def threshold_adjacency_matrices_newfc(orig_connection, ratio):
    """
    Threshold adjacency matrices for non-correlation FCs (e.g., coherence, PLV, GC, PDC).

    Behavior:
    - Per subject matrix validation. If a matrix contains any non-finite values (NaN/Inf),
      replace the entire matrix with an identity matrix (diagonal=1) as a robust fallback.
    - Otherwise, keep the top `ratio` proportion of off-diagonal edges by absolute weight
      (no symmetry enforcement; suitable for directed FCs). Zeros out the rest.
    - Always set the diagonal to 1.

    Parameters:
    - orig_connection (torch.Tensor): shape (batch_size, num_nodes, num_nodes)
    - ratio (float): 0..1, proportion of off-diagonal edges to keep by |weight|

    Returns:
    - torch.Tensor: thresholded adjacency matrices, same shape as input.
    """
    X = orig_connection.clone()

    batch_size, num_nodes, _ = X.size()

    # ratio edge cases
    if ratio <= 0:
        out = torch.zeros_like(X)
        for i in range(batch_size):
            out[i].fill_diagonal_(1)
        return out

    if ratio >= 1:
        # Validate per subject; fallback to identity if invalid
        out = X.clone()
        for i in range(batch_size):
            mat = out[i]
            if not torch.isfinite(mat).all():
                out[i] = torch.eye(num_nodes, dtype=mat.dtype, device=mat.device)
            else:
                out[i].fill_diagonal_(1)
        return out

    # General case: 0 < ratio < 1
    thresholded = torch.zeros_like(X)

    # Precompute masks
    eye_mask = torch.eye(num_nodes, dtype=torch.bool, device=X.device)
    offdiag_mask = ~eye_mask

    for i in range(batch_size):
        mat = X[i]

        # Fallback for invalid matrices
        if not torch.isfinite(mat).all():
            thresholded[i] = torch.eye(num_nodes, dtype=mat.dtype, device=mat.device)
            continue

        # Work on off-diagonal entries only and exclude zeros
        off_abs_flat = torch.abs(mat)[offdiag_mask]
        num_nonzero = int((off_abs_flat > 0).sum().item())

        if num_nonzero == 0:
            thresholded[i] = torch.eye(num_nodes, dtype=mat.dtype, device=mat.device)
            continue

        k = int(ratio * num_nonzero)
        if k <= 0:
            thresholded[i] = torch.zeros_like(mat)
            thresholded[i].fill_diagonal_(1)
            continue
        if k > num_nonzero:
            k = num_nonzero

        # Select top-k by absolute value among nonzero off-diagonal entries
        # Since k <= num_nonzero, torch.topk will not select zeros
        topk_vals, topk_idx = torch.topk(off_abs_flat, k)

        # Map flattened off-diagonal indices back to (row, col)
        idx_off = torch.nonzero(offdiag_mask, as_tuple=False)  # (num_off, 2)
        selected_pairs = idx_off[topk_idx]

        keep_mask = torch.zeros_like(mat, dtype=torch.bool)
        keep_mask[selected_pairs[:, 0], selected_pairs[:, 1]] = True

        thr_mat = torch.zeros_like(mat)
        thr_mat[keep_mask] = mat[keep_mask]
        thr_mat.fill_diagonal_(1)
        thresholded[i] = thr_mat

    return thresholded




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
