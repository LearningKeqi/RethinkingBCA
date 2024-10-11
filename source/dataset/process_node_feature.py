import numpy as np
import os 
from omegaconf import DictConfig, open_dict


def preprocess_nodefeature(cfg, orig_connection, sparse_connection, final_timeseires):

    if cfg.dataset.feature_orig_or_sparse == 'orig':
        connection = orig_connection
    elif cfg.dataset.feature_orig_or_sparse == 'sparse':
        connection = sparse_connection
    elif cfg.dataset.feature_orig_or_sparse == 'binary_sparse':
        binary_sparse_connection = sparse_connection.copy()
        binary_sparse_connection[binary_sparse_connection!=0] = 1
        assert np.all((binary_sparse_connection == 0) | (binary_sparse_connection == 1)), "Tensor should only contain 0 and 1."
        
        connection = binary_sparse_connection


    if cfg.dataset.node_feature_type == 'identity':
        return Identity(cfg, connection)
    
    elif cfg.dataset.node_feature_type == 'eigenvec':
        return Eigenvec(cfg, connection)
    
    elif cfg.dataset.node_feature_type == 'connection':
        return Connection_profile(cfg, connection)
    
    elif cfg.dataset.node_feature_type == 'centrality':
        return Centrality_enc(cfg, connection)
    
    elif cfg.dataset.node_feature_type == 'degree':
        return Degree_enc(cfg, connection)
    
    elif cfg.dataset.node_feature_type == 'degree_bin':
        return Degree_bin(cfg, connection, cfg.dataset.num_bins)
    
    elif cfg.dataset.node_feature_type.endswith('time_series'):
        return Time_series(cfg, final_timeseires)

    elif cfg.dataset.node_feature_type == 'learnable_eigenvec':
        return Learnable_eigenvec(cfg, connection)
    
    elif cfg.dataset.node_feature_type.startswith('gnn_'):
        if cfg.dataset.node_feature_type.endswith('identity'):
            return np.concatenate((connection, Identity(cfg, connection)), axis=2) 
        elif cfg.dataset.node_feature_type.endswith('eigenvec'):
            return np.concatenate((connection, Eigenvec(cfg, connection)), axis=2)
        elif cfg.dataset.node_feature_type.endswith('connection'):
            return np.concatenate((connection, Connection_profile(cfg, connection)), axis=2)
    

def Identity(cfg, connection):
    num_subjects = connection.shape[0]
    num_nodes = connection.shape[-1]
    process_feature = np.eye(num_nodes)
    process_feature = np.tile(process_feature, (num_subjects, 1, 1))

    print(f'process_feature.shape={process_feature.shape}')

    if cfg.dataset.node_feature_type == 'identity':
        with open_dict(cfg):
            cfg.dataset.node_feature_dim = num_nodes

    return process_feature


def Eigenvec(cfg, connection):

    file_path = f'./exp_results/eigenvectors/eigen_{cfg.dataset.name}.npy'

    if not os.path.exists(file_path):
        matrix_array = connection

        # special case: PPMI's diagnoal values are all zeros, it seems that some nodes don't have any connections with other nodes  
        if cfg.dataset.name == 'ppmi':
            num_graphs, _, _ = matrix_array.shape

            for i in range(num_graphs):
                np.fill_diagonal(matrix_array[i],1)
    

        num_samples = matrix_array.shape[0]
        num_nodes = matrix_array.shape[-1]
        
        result_array = np.zeros((num_samples, num_nodes, num_nodes + 1))
        
        for i in range(num_samples):
            # cal normalized laplacian matrix
            A = np.abs(matrix_array[i])  # calculate absolute matrix

            D = np.diag(np.sum(A, axis=1))
            D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
            I = np.eye(num_nodes) 
            Laplacian = I - np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)

            # cal eigen vectors and eigen values
            eigenvalues, eigenvectors = np.linalg.eig(Laplacian)
            
            # increasing order
            sort_index = np.argsort(eigenvalues)
            sorted_eigenvalues = eigenvalues[sort_index]
            sorted_eigenvectors = eigenvectors[:, sort_index]
            
            # concate eigenvalues as the last column of eigenvectors matrix
            result_array[i, :, :-1] = sorted_eigenvectors
            result_array[i, :, -1] = sorted_eigenvalues
            
        process_feature = result_array

        np.save(file_path, process_feature)

    else:
        process_feature = np.load(file_path)

    if cfg.dataset.node_feature_type == 'gnn_eigenvec':
        process_feature = process_feature[:,:,:cfg.dataset.node_feature_eigen_topk]
    else:
        process_feature = process_feature[:,:,:cfg.dataset.node_feature_dim]

    print(f'process_feature.shape={process_feature.shape}')
    return process_feature


def Connection_profile(cfg, connection):
    process_feature = connection
    print(f'process_feature.shape={process_feature.shape}')

    if cfg.dataset.node_feature_type == 'connection':
        with open_dict(cfg):
            cfg.dataset.node_feature_dim = process_feature.shape[-1]


    return process_feature 


def Centrality_enc(cfg, connection):
    A = np.abs(connection)
    process_feature = np.sum(A, axis=2, keepdims=True)
    print(f'process_feature.shape={process_feature.shape}')

    return process_feature


def Degree_enc(cfg, connection):
    A = connection
    process_feature = np.sum(A, axis=2, keepdims=True)
    print(f'process_feature.shape={process_feature.shape}')

    return process_feature


def Degree_bin(cfg, connection, num_bins):
    '''
    Calculate Node degree and map the dergee d into 
    one of the T buckets based on its degree value.
    Each bucket corresponds to a One-hot vector, 
    which will be assigned as the initial node feature.

    Input: 
        connection: np.array. shape: [num_samples, num_nodes, num_nodes]
        num_bins: int.
    Output:
        degree_bin_feature: shape: [num_samples, num_nodes, num_bins]. The one-hot vector initial feature for each node

    '''
    # Preparation for Degree Bin node feature

    num_samples, num_nodes, _ = connection.shape
    degree = np.sum(np.abs(connection), axis=2)

    # Get quantile edges
    edges = np.quantile(degree, np.linspace(0, 1, num_bins + 1))
    bin_indices = np.digitize(degree, edges, right=True) - 1
    bin_indices[bin_indices == -1] = 0
    bin_indices = np.expand_dims(bin_indices, axis=-1)

    print(f'bin_indices.shape={bin_indices.shape}')
    # degree_bin_feature = np.eye(num_bins)[bin_indices]

    return bin_indices



def Time_series(cfg, time_series):
    print(f'time series shape = {time_series.shape}')

    return time_series



def Learnable_eigenvec(cfg, connection):
    file_path = f'./exp_results/eigenvectors/eigen_{cfg.dataset.name}.npy'

    if not os.path.exists(file_path):
        matrix_array = connection

        num_samples = matrix_array.shape[0]
        num_nodes = matrix_array.shape[-1]
        
        result_array = np.zeros((num_samples, num_nodes, num_nodes + 1))
        
        for i in range(num_samples):
            # cal normalized laplacian matrix
            A = np.abs(matrix_array[i])  # calculate absolute matrix

            D = np.diag(np.sum(A, axis=1))
            D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
            I = np.eye(num_nodes) 
            Laplacian = I - np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)

            # cal eigen vectors and eigen values
            eigenvalues, eigenvectors = np.linalg.eig(Laplacian)
            
            # increasing order
            sort_index = np.argsort(eigenvalues)
            sorted_eigenvalues = eigenvalues[sort_index]
            sorted_eigenvectors = eigenvectors[:, sort_index]
            
            # concate eigenvalues as the last column of eigenvectors matrix
            result_array[i, :, :-1] = sorted_eigenvectors
            result_array[i, :, -1] = sorted_eigenvalues
            
        process_feature = result_array

        np.save(file_path, process_feature)

    else:
        process_feature = np.load(file_path)
    
    print(f'process_feature.shape={process_feature.shape}')
    return process_feature



