import numpy as np
import os
import argparse
from load_pc import load_abcd_pc, load_abide_pc, load_pnc_pc, load_hcp_pc


def Eigenvec(dataset_name, connection):

    file_path = f'/local/scratch3/khan58/BrainNetworkTransformer/testrun/eigenvectors/eigen_{dataset_name}.npy'

    if not os.path.exists(file_path):
        print(f'{dataset_name} eigenvectors not found, generating...')
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
            
            print(f'{i+1} th sample done!')
            
        process_feature = result_array

        np.save(file_path, process_feature)

        print(f'{dataset_name} eigenvectors saved!')
    else:
        print(f'{dataset_name} eigenvectors found, skipping...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()

    connection = eval(f'load_{args.dataset_name}_pc()')
    print(f'connection.shape={connection.shape}')

    Eigenvec(args.dataset_name, connection)

    # the command to use this script is: python generate_eigenvec_nf.py --dataset_name abcd