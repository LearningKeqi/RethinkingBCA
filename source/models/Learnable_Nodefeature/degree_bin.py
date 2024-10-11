import torch
import torch.nn as nn


class DegreeBin_Nodefeature(nn.Module):
    """
    Compute the Degree Bin node features for each node in the graph using embedding.
    """
    def __init__(self, num_bins, embedding_dim):
        super(DegreeBin_Nodefeature, self).__init__()
        
        self.embedding = nn.Embedding(num_bins, embedding_dim)

    def forward(self, bin_index):
        '''
        one_hot_vectors: [bz, num_nodes, num_bins]
        Indices are the indices for each node's degree bin.
        '''
        indices = bin_index.squeeze().long()
        embedding = self.embedding(indices)
        
        return embedding
