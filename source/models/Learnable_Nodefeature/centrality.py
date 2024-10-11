import torch
import torch.nn as nn
import torch.nn.functional as F


class Centrality_Nodefeature(nn.Module):
    """
    Compute the centrality node features for each node in the graph.
    """

    def __init__(self, degree_dim, embedding_dim):
        
        super(Centrality_Nodefeature, self).__init__()
        self.fc = nn.Linear(degree_dim, embedding_dim)

    def forward(self, degree):
        '''
        degree: batch_sz x num_nodes x 1 
        '''
        embedding = F.relu(self.fc(degree))  
        
        return embedding