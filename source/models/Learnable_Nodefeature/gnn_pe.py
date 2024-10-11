import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, X, adj):
        # Apply the linear transformation
        h = self.linear(X)
        # Apply the normalized adjacency matrix to the transformed features
        h = torch.bmm(adj, h)
        return h
    
class GCN_Nodefeature(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, out_dim):
        super(GCN_Nodefeature, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        # Output layer
        self.layers.append(GCNLayer(hidden_dim, out_dim))
        
    def forward(self, orig_feature):
        num_nodes = orig_feature.shape[1]
        adj = orig_feature[:,:,:num_nodes]
        X = orig_feature[:,:,num_nodes:]

        adj = torch.abs(adj)  # absolute value, since GCN cannot deal with negative value

        # Normalize adjacency matrix
        D = torch.sum(adj, dim=2)  # Degree matrix
        D_inv_sqrt = torch.pow(D, -0.5)  # D^-0.5
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        adj_normalized = D_inv_sqrt.unsqueeze(2) * adj * D_inv_sqrt.unsqueeze(1)
        
        h = X
        for layer in self.layers:
            h = F.relu(layer(h, adj_normalized))
        
        return h
