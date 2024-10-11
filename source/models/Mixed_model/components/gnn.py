import torch
from torch_geometric.nn.models import GCN, GAT, GIN, GraphSAGE
from torch_geometric.nn.models import MLP
from torch import Tensor
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor
import copy
from torch_geometric.nn.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)


        
class MyGNN(torch.nn.Module):
    def __init__(self, inchannels, hidden_channels, num_layers, gnn_type, **kwargs):
        super(MyGNN, self).__init__()
        self.in_channels = inchannels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.gnn_type = gnn_type 

        self.gat_attention_score = None

        if self.gnn_type == 'gin':
            # self.gnn_module = GIN(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.num_layers)
            eps = kwargs.pop('eps', 0.0)
            self.gnn_module = My_GIN(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.num_layers, eps=eps)
        elif self.gnn_type == 'gcn':
            self.gnn_module = GCN(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.num_layers)
        elif self.gnn_type == 'gat':
            self.num_heads = kwargs.pop('num_heads', 1)
            print(f'self.num_heads={self.num_heads}')
            # self.gnn_module = GAT(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.num_layers, heads=self.num_heads)
            self.gnn_module =Exp_GAT(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.num_layers, heads=self.num_heads)
        elif self.gnn_type == 'graphsage':
            print(f'using graph sage')
            aggr = kwargs.pop('aggr', 'mean')  # mean, max, lstm
            self.gnn_module = My_GraphSAGE(in_channels=self.in_channels, hidden_channels=self.hidden_channels, num_layers=self.num_layers, aggr=aggr)


        # record layer embedding for analyzing over-smoothing
        self.layer_embd_list = []

     
    def forward(self, node_feature, adjacency_matrix):
        
       node_feature = node_feature.float()

       if node_feature.dim()==2:
           node_feature = node_feature.unsqueeze(0)
        
       bz, num_nodes, _ = node_feature.shape
       x, edge_index, edge_weight = self.transfer_data(node_feature, adjacency_matrix)
       
       attention_map_list = []

       if self.gnn_type == 'gin':
           output = self.gnn_module(x, edge_index)   # Note: output.shape=(bz*num_nodes, emb_dim)

       elif self.gnn_type == 'gcn':
           output = self.gnn_module(x, edge_index, edge_weight)

       elif self.gnn_type == 'gat':
           output, (edge_index_list, alpha_list), layer_embedding_list = self.gnn_module(x, edge_index) 

            #  output, (edge_index_list, alpha_list)= self.gnn_module(x, edge_index) 

           # average attention score
           for cur_edge_index, cur_alpha in zip(edge_index_list, alpha_list):
               attention_map_list.append(self.get_attention_score(num_nodes, cur_edge_index, cur_alpha, bz))
           
           self.gat_attention_score = attention_map_list
        
       elif self.gnn_type == 'graphsage':
            output = self.gnn_module(x, edge_index)

        
       output = output.reshape((bz, num_nodes, -1))


        # record layer embeddings
       if self.gnn_type == 'gat':
            self.layer_embd_list.clear()
            # print(f'debugging record layer embdding')

            assert len(self.layer_embd_list) == 0, "layer embd list is not cleared"
            for embd in layer_embedding_list:
                self.layer_embd_list.append(embd.reshape((bz, num_nodes, -1)).clone())  # size = number of layers, element is embeddings of this batch 

       return output


    def get_layer_embd(self):
        return self.layer_embd_list

        
    
    def get_attention_score(self, num_nodes, edge_index, alpha, batch_size):       
        '''
        Convert torch_geometric edge_index and edge_weight to a weighted adjacency matrix.
        Input:
            num_nodes: Number of nodes per graph (assumed same for all graphs in batch)
            edge_index: [2, num_edges * batch_size] edge indices
            edge_weight: [num_edges * batch_size] edge weights
            batch_size: Number of graphs in the batch
        Output:
            adjacency_matrix: [batch_size, num_nodes, num_nodes] weighted adjacency matrix
        '''

        edge_weight = torch.mean(alpha, dim = -1)   # shape = # edges

        # Initialize an empty adjacency matrix
        adjacency_matrix = torch.zeros((batch_size, num_nodes, num_nodes), dtype=edge_weight.dtype, device=edge_weight.device)

        # Compute batch offsets
        batch_offsets = torch.div(edge_index[0], num_nodes, rounding_mode='trunc') * num_nodes
        adjusted_edge_index = edge_index - batch_offsets

        # Flatten batch indices
        batch_indices = torch.div(edge_index[0], num_nodes, rounding_mode='trunc')

        # Fill the adjacency matrix
        adjacency_matrix[batch_indices, adjusted_edge_index[0] % num_nodes, adjusted_edge_index[1] % num_nodes] = edge_weight

        return adjacency_matrix

        
    def transfer_data(self, node_feature, adjacency_matrix):
        '''
        transfer input data to the form accepted by GIN
        Input:
            node_feature: [bz, num_nodes, feature_size]
            adjacency_matrix: [bz, num_nodes, num_nodes]
        Output:
            x: [bz * num_nodes, feature_size]
            edge_index: [2, num_edges in this batch]
        '''
        bz, num_nodes, feature_size = node_feature.shape

        # Flatten node features
        x = node_feature.view(bz * num_nodes, feature_size)

        # Prepare to collect edge indices for all batches
        edge_indices = []
        edge_weights = []

        # Process each batch's adjacency matrix
        for b in range(bz):
            # Get the current batch adjacency matrix
            adj = adjacency_matrix[b]
            
            # Convert adjacency matrix to edge indices
            # (row, col) pairs where adj[row, col] == 1 are edges
            rows, cols = adj.nonzero(as_tuple=True)

            # Get the weights for the edges
            weights = adj[rows, cols].float()

            # Adjust indices for flattened node feature indexing
            batch_offset = b * num_nodes
            batch_rows = rows + batch_offset
            batch_cols = cols + batch_offset

            # Stack and add to list of edge indices
            batch_edge_index = torch.stack([batch_rows, batch_cols], dim=0)
            edge_indices.append(batch_edge_index)

            # Add weights to the list of edge weights
            edge_weights.append(weights)

        # Concatenate all batch edge indices along the second dimension
        edge_index = torch.cat(edge_indices, dim=1)
        edge_weight = torch.cat(edge_weights, dim=0)


        return x, edge_index, edge_weight


    def get_attention_weights(self) -> Optional[Tensor]:
        return self.gat_attention_score




class Exp_GAT(GAT):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,**kwargs):
        super().__init__(in_channels, hidden_channels, num_layers, **kwargs)


    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        assert len(self.convs) == len(self.norms)

        edge_index_list = []
        alpha_list = []
        layer_embedding_list = []

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if (not torch.jit.is_scripting()
                    and num_sampled_nodes_per_hop is not None):
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight,
                         edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                # this code is used
                # x = conv(x, edge_index, edge_attr=edge_attr)
                x, (edge_index, alpha) = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)

                # print(f'x.shape={x.shape}, edge_index.shape={edge_index.shape}, alpha.shape={alpha.shape}')
                edge_index_list.append(edge_index.clone())
                alpha_list.append(alpha.clone())
                
            else:
                x = conv(x, edge_index)
                

            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                x = self.dropout(x)
                if hasattr(self, 'jk'):
                    xs.append(x)

            layer_embedding_list.append(x.clone())


        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x

        # return x, (edge_index_list, alpha_list)
        return x, (edge_index_list, alpha_list), layer_embedding_list





class My_GIN(GIN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,**kwargs):
        super().__init__(in_channels, hidden_channels, num_layers, **kwargs)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

        eps = kwargs.pop('eps', 0.0)
        # print(f'eps={eps}')
    
        return GINConv(mlp, eps, **kwargs)




class My_GraphSAGE(GraphSAGE):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,**kwargs):
        super().__init__(in_channels, hidden_channels, num_layers, **kwargs)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        aggr = kwargs.pop('aggr', 'mean')
        print(f'aggr={aggr}')
    
        return SAGEConv(in_channels, out_channels, aggr, **kwargs)

       

