import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder, Exp_InterpretableTransformerEncoder, gatv2_Exp_InterpretableTransformerEncoder, MyGNN
from omegaconf import DictConfig
from ..base import BaseModel
import torch.nn.functional as F
import math
from ..Learnable_Nodefeature import Centrality_Nodefeature, SAN_Nodefeature, GCN_Nodefeature, DegreeBin_Nodefeature, TimeSeriesEncoder
from omegaconf import DictConfig, open_dict


class Exp_TransPoolingEncoder(nn.Module):
    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, config, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, num_heads=4):
        super().__init__()
        self.transformer = Exp_InterpretableTransformerEncoder(d_model=input_feature_size, nhead=num_heads, config = config,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)
        

        self.cfg = config


        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)


    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x, edge_mask, mask_pos, is_explain, attn_map_bias, attn_solid_mask):

        x = self.transformer(x, attn_map_bias = attn_map_bias, attn_solid_mask = attn_solid_mask)        
    
        if self.pooling:
            x, assignment = self.dec(x)

            return x, assignment

        
        return x, None


    def get_attention_weights(self):
        return self.transformer.get_attention_weights()


    def loss(self, assignment):
        return self.dec.loss(assignment)


class mixed_model(BaseModel):

    def __init__(self, config: DictConfig):
        
        super().__init__()
        self.cfg = config
        self.attention_list = nn.ModuleList()

        forward_dim = self.cfg.dataset.node_feature_dim
        print(f'forward_dim = {forward_dim}')

        ############  only used for BNT model, not for others  #######
        sizes = config.model.sizes
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        if self.cfg.model.readout_strategy != 'OCRead':   
            do_pooling[-1] = False

        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                Exp_TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=self.cfg.model.hidden_size,
                                    output_node_num=size,
                                    config=config,
                                    pooling=do_pooling[index],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment,
                                    num_heads = config.model.num_heads))


        red_dim = 8
        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, red_dim),
            nn.LeakyReLU()
        )
        
        ############  only used for BNT model, not others  #######

        if self.cfg.dataset.task == 'classification' and not self.cfg.log_reg:
            last_output_dim = 2
        elif self.cfg.dataset.task == 'classification' and self.cfg.log_reg:
            last_output_dim = 1
        elif self.cfg.dataset.task == 'regression':
            last_output_dim = 1

 
        num_nodes = config.dataset.node_sz
        if config.has_self_loop:
            residual_input_dim = int((num_nodes*(num_nodes+1))/2) 
        else:
            residual_input_dim = int((num_nodes*(num_nodes-1))/2) # do not take diagonal into consideration
        

        if self.cfg.model.has_nonaggr_module and self.cfg.model.has_aggr_module and self.cfg.model.aggr_combine_type=='concat':
            print(f'residual_input_dim={residual_input_dim}, self.cfg.dataset.gnn_hidden_channels*num_nodes={self.cfg.dataset.gnn_hidden_channels*num_nodes}')
            fc_input_dim = residual_input_dim + self.cfg.dataset.gnn_hidden_channels*num_nodes
                
        elif not self.cfg.model.has_nonaggr_module and self.cfg.model.has_aggr_module:
            if self.cfg.model.aggr_module != 'bnt':
                if self.cfg.dataset.gnn_pool == 'concat':
                    fc_input_dim = self.cfg.dataset.gnn_hidden_channels*num_nodes
                elif self.cfg.dataset.gnn_pool == 'mean':
                    fc_input_dim = self.cfg.dataset.gnn_hidden_channels
            elif self.cfg.model.aggr_module == 'bnt':
                num_features_last_year = self.cfg.model.sizes[-1] if do_pooling[-1] else num_nodes

                if self.cfg.model.dim_reduction:
                    fc_input_dim = num_features_last_year*8
                else:
                    fc_input_dim = num_features_last_year*num_nodes
                    
        
        print(f'fc_input_dim = {fc_input_dim}')

        # Final prediction layer
        if not self.cfg.model.one_layer_fc:
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(), 
                nn.Linear(32, last_output_dim)
            )
        else:
            self.nonaggr_fc = nn.Linear(residual_input_dim, last_output_dim, bias=True)
            
            if self.cfg.dataset.gnn_pool == 'mean':
                self.aggr_fc = nn.Linear(self.cfg.dataset.gnn_hidden_channels, last_output_dim, bias=True)
            else:
                self.aggr_fc = nn.Linear(self.cfg.dataset.gnn_hidden_channels*num_nodes, last_output_dim, bias=True)


        # learnable node feature
        if self.cfg.dataset.node_feature_type in ['centrality', 'degree', 'degree_bin', 'learnable_time_series', 'learnable_eigenvec', 'gnn_identity', 'gnn_eigenvec', 'gnn_connection']:
            if self.cfg.dataset.node_feature_type == 'centrality' or self.cfg.dataset.node_feature_type == 'degree':
                self.node_feature_generator = Centrality_Nodefeature(degree_dim=1, embedding_dim = self.cfg.dataset.node_feature_dim)
            elif self.cfg.dataset.node_feature_type == 'degree_bin':
                self.node_feature_generator = DegreeBin_Nodefeature(num_bins=self.cfg.dataset.num_bins, embedding_dim = self.cfg.dataset.node_feature_dim)
            elif self.cfg.dataset.node_feature_type == 'learnable_time_series':
                self.node_feature_generator = TimeSeriesEncoder(input_size=1, hidden_size=self.cfg.dataset.time_series_hidden_size, embedding_size= self.cfg.dataset.node_feature_dim, model_type=self.cfg.dataset.time_series_encoder, timeseries_embedding_type=self.cfg.dataset.timeseries_embedding_type)
            elif self.cfg.dataset.node_feature_type == 'learnable_eigenvec':
                self.node_feature_generator = SAN_Nodefeature(m = self.cfg.dataset.node_feature_eigen_topk, k=self.cfg.dataset.node_feature_dim)

        
        
        if self.cfg.model.has_aggr_module and self.cfg.model.aggr_module in ['gcn','gat','gin','graphsage']:
            if self.cfg.dataset.node_feature_type == 'orig_time_series':
                gnn_inchannels = self.cfg.dataset.time_series_input_size
            else:
                gnn_inchannels = self.cfg.dataset.node_feature_dim

            self.gnn_module = MyGNN(inchannels=gnn_inchannels, hidden_channels=self.cfg.dataset.gnn_hidden_channels, 
                                    num_layers=self.cfg.dataset.gnn_num_layers, gnn_type=self.cfg.model.aggr_module, 
                                    num_heads=self.cfg.dataset.gnn_num_heads, eps=self.cfg.dataset.gin_eps, aggr=self.cfg.dataset.sage_aggr)

        if self.cfg.model.has_aggr_module and self.cfg.model.aggr_module in ['gcn','gat','gin','graphsage']:
            self.norm0 = nn.LayerNorm(self.cfg.dataset.gnn_hidden_channels)
        else:
            self.norm0 = nn.LayerNorm(self.cfg.dataset.node_sz)

        self.nonaggr_coefficient = self.cfg.nonaggr_coef
        self.aggr_coefficient = self.cfg.aggr_coef




    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor,
                edge_mask: torch.tensor,
                mask_pos: str,
                is_explain: bool,
                orig_connection: torch.tensor,
                saved_eigenvectors: torch.tensor,
                sparse_connection: torch.tensor):
        
        
        
        bz, _, _, = node_feature.shape

        assignments = []

        # learnable node feature
        if self.cfg.dataset.node_feature_type in ['centrality', 'degree', 'degree_bin', 'learnable_time_series','learnable_eigenvec']:
            node_feature = self.node_feature_generator(node_feature)

        # obtain attn_solid_mask from sparse_connection
        attn_solid_mask = torch.full_like(sparse_connection, float('-inf')).cuda()
        attn_solid_mask[sparse_connection != 0] = 0


        layer_cnt = 0

        # input of the linear pathway model
        if self.cfg.model.has_nonaggr_module:
            LM_feature = orig_connection
        else:
            LM_feature = None


        if self.cfg.model.has_aggr_module:
            if self.cfg.model.aggr_module == 'bnt':
                aggr_feature = node_feature.clone()
                for atten in self.attention_list:
                    aggr_feature, assignment = atten(aggr_feature, edge_mask, mask_pos, is_explain, attn_map_bias = None, attn_solid_mask = attn_solid_mask)
                    assignments.append(assignment)
                    layer_cnt += 1

            elif self.cfg.model.aggr_module in['gcn','gat','gin','graphsage']:
                aggr_feature = self.gnn_module(node_feature, sparse_connection)
        else:
            aggr_feature = None

        if self.cfg.model.has_nonaggr_module and self.cfg.model.has_aggr_module:
            aggr_feature = self.norm0(aggr_feature)


        if self.cfg.model.has_aggr_module and self.cfg.model.aggr_module == 'bnt' and self.cfg.model.dim_reduction:   # used for BNT model
            aggr_feature = self.dim_reduction(aggr_feature)
        
        # pooling strategy for gnn
        if not self.cfg.model.has_nonaggr_module and self.cfg.model.has_aggr_module and self.cfg.model.aggr_module != 'bnt' and self.cfg.dataset.gnn_pool == 'mean':
            aggr_feature = torch.mean(aggr_feature, dim=1)
        else:
            aggr_feature = aggr_feature.reshape((bz, -1))
             

        # input concat other feature
        if self.cfg.model.has_nonaggr_module and self.cfg.model.nonaggr_type == 'input':
            num_nodes = orig_connection.shape[-1]
            offset = 0 if self.cfg.has_self_loop else 1
            rows, cols = torch.triu_indices(row=num_nodes, col=num_nodes, offset=offset)  # offset 0 including diag

            LM_feature = LM_feature[:, rows, cols]

            if self.cfg.model.has_aggr_module:
                aggr_feature = aggr_feature.reshape((bz, -1))
                                
                # separate fc into two parts
                nonaggr_output = self.nonaggr_fc(self.nonaggr_coefficient * LM_feature)
                aggr_output = self.aggr_fc(self.aggr_coefficient * aggr_feature)

                return nonaggr_output + aggr_output


            else:
                nonaggr_output = self.nonaggr_fc(LM_feature)

                return nonaggr_output
            

        if not self.cfg.model.has_nonaggr_module and self.cfg.model.aggr_module=='bnt':
            return self.fc(aggr_feature)
        elif not self.cfg.model.has_nonaggr_module and self.cfg.model.aggr_module!='bnt':
            return self.aggr_fc(aggr_feature)


    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]
    

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
