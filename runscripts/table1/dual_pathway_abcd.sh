cfg={'dataset': {'name': 'abcd', 'batch_size': 16, 'test_batch_size': 16, 'val_batch_size': 16, 'train_set': 0.7, 'val_set': 0.1, 'node_feature': '/local/scratch/xkan/ABCD/abcd_rest-pearson-HCP2016.npy', 'time_seires': '/local/scratch/xkan/ABCD/ABCD/abcd_rest-timeseires-HCP2016.npy', 'node_id': '/local/scratch/xkan/ABCD/ids_HCP2016.txt', 'seires_id': '/local/scratch/xkan/ABCD/ABCD/ids_HCP2016_timeseires.txt', 'label': '/local/scratch/xkan/ABCD/ABCD/id2sex.txt', 'drop_last': True, 'stratified': True, 'cog_index': 0, 'task': 'regression', 'comb_measures_path': '/local/scratch3/khan58/Datasets/ABCD/labels37_final_comb_measures.csv', 'measure': 'pea_wiscv_trs', 'node_feature_type': 'learnable_time_series', 'node_feature_dim': 128, 'node_feature_eigen_topk': 10, 'num_bins': 100, 'time_series_hidden_size': 128, 'time_series_encoder': 'cnn', 'timeseries_embedding_type': 'last', 'timeseries_used_length': -1, 'feature_orig_or_sparse': 'orig', 'pos_enc': 'none', 'pe_dim': 16, 'pe_eigen_topk': 10, 'sparsity_topk': 30, 'pe_add_or_concat': 'concat', 'attn_bias': 'none', 'proximity_m': 5, 'sparse_ratio': 1, 'plot_figures': True, 'itype': 'fmri', 'binary_sparse': True, 'only_positive_corr': True, 'gnn_hidden_channels': 128, 'gnn_num_layers': 1, 'gnn_pool': 'concat', 'gnn_num_heads': 2, 'gin_eps': 0.0, 'sage_aggr': 'mean', 'tune_gnn_hidden_channels': [128], 'tune_gnn_num_layers': [2], 'tune_gnn_pool': ['concat'], 'tune_gnn_num_heads': [2], 'tune_gin_eps': [0.0], 'tune_sage_aggr': ['mean']}, 'model': {'name': 'Exp_BrainNetworkTransformer', 'sizes': [360, 100], 'pooling': [False, False], 'pos_encoding': 'none', 'orthogonal': True, 'freeze_center': True, 'project_assignment': True, 'pos_embed_dim': 360, 'num_heads': 4, 'hidden_size': 1024, 'readout_strategy': 'OCRead', 'sortpooling_k': 128, 'dim_reduction': False, 'all_layer_concat': False, 'input_or_layer': 'layer', 'attn_or_cosine': 'attn', 'soft_v': 'add', 'rm_x': False, 'rm_norm1': False, 'rm_norm2': False, 'rm_ffblock': False, 'simple_mlp': False, 'only_ffblock': False, 'pure_mlp': False, 'fix_attn': 'none', 'l1_reg': False, 'tanh_sym': False, 'upper_tria': False, 'one_layer_fc': True, 'delta_from_input': False, 'num_ff_block': 3, 'combine_ff': 'mean', 'simple_mask': False, 'has_nonaggr_module': True, 'has_aggr_module': True, 'combine_alpha': 0.5, 'num_combine_mlp_layer': 2, 'nonaggr_type': 'input', 'aggr_combine_type': 'concat', 'only_sa_block': False, 'aggr_module': 'gat', 'tune_sizes': [[360, 100]], 'tune_num_heads': [4]}, 'optimizer': [{'name': 'Adam', 'lr': 0.0001, 'match_rule': 'None', 'except_rule': 'None', 'no_weight_decay': False, 'weight_decay': 0.0001, 'alpha': 0.01, 'l1_ratio': 0.5, 'lr_scheduler': {'mode': 'cos', 'base_lr': 0.0001, 'target_lr': 1e-05, 'decay_factor': 0.1, 'milestones': [0.3, 0.6, 0.9], 'poly_power': 2.0, 'lr_decay': 0.98, 'warm_up_from': 0.0, 'warm_up_steps': 0, 'combine_base_lr': 0.0001, 'combine_target_lr': 1e-05}}], 'training': {'name': 'Train', 'epochs': 100, 'less_epoch': 100}, 'datasz': {'percentage': 1.0}, 'preprocess': {'name': 'continus_mixup', 'continus': True}, 'explain': {'mask_pos': 'input', 'epochs': 15, 'batch_size': 16, 'lr': 0.001, 'strategy': 'const', 'const_val': 1, 'only_first_layer': False, 'comb_exp': True, 'add_reg': False, 'select_mask_interval': 15, 'self_gated': False, 'dot_or_matmul': 'dot', 'xxt': False, 'bothsides': True}, 'repeat_time': 10, 'log_path': 'result', 'save_learnable_graph': False, 'wandb_entity': 'keqihan998', 'project': 'bnt_cog', 'ifexplain': False, 'save_mask': True, 'save_mask_interval': 5, 'exp_name': '128lay2_ep20', 'new_weight_decay': 0.0001, 'new_l1_norm_weight': 0.0001, 'new_alpha': 0.1, 'new_l1_ratio': 0.5, 'ig_steps': 20, 'cal_ig': False, 'topk_important_features_mlp': 200, 'topk_important_features_ig': 200, 'x_topk': 200, 'num_runs_save_interpretability': 2, 'save_mlp_weight': True, 'has_self_loop': False, 'new_learning_rates': [0.0001, 1e-05], 'tune_new_learning_rates': [[0.0001, 1e-05]], 'pretrain_lower_epoch': 20, 'combine_learning_rates': [0.0001, 1e-05], 'tune_combine_learning_rates': [[1e-05, 1e-06]], 'pretrain_nonaggr_coef': 1, 'nonaggr_coef': 1, 'aggr_coef': 1, 'aggr_first': True}



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=learnable_time_series\
  model.pooling=[False,False]\
  model.sizes=[360,100] model.dim_reduction=False\
  exp_name=dual_pathway_abcd\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  dataset.time_series_hidden_size=128\
  dataset.time_series_encoder=cnn\
  dataset.node_feature_dim=128\
  dataset.gnn_hidden_channels=128\
  dataset.gnn_num_layers=1\
  model.has_nonaggr_module=True\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gat\
  model.aggr_combine_type=concat\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  dataset.plot_figures=True\
  dataset.tune_gnn_num_layers=[2]\
  dataset.batch_size=16\
  tune_combine_learning_rates=[[1.0e-5,1.0e-6]]\
  pretrain_lower_epoch=20\
  pretrain_nonaggr_coef=1\
  nonaggr_coef=1\
  dataset.tune_gnn_hidden_channels=[128]\
  save_mlp_weight=True
