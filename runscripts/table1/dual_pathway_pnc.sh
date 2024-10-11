python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=learnable_time_series\
  model.pooling=[False,False]\
  model.sizes=[264,100] model.dim_reduction=False\
  exp_name=dual_pathway_pnc\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  dataset.time_series_hidden_size=32\
  dataset.time_series_encoder=cnn\
  dataset.node_feature_dim=32\
  dataset.gnn_hidden_channels=32\
  dataset.gnn_num_layers=1\
  model.has_nonaggr_module=True\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gat\
  model.aggr_combine_type=concat\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  dataset.plot_figures=True\
  dataset.tune_gnn_num_layers=[1]\
  dataset.batch_size=16\
  tune_combine_learning_rates=[[1.0e-4,1.0e-5]]\
  pretrain_lower_epoch=20\
  pretrain_nonaggr_coef=1\
  nonaggr_coef=1\
  dataset.tune_gnn_hidden_channels=[32]\
  save_mlp_weight=True\
  draw_heatmap=False\
  new_weight_decay=1.0e-4