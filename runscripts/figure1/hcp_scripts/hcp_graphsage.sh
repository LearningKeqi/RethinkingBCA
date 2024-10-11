python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=graphsage_hcp\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]