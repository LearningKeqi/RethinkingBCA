
python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]
