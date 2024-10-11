python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=ABCD_gin\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_gin_eps=[0.2]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]