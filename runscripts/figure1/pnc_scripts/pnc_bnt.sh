python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[264,100] model.dim_reduction=True\
  exp_name=bnt_pnc\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[264,5]]\
  model.tune_num_heads=[4]