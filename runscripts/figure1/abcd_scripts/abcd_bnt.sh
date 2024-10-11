python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,100] model.dim_reduction=True\
  exp_name=ABCD_bnt\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]
