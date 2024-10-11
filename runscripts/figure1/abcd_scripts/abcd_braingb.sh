python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2