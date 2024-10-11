python -u -m source --multirun datasz=100p model=braingb\
  dataset=PNC repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingb_pnc\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]