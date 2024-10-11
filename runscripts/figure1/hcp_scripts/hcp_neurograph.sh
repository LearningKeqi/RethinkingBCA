python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  ataset.node_feature_type=connection\
  exp_name=neurograph_hcp\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=True\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]
