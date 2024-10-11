python -u -m source --multirun datasz=100p model=neurograph\
  dataset=ABCD repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=neurograph_abcd\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=True\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-4