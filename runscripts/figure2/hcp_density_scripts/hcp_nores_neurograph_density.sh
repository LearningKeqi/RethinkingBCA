python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density0\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density005\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density01\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density015\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density02\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density025\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density03\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density05\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density07\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=HCP_nores_neurograph_density1\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[3]\
  model.tune_hidden_channels=[64]\
  tune_new_learning_rates=[[1.0e-3,1.0e-4]]\
  new_weight_decay=1.0e-3


