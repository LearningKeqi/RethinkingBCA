python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_0\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]



python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_005\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]



python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_01\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_015\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_02\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_025\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_03\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_05\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_07\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=neurograph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=density_PNC_nores_neurograph_1\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_residual=False\
  model.has_mid_layer=True\
  model.tune_num_layers=[2]\
  model.tune_hidden_channels=[128]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


