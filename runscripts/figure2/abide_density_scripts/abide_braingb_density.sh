python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_0\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]


python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_005\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]


python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_01\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]


python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_015\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_02\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_025\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]


python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_03\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]


python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_05\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]


python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_07\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingb_1\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[1e-2,1e-3]]
