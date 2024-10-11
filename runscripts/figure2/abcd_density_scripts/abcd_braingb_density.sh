
python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density0\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2


python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density005\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density01\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density015\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density02\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density025\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density03\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density05\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density07\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2



python -u -m source --multirun datasz=100p model=braingb\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  exp_name=braingb_abcd_density1\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_pooling=[concat]\
  model.tune_hidden_dim=[256]\
  model.tune_gcn_mp_type=[node_concate]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]\
  new_weight_decay=1.0e-2