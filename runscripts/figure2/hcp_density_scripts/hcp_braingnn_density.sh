python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density0\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]



python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density005\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]



python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density01\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]




python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density015\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]


python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density02\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]


python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density025\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]


python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density03\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]


python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density05\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]


python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density07\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]


python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp_density1\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]

  