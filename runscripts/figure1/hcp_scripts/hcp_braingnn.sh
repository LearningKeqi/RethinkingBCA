python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  exp_name=braingnn_hcp\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.tune_dim1=[32]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]
