python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  exp_name=density_ABIDE_braingnn_0\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0]\
  model.tune_lamb2=[0]

