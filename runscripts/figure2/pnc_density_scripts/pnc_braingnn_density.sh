python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density000\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]


python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density005\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]



python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density01\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]



python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density015\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]

python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density02\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]

python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density025\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]

python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density03\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]

python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density05\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]

python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density07\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]

python -u -m source --multirun datasz=100p model=braingnn_orig\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=braingnn_pnc_density1\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  tune_new_learning_rates=[[5.0e-3,5.0e-4]]\
  dataset.stratified=True\
  model.tune_dim1=[128]\
  model.tune_dim2=[128]\
  model.tune_lamb1=[0.1]\
  model.tune_lamb2=[0]