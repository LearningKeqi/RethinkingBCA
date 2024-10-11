python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density0\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density005\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density01\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density015\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density02\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density025\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density03\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density05\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density07\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=HCP repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[132,64] model.dim_reduction=True\
  exp_name=bnt_hcp_density1\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[132,5]]\
  model.tune_num_heads=[2]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]