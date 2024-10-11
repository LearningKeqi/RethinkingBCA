python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,64] model.dim_reduction=True\
  exp_name=nores_bnt_abcd_density0\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  model.only_sa_block=True\
  new_weight_decay=1.0e-4



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,64] model.dim_reduction=True\
  exp_name=nores_bnt_abcd_density005\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  model.only_sa_block=True\
  new_weight_decay=1.0e-4



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,64] model.dim_reduction=True\
  exp_name=nores_bnt_abcd_density01\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  model.only_sa_block=True\
  new_weight_decay=1.0e-4



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,64] model.dim_reduction=True\
  exp_name=nores_bnt_abcd_density015\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  model.only_sa_block=True\
  new_weight_decay=1.0e-4



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,64] model.dim_reduction=True\
  exp_name=nores_bnt_abcd_density02\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  model.only_sa_block=True\
  new_weight_decay=1.0e-4



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,64] model.dim_reduction=True\
  exp_name=nores_bnt_abcd_density03\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  model.only_sa_block=True\
  new_weight_decay=1.0e-4




python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,64] model.dim_reduction=True\
  exp_name=nores_bnt_abcd_density05\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  model.only_sa_block=True\
  new_weight_decay=1.0e-4



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,64] model.dim_reduction=True\
  exp_name=nores_bnt_abcd_density07\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  model.only_sa_block=True\
  new_weight_decay=1.0e-4



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 preprocess=mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[360,64] model.dim_reduction=True\
  exp_name=nores_bnt_abcd_density1\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[360,10]]\
  model.tune_num_heads=[4]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  model.only_sa_block=True\
  new_weight_decay=1.0e-4