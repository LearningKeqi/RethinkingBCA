
python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_0\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_005\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_01\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_015\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_02\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_025\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_03\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_05\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_07\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=mixup\
  training.epochs=100 training.less_epoch=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,True]\
  model.sizes=[200,100] model.dim_reduction=True\
  exp_name=bnt_abide_density_1\
  dataset.only_positive_corr=False\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=bnt\
  model.tune_sizes=[[200,100]]\
  model.tune_num_heads=[2]