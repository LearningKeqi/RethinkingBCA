python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density0\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density005\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density01\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density015\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density02\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density025\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density03\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density05\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density07\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gin_abide_density1\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gin\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gin_eps=[0.5]
