
python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_0\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_005\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_01\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_015\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_02\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_025\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_03\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_05\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_07\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]

python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  dataset.node_feature_type=connection\
  model.pooling=[False,False]\
  model.sizes=[200,200] model.dim_reduction=False\
  exp_name=gcn_abide_density_1\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=False\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=gcn\
  dataset.tune_gnn_num_layers=[2]\
  dataset.tune_gnn_hidden_channels=[128]\
  dataset.tune_gnn_pool=[concat]


