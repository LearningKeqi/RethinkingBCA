python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density0\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density005\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.05\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]



python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density01\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density015\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.15\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density02\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.2\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density025\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.25\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density03\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.3\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density05\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.5\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density07\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=0.7\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]


python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  dataset.node_feature_type=connection\
  exp_name=graphsage_pnc_density1\
  model.one_layer_fc=True\
  dataset.only_positive_corr=True\
  dataset.sparse_ratio=1\
  dataset.feature_orig_or_sparse=orig\
  dataset.binary_sparse=True\
  model.has_nonaggr_module=False\
  model.nonaggr_type=input\
  model.has_aggr_module=True\
  model.aggr_module=graphsage\
  dataset.tune_gnn_num_layers=[3]\
  dataset.tune_gnn_hidden_channels=[32]\
  dataset.tune_sage_aggr=[mean]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]
