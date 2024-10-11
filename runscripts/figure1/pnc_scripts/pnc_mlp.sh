python -u -m source --multirun datasz=100p model=mlp_graph\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  exp_name=PNC_mlp_graph_layer2\
  model.num_mlp_layer=2



python -u -m source --multirun datasz=100p model=mlp_node\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  exp_name=PNC_mlp_node\
  dataset.num_bins=100\