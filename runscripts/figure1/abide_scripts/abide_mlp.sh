python -u -m source --multirun datasz=100p model=mlp_graph\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  exp_name=ABIDE_mlp_graph_layer2\
  model.num_mlp_layer=2


python -u -m source --multirun datasz=100p model=mlp_node\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\ 
  dataset.measure=Autism\
  exp_name=ABIDE_mlp_node\
  model.num_mlp_layer=2
