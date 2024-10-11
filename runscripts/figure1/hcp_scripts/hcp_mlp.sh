python -u -m source --multirun datasz=100p model=mlp_graph\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  exp_name=hcp_mlp_graph_layer2\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]



python -u -m source --multirun datasz=100p model=mlp_node\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=PMAT24_A_CR\
  exp_name=hcp_mlp_node\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]
