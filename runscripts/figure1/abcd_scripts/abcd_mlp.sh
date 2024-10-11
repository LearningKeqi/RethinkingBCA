python -u -m source --multirun datasz=100p model=mlp_graph\
  dataset=ABCD repeat_time=10 preprocess=non_mixup\
  dataset.measure=pea_wiscv_trs\
  exp_name=ABCD_mlp_graph_layer2\
  model.num_mlp_layer=2



python -u -m source --multirun datasz=100p model=mlp_node\
  dataset=ABCD repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  exp_name=ABCD_mlp_node\
