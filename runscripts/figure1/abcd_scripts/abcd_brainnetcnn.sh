
python -u -m source --multirun datasz=100p model=brainnetcnn\
  dataset=ABCD repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=pea_wiscv_trs\
  exp_name=brainnetcnn_abcd\
  model.tune_dim1=[32]\
  model.tune_dim2=[32]\
  model.tune_dropout_rate=[0.3]\
  tune_new_learning_rates=[[1.0e-4,1.0e-5]]