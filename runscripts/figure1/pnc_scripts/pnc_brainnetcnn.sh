python -u -m source --multirun datasz=100p model=brainnetcnn\
  dataset=PNC repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=sex\
  exp_name=brainnetcnn_pnc\
  model.tune_dim1=[32]\
  model.tune_dim2=[32]\
  model.tune_dropout_rate=[0.3]
