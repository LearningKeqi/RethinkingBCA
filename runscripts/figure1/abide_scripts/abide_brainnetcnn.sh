python -u -m source --multirun datasz=100p model=brainnetcnn\
  dataset=ABIDE repeat_time=10 preprocess=non_mixup\
  training.epochs=100\
  dataset.measure=Autism\
  exp_name=ABIDE_brainnetcnn\
  model.tune_dim1=[32]\
  model.tune_dim2=[64]\
  model.tune_dropout_rate=[0.3]
