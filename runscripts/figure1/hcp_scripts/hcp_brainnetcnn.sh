
python -u -m source --multirun datasz=100p model=brainnetcnn\
  dataset=HCP repeat_time=10 preprocess=non_mixup\
  training.epochs=100 \
  dataset.measure=PMAT24_A_CR\
  exp_name=brainnetcnn_hcp\
  model.tune_dim1=[32]\
  model.tune_dim2=[32]\
  model.tune_dropout_rate=[0.3]\
  tune_new_learning_rates=[[5.0e-4,5.0e-5]]
