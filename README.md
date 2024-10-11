# Rethinking Brain Connectome Analysis: Do Aggregation-based Graph Deep Learning Models Help?


In this work, we investigate whether Graph Deep Learning models are effective for Brain Connectome Analysis.


From this repository, you can find the code for the baseline experiments and 
the proposed dual-pathway model, along with detailed experimental settings and 
parameters used in each part of the study.

---

## Repository Organization

- **classical_ML_baselines**: this folder contains all the implementation for the Classical Machine Learning models used in our work.
- **plot_figures**: this folder includes the summarized experimental results and the scripts to plot the figures in our work.
- **runscripts**: it includes all the running scripts/commands for graph deep learning models used in this work.
- **source**: this folder contains the implementation of all the Graph Deep Learning models used in our work.

---


## Datasets
All the datasets used in this work are publicly available after applying for access. The **ABCD** data is publicly available via the [NIMH Data Archive (NDA)](https://nda.nih.gov/abcd), 
and the **HCP** data through [ConnectomeDB](https://db.humanconnectome.org/). 
The **PNC** data can be accessed through the [Philadelphia Neurodevelopmental Cohort (PNC) initiative](https://www.nitrc.org/projects/pnc/). 
Access to the ABCD, HCP, and PNC datasets requires an application. The **ABIDE** data is publicly accessible without 
restrictions through the [Preprocessed Connectomes Project (PCP)](http://preprocessed-connectomes-project.org/abide). This study does not involve any new datasets. 
All datasets used, along with their preprocessing steps, are properly cited within the text.



## Dependencies

  - python=3.8.10
  - pytorch=1.12.1+cu116
  - torch-geometric=2.5.1
  - torch-scatter=2.1.0+pt112cu116
  - torch-sparse=0.6.16+pt112cu116
  - scikit-learn=1.1.1
  - pandas=2.0.3   
  - numpy=1.24.4
  - scipy=1.10.1
  - wandb=0.16.4


## Usage

Run the following command to train the **deep learning models** used in this study.

```bash
python -u -m source --multirun datasz=100p model=mixed_model\
  dataset=ABCD repeat_time=10 training.epochs=100 dataset.measure=pea_wiscv_trs\
  dataset.node_feature_type=learnable_time_series exp_name=dual_pathway_abcd\
  dataset.only_positive_corr=True dataset.sparse_ratio=1 dataset.binary_sparse=True\
  dataset.time_series_hidden_size=128 dataset.time_series_encoder=cnn\
  dataset.node_feature_dim=128 dataset.gnn_hidden_channels=128\
  dataset.gnn_num_layers=1 model.one_layer_fc=True model.has_nonaggr_module=True\
  model.has_aggr_module=True model.aggr_module=gat tune_new_learning_rates=[[1.0e-4,1.0e-5]]\
  pretrain_lower_epoch=20
```
- **datasz**, default=100p, optional values: (10p, 20p, 30p, 40p, 50p, 60p, 70p, 80p, 90p, 100p). How much data to use for training. The
  value is a percentage of the total number of samples in the dataset. For example, 10p means 10% of the total number of samples in the training set.
  
- **model**, optional values: (mixed_model, braingb, braingnn_orig, brainnetcnn, mlp_graph, mlp_node, neurograph).
  Notably, 'mixed_model' includes GCN, GAT, GIN, GraphSage, Brain Network Tranformer and the proposed Dual-pathway model,
  it needs to be used in combination with parameters 'model.has_nonaggr_module','model.has_aggr_module' and 'model.aggr_module'.
  
- **dataset**, optional values: (ABCD, HCP, ABIDE, PNC)
  
- **repeat_time**, number of cross-validation runs, default=10
  
- **dataset.measure**, measures (labels) needed to be predicted. ABCD:pea_wiscv_trs (fluid intelligence), HCP:PMAT24_A_CR (fluid intelligence)
  ABIDE:Autism, PNC:sex

- **dataset.node_feature_type**, choice of node feature for graph deep learning models' input, optional values: (connection, learnable_time_series).
  'conncetion' denotes connection profile. 'learnable_time_series' denotes learnable node features from BOLD timeseries.

- **dataset.only_positive_corr**, retain only positive correlation as edges in the brain networks or keep both positive and negative conncetions.

- **dataset.sparse_ratio**, graph densities, retaining the top K% edges in the graphs. Value ranges from (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1).
  Notably, dataset.sparse_ratio=0 denotes no edges exist between ROIs, meaning node features are transformed independently without any aggregation.

- **model.has_nonaggr_module, model.has_aggr_module, model.aggr_module**. When training the GCN/GAT/GIN/GraphSage/BrainNetTF model,
  set model.has_nonaggr_module=False, model.has_aggr_module=True, model.aggr_module=gcn/gat/gin/graphsage/bnt.
  On the other hand, when training the Dual-pathway model, set model.has_nonaggr_module=True, model.has_aggr_module=True, model.aggr_module=gat.

- **pretrain_lower_epoch**, only used for the proposed Dual-pathway model. Number of independent training epochs for the GAT pathway of the proposed Dual-pathway model.

More running parameters could be referred to the `\source\conf` folders and the provided running scripts for each part of the study.

---

Run the following command to train the **classical machine learning models** used in this study.

```bash
python -u ../para_baseline_main.py --dataset_name ABCD --measure pea_wiscv_trs\
 --model_name linear --feature_selection_type corr --topk_feature_list 100 200 500 1000 5000 10000 20000 1\
 --repeat 10
```
- **dataset_name**, optional values: (ABCD, HCP, ABIDE, PNC)

- **measure**, measures (labels) needed to be predicted. ABCD:pea_wiscv_trs (fluid intelligence), HCP:PMAT24_A_CR (fluid intelligence)
  ABIDE:Autism, PNC:sex

- **model_name**, classical ML model used to train and make prediction.
  Optional values: (cpm, cmep, linear, elastic_net, svm, random_forest, kernel_ridge_reg, naive_bayes)

- **feature_selection_type**, metrics used to conduct feature selection. optional values: (corr, ttest). 'corr' denotes using p-value to the correlation between
each connection (feature) and the target variable as the selection metrics. 'ttest' denotes using p-values determined through a t-test as the selection metrics.

- **topk_feature_list**, number of features selected to train the model and make predictions.


## Reproducing the results

### Figure 1
The running scripts (commands) and the corresponding hyparameters to get the results shown in Fig. 1 can be found at `runscripts/figure1` and `classical_ML_baselines/runscripts`.


### Figure 2
The running scripts (commands) and the corresponding hyparameters to get the results shown in Fig. 2 can be found at `runscripts/figure2`.

### Table 1
The running scripts (commands) and the corresponding hyparameters to get the results shown in Tab. 1 can be found at `runscripts/table1`.



## Plotting the results
The summarized experimental results from this study (including Fig.1, Fig.2, Tab.1 and parts of the interpertability analysis) 
and the scripts to plot the figures can be found at `plot_figures`.




