
python -u ../para_baseline_main.py --dataset_name HCP_CARL\
 --measure PMAT24_A_CR --model_name cpm --feature_selection_type corr --repeat 10


python -u ../para_baseline_main.py --dataset_name HCP_CARL\
 --measure PMAT24_A_CR --model_name cpm --cpm_type negative --feature_selection_type corr --repeat 10


python -u ../para_baseline_main.py --dataset_name HCP_CARL\
 --measure PMAT24_A_CR --model_name cmep --feature_selection_type corr --repeat 10


python -u ../para_baseline_main.py --dataset_name HCP_CARL\
 --measure PMAT24_A_CR --model_name linear --feature_selection_type corr --topk_feature_list 100 200 500 1000 5000 10000\
 --repeat 10


python -u ../para_baseline_main.py --dataset_name HCP_CARL\
 --measure PMAT24_A_CR --model_name elastic_net --feature_selection_type corr --topk_feature_list 100 200 500 1000 5000 10000\
 --repeat 10


python -u ../para_baseline_main.py --dataset_name HCP_CARL\
 --measure PMAT24_A_CR --model_name svm --feature_selection_type corr --topk_feature_list 100 200 500 1000 5000 10000


python -u ../para_baseline_main.py --dataset_name HCP_CARL\
 --measure PMAT24_A_CR --model_name random_forest --feature_selection_type corr --topk_feature_list 100 200 500 1000 5000 10000\
 --repeat 10


python -u ../para_baseline_main.py --dataset_name HCP_CARL\
 --measure PMAT24_A_CR --model_name kernel_ridge_reg --feature_selection_type corr --topk_feature_list 100 200 500 1000 5000 10000\
 --repeat 10