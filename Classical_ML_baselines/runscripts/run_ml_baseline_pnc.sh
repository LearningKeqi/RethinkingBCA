
python -u ../para_baseline_main.py --dataset_name PNC\
 --measure sex --model_name cpm --feature_selection_type corr --repeat 10


python -u ../para_baseline_main.py --dataset_name PNC\
 --measure sex --model_name cpm --cpm_type negative --feature_selection_type corr --repeat 10


python -u ../para_baseline_main.py --dataset_name PNC\
 --measure sex --model_name cmep --feature_selection_type corr --repeat 10


python -u ../para_baseline_main.py --dataset_name PNC\
 --measure sex --model_name linear --feature_selection_type ttest --topk_feature_list 100 200 500 1000 5000 10000 1\
 --repeat 10


python -u ../para_baseline_main.py --dataset_name PNC\
 --measure sex --model_name elastic_net --feature_selection_type ttest --topk_feature_list 100 200 500 1000 5000 10000 1\
 --repeat 10


python -u ../para_baseline_main.py --dataset_name PNC\
 --measure sex --model_name svm --feature_selection_type ttest --topk_feature_list 100 200 500 1000 5000 10000 1


python -u ../para_baseline_main.py --dataset_name PNC\
 --measure sex --model_name random_forest --feature_selection_type ttest --topk_feature_list 100 200 500 1000 5000 10000 1\
 --repeat 10


python -u ../para_baseline_main.py --dataset_name PNC\
 --measure sex --model_name naive_bayes --feature_selection_type ttest --topk_feature_list 100 200 500 1000 5000 10000 1\
 --repeat 10