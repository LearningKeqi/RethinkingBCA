from load_dataset import *
from scipy import stats
import argparse
from ml_baseline_models.para_cpm import cpm_train_and_evaluate
from ml_baseline_models.para_cmep import cmep_train_and_evaluate
from ml_baseline_models.para_svm import svm_train_and_evaluate
from ml_baseline_models.para_linear import linear_train_and_evaluate
from ml_baseline_models.para_kernel_ridge_reg import kernel_ridge_reg_train_and_evaluate
from ml_baseline_models.para_random_forest import random_forest_train_and_evaluate
from ml_baseline_models.para_elastic_net import elastic_net_train_and_evaluate
from ml_baseline_models.para_naive_bayes import naive_bayes_train_and_evaluate
import logging
from scipy.stats import ttest_ind
import random
import itertools
from collections import defaultdict
import pprint
import pandas as pd


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def load_dataset(args):
    if args.dataset_name == 'ABCD':
        adj, y = load_abcd_data(args)
    elif args.dataset_name == 'ABIDE':
        adj, y = load_abide_data(args)
    elif args.dataset_name == 'HCP':
        adj, y = load_hcp_data(args)
    elif args.dataset_name == 'PNC':
        adj, y = load_pnc_data(args)

    return adj, y



def feature_selection_corr(adj, y):
    num_nodes = adj.shape[1]
    cc = np.zeros((num_nodes, num_nodes))   # pearson correlation between each edge and labels
    p_values = np.ones((num_nodes, num_nodes)) 

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if i==j:
                cc[i,j] = 0
                p_values[i,j] = 1 
            else:
                corr, p = stats.pearsonr(y,adj[:,i,j])
                cc[i,j] = corr
                p_values[i,j] = p

    return cc, p_values



def feature_selection_ttest(adj, y):
    num_subjects, num_nodes = adj.shape[0], adj.shape[1]
    X = adj
    p_values_mat = np.ones((num_nodes,num_nodes))
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            group1 = X[y == 0, i, j]
            group2 = X[y == 1, i, j]
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False) 
            p_values_mat[i,j] = p_val

    return None, p_values_mat



def load_indices(args, cur_repeat):

    strafied_flag = True

    load_path = f'/local/scratch3/khan58/BCA_Code_avail/RethinkBCA/exp_results/split_with_valid/{args.dataset_name.lower()}_repeat{str(cur_repeat)}_indices.csv'

    indices = pd.read_csv(load_path)

    train_index = indices['Train Index'].dropna().to_numpy().astype(int)
    valid_index = indices['Valid Index'].dropna().to_numpy().astype(int)
    test_index = indices['Test Index'].dropna().to_numpy().astype(int)

    indices = (train_index, valid_index, test_index)

    return indices


if __name__ == "__main__":
    set_seed()

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str,
                        choices=['ABCD', 'PNC', 'ABIDE', 'HCP'],
                        default="ABCD")
    parser.add_argument('--model_name', type=str, default='cpm')
    parser.add_argument('--p_threshold_list', nargs='+', type=float, default=[0.001, 0.01, 0.05, 0.1, 0.5, 1], help='List of p-values')
    parser.add_argument('--topk_feature_list', nargs='+', type=int, default=[100, 200, 500, 1000, 5000, 1], help='List of topk feature, special case, topk = 1 denotes w/o dim reduction')
    parser.add_argument('--measure', type=str)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--cpm_type', type=str, default='positive', help='positive, negative, combine')
    parser.add_argument('--view', type=int, default=2)  # original value is 1
    parser.add_argument('--feature_selection_type', type=str, help='corr or ttest') 

    args = parser.parse_args()
    logging.info(f'\n\n\n=================================================')
    logging.info(f'args={args}')
    
    logging.info(f'Start Exp [{args.model_name}] on [{args.dataset_name}] dataset with measure [{args.measure}]')

    # load dataset
    adj, y = load_dataset(args)
    logging.info(f'load dataset {args.dataset_name} done!')

    # train model and evaluate model
    if args.measure in ['sex', 'Autism','Gender','Parkinson']:  # classification task
        aucs, accs = [], []
    else:  # regression task
        mses, maes, corrs = [], [], []


    all_results = []

    for cur_repeat in range(args.repeat):  
        logging.info(f'\nRepeat {cur_repeat} starts...')                  
        indices = load_indices(args, cur_repeat)
        
        # get feature selection mask
        train_index = indices[0]
        train_adj, train_y = adj[train_index], y[train_index]

        cc, p_values = eval(f'feature_selection_{args.feature_selection_type}')(train_adj, train_y)
        logging.info(f'Calculate correlation and p-value for {args.dataset_name} - {args.measure} done!')

        cur_results = eval(f'{args.model_name}_train_and_evaluate')(args, adj, y, indices, cc, p_values, cur_repeat)

        all_results.extend(cur_results)

    # record performance

    logging.info(f'------------ Done! ------------')
    if args.measure in ['Autism','Gender']:  # classification task
        grouped_results = defaultdict(list)

        for record in all_results:
            para_key = frozenset(record['para'].items()) 
            grouped_results[para_key].append(record)

        group_avg_results = {}

        for para_key, records in grouped_results.items():
            performance_list = [record['performance'] for record in records]
            df = pd.DataFrame(performance_list)
            avg_performance = df.mean().to_dict()
            std_performance = df.std().to_dict()
            group_avg_results[para_key] = {
                'mean': avg_performance,
                'std': std_performance,
                'records': performance_list  
            }

        # find the group with higher 'valid_auc' score
        best_para_key = max(group_avg_results, key=lambda k: group_avg_results[k]['mean']['valid_auc'])
        best_parameters = dict(best_para_key)
        best_performance = group_avg_results[best_para_key]

        # print hyparameters and the corresponding average results
        logging.info(f'[Model]:{args.model_name}\n[{args.measure}]')
        print("Best Parameters:", best_parameters)

        # print the result of each single run
        print("Individual Results:")
        for record in best_performance['records']:
            print(record)

        print("Average Performance with Standard Deviation:")
        for metric in best_performance['mean'].keys():
            mean_value = best_performance['mean'][metric]
            std_value = best_performance['std'][metric]
            print(f"{metric}: {mean_value} +- {std_value}")        
    else:
        grouped_results = defaultdict(list)

        for record in all_results:
            para_key = frozenset(record['para'].items())  # 将 'para' 转换为可哈希的类型
            grouped_results[para_key].append(record)

        group_avg_results = {}

        for para_key, records in grouped_results.items():
            performance_list = [record['performance'] for record in records]
            df = pd.DataFrame(performance_list)
            avg_performance = df.mean().to_dict()
            std_performance = df.std().to_dict()
            group_avg_results[para_key] = {
                'mean': avg_performance,
                'std': std_performance,
                'records': performance_list
            }

        best_para_key = min(group_avg_results, key=lambda k: group_avg_results[k]['mean']['valid_mse'])
        best_parameters = dict(best_para_key)
        best_performance = group_avg_results[best_para_key]
        
        logging.info(f'[Model]:{args.model_name}\n[{args.measure}]')
        print("Best Parameters:", best_parameters)

        print("Individual Results:")
        for record in best_performance['records']:
            print(record)

        print("Average Performance with Standard Deviation:")
        for metric in best_performance['mean'].keys():
            mean_value = best_performance['mean'][metric]
            std_value = best_performance['std'][metric]
            print(f"{metric}: {mean_value} +- {std_value}")
