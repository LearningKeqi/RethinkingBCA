import numpy as np
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
import copy


def convert_2_feature(adj, indices, p_values, topk):       
    cur_adj = adj[indices]
    num_subjets, num_nodes = cur_adj.shape[0], cur_adj.shape[1]
    cur_adj = cur_adj.reshape(num_subjets, -1)   # num_subjects, num_nodes*num_nodes
    
    if topk == 1 or topk > num_nodes*(num_nodes-1)/2:  # special case, topk = 1 denotes w/o dim reduction
        topk = int(num_nodes*(num_nodes-1)/2)

    flatten_p_values = p_values.reshape(1, -1).squeeze()
    sorted_indices = np.argsort(flatten_p_values)[:topk]

    features = cur_adj[:, sorted_indices]

    return features


def kernel_ridge_reg_train_and_evaluate(args, adj, y, indices, cc, p_values, cur_repeat):
    results_list = []

    train_indices = indices[0]
    valid_indices = indices[1]
    test_indices = indices[2]

    y_train, y_valid, y_test = y[train_indices], y[valid_indices], y[test_indices]

    for topk in args.topk_feature_list:
        logging.info(f'Start with topk:{topk}...')

        train_features = convert_2_feature(adj, train_indices, p_values, topk)
        valid_features = convert_2_feature(adj, valid_indices, p_values, topk)
        test_features = convert_2_feature(adj, test_indices, p_values, topk)


        print(f'feature_dim = {train_features.shape}')

        param_combinations = [
            {'alpha': 0.1, 'kernel': 'linear'},
            {'alpha': 1, 'kernel': 'linear'},
            {'alpha': 10, 'kernel': 'linear'},
            {'alpha': 1, 'kernel': 'rbf', 'gamma': 0.1},
            {'alpha': 1, 'kernel': 'rbf', 'gamma': 1},
            {'alpha': 1, 'kernel': 'poly', 'degree': 3},
            {'alpha': 0.1, 'kernel': 'poly', 'degree': 3},
            {'alpha': 1, 'kernel': 'poly', 'degree': 2}
        ]
        
        for params in param_combinations:
            model = KernelRidge(**params)
            model.fit(train_features, y_train)
            predictions = model.predict(valid_features)
            valid_score = mean_squared_error(y_valid, predictions)

            final_predictions = model.predict(test_features)
            best_mse_test = mean_squared_error(y_test, final_predictions)
            best_mae_test = mean_absolute_error(y_test, final_predictions)
            best_corr_test = np.corrcoef(y_test, final_predictions)[0,1]

            cur_para = copy.deepcopy(params)
            cur_para['topk']=topk

            cur_performance = {}
            cur_performance['valid_mse']=valid_score
            cur_performance['test_mse']=best_mse_test
            cur_performance['test_mae']=best_mae_test
            cur_performance['test_corr']=best_corr_test

            cur_result = {}
            cur_result['para'] = cur_para
            cur_result['repeat'] = cur_repeat
            cur_result['performance'] = cur_performance        

            results_list.append(cur_result)        


    return results_list
        