import numpy as np
import logging
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
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


def naive_bayes_train_and_evaluate(args, adj, y, indices, cc, p_values, cur_repeat):
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

        if args.measure in ['sex', 'Autism','Gender','Parkinson']:
            best_score = 0
            best_auc_test = 0
            best_acc_test = 0

            logging.info(f'start training GaussianNB')
            model = GaussianNB()
            model.fit(train_features, y_train)
            valid_score = roc_auc_score(y_valid, model.predict_proba(valid_features)[:, 1])

            predictions_test = model.predict_proba(test_features)[:, 1]
            best_auc_test = roc_auc_score(y_test, predictions_test)
            predictions_test[predictions_test>0.5]=1
            predictions_test[predictions_test<=0.5]=0
            best_acc_test = accuracy_score(y_test, predictions_test)

            
            confusion_mat = confusion_matrix(y_test, predictions_test)
            tn, fp, fn, tp = confusion_mat.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            # save result
            cur_para = {}
            cur_para['topk']=topk

            cur_performance = {}
            cur_performance['valid_auc']=valid_score
            cur_performance['test_auc']=best_auc_test
            cur_performance['test_acc']=best_acc_test
            cur_performance['test_sens']=sensitivity
            cur_performance['test_spec']=specificity

            cur_result = {}
            cur_result['para'] = cur_para
            cur_result['repeat'] = cur_repeat
            cur_result['performance'] = cur_performance        

            results_list.append(cur_result)

        else:
            best_score = np.inf
            best_mse_test = np.inf
            best_mae_test = np.inf
            best_corr_test = 0
            
            logging.warning("GaussianNB is typically used for classification tasks. For regression tasks, consider using appropriate regression models.")
            
            model = GaussianNB()
            model.fit(train_features, y_train)
            predictions = model.predict(valid_features)
            score = mean_squared_error(y_valid, predictions)
            if score < best_score:
                best_score = score

                final_predictions = model.predict(test_features)
                best_mse_test = mean_squared_error(y_test, final_predictions)
                best_mae_test = mean_absolute_error(y_test, final_predictions)
                best_corr_test = np.corrcoef(y_test, final_predictions)[0,1]

            results = {'mse_valid':best_score, 'mse_test':best_mse_test, 'mae_test':best_mae_test, 'corr_test':best_corr_test}

            logging.info(f'cur_resutls = {results}')
            
            results_list.append(results)
        

    return results_list