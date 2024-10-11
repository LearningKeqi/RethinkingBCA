import numpy as np 
import scipy as sp
import pandas as pd
import glob
from scipy import stats
import random
import glob
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error,mean_absolute_error
from sklearn.metrics import confusion_matrix


def train_cpm(args, X, y, feature_corr, feature_pvalues, p_threshold):
    ipmat = X
    rmat = feature_corr
    pmat = feature_pvalues

    if args.cpm_type == 'positive':
        sel_edges = (rmat > 0) & (pmat < p_threshold)
    elif args.cpm_type == 'negative':
        sel_edges = (rmat < 0) & (pmat < p_threshold)
    elif args.cpm_type == 'combine':
        sel_edges = (rmat != 0) & (pmat < p_threshold)

    sel_edges = sel_edges.astype(int)

    features = ipmat[sel_edges.flatten().astype(bool),:]

    sum_feature = features.sum(axis=0)/2   # univariate

    if args.measure in ['sex', 'Autism','Gender','Parkinson']:
        model = LogisticRegression(random_state=42)
        model.fit(sum_feature.reshape(-1, 1), y)
    else:
        model = np.polyfit(sum_feature,y,1)


    return model, sel_edges



def inter_process(args, testmats, model, sel_edges):
    sum_feature=np.sum(testmats[sel_edges.flatten().astype(bool),:], axis=0)/2

    if args.measure in ['sex', 'Autism','Gender','Parkinson']:
        predictions = model.predict_proba(sum_feature.reshape(-1, 1))[:, 1]
    else:
        predictions = model[0]*sum_feature + model[1]

    return predictions



def cpm_train_and_evaluate(args, X, y, indices, feature_corr, feature_pvalues, cur_repeat):
    
    results_list = []

    X = X.transpose(1, 2, 0)
    numsubs=X.shape[2]
    X=np.reshape(X,[-1,numsubs])
    
    for p_threshold in args.p_threshold_list:
        logging.info(f'Start with p_threshold:{p_threshold}...')

        train_indices = indices[0]
        valid_indices = indices[1]
        test_indices = indices[2]

        trainmats=X[:,train_indices]
        trainpheno=y[train_indices]

        validmats=X[:,valid_indices]
        validpheno=y[valid_indices]

        testmats=X[:,test_indices]
        testpheno=y[test_indices]

        model, sel_edges = train_cpm(args, trainmats, trainpheno, feature_corr, feature_pvalues, p_threshold)

        prediction_val = inter_process(args, validmats, model, sel_edges)
        prediction_test = inter_process(args, testmats, model, sel_edges)

        if args.measure in ['sex', 'Autism','Gender','Parkinson']:
            auc_valid = roc_auc_score(validpheno, prediction_val)
            auc_test = roc_auc_score(testpheno, prediction_test)

            prediction_val[prediction_val > 0.5] = 1
            prediction_val[prediction_val <= 0.5] = 0

            prediction_test[prediction_test > 0.5] = 1
            prediction_test[prediction_test <= 0.5] = 0

            acc_valid = accuracy_score(validpheno, prediction_val)
            acc_test = accuracy_score(testpheno, prediction_test)

            confusion_mat = confusion_matrix(testpheno, prediction_test)
            tn, fp, fn, tp = confusion_mat.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            # save result
            cur_para = {}
            cur_para['p_threshold']=p_threshold

            cur_performance = {}
            cur_performance['valid_auc']=auc_valid
            cur_performance['test_auc']=auc_test
            cur_performance['test_acc']=acc_test
            cur_performance['test_sens']=sensitivity
            cur_performance['test_spec']=specificity

            cur_result = {}
            cur_result['para'] = cur_para
            cur_result['repeat'] = cur_repeat
            cur_result['performance'] = cur_performance        

            results_list.append(cur_result)


        else:
            mse_valid = mean_squared_error(validpheno, prediction_val)
            mse_test = mean_squared_error(testpheno, prediction_test)

            mae_valid = mean_absolute_error(validpheno, prediction_val)
            mae_test = mean_absolute_error(testpheno, prediction_test)

            corr_valid = np.corrcoef(validpheno, prediction_val)[0,1]
            corr_test = np.corrcoef(testpheno, prediction_test)[0,1]

            # save result
            cur_para = {}
            cur_para['p_threshold']=p_threshold

            cur_performance = {}
            cur_performance['valid_mse']=mse_valid
            cur_performance['test_mse']=mse_test
            cur_performance['test_mae']=mae_test
            cur_performance['test_corr']=corr_test
            
            cur_result = {}
            cur_result['para'] = cur_para
            cur_result['repeat'] = cur_repeat
            cur_result['performance'] = cur_performance        

            results_list.append(cur_result)    

    return results_list
    






    






