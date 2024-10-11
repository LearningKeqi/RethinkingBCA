from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import seaborn as sns
from itertools import product


def set_seed(seed=338): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def model_training(cfg: DictConfig):
    print(cfg)

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
    
    dataloaders = dataset_factory(cfg)
    
    logger = logger_factory(cfg)

    model = model_factory(cfg)

    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    
    print(f'optimizers={optimizers[0]}')
    
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)


    if cfg.dataset.task == 'regression':
        test_mse, test_mae, test_corr, test_mse_m, test_mae_m, test_corr_m, attention_weights_m, best_model_dict,\
            valid_mse_m, valid_mae_m, valid_corr_m = training.train(edge_mask=None, mask_pos=None, is_explain=False)
        
        return test_mse, test_mae, test_corr, test_mse_m, test_mae_m, test_corr_m, attention_weights_m, valid_mse_m, valid_mae_m, valid_corr_m

    elif cfg.dataset.task == 'classification':
        test_auc, test_acc, test_sens, test_spec, attention_weights, best_model_dict,\
            valid_auc, valid_acc, valid_sens, valid_spec = training.train(edge_mask=None, mask_pos=None, is_explain=False)
        
        return test_auc, test_acc, test_sens, test_spec, attention_weights, valid_auc, valid_acc, valid_sens, valid_spec            

    
     

def generate_param_combinations(param_dict):
    param_names = list(param_dict.keys())
    
    param_values = [param_dict[param] for param in param_names]
    
    combinations = list(product(*param_values))
    
    param_combinations = [dict(zip(param_names, combination)) for combination in combinations]
    
    return param_combinations



def find_tune_keys(nested_dict, parent_keys=[]):
    result_dict = {}
    
    for key, value in nested_dict.items():
        current_keys = parent_keys + [key.replace('tune_', '')]
        if key.startswith('tune_'):
            result_dict['.'.join(current_keys)] = value
            
        if isinstance(value, dict):
            result_dict.update(find_tune_keys(value, current_keys))
    
    return result_dict



def set_optimizer_param(cfg):
    new_base_lr = cfg.new_learning_rates[0]
    new_target_lr = cfg.new_learning_rates[1]
    combine_base_lr = cfg.combine_learning_rates[0]
    combine_target_lr = cfg.combine_learning_rates[1]

    cfg.optimizer[0].lr_scheduler.base_lr=new_base_lr
    cfg.optimizer[0].lr_scheduler.target_lr=new_target_lr
    cfg.optimizer[0].lr_scheduler.combine_base_lr=combine_base_lr
    cfg.optimizer[0].lr_scheduler.combine_target_lr=combine_target_lr
    
    if cfg.optimizer[0].name == 'Adam':
        if not cfg.model.l1_reg:
            cfg.optimizer[0].weight_decay=cfg.new_weight_decay
        else:
            cfg.optimizer[0].l1_ratio = cfg.new_l1_ratio
            cfg.optimizer[0].alpha = cfg.new_alpha
            cfg.optimizer[0].weight_decay = cfg.optimizer[0].alpha * (1 - cfg.optimizer[0].l1_ratio)
    else:
        cfg.optimizer[0].weight_decay=cfg.new_weight_decay
        cfg.optimizer[0].alpha = cfg.new_alpha
        cfg.optimizer[0].l1_ratio = cfg.new_l1_ratio



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # For parameter tuning
    print(f'cfg={cfg}')
    temp_cfg = OmegaConf.to_container(cfg, resolve=True)
    tune_para = find_tune_keys(temp_cfg)
    print(f'tune_para={tune_para}')
    para_comb = generate_param_combinations(tune_para)
    print(f'para_comb={para_comb}\n')
    print(f'len(para_comb) = {len(para_comb)}')

    tune_results = []

    # grid search for hyperparameters
    for param_index, grid_param in enumerate(para_comb):
        print(f'grid_param={grid_param}')
        for key, value in grid_param.items():
            print(f'key={key}, value={value}')
            OmegaConf.update(cfg, key, value)        

        # set learning rate
        with open_dict(cfg):
            set_optimizer_param(cfg)


        if cfg.model.name == 'mixed_model':
            group_name = f"{cfg.exp_name}_{int(cfg.dataset.sparse_ratio*10000)}_{cfg.model.aggr_module}_{cfg.dataset.node_feature_type}_{cfg.dataset.name}_{cfg.dataset.measure}"
        else: 
            group_name = 'other'

        # regression
        mses_m, maes_m, corrs_m = [], [], []
        mses_valid, maes_valid, corrs_valid = [], [], []

        # classification
        aucs, accs = [], []
        senss, specs = [], []
        aucs_valid, accs_valid, senss_valid, specs_valid = [], [], [], []
        

        with open_dict(cfg):
            cfg.common_save = (f'{cfg.exp_name}_{cfg.dataset.measure}_{cfg.dataset.name}_param{param_index}')

        print(f'Start Exp Name = {cfg.common_save}')

        # Independently run the experiments multiple times
        for i in range(cfg.repeat_time): 
            with open_dict(cfg):     # added for k-fold cross validation
                cfg.dataset.cur_repeat = i
            
            print(f'start repeat{i} for {cfg.common_save}')

            name = f'{group_name}_repreat{cfg.dataset.cur_repeat}'

            run = wandb.init(project=cfg.project, entity=cfg.wandb_entity, reinit=True,
                            group=f"{group_name}", tags=[f"{cfg.dataset.name}"],
                            name = name, mode="offline")
            

            if cfg.dataset.task == 'regression':
                _, _, _, test_mse_m, test_mae_m, test_corr_m, _, valid_mse_m, valid_mae_m, valid_corr_m = model_training(cfg)
            elif cfg.dataset.task == 'classification':
                test_auc, test_acc, test_sens, test_spec, _, valid_auc, valid_acc, valid_sens, valid_spec = model_training(cfg)


            if cfg.dataset.task == 'regression':
                mses_m.append(test_mse_m)
                maes_m.append(test_mae_m)
                corrs_m.append(test_corr_m)

                mses_valid.append(valid_mse_m)
                maes_valid.append(valid_mae_m)
                corrs_valid.append(valid_corr_m)

            elif cfg.dataset.task == 'classification':
                aucs.append(test_auc)
                accs.append(test_acc)
                senss.append(test_sens)
                specs.append(test_spec)

                aucs_valid.append(valid_auc)
                accs_valid.append(valid_acc)
                senss_valid.append(valid_sens)
                specs_valid.append(valid_spec)


            run.finish()
        

        print(f'End Exp Name = {cfg.common_save}')

        if cfg.dataset.task == 'regression':      
            cur_results = {}

            result_str = f'(Average Results Best MSE)| avg_mse={(np.mean(mses_m)):.4f} +- {(np.std(mses_m)): .4f}, avg_mae={(np.mean(maes_m)):.4f} +- {(np.std(maes_m)): .4f}, avg_corr={(np.mean(corrs_m)):.4f} +- {(np.std(corrs_m)): .4f}' 
            print(result_str)   
            cur_results['test_mse'], cur_results['test_mae'], cur_results['test_corr'] = (np.mean(mses_m), np.std(mses_m)), (np.mean(maes_m), np.std(maes_m)), (np.mean(corrs_m), np.std(corrs_m))

            result_str = f'(Average Results Valid Best MSE)| avg_mse={(np.mean(mses_valid)):.4f} +- {(np.std(mses_valid)): .4f}, avg_mae={(np.mean(maes_valid)):.4f} +- {(np.std(maes_valid)): .4f}, avg_corr={(np.mean(corrs_valid)):.4f} +- {(np.std(corrs_valid)): .4f}' 
            print(result_str)   
            cur_results['valid_mse'], cur_results['valid_mae'], cur_results['valid_corr'] = (np.mean(mses_valid), np.std(mses_valid)), (np.mean(maes_valid), np.std(maes_valid)), (np.mean(corrs_valid), np.std(corrs_valid))

            tune_results.append({'param': grid_param, 'results': cur_results})


        elif cfg.dataset.task == 'classification':
            cur_results = {}
            result_str = f'(Average Results Best AUC)| avg_auc={(np.mean(aucs)):.4f} +- {(np.std(aucs)): .4f}, avg_acc={(np.mean(accs)):.4f} +- {(np.std(accs)): .4f}, avg_sens={(np.mean(senss)):.4f} +- {(np.std(senss)): .4f}, , avg_specs={(np.mean(specs)):.4f} +- {(np.std(specs)): .4f}' 
            print(result_str)
            cur_results['test_auc'], cur_results['test_acc'], cur_results['test_sens'], cur_results['test_spec'] = (np.mean(aucs), np.std(aucs)), (np.mean(accs), np.std(accs)), (np.mean(senss), np.std(senss)), (np.mean(specs), np.std(specs))

            result_str = f'(Average Results Valid Best AUC)| avg_auc={(np.mean(aucs_valid)):.4f} +- {(np.std(aucs_valid)): .4f}, avg_acc={(np.mean(accs_valid)):.4f} +- {(np.std(accs_valid)): .4f}, avg_sens={(np.mean(senss_valid)):.4f} +- {(np.std(senss_valid)): .4f}, , avg_specs={(np.mean(specs_valid)):.4f} +- {(np.std(specs_valid)): .4f}\n' 
            print(result_str)
            cur_results['valid_auc'], cur_results['valid_acc'], cur_results['valid_sens'], cur_results['valid_spec'] = (np.mean(aucs_valid), np.std(aucs_valid)), (np.mean(accs_valid), np.std(accs_valid)), (np.mean(senss_valid), np.std(senss_valid)), (np.mean(specs_valid), np.std(specs_valid))

            tune_results.append({'param': grid_param, 'results': cur_results})



    print('=========================================')
    if cfg.dataset.task == 'classification':
        valid_aucs = np.array([this_result['results']['valid_auc'][0] for this_result in tune_results])
        max_auc_index = np.argmax(valid_aucs)
        print(f'Tune finished! best valid set auc setting: {tune_results[max_auc_index]["param"]}')
        
        cur_performance = tune_results[max_auc_index]["results"]
        test_result_str = f'avg_auc={cur_performance["test_auc"][0]:.4f} +- {cur_performance["test_auc"][1]: .4f}, avg_acc={cur_performance["test_acc"][0]:.4f} +- {cur_performance["test_acc"][1]: .4f}, avg_sens={cur_performance["test_sens"][0]:.4f} +- {cur_performance["test_sens"][1]: .4f}, avg_specs={cur_performance["test_spec"][0]:.4f} +- {cur_performance["test_spec"][1]: .4f}' 
        print(f"(Tune best test): {test_result_str}")

        valid_result_str = f'avg_auc={cur_performance["valid_auc"][0]:.4f} +- {cur_performance["valid_auc"][1]: .4f}, avg_acc={cur_performance["valid_acc"][0]:.4f} +- {cur_performance["valid_acc"][1]: .4f}, avg_sens={cur_performance["valid_sens"][0]:.4f} +- {cur_performance["valid_sens"][1]: .4f}, avg_specs={cur_performance["valid_spec"][0]:.4f} +- {cur_performance["valid_spec"][1]: .4f}\n' 
        print(f"(Tune best valid): {valid_result_str}")
    elif cfg.dataset.task == 'regression':
        valid_mses = np.array([this_result['results']['valid_mse'][0] for this_result in tune_results])
        min_mse_index = np.argmin(valid_mses)
        print(f'Tune finished! best valid set mse setting: {tune_results[min_mse_index]["param"]}')

        cur_performance = tune_results[min_mse_index]['results']
        test_result_str = f'avg_mse={cur_performance["test_mse"][0]:.4f} +- {cur_performance["test_mse"][1]: .4f}, avg_mae={cur_performance["test_mae"][0]:.4f} +- {cur_performance["test_mae"][1]: .4f}, avg_corr={cur_performance["test_corr"][0]:.4f} +- {cur_performance["test_corr"][1]: .4f}' 
        print(f"(Tune best test): {test_result_str}")

        valid_result_str = f'avg_mse={cur_performance["valid_mse"][0]:.4f} +- {cur_performance["valid_mse"][1]: .4f}, avg_mae={cur_performance["valid_mae"][0]:.4f} +- {cur_performance["valid_mae"][1]: .4f}, avg_corr={cur_performance["valid_corr"][0]:.4f} +- {cur_performance["valid_corr"][1]: .4f}' 
        print(f"(Tune best valid): {valid_result_str}")


    


if __name__ == '__main__':
    print('start!!')
    set_seed()
    main()

