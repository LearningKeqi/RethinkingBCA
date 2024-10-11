from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error,mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
import copy
import time
import matplotlib.pyplot as plt
from ..utils import draw_single_attn, draw_single_connectivity, draw_single_x, draw_multiple_connectivity
import math


class MLP_Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders

        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers

        if self.config.dataset.task == 'classification':
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        elif self.config.dataset.task == 'regression':
            self.loss_fn = torch.nn.MSELoss()

        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph

        self.init_meters()


    def init_meters(self):        
        self.train_loss, self.valid_loss, self.test_loss,\
            self.train_mse, self.valid_mse, self.test_mse,\
            self.train_mae, self.train_corr = [
                TotalMeter() for _ in range(8)]

    def reset_meters(self):
        for meter in [self.train_mse, self.valid_mse, self.test_mse, self.train_loss, self.valid_loss, self.test_loss, self.train_mae, self.train_corr]:
            meter.reset()


    def train_per_epoch(self, optimizer, lr_scheduler, edge_mask, mask_pos, is_explain):
        self.model.train()

        for time_series, node_feature, label, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids in self.train_dataloader:
        
            self.current_step += 1

            label = label.float()


            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series, node_feature, label, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids = time_series.cuda(), node_feature.cuda(), label.cuda(), orig_connection.cuda(), saved_eigenvectors.cuda(), sparse_connection.cuda(), used_subjectids.cuda()


            if self.config.preprocess.continus:
                time_series, node_feature, orig_connection, label = continus_mixup_data(
                    time_series, node_feature, orig_connection, y=label)

            predict = self.model(time_series, node_feature)

            if self.config.dataset.task == 'regression':
                loss = self.loss_fn(predict.squeeze(), label)
                    
                if self.config.model.l1_reg:
                    loss += self.l1_regularization(self.config.new_l1_norm_weight)

            else:
                loss = self.loss_fn(predict, label)
                
                if self.config.model.l1_reg:
                    loss += self.l1_regularization(self.config.new_l1_norm_weight)



            self.train_loss.update_with_weight(loss.item(), label.shape[0])

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    

    def test_per_epoch_cog(self, dataloader, loss_meter, mse_meter, edge_mask, mask_pos, is_explain):
        labels = []
        result = []
        
        num_layer = len(self.config.model.sizes)

        cnt_subjects = 0

        attention_weights_list = [[] for _ in range(num_layer)]
        orig_attention_weights_list = [[] for _ in range(num_layer)]

        orig_x_list = []
        delta_x_list = [[] for _ in range(num_layer)]
        modified_x_list = [[] for _ in range(num_layer)]

        self.model.eval()


        if self.config.dataset.task == 'regression':
            for time_series, node_feature, label, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids in dataloader:
                time_series, node_feature, label, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids = time_series.cuda(), node_feature.cuda(), label.cuda(), orig_connection.cuda(), saved_eigenvectors.cuda(), sparse_connection.cuda(), used_subjectids.cuda()
                
                output = self.model(time_series, node_feature).squeeze()

                cnt_subjects += node_feature.shape[0]


                label = label.float()


                if output.dim() == 0:
                    output = output.unsqueeze(0)

                loss = self.loss_fn(output, label)
                loss_meter.update_with_weight(
                    loss.item(), len(label))

                mse = mean_squared_error(output.detach().cpu(), label.detach().cpu())
                mse_meter.update_with_weight(mse, len(label))

                result += output.detach().cpu().tolist()
                labels += label.detach().cpu().tolist()


            mae = mean_absolute_error(labels, result)

            corr = np.corrcoef(labels, result)[0,1]

            return [mae, corr], attention_weights_list, orig_attention_weights_list, orig_x_list, delta_x_list, modified_x_list
        
        elif self.config.dataset.task == 'classification':
            num_samples = 0

            for time_series, node_feature, label, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids in dataloader:
                time_series, node_feature, label, orig_connection, saved_eigenvectors, sparse_connection, used_subjectids = time_series.cuda(), node_feature.cuda(), label.cuda(), orig_connection.cuda(), saved_eigenvectors.cuda(), sparse_connection.cuda(), used_subjectids.cuda()
                
                output = self.model(time_series, node_feature)

                num_samples += node_feature.shape[0]

                label = label.float()

                loss = self.loss_fn(output, label)
                loss_meter.update_with_weight(
                    loss.item(), len(label))
                
                
                result += F.softmax(output, dim=1)[:, 1].tolist()
                labels += label[:, 1].tolist()


            auc = roc_auc_score(labels, result)
            result, labels = np.array(result), np.array(labels)
            result[result > 0.5] = 1
            result[result <= 0.5] = 0
            
            acc = accuracy_score(labels, result)
            confusion_mat = confusion_matrix(labels, result)
            tn, fp, fn, tp = confusion_mat.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)


            return auc, acc, sensitivity, specificity, attention_weights_list, orig_attention_weights_list, orig_x_list, delta_x_list, modified_x_list
    

    def generate_save_learnable_matrix(self):

        # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(x_labels, y_labels, matrix_values, show_text=False)})
        learable_matrixs = []

        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            _, learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")


    def save_trained_model(self, model_dict, is_explain):
        save_model_file = f'./exp_results/trained_models/model_{self.config.common_save}_isexplain{is_explain}_repeat{self.config.dataset.cur_repeat}.pth'
        torch.save(model_dict, save_model_file)
    



    def show_initial_performance(self, edge_mask, mask_pos, is_explain):

        if self.config.dataset.task == 'regression':
            self.reset_meters()
            
            train_result, _, _, _, _, _ = self.test_per_epoch_cog(self.train_dataloader,
                                                self.train_loss, self.train_mse, edge_mask, mask_pos, is_explain)
            
            valid_result, _, _, _, _, _ = self.test_per_epoch_cog(self.val_dataloader,
                                                self.valid_loss, self.valid_mse, edge_mask, mask_pos, is_explain)
            
            test_result, _, _, _, _, _ = self.test_per_epoch_cog(self.test_dataloader,
                                                self.test_loss, self.test_mse, edge_mask, mask_pos, is_explain)



            self.logger.info(" | ".join([
                f'Initial Performance',
                f'Train Loss:{self.train_loss.avg: .4f}',
                f'Train MSE:{self.train_mse.avg: .4f}',
                f'Train MAE:{train_result[0]:.4f}',
                f'Train Corr:{train_result[-1]:.4f}',
                f'Valid Loss:{self.valid_loss.avg: .4f}',
                f'Valid MSE:{self.valid_mse.avg: .4f}',
                f'Valid MAE:{valid_result[0]:.4f}',
                f'Valid Corr:{valid_result[-1]:.4f}',
                f'Test Loss:{self.test_loss.avg: .4f}',
                f'Test MSE:{self.test_mse.avg: .4f}',
                f'Test MAE:{test_result[0]:.4f}',
                f'Test Corr:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train MSE": self.train_mse.avg,
                "Train MAE": train_result[0],
                "Train Corr": train_result[-1],
                "Valid Loss": self.valid_loss.avg,
                "Valid MSE": self.valid_mse.avg,
                "Valid MAE": valid_result[0],
                'Valid Corr': valid_result[-1],
                "Test Loss": self.test_loss.avg,
                "Test MSE": self.test_mse.avg,
                "Test MAE": test_result[0],
                'Test Corr': test_result[-1]
            })
        
        elif self.config.dataset.task == 'classification':
            self.reset_meters()
            
            train_auc, train_acc, train_sensitivity, train_specificity, _, _, _, _, _ = self.test_per_epoch_cog(self.train_dataloader,
                                                self.train_loss, None, edge_mask, mask_pos, is_explain)
            
            valid_auc, valid_acc, valid_sensitivity, valid_specificity, _, _, _, _, _ = self.test_per_epoch_cog(self.val_dataloader,
                                                self.valid_loss, None, edge_mask, mask_pos, is_explain)
            
            test_auc, test_acc, test_sensitivity, test_specificity, _, _, _, _, _ = self.test_per_epoch_cog(self.test_dataloader,
                                                self.test_loss, None, edge_mask, mask_pos, is_explain)
            

            self.logger.info(" | ".join([
                f'Initial Performance',
                f'Train Loss:{self.train_loss.avg: .4f}',
                f'Train AUC:{train_auc: .4f}',
                f'Train ACC:{train_acc:.4f}',
                f'Train SENS:{train_sensitivity: .4f}',
                f'Train SPEC:{train_specificity:.4f}',
                f'Valid Loss:{self.valid_loss.avg: .4f}',
                f'Valid AUC:{valid_auc: .4f}',
                f'Valid ACC:{valid_acc:.4f}',
                f'Valid SENS:{valid_sensitivity: .4f}',
                f'Valid SPEC:{valid_specificity:.4f}',
                f'Test Loss:{self.test_loss.avg: .4f}',
                f'Test AUC:{test_auc: .4f}',
                f'Test ACC:{test_acc:.4f}',
                f'Test SENS:{test_sensitivity: .4f}',
                f'Test SPEC:{test_specificity:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train AUC": train_auc,
                "Train ACC": train_acc,
                "Train SENS": train_sensitivity,
                "Train SPEC": train_specificity,
                "Valid Loss": self.valid_loss.avg,
                "Valid AUC": valid_auc,
                "Valid ACC": valid_acc,
                "Valid SENS": valid_sensitivity,
                "Valid SPEC": valid_specificity,
                "Test Loss": self.test_loss.avg,
                "Test AUC": test_auc,
                "Test ACC": test_acc,
                "Test SENS": test_sensitivity,
                "Test SPEC": test_specificity
            })
        



    def train(self, edge_mask, mask_pos, is_explain):
        training_process = []
        self.current_step = 0

        self.show_initial_performance(edge_mask, mask_pos, is_explain)

        if self.config.dataset.task == 'regression':
            mses_valid, maes_valid, corrs_valid= [], [], []
            mses_test, maes_test, corrs_test= [], [], []
        elif self.config.dataset.task == 'classification':
            aucs_valid, accs_valid, senss_valid, specs_valid = [], [], [], []
            aucs_test, accs_test, senss_test, specs_test  = [], [], [], []
            


        act_epoch = self.epochs if self.config.training.less_epoch == 'none' else self.config.training.less_epoch

        if self.config.dataset.task == 'regression':

            best_mse = float('inf')
            best_model_dict = copy.deepcopy(self.model.state_dict())
            best_index = -1

            for epoch in range(act_epoch):

                self.reset_meters()
                
                self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0], edge_mask, mask_pos, is_explain)
                self.reset_meters()


                train_result, _, _, \
                    _, _, _ = self.test_per_epoch_cog(self.train_dataloader,
                                                self.train_loss, self.train_mse, edge_mask, mask_pos, is_explain)
                
                valid_result, _, _, \
                    _, _, _ = self.test_per_epoch_cog(self.val_dataloader,
                                                self.valid_loss, self.valid_mse, edge_mask, mask_pos, is_explain)
                

                test_result, _, _, \
                    _, _, _ = self.test_per_epoch_cog(self.test_dataloader,
                                                self.test_loss, self.test_mse, edge_mask, mask_pos, is_explain)
                


                mses_valid.append(self.valid_mse.avg)
                maes_valid.append(valid_result[0])
                corrs_valid.append(valid_result[-1])

                mses_test.append(self.test_mse.avg)
                maes_test.append(test_result[0])
                corrs_test.append(test_result[-1])

                if best_mse > self.valid_mse.avg:
                    best_mse = self.valid_mse.avg
                    best_model_dict = copy.deepcopy(self.model.state_dict())
                    best_index = epoch


                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .4f}',
                    f'Train MSE:{self.train_mse.avg: .4f}',
                    f'Train MAE:{train_result[0]:.4f}',
                    f'Train Corr:{train_result[-1]:.4f}',
                    f'Valid Loss:{self.valid_loss.avg: .4f}',
                    f'Valid MSE:{self.valid_mse.avg: .4f}',
                    f'Valid MAE:{valid_result[0]:.4f}',
                    f'Valid Corr:{valid_result[-1]:.4f}',
                    f'Test Loss:{self.test_loss.avg: .4f}',
                    f'Test MSE:{self.test_mse.avg: .4f}',
                    f'Test MAE:{test_result[0]:.4f}',
                    f'Test Corr:{test_result[-1]:.4f}',
                    f'LR:{self.lr_schedulers[0].lr:.4f}'
                ]))

                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Train MSE": self.train_mse.avg,
                    "Train MAE": train_result[0],
                    "Train Corr": train_result[-1],
                    "Valid Loss": self.valid_loss.avg,
                    "Valid MSE": self.valid_mse.avg,
                    "Valid MAE": valid_result[0],
                    'Valid Corr': valid_result[-1],
                    "Test Loss": self.test_loss.avg,
                    "Test MSE": self.test_mse.avg,
                    "Test MAE": test_result[0],
                    'Test Corr': test_result[-1]
                })

                training_process.append({
                    "Epoch": epoch,
                    "Train Loss": self.train_loss.avg,
                    "Train MSE": self.train_mse.avg,
                    "Train MAE": train_result[0],
                    "Train Corr": train_result[-1],
                    "Valid Loss": self.valid_loss.avg,
                    "Valid MSE": self.valid_mse.avg,
                    "Valid MAE": valid_result[0],
                    'Valid Corr': valid_result[-1],
                    "Test Loss": self.test_loss.avg,
                    "Test MSE": self.test_mse.avg,
                    "Test MAE": test_result[0],
                    'Test Corr': test_result[-1]
                })

                torch.cuda.empty_cache()


            if self.save_learnable_graph:
                self.generate_save_learnable_matrix()
            self.save_result(training_process)

            mses_valid=np.array(mses_valid)
            maes_valid=np.array(maes_valid)
            corrs_valid=np.array(corrs_valid)

            mses_test=np.array(mses_test)
            maes_test=np.array(maes_test)
            corrs_test=np.array(corrs_test)

            max_index = np.argmax(corrs_valid)
            self.logger.info(f"best corr index: {max_index}")
            self.logger.info(f"best valid performance: best_val_mse={mses_valid[max_index]}, best_val_mae={maes_valid[max_index]}, best_val_corr={corrs_valid[max_index]}")
            self.logger.info(f"best test performance: best_test_mse={mses_test[max_index]}, best_test_mae={maes_test[max_index]}, best_test_corr={corrs_test[max_index]}")

            min_index = np.argmin(mses_valid)
            self.logger.info(f"best mses index: {min_index}")
            self.logger.info(f"best valid performance: best_val_mse={mses_valid[min_index]}, best_val_mae={maes_valid[min_index]}, best_val_corr={corrs_valid[min_index]}")
            self.logger.info(f"best test performance: best_test_mse={mses_test[min_index]}, best_test_mae={maes_test[min_index]}, best_test_corr={corrs_test[min_index]}")
            
            assert min_index == best_index, "min_index should be equal to best_index"
            

            # save trained model
            self.save_trained_model(best_model_dict, is_explain)
                        

            return mses_test[max_index], maes_test[max_index], corrs_test[max_index], mses_test[min_index], maes_test[min_index], corrs_test[min_index], None, best_model_dict,\
                  mses_valid[min_index], maes_valid[min_index], corrs_valid[min_index]   
        
            
        elif self.config.dataset.task == 'classification':
            best_auc = -float('inf')
            best_model_dict = copy.deepcopy(self.model.state_dict())
            best_index = -1

            for epoch in range(act_epoch):                

                self.reset_meters()
                
                self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0], edge_mask, mask_pos, is_explain)
                self.reset_meters()

                train_auc, train_acc, train_sensitivity, train_specificity, _, _, \
                    _, _, _ = self.test_per_epoch_cog(self.train_dataloader,
                                                self.train_loss, None, edge_mask, mask_pos, is_explain)


                valid_auc, valid_acc, valid_sensitivity, valid_specificity, _, _, \
                    _, _, _ = self.test_per_epoch_cog(self.val_dataloader,
                                                self.valid_loss, None, edge_mask, mask_pos, is_explain)
                

                test_auc, test_acc, test_sensitivity, test_specificity, _, _, \
                    _, _, _ = self.test_per_epoch_cog(self.test_dataloader,
                                                self.test_loss, None, edge_mask, mask_pos, is_explain)
                

                aucs_valid.append(valid_auc)
                accs_valid.append(valid_acc)
                senss_valid.append(valid_sensitivity)
                specs_valid.append(valid_specificity)

                aucs_test.append(test_auc)
                accs_test.append(test_acc)
                senss_test.append(test_sensitivity)
                specs_test.append(test_specificity)

  
                if best_auc < valid_auc:
                    best_auc = valid_auc
                    best_model_dict = copy.deepcopy(self.model.state_dict())
                    best_index = epoch

                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .4f}',
                    f'Train AUC:{train_auc: .4f}',
                    f'Train ACC:{train_acc: .4f}',
                    f'Train SENS:{train_sensitivity:.4f}',
                    f'Train SPEC:{train_specificity:.4f}',
                    f'Valid Loss:{self.valid_loss.avg: .4f}',
                    f'Valid AUC:{valid_auc: .4f}',
                    f'Valid ACC:{valid_acc:.4f}',
                    f'Valid SENS:{valid_sensitivity:.4f}',
                    f'Valid SPEC:{valid_specificity:.4f}',
                    f'Test Loss:{self.test_loss.avg: .4f}',
                    f'Test AUC:{test_auc: .4f}',
                    f'Test ACC:{test_acc:.4f}',
                    f'Test SENS:{test_sensitivity:.4f}',
                    f'Test SPEC:{test_specificity:.4f}',
                    f'LR:{self.lr_schedulers[0].lr:.4f}'
                ]))

                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Train AUC": train_auc,
                    "Train ACC": train_acc,
                    "Train SENS": train_sensitivity,
                    "Train SPEC": train_specificity,
                    "Valid Loss": self.valid_loss.avg,
                    "Valid AUC": valid_auc,
                    "Valid ACC": valid_acc,
                    "Valid SENS": valid_sensitivity,
                    "Valid SPEC": valid_specificity,
                    "Test Loss": self.test_loss.avg,
                    "Test AUC": test_auc,
                    "Test ACC": test_acc,
                    "Test SENS": test_sensitivity,
                    "Test SPEC": test_specificity,
                })

                training_process.append({
                    "Epoch": epoch,
                    "Train Loss": self.train_loss.avg,
                    "Train AUC": train_auc,
                    "Train ACC": train_acc,
                    "Train SENS": train_sensitivity,
                    "Train SPEC": train_specificity,
                    "Valid Loss": self.valid_loss.avg,
                    "Valid AUC": valid_auc,
                    "Valid ACC": valid_acc,
                    "Valid SENS": valid_sensitivity,
                    "Valid SPEC": valid_specificity,
                    "Test Loss": self.test_loss.avg,
                    "Test AUC": test_auc,
                    "Test ACC": test_acc,
                    "Test SENS": test_sensitivity,
                    "Test SPEC": test_specificity
                })


                torch.cuda.empty_cache()

            if self.save_learnable_graph:
                self.generate_save_learnable_matrix()
            self.save_result(training_process)

            aucs_valid=np.array(aucs_valid)
            accs_valid=np.array(accs_valid)
            senss_valid = np.array(senss_valid)
            specs_valid = np.array(specs_valid)

            aucs_test=np.array(aucs_test)
            accs_test=np.array(accs_test)
            senss_test = np.array(senss_test)
            specs_test = np.array(specs_test)


            max_index = np.argmax(aucs_valid)
            self.logger.info(f"best auc index: {max_index}")
            self.logger.info(f"best valid performance: best_val_auc={aucs_valid[max_index]}, best_val_acc={accs_valid[max_index]}, best_val_sens={senss_valid[max_index]}, best_val_spec={specs_valid[max_index]}")
            self.logger.info(f"best test performance: best_test_auc={aucs_test[max_index]}, best_test_acc={accs_test[max_index]}, best_test_sens={senss_test[max_index]}, best_test_spec={specs_test[max_index]}")
            

            # save trained model
            self.save_trained_model(best_model_dict, is_explain)

            assert max_index == best_index, "min_index does not equal to best_index"

            return aucs_test[max_index], accs_test[max_index], senss_test[max_index], specs_test[max_index], None, best_model_dict,\
                    aucs_valid[max_index], accs_valid[max_index], senss_valid[max_index], specs_valid[max_index]    # modified for best metric score


    def l1_regularization(self, lambda_l1=1e-4):
        l1_loss = 0
        for param in self.model.parameters():
            l1_loss += torch.sum(torch.abs(param))
            
        # return lambda_l1 * l1_loss
        return self.config.new_l1_ratio * self.config.new_alpha * l1_loss