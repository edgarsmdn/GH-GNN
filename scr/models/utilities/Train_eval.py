'''
Project: GNN_IAC

                               Train and evaluation

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import torch
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
from  torch.cuda.amp import autocast

# Cost functions
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def MAPELoss(prediction, real):
    return torch.mean(torch.abs((real - prediction) / real))*100 

def train(model, device, loader, optimizer, task_type, stats, scaler):
  model.train()
  loss_sum = 0
  for batch_solvent, batch_solute, T in loader:
      batch_solvent = batch_solvent.to(device)
      batch_solute  = batch_solute.to(device)
      T = T.to(device)
      if torch.cuda.is_available():
          with autocast(enabled=False):
              pred  = model(batch_solvent.cuda(), batch_solute.cuda(), T.cuda(), scaler=None, ln_gamma=True)
              optimizer.zero_grad()
    
              prediction = pred.to(torch.float32)
              real       = batch_solvent.y.to(torch.float32).reshape(prediction.shape)
              if task_type=="classification": 
                    loss = cls_criterion(prediction, real)
              elif task_type=="regression":
                    loss = reg_criterion(prediction, real)
              else:
                  ValueError(f'Invalid task_type {task_type}')
              scaler.scale(loss).backward()
              loss_sum += loss.item() * batch_solvent.num_graphs   # Loss_i * num_graphs
              scaler.step(optimizer)
              scaler.update()
              
          
          # pred  = model(batch_solvent, batch_solute, T)
          # optimizer.zero_grad()
    
          # prediction = pred.to(torch.float32)
          # real       = batch_solvent.y.to(torch.float32).reshape(prediction.shape)
          # if task_type=="classification": 
          #     loss = cls_criterion(prediction, real)
          # elif task_type=="regression":
          #     loss = reg_criterion(prediction, real)
          # else:
          #   ValueError(f'Invalid task_type {task_type}')
          # loss.backward()
          # loss_sum += loss.item() * batch_solvent.num_graphs   # Loss_i * num_graphs
          # optimizer.step()
      else:
          pred  = model(batch_solvent, batch_solute, T)
          optimizer.zero_grad()
    
          prediction = pred.to(torch.float32)
          real       = batch_solvent.y.to(torch.float32).reshape(prediction.shape)
          if task_type=="classification": 
              loss = cls_criterion(prediction, real)
          elif task_type=="regression":
              loss = reg_criterion(prediction, real)
              
          else:
            ValueError(f'Invalid task_type {task_type}')
          loss.backward()
          loss_sum += loss.item() * batch_solvent.num_graphs   # Loss_i * num_graphs
          optimizer.step()

  n = float(sum([batch.num_graphs for batch, _, _ in loader]))     
  stats.update({'train_loss': loss_sum/n})
  return stats


def eval(model, device, loader, evaluator, stats, split_label, task_type):
    model.eval()
    y_true = []
    y_pred = []

    loss_sum = 0
    for batch_solvent, batch_solute, T in loader:
        batch_solvent = batch_solvent.to(device)
        batch_solute  = batch_solute.to(device)
        T = T.to(device)
        y_true_batch = batch_solvent.y
        num_graphs_batch = batch_solvent.num_graphs
        with torch.no_grad():
            if torch.cuda.is_available():
                pred  = model(batch_solvent.cuda(), batch_solute.cuda(), T.cuda())
            else:
                pred  = model(batch_solvent, batch_solute, T)
        y_true.append(y_true_batch.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

        prediction = pred.to(torch.float32)
        real       = y_true_batch.to(torch.float32).reshape(prediction.shape)
        if task_type=="classification": 
          loss = cls_criterion(prediction, real)
        elif task_type=="regression":
          loss = reg_criterion(prediction, real)
        else:
          ValueError(f'Invalid task_type {task_type}') 
        loss_sum += loss.item() * num_graphs_batch   # Loss_i * num_graphs

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    pred_dict  = {"y_true": y_true, "y_pred": y_pred}
    prediction = evaluator(pred_dict)

    n = float(sum([batch.num_graphs for batch, _, _ in loader]))     
    stats.update({split_label+'_loss': loss_sum/n})
    stats.update({evaluator.__name__ + '_' + split_label: prediction})
    return stats

def ROC_AUC(pred_dict):
    y_true = pred_dict['y_true']; y_pred = pred_dict['y_pred']

    rocauc_list = []
    # Can work for multiclass
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            rocauc_list.append(roc_auc_score(y_true[:,i], y_pred[:,i]))
    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

def MAE(pred_dict):
    y_true = pred_dict['y_true']; y_pred = pred_dict['y_pred']
    # check for NaNs
    if np.any(np.isnan(y_pred)):
        return np.inf
        #raise Exception('Model is predicting NaN')
    else:
        mae = mean_absolute_error(y_true, y_pred)
        return mae

def MAPE(pred_dict):
    y_true = pred_dict['y_true']; y_pred = pred_dict['y_pred']
    mape = mean_absolute_percentage_error(y_true, y_pred)*100
    return mape

def R2(pred_dict):
    y_true = pred_dict['y_true']; y_pred = pred_dict['y_pred']
    r2  = r2_score(y_true, y_pred)
    return r2
