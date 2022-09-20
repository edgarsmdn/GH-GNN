'''
Project: GNN_IAC_T
                    SolvGNNGH train
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

# Scientific computing
import numpy as np
import pandas as pd

# RDKiT
from rdkit import Chem

# Sklearn
from sklearn.model_selection import KFold

# Internal utilities
from SolvGNNGH_architecture import SolvGNNGH, count_parameters
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features
from utilities.Train_eval import train, eval, MAE
from utilities.save_info import save_train_traj

# External utilities
from tqdm import tqdm
#tqdm.pandas()
from collections import OrderedDict
import copy
import time
import os

# Pytorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
from torch.cuda.amp import GradScaler


def train_SolvGNNGH_kfold(df, model_name, hyperparameters):
    
    path = os.getcwd()
    path = path + '/../../models/temperature_dependency/' + model_name
    df_split = df.copy()
    
    if not os.path.exists(path):
        os.makedirs(path)

    # Open report file
    report = open('../../reports/Report_training_kfolds_'+ model_name + '.txt', 'w')
    def print_report(string, file=report):
        print(string)
        file.write('\n' + string)

    print_report(' Report for ' + model_name)
    print_report('-'*50)
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    indices = df.index.tolist()
    
    kfolds=10
    kf = KFold(n_splits=kfolds, random_state=0, shuffle=True)
    cv_train =[]
    cv_valid = []
    for train_kf, valid_kf in kf.split(indices):
        cv_train.append(train_kf)
        cv_valid.append(valid_kf)
        
    print_report('Total points   : ' + str(df.shape[0]))
    print_report('Train points   : ' + str(len(train_kf)))
    print_report('Valid points   : ' + str( len(valid_kf)))
    
    target = 'log-gamma'
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']
    lr          = hyperparameters['lr']
    n_epochs    = hyperparameters['n_epochs']
    batch_size  = hyperparameters['batch_size']
    
    for e in range(1, kfolds+1):
        print('---> kfold: ', e)
        train_index = cv_train[e-1]
        valid_index = cv_valid[e-1]
        
        spliting_values = [0]*df_split.shape[0]
        for k in range(df_split.shape[0]):
            if k in train_index:
                spliting_values[k] = 'Train'
            elif k in valid_index:
                spliting_values[k] = 'Valid'
        df_split['Kfold_'+str(e)] = spliting_values
        
        start       = time.time()
        
        # Data loaders
        train_loader = get_dataloader_pairs_T(df, 
                                              train_index, 
                                              graphs_solv,
                                              graphs_solu,
                                              batch_size, 
                                              shuffle=True, 
                                              drop_last=True)
        valid_loader = get_dataloader_pairs_T(df, 
                                              valid_index, 
                                              graphs_solv,
                                              graphs_solu,
                                              batch_size, 
                                              shuffle=False, 
                                              drop_last=True)
        
        # Model
        model    = SolvGNNGH(n_atom_features(), hidden_dim)
        device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model    = model.to(device)
        
        if e == 1:
            print('    Number of model parameters: ', count_parameters(model))
        
        # Optimizer                                                           
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)     
        task_type = 'regression'
        scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
        
        # Mixed precision training with autocast
        if torch.cuda.is_available():
            pbar = range(n_epochs)
            scaler = GradScaler()
        else:
            pbar = tqdm(range(n_epochs))
            scaler=None
        
        # To save trajectory
        mae_train = []; train_loss = []
        mae_valid = []; valid_loss = []
        
        best_MAE = np.inf
        
        path_model_info = path + '/Kfold_' + str(e)
        
        for epoch in pbar:
            stats = OrderedDict()
            
            # Train
            stats.update(train(model, device, train_loader, optimizer, task_type, stats, scaler))
          
            # Evaluation
            stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
            stats.update(eval(model, device, valid_loader, MAE, stats, 'Valid', task_type))
            
            # Scheduler
            scheduler.step(stats['MAE_Valid'])
          
            # Save info
            train_loss.append(stats['Train_loss'])
            valid_loss.append(stats['Valid_loss'])
            mae_train.append(stats['MAE_Train'])
            mae_valid.append(stats['MAE_Valid'])
            
            # Save best model
            if mae_valid[-1] < best_MAE:
                best_model = copy.deepcopy(model.state_dict())
                best_MAE   = mae_valid[-1]
                
            #pbar.set_postfix(stats) # include stats in the progress bar
    
        best_val_epoch = np.argmin(np.array(mae_valid))
        
        print_report('\n\nKfold ' + str(e))
        print_report('-'*30)
        print_report('Best epoch     : '+ str(best_val_epoch))
        print_report('Training MAE   : '+ str(mae_train[best_val_epoch]))
        print_report('Validation MAE : '+ str(mae_valid[best_val_epoch]))
        
        # Save training trajectory
        df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
        df_model_training['Valid_loss'] = valid_loss
        df_model_training['MAE_Train']  = mae_train
        df_model_training['MAE_Valid']  = mae_valid
        
        save_train_traj(path_model_info, df_model_training)
        
        # Save best model
        torch.save(best_model, path_model_info + '/Kfold_' + str(e) + '.pth')
        
        end       = time.time()
        print_report('\nTraining time (min): ' + str((end-start)/60))
    
    df_split.to_csv(path+'/Split_'+model_name+'.csv', index=False) # Save splits
    report.close()
    
def train_SolvGNNGH(df, model_name, hyperparameters):
    
    path = os.getcwd()
    path = path + '/../../models/temperature_dependency/' + model_name
    
    if not os.path.exists(path):
        os.makedirs(path)

    # Open report file
    report = open('../../reports/Report_training_' + model_name + '.txt', 'w')
    def print_report(string, file=report):
        print(string)
        file.write('\n' + string)

    print_report(' Report for ' + model_name)
    print_report('-'*50)
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    train_index = df.index.tolist()
    
    target = 'log-gamma'
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']
    lr          = hyperparameters['lr']
    n_epochs    = hyperparameters['n_epochs']
    batch_size  = hyperparameters['batch_size']
    
    start       = time.time()
    
    # Data loaders
    train_loader = get_dataloader_pairs_T(df, 
                                          train_index, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size, 
                                          shuffle=True, 
                                          drop_last=True)
    
    # Model
    model    = SolvGNNGH(n_atom_features(), hidden_dim)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    
    print('    Number of model parameters: ', count_parameters(model))
    
    # Optimizer                                                           
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)     
    task_type = 'regression'
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
    
    # Mixed precision training with autocast
    if torch.cuda.is_available():
        pbar = range(n_epochs)
        scaler = GradScaler()
    else:
        pbar = tqdm(range(n_epochs))
        scaler=None
    
    # To save trajectory
    mae_train = []; train_loss = []
    
    for epoch in pbar:
        stats = OrderedDict()
        
        # Train
        stats.update(train(model, device, train_loader, optimizer, task_type, stats, scaler))
        # Evaluation
        stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
        # Scheduler
        scheduler.step(stats['MAE_Train'])
        # Save info
        train_loss.append(stats['Train_loss'])
        mae_train.append(stats['MAE_Train'])
        #pbar.set_postfix(stats) # include stats in the progress bar
    
    print_report('-'*30)
    print_report('Training MAE   : '+ str(mae_train[-1]))
    
    # Save training trajectory
    df_model_training = pd.DataFrame(train_loss, columns=['Train_loss'])
    df_model_training['MAE_Train']  = mae_train
    save_train_traj(path, df_model_training, valid=False)
    
    # Save best model
    final_model = copy.deepcopy(model.state_dict())
    torch.save(final_model, path + '/' + model_name + '.pth')
    
    end       = time.time()
    
    print_report('\nTraining time (min): ' + str((end-start)/60))
    report.close()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("epochs", type=int)
args = parser.parse_args()    


hyperparameters_dict = {'hidden_dim'  : 128,
                        'lr'          : 0.0005474078198480248,
                        'n_epochs'    : args.epochs,
                        'batch_size'  : 23
                        }
    
df = pd.read_csv('../../data/processed/molecular_train.csv')
train_SolvGNNGH(df, 'SolvGNNGH_epochs_'+str(args.epochs), hyperparameters_dict)
# train_SolvGNNGH_kfold(df, 'SolvGNNGH_epochs_'+str(args.epochs), hyperparameters_dict)