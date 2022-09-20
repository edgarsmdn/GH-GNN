'''
Project: GNN_IAC_T
                    GNNprevious isothermal models
                    
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
from GNNprevious_architecture import GNN
from utilities.mol2graph import get_dataloader_pairs, mol2torchdata
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


def train_GNNprevious_kfold(df, model_name, hyperparameters):
    
    path = os.getcwd()
    path = path + '/../../models/isothermal/' + model_name
    df_split = df.copy()
    
    if not os.path.exists(path):
        os.makedirs(path)

    # Open report file
    report = open('../../reports/Report_training_kfolds_' + model_name + '.txt', 'w')
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
    
    kfolds=5
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
    
    # Construct graphs from molecules
    graph_column_solvent      = 'Graphs_Solvent'
    df[graph_column_solvent]  = mol2torchdata(df, mol_column_solvent, target, y_scaler=None)

    graph_column_solute      = 'Graphs_Solute'
    df[graph_column_solute]  = mol2torchdata(df, mol_column_solute, target, y_scaler=None)
    
    
    # Hyperparameters
    num_layer   = hyperparameters['num_layer']
    drop_ratio  = hyperparameters['drop_ratio']
    conv_dim    = hyperparameters['conv_dim']
    lr          = hyperparameters['lr']
    n_ms        = hyperparameters['n_ms']
    n_epochs    = hyperparameters['n_epochs']
    batch_size  = hyperparameters['batch_size']
    JK          = hyperparameters['JK']
    graph_pooling = hyperparameters['graph_pooling']
    mlp_layers  = hyperparameters['mlp_layers']
    mlp_dims    = hyperparameters['mlp_dims']
    
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
        train_loader = get_dataloader_pairs(df, train_index, target, 
                                            graph_column_solvent, 
                                            graph_column_solute, 
                                            batch_size, 
                                            shuffle=True, 
                                            drop_last=True)
        valid_loader = get_dataloader_pairs(df, valid_index, target, 
                                            graph_column_solvent, 
                                            graph_column_solute, 
                                            batch_size, 
                                            shuffle=False, 
                                            drop_last=True)
        
        
        # Model
        model    = GNN(num_layer=num_layer, drop_ratio=drop_ratio, conv_dim=conv_dim, 
                   gnn_type='NNConv', JK=JK, graph_pooling=graph_pooling,
                   neurons_message=n_ms, mlp_layers=mlp_layers, mlp_dims=mlp_dims)
        device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model    = model.to(device)
        
        # Optimizer                                                           
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)     
        task_type = 'regression'
        scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
        
        # To save trajectory
        mae_train = []; train_loss = []
        mae_valid = []; valid_loss = []
        
        pbar = tqdm(range(n_epochs))
        #pbar = range(n_epochs)
        best_MAE = np.inf
        
        for epoch in pbar:
            stats = OrderedDict()
            # Train
            stats.update(train(model, device, train_loader, optimizer, task_type, stats))
          
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
          
            pbar.set_postfix(stats) # include stats in the progress bar
    
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
        
        
        path_model_info = path + '/Kfold_' + str(e)
        save_train_traj(path_model_info, df_model_training)
        
        # Save best model
        torch.save(best_model, path_model_info + '/Kfold_' + str(e) + '.pth')
        
        end       = time.time()
        print_report('\nTraining time (min): ' + str((end-start)/60))
    
    df_split.to_csv(path+'/Split_'+model_name+'.csv', index=False) # Save splits
    report.close()
    

def train_GNNprevious(df, model_name, hyperparameters):
    
    path = os.getcwd()
    path = path + '/../../models/isothermal/' + model_name
    
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
    
    print_report('Total points   : ' + str(df.shape[0]))
    
    target = 'log-gamma'
    
    # Construct graphs from molecules
    graph_column_solvent      = 'Graphs_Solvent'
    df[graph_column_solvent]  = mol2torchdata(df, mol_column_solvent, target, y_scaler=None)

    graph_column_solute      = 'Graphs_Solute'
    df[graph_column_solute]  = mol2torchdata(df, mol_column_solute, target, y_scaler=None)
    
    
    # Hyperparameters
    num_layer   = hyperparameters['num_layer']
    drop_ratio  = hyperparameters['drop_ratio']
    conv_dim    = hyperparameters['conv_dim']
    lr          = hyperparameters['lr']
    n_ms        = hyperparameters['n_ms']
    n_epochs    = hyperparameters['n_epochs']
    batch_size  = hyperparameters['batch_size']
    JK          = hyperparameters['JK']
    graph_pooling = hyperparameters['graph_pooling']
    mlp_layers  = hyperparameters['mlp_layers']
    mlp_dims    = hyperparameters['mlp_dims']
    

    start       = time.time()
    # Data loaders
    train_loader = get_dataloader_pairs(df, train_index, target, 
                                        graph_column_solvent, 
                                        graph_column_solute, 
                                        batch_size, 
                                        shuffle=True, 
                                        drop_last=True)
    
    
    # Model
    model    = GNN(num_layer=num_layer, drop_ratio=drop_ratio, conv_dim=conv_dim, 
               gnn_type='NNConv', JK=JK, graph_pooling=graph_pooling,
               neurons_message=n_ms, mlp_layers=mlp_layers, mlp_dims=mlp_dims)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    
    # Optimizer                                                           
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)     
    task_type = 'regression'
    scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
    
    # To save trajectory
    mae_train = []; train_loss = []
    
    pbar = tqdm(range(n_epochs))
    #pbar = range(n_epochs)
    
    for epoch in pbar:
        stats = OrderedDict()
        # Train
        stats.update(train(model, device, train_loader, optimizer, task_type, stats))
        # Evaluation
        stats.update(eval(model, device, train_loader, MAE, stats, 'Train', task_type))
        # Scheduler
        scheduler.step(stats['MAE_Train'])
        # Save info
        train_loss.append(stats['Train_loss'])
        mae_train.append(stats['MAE_Train'])
        pbar.set_postfix(stats) # include stats in the progress bar
    
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


Ts = [20,25,30,40,50,60,70,80,100]


hyperparameters_dict ={
    20:{
        'num_layer'   : 4,
        'drop_ratio'  : 0.1301932927161807,
        'conv_dim'    : 234,
        'lr'          : 0.0018940595132654653,
        'n_ms'        : 21,
        'n_epochs'    : 188,
        'batch_size'  : 39,
        'JK'          : 'last',
        'graph_pooling' : 'sum',
        'mlp_layers'  : 3,
        'mlp_dims'    : [105, 74, 1]
        },
    25:{
        'num_layer'   : 3,
        'drop_ratio'  : 0.05138657006194203,
        'conv_dim'    : 41,
        'lr'          : 0.0018322864951079414,
        'n_ms'        : 31,
        'n_epochs'    : 239,
        'batch_size'  : 59,
        'JK'          : 'sum',
        'graph_pooling' : 'sum',
        'mlp_layers'  : 3,
        'mlp_dims'    : [83, 30, 1]
        },
    30:{
        'num_layer'   : 4,
        'drop_ratio'  : 0.12625744000663644,
        'conv_dim'    : 154,
        'lr'          : 0.01120767275290266,
        'n_ms'        : 18,
        'n_epochs'    : 274,
        'batch_size'  : 62,
        'JK'          : 'mean',
        'graph_pooling' : 'set2set',
        'mlp_layers'  : 2,
        'mlp_dims'    : [125, 1]
        },
    40:{
        'num_layer'   : 2,
        'drop_ratio'  : 0.0508238699714836,
        'conv_dim'    : 209,
        'lr'          : 0.0029729610735225397,
        'n_ms'        : 37,
        'n_epochs'    : 238,
        'batch_size'  : 44,
        'JK'          : 'mean',
        'graph_pooling' : 'sum',
        'mlp_layers'  : 3,
        'mlp_dims'    : [110, 112, 1]
        },
    50:{
        'num_layer'   : 3,
        'drop_ratio'  : 0.05105811480090903,
        'conv_dim'    : 72,
        'lr'          : 0.002458567940631729,
        'n_ms'        : 12,
        'n_epochs'    : 168,
        'batch_size'  : 53,
        'JK'          : 'mean',
        'graph_pooling' : 'sum',
        'mlp_layers'  : 3,
        'mlp_dims'    : [104, 126, 1]
        },
    60:{
        'num_layer'   : 3,
        'drop_ratio'  : 0.06296118449816113,
        'conv_dim'    : 88,
        'lr'          : 0.011030631348354433,
        'n_ms'        : 42,
        'n_epochs'    : 289,
        'batch_size'  : 44,
        'JK'          : 'mean',
        'graph_pooling' : 'sum',
        'mlp_layers'  : 3,
        'mlp_dims'    : [113, 122, 1]
        },
    70:{
        'num_layer'   : 4,
        'drop_ratio'  : 0.05137698200239596,
        'conv_dim'    : 250,
        'lr'          : 0.0013392798222914527,
        'n_ms'        : 37,
        'n_epochs'    : 226,
        'batch_size'  : 42,
        'JK'          : 'sum',
        'graph_pooling' : 'sum',
        'mlp_layers'  : 3,
        'mlp_dims'    : [105, 18, 1]
        },
    80:{
        'num_layer'   : 3,
        'drop_ratio'  : 0.06589047058641667,
        'conv_dim'    : 64,
        'lr'          : 0.007198444282887452,
        'n_ms'        : 39,
        'n_epochs'    : 144,
        'batch_size'  : 53,
        'JK'          : 'mean',
        'graph_pooling' : 'sum',
        'mlp_layers'  : 3,
        'mlp_dims'    : [97, 56, 1]
        },
    100:{
        'num_layer'   : 5,
        'drop_ratio'  : 0.06608094458428482,
        'conv_dim'    : 30,
        'lr'          : 0.00934966271701091,
        'n_ms'        : 10,
        'n_epochs'    : 256,
        'batch_size'  : 62,
        'JK'          : 'mean',
        'graph_pooling' : 'mean',
        'mlp_layers'  : 3,
        'mlp_dims'    : [100, 30, 1]
        },
        }



for T in Ts:
    print('-'*50)
    print('Temperature: ', T)
    
    # Given that the data is not open-source, the paths to the data are here 
    # just representative

    df = pd.read_csv('../../data/processed/'+str(T)+'_train.csv')
    train_GNNprevious(df, model_name='GNNprevious_'+str(T), 
                      hyperparameters=hyperparameters_dict[T])
    # train_GNNprevious_kfold(df, model_name='GNNprevious_'+str(T),
    #                         hyperparameters=hyperparameters_dict[T])




