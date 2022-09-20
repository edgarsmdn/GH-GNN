'''
Project: GNN_IAC_T
                    SolvGNN isothermal models
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from rdkit import Chem
from utilities.mol2graph import get_dataloader_pairs, sys2graph, n_atom_features
from SolvGNN_architecture import SolvGNN
import torch
import os
import numpy as np
from tqdm import tqdm

def pred_SolvGNN(df, model_name, hyperparameters):
    path = os.getcwd()
    path = path + '/../../models/isothermal'
    
    target = 'log-gamma'
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']    
    
    # Dataloader
    indices = df.index.tolist()
    predict_loader = get_dataloader_pairs(df, 
                                          indices, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size=df.shape[0], 
                                          shuffle=False, 
                                          drop_last=False)
    
    
    ######################
    # --- Prediction --- #
    ######################

    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    in_dim   = n_atom_features()
    model    = SolvGNN(in_dim, hidden_dim)
    device   = torch.device(available_device)
    model.load_state_dict(torch.load(path + '/' + model_name + '.pth', 
                                     map_location=torch.device(available_device)))
    model    = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute in predict_loader:
            with torch.no_grad():
                if torch.cuda.is_available():
                    y_pred = model(batch_solvent.cuda(), batch_solute.cuda()).cpu().numpy().reshape(-1,)
                else:
                    y_pred = model(batch_solvent, batch_solute).numpy().reshape(-1,)
            
    df[model_name] = y_pred
    
    return df

def pred_SolvGNN_kfolds(df, model_name, hyperparameters, k=5):
    path = os.getcwd()
    path = path + '/../../models/isothermal'
    
    target = 'log-gamma'
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    graphs_solv, graphs_solu = 'g_solv', 'g_solu'
    df[graphs_solv], df[graphs_solu] = sys2graph(df, mol_column_solvent, mol_column_solute, target)
    
    # Hyperparameters
    hidden_dim  = hyperparameters['hidden_dim']    
    
    # Dataloader
    indices = df.index.tolist()
    predict_loader = get_dataloader_pairs(df, 
                                          indices, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size=df.shape[0], 
                                          shuffle=False, 
                                          drop_last=False)

    
    ######################
    # --- Prediction --- #
    ######################

    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    in_dim   = n_atom_features()
    model    = SolvGNN(in_dim, hidden_dim)
    device   = torch.device(available_device)
    
    
    Y_pred = np.zeros((df.shape[0], k))
    
    for i in tqdm(range(k)):
        kfold_name = 'Kfold_' + str(i+1)

        model.load_state_dict(torch.load(path + '/' + kfold_name + '/' + kfold_name + '.pth', 
                                         map_location=torch.device(available_device)))
        model    = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_solvent, batch_solute in predict_loader:
                if torch.cuda.is_available():
                    y_pred = model(batch_solvent.cuda(), batch_solute.cuda()).cpu().numpy().reshape(-1,)
                else:
                    y_pred = model(batch_solvent, batch_solute).numpy().reshape(-1,) 
        Y_pred[:,i] = y_pred
    
    y_pred         = np.mean(Y_pred, axis=1)
    y_pred_std     = np.std(Y_pred, axis=1)  
       
    df[model_name] = y_pred
    df[model_name+'_std'] = y_pred_std
    
    return df

Ts = [20,25,30,40,50,60,70,80,100]


hyperparameters_dict ={
    20:{
        'hidden_dim'  : 242,
        'lr'          : 0.00036936165073207783,
        'n_epochs'    : 156,
        'batch_size'  : 5
        },
    25:{
        'hidden_dim'  : 226,
        'lr'          : 0.0004452246905217191,
        'n_epochs'    : 178,
        'batch_size'  : 12
        },
    30:{
        'hidden_dim'  : 236,
        'lr'          : 0.0001064539542772283,
        'n_epochs'    : 287,
        'batch_size'  : 4
        },
    40:{
        'hidden_dim'  : 186,
        'lr'          : 0.00024982359554047667,
        'n_epochs'    : 151,
        'batch_size'  : 4
        },
    50:{
        'hidden_dim'  : 197,
        'lr'          : 0.0005870732537897345,
        'n_epochs'    : 212,
        'batch_size'  : 9
        },
    60:{
        'hidden_dim'  : 182,
        'lr'          : 0.00034166273626446927,
        'n_epochs'    : 299,
        'batch_size'  : 7
        },
    70:{
        'hidden_dim'  : 252,
        'lr'          : 0.0005625406199256793,
        'n_epochs'    : 256,
        'batch_size'  : 8
        },
    80:{
        'hidden_dim'  : 162,
        'lr'          : 0.0002417916646587825,
        'n_epochs'    : 254,
        'batch_size'  : 4
        },
    100:{
        'hidden_dim'  : 177,
        'lr'          : 0.000637356514233681,
        'n_epochs'    : 260,
        'batch_size'  : 4
        },
        }


for T in Ts:
    print('-'*50)
    print('Temperature: ', T)
    
    # Models trained on the complete train/validation set
    
    # Given that the data is not open-source, the paths to the data are here 
    # just representative, the same is true for the destination paths of the 
    # predictions
    
    print('Predicting with SolvGNN')
    df = pd.read_csv('../../data/processed/'+str(T)+'_train.csv')
    df_pred = pred_SolvGNN(df, model_name='SolvGNN_'+str(T), 
                      hyperparameters=hyperparameters_dict[T])
    df_pred.to_csv('../../models/isothermal/predictions/SolvGNN/'
                   +str(T)+'_train_pred.csv')
    
    df = pd.read_csv('../../data/processed/'+str(T)+'_test.csv')
    df_pred = pred_SolvGNN(df, model_name='SolvGNN_'+str(T), 
                      hyperparameters=hyperparameters_dict[T])
    df_pred.to_csv('../../models/isothermal/predictions/SolvGNN/'
                   +str(T)+'_test_pred.csv')
    print('Done!')
    
    # # Models trained using kfold CV
    # print('Predicting with SolvGNN_kfolds')
    # df = pd.read_csv('../../data/processed/'+str(T)+'_train.csv')
    # df_pred = pred_SolvGNN_kfolds(df, model_name='SolvGNN_'+str(T), 
    #                   hyperparameters=hyperparameters_dict[T],
    #                   k=5)
    # df_pred.to_csv('../../models/isothermal/predictions/SolvGNN/'
    #                +str(T)+'_train_pred_kfolds.csv')
    
    # df = pd.read_csv('../../data/processed/'+str(T)+'_test.csv')
    # df_pred = pred_SolvGNN_kfolds(df, model_name='SolvGNN_'+str(T), 
    #                   hyperparameters=hyperparameters_dict[T],
    #                   k=5)
    # df_pred.to_csv('../../models/isothermal/predictions/SolvGNN/'
    #                +str(T)+'_test_pred_kfolds.csv')
    # print('Done!')
    



