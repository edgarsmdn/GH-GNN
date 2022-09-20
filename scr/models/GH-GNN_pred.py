'''
Project: GNN_IAC_T
                    GNN-Gibbs-Helmholtz - T prediction
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from rdkit import Chem
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from GNNGH_T_architecture import GNNGH_T
import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as MAE

def pred_GNNGH_T(df, model_name, hyperparameters):
    path = os.getcwd()
    path = path + '/../../models/temperature_dependency'
    
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
    predict_loader = get_dataloader_pairs_T(df, 
                                          indices, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size=32, 
                                          shuffle=False, 
                                          drop_last=False)
    
    
    ######################
    # --- Prediction --- #
    ######################
    
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3 # ap, bp, topopsa
    model    = GNNGH_T(v_in, e_in, u_in, hidden_dim)
    model.load_state_dict(torch.load(path + '/' + model_name + '.pth', 
                                     map_location=torch.device(available_device)))
    device   = torch.device(available_device)
    model    = model.to(device)
    
    y_pred_final = np.array([])
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute, batch_T in predict_loader:
            batch_solvent = batch_solvent.to(device)
            batch_solute  = batch_solute.to(device)
            batch_T = batch_T.to(device)
            with torch.no_grad():
                if torch.cuda.is_available():
                    y_pred  = model(batch_solvent.cuda(), batch_solute.cuda(), batch_T.cuda()).cpu()
                    y_pred  = y_pred.numpy().reshape(-1,)
                else:
                    y_pred  = model(batch_solvent, batch_solute, batch_T).numpy().reshape(-1,)
                y_pred_final = np.concatenate((y_pred_final, y_pred))
            
    df[model_name] = y_pred_final
    
    return df

def pred_GNNGH_T_kfolds(df, model_name, hyperparameters, k=5):
    path = os.getcwd()
    path = path + '/../../models/temperature_dependency'
    
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
    predict_loader = get_dataloader_pairs_T(df, 
                                          indices, 
                                          graphs_solv,
                                          graphs_solu,
                                          batch_size=32, 
                                          shuffle=False, 
                                          drop_last=False)

    
    ######################
    # --- Prediction --- #
    ######################
    
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    v_in = n_atom_features()
    e_in = n_bond_features()
    u_in = 3 # ap, bp, topopsa
    model    = GNNGH_T(v_in, e_in, u_in, hidden_dim)
    device   = torch.device(available_device)
    model    = model.to(device)
    
    Y_pred = np.zeros((df.shape[0], k))
    
    for i in tqdm(range(k)):
        kfold_name = 'Kfold_' + str(i+1)

        model.load_state_dict(torch.load(path + '/' + kfold_name + '/' + kfold_name + '.pth', 
                                         map_location=torch.device(available_device)))
        
        y_pred_final = np.array([])
        model.eval()
        with torch.no_grad():
            for batch_solvent, batch_solute, batch_T in predict_loader:
                batch_solvent = batch_solvent.to(device)
                batch_solute  = batch_solute.to(device)
                batch_T = batch_T.to(device)
                with torch.no_grad():
                    if torch.cuda.is_available():
                        y_pred  = model(batch_solvent.cuda(), batch_solute.cuda(), batch_T.cuda()).cpu()
                        y_pred  = y_pred.numpy().reshape(-1,)
                    else:
                        y_pred  = model(batch_solvent, batch_solute, batch_T).numpy().reshape(-1,)
                    y_pred_final = np.concatenate((y_pred_final, y_pred))
        Y_pred[:,i] = y_pred_final
        df[model_name+'_kfold_'+str(i+1)] = y_pred_final
    
    y_pred         = np.mean(Y_pred, axis=1)
    y_pred_std     = np.std(Y_pred, axis=1)  
       
    df[model_name] = y_pred
    df[model_name+'_std'] = y_pred_std
    
    return df



hyperparameters_dict = {'hidden_dim'  : 113,
                        'lr'          : 0.0002532501358651798,
                        'batch_size'  : 32
                        }


model_name = 'GH-GNN'

# Models trained on the complete train/validation set
print('Predicting with GNNGH_T')
df = pd.read_csv('../../data/processed/molecular_train.csv')
df_pred = pred_GNNGH_T(df, model_name=model_name, 
                  hyperparameters=hyperparameters_dict)
df_pred.to_csv('../../models/temperature_dependency/predictions/'
               +model_name+'_train_pred.csv')

df = pd.read_csv('../../data/processed/molecular_test.csv')
df_pred = pred_GNNGH_T(df, model_name=model_name, 
                  hyperparameters=hyperparameters_dict)
df_pred.to_csv('../../models/temperature_dependency/predictions/'
               +model_name+'_test_pred.csv')
print('Done!')

# # Models trained using kfold CV
# print('Predicting with GNNGH_T_kfolds')
# df = pd.read_csv('../../data/processed/molecular_train.csv')
# df_pred = pred_GNNGH_T_kfolds(df, model_name=model_name, 
#                   hyperparameters=hyperparameters_dict,
#                   k=10)
# df_pred.to_csv('../../models/temperature_dependency/predictions/'
#                +model_name+'_train_pred_kfolds.csv')


# ### Get performance on validation set
# df_split = pd.read_csv(model_name+'/Split_'+model_name+'.csv')
# maes=[]
# for k in range(1,10+1):
#     #print(' -----> kfold: ', k)
#     val_idx = df_split[df_split['Kfold_'+str(k)] == 'Valid'].index.tolist()
    
#     y_true = df_pred.loc[val_idx, 'log-gamma'].to_numpy()
#     y_pred = df_pred.loc[val_idx, model_name+'_kfold_'+str(k)].to_numpy()
#     mae = MAE(y_true, y_pred)
#     maes.append(mae)
#     #print(' -------- MAE: ', mae)
# print('--> MAE validation: ', np.mean(maes))

# df = pd.read_csv('../../data/processed/molecular_test.csv')
# df_pred = pred_GNNGH_T_kfolds(df, model_name=model_name, 
#                   hyperparameters=hyperparameters_dict,
#                   k=10)
# df_pred.to_csv('../../models/temperature_dependency/predictions/'
#                +model_name+'_test_pred_kfolds.csv')
# print('Done!')

###################################
# --- Predict Brouwer dataset --- #
###################################

# Models trained on the complete train/validation set
print('Predicting with GNNGH_T')
df = pd.read_csv('../../data/processed/brouwer_edge_test.csv')
df_pred = pred_GNNGH_T(df, model_name=model_name, 
                  hyperparameters=hyperparameters_dict)
df_pred.to_csv('../../models/temperature_dependency/predictions/'
               +model_name+'_brouwer_edge_pred.csv')

df = pd.read_csv('../../data/processed/brouwer_extrapolation_test.csv')
df_pred = pred_GNNGH_T(df, model_name=model_name, 
                  hyperparameters=hyperparameters_dict)
df_pred.to_csv('../../models/temperature_dependency/predictions/'
               +model_name+'_brouwer_extrapolation_pred.csv')
print('Done!')

# # Models trained using kfold CV
# print('Predicting with GNNGH_T_kfolds')
# df = pd.read_csv('../../data/processed/brouwer_edge_test.csv')
# df_pred = pred_GNNGH_T_kfolds(df, model_name=model_name, 
#                   hyperparameters=hyperparameters_dict,
#                   k=10)
# df_pred.to_csv('../../models/temperature_dependency/predictions/'
#                +model_name+'_brouwer_edge_pred_kfolds.csv')

# df = pd.read_csv('../../data/processed/brouwer_extrapolation_test.csv')
# df_pred = pred_GNNGH_T_kfolds(df, model_name=model_name, 
#                   hyperparameters=hyperparameters_dict,
#                   k=10)
# df_pred.to_csv('../../models/temperature_dependency/predictions/'
#                +model_name+'_brouwer_extrapolation_pred_kfolds.csv')
# print('Done!')



