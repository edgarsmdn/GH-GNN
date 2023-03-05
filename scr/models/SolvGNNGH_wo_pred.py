'''
Project: GNN_IAC_T
                    SolvGNNGH wo MTL
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from rdkit import Chem
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features
from SolvGNNGH_wo_architecture import SolvGNNGH_wo
import torch
import os
import numpy as np
from sklearn.metrics import mean_absolute_error as MAE

def pred_SolvGNNGH(df, model_name, hyperparameters):
    path = os.getcwd()
    path = path + '/' + model_name
    
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
                                          batch_size=23, 
                                          shuffle=False, 
                                          drop_last=False)
    
    
    ######################
    # --- Prediction --- #
    ######################
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    model    = SolvGNNGH_wo(n_atom_features(), hidden_dim)
    device   = torch.device(available_device)
    model    = model.to(device)
    model.load_state_dict(torch.load(path + '/' + model_name + '.pth', 
                                     map_location=torch.device(available_device)))
    
    
    y_pred_final = np.array([])
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute, batch_T in predict_loader:
            with torch.no_grad():
                if torch.cuda.is_available():
                    y_pred  = model(batch_solvent.cuda(), batch_solute.cuda(), batch_T.cuda()).cpu()
                    y_pred  = y_pred.numpy().reshape(-1,)
                else:
                    y_pred  = model(batch_solvent, batch_solute, batch_T).numpy().reshape(-1,)
                y_pred_final = np.concatenate((y_pred_final, y_pred))
            
    df[model_name] = y_pred_final
    
    return df


epochs = [250]


hyperparameters_dict = {'hidden_dim'  : 193,
                        'lr'          : 0.00011559310094158379,
                        'batch_size'  : 16
                        }


for e in epochs:
    print('-'*50)
    print('Epochs: ', e)
    
    model_name = 'SolvGNNGH_epochs_'+str(e)
    
    # Models trained on the complete train/validation set
    print('Predicting with SolvGNNGH')
    df = pd.read_csv('../../data/processed/molecular_train.csv')
    df_pred = pred_SolvGNNGH(df, model_name=model_name, 
                      hyperparameters=hyperparameters_dict)
    df_pred.to_csv(model_name+'/train_pred.csv')
    
    df = pd.read_csv('../../data/processed/molecular_test.csv')
    df_pred = pred_SolvGNNGH(df, model_name=model_name, 
                      hyperparameters=hyperparameters_dict)
    df_pred.to_csv(model_name+'/test_pred.csv')
    print('Done!')
    
    ###################################
    # --- Predict Brouwer dataset --- #
    ###################################
    
    model_name = 'SolvGNNGH_epochs_'+str(e)
    
    # Models trained on the complete train/validation set
    print('Predicting with SolvGNNGH')
    df = pd.read_csv('../../data/processed/brouwer_edge_test.csv')
    df_pred = pred_SolvGNNGH(df, model_name=model_name, 
                      hyperparameters=hyperparameters_dict)
    df_pred.to_csv(model_name+'/brouwer_edge_pred.csv')
    
    df = pd.read_csv('../../data/processed/brouwer_extrapolation_test.csv')
    df_pred = pred_SolvGNNGH(df, model_name=model_name, 
                      hyperparameters=hyperparameters_dict)
    df_pred.to_csv(model_name+'/brouwer_extrapolation_pred.csv')
    print('Done!')
    


