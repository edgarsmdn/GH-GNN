'''
Project: GNN_IAC_T
                    GNNprevious isothermal models
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from rdkit import Chem
from utilities.mol2graph import mol2torchdata, get_dataloader_pairs
from GNNprevious_architecture import GNN
import torch
import os
import numpy as np
from tqdm import tqdm

def pred_GNNprevious(df, model_name, hyperparameters):
    path = os.getcwd()
    path = path + '/../../models/isothermal'
    
    target = 'log-gamma'
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    # Construct graphs from molecules
    graph_column_solvent      = 'Graphs_Solvent'
    df[graph_column_solvent]  = mol2torchdata(df, mol_column_solvent, target, y_scaler=None)

    graph_column_solute      = 'Graphs_Solute'
    df[graph_column_solute]  = mol2torchdata(df, mol_column_solute, target, y_scaler=None)
    
    # Dataloader
    indices = df.index.tolist()
    predict_loader = get_dataloader_pairs(df, indices, target, 
                                          graph_column_solvent, 
                                          graph_column_solute, 
                                          batch_size=df.shape[0], 
                                          shuffle=False, drop_last=False)

    # Hyperparameters
    num_layer   = hyperparameters['num_layer']
    drop_ratio  = hyperparameters['drop_ratio']
    conv_dim    = hyperparameters['conv_dim']
    n_ms        = hyperparameters['n_ms']
    JK          = hyperparameters['JK']
    graph_pooling = hyperparameters['graph_pooling']
    mlp_layers  = hyperparameters['mlp_layers']
    mlp_dims    = hyperparameters['mlp_dims']
    
    ######################
    # --- Prediction --- #
    ######################
    
    model = GNN(num_layer=num_layer, drop_ratio=drop_ratio, conv_dim=conv_dim, 
                   gnn_type='NNConv', JK=JK, graph_pooling=graph_pooling,
                   neurons_message=n_ms, mlp_layers=mlp_layers, mlp_dims=mlp_dims)

    model.load_state_dict(torch.load(path + '/' + model_name + '.pth', 
                                     map_location=torch.device('cpu')))
    
    model.eval()
    with torch.no_grad():
        for batch_solvent, batch_solute in predict_loader:
            with torch.no_grad():
                y_pred = model(batch_solvent, batch_solute).numpy().reshape(-1,)
            
    df[model_name] = y_pred
    
    return df

def pred_GNNprevious_kfolds(df, model_name, hyperparameters, k=5):
    path = os.getcwd()
    path = path + '/../../models/isothermal'
    
    target = 'log-gamma'
    
    # Build molecule from SMILES
    mol_column_solvent     = 'Molecule_Solvent'
    df[mol_column_solvent] = df['Solvent_SMILES'].apply(Chem.MolFromSmiles)

    mol_column_solute      = 'Molecule_Solute'
    df[mol_column_solute]  = df['Solute_SMILES'].apply(Chem.MolFromSmiles)
    
    # Construct graphs from molecules
    graph_column_solvent      = 'Graphs_Solvent'
    df[graph_column_solvent]  = mol2torchdata(df, mol_column_solvent, target, y_scaler=None)

    graph_column_solute      = 'Graphs_Solute'
    df[graph_column_solute]  = mol2torchdata(df, mol_column_solute, target, y_scaler=None)
     
    # Dataloader
    indices = df.index.tolist()
    predict_loader = get_dataloader_pairs(df, indices, target, 
                                          graph_column_solvent, 
                                          graph_column_solute, 
                                          batch_size=df.shape[0], 
                                          shuffle=False, drop_last=False)

    # Hyperparameters
    num_layer   = hyperparameters['num_layer']
    drop_ratio  = hyperparameters['drop_ratio']
    conv_dim    = hyperparameters['conv_dim']
    n_ms        = hyperparameters['n_ms']
    JK          = hyperparameters['JK']
    graph_pooling = hyperparameters['graph_pooling']
    mlp_layers  = hyperparameters['mlp_layers']
    mlp_dims    = hyperparameters['mlp_dims']
    
    ######################
    # --- Prediction --- #
    ######################
    
    model = GNN(num_layer=num_layer, drop_ratio=drop_ratio, conv_dim=conv_dim, 
                   gnn_type='NNConv', JK=JK, graph_pooling=graph_pooling,
                   neurons_message=n_ms, mlp_layers=mlp_layers, mlp_dims=mlp_dims)
    
    Y_pred = np.zeros((df.shape[0], k))
    
    for i in tqdm(range(k)):
        kfold_name = 'Kfold_' + str(i+1)

        model.load_state_dict(torch.load(path + '/' + kfold_name + '/' + kfold_name + '.pth', 
                                         map_location=torch.device('cpu')))
        
        model.eval()
        with torch.no_grad():
            for batch_solvent, batch_solute in predict_loader:
                with torch.no_grad():
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
    
    # Models trained on the complete train/validation set
    
    # Given that the data is not open-source, the paths to the data are here 
    # just representative, the same is true for the destination paths of the 
    # predictions
    
    print('Predicting with GNNprevious')
    df = pd.read_csv('../../data/processed/'+str(T)+'_train.csv')
    df_pred = pred_GNNprevious(df, model_name='GNNprevious_'+str(T), 
                      hyperparameters=hyperparameters_dict[T])
    df_pred.to_csv('../../models/isothermal/predictions/GNNprevious/'
                   +str(T)+'_train_pred.csv')
    
    df = pd.read_csv('../../data/processed/'+str(T)+'_test.csv')
    df_pred = pred_GNNprevious(df, model_name='GNNprevious_'+str(T), 
                      hyperparameters=hyperparameters_dict[T])
    df_pred.to_csv('../../models/isothermal/predictions/GNNprevious/'
                   +str(T)+'_test_pred.csv')
    print('Done!')
    
    # # Models trained using kfold CV
    # print('Predicting with GNNprevious_kfolds')
    # df = pd.read_csv('../../data/processed/'+str(T)+'_train.csv')
    # df_pred = pred_GNNprevious_kfolds(df, model_name='GNNprevious_'+str(T), 
    #                   hyperparameters=hyperparameters_dict[T],
    #                   k=5)
    # df_pred.to_csv('../../models/isothermal/predictions/GNNprevious/'
    #                +str(T)+'_train_pred_kfolds.csv')
    
    # df = pd.read_csv('../../data/processed/'+str(T)+'_test.csv')
    # df_pred = pred_GNNprevious_kfolds(df, model_name='GNNprevious_'+str(T), 
    #                   hyperparameters=hyperparameters_dict[T],
    #                   k=5)
    # df_pred.to_csv('../../models/isothermal/predictions/GNNprevious/'
    #                +str(T)+'_test_pred_kfolds.csv')
    # print('Done!')
    



