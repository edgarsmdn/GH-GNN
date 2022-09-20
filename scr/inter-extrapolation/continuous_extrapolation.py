'''
Project: GNN_IAC_T

Analize the continous extrapolation/interpolation capabilitites
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE
import os
import numpy as np

df_train = pd.read_csv('../../models/temperature_dependency/predictions/train_predictions.csv')
df_test = pd.read_csv('../../models/temperature_dependency/predictions/test_predictions.csv')

methods = [
    'SolvGNNGH',
    'GNNGH_T',
           ]

folder='../../models/temperature_dependency/continuous_extrapolation'
if not os.path.exists(folder):
    os.makedirs(folder)

# Get keys for unique binary systems and collect temperatures
def get_unique_binary_sys(df):
    keys = {}
    for i in range(df.shape[0]):
        key = df['Solvent_name'].iloc[i] + '_' + df['Solute_name'].iloc[i]
        T = df['T'].iloc[i]
        if key not in keys:
            keys[key] = [T]
        else:
            if T not in keys[key]:
                keys[key].append(T)

    # Sort the temperatures
    for j in keys:
        keys[j] = sorted(keys[j])
    return keys
    
unique_train = get_unique_binary_sys(df_train)
unique_test = get_unique_binary_sys(df_test)

only_in_test=[]
for key in unique_test:
    if key in unique_train:
        pass
    else:
        only_in_test.append(key)

print('Number of unique binary systems in train: ', len(unique_train))
print('Number of unique binary systems in test: ', len(unique_test))
print('Number of binary systems in test that are not in train: ', len(only_in_test))

# Get extrapolation and interpolation systems
lower_extrapolation = {}
upper_extrapolation = {}
interpolation = {}

for key in unique_test:
    if key in unique_train:
        Ts_train = unique_train[key]
        Ts_test = unique_test[key]
        for T in Ts_test:
            if T < min(Ts_train):
                if key not in lower_extrapolation:
                    lower_extrapolation[key] = [T]
                else:
                    lower_extrapolation[key].append(T)
            elif T > max(Ts_train):
                if key not in upper_extrapolation:
                    upper_extrapolation[key] = [T]
                else:
                    upper_extrapolation[key].append(T)
            else:
                if key not in interpolation:
                    interpolation[key] = [T]
                else:
                    interpolation[key].append(T)
                

print('\n')
print('Number of lower extrapolation systems: ', len(lower_extrapolation))
print('Number of upper extrapolation systems: ', len(upper_extrapolation))
print('Number of interpolation systems: ', len(interpolation))


keys_test_T=[]
for i in range(df_test.shape[0]):
    keys_test_T.append( df_test['Solvent_name'].iloc[i] + '_' + df_test['Solute_name'].iloc[i] + '_' + str(df_test['T'].iloc[i]))

df_test['key_T'] = keys_test_T  

def get_extra_inter_results(mode, dict_systems):
    y_true = []
    y_pred = {method:[] for method in methods}
    distance = []
    system_keys = []
    solvent_smiles = []
    solute_smiles = []
    solvent_name = []
    solute_name = []
    Ts = []
    for key in dict_systems:
        for T in dict_systems[key]:
            key_T = key + '_' + str(T)
            y_true.append(df_test['log-gamma'][df_test['key_T'] == key_T].values[0])
            solvent_smiles.append(df_test['Solvent_SMILES'][df_test['key_T'] == key_T].values[0])
            solute_smiles.append(df_test['Solute_SMILES'][df_test['key_T'] == key_T].values[0])
            solvent_name.append(df_test['Solvent_name'][df_test['key_T'] == key_T].values[0])
            solute_name.append(df_test['Solute_name'][df_test['key_T'] == key_T].values[0])
            Ts.append(df_test['T'][df_test['key_T'] == key_T].values[0])
            if mode == 'lower_extrapolation':
                distance.append(min(unique_train[key]) - T)
            elif mode == 'upper_extrapolation':
                distance.append(T - max(unique_train[key]))
            elif mode == 'interpolation':
                # Distance to the closest T in training
                distance.append(min(np.abs(T - np.array(unique_train[key]))))
            system_keys.append(key_T)
            for method in methods:
                y_pred[method].append(df_test[method][df_test['key_T'] == key_T].values[0])
                
    df = pd.DataFrame({'System_key':system_keys,
                       'Solvent_SMILES': solvent_smiles,
                       'Solute_SMILES': solute_smiles,
                       'Solvent_name': solvent_name,
                       'Solute_name': solute_name,
                       'T': Ts,
                       'Distance':distance,
                       'log-gamma':y_true})
    for method in methods:
        df[method] = y_pred[method]
        
    df.to_csv(folder+'/' + mode + '_predictions.csv', index=False)

    # Open report file
    report = open(folder+'/Report_'+mode+'.txt', 'w')
    def print_report(string, file=report):
        print(string)
        file.write('\n' + string)
    print_report('-'*30)
    print_report(mode)
    for method in methods:
        mae = MAE(y_true, y_pred[method])
        print_report('%12s  %20s  %12s' % (' --- MAE ', method, str(mae)))
    report.close() 
    
    
##########################################################
# --- Get performance on lower extrapolation systems --- #
##########################################################
get_extra_inter_results('lower_extrapolation', lower_extrapolation)

##########################################################
# --- Get performance on upper extrapolation systems --- #
##########################################################
get_extra_inter_results('upper_extrapolation', upper_extrapolation)

####################################################
# --- Get performance on interpolation systems --- #
####################################################
get_extra_inter_results('interpolation', interpolation)



###########################################################
# --- Parity plot for interpolation and extrapolation --- #
###########################################################

import matplotlib.pyplot as plt

method = 'GNNGH_T'
folder = 'results/continuous_extrapolation/'
modes = ['interpolation', 'lower_extrapolation', 'upper_extrapolation']
titles = ['Interpolation', 'Lower extrapolation', 'Upper extrapolation']

for mode, title in zip(modes, titles):
    df = pd.read_csv(folder+mode+'_predictions.csv')
    fig = plt.figure(figsize=(7, 6))
    
    df = df.sort_values('Distance')
    y_true = df['log-gamma'].to_numpy()
    y_pred = df[method].to_numpy()
    distances = df['Distance'].to_numpy()
    
    plt.plot([np.min(y_true), np.max(y_true)],
              [np.min(y_true), np.max(y_true)], '--', c='0.5')
    plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
    im = plt.scatter(y_true, y_pred, 
                s=40, 
                edgecolors='k', 
                alpha=0.5, 
                marker='o', 
                cmap='plasma', 
                c=distances,
                linewidth=0.5)
    plt.title(title, fontsize=22)
    plt.xlabel('Experimental $ln(\gamma_{ij}^{\infty})$', fontsize=18); plt.ylabel('Predicted $ln(\gamma_{ij}^{\infty})$', fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=15)
    cax = fig.add_axes([0.15, 0.81, 0.5, 0.05])
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.close(fig)
    fig.savefig(folder+'/parity_'+mode+'_'+method+'.png', dpi=300, format='png')
    # fig.savefig(folder+'/parity_'+mode+'_'+method+'.svg', dpi=300, format='svg')




            
    
    
    