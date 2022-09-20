'''
Project: GNN_IAC_T

Get results from the Brouwer dataset for extrapolation
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
import os

methods = [
    'SolvGNNGH',
    'GNNGH_T'
           ]

folder = '../../models/temperature_dependency/discrete_extrapolation'
if not os.path.exists(folder):
    os.makedirs(folder)
    
# Collect predictions of all

for split in ['edge', 'extrapolation']:
    df_all = pd.read_csv('../../data/processed/brouwer_'+ split +'_test.csv')
    for method in methods:
        if method == 'SolvGNNGH':
            path_pred = '../../models/temperature_dependency/predictions/SolvGNNGH_brouwer_' + split + '_pred.csv'
        elif method == 'GNNGH_T':
            path_pred = '../../models/temperature_dependency/predictions/GH-GNN_brouwer_' + split + '_pred.csv'
        
        df = pd.read_csv(path_pred)
        pred = df[method].to_numpy()
        df_all[method] = pred
        
        # path_pred = method + '/' + best_model + '/brouwer_' + split + '_pred_kfolds.csv'
        # df = pd.read_csv(path_pred)
        # pred = df[best_model].to_numpy()
        # df_all[method+'_kfolds'] = pred
        
    df_all.to_csv(folder + '/brouwer_' + split + '_predictions.csv', index=False)




