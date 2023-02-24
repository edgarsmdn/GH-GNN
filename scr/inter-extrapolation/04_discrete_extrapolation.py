'''
Project: GNN_IAC_T

Analize the discrete extrapolation/interpolation capabilitites
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE
import os
from rdkit import Chem
from tqdm import tqdm
import numpy as np
from rdkit.DataStructs import FingerprintSimilarity as FPS
import matplotlib.pyplot as plt

df_train = pd.read_csv('results/train_predictions.csv')
df_test = pd.read_csv('results/test_predictions.csv')

methods = [
    # 'GNNCat',
    # 'GH-GNN_wo',
    # 'SolvGNNCat',
    # 'SolvGNNGH_wo',
    'GH-GNN'
           ]

folder='results/discrete_extrapolation'
if not os.path.exists(folder):
    os.makedirs(folder)
    

df_train['key'] = df_train['Solvent_name'] + '_' + df_train['Solute_name']
df_test['key'] = df_test['Solvent_name'] + '_' + df_test['Solute_name']
unique_train = df_train['key'].unique()
unique_test = df_test['key'].unique()

interpolation = []
edge = []
extrapolation = []

same_sys = 0
for key in df_test['key'].tolist():
    if key in df_train['key'].tolist():
        same_sys += 1
    else:
        delimiter = key.find('_')
        solvent = key[:delimiter]
        solute = key[delimiter+1:]
        if solvent in df_train['Solvent_name'].tolist() and solute in df_train['Solute_name'].tolist():
            # Both species are present but not in this precise combination
            interpolation.append(key)
        elif solvent in df_train['Solvent_name'].tolist() or solute in df_train['Solute_name'].tolist():
            # Only one of the species is present in the training set
            edge.append(key)
        else:
            # None of the species is present in the training set
            extrapolation.append(key)
            
print('Number of discrete interpolation systems: ', len(interpolation))
print('Number of discrete edge systems         : ', len(edge))
print('Number of discrete extrapolation systems: ', len(extrapolation))

# Get extrapolation systems from the Brouwer, 2021 dataset 
extrapol_brouwer_path = '../data/processed/brouwer_extrapolation_test.csv'
edge_brouwer_path = '../data/processed/brouwer_edge_test.csv'

def feasible_sys(solv, solu):
    # Check for consistency of molecular classes
    possible_atom_list = ['C','N','O','Cl','S','F','Br','I','Si','Sn','Pb','Ge',
                          'H','P','Hg', 'Te']

    possible_hybridization = [Chem.rdchem.HybridizationType.S,
                              Chem.rdchem.HybridizationType.SP, 
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3]

    possible_chiralities =[Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                           Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                           Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]

    possible_num_bonds = [0,1,2,3,4]

    possible_formal_charge = [0,1,-1]

    possible_num_Hs  = [0,1,2,3]
    
    solv = Chem.MolFromSmiles(solv)
    solu = Chem.MolFromSmiles(solu)
    
    feasible_sys = True
    for mol in [solv, solu]:
        atoms = mol.GetAtoms()
        for atom in atoms:
            if atom.GetSymbol() not in possible_atom_list:
                feasible_sys = False
            elif atom.GetHybridization() not in possible_hybridization:
                feasible_sys = False
            elif atom.GetChiralTag() not in possible_chiralities:
                feasible_sys = False
            elif len(atom.GetNeighbors()) not in possible_num_bonds:
                feasible_sys = False
            elif atom.GetFormalCharge() not in possible_formal_charge:
                feasible_sys = False
            elif atom.GetTotalNumHs() not in possible_num_Hs:
                feasible_sys = False
            elif atom.GetTotalNumHs() not in possible_num_Hs:
                feasible_sys = False
    return feasible_sys
    

if not os.path.exists(extrapol_brouwer_path) or not os.path.exists(edge_brouwer_path):
    print('\n Getting extrapolation/edge systems from Brouwer dataset...')
    df_brouwer = pd.read_csv('../data/raw/Brouwer_2021.csv')
    
    extrapolation_idx = []
    edge_idx = [] 
    edge_rare = []
    rare_solu_solv = []
    for i in tqdm(range(df_brouwer.shape[0])):
        solv = Chem.MolToSmiles(Chem.MolFromSmiles(df_brouwer['Solvent_SMILES'].iloc[i]))
        solu = Chem.MolToSmiles(Chem.MolFromSmiles(df_brouwer['Solute_SMILES'].iloc[i]))
        if solv in df_train['Solvent_SMILES'].tolist() and solu in df_train['Solute_SMILES'].tolist():
            pass
        elif solv in df_train['Solvent_SMILES'].tolist() or solu in df_train['Solute_SMILES'].tolist():
            if feasible_sys(solv, solu):
                edge_idx.append(i)
                if solv in df_train['Solvent_SMILES'].tolist():
                    edge_rare.append(solu)
                    rare_solu_solv.append('Solute')
                else:
                    edge_rare.append(solv)
                    rare_solu_solv.append('Solvent')
        else:
            if feasible_sys(solv, solu):
                extrapolation_idx.append(i)    
    
    df_brouwer_edge = df_brouwer.loc[edge_idx]
    df_brouwer_edge['Rare_species'] = edge_rare
    df_brouwer_edge['Rare_solvent_solute'] = rare_solu_solv
    edge_classyfire = pd.read_csv('../data/interim/brouwer_edge_classified_with_classyfire.csv')
    key_class_edge = []
    for i in range(df_brouwer_edge.shape[0]):
        try:
            solv_key = edge_classyfire['class'][edge_classyfire['SMILES'] == df_brouwer_edge['Solvent_SMILES'].iloc[i]].values[0]
        except:
            solv_key = ''
        try:
            solu_key = edge_classyfire['class'][edge_classyfire['SMILES'] == df_brouwer_edge['Solute_SMILES'].iloc[i]].values[0]
        except:
            solu_key = ''
        key_class_edge.append(str(solv_key) + '_' + str(solu_key))
    df_brouwer_edge['key_class'] = key_class_edge
    df_brouwer_edge.to_csv('../data/processed/brouwer_edge_test.csv', index=False)
    
    df_brouwer_extra = df_brouwer.loc[extrapolation_idx]
    extra_classyfire = pd.read_csv('../data/interim/brouwer_extrapolation_classified_with_classyfire.csv')
    key_class_extra = []
    for i in range(df_brouwer_extra.shape[0]):
        solv_key = extra_classyfire['class'][extra_classyfire['SMILES'] == df_brouwer_extra['Solvent_SMILES'].iloc[i]].values[0]
        solu_key = extra_classyfire['class'][extra_classyfire['SMILES'] == df_brouwer_extra['Solute_SMILES'].iloc[i]].values[0]
        key_class_extra.append(str(solv_key) + '_' + str(solu_key))
    df_brouwer_extra['key_class'] = key_class_extra
    df_brouwer_extra.to_csv('../data/processed/brouwer_extrapolation_test.csv', index=False)
     
    #print('You still need to clean according to class from classyfire and save the clean version in the processed folder!')
else:
    df_brouwer_edge = pd.read_csv(edge_brouwer_path)
    df_brouwer_extra = pd.read_csv(extrapol_brouwer_path)

print('\n Number of discrete edge systems in Brouwer dataset: ', df_brouwer_edge.shape[0])
print('\n Number of discrete extrapolation systems in Brouwer dataset: ', df_brouwer_extra.shape[0])  

def get_inter_edge_results(mode, keys_lst):
    y_true = []
    y_pred = {method:[] for method in methods}
    solvent_smiles = []
    solute_smiles = []
    solvent_name = []
    solute_name = []
    Ts = []
    key_class_lst = []
    for key in keys_lst:
        y_true.append(df_test['log-gamma'][df_test['key'] == key ].values[0])
        solvent_smiles.append(df_test['Solvent_SMILES'][df_test['key'] == key ].values[0])
        solute_smiles.append(df_test['Solute_SMILES'][df_test['key'] == key ].values[0])
        solvent_name.append(df_test['Solvent_name'][df_test['key'] == key ].values[0])
        solute_name.append(df_test['Solute_name'][df_test['key'] == key ].values[0])
        Ts.append(df_test['T'][df_test['key'] == key ].values[0])
        key_class_lst.append(df_test['key_class'][df_test['key'] == key ].values[0])
        for method in methods:
            y_pred[method].append(df_test[method][df_test['key'] == key].values[0])
    
    df = pd.DataFrame({'Solvent_SMILES': solvent_smiles,
                                     'Solute_SMILES': solute_smiles,
                                     'Solvent_name': solvent_name,
                                     'Solute_name': solute_name,
                                     'T': Ts,
                                     'log-gamma':y_true,
                                     'key_class':key_class_lst
                                     })
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
        
        bin03 = (np.abs(np.array(y_true) - np.array(y_pred[method])) <= 0.3).sum()/np.array(y_true).shape[0]*100
        print_report('%12s  %20s  %12s' % (' --- Bin03 ', method, str(bin03)))
    report.close() 
    
def get_clean_systems_results(df, threshold):
    y_true = df['log-gamma'].to_numpy()
    y_pred = df['GH-GNN'].to_numpy()

    y_true_clean = []
    y_pred_clean = []
    for i in range(len(y_true)):
        if np.abs(y_true[i] - y_pred[i]) <= threshold:
            y_true_clean.append(y_true[i])
            y_pred_clean.append(y_pred[i])
            

    print('\nPercentage of predictions clean: ', len(y_true_clean)/len(y_true)*100)
    print('Number of systems clean          : ', len(y_true_clean))
    print(' --- MAE clean                   : ', MAE(y_true_clean, y_pred_clean))    
    
def similarity_analysis(df, df_train, mode, method='GNNGH_T_kfolds'):
    solvs_train = df_train['Solvent_SMILES'].unique()
    solus_train = df_train['Solute_SMILES'].unique()
    
    solvs_train = [Chem.MolFromSmiles(x) for x in solvs_train]
    solus_train = [Chem.MolFromSmiles(x) for x in solus_train]
    
    solvs_fps = [Chem.RDKFingerprint(x) for x in solvs_train]
    solus_fps = [Chem.RDKFingerprint(x) for x in solus_train]
    
    tanimoto_similarities = np.zeros((df.shape[0], len(solvs_train) + len(solus_train)))
    if mode =='edge':
        edge_tanimoto_path = 'results/discrete_extrapolation/tanimoto_brouwer_edge.csv'
        if not os.path.exists(edge_tanimoto_path):
        
            rare_mols = df['Rare_species'].apply(Chem.MolFromSmiles).tolist()
            rare_fps = [Chem.RDKFingerprint(x) for x in rare_mols]
            
            # Compute similarities
            print(' ---> Computing Tanimoto similarities for edge brouwer')
            for i in tqdm(range(tanimoto_similarities.shape[0])):
                for j in range(tanimoto_similarities.shape[1]):
                    if j <= len(solvs_train)-1:
                        tanimoto_similarities[i,j] = FPS(rare_fps[i], solvs_fps[j])
                    else:
                        tanimoto_similarities[i,j] = FPS(rare_fps[i], solus_fps[j - len(solvs_train)])
                        
            df_tanimoto = pd.DataFrame(tanimoto_similarities)
            df_tanimoto.to_csv(edge_tanimoto_path)
        else:
            df_tanimoto = pd.read_csv(edge_tanimoto_path)
            
        # Get group ofmost similiar species per molecule
        print('\nSorting similarities...', end='')
        most_similars = 3
        similarities = np.mean(np.sort(df_tanimoto.to_numpy(), axis=1)[:, :-most_similars], axis=1)
        errors = np.abs(df['log-gamma'].to_numpy() - df[method].to_numpy())
        print('Done!')
        return similarities, errors
                     
    
##################################
# --- Discrete interpolation --- #
##################################
get_inter_edge_results('interpolation', interpolation)

interpolation_df = pd.read_csv(folder+'/interpolation_predictions.csv')
y_true = interpolation_df['log-gamma'].to_numpy()
y_pred = interpolation_df['GH-GNN'].to_numpy()

mae_interpolation = MAE(y_true, y_pred)
bin03_interpolation = (np.abs(y_true - y_pred) <= 0.3).sum()/y_true.shape[0]*100

fig = plt.figure(figsize=(7, 6))
plt.plot([np.min(y_true), np.max(y_true)],
          [np.min(y_true), np.max(y_true)], '--', c='0.5')
plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
im = plt.scatter(y_true, y_pred, 
            s=40, 
            edgecolors='k', 
            alpha=0.5, 
            marker='o', 
            linewidth=0.5)
plt.title('Discrete interpolation', fontsize=22)
plt.xlabel('Experimental $ln(\gamma_{ij}^{\infty})$', fontsize=18); plt.ylabel('Predicted $ln(\gamma_{ij}^{\infty})$', fontsize=18)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=15)

stats_str = f'MAE: {mae_interpolation:.2f}'
plt.text(12, 2, stats_str, fontsize=18, bbox=dict(boxstyle="round",
                   facecolor='white', edgecolor='black'
                   ))
gamma_str = '$ | \gamma_{ij}^\infty | \leq 0.3$'
bin03_ss = f'ln {gamma_str}: {bin03_interpolation:.2f}%'
plt.text(9, -1, bin03_ss, fontsize=18, bbox=dict(boxstyle="round",
                   facecolor='white', edgecolor='black'
                   ))

plt.close(fig)
fig.savefig(folder+'/parity_discrete_interpolation_GNNGH.png', dpi=300, format='png')
fig.savefig(folder+'/parity_discrete_interpolation_GNNGH.svg', dpi=300, format='svg')

#########################
# --- Discrete edge --- #
#########################
get_inter_edge_results('edge', edge)
df_edge = pd.read_csv(folder + '/edge_predictions.csv')
get_clean_systems_results(df_edge, threshold=0.4)

extrapolation_df = pd.read_csv(folder+'/edge_predictions.csv')
extrapolation_df['error_gnngh'] = np.abs(extrapolation_df['log-gamma'].to_numpy() - extrapolation_df['GH-GNN'].to_numpy())
extrapolation_df = extrapolation_df.sort_values(by=['error_gnngh'])

n_best_systems = 73
y_true = extrapolation_df['log-gamma'].to_numpy()
y_pred = extrapolation_df['GH-GNN'].to_numpy()

y_true_good = extrapolation_df['log-gamma'].to_numpy()[:n_best_systems]
y_pred_good = extrapolation_df['GH-GNN'].to_numpy()[:n_best_systems]
error_good = extrapolation_df['error_gnngh'].to_numpy()[:n_best_systems]

y_true_bad = extrapolation_df['log-gamma'].to_numpy()[-(extrapolation_df.shape[0]-n_best_systems):]
y_pred_bad = extrapolation_df['GH-GNN'].to_numpy()[-(extrapolation_df.shape[0]-n_best_systems):]
error_bad = extrapolation_df['error_gnngh'].to_numpy()[-(extrapolation_df.shape[0]-n_best_systems):]

fig = plt.figure(figsize=(7, 6))
plt.plot([np.min(y_true), np.max(y_true)],
          [np.min(y_true), np.max(y_true)], '--', c='0.5')
plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
plt.scatter(y_true_good, y_pred_good, 
            s=40, 
            edgecolors='k', 
            alpha=0.5, 
            marker='o', 
            linewidth=0.5)
plt.scatter(y_true_bad, y_pred_bad, 
            s=40, 
            edgecolors='k', 
            alpha=1, 
            marker='^', 
            linewidth=0.5,
            color='red')
plt.title('Discrete extrapolation', fontsize=22)
plt.xlabel('Experimental $ln(\gamma_{ij}^{\infty})$', fontsize=18); plt.ylabel('Predicted $ln(\gamma_{ij}^{\infty})$', fontsize=18)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=15)
plt.close(fig)
fig.savefig(folder+'/parity_discrete_extrapolation_GNNGH.svg', dpi=300, format='svg')



#################################
# --- Discrete edge Brouwer --- #
#################################
df_edge_brouwer = pd.read_csv(folder + '/brouwer_edge_predictions.csv')
y_true = df_edge_brouwer['log-gamma'].to_numpy()
# Open report file
report = open(folder+'/Report_brouwer_edge.txt', 'w')
def print_report(string, file=report):
    print(string)
    file.write('\n' + string)
print_report('-'*30)
print_report('brouwer_edge')
for method in methods:
    mae = MAE(y_true, df_edge_brouwer[method].to_numpy())
    print_report('%12s  %20s  %12s' % (' --- MAE ', method, str(mae)))
report.close() 

get_clean_systems_results(df_edge_brouwer, threshold=0.4)

df = pd.read_csv('results/discrete_extrapolation/edge_Brouwer_GNNGH_for_parity_Nsystems.csv')
fig = plt.figure(figsize=(7, 6))
df = df.sort_values('Same_key_class_n')
y_true = df['log-gamma'].to_numpy()
y_pred = df['GH-GNN'].to_numpy()
same_class = df['Same_key_class_n'].to_numpy()

plt.plot([np.min(y_true), np.max(y_true)],
          [np.min(y_true), np.max(y_true)], '--', c='0.5')
plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
im = plt.scatter(y_true, y_pred, 
            s=40, 
            edgecolors='k', 
            alpha=0.5, 
            marker='o', 
            cmap='plasma', 
            c=same_class,
            linewidth=0.5)
plt.title('Discrete extrapolation\n external dataset', fontsize=22)
plt.xlabel('Experimental $ln(\gamma_{ij}^{\infty})$', fontsize=18); plt.ylabel('Predicted $ln(\gamma_{ij}^{\infty})$', fontsize=18)
plt.tight_layout()
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=15)
cax = fig.add_axes([0.15, 0.81, 0.5, 0.05])
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.close(fig)
fig.savefig('../Temperature_dependency/results/discrete_extrapolation/parity_extrapolation_Brouwer_GNNGH.png', dpi=300, format='png')
fig.savefig('../Temperature_dependency/results/discrete_extrapolation/parity_extrapolation_Brouwer_GNNGH.svg', dpi=300, format='svg')




bins_classes = np.arange(0, 160, 25)
bins_mae =np.zeros(bins_classes.shape[0])
for i in range(bins_mae.shape[0]):
    df_feasible = df[df['Same_key_class_n'] >= bins_classes[i]]
    bins_mae[i] = MAE(df_feasible['log-gamma'].to_numpy(), df_feasible['GH-GNN'].to_numpy())

fig = plt.figure(figsize=(7, 6))
plt.plot(bins_classes, bins_mae, 'ko-')
plt.xlabel('Threshold of systems with same \nsolute-solvent classes in training set', fontsize=18)
plt.ylabel('Mean absolute error (MAE)', fontsize=18)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=15)
plt.grid(axis='x', color='0.95')
plt.tight_layout()
plt.close(fig)
fig.savefig('../Temperature_dependency/results/discrete_extrapolation/Threshold_n_systems_extrapolation_Brouwer_GNNGH.png', dpi=650, format='png')
fig.savefig('../Temperature_dependency/results/discrete_extrapolation/Threshold_n_systems_extrapolation_Brouwer_GNNGH.svg', dpi=650, format='svg')




train_smiles = list(set(df_train['Solvent_SMILES'].unique().tolist() + df_train['Solute_SMILES'].unique().tolist()))
fps_train = [Chem.RDKFingerprint(Chem.MolFromSmiles(smiles_i)) for smiles_i in train_smiles]
similarities_brouwer_lst = []
for i in range(df.shape[0]):
    fp = Chem.RDKFingerprint(Chem.MolFromSmiles(df['Rare_species'].iloc[i]))
    similarities_brouwer_lst.append(sorted([FPS(fp, fp_train) for fp_train in fps_train]))
    
max_10_sim = []
for similarities in similarities_brouwer_lst:
    max_10_sim.append(np.mean(similarities[-10]))
df['Similarity_max_10']  = max_10_sim

fig = plt.figure(figsize=(7, 6))
df = df.sort_values('Similarity_max_10')
y_true = df['log-gamma'].to_numpy()
y_pred = df['GH-GNN'].to_numpy()
same_class = df['Similarity_max_10'].to_numpy()

plt.plot([np.min(y_true), np.max(y_true)],
          [np.min(y_true), np.max(y_true)], '--', c='0.5')
plt.grid(True, which='major', color='gray', linestyle='dashed', alpha=0.3)
im = plt.scatter(y_true, y_pred, 
            s=40, 
            edgecolors='k', 
            alpha=0.5, 
            marker='o', 
            cmap='plasma', 
            c=same_class,
            linewidth=0.5)
plt.title('Discrete extrapolation\n external dataset', fontsize=22)
plt.xlabel('Experimental $ln(\gamma_{ij}^{\infty})$', fontsize=18); plt.ylabel('Predicted $ln(\gamma_{ij}^{\infty})$', fontsize=18)
plt.tight_layout()
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=15)
cax = fig.add_axes([0.15, 0.81, 0.5, 0.05])
fig.colorbar(im, cax=cax, orientation='horizontal')
plt.close(fig)
fig.savefig('../Temperature_dependency/results/discrete_extrapolation/parity_extrapolation_Brouwer_GNNGH_Tanimoto.png', dpi=300, format='png')
fig.savefig('../Temperature_dependency/results/discrete_extrapolation/parity_extrapolation_Brouwer_GNNGH_Tanimoto.svg', dpi=300, format='svg')




bins_similarity = np.arange(0, 0.8, 0.05)
bins_mae =np.zeros(bins_similarity.shape[0])

for i in range(bins_mae.shape[0]):
    df_feasible = df[df['Similarity_max_10'] >= bins_similarity[i]]
    bins_mae[i] = MAE(df_feasible['log-gamma'].to_numpy(), df_feasible['GH-GNN'].to_numpy())

fig = plt.figure(figsize=(7, 6))
plt.plot(bins_similarity, bins_mae, 'ko-')
plt.xlabel('Threshold of mean Tanimoto similarity \n', fontsize=18)
plt.ylabel('Mean absolute error (MAE)', fontsize=18)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=15)
plt.grid(axis='x', color='0.95')
plt.tight_layout()
plt.close(fig)
fig.savefig('../Temperature_dependency/results/discrete_extrapolation/Threshold_tanimoto_extrapolation_Brouwer_GNNGH.png', dpi=650, format='png')
fig.savefig('../Temperature_dependency/results/discrete_extrapolation/Threshold_tanimoto_extrapolation_Brouwer_GNNGH.svg', dpi=650, format='svg')



##########################################
# --- Discrete extrapolation Brouwer --- #
##########################################
df_extra = pd.read_csv(folder + '/brouwer_extrapolation_predictions.csv')
y_true = df_extra['log-gamma'].to_numpy()
# Open report file
report = open(folder+'/Report_brouwer_extrapolation.txt', 'w')
def print_report(string, file=report):
    print(string)
    file.write('\n' + string)
print_report('-'*30)
print_report('brouwer_extrapolation')
for method in methods:
    mae = MAE(y_true, df_extra[method].to_numpy())
    print_report('%12s  %20s  %12s' % (' --- MAE ', method, str(mae)))
report.close() 

get_clean_systems_results(df_extra, threshold=0.4)















         
