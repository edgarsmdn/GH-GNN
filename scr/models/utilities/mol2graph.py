'''
Project: GNN_IAC_T

                              IAC_T mol2graph specific 

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''

from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from rdkit.Chem import rdMolDescriptors
from mordred.Polarizability import APol, BPol
from mordred.TopoPSA import TopoPSA
from tqdm import tqdm

##################################################
# --- Parameters to be specified by the user --- #
##################################################

'''
possible_atom_list: Symbols of possible atoms

possible_atom_degree: The degree of an atom is defined to be its number of directly-bonded neighbors. 
                        The degree is independent of bond orders, but is dependent on whether or not Hs are 
                        explicit in the graph.
                        
possible_atom_valence: The implicit atom valence: number of implicit Hs on the atom

possible_hybridization: Type of Hybridazation. Nice reminder here:
    https://www.youtube.com/watch?v=vHXViZTxLXo
    
NOTE!!: Bond features are within the function bond_features

'''

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

possible_stereo  = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE]

#########################
# --- Atom features --- #
#########################

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]                               # Specify as Unknown
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    '''
    Get atom features
    '''
    Symbol       = atom.GetSymbol()
    
    # Features
    Type_atom     = one_of_k_encoding(Symbol, possible_atom_list)
    Ring_atom     = [atom.IsInRing()]
    Aromaticity   = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Bonds_atom    = one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    Formal_charge = one_of_k_encoding(atom.GetFormalCharge(), possible_formal_charge)
    num_Hs        = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)
    Type_chirality= one_of_k_encoding(atom.GetChiralTag(), possible_chiralities)
    
    # Merge features in a list
    results = Type_atom + Ring_atom + Aromaticity + Hybridization + \
        Bonds_atom + Formal_charge + num_Hs + Type_chirality
    
    return np.array(results).astype(np.float32)

#########################
# --- Bond features --- #
#########################

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def bond_features(bond):
    '''
    Get bond features
    '''
    bt = bond.GetBondType()
    
    type_stereo = one_of_k_encoding(bond.GetStereo(), possible_stereo)
    
    # Features
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()] + type_stereo
    return np.array(bond_feats).astype(np.float32)


###################################
# --- Molecule to torch graph --- #
###################################
def mol2torchdata(df, mol_column, target, y_scaler=None):
    '''
    Takes a molecule and return a graph
    '''
    graphs=[]
    mols = df[mol_column].tolist()
    ys   = df[target].tolist()
    for mol, y in zip(mols, ys):
        atoms  = mol.GetAtoms()
        bonds  = mol.GetBonds()
        
        # Information on nodes
        node_f = [atom_features(atom) for atom in atoms]
        
        # Information on edges
        edge_index = get_bond_pair(mol)
        edge_attr  = []
        
        for bond in bonds:
            edge_attr.append(bond_features(bond))
            edge_attr.append(bond_features(bond))
        
        # Store all information in a graph
        nodes_info = torch.tensor(np.array(node_f), dtype=torch.float)
        edges_indx = torch.tensor(np.array(edge_index), dtype=torch.long)
        edges_info = torch.tensor(np.array(edge_attr), dtype=torch.float)
        
        
        graph = Data(x=nodes_info, edge_index=edges_indx, edge_attr=edges_info)
        
        if y_scaler != None:
            y = np.array(y).reshape(-1,1)
            y = y_scaler.transform(y).astype(np.float32)
            graph.y = torch.tensor(y[0], dtype=torch.float)
        else:
            #y = y.astype(np.float32)
            graph.y = torch.tensor(y, dtype=torch.float)
        
        
        graphs.append(graph)
    
    return graphs


##########################################
# --- System to dictionary with data --- #
##########################################

def sys2graph(df, mol_column_1, mol_column_2, target, y_scaler=None, single_system=False):
    '''
    Return graphs for solvent and solute with hydrogen-bonding info
    '''
    if single_system:
        solvents = [df[mol_column_1]]
        solutes = [df[mol_column_2]]
        ys   = [df[target]]
    else:
        print('-- Constructing graphs...')
        solvents = df[mol_column_1].tolist()
        solutes = df[mol_column_2].tolist()
        ys   = df[target].tolist()
    
    graphs_solvent, graphs_solute = [], []
    for y, solv, solu in tqdm(zip(ys, solvents, solutes), total=len(ys)):
        atoms_solv  = solv.GetAtoms()
        bonds_solv  = solv.GetBonds()
        
        atoms_solu  = solu.GetAtoms()
        bonds_solu  = solu.GetBonds()
        
        # Information on nodes
        node_f_solv = [atom_features(atom) for atom in atoms_solv]
        node_f_solu = [atom_features(atom) for atom in atoms_solu]
        
        # Information on edges
        edge_index_solv = get_bond_pair(solv)
        edge_attr_solv  = []
        
        for bond in bonds_solv:
            edge_attr_solv.append(bond_features(bond))
            edge_attr_solv.append(bond_features(bond))
            
        edge_index_solu = get_bond_pair(solu)
        edge_attr_solu  = []
        
        for bond in bonds_solu:
            edge_attr_solu.append(bond_features(bond))
            edge_attr_solu.append(bond_features(bond))
        
        # Atomic polarizability
        calc = APol()
        ap_solv = calc(solv)
        ap_solu = calc(solu)
        
        # Bond polarizability
        calc = BPol()
        bp_solv = calc(solv)
        bp_solu = calc(solu)
        
        # Topological Polar Surface Area
        calc = TopoPSA()
        topopsa_solv = calc(solv)
        topopsa_solu = calc(solu)
        
        # Intra hydrogen-bond acidity and basicity
        hb_solv = min(rdMolDescriptors.CalcNumHBA(solv), rdMolDescriptors.CalcNumHBD(solv))
        hb_solu = min(rdMolDescriptors.CalcNumHBA(solu), rdMolDescriptors.CalcNumHBD(solu))
        
        # Inter hydrogen-bond
        inter_hb = min(rdMolDescriptors.CalcNumHBA(solv), rdMolDescriptors.CalcNumHBD(solu)) \
                             + min(rdMolDescriptors.CalcNumHBA(solu),rdMolDescriptors.CalcNumHBD(solv))
        
        # Store all information in a graph
        nodes_info_solv = torch.tensor(np.array(node_f_solv), dtype=torch.float)
        edges_indx_solv = torch.tensor(np.array(edge_index_solv), dtype=torch.long)
        edges_info_solv = torch.tensor(np.array(edge_attr_solv), dtype=torch.float)
        graph_solv = Data(x=nodes_info_solv, edge_index=edges_indx_solv, edge_attr=edges_info_solv,
                          ap=ap_solv,
                          bp=bp_solv,
                          topopsa=topopsa_solv,
                          hb=hb_solv, 
                          inter_hb=inter_hb)
        
        nodes_info_solu = torch.tensor(np.array(node_f_solu), dtype=torch.float)
        edges_indx_solu = torch.tensor(np.array(edge_index_solu), dtype=torch.long)
        edges_info_solu = torch.tensor(np.array(edge_attr_solu), dtype=torch.float)
        graph_solu = Data(x=nodes_info_solu, edge_index=edges_indx_solu, edge_attr=edges_info_solu,
                          ap=ap_solu,
                          bp=bp_solu,
                          topopsa=topopsa_solu, 
                          hb=hb_solu, 
                          inter_hb=inter_hb)
        
        if y_scaler != None:
            y = np.array(y).reshape(-1,1)
            y = y_scaler.transform(y).astype(np.float32)
            graph_solv.y = torch.tensor(y[0], dtype=torch.float)
            graph_solu.y = torch.tensor(y[0], dtype=torch.float)
        else:
            #y = y.astype(np.float32)
            graph_solv.y = torch.tensor(y, dtype=torch.float)
            graph_solu.y = torch.tensor(y, dtype=torch.float)
        
        graphs_solvent.append(graph_solv)
        graphs_solute.append(graph_solu)
        
    return graphs_solvent, graphs_solute
    
def get_dummy_graph():
    solv = Chem.MolFromSmiles('[2H]O[2H]')
    atoms_solv  = solv.GetAtoms()
    bonds_solv  = solv.GetBonds()
    node_f_solv = [atom_features(atom) for atom in atoms_solv]
    edge_index_solv = get_bond_pair(solv)
    edge_attr_solv  = []
    
    for bond in bonds_solv:
        edge_attr_solv.append(bond_features(bond))
        edge_attr_solv.append(bond_features(bond))
        
    # Store all information in a graph
    nodes_info_solv = torch.tensor(np.array(node_f_solv), dtype=torch.float)
    edges_indx_solv = torch.tensor(np.array(edge_index_solv), dtype=torch.long)
    edges_info_solv = torch.tensor(np.array(edge_attr_solv), dtype=torch.float)
    ap, bp, topopsa, hb, inter_hb, y = [torch.tensor([1])]*6
    graph_dummy = Data(x=nodes_info_solv, edge_index=edges_indx_solv, edge_attr=edges_info_solv,
                       ap=ap, bp=bp, topopsa=topopsa, hb=hb, inter_hb=inter_hb, y=y)
    return graph_dummy

def cat_dummy_graph(batch):
    dummy_graph = get_dummy_graph()
    if torch.cuda.is_available():
        dummy_graph  = dummy_graph.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        dummy_graph = dummy_graph.cuda()
    if batch.edge_index.shape[1] == 0:
        max_node_idx = batch.x.shape[0]
    else:
        max_node_idx = batch.edge_index.max() + 1
        
    batch.x = torch.cat((batch.x, dummy_graph.x), axis=0)
    batch.edge_index = torch.cat((batch.edge_index, dummy_graph.edge_index + max_node_idx), axis=1)
    batch.edge_attr = torch.cat((batch.edge_attr, dummy_graph.edge_attr), axis=0)
    batch.ap = torch.cat((batch.ap, dummy_graph.ap))
    batch.bp = torch.cat((batch.bp, dummy_graph.bp))
    batch.topopsa = torch.cat((batch.topopsa, dummy_graph.topopsa))
    batch.hb = torch.cat((batch.hb, dummy_graph.hb))
    batch.inter_hb = torch.cat((batch.inter_hb, dummy_graph.inter_hb))
    batch.y = torch.cat((batch.y, dummy_graph.y))
    
    dummy_n_nodes = dummy_graph.x.shape[0]
    batch_dummy = torch.tensor([batch.num_graphs]).repeat(dummy_n_nodes)
    ptr_dummy = torch.tensor([batch.ptr[-1] + dummy_n_nodes])
    if torch.cuda.is_available():
        batch_dummy = batch_dummy.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        batch_dummy = batch_dummy.cuda()
        ptr_dummy = ptr_dummy.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        ptr_dummy = ptr_dummy.cuda()
    batch.batch = torch.cat((batch.batch, batch_dummy))
    batch.ptr = torch.cat((batch.ptr, ptr_dummy))
    return batch

def cat_dummy_T(T):
    T.x = torch.cat((T.x, torch.tensor([25])))
    T.batch = torch.cat((T.batch, torch.tensor([T.num_graphs])))
    T.ptr = torch.cat((T.ptr, torch.tensor([T.ptr[-1]+1])))
    return T
    

##########################
# --- Count features --- #
##########################

def n_atom_features():
    atom = Chem.MolFromSmiles('CC').GetAtomWithIdx(0)
    return len(atom_features(atom))


def n_bond_features():
    bond = Chem.MolFromSmiles('CC').GetBondWithIdx(0)
    return len(bond_features(bond))

#######################
# --- Data loader --- #
#######################

def get_dataloader(df, index, target, graph_column, batch_size, shuffle=False, drop_last=False):
    
    x = df.loc[index, graph_column].tolist() # Get graphs (x)
    data_loader = DataLoader(x, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) 
    
    # Note: drop_last argument drops the last non-full batch of each worker’s 
    #       iterable-style dataset replica. This ensure all batches to be of equal size
    
    return data_loader

######################################
# --- Data loader 2Graphs_1Output--- #
######################################

from torch_geometric.data import Batch
from torch.utils.data import Dataset

"""

class PairDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, idx):
        return self.datasetA[idx], self.datasetB[idx]
    
    def __len__(self):
        return len(self.datasetA)
    

def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB


# For 2Graph-1Output
def get_dataloader_pairs(df, index, graphs_solvent, graphs_solute, batch_size, shuffle=False, drop_last=False):
    
    x_solvent = df.loc[index, graphs_solvent].tolist() # Get graphs for solvent
    x_solute  = df.loc[index, graphs_solute].tolist()  # Get graphs for solute
    
    pair_dataset = PairDataset(x_solvent, x_solute)
    data_loader  = torch.utils.data.DataLoader(pair_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate) 
    
    return data_loader
"""
###########################################################
# --- Data loader 2Graphs_1Output and system property --- #
###########################################################

class PairDataset_T(Dataset):
    def __init__(self, datasetA, datasetB, datasetT):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetT = datasetT

    def __getitem__(self, idx):
        return self.datasetA[idx], self.datasetB[idx], self.datasetT[idx]
    
    def __len__(self):
        return len(self.datasetA)
    

def collate_T(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, batchC

# For 2Graph-1Output
def get_dataloader_pairs_T(df, index, graphs_solvent, graphs_solute, batch_size, shuffle=False, drop_last=False):
    
    x_solvent = df.loc[index, graphs_solvent].tolist() # Get graphs for solvent
    x_solute  = df.loc[index, graphs_solute].tolist()  # Get graphs for solute
    Temp      = [Data(x=torch.tensor(t, dtype=torch.float).reshape(1)) for t in df.loc[index, 'T'].tolist()] # Get system temperatures
    
    pair_dataset = PairDataset_T(x_solvent, x_solute, Temp)
    data_loader  = torch.utils.data.DataLoader(pair_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_T) 
    
    return data_loader



