'''
Project: GNN_IAC_T
                    GNN-Gibbs-Helmholtz architecture
                    
Author: Edgar Sanchez
-------------------------------------------------------------------------------
'''
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
import torch_geometric.nn as gnn
import numpy as np
from torch_scatter import scatter_mean
from torch_scatter import scatter_add
from GHGNN.mol2graph import cat_dummy_graph


class MPNNconv(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats,
                 edge_hidden_feats=32, num_step_message_passing=1):
        super(MPNNconv, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats*node_out_feats)
        )
        self.gnn_layer = gnn.NNConv(
            node_out_feats,
            node_out_feats,
            edge_network,
            aggr='add'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()
        
    def forward(self, system_graph):
        
        node_feats = system_graph.x
        edge_index = system_graph.edge_index
        edge_feats = system_graph.edge_attr
        
        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            if torch.cuda.is_available():
                node_feats = F.relu(self.gnn_layer(x=node_feats.type(torch.FloatTensor).cuda(), 
                                                   edge_index=edge_index.type(torch.LongTensor).cuda(),
                                                   edge_attr=edge_feats.type(torch.FloatTensor).cuda()))
            else:
                node_feats = F.relu(self.gnn_layer(x=node_feats.type(torch.FloatTensor), 
                                                   edge_index=edge_index.type(torch.LongTensor),
                                                   edge_attr=edge_feats.type(torch.FloatTensor)))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats


class EdgeModel(torch.nn.Module):
    def __init__(self, v_in, e_in, u_in, hidden_dim):
        super().__init__()

        layers = [nn.Linear(v_in*2 + e_in + u_in, hidden_dim),
                  nn.ReLU(),
                  nn.Linear(hidden_dim, hidden_dim)]

        self.edge_mlp = nn.Sequential(*layers)

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [N] with max entry B - 1.
        # print(' ')
        # print(src.shape)
        # print(dest.shape)
        # print(edge_attr.shape)
        # print(u[batch].shape)
        # print(' ')
        out = torch.cat([src, dest, edge_attr, u[batch]], axis=1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, v_in, u_in, hidden_dim):
        super().__init__()

        layers = [nn.Linear(v_in + hidden_dim + u_in, hidden_dim),
                  nn.ReLU(),
                  nn.Linear(hidden_dim, hidden_dim)]

        self.node_mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, u_in, hidden_dim):
        super().__init__()
        
        layers = [nn.Linear(hidden_dim + hidden_dim + u_in, hidden_dim),
                  nn.ReLU(),
                  nn.Linear(hidden_dim, hidden_dim)]

        self.global_mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        node_aggregate = scatter_mean(x, batch, dim=0)
        edge_aggregate = scatter_mean(edge_attr, batch[edge_index[1]], 
                                      dim=0, 
                                      out=edge_attr.new_zeros(node_aggregate.shape))
        out = torch.cat([u, node_aggregate, edge_aggregate], dim=1)
        return self.global_mlp(out)



class GHGNN_model(nn.Module):
    def __init__(self, v_in, e_in, u_in, hidden_dim):
        super(GHGNN_model, self).__init__()
        
        self.graphnet1 = gnn.MetaLayer(EdgeModel(v_in, e_in, u_in, hidden_dim),
                                      NodeModel(v_in, u_in, hidden_dim), 
                                      GlobalModel(u_in, hidden_dim))
        self.graphnet2 = gnn.MetaLayer(EdgeModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim),
                                      NodeModel(hidden_dim, hidden_dim, hidden_dim), 
                                      GlobalModel(hidden_dim, hidden_dim))
        
        self.gnorm1 = gnn.GraphNorm(hidden_dim)
        self.gnorm2 = gnn.GraphNorm(hidden_dim)
        
        self.pool = global_mean_pool
        
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim*2,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim*2)
        
        # MLP for A
        self.mlp1a = nn.Linear(hidden_dim*4, hidden_dim)
        self.mlp2a = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3a = nn.Linear(hidden_dim, 1)
        
        # MLP for B
        self.mlp1b = nn.Linear(hidden_dim*4, hidden_dim)
        self.mlp2b = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3b = nn.Linear(hidden_dim, 1)
        
        # MLP for C
        self.mlp1c = nn.Linear(hidden_dim*4, hidden_dim)
        self.mlp2c = nn.Linear(hidden_dim, hidden_dim)
        self.mlp3c = nn.Linear(hidden_dim, 1)
        
        
    def generate_sys_graph(self, x, edge_attr, batch_size, n_mols=2):
        
        src = np.arange(batch_size)
        dst = np.arange(batch_size, n_mols*batch_size)
        
        # Self-connections (within same molecule)
        self_connection = np.arange(n_mols*batch_size)
        
        # Biderectional connections (between each molecule in the system)
        # and self-connection
        one_way = np.concatenate((src, dst, self_connection))
        other_way = np.concatenate((dst, src, self_connection))
        
        edge_index = torch.tensor([list(one_way),
                                   list(other_way)], dtype=torch.long)
        sys_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return sys_graph

    def forward(self, solvent, solute, T):
        
        # Molecular descriptors based on MOSCED model
        
        # -- Induction via polarizability 
        # -- (i.e., dipole-induced dipole or induced dipole - induced dipole)
        
        # ---- Atomic polarizability
        ap1 = solvent.ap.reshape(-1,1)
        ap2 = solute.ap.reshape(-1,1)
        
        # ---- Bond polarizability
        bp1 = solvent.bp.reshape(-1,1)
        bp2 = solute.bp.reshape(-1,1)
        
        # -- Polarity via topological polar surface area
        topopsa1 = solvent.topopsa.reshape(-1,1)
        topopsa2 = solute.topopsa.reshape(-1,1)
        
        # -- Hydrogen-bond acidity and basicity
        intra_hb1 = solvent.inter_hb
        intra_hb2 = solute.inter_hb
        
        u1 = torch.cat((ap1,bp1,topopsa1), axis=1) # Molecular descriptors solvent
        u2 = torch.cat((ap2,bp2,topopsa2), axis=1) # Molecular descriptors solute
        
        #### - Security check for predicting single node graphs (e.g. water)
        single_node_batch=False
        if solvent.edge_attr.shape[0] == 0 or solute.edge_attr.shape[0] == 0:
            solvent = cat_dummy_graph(solvent)
            solute = cat_dummy_graph(solute)
            u1_dummy = torch.tensor([1,1,1]).reshape(1,-1)
            u2_dummy = torch.tensor([1,1,1]).reshape(1,-1)
            if torch.cuda.is_available():
                u1_dummy = u1_dummy.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                u1_dummy = u1_dummy.cuda() 
                u2_dummy = u1_dummy.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                u2_dummy = u1_dummy.cuda() 
            
            u1 = torch.cat((u1, u1_dummy), axis=0)
            u2 = torch.cat((u2, u2_dummy), axis=0)
            single_node_batch=True
        
        # Solvent GraphNet
        x1, edge_attr1, u1 = self.graphnet1(solvent.x, 
                                           solvent.edge_index, 
                                           solvent.edge_attr, 
                                           u1, 
                                           solvent.batch)
        x1 = self.gnorm1(x1, solvent.batch)
        
        x1, edge_attr1, u1 = self.graphnet2(x1, 
                                           solvent.edge_index, 
                                           edge_attr1, 
                                           u1, 
                                           solvent.batch) 
        x1 = self.gnorm2(x1, solvent.batch)

        xg1 = self.pool(x1, solvent.batch)
        
        # Solute GraphNet
        x2, edge_attr2, u2 = self.graphnet1(solute.x, 
                                           solute.edge_index, 
                                           solute.edge_attr, 
                                           u2, 
                                           solute.batch) 
        x2 = self.gnorm1(x2, solute.batch)
        x2, edge_attr2, u2 = self.graphnet2(x2, 
                                           solute.edge_index, 
                                           edge_attr2, 
                                           u2, 
                                           solute.batch)
        x2 = self.gnorm2(x2, solute.batch)

        xg2 = self.pool(x2, solute.batch)
        
        if single_node_batch: # Eliminate prediction for dummy graph
            xg1 = xg1[:-1,:]
            xg2 = xg2[:-1,:]
            u1 = u1[:-1,:]
            u2 = u2[:-1,:]
            solvent.inter_hb = solvent.inter_hb[:-1]
            solute.inter_hb = solute.inter_hb[:-1]
            batch_size = solvent.y.shape[0] - 1
        else:
            batch_size = solvent.y.shape[0]
            
        # Intermolecular descriptors
        # -- Hydrogen bonding 
        inter_hb  = solvent.inter_hb
        
        # Construct binary system graph
        node_feat = torch.cat((
            torch.cat((xg1, u1), axis=1),
            torch.cat((xg2, u2), axis=1)),axis=0)
        edge_feat = torch.cat((inter_hb.repeat(2),
                             intra_hb1,
                             intra_hb2)).unsqueeze(1)
        
        binary_sys_graph = self.generate_sys_graph(x=node_feat,
                                                   edge_attr=edge_feat,
                                                   batch_size=batch_size)
        
        # Binary system fingerprint
        xg = self.global_conv1(binary_sys_graph)
        xg = torch.cat((xg[0:len(xg)//2,:], xg[len(xg)//2:,:]), axis=1)
        
        T = T.x.reshape(-1,1) + 273.15
        
        A = F.relu(self.mlp1a(xg))
        A = F.relu(self.mlp2a(A))
        A = self.mlp3a(A)   
        
        B = F.relu(self.mlp1b(xg))
        B = F.relu(self.mlp2b(B))
        B = self.mlp3b(B)
        
        C = F.relu(self.mlp1c(xg*T))
        C = F.relu(self.mlp2c(C))
        C = self.mlp3c(C)
        
        output = A + B/T + C
        
        
        return output    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

