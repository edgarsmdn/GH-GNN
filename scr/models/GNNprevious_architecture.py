'''
Project: GNN_IAC_T
                    GNNprevious architecture
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from utilities.mol2graph import n_atom_features, n_bond_features
import torch


class GNN_mol(nn.Module):
    def __init__(self, num_layer, drop_ratio, conv_dim, gnn_type, JK, neurons_message=None):
        super(GNN_mol, self).__init__()

        # Initialization
        self.num_layer  = num_layer
        self.drop_ratio = drop_ratio
        self.conv_dim   = conv_dim
        self.gnn_type   = gnn_type
        self.JK         = JK
        self.neurons_message = neurons_message

        self.node_dim   =  n_atom_features()      # Num features nodes
        self.edge_dim   =  n_bond_features()      # Num features edges

        # List of GNNs
        self.convs       = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First embedding atom_dim to conv_dim
        self.linatoms = nn.Linear(self.node_dim , conv_dim)
        
        # GRU
        #self.gru = nn.GRU(conv_dim, conv_dim)

        # GNN layers
        for layer in range(num_layer):
            if gnn_type == 'NNConv':
              neurons_message = self.neurons_message
              mes_nn = nn.Sequential(nn.Linear(self.edge_dim, neurons_message), nn.ReLU(), nn.Linear(neurons_message, conv_dim**2))
              self.convs.append(gnn.NNConv(conv_dim, conv_dim, mes_nn, aggr='mean'))
            else:
                ValueError(f'Undefined GNN type called {gnn_type}') 
            self.batch_norms.append(nn.BatchNorm1d(conv_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr

        # Original node dimension to first conv dimension
        x = F.leaky_relu(self.linatoms(x))
        #h = x.unsqueeze(0)

        # GNN layers
        x_list = [x]
        for layer in range(self.num_layer):
            x = self.convs[layer](x_list[layer], edge_index, edge_attr)
            x = self.batch_norms[layer](x)
            
            # Remove activation function from last layer
            if layer == self.num_layer - 1 and False:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.leaky_relu(x), self.drop_ratio, training=self.training)
            #out, h = self.gru(x.unsqueeze(0), h)
            #out = out.squeeze(0)
            x_list.append(x)

        ### Jumping knowledge https://arxiv.org/pdf/1806.03536.pdf
        if self.JK == "last":
            x = x_list[-1]
        elif self.JK == "sum":
            x = 0
            for layer in range(self.num_layer):
                x += x_list[layer]
        elif self.JK == "mean":
            x = 0
            for layer in range(self.num_layer):
                x += x_list[layer]
            x = x/self.num_layer
        return x
    
class GNN(nn.Module):
    def __init__(self, num_layer=3, drop_ratio=0.5, conv_dim=100, gnn_type='NNConv', JK='last', graph_pooling='mean', neurons_message=None,
                 mlp_layers=None, mlp_dims=None):
        super(GNN, self).__init__()
        
        # Initialization
        self.num_layer     = num_layer
        self.drop_ratio    = drop_ratio
        self.conv_dim      = conv_dim
        self.gnn_type      = gnn_type
        self.JK            = JK
        self.graph_pooling = graph_pooling
        self.neurons_message = neurons_message
        self.mlp_layers    = mlp_layers
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNNs
        self.gnn_solvent = GNN_mol(num_layer, drop_ratio, conv_dim, gnn_type, JK, neurons_message)
        self.gnn_solute  = GNN_mol(num_layer, drop_ratio, conv_dim, gnn_type, JK, neurons_message)

        # Whole-graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(conv_dim, 2*conv_dim), 
                                                                    nn.BatchNorm1d(2*conv_dim), 
                                                                    nn.ReLU(), 
                                                                    nn.Linear(2*conv_dim, 1)))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(conv_dim, processing_steps=3)
        else:
            raise ValueError("Invalid graph pooling type.")

        # Multi-layer perceptron
        self.mlp             = nn.ModuleList()
        self.batch_norms_mlp = nn.ModuleList()
        if graph_pooling == "set2set":
            mlp_layer_1 = 4*self.conv_dim
        else:
            mlp_layer_1 = 2*self.conv_dim
        mlp_dims_complete = [mlp_layer_1] + mlp_dims
        for layer in range(mlp_layers):
            self.mlp.append(nn.Linear(mlp_dims_complete[layer], mlp_dims_complete[layer+1]))
            self.batch_norms_mlp.append(nn.BatchNorm1d(mlp_dims_complete[layer+1]))
            
    def forward(self, solvent, solute):
        x_solvent = self.gnn_solvent(solvent)   
        x_solvent = self.pool(x_solvent, solvent.batch)
        
        x_solute  = self.gnn_solute(solute)
        x_solute  = self.pool(x_solute, solute.batch)
        
        x = torch.cat((x_solvent, x_solute), dim=1) # Concatenate solvent and solute embeddings
        
        # MLP
        for layer in range(self.mlp_layers):
            x = self.mlp[layer](x)
            # Remove activation function from last layer
            if layer == self.mlp_layers - 1:
                x = x
            else:
                x = self.batch_norms_mlp[layer](x)
                x = F.dropout(F.leaky_relu(x), self.drop_ratio, training=self.training) 
        return x