import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GINConv, TopKPooling
from torch.nn import Sequential
import pdb
F_act = F.relu
class GNN_encoder(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, gnn_name = 'gin', node_feature_size = 2):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_encoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        ### add residual connection or not
        self.node_feature_size = node_feature_size

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        if gnn_name == 'gin':
            self.convs.append(GINConv(Sequential(nn.Linear(self.node_feature_size, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))))
        elif gnn_name == 'gcn':
            self.convs.append(GCNConv(self.node_feature_size, emb_dim)) 
        elif gnn_name == 'sage':
            self.convs.append(SAGEConv(self.node_feature_size, emb_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim,track_running_stats=False))

        for layer in range(num_layer-1):
            if gnn_name == 'gin':
                self.convs.append(GINConv(Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))))
            elif gnn_name == 'gcn':
                self.convs.append(GCNConv(emb_dim, emb_dim))
            elif gnn_name == 'sage':
                self.convs.append(SAGEConv(emb_dim, emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_name))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim,track_running_stats=False))
            # self.batch_norms.append(GraphNorm(emb_dim))

    def forward(self, nodes, edge_indexs, graph_indicators):
        x, edge_index, batch = nodes, edge_indexs, graph_indicators
        ### computing input node embedding
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F_act(h), self.drop_ratio, training = self.training)
            h_list.append(h)
        ### Different implementations of Jk-concat
        node_representation = h_list[-1]
        return node_representation


