import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv, GINConv, SuperGATConv, TransformerConv, DNAConv, MLP, Sequential

class MLP(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=64, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.loss=nn.MSELoss()

        self.mlp_layers = torch.nn.ModuleList()
        self.mlp_layers.append(torch.nn.Linear(self.in_channels, self.hidden_channels))
        self.mlp_layers.append(torch.nn.ReLU())
        self.mlp_layers.append(torch.nn.Dropout(0.3))
        for _ in range(self.num_layers-2):
            self.mlp_layers.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
            self.mlp_layers.append(torch.nn.ReLU())
            self.mlp_layers.append(torch.nn.Dropout(0.3))
        self.mlp_layers.append(torch.nn.Linear(self.hidden_channels, self.out_channels))
    
    def forward(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        return x


class GNN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_targets, dropout, attn=False, num_lin=2, conv='sage'):
        super().__init__()
        self.num_features = num_features
        self.num_targets = num_targets
        self.hidden_channels = hidden_channels
        self.num_lin = num_lin
        self.dropout = dropout
        self.attn = attn
        self.loss=nn.MSELoss()

        if conv == 'sage':
            conv_layer1 = SAGEConv(num_features, hidden_channels)
            conv_layer2 = SAGEConv(hidden_channels, hidden_channels)
        elif conv == 'gin':
            lin1 = nn.ModuleList()
            lin1.append(nn.Linear(num_features, hidden_channels))
            for _ in range(self.num_lin-1):
             lin1.append(nn.Linear(hidden_channels, hidden_channels))
            lin2 = nn.ModuleList()
            for _ in range(self.num_lin):
              lin2.append(nn.Linear(hidden_channels, hidden_channels))
            conv_layer1 = GINConv(nn.Sequential(*lin1))
            conv_layer2 = GINConv(nn.Sequential(*lin2))
        elif conv == 'transformer':
            conv_layer1 = TransformerConv(num_features, hidden_channels)
            conv_layer2 = TransformerConv(hidden_channels, hidden_channels)
        elif conv == 'gat':
            conv_layer1 = SuperGATConv(num_features, hidden_channels)
            conv_layer2 = SuperGATConv(hidden_channels, hidden_channels)
        elif conv == 'dna':
            conv_layer1 = DNAConv(num_features, hidden_channels)
            conv_layer2 = DNAConv(hidden_channels, hidden_channels)

        else:
            raise ValueError(f'Invalid convolution layer {conv}')

        self.convs = Sequential('x, edge_index',[
            (conv_layer1, 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (conv_layer2, 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (nn.Dropout(p=dropout), 'x -> x')],)
        self.fc = nn.Linear(hidden_channels, num_targets)


    def forward(self, x, edge_index):
        x = self.convs(x, edge_index)
        return self.fc(x)






