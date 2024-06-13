import torch
import torch.nn as nn

from models.layers import GraphSAGE
from parser import parameter_parser

args = parameter_parser()

class GraphSAGEModel(nn.Module):
    """
    Baseline GraphSAGE Network with a stack of GraphSAGE Layers and global pooling over nodes.
    """
    def __init__(self, in_features, out_features, filters=args.filters,
                 n_hidden=args.n_hidden, dropout=args.dropout):
        super(GraphSAGEModel, self).__init__()
        # GraphSAGE layers
        self.gconv = nn.ModuleList([GraphSAGE(in_features=in_features if layer == 0 else filters[layer - 1],
                                              out_features=f, activation=nn.ReLU(inplace=True)) for layer, f in enumerate(filters)])
        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x, adj = data.x, data.adj
        for layer in self.gconv:
            x = layer(x, adj)
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x = self.fc(x)
        return x
