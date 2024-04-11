import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import Sequential, to_hetero

import lightning as L

class BaseModel(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        metadata,
        layer,
        edge_types,
        rev_edge_types,
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        optim=None,
        crit=None,
        **kwargs
    ):
        super(BaseModel, self).__init__()

        self.edge_types = edge_types
        self.rev_edges_types = rev_edge_types
        
        for i in range(len(hidden_channels)):
            if i == 0:
                layers = [
                    (layer(in_channels, hidden_channels[i]), 'x, edge_index -> x'),
                    activation
                ]
            else:
                layers.extend([
                    (layer(hidden_channels[i-1], hidden_channels[i]), 'x, edge_index -> x'),
                    activation
                ])
            if batch_norm:
                layers.append(
                    nn.BatchNorm1d(hidden_channels[i])
                )
            if drop_out is not None:
                layers.append(
                    nn.Dropout(drop_out)
                )

        layers.append(
            (layer(hidden_channels[-1], out_channels), 'x, edge_index -> x')
        )

        self.encoder = to_hetero(Sequential('x, edge_index', layers), metadata)

        self.set_optimizer(optim)
        self.set_criteria(crit)

    def set_optimizer(self, optim=None):
        if optim is None:
            optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.optim = optim

    def set_criteria(self, crit=None):
        if crit is None:
            crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.crit = crit

    def configure_optimizers(self):
        return self.optim
    
    def encode(self, x_dict, edge_index_dict):            
        return self.encoder(x_dict, edge_index_dict)
    