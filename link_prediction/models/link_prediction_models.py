import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from sklearn.metrics import roc_auc_score, f1_score

from torch_geometric.nn import Sequential, GCNConv, SAGEConv, GATv2Conv, to_hetero

import lightning as L



class BaseLPModel(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        metadata,
        layer=GCNConv,
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        optim=None,
        crit=None,
        **kwargs
    ):
        super(BaseLPModel, self).__init__()
        for i in range(len(hidden_channels)):
            if i == 0:
                self.layers = [
                    (layer(in_channels, hidden_channels[i], add_self_loops=False), 'x, edge_index -> x'),
                    activation
                ]
            else:
                self.layers.extend([
                    (layer(hidden_channels[i-1], hidden_channels[i], add_self_loops=False), 'x, edge_index -> x'),
                    activation
                ])
            if batch_norm:
                self.layers.append(
                    nn.BatchNorm1d(hidden_channels[i])
                )
            if drop_out is not None:
                self.layers.append(
                    nn.Dropout(drop_out)
                )
            
        self.layers.append(
            (layer(hidden_channels[-1], out_channels, add_self_loops=False), 'x, edge_index -> x')
        )
        
        self.layers = to_hetero(Sequential('x, edge_index', self.layers), metadata)

        self.set_optimizer(optim)
        self.set_criteria(crit)

    def encode(self, x_dict, edge_index_dict):            
        return self.layers(x_dict, edge_index_dict)

    def decode(self, node1, node2, edge_index): # only pos and neg edges
        # edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) # concatenate pos and neg edges
        # logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product

        logits = nn.Sigmoid()(node1[edge_index[0]] * node2[edge_index[1]]).sum(dim=-1)
        return logits
    
    def decode_all(self, z):
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list
    
    def forward(self, data, mode='infer'):
        if mode=='trainval':
            data, edge_type = data
            # data, pos_edge_index, neg_edge_index = data
            z = self.encode(data.x_dict, data.edge_index_dict)
            return self.decode(z[edge_type[0]], z[edge_type[2]], data[edge_type].edge_label_index)
        if mode=='embed':
            return self.encode(data.x_dict, data.edge_index_dict)
            
        return self.decode_all(self.encode(data.x_dict, data.edge_index_dict))

    def set_optimizer(self, optim=None):
        if optim is None:
            optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.optim = optim

    def set_criteria(self, crit=None):
        if crit is None:
            crit = nn.BCEWithLogitsLoss()
        self.crit = crit

    def set_trainval_info(self, edge_type, train_batch_size, val_batch_size):
        self.edge_type = edge_type
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def configure_optimizers(self):
        return self.optim

    def training_step(self, train_batch, batch_idx):
        # forward pass
        pred = self.forward((train_batch, self.edge_type), mode='trainval')

        # loss calculation
        ground_truth = train_batch[self.edge_type].edge_label
        loss = self.crit(pred, ground_truth)

        # log training loss
        self.log(f'{self.edge_type[1]}_train_loss', loss, batch_size=self.train_batch_size)
        
        return loss


    def validation_step(self, val_batch, batch_idx):
        # forward pass
        pred = self.forward((val_batch, self.edge_type), mode='trainval')

        # loss calculation
        ground_truth = val_batch[self.edge_type].edge_label
        loss = self.crit(pred, ground_truth)

        # metrics calculation
        roc_auc = roc_auc_score(ground_truth.long().cpu(), pred.cpu())
        f1 = f1_score(ground_truth.long().cpu(), pred.cpu().round(),average="micro")

        # log validation loss and metrics
        self.log(f'{self.edge_type[1]}_val_loss', loss, batch_size=self.val_batch_size)
        self.log(f'{self.edge_type[1]}_val_roc_auc', roc_auc, batch_size=self.val_batch_size)
        self.log(f'{self.edge_type[1]}_val_f1_score', f1, batch_size=self.val_batch_size)
        


class GCN(BaseLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        metadata,
        improved=False,
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        **kwargs
    ):
        layer = partial(GCNConv, improved=improved)
        super(GCN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            metadata=metadata,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            **kwargs
        )


class GraphSAGE(BaseLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        metadata,
        aggr_scheme='mean',
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        **kwargs
    ):
        layer = partial(SAGEConv, aggr=aggr_scheme, normalize=False)
        super(GraphSAGE, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            metadata=metadata,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            **kwargs
        )


class GATv2(BaseLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        metadata,
        num_heads=1,
        concat=True,
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        **kwargs
    ):
        layer = partial(GATv2Conv, heads=num_heads, concat=concat)
        super(GATv2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            metadata=metadata,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            **kwargs
        )



    