import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

from torch_geometric.nn import Sequential, SAGEConv, GATConv, GATv2Conv, GraphConv
import lightning as L

from .base_model import BaseModel
        
        
class BaseHomoLPModel(BaseModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        layer,
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        optim=None,
        crit=None,
        **kwargs
    ):
        super(BaseModel, self).__init__()
        
        # self.eid = {k: v for v, k in enumerate(edge_types)}
        
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

        self.encoder = Sequential('x, edge_index', layers)
        
        self.set_optimizer(optim)
        self.set_criteria(crit)

    def set_criteria(self, crit=None):
        if crit is None:
            crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.crit = crit
    
    def encode(self, x, edge_index):            
        return self.encoder(x, edge_index)

    def decode(self, x, edge_index, sigmoid=False): # only pos and neg edges
        logits = (F.normalize(x[edge_index[0]]) * F.normalize(x[edge_index[1]])).sum(dim=-1)
        return torch.sigmoid(logits) if sigmoid else logits
    
    def decode_all(self, x, sigmoid=True):
        logits = torch.matmul(x, x.t())
        return torch.sigmoid(logits) if sigmoid else logits
    
    def forward(self, data, edge_type, edge_index):
        z = self.encode(data.x, data.edge_index)
        return self.decode(z, edge_index)
    
    def forward_all(self, data):
        z = self.encode(data.x_dict, data.edge_index_dict, edge_type)
        output = self.decode_all(z)
        return output
    
    def forward_trainval(self, data, key):
        z = self.encode(data.x, data.edge_index)
        pred = self.decode(z, data[f'{key}_index'])
        label = data[key]
        
        return pred, label
    
    def training_step(self, train_batch, batch_idx):
        # forward pass
        data, key = train_batch
        pred, label = self.forward_trainval(data, key)
        
        # loss calculation
        loss = self.crit(pred, label.float())
        
        # log training loss
        self.log(f'train_loss', loss.item(), batch_size=1)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        # forward pass
        data, key = val_batch
        pred, label = self.forward_trainval(data, key)
        
        # loss calculation
        loss = self.crit(pred, label.float())

        # metrics calculation
        metrics = self.metrics_cal(pred, label)
        
        # log validation loss and metrics
        self.log(f'val_loss', loss.item(), batch_size=1)
        self.log_dict(metrics, batch_size=1)
    
    def metrics_cal(self, pred, label, sigmoid=True):
        # metrics calculation
        pred = nn.Sigmoid()(pred).cpu() if sigmoid else pred.cpu()
        label = label.long().cpu()
        
        metrics = {}
        metrics['val_roc_auc'] = roc_auc_score(label, pred.round())
        metrics['val_f1'] = f1_score(label, pred.round(),average="micro")
        metrics['val_precision'] = precision_score(label, pred.round())
        metrics['val_recall'] = recall_score(label, pred.round())
        metrics['val_accuracy'] = accuracy_score(label, pred.round())
        return metrics
        

class GraphConvNet(BaseHomoLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        aggr_scheme='mean',
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        optim=None,
        crit=None,
        **kwargs
    ):
        layer = partial(GraphConv, aggr=aggr_scheme)
        super(GraphConvNet, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        
        
class GraphSAGE(BaseHomoLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        aggr_scheme='mean',
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        optim=None,
        crit=None,
        **kwargs
    ):
        layer = partial(SAGEConv, aggr=aggr_scheme, normalize=False)
        super(GraphSAGE, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )


class GAT(BaseHomoLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        num_heads=1,
        concat=True,
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        optim=None,
        crit=None,
        **kwargs
    ):
        layer = partial(GATConv, heads=num_heads, concat=concat, add_self_loops=False)
        super(GAT, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        
        
class GATv2(BaseHomoLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list,
        num_heads=1,
        concat=True,
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        optim=None,
        crit=None,
        **kwargs
    ):
        layer = partial(GATv2Conv, heads=num_heads, concat=concat, add_self_loops=False)
        super(GATv2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        
            