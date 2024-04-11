import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

from torch_geometric.nn import Sequential, SAGEConv, GATConv, GATv2Conv, GraphConv

import lightning as L

from .base_model import BaseModel
        
        
class BaseHomoCDModel(BaseModel):
    def __init__(
        self,
        in_channels: int,
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
        for i in range(len(hidden_channels)-1):
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
            (layer(hidden_channels[-2], hidden_channels[-1]), 'x, edge_index -> x')
        )

        self.encoder = Sequential('x, edge_index', layers)
        
        self._build_mlp_head(hidden_channels[-1], activation)
        
        self.set_criteria(crit)
        self.set_optimizer(optim)
    
        
    def _build_mlp_head(self, in_channels, activation):
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels*2),
            activation,
            nn.Linear(in_channels*2, in_channels//4),
            activation,
            nn.Linear(in_channels//4, 1),
        )
            
    
    def set_criteria(self, crit=None):
        if crit is None:
            crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.crit = crit
        
    def encode(self, x, edge_index):         
        z = self.encoder(x, edge_index)
        feat = self.mlp(z)
        return feat.squeeze(-1)
    
    def forward(self, data):
        return nn.Sigmoid()(self.encode(data.x, data.edge_index))
    
    def forward_trainval(self, data, mask):
        z = self.encoder(data.x, data.edge_index)
        pred = self.mlp(z[mask])
        label = data['node_label'][mask]
        return pred.squeeze(-1), label
    
    def training_step(self, train_batch, batch_idx):
        # forward pass
        data, mask = train_batch
        pred, label = self.forward_trainval(data, mask)
        
        # loss calculation
        loss = self.crit(pred, label.float())
        
        # log training loss
        self.log(f'train_loss', loss.item(), batch_size=1)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        # forward pass
        data, mask = val_batch
        pred, label = self.forward_trainval(data, mask)
        
        # loss calculation
        loss = self.crit(pred, label.float())

        # metrics calculation
        metrics = {f'val_{k}': v for k, v in self.metrics_cal(pred, label).items()}
        
        # log validation loss and metrics
        self.log(f'val_loss', loss.item(), batch_size=1)
        self.log_dict(metrics, batch_size=1)  
    
    def metrics_cal(self, pred, label, sigmoid=True):
        # metrics calculation
        pred = nn.Sigmoid()(pred).cpu() if sigmoid else pred.cpu()
        label = label.long().cpu()
        
        metrics = {}
        metrics['roc_auc'] = roc_auc_score(label, pred)
        metrics['f1'] = f1_score(label, pred.round(),average="micro")
        metrics['precision'] = precision_score(label, pred.round())
        metrics['recall'] = recall_score(label, pred.round())
        metrics['accuracy'] = accuracy_score(label, pred.round())
        
        return metrics
        

class GraphConvNet(BaseHomoCDModel):
    def __init__(
        self,
        in_channels: int,
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
            hidden_channels=hidden_channels,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        

class GraphSAGE(BaseHomoCDModel):
    def __init__(
        self,
        in_channels: int,
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
            hidden_channels=hidden_channels,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )


class GAT(BaseHomoCDModel):
    def __init__(
        self,
        in_channels: int,
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
            hidden_channels=hidden_channels,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        
        
class GATv2(BaseHomoCDModel):
    def __init__(
        self,
        in_channels: int,
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
            hidden_channels=hidden_channels,
            layer=layer,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        