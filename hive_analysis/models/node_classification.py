import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

from torch_geometric.nn import Sequential, SAGEConv, GATConv, GATv2Conv, to_hetero, HGTConv, GraphConv

import lightning as L

from .base_model import BaseModel
        
        
class BaseCDModel(BaseModel):
    def __init__(
        self,
        in_channels: int,
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
        super(BaseCDModel, self).__init__(
            in_channels=in_channels,
            out_channels=hidden_channels[-1],
            hidden_channels=hidden_channels[:-1],
            metadata=metadata,
            layer=layer,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        
        self.node_types = metadata[0]
        self._build_mlp_head(hidden_channels[-1], activation)
        
        self.set_optimizer(optim)
        
    def _build_mlp_head(self, in_channels, activation):
        self.mlp = {}
        # self.nid = {k:v for v, k in enumerate(self.node_types)}
        for node_type in self.node_types:
            self.mlp[node_type] = nn.Sequential(
                nn.Linear(in_channels, in_channels*2),
                activation,
                nn.Linear(in_channels*2, in_channels//4),
                activation,
                nn.Linear(in_channels//4, 1),
            )
        self.mlp = nn.ModuleDict(self.mlp)
            
    
    def set_criteria(self, crit=None):
        if crit is None:
            crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.crit = crit
        
    def encode(self, x_dict, edge_index_dict, node_types=None):         
        z = self.encoder(x_dict, edge_index_dict)
        feats = []
        if node_types is None:
            node_types = x.dict.keys()
        for node_type in node_types:
            feats.append(self.mlp[node_type](z[node_type]))
        return torch.cat(feats).squeeze(-1)
    
    def forward(self, data):
        return nn.Sigmoid()(self.encode(data.x_dict, data.edge_index_dict))
    
    def forward_trainval(self, data, node_types, masks):
        labels = []
        preds = []
        z = self.encoder(data.x_dict, data.edge_index_dict)
        for node_type, mask in zip(node_types, masks):
            preds.append(self.mlp[node_type](z[node_type][mask]))
            labels.append(data[node_type]['node_label'][mask])
        return torch.cat(preds).squeeze(-1), torch.cat(labels)
    
    def training_step(self, train_batch, batch_idx):
        # forward pass
        data, node_types, masks = train_batch
        pred, label = self.forward_trainval(data, node_types, masks)
        
        # loss calculation
        loss = self.crit(pred, label.float())
        
        # log training loss
        self.log(f'train_loss', loss.item(), batch_size=1)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        # forward pass
        data, node_types, masks = val_batch
        pred, label = self.forward_trainval(data, node_types, masks)
        
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
        

class GraphConvNet(BaseCDModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        metadata,
        edge_types,
        rev_edge_types,
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
            metadata=metadata,
            layer=layer,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        
        
class GraphSAGE(BaseCDModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        metadata,
        edge_types,
        rev_edge_types,
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
            metadata=metadata,
            layer=layer,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )


class GAT(BaseCDModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        metadata,
        edge_types,
        rev_edge_types,
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
            metadata=metadata,
            layer=layer,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        
        
class GATv2(BaseCDModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        metadata,
        edge_types,
        rev_edge_types,
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
            metadata=metadata,
            layer=layer,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            optim=optim,
            crit=crit,
            **kwargs
        )
        

class HGT(BaseCDModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        metadata,
        edge_types,
        rev_edge_types,
        num_heads=1,
        batch_norm=False,
        drop_out=0.2,
        activation=nn.ReLU(),
        optim=None,
        crit=None,
        **kwargs
    ):
        L.LightningModule.__init__(self)
        self.num_inner_layers = len(hidden_channels)-1
        self.edge_types = edge_types
        self.rev_edges_types = rev_edge_types
        
        for i in range(self.num_inner_layers):
            layer_head = [activation,]
            if batch_norm:
                layer_head.append(
                    nn.BatchNorm1d(hidden_channels[i])
                )
            if drop_out is not None:
                layer_head.append(
                    nn.Dropout(drop_out)
                )
            layer_head = nn.Sequential(*layer_head)
            
            if i == 0:
                self.encoder = nn.ModuleList([
                    HGTConv(in_channels, hidden_channels[i], metadata=metadata, heads=num_heads),
                    layer_head
                ])
            else:
                self.encoder.extend([
                    HGTConv(hidden_channels[i-1], hidden_channels[i], metadata=metadata, heads=num_heads),
                    layer_head
                ])
                
        self.encoder.append(
            HGTConv(hidden_channels[-2], hidden_channels[-1], metadata=metadata, heads=num_heads)
        )        
        
        
        self.node_types = metadata[0]
        self._build_mlp_head(hidden_channels[-1], activation)
        
        self.set_optimizer(optim)
        self.set_criteria(crit)
        
    def encode(self, x_dict, edge_index_dict):
        for i in range(self.num_inner_layers):
            x_dict = self.encoder[i*2](x_dict, edge_index_dict)
            x_dict = {
                node_type: self.encoder[i*2+1](x)
                for node_type, x in x_dict.items()
            }
        z = self.encoder[-1](x_dict, edge_index_dict)
        feats = []
        for node_type in x_dict:
            feats.append(self.mlp[node_type](z[node_type]))
        return torch.cat(feats).squeeze(-1)
    
    def forward_trainval(self, data, node_types, masks):
        labels = []
        preds = []
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for i in range(self.num_inner_layers):
            x_dict = self.encoder[i*2](x_dict, edge_index_dict)
            x_dict = {
                node_type: self.encoder[i*2+1](x)
                for node_type, x in x_dict.items()
            }
        z = self.encoder[-1](x_dict, edge_index_dict)
        
        for node_type, mask in zip(node_types, masks):
            preds.append(self.mlp[node_type](z[node_type][mask]))
            labels.append(data[node_type]['y'][mask])
        return torch.cat(preds).squeeze(-1), torch.cat(labels)
            