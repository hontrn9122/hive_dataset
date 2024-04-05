import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

from torch_geometric.nn import Sequential, SAGEConv, GATConv, GATv2Conv, to_hetero, HGTConv

import lightning as L

from .base_model import BaseModel
        
        
class BaseLPModel(BaseModel):
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
        
        self.encoder = {}
        self.edge_types = edge_types
        self.rev_edges_types = rev_edge_types
        
        # self.eid = {k: v for v, k in enumerate(edge_types)}
        
        for edge_type in edge_types:
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

            self.encoder[''.join(edge_type)] = to_hetero(Sequential('x, edge_index', layers), metadata)
        
        self.encoder = nn.ModuleDict(self.encoder)
        self.set_optimizer(optim)
        self.set_criteria(crit)

    def set_criteria(self, crit=None):
        if crit is None:
            crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.crit = crit
    
    def encode(self, x_dict, edge_index_dict, edge_type):            
        return self.encoder[''.join(edge_type)](x_dict, edge_index_dict)

    def decode(self, node1, node2, edge_index, sigmoid=False): # only pos and neg edges
        logits = (F.normalize(node1[edge_index[0]]) * F.normalize(node2[edge_index[1]])).sum(dim=-1)
        return torch.sigmoid(logits) if sigmoid else logits
    
    def decode_all(self, node1, node2, sigmoid=True):
        logits = torch.matmul(node1, node2.t())
        return torch.sigmoid(logits) if sigmoid else logits
    
    def forward(self, data, edge_type, edge_index):
        z = self.encode(data.x_dict, data.edge_index_dict, edge_type)
        return self.decode(z[edge_type[0]], z[edge_type[2]], edge_index)
    
    def forward_all(self, data, edge_types=None):
        outputs = {}
        if edge_types is None:
            edge_types = [etype for etype in data.edge_types if etype in self.edge_types]
        for edge_type in edge_types:
            z = self.encode(data.x_dict, data.edge_index_dict, edge_type)
            outputs[edge_type] = self.decode_all(z[edge_type[0]], z[edge_type[2]])
        return outputs
    
    def forward_trainval(self, data, key, edge_types=None):
        preds, labels = [], []
        if edge_types is None:
            edge_types = [etype for etype in data.edge_types if etype in self.edge_types]
        for edge_type in edge_types:
            z = self.encode(data.x_dict, data.edge_index_dict, edge_type)
            preds.append(self.decode(z[edge_type[0]], z[edge_type[2]], data[edge_type][f'{key}_index']))
            labels.append(data[edge_type][key])
        return torch.cat(preds), torch.cat(labels)
    
    def training_step(self, train_batch, batch_idx):
        # forward pass
        data, edge_types, key = train_batch
        edge_types = [tuple(edge_type) for edge_type in edge_types]
        pred, label = self.forward_trainval(data, key, edge_types)
        
        # loss calculation
        loss = self.crit(pred, label)
        
        # log training loss
        self.log(f'train_loss', loss.item(), batch_size=1)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        # forward pass
        data, edge_types, key = val_batch
        edge_types = [tuple(edge_type) for edge_type in edge_types]
        pred, label = self.forward_trainval(data, key, edge_types)
        
        # loss calculation
        loss = self.crit(pred, label)

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
        

class GraphSAGE(BaseLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            metadata=metadata,
            layer=layer,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            **kwargs
        )


class GAT(BaseLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            metadata=metadata,
            layer=layer,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
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
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            metadata=metadata,
            layer=layer,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            **kwargs
        )
        

class HGT(BaseLPModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        self.num_inner_layers = len(hidden_channels)
        self.encoder = {}
        self.edge_types = edge_types
        self.rev_edges_types = rev_edge_types
        # self.eid = {k: v for v, k in enumerate(edge_types)}
        
        for edge_type in edge_types:
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
                    layers = nn.ModuleList([
                        HGTConv(in_channels, hidden_channels[i], metadata=metadata, heads=num_heads),
                        layer_head
                    ])
                else:
                    layers.extend([
                        HGTConv(hidden_channels[i-1], hidden_channels[i], metadata=metadata, heads=num_heads),
                        layer_head
                    ])

            layers.append(
                HGTConv(hidden_channels[-1], out_channels, metadata=metadata, heads=num_heads)
            )
            
            self.encoder[''.join(edge_type)] = layers
        
        self.encoder = nn.ModuleDict(self.encoder)
        self.set_optimizer(optim)
        self.set_criteria(crit)
        
    def encode(self, x_dict, edge_index_dict, edge_type):
        edge_type = ''.join(edge_type)
        for i in range(self.num_inner_layers):
            x_dict = self.encoder[edge_type][i*2](x_dict, edge_index_dict)
            x_dict = {
                node_type: self.encoder[edge_type][i*2+1](x)
                for node_type, x in x_dict.items()
            }
        return self.encoder[edge_type][-1](x_dict, edge_index_dict)
            