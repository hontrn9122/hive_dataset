import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# from sklearn.cluster import DBSCAN
from dbscan import DBSCAN
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from link_prediction.utils import silhouette
from torchmetrics.clustering import CalinskiHarabaszScore, DaviesBouldinScore, DunnIndex

from torch_geometric.nn import Sequential, SAGEConv, GATConv, GATv2Conv, to_hetero, HGTConv

import lightning as L
from fast_pytorch_kmeans import KMeans

from .base_model import BaseModel
        
# def euclidean_distance(x1, x2):
#     return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))

# def dbscan(X, eps, min_samples):
#     device = X.device
#     n_samples = X.size(0)
#     labels = torch.zeros(n_samples, dtype=torch.int, device=device)
 
#     # Initialize cluster label and visited flags
#     cluster_label = 0
#     visited = torch.zeros(n_samples, dtype=torch.bool, device=device)
 
#     # Iterate over each point
#     for i in range(n_samples):
#         if visited[i]:
#             continue
#         visited[i] = True
 
#         # Find neighbors
#         neighbors = torch.nonzero(euclidean_distance(X[i], X) < eps).squeeze(-1)
         
#         if neighbors.size(0) < min_samples:
#             # Label as noise
#             labels[i] = -1
#         else:
#             # Expand cluster
#             labels[i] = cluster_label
#             expand_cluster(X, labels, visited, neighbors, cluster_label, eps, min_samples)
#             cluster_label += 1
#     print('Done clustering!')
#     return labels


# def expand_cluster(X, labels, visited, neighbors, cluster_label, eps, min_samples):
#     i = 0
#     while i < neighbors.shape[0]:
#         neighbor_index = neighbors[i].item()
#         if not visited[neighbor_index]:
#             visited[neighbor_index] = True
#             neighbor_neighbors = torch.nonzero(euclidean_distance(X[neighbor_index], X) < eps).squeeze()
#             if neighbor_neighbors.shape[0] >= min_samples:
#                 neighbors = torch.cat((neighbors, neighbor_neighbors))
#         if labels[neighbor_index] == -1:
#             labels[neighbor_index] = cluster_label
#         i += 1

class BaseCDModel(BaseModel):
    def set_criteria(self, crit=None):
        if crit is None:
            crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.crit = crit
    
    def decode(self, node1, node2, edge_index, sigmoid=False): # only pos and neg edges
        logits = (F.normalize(node1[edge_index[0]]) * F.normalize(node2[edge_index[1]])).sum(dim=-1)
        return torch.sigmoid(logits) if sigmoid else logits
    
    def forward(self, data):
        return nn.Sigmoid()(self.encode(data.x_dict, data.edge_index_dict))
    
    def forward_trainval(self, data, edge_types=None):
        preds, labels = [], []
        if edge_types is None:
            edge_types = [etype for etype in data.edge_types if etyp in self.edge_types]
        z = self.encode(data.x_dict, data.edge_index_dict)
        embed = torch.cat([feat for feat in z.values()])
        for edge_type in edge_types:
            preds.append(self.decode(z[edge_type[0]], z[edge_type[2]], data[edge_type].edge_label_index))
            labels.append(data[edge_type].edge_label)
        return embed, torch.cat(preds), torch.cat(labels)
    
    def training_step(self, train_batch, batch_idx):
        # forward pass
        data, edge_types = train_batch
        edge_types = [tuple(edge_type) for edge_type in edge_types]
        _, pred, label = self.forward_trainval(data, edge_types)
        
        # loss calculation
        loss = self.crit(pred, label)
        
        # log training loss
        self.log(f'train_loss', loss.item(), batch_size=1)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        # forward pass
        data, edge_types = val_batch
        edge_types = [tuple(edge_type) for edge_type in edge_types]
        embed, pred, label = self.forward_trainval(data, edge_types)
        
        # loss calculation
        loss = self.crit(pred, label)

        # metrics calculation
        metrics = self.metrics_cal(embed)
        
        # log validation loss and metrics
        self.log(f'val_loss', loss.item(), batch_size=1)
        self.log_dict(metrics, batch_size=1)        
    
    def metrics_cal(self, pred):
        # metrics calculation
        # kmeans = KMeans(n_clusters=1000, mode='euclidean', verbose=1)
        # label = kmeans.fit_predict(pred).cpu()
        # label = dbscan(pred, eps=1, min_samples=20).cpu()
        
        # label, _ = DBSCAN(pred.cpu(), eps=0.0001, min_samples=10)
        pred = pred.cpu()

        import numpy as np
        print(np.unique(_).shape)

        metrics = {}
        metrics['silhouette_coefficient'] = silhouette().score(pred, label)
        print(1)
        metrics['calinski_harabasz_index'] = CalinskiHarabaszScore()(pred, label)
        print(2)
        metrics['dunn_index'] = DunnIndex()(pred, label)
        print(3)
        metrics['davies-bouldin_index'] = DaviesBouldinScore()(pred, label)
        print(4)

        return metrics
        

class GraphSAGE(BaseCDModel):
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


class GAT(BaseCDModel):
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
        
        
class GATv2(BaseCDModel):
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
        

class HGT(BaseCDModel):
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
            HGTConv(hidden_channels[-1], out_channels, metadata=metadata, heads=num_heads)
        )        
        
        
        self.set_optimizer(optim)
        self.set_criteria(crit)
        
    def encode(self, x_dict, edge_index_dict):
        for i in range(self.num_inner_layers):
            x_dict = self.encoder[i*2](x_dict, edge_index_dict)
            x_dict = {
                node_type: self.encoder[i*2+1](x)
                for node_type, x in x_dict.items()
            }
        return self.encoder[-1](x_dict, edge_index_dict)
            