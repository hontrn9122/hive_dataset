import torch
import json
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData


def hive_preprocessing(
    nodes_file,
    edges_file,
    out_dir = 'hive.pt'
):
    '''
    This function imports the dataset from csv file and preprocess it to make a Pytorch Geometric HeteroData object and save to given directory.
    Input:
        nodes_file: str, path of the node csv file
        edges_file: str, path of the edge csv file
        out_file: str, path of the preprocessed data file if out_file is not None, default: 'hive.pt'
    Return:
        torch_geometric.data.HeteroData, preprocessed HeteroData object
    '''
    # load dataset from csv file
    node_df = pd.read_csv(nodes_file).drop(['Value', 'Created'], axis=1)

    
    edge_df = pd.read_csv(edges_file).drop(
        ['Timestamp'], axis=1
    )

    
    #------------------------------- dataset processing ---------------------------#

    data, idx, idx_map = {}, {}, {}
    node_types = []
    
    #>>>>>>>>>>>>>>>>>>>>> Extract node features for each node types >>>>>>>>>>>>>>>>>>>>>>
    for t in node_df.Type.unique():
        if t == 'User':
            nodes = node_df.loc[node_df.Type == t].drop(['Type', 'Total vote'], axis=1)
        elif t in ['Post', 'Comment']:
            nodes = node_df.loc[node_df.Type == t, ['ID', 'Total vote', 'Abnormally']]
        else:
            print("[Warning] This node data has other types than 'User', 'Post' and 'Comment', which is curently not supported and will be ignored while processing!")
            continue
        t = t.lower()
        idx[t] = nodes.ID.to_list()
        idx_map[t] = {k:v for v, k in enumerate(idx[t])}
        data[t] = {
            'x': torch.tensor(nodes.drop(['ID', 'Abnormally'], axis=1).values).float(),
            'y': torch.tensor(pd.to_numeric(nodes['Abnormally'].replace({True:'1', False:'0'})).values)
        }
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Extract edges >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    for t in edge_df.Type.unique():
        edge_type = edge_df.loc[edge_df.Type == t].drop('Type', axis=1)
        if t == "Belong_to":
            source_target = (
                ('comment', 'comment'),
                ('comment', 'post')
            )
        else:
            source_target = (
                ('user', 'comment'),
                ('user', 'post'),
            )
        for source, target in source_target:
            edges = edge_type.loc[edge_type.Source.isin(idx[source]) & edge_type.Target.isin(idx[target])]
            edges = edges.replace({"Source": idx_map[source], "Target": idx_map[target]})
            edge_idx = np.stack([
                edges['Source'].values,
                edges['Target'].values
            ], axis=0)
            edge_attr = np.array(edges.Weight)
    
            edge_label = pd.to_numeric(edges['Abnormally'].replace({True:'1', False:'0'})).values


            data[source, t.lower(), target]= {
                'edge_index': torch.from_numpy(edge_idx).long(),
                'edge_attr': torch.from_numpy(edge_attr).float().view(-1,1),
                'y': torch.from_numpy(edge_label).long()
            }
    if out_file is not None:
        torch.save(data, out_file)
    
    return HeteroData.from_dict(data)



# def getLinkHiveDataset(
#     node_file,
#     user_link_edge_file,
#     user_user_edge_file=None,
# ):
    
#     nodes, edges = getHiveDataset(
#         node_file,
#         user_link_edge_file,
#         user_user_edge_file,
#     )
#     link_data_dict = {}
#     for edge in edges.keys():
#         data = nodes.copy()
#         data[edge] = edges[edge]
#         link_data_dict[edge] = HeteroData.from_dict(
#             data
#         )
#     return link_data_dict

    
            
        
        