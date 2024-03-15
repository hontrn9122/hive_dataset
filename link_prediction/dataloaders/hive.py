import torch
import json
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData


def getHiveDataset(
    node_file,
    user_link_edge_file,
    user_user_edge_file=None,
):
    assert user_user_edge_file is None, "Not support User-User relation yet!"
    # load dataset from csv file
    node_df = pd.read_csv(node_file).drop(['Value', 'Created'], axis=1)

    
    ul_edge_df = pd.read_csv(user_link_edge_file).drop(
        ['Timestamp', 'Frequency'], axis=1
    )
    ul_edge_df.Interaction_Strength = pd.to_numeric(
        ul_edge_df.Interaction_Strength
    ).fillna(0)

    
    #------------------------------- dataset processing ---------------------------#

    data = {}
    # split user nodes and post nodes from the node data
    u_node_df = node_df.loc[node_df.Type == "User"].drop(
        ['Type', 'Total vote'], axis=1
    ).fillna(0)
    
    l_node_df = node_df.loc[node_df.Type == "Link", ['ID', 'Total vote']]
    l_node_df['Total vote'] = pd.to_numeric(l_node_df['Total vote']).fillna(0)

    # split edge type
    uid = u_node_df.ID
    lid = l_node_df.ID
    # print(user_id.rename("Source"))

    # map node id to graph node id
    map_uid = {k:v for v, k in enumerate(uid.to_list())}
    map_lid = {k:v for v, k in enumerate(lid.to_list())}
    
    # user_map_id_reverse = {k:v for k, v in enumerate(user_id.to_list())}
    # link_map_id_reverse = {k:v for k, v in enumerate(link_id.to_list())}
    
    ul_edge_df = ul_edge_df.replace({"Source": map_uid, "Target": map_lid})      
    
    data['user'] = {
        'x': torch.tensor(u_node_df.drop('ID', axis=1).values).float(),
        'y': torch.tensor(
            pd.to_numeric(u_node_df['Abnormally'].replace({True:1, False:0})).fillna(0).values
        ).long()
    }
    data['link'] = {
        'x': torch.tensor(l_node_df.drop('ID', axis=1).values).float(),
        'y': torch.tensor(
            pd.to_numeric(l_node_df['Abnormally'].replace({True:1, False:0})).fillna(0).values
        ).long()
    }

    

    for type in ul_edge_df.Type.unique():
        edge_id = np.stack([
            ul_edge_df.loc[ul_edge_df.Type==type]['Source'].values,
            ul_edge_df.loc[ul_edge_df.Type==type]['Target'].values
        ], axis=0)
        
        edge_attr = np.stack([
            ul_edge_df.loc[ul_edge_df.Type==type] ['Interaction_Strength'].values,
            ul_edge_df.loc[ul_edge_df.Type==type] ['TimeDifference'].values
        ], axis=1)

        edge_label = pd.to_numeric(ul_edge_df.loc[ul_edge_df.Type==type]['Abnormally'].replace({True:1, False:0})).fillna(0).values

        
        data['user', type.lower(), 'link']= {
            'edge_index': torch.from_numpy(edge_id).long(),
            'edge_attr': torch.from_numpy(edge_attr).float(),
            'y': torch.from_numpy(edge_label).long()
        }

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

    
            
        
        