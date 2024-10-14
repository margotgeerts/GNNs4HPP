from src.data.benchmark_dataset import MyDataset
import os
import torch
import torch_geometric
from torch_geometric.utils import to_undirected, subgraph, to_scipy_sparse_matrix
from torch_geometric import transforms as T
import json
from scipy.sparse.csgraph import connected_components
import pandas as pd
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='kc')
args=parser.parse_args()

data_name = args.data

# Load the dataset
network_data = json.load(open(f'config/data/network_data_{data_name}.json'))
print(networks)

dicts = []

def generate_descriptions(network, key, v, dicts):
    edge_kwargs = {key:v} if key is not None else None
    dataset = MyDataset(data=data_name,
                edge_attr=False, edge_generator=network,  
                edge_kwargs = edge_kwargs, withxy=True, time_split=True)

    dicts.append({'network':network, 'arg':v, 'is_undirected':data.is_undirected(), 'num_nodes':data.num_nodes, 'num_edges':data.num_edges, 'has_isolated_nodes':data.has_isolated_nodes(), 'has_self_loops':data.has_self_loops(), 'avg_degree':torch.bincount(data.edge_index[0]).mean(dtype=float).numpy(), 'connected_components':connected_components(to_scipy_sparse_matrix(data.edge_index), directed=False, return_labels=False)})
    return dicts


for network, arg in sorted(networks.items()):
    print(f'Generating description for {network, arg}...')
    if arg != "":
        for key in arg.keys():
            for v in arg[key]:
                dicts = generate_descriptions(network, key, v, dicts)
                
    else:
        dicts = generate_descriptions(network,None, None, dicts)

df=pd.DataFrame(dicts)    
print(df)
df.to_csv(f'results/network_descriptions_{data_name}.csv')