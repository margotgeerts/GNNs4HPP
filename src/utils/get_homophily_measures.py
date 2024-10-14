import torch
import torch_geometric
from src.data.benchmark_dataset import MyDataset
import argparse
import os
import json
import numpy as np
import pandas as pd
import scipy

def calculate_target_assortativity(edge_index, y):
    # Get source and target attribute values
    source_attr = y[edge_index[0]]
    target_attr = y[edge_index[1]]

    # Compute the mean attribute value
    mean_attr = torch.mean(y).item()

    # Compute the numerator and denominator for the assortativity coefficient
    numerator = torch.sum((source_attr - mean_attr) * (target_attr - mean_attr)).item()
    denominator = torch.sum((source_attr - mean_attr)**2).item() * torch.sum((target_attr - mean_attr)**2).item()

    # Calculate the assortativity coefficient
    r = numerator / np.sqrt(denominator)
    return r

def calculate_homophily(edge_index, y):
    return torch_geometric.utils.assortativity(edge_index), calculate_target_assortativity(edge_index, y)

def get_homophily_measures(dataset_name):
    try:
        network_data = json.load(open(f'config/data/network_data_{dataset_name}.json'))
    except:
        print(f"Network data for dataset {dataset_name} not found.")
    
    df = pd.DataFrame(columns=['network', 'arg', 'avg_degree', 'connected_components', 'degree_assortativity', 'target_assortativity'])

    for edge_gen, args in network_data.items():
        if edge_gen == 'random':
            continue
        print(f"Edge generation: {edge_gen}")
        print(f"Arguments: {args}")

        if args == "":
            dataset = MyDataset(data=dataset_name,
                                withxy=True, edge_attr=False, time_split=True,
                                edge_generator=edge_gen, edge_kwargs=None)
            deg_assort, target_assort = calculate_homophily(data.edge_index, data.y)
            avg_degree = torch.mean(torch.bincount(data.edge_index[0]), dtype=torch.float).item()
            connected_components = scipy.sparse.csgraph.connected_components(torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index), directed=False, return_labels=False)
            df.loc[len(df)] = [edge_gen, None, avg_degree, connected_components, deg_assort, target_assort]
        else:
            for arg, params in args.items():
                for p in params:
                    dataset = MyDataset(data=dataset_name,
                                withxy=True, edge_attr=False, time_split=True,
                                edge_generator=edge_gen, edge_kwargs={arg:p})
                    deg_assort, target_assort = calculate_homophily(data.edge_index, data.y)
                    avg_degree = torch.mean(torch.bincount(data.edge_index[0]), dtype=torch.float).item()
                    connected_components = scipy.sparse.csgraph.connected_components(torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index), directed=False, return_labels=False)
                    df.loc[len(df)] = [edge_gen, p, avg_degree, connected_components, deg_assort, target_assort]
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    pargs = parser.parse_args()

    df = get_homophily_measures(pargs.dataset)
    df.to_csv(f'results/homophily_measures_{pargs.dataset}.csv', index=False)