import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric import utils
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric import utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import copy
import os
import json
from src.data.process_kc import *
from src.data.edge_generation import *

def transform_data(data, scaler=None, imputer=None, time_split=False):
    print('Transforming data...')

    if data.edge_attr:
        data.edge_attr = data.edge_attr.view(-1, 1)
        scaler2 = MinMaxScaler()
        data.edge_attr = torch.tensor(scaler2.fit_transform(data.edge_attr), dtype=torch.float)
        data.edge_attr = data.edge_attr.view(data.edge_attr.shape[0])
        transform = T.Compose([
            T.AddRemainingSelfLoops(attr="edge_attr", fill_value=1),
            T.ToUndirected(reduce="mean")
            ])
        data = transform(data)
    else:
        transform = T.Compose([
            T.AddRemainingSelfLoops(),
            T.ToUndirected(reduce="mean")            
            ])
        data = transform(data)

    test_ratio = 0.2
    num_nodes = data.x.shape[0]
    num_test = int(num_nodes * test_ratio)
    num_train = num_nodes - (2*num_test)
    num_val = num_nodes - num_test
    if hasattr(data, 'date'):
        # sort idx by date
        idx = np.argsort(data.date.numpy())
    else:
        idx = [i for i in range(num_nodes)]

    # if data does not contain train test masks:
    if not hasattr(data, 'train_mask'):
        if not time_split:
            np.random.seed(42)
            np.random.shuffle(idx)
        train_mask = torch.full_like(data.y, False, dtype=bool)
        train_mask[idx[:num_train]] = True
        val_mask = torch.full_like(data.y, False, dtype=bool)
        val_mask[idx[num_train:num_val]] = True
        test_mask = torch.full_like(data.y, False, dtype=bool)
        test_mask[idx[num_val:]] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    scaler = StandardScaler() if scaler is None else scaler
    data.x[data.train_mask] = torch.tensor(scaler.fit_transform(data.x[data.train_mask]), dtype=torch.float)
    data.x[data.val_mask] = torch.tensor(scaler.transform(data.x[data.val_mask]), dtype=torch.float)
    data.x[data.test_mask] = torch.tensor(scaler.transform(data.x[data.test_mask]), dtype=torch.float)

    

    if data.x.isnan().any():
        imputer = SimpleImputer(strategy="median") if imputer is None else imputer
        data.x[data.train_mask] = torch.tensor(imputer.fit_transform(data.x[data.train_mask]), dtype=torch.float)
        data.x[data.val_mask] = torch.tensor(imputer.transform(data.x[data.val_mask]), dtype=torch.float)
        data.x[data.test_mask] = torch.tensor(imputer.transform(data.x[data.test_mask]), dtype=torch.float)
    
    return data



class MyDataset(InMemoryDataset):
    def __init__(self, data='kc', pre_transform=transform_data, 
                    withxy=True, withdate=False, edge_attr=False, 
                    edge_generator=None, edge_kwargs=None, 
                    time_split=True):
        self.data_name = data
        # json file should contain variables
        vars = json.load(open(f'config/data/{self.data_name}.json'))
        self.features = vars['hedonic_vars']
        self.withxy = withxy
        self.withdate = withdate
        self.edge_attr = edge_attr
        self.edge_generator = edge_generator
        self.edge_kwargs = edge_kwargs
        self.root = 'data/'
        self.time_split = time_split
        #print(self.processed_file_names)
        super().__init__(root=self.root, transform=None, pre_transform=pre_transform)
        self.data = torch.load(self.processed_paths[0])
        print(self.processed_paths[0])
        print(self.data)
        
        
    @property
    def raw_file_names(self):
        return [self.file]

    # @property
    # def processed_paths(self):
    #     return [self.root+'/'+self.processed_file_names[0]]

    @property
    def processed_file_names(self):
        edge_kwargs_str = '_'.join([str(k)+str(v) for k,v in self.edge_kwargs.items()]) if self.edge_kwargs is not None else ""
        file_name = f"{self.data_name}"
        file_name += f"_{self.edge_generator}" if self.edge_generator is not None else ""
        file_name += f"_{edge_kwargs_str}" if self.edge_kwargs is not None else ""
        file_name += ".pt"
        return [file_name]
    

    def process(self):
        if os.path.exists(f'data/processed/{self.data_name}.csv'):
            df = pd.read_csv(f'data/processed/{self.data_name}.csv')
        else:
            df = process_kc(input_file=f'data/raw/kc_final.csv',
                            output_file=f'data/processed/{self.data_name}.csv',
                            remove_duplicates=True, remove_missing=True)

        # processed files should have the following columns:
        # x_geo, y_geo, transaction_date, log_price, set
        # other features are from the json file in config/data/data_name.json
        pos = torch.tensor(df[['x_geo','y_geo']].values, dtype=torch.float)
        y = torch.tensor(df['log_price'].values, dtype=torch.float)
        x = torch.tensor(df[self.features].values, dtype=torch.float)
        date = pd.to_numeric(pd.to_datetime(df['transaction_date'])) 
        train_test = df['set']

        if date is not None and self.withdate:
            x = torch.concat((x, torch.tensor(date.values.reshape(-1,1), dtype=torch.float)), axis=1)
        
        if self.withxy:
            x = torch.concat((x,pos), axis=1)
        
        print(f'Generating {self.edge_generator} edges...')
        edge_index = generate_edges(pos, self.edge_generator, self.edge_kwargs)
        self.data = Data(x=x,
                            y = y,
                            pos = pos, 
                            date = torch.tensor(date.values.reshape(-1,1), dtype=torch.float) if date is not None else None,
                            edge_index=edge_index)

        if train_test is not None:
            self.data.train_mask = torch.tensor(train_test == 'train', dtype=torch.bool)
            self.data.val_mask = torch.tensor(train_test == 'val', dtype=torch.bool)
            self.data.test_mask = torch.tensor(train_test == 'test', dtype=torch.bool)
        
        if self.pre_transform is not None:
            self.data = self.pre_transform(self.data, time_split=self.time_split)
        
        torch.save(self.data, self.processed_paths[0])
    
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data

    
    @property
    def num_features(self):
        return self.data.num_features
    
    @property
    def train_data(self):
        train_data = copy.deepcopy(self.data)
        if self.edge_attr:
            train_data.edge_index, train_data.edge_attr = utils.subgraph(self.data.train_mask, self.data.edge_index, self.data.edge_attr)
        else:
            train_data.edge_index, _ = utils.subgraph(self.data.train_mask, self.data.edge_index)
        train_data.edge_index = utils.to_undirected(train_data.edge_index)
        return train_data

    @property
    def val_data(self):
        val_data = copy.deepcopy(self.data)
        if self.edge_attr:
            val_data.edge_index, val_data.edge_attr = utils.subgraph(self.data.train_mask+self.data.val_mask, self.data.edge_index, self.data.edge_attr)
        else:
            val_data.edge_index, _ = utils.subgraph(self.data.train_mask+self.data.val_mask, self.data.edge_index)
        val_data.edge_index = utils.to_undirected(val_data.edge_index)
        return val_data

    @property
    def test_data(self):
        test_data = copy.deepcopy(self.data)
        if self.edge_attr:
            test_data.edge_index, test_data.edge_attr = utils.subgraph(self.data.train_mask+self.data.val_mask+self.data.test_mask, self.data.edge_index, self.data.edge_attr)
        else:
            test_data.edge_index, _ = utils.subgraph(self.data.train_mask+self.data.val_mask+self.data.test_mask, self.data.edge_index)
        test_data.edge_index = utils.to_undirected(test_data.edge_index)
        return test_data
    

