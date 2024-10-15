# NOTE: Your script is not in the root directory. We must hence change the system path
import os
import sys
DIR = "C:/Users/folder/subfolder/GNNs4HPP"
os.chdir(DIR)
sys.path.append(DIR)
import torch
import wandb
from src.data.benchmark_dataset import MyDataset
from src.methods.model import GNN
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import segregate_self_loops, add_remaining_self_loops
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, loader):
    model.eval()
    preds = []
    trues = []
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)[:batch_size].view(batch_size)
        preds.append(out.detach().cpu())
        trues.append(batch.y[:batch_size].detach().cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return preds, trues





dir = "results/kc_sage_knn_5/"
data_name = 'kc'
gnn_model = 'sage'
trial = 1
res = np.load(f"{dir}/res_trial_{trial}.npz", allow_pickle=True)
withxy = res['withxy'].item() if 'withxy' in res else True
time_split = res['split'].item() == 'time_based' if 'split' in res else True
n1 = res['n1'].item()
n2 = res['n2'].item()
hidden_dim = res['hidden_dim'].item()
dropout = res['dropout'].item()
generator = 'knn'
gen_param = 5
edge_kwargs = {'n_neighbors': gen_param}

dataset = MyDataset(data=data_name, edge_attr=False, 
edge_generator=generator,  edge_kwargs = edge_kwargs,
 withxy=withxy, time_split=time_split)

train_data = dataset.train_data
val_data = dataset.val_data

val_loader=NeighborLoader(val_data, 
            num_neighbors=[n1,n2], 
            batch_size=256, 
            input_nodes=val_data.val_mask)

print(f"Original node degree: {torch.bincount(val_data.edge_index[0]).float().mean(dtype=torch.float)}")

# Set all xy features to the mean to calculate feature importance of xy features
val_xy = copy.deepcopy(val_data)
x_mean = train_data.x[:, -2].mean()
y_mean = train_data.x[:, -1].mean()
val_xy.x[:, -2] = x_mean
val_xy.x[:, -1] = y_mean

# Set all hedonic features to the mean to calculate feature importance of hedonic features
val_hedonic = copy.deepcopy(val_data)
hedonic_mean = train_data.x[:, :-2].mean(dim=0)
val_hedonic.x[:, :-2] = hedonic_mean


val_loader_xy=NeighborLoader(val_xy,
            num_neighbors=[n1,n2], 
            batch_size=256, 
            input_nodes=val_xy.val_mask)

val_loader_hedonic=NeighborLoader(val_hedonic,
            num_neighbors=[n1,n2], 
            batch_size=256, 
            input_nodes=val_hedonic.val_mask)



model = GNN(dataset.num_features, hidden_dim, 1, dropout, conv=gnn_model)
print(model)

best_model_name = f"{dir}best_model_trial_{trial}.pth"
model.load_state_dict(torch.load(best_model_name, map_location=device)['model_state_dict'])

pred_orig, true_orig = evaluate(model, val_loader)
pred_xy, true_xy = evaluate(model, val_loader_xy)
pred_hedonic, true_hedonic = evaluate(model, val_loader_hedonic)

preds = pd.DataFrame({'original_pred':pred_orig.numpy(), 'original_true': true_orig.numpy(), 
'xy_pred':pred_xy.numpy(), 'xy_true': true_xy.numpy(),
'hedonic_pred':pred_hedonic.numpy(), 'hedonic_true': true_hedonic.numpy()})
preds.to_csv(f'{dir}preds_feat_import_trial_{trial}.csv')

print(f'Original MSE: {np.mean((pred_orig.numpy()-true_orig.numpy())**2)}')
# XY feature attribution / importance
print(f'xy MSE: {np.mean((pred_xy.numpy()-true_xy.numpy())**2)}')
print(f'xy feature attribution (MSDiff): {np.mean((pred_orig.numpy()-pred_xy.numpy())**2)}')
# Hedonic feature attribution / importance
print(f'hedonic MSE: {np.mean((pred_hedonic.numpy()-true_hedonic.numpy())**2)}')
print(f'hedonic feature attribution (MSDiff): {np.mean((pred_orig.numpy()-pred_hedonic.numpy())**2)}')

