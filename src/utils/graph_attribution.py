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


def get_random_edges(edge_index, n):
    edge_index = edge_index.cpu().numpy()

    edge_index[1] = np.random.choice(edge_index.shape[1], edge_index.shape[1], replace=True)
    edge_index = torch.tensor(edge_index)

    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=torch.bincount(edge_index[0]).shape[0])
    return edge_index


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

val_unconnected = copy.deepcopy(val_data)
org_index, _, loop_index, _ = segregate_self_loops(val_unconnected.edge_index)
val_unconnected.edge_index = loop_index
print(f"Unconnected node degree: {torch.bincount(val_unconnected.edge_index[0]).float().mean(dtype=torch.float)}")

val_random = copy.deepcopy(val_data)
val_random.edge_index = get_random_edges(org_index, n=torch.bincount(val_random.edge_index[0]).max().item())
print(f"Random node degree: {torch.bincount(val_random.edge_index[0]).float().mean(dtype=torch.float)}")

val_loader_unconnected=NeighborLoader(val_unconnected,
            num_neighbors=[n1,n2], 
            batch_size=256, 
            input_nodes=val_unconnected.val_mask)

val_loader_random=NeighborLoader(val_random,
            num_neighbors=[n1,n2], 
            batch_size=256, 
            input_nodes=val_random.val_mask)


model = GNN(dataset.num_features, hidden_dim, 1, dropout, conv=gnn_model)
print(model)

best_model_name = f"{dir}best_model_trial_{trial}.pth"
model.load_state_dict(torch.load(best_model_name, map_location=device)['model_state_dict'])

pred_orig, true_orig = evaluate(model, val_loader)
pred_unconnected, true_unconnected = evaluate(model, val_loader_unconnected)
pred_random, true_random = evaluate(model, val_loader_random)

preds = pd.DataFrame({'original_pred':pred_orig.numpy(), 'original_true': true_orig.numpy(), 
'unconnected_pred':pred_unconnected.numpy(), 'unconnected_true': true_unconnected.numpy(),
'random_pred':pred_random.numpy(), 'random_true': true_random.numpy()})
preds.to_csv(f'{dir}preds_graph_import_trial_{trial}.csv')

print(f'Original MSE: {np.mean((pred_orig.numpy()-true_orig.numpy())**2)}')
print(f'Unconnected MSE: {np.mean((pred_unconnected.numpy()-true_unconnected.numpy())**2)}')
# Graph attribution wrt unconnected grap
print(f'Unconnected graph attribution: {np.mean((pred_orig.numpy()-pred_unconnected.numpy())**2)}')
print(f'Random MSE: {np.mean((pred_random.numpy()-true_random.numpy())**2)}')
# Graph attribution wrt random graph
print(f'Random graph attribution: {np.mean((pred_orig.numpy()-pred_random.numpy())**2)}')

