# NOTE: Your script is not in the root directory. We must hence change the system path
import os
import sys
DIR = "/Users/margotgeerts/OneDrive - KU Leuven/GNNs4HPP"
os.chdir(DIR)
sys.path.append(DIR)
import argparse
from src.data.benchmark_dataset import MyDataset
from src.methods.utils import *
import copy
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.utils import subgraph
from sklearn.metrics import mean_squared_error
import optuna
from optuna.trial import TrialState
from src.methods.pegnn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='kc')
args = parser.parse_args()

data_name = args.data_name

generator = 'gabriel'

seed_everything()


save_dir = f'results/{data_name}_pegnn/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

num_targets = 1


epochs = 1000


dataset = MyDataset(data = data_name, edge_attr=False, 
edge_generator=generator,  edge_kwargs = None, 
withxy=True, time_split=True)
data = dataset.data
y_train = data.y[data.train_mask]
y_val =data.y[data.val_mask]
y_test = data.y[data.test_mask]


X_hedonic_train = data.x[data.train_mask, :-2]
X_hedonic_val = data.x[data.val_mask, :-2]
X_hedonic_test = data.x[data.test_mask, :-2]

c_train = data.x[data.train_mask, -2:]
c_val = data.x[data.val_mask, -2:]
c_test = data.x[data.test_mask, -2:]

train_data = GeoDataset(X_hedonic_train, y_train, c_train)
val_data = GeoDataset(X_hedonic_val, y_val, c_val)
test_data = GeoDataset(X_hedonic_test, y_test, c_test)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
score1 = nn.MSELoss()

def objective(trial):
    train_edge_index = False
    train_edge_weight = False
    val_edge_index = False
    val_edge_weight = False
    train_y_moran = False

    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
    model = PEGCN(num_features_in=X_hedonic_train.shape[1],k=5,MAT=True,emb_dim=hidden_channels).to(device)
    
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    loss_wrapper = LossWrapper(model, task_num=2, loss='mse', uw=False, lamb=0.25, k=5, batch_size=256).to(device)
    optimizer = torch.optim.Adam(loss_wrapper.parameters(), lr=lr)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    batch = next(iter(train_loader))
    loss_wrapper(batch[0].to(device).float(), batch[1].to(device).float(), batch[2].to(device).float(), None, None, None)

    early_stopper = EarlyStopper(patience=250, min_delta=0.)

    
    save_best_model = SaveBestModel(dir=save_dir)
    print(model.device)

    pbar = trange(1, epochs+1)
    for epoch in pbar:
        train_loss = train_examples = 0
        for batch in train_loader:
            model.train()
            x = batch[0].to(device).float()
            y = batch[1].to(device).float()
            c = batch[2].to(device).float()
            optimizer.zero_grad()          
            loss = loss_wrapper(x, y, c, train_edge_index, train_edge_weight, train_y_moran)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_examples += x.size(0)
        
        model.eval()
        with torch.no_grad():
            val_score1 = evaluate_pegnn(model, val_loader, score1)['loss']
        pbar.set_postfix(train_loss=np.round(train_loss/train_examples,2), val_loss=np.round(val_score1,3))
        
        lr_scheduler.step(val_score1)
        
        save_best_model(current_valid_loss = val_score1,
            epoch=epoch,
            model = model,
            optimizer = optimizer,
            criterion = score1,
            trial_number = trial.number)
        if epoch %100 == 0:
            print(f'Epoch: {epoch:03d}')
            print(f'train_loss: {train_loss:.4f} - val_loss {val_score1:.4f}')

        if epoch == epochs or early_stopper.early_stop(val_score1):
            trial.set_user_attr('train_loss',train_loss)
            trial.set_user_attr('val_loss',val_score1)
            best_val_loss = save_best_model.best_valid_loss
            trial.set_user_attr('best_val_loss', best_val_loss)
            np.savez(f'{save_dir}/res_trial_{trial.number}.npz', train_loss=train_loss,
            val_loss=val_score1, best_val_loss=best_val_loss,
            hidden_channels=hidden_channels, lr=lr)

            torch.save({
                'epoch': epoch,
                'val_loss': val_score1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': score1,
                }, f'{save_dir}/final_model_trial_{trial.number}.pth')

            break

        trial.report(save_best_model.best_valid_loss, epoch)


    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

        print("  Final values: ")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
    
    return save_best_model.best_valid_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Number: ", trial.number)
print("  Final values: ")
for key, value in trial.user_attrs.items():
    print("    {}: {}".format(key, value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# retrains the best model
hidden_channels = trial.params['hidden_channels']


model = PEGCN(num_features_in=X_hedonic_train.shape[1],k=5,MAT=True,emb_dim=hidden_channels).to(device)

model.load_state_dict(torch.load(f'{save_dir}/best_model_trial_{trial.number}.pth')['model_state_dict'])
model.eval()

train_loader = DataLoader(train_data, batch_size=256, shuffle=False)
train_metrics = evaluate_pegnn(model, loader=train_loader, model_loss=score1)
val_metrics = evaluate_pegnn(model, loader=val_loader, model_loss=score1)
test_metrics = evaluate_pegnn(model, loader=test_loader, model_loss=score1)

print("Best model")
print(f'Train: {" - ".join([k+" "+str(round(v,4)) for k, v in train_metrics.items()])}')
print(f'Val: {" - ".join([k+" "+str(round(v,4)) for k, v in val_metrics.items()])}')
print(f'Test: {" - ".join([k+" "+str(round(v,4)) for k, v in test_metrics.items()])}')

# save metrics
metrics = {**{f'train_{k}':v for k,v in train_metrics.items()}, 
           **{f'val_{k}':v for k,v in val_metrics.items()}, 
           **{f'test_{k}':v for k,v in test_metrics.items()}}
pd.DataFrame(metrics, index=[0]).to_csv(f'{save_dir}/best_metrics_trial{trial.number}.csv')




 


