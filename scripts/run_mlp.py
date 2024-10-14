# NOTE: Your script is not in the root directory. We must hence change the system path
import os
import sys
DIR = "/Users/margotgeerts/OneDrive - KU Leuven/GNNs4HPP"
os.chdir(DIR)
sys.path.append(DIR)
import argparse
from src.data.benchmark_dataset import MyDataset
from src.methods.utils import *
from src.methods.model import MLP
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='kc')
args = parser.parse_args()

data_name = args.data_name

generator = 'gabriel'

seed_everything()


save_dir = f'results/{data_name}_mlp/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

num_targets = 1


epochs = 1000


dataset = MyDataset(data = data_name, edge_attr=False, 
edge_generator=generator,  edge_kwargs = None, 
withxy=True, time_split=True)
data = dataset.data
X_train, y_train = data.x[data.train_mask], data.y[data.train_mask]
X_val, y_val = data.x[data.val_mask], data.y[data.val_mask]
X_test, y_test = data.x[data.test_mask], data.y[data.test_mask]

dataset = MyDataset(data = data_name, edge_attr=False, 
edge_generator=generator,  edge_kwargs = None, 
withxy=False, time_split=True)
data = dataset.data
X_hedonic_train = data.x[data.train_mask]
X_hedonic_val = data.x[data.val_mask]
X_hedonic_test = data.x[data.test_mask]

X_xy_train = X_train[:, -2:]
X_xy_val = X_val[:, -2:]
X_xy_test = X_test[:, -2:]

train_data = TabularDataset(X_train, y_train)
val_data = TabularDataset(X_val, y_val)
test_data = TabularDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)


def objective(trial):
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 5)
    model = MLP(in_channels=X_train.shape[1], 
                out_channels=num_targets, 
                hidden_channels=hidden_channels, 
                num_layers=num_layers).to(device)
    opt = trial.suggest_categorical("optimizer", ['adam', 'sgd'])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    if opt == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
      optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    batch = next(iter(train_loader))
    model(batch[0].to(device))

    early_stopper = EarlyStopper(patience=250, min_delta=0.)

    
    save_best_model = SaveBestModel(dir=save_dir)

    pbar = trange(1, epochs+1)
    for epoch in pbar:
        total_loss = total_examples = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(batch_x).view(-1)
            loss = model.loss(out, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            total_examples += batch_x.size(0)
            loss.backward()
            optimizer.step()
        train_loss= float(total_loss) / total_examples

        val_loss = evaluate_mlp(model, loader = val_loader)['loss']
        
        lr_scheduler.step(val_loss)
        pbar.set_postfix(train_loss=np.round(train_loss,2), val_loss=np.round(val_loss,3))
        
        save_best_model(current_valid_loss = val_loss,
            epoch=epoch,
            model = model,
            optimizer = optimizer,
            criterion = model.loss,
            trial_number = trial.number)
        if epoch %100 == 0:
            print(f'Epoch: {epoch:03d}')
            print(f'train_loss: {train_loss:.4f} - val_loss {val_loss:.4f}')

        if epoch == epochs or early_stopper.early_stop(val_loss):
            trial.set_user_attr('train_loss',train_loss)
            trial.set_user_attr('val_loss',val_loss)
            best_val_loss = save_best_model.best_valid_loss
            trial.set_user_attr('best_val_loss', best_val_loss)
            np.savez(f'{save_dir}/res_trial_{trial.number}.npz', train_loss=train_loss,
            val_loss=val_loss, best_val_loss=best_val_loss,
            hidden_channels=hidden_channels, lr=lr)

            torch.save({
                'epoch': epoch,
                'val_loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': model.loss,
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
study.optimize(objective, n_trials=50)
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
num_layers = trial.params['num_layers']


model = MLP(in_channels=X_train.shape[1],
            out_channels=num_targets,
            hidden_channels=hidden_channels,
            num_layers=num_layers).to(device)

model.load_state_dict(torch.load(f'{save_dir}/best_model_trial_{trial.number}.pth')['model_state_dict'])
model.eval()

train_metrics = evaluate_mlp(model, loader=train_loader)
val_metrics = evaluate_mlp(model, loader=val_loader)
test_metrics = evaluate_mlp(model, loader=test_loader)

print("Best model")
print(f'Train: {" - ".join([k+" "+str(round(v,4)) for k, v in train_metrics.items()])}')
print(f'Val: {" - ".join([k+" "+str(round(v,4)) for k, v in val_metrics.items()])}')
print(f'Test: {" - ".join([k+" "+str(round(v,4)) for k, v in test_metrics.items()])}')

# save metrics
metrics = {**{f'train_{k}':v for k,v in train_metrics.items()}, 
           **{f'val_{k}':v for k,v in val_metrics.items()}, 
           **{f'test_{k}':v for k,v in test_metrics.items()}}
pd.DataFrame(metrics, index=[0]).to_csv(f'{save_dir}/best_metrics_trial{trial.number}.csv')




 


