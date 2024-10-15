# NOTE: Your script is not in the root directory. We must hence change the system path
import os
import sys
DIR = "C:/Users/folder/subfolder/GNNs4HPP"
os.chdir(DIR)
sys.path.append(DIR)
import argparse
import optuna
from optuna.trial import TrialState
from src.methods.model import GNN
from src.data.benchmark_dataset import MyDataset
from src.methods.utils import *
import copy
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch
import torch_geometric
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.loader import NeighborLoader
import json

parser = argparse.ArgumentParser()
parser.add_argument('--generator', type=str, default='knn')
parser.add_argument('--generator_arg', nargs='+')
parser.add_argument('--features', type=str, default='full', choices=['full', 'hedonic'])
parser.add_argument('--gnn_model', type=str, default='sage', choices=['sage', 'gin', 'gat', 'transformer', 'dna'])
parser.add_argument('--data_name', type=str, default='kc')
parser.add_argument('--loader', type=bool, default=False)
parser.add_argument('--wandb_project', type=str, default=None)

args = parser.parse_args()
generator = args.generator
generator_arg = args.generator_arg
features = args.features
gnn_model = args.gnn_model
data_name = args.data_name
loader = args.loader
wandb_project = args.wandb_project


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network_data = json.load(open(f'config/data/network_data_{data_name}.json'))

if generator not in ['gabriel', 'relative_neighborhood', 'unconnected']:
  key_gen = list(network_data[generator].keys())[0]
  params = network_data[generator][key_gen]
  if generator_arg is not None:
    
    if generator in ['knn', 'kmeans']:
      generator_arg = [int(x) for x in generator_arg]
    elif generator in ['gaussian']:
      generator_arg = [float(x) for x in generator_arg]
    for p in generator_arg:
      assert p in params
  gen_arg = params if generator_arg is None else generator_arg
  
else:
  
  gen_arg=list([None])

print(f"Gen arg: {gen_arg}")
print(f'use loader: {loader}')

for gen_param in gen_arg:
  
  edge_kwargs = {key_gen:gen_param} if gen_param is not None else None
  

  def objective(trial):
    seed_everything()
    print(device, generator, gen_param)

    withxy = True if features == 'full' else False
    dataset = MyDataset(data = data_name, edge_attr=False, 
    edge_generator=generator,  edge_kwargs = edge_kwargs, withxy=withxy, time_split=True)
    
    print(dataset.num_features)
    num_targets = 1

    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64,128, 256])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    n1 = 25
    n2 = 10
    epochs = 1000
    dropout = trial.suggest_categorical("dropout",[0., 0.1,0.2, 0.3,0.4, 0.5])
    opt = trial.suggest_categorical("optimizer", ['adam', 'sgd'])
    print(generator, gen_param, hidden_dim, dropout, opt, lr, n1, n2)


    train_data = dataset.train_data
    val_data = dataset.val_data
    test_data = dataset.test_data

    train_loader= NeighborLoader(train_data, num_neighbors=[n1,n2], 
            batch_size=256, 
            input_nodes=train_data.train_mask) if loader else None
    val_loader=NeighborLoader(val_data, 
              num_neighbors=[n1,n2], 
              batch_size=256, 
              input_nodes=val_data.val_mask) if loader else None
    test_loader = NeighborLoader(test_data, 
          num_neighbors=[n1,n2], 
          batch_size=256, 
          input_nodes=test_data.test_mask) if loader else None

    model = GNN(dataset.num_features, hidden_dim, num_targets, dropout, conv=gnn_model)
    print(model)

    if wandb_project is not None:
      run = wandb.init(
        # set the wandb project where this run will be logged
        project=wandb,
        
        # track hyperparameters and run metadata
        config={
        "learning_rate":lr,
        "hidden_dim":hidden_dim,
        "n1":n1,
        "n2":n2,
        "dropout": dropout,
        "architecture": gnn_model,
        "dataset": data_name,
        "generator": generator,
        "gen_param": gen_param,
        "optimizer": opt,
        "split": 'time_based' if dataset.time_split else 'random',
        "epochs": epochs,
        "feature_set": features,
        "loader": 'neighborloader' if loader else 'none'
        }
    )
    # construct the optimizer based on the trial suggestion
    if opt == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
      optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)


    model = model.to(device)

    if train_loader is not None:
      batch = next(iter(train_loader))
      batch = batch.to(device)
      model(batch.x, batch.edge_index)
    else:
      x = train_data.x.to(device)
      edge_index = train_data.edge_index.to(device)
      model(x, edge_index)
    results = []

    early_stopper = EarlyStopper(patience=250, min_delta=0.)
    save_dir = f"results/{data_name}_{gnn_model}_{generator}_{gen_param}/"
    if not os.path.exists('results/'):
      os.mkdir('results/')
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    save_best_model = SaveBestModel(dir=save_dir)

    pbar = trange(1, epochs+1)
    for epoch in pbar:

        train_loss = train(model, optimizer, loader=train_loader) if train_loader is not None else \
          train(model, optimizer, data=train_data, mask=train_data.train_mask)

        val_loss = evaluate(model, loader = val_loader)['loss'] if train_loader is not None else \
          evaluate(model, data=val_data, mask=val_data.val_mask)['loss']
        lr_scheduler.step(val_loss)
        pbar.set_postfix(train_loss=np.round(train_loss,2), val_loss=np.round(val_loss,3))
        if wandb_project is not None:
          wandb.log({"train_loss":train_loss, "val_loss":val_loss, "lr": optimizer.param_groups[0]["lr"]})
          
        results.append([epoch, train_loss, val_loss])
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
            np.savez(f'{save_dir}res_trial_{trial.number}.npz', train_loss=train_loss,
            val_loss=val_loss, best_val_loss=best_val_loss,
            hidden_dim=hidden_dim, lr=lr, dropout=dropout,
            withxy=withxy, split = 'time_based' if dataset.time_split else 'random',
            n1=n1, n2=n2)

            torch.save({
                'epoch': epoch,
                'val_loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': model.loss,
                }, f'{save_dir}final_model_trial_{trial.number}.pth')
            
            if not np.isnan(val_loss):
              
              artifact = wandb.Artifact('best_model', type="model")
              artifact.add_file(f'{save_dir}best_model_trial_{trial.number}.pth')
              if wandb_project is not None:
                run.log_artifact(artifact)

              artifact = wandb.Artifact('final_model', type="model")
              artifact.add_file(f'{save_dir}final_model_trial_{trial.number}.pth')
              if wandb_project is not None:
                run.log_artifact(artifact)

            results = np.array(results)
            results = pd.DataFrame({'index':results[:,0],
            'train_loss':results[:,1],
            'val_loss': results[:,2]})
            results.to_csv(f'{save_dir}loss_trial_{trial.number}.csv')

            break
        trial.report(save_best_model.best_valid_loss, epoch)


        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            if wandb_project is not None:
              wandb.finish()
            raise optuna.exceptions.TrialPruned()

    print("  Final values: ")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))

    if not np.isnan(val_loss):
        loaders = {'train':{"loader":train_loader},
                  'val':{'loader':val_loader},
                  'test':{'loader':test_loader}} if loader else \
          {'train':{"data":train_data, "mask": train_data.train_mask},
          'val':{"data":val_data, "mask": val_data.val_mask},
          'test':{"data":test_data, "mask": test_data.test_mask}}

        model = GNN(num_features=dataset.num_features,
          num_targets=1,
          hidden_channels=hidden_dim, dropout=dropout,
          conv=gnn_model)

        evaluate_best_model(dir = save_dir, model=model, 
        trial_number=trial.number, loaders=loaders, 
        num_features=dataset.num_features,
        save_to_run=run if wandb_project is not None else None
        )

        results = pd.read_csv(f'{save_dir}loss_trial_{trial.number}.csv', index_col=0)

        results.plot(y=['train_loss', 'val_loss'])

        plt.savefig(f'{save_dir}loss_trial_{trial.number}.png')
        plt.close()

    if wandb_project is not None:
      wandb.finish()
    return save_best_model.best_valid_loss


    
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=10)
  pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
  complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

  print(f"Optuna study completed for Generator: {generator} - Parameter: {gen_param}")

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