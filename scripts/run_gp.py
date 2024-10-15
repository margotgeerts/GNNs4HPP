# NOTE: Your script is not in the root directory. We must hence change the system path
import os
import sys
DIR = "C:/Users/folder/subfolder/GNNs4HPP"
os.chdir(DIR)
sys.path.append(DIR)
import argparse
from src.data.benchmark_dataset import MyDataset
from src.methods.utils import *
import copy
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric
from torch_geometric.utils import subgraph
from sklearn.metrics import mean_squared_error
import optuna
from optuna.trial import TrialState
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


"""
Based on the example from https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/index.html
"""
class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='kc')
args = parser.parse_args()

data_name = args.data_name

generator = 'gabriel'

seed_everything()


save_dir = f'results/{data_name}_gp/'
if not os.path.exists('results/'):
      os.mkdir('results/')
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


X_train = data.x[data.train_mask]
X_val = data.x[data.val_mask]
X_test = data.x[data.test_mask]


train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


def objective(trial):
    inducing_points = X_train[:500, :]
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    print(f"lr: {lr}")
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))
    mse_loss = torch.nn.MSELoss()
    early_stopper = EarlyStopper(patience=250, min_delta=0.)

    
    save_best_model = SaveBestModel(dir=save_dir)

    epochs_iter = tqdm.tqdm(range(epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        #minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        train_mse = train_examples = 0
        for x_batch, y_batch in train_loader:
            model.train()
            likelihood.train()
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            mse = mse_loss(output.mean, y_batch)
            train_mse += mse*y_batch.size(0)
            train_examples += y_batch.size(0)
            #minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
        train_mse /= train_examples
        model.eval()
        likelihood.eval()
        means = torch.tensor([0.])
        with torch.no_grad():
            preds = model(X_val)
            means = torch.cat([means, preds.mean.cpu()])
        means = means[1:]
        val_mse = torch.sqrt(torch.mean((means - y_val.cpu())**2))
        epochs_iter.set_postfix(loss=loss.item(), val_mse=val_mse.item())
        save_best_model(current_valid_loss = val_mse,
                epoch=i,
                model = model,
                optimizer = optimizer,
                criterion = mll,
                trial_number=trial.number)

    
        if i == epochs or early_stopper.early_stop(val_mse.detach().cpu().numpy()):
            trial.set_user_attr('train_loss',train_mse.detach().cpu().numpy())
            trial.set_user_attr('val_loss',val_mse.detach().cpu().numpy())
            best_val_loss = save_best_model.best_valid_loss
            trial.set_user_attr('best_val_loss', best_val_loss)
            np.savez(f'{save_dir}/res_trial_{trial.number}.npz', train_loss=train_mse.detach().cpu().numpy(),
            val_loss=val_mse.detach().cpu().numpy(), best_val_loss=best_val_loss,
            lr=lr)

            torch.save({
                'epoch': i,
                'val_loss': val_mse.detach().cpu().numpy(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mll,
                }, f'{save_dir}/final_model_trial_{trial.number}.pth')

            break

        trial.report(save_best_model.best_valid_loss, i)


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


inducing_points = X_train[:500, :]
model = GPModel(inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()


model.load_state_dict(torch.load(f'{save_dir}/best_model_trial_{trial.number}.pth')['model_state_dict'])
model.eval()
likelihood.eval()
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
means_train = torch.tensor([0.])
means_val = torch.tensor([0.])
means_test = torch.tensor([0.])
with torch.no_grad():
    for x_batch, y_batch in train_loader:
        preds = model(x_batch)
        means_train = torch.cat([means_train, preds.mean.cpu()])
    #preds_train = model(X_train)
    preds_val = model(X_val)
    preds_test = model(X_test)
    #means_train = torch.cat([means_train, preds_train.mean.cpu()])
    means_val = torch.cat([means_val, preds_val.mean.cpu()])
    means_test = torch.cat([means_test, preds_test.mean.cpu()])
means_train = means_train[1:]
means_val = means_val[1:]
means_test = means_test[1:]

train_rmse = torch.sqrt(torch.mean((means_train - y_train.cpu())**2))
val_rmse = torch.sqrt(torch.mean((means_val - y_val.cpu())**2))
test_rmse = torch.sqrt(torch.mean((means_test - y_test.cpu())**2))
print(f"Train RMSE: {train_rmse.item():.4f} - Val RMSE: {val_rmse.item():.4f} - Test RMSE: {test_rmse.item():.4f}")

mae = MeanAbsoluteError().to(device)
mape = MeanAbsolutePercentageError().to(device)
mse = MeanSquaredError().to(device)
r2 = R2Score().to(device)

train_metrics = {
    "mae": mae(means_train, y_train).item(), 
    "mape": mape(means_train, y_train).item(), 
    "mse": mse(means_train, y_train).item(), 
    "r2": r2(means_train, y_train).item()}

val_metrics = {
    "mae": mae(means_val, y_val).item(), 
    "mape": mape(means_val, y_val).item(), 
    "mse": mse(means_val, y_val).item(), 
    "r2": r2(means_val, y_val).item()}

test_metrics = {
    "mae": mae(means_test, y_test).item(), 
    "mape": mape(means_test, y_test).item(), 
    "mse": mse(means_test, y_test).item(), 
    "r2": r2(means_test, y_test).item()}

# save metrics
metrics = {**{f'train_{k}':v for k,v in train_metrics.items()}, 
           **{f'val_{k}':v for k,v in val_metrics.items()}, 
           **{f'test_{k}':v for k,v in test_metrics.items()}}
pd.DataFrame(metrics, index=[0]).to_csv(f'{save_dir}/best_metrics_trial{trial.number}.csv')




 


