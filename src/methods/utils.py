import torch
import numpy as np
import copy
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError, R2Score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.nan_counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.nan_counter = 0
        elif (validation_loss > (self.min_validation_loss + self.min_delta)):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif np.isnan(validation_loss):
          self.nan_counter += 1
          if self.nan_counter >= 35:
            return True
        return False

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf'), dir='outputs'
    ):
        self.best_valid_loss = best_valid_loss
        self.dir=dir

    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion, trial_number
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            # print(f"\nBest validation loss: {self.best_valid_loss}")
            # print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'val_loss': current_valid_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{self.dir}/best_model_trial_{trial_number}.pth')


def load_best_model(dir, trial_number):
    best_model = torch.load(f'{dir}/best_model_trial_{trial_number}.pth')
    return best_model

def evaluate_best_model(dir, model, trial_number, loaders,num_features, save_to_run=False):
    res = np.load(f'{dir}/res_trial_{trial_number}.npz')
    hidden_dim=res['hidden_dim'].astype(int).item()
    n1 = res['n1'].item()
    n2=res['n2'].item()
    dropout=res['dropout'].item()

    model2 = copy.deepcopy(model)


    best_model =torch.load(f'{dir}/best_model_trial_{trial_number}.pth')
    print(best_model['epoch'])
    model.load_state_dict(best_model['model_state_dict'])
    model.eval()
    model = model.to(device)

    final_model =torch.load(f'{dir}/final_model_trial_{trial_number}.pth')

    model2.load_state_dict(final_model['model_state_dict'])
    model2.eval()
    model2 = model2.to(device)


    best_train = evaluate(model, **loaders["train"])
    best_val = evaluate(model, **loaders["val"])
    best_test = evaluate(model, **loaders["test"])
    if save_to_run:
        # update keys of best train and merge with best val and best test:
        best_train = {'best_train_'+k:v for k,v in best_train.items()}
        best_val = {f'best_val_{k}':v for k,v in best_val.items()}
        best_test = {f'best_test_{k}':v for k,v in best_test.items()}
        d = {**best_train, **best_val, **best_test,
        'best_epoch':best_model['epoch']}

        save_to_run.summary.update(d)
    print("Best model")
    print(f'Train: {" - ".join([k+" "+str(round(v,4)) for k, v in best_train.items()])}')
    print(f'Val: {" - ".join([k+" "+str(round(v,4)) for k, v in best_val.items()])}')
    print(f'Test: {" - ".join([k+" "+str(round(v,4)) for k, v in best_test.items()])}')

    print("Final model")
    print(f'Train: {" - ".join([k+" "+str(round(v,4)) for k, v in evaluate(model2, **loaders["train"]).items()])}')
    print(f'Val: {" - ".join([k+" "+str(round(v,4)) for k, v in evaluate(model2,  **loaders["val"]).items()])}')
    print(f'Test: {" - ".join([k+" "+str(round(v,4)) for k, v in evaluate(model2, **loaders["test"]).items()])}')

def train(model, optimizer, loader=None, data=None, mask=None):
    assert (loader is None) or ((data is None) and (mask is None))
    model.train()
    optimizer.zero_grad()
    if loader:
        total_loss = total_examples = 0
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.batch_size
            out = model(batch.x, batch.edge_index)[:batch_size]
            out = out.view(out.size(0))
            loss = model.loss(out, batch.y[:batch_size])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
            total_examples += batch_size
        train_loss= float(total_loss) / total_examples
    else:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        out = model(x, edge_index).view(-1)
        loss = model.loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
    return train_loss

def evaluate(model, loader=None, data=None, mask=None):
    assert (loader is None) or ((data is None) and (mask is None))

    mae = MeanAbsoluteError().to(device)
    mape = MeanAbsolutePercentageError().to(device)
    mse = MeanSquaredError().to(device)
    r2 = R2Score().to(device)


    model.eval()
    if loader:
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
    else:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        preds = model(x, edge_index).view(-1)[mask].detach().cpu()
        trues = data.y[mask].detach().cpu()
    return {"loss": model.loss(preds, trues).item(), 
    "mae": mae(preds, trues).item(), 
    "mape": mape(preds, trues).item(), 
    "mse": mse(preds, trues).item(), 
    "r2": r2(preds, trues).item()}

def evaluate_mlp(model, loader=None):

    mae = MeanAbsoluteError().to(device)
    mape = MeanAbsolutePercentageError().to(device)
    mse = MeanSquaredError().to(device)
    r2 = R2Score().to(device)


    model.eval()
    
    preds = []
    trues = []
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        out = model(batch_x).view(-1)
        preds.append(out.detach().cpu())
        trues.append(batch_y.detach().cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return {"loss": model.loss(preds, trues).item(), 
    "mae": mae(preds, trues).item(), 
    "mape": mape(preds, trues).item(), 
    "mse": mse(preds, trues).item(), 
    "r2": r2(preds, trues).item()}

def evaluate_pegnn(model, loader=None, model_loss=torch.nn.MSELoss()):

    mae = MeanAbsoluteError().to(device)
    mape = MeanAbsolutePercentageError().to(device)
    mse = MeanSquaredError().to(device)
    r2 = R2Score().to(device)


    model.eval()
    
    preds = []
    trues = []
    for batch in loader:
        batch_x, batch_y, batch_c = batch
        batch_x, batch_y, batch_c = batch_x.to(device), batch_y.to(device), batch_c.to(device)
        out,_ = model(batch_x, batch_c, None, None).view(-1)
        preds.append(out.detach().cpu())
        trues.append(batch_y.detach().cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return {"loss": model_loss(preds, trues).item(), 
    "mae": mae(preds, trues).item(), 
    "mape": mape(preds, trues).item(), 
    "mse": mse(preds, trues).item(), 
    "r2": r2(preds, trues).item()}


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class GeoDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, coords):
      self.features = x
      self.target = y
      self.coords = coords
    def __len__(self):
      return len(self.features)
    def __getitem__(self, idx):
      return torch.tensor(self.features[idx]), torch.tensor(self.target[idx]), torch.tensor(self.coords[idx])       