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
import wandb
import torch
import torch_geometric
from torch_geometric.utils import subgraph
import lightgbm
import catboost
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import shap


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='kc')
args = parser.parse_args()

data_name = args.data_name

generator = 'gabriel'

seed_everything()

num_targets = 1


n1 = 25
n2 = 10
epochs = 1000


dataset = MyDataset(data = data_name, edge_attr=False, 
edge_generator=generator,  edge_kwargs = None, 
withxy=True, time_split=True)
data = dataset.data

X_train, y_train = data.x[data.train_mask].numpy(), data.y[data.train_mask].numpy()
X_val, y_val = data.x[data.val_mask].numpy(), data.y[data.val_mask].numpy()
X_test, y_test = data.x[data.test_mask].numpy(), data.y[data.test_mask].numpy()
print(X_train.shape)

del dataset, data
dataset = MyDataset(data = data_name, edge_attr=False, 
edge_generator=generator,  edge_kwargs = None, 
withxy=False, time_split=True)
data = dataset.data
X_hedonic_train = data.x[data.train_mask].numpy()
print(X_hedonic_train.shape)
X_hedonic_val = data.x[data.val_mask].numpy()
X_hedonic_test = data.x[data.test_mask].numpy()

X_xy_train = X_train[:, -2:]
X_xy_val = X_val[:, -2:]
X_xy_test = X_test[:, -2:]

lgbm_model = lightgbm.LGBMRegressor()
lgbm_model_hedonic = lightgbm.LGBMRegressor()
lgbm_model_xy = lightgbm.LGBMRegressor()

rf_model = RandomForestRegressor()
rf_model_hedonic = RandomForestRegressor()
rf_model_xy = RandomForestRegressor()

catboost_model = catboost.CatBoostRegressor(silent=True)
catboost_model_hedonic = catboost.CatBoostRegressor(silent=True)
catboost_model_xy = catboost.CatBoostRegressor(silent=True)

xgboost_model = xgboost.XGBRegressor()
xgboost_model_hedonic = xgboost.XGBRegressor()
xgboost_model_xy = xgboost.XGBRegressor()


lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
lgbm_model_hedonic.fit(X_hedonic_train, y_train, eval_set=[(X_hedonic_val, y_val)] )
lgbm_model_xy.fit(X_xy_train, y_train, eval_set=[(X_xy_val, y_val)])

rf_model.fit(X_train, y_train)
rf_model_hedonic.fit(X_hedonic_train, y_train)
rf_model_xy.fit(X_xy_train, y_train)

catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val))
catboost_model_hedonic.fit(X_hedonic_train, y_train, eval_set=(X_hedonic_val, y_val))
catboost_model_xy.fit(X_xy_train, y_train, eval_set=(X_xy_val, y_val))

xgboost_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgboost_model_hedonic.fit(X_hedonic_train, y_train, eval_set=[(X_hedonic_val, y_val)], verbose=False)
xgboost_model_xy.fit(X_xy_train, y_train, eval_set=[(X_xy_val, y_val)], verbose=False)

print(f"LGBM: train {mean_squared_error(y_train, lgbm_model.predict(X_train))}, val {mean_squared_error(y_val, lgbm_model.predict(X_val))}, test {mean_squared_error(y_test, lgbm_model.predict(X_test))}")
print(f"CatBoost: train {mean_squared_error(y_train, catboost_model.predict(X_train))}, val {mean_squared_error(y_val, catboost_model.predict(X_val))}, test {mean_squared_error(y_test, catboost_model.predict(X_test))}")
print(f"XGBoost: train {mean_squared_error(y_train, xgboost_model.predict(X_train))}, val {mean_squared_error(y_val, xgboost_model.predict(X_val))}, test {mean_squared_error(y_test, xgboost_model.predict(X_test))}")

m = [
    {'train_mse': mean_squared_error(y_train, lgbm_model.predict(X_train)), 
                            'val_mse': mean_squared_error(y_val, lgbm_model.predict(X_val)),
                            'test_mse': mean_squared_error(y_test, lgbm_model.predict(X_test)),
                            'model': 'lgbm', 'feature_set': 'full'},
{'train_mse': mean_squared_error(y_train, lgbm_model_hedonic.predict(X_hedonic_train)),
                            'val_mse': mean_squared_error(y_val, lgbm_model_hedonic.predict(X_hedonic_val)),
                            'test_mse': mean_squared_error(y_test, lgbm_model_hedonic.predict(X_hedonic_test)),    
                            'model': 'lgbm', 'feature_set': 'hedonic'},
{'train_mse': mean_squared_error(y_train, lgbm_model_xy.predict(X_xy_train)),
                            'val_mse': mean_squared_error(y_val, lgbm_model_xy.predict(X_xy_val)),  
                            'test_mse': mean_squared_error(y_test, lgbm_model_xy.predict(X_xy_test)),
                            'model': 'lgbm', 'feature_set': 'xy'},

{'train_mse': mean_squared_error(y_train, rf_model.predict(X_train)),
                            'val_mse': mean_squared_error(y_val, rf_model.predict(X_val)),
                            'test_mse': mean_squared_error(y_test, rf_model.predict(X_test)),
                            'model': 'rf', 'feature_set': 'full'},  
{'train_mse': mean_squared_error(y_train, rf_model_hedonic.predict(X_hedonic_train)),
                            'val_mse': mean_squared_error(y_val, rf_model_hedonic.predict(X_hedonic_val)),  
                            'test_mse': mean_squared_error(y_test, rf_model_hedonic.predict(X_hedonic_test)),
                            'model': 'rf', 'feature_set': 'hedonic'},
{'train_mse': mean_squared_error(y_train, rf_model_xy.predict(X_xy_train)),
                            'val_mse': mean_squared_error(y_val, rf_model_xy.predict(X_xy_val)),
                            'test_mse': mean_squared_error(y_test, rf_model_xy.predict(X_xy_test)),
                            'model': 'rf', 'feature_set': 'xy'},

{'train_mse': mean_squared_error(y_train, catboost_model.predict(X_train)),
                            'val_mse': mean_squared_error(y_val, catboost_model.predict(X_val)),
                            'test_mse': mean_squared_error(y_test, catboost_model.predict(X_test)),
                            'model': 'catboost', 'feature_set': 'full'},
{'train_mse': mean_squared_error(y_train, catboost_model_hedonic.predict(X_hedonic_train)),
                            'val_mse': mean_squared_error(y_val, catboost_model_hedonic.predict(X_hedonic_val)),
                            'test_mse': mean_squared_error(y_test, catboost_model_hedonic.predict(X_hedonic_test)),
                            'model': 'catboost', 'feature_set': 'hedonic'},

{'train_mse': mean_squared_error(y_train, catboost_model_xy.predict(X_xy_train)),
                            'val_mse': mean_squared_error(y_val, catboost_model_xy.predict(X_xy_val)),
                            'test_mse': mean_squared_error(y_test, catboost_model_xy.predict(X_xy_test)),
                            'model': 'catboost', 'feature_set': 'xy'},
                            
{'train_mse': mean_squared_error(y_train, xgboost_model.predict(X_train)),
                            'val_mse': mean_squared_error(y_val, xgboost_model.predict(X_val)),
                            'test_mse': mean_squared_error(y_test, xgboost_model.predict(X_test)),
                            'model': 'xgboost', 'feature_set': 'full'},
{'train_mse': mean_squared_error(y_train, xgboost_model_hedonic.predict(X_hedonic_train)),
                            'val_mse': mean_squared_error(y_val, xgboost_model_hedonic.predict(X_hedonic_val)),
                            'test_mse': mean_squared_error(y_test, xgboost_model_hedonic.predict(X_hedonic_test)),
                            'model': 'xgboost', 'feature_set': 'hedonic'},
{'train_mse': mean_squared_error(y_train, xgboost_model_xy.predict(X_xy_train)),
                            'val_mse': mean_squared_error(y_val, xgboost_model_xy.predict(X_xy_val)),
                            'test_mse': mean_squared_error(y_test, xgboost_model_xy.predict(X_xy_test)),
                            'model': 'xgboost', 'feature_set': 'xy'}]



metrics = pd.read_csv(f'results/tree_baselines_{data_name}.csv', index_col=0)
metrics = pd.concat([metrics, pd.DataFrame(m)], ignore_index=True)
metrics.to_csv(f'results/tree_baselines_{data_name}.csv')

tree_explainer1 = shap.TreeExplainer(lgbm_model)
tree_explainer2 = shap.TreeExplainer(rf_model)
tree_explainer3 = shap.TreeExplainer(xgboost_model)

print("Shap values LGBM...")
shap_values1 = tree_explainer1.shap_values(X_val)
print("Shap values RF...")
shap_values2 = tree_explainer2.shap_values(X_val)
print("Shap values CatBoost...")
shap_values3 = catboost_model.get_feature_importance(data=catboost.Pool(X_val, label=y_val), type='ShapValues')
print("Shap values XGBoost..."  )
shap_values4 = tree_explainer3.shap_values(X_val)

np.save(f'results/shap_values_lgbm_{data_name}', shap_values1)
np.save(f'results/shap_values_rf_{data_name}', shap_values2)
np.save(f'results/shap_values_catboost_{data_name}', shap_values3)
np.save(f'results/shap_values_xgboost_{data_name}', shap_values4)

joblib.dump(lgbm_model, f'results/lgbm_model_full_{data_name}.h5')
joblib.dump(rf_model, f'results/rf_model_full_{data_name}.h5')
joblib.dump(catboost_model, f'results/catboost_model_full_{data_name}.h5')
joblib.dump(xgboost_model, f'results/xgboost_model_full_{data_name}.h5')