# *Graph Neural Networks for House Price Prediction: Do or Don't?* </br><sub><sub>*Margot Geerts, Seppe vanden Broucke, Jochen De Weerdt* [[*Outlet Year*]](*url to paper*)</sub></sub>
*Write a short summary of your paper*

## Repository structure
This repository is organised as follows:
```bash
|- config/
|- data/
|- notebooks/
|- results/
|- scripts/
|- src/
```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
The [`pyg-lib`](https://github.com/pyg-team/pyg-lib) package should be installed additionally, depending on the Torch and CUDA version. Please use the above in a newly created virtual environment to avoid clashing dependencies.

Alternatively, the `GNNs4HPP.yml` file can be used to create a conda environment:
```bash
conda env create -f GNNs4HPP.yml
```

## Usage
After everything is installed properly, we can replicate the experiments in the paper. To run the GNN experiments with the King County dataset, kNN graph and GraphSAGE model:
```bash
python scripts/main.py --data_name kc --generator knn --gnn_model sage --loader TRUE
```

To run the tree-based baselines on the King County dataset:
```bash
python scripts/run_baselines.py --data_name kc
```



## Citing
Please cite our paper and/or code as follows:
*Use the BibTeX citation*

```tex

@article{}

```