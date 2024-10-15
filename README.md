# *Graph Neural Networks for House Price Prediction: Do or Don't?* </br><sub><sub>*Margot Geerts, Seppe vanden Broucke, Jochen De Weerdt* [[*Outlet Year*]](*url to paper*)</sub></sub>
*Write a short summary of your paper*

## Repository structure
This repository is organised as follows:
```
GNNs4HPP
│
├── config/
│    └── data/
│         ├── kc.json: defines the variables for the King County dataset
│         └── network_data_kc.json: defines the graph parameter settings for the different graph construction methods; specific to the KC dataset
│
├── data/
│    ├── raw/
│    │    └── kc_final.csv the King County dataset (from https://www.kaggle.com/datasets/astronautelvis/kc-house-data)
│    └── processed/
│
├── results/
│
├── scripts/
│    ├── main.py: run this to replicate the GNN experiments
│    ├── run_baselines.py: runs the tree-based baseline experiments
│    ├── run_gp.py: runs the GPR baseline experiments
│    ├── run_mlp.py: runs the MLP baseline experiments
│    └── run_pegnn.py: runs the PE-GNN experiments
│
└── src/
     ├── data/
     │    ├── benchmark_dataset.py: defines the MyDataset class
     │    ├── edge_generation.py: defines the functions for graph construction
     │    └── process_kc.py: defines a processing function for the KC dataset
     ├── methods/
     │    ├── model.py: defines the GNN and MLP classes
     │    ├── pegnn.py: defines the classes and functions for PE-GNN
     │    └── utils.py: defines early stopping, training, and evaluation of models
     └── utils/
          ├── dataset_descriptions.py: generates descriptive statistics of all graphs
          ├── feature_attribution.py: calculates feature attribution of XY and hedonic features
          ├── graph_attribution.py: calculates graph attribution with respect to unconnected and random graphs
          └── homophily_measures.py: calculates homophily measures of graphs
```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
The [`pyg-lib`](https://github.com/pyg-team/pyg-lib) package should be installed additionally, depending on the Torch and CUDA version. This can be done as follows:
```bash
pip install pyg-lib==0.3.0 -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Usage
After everything is installed properly, we can replicate the experiments in the paper. Firstly, set the project directory in the scripts you want to run. For example, `DIR = 'C:\Users\folder\subfolder\GNNs4HPP'`

To run the GNN experiments with the King County dataset, kNN graph and GraphSAGE model:
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