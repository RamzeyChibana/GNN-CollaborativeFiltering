GNN CollaborativeFiltering
===
## Overview
This repository implements the Neural Graph Collaborative Filtering (NGCF) model based on the paper [Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108). NGCF uses graph neural networks (GNNs) to model user-item interactions as a graph and propagate embeddings to improve recommendations.

Our implementation reproduces the key components of NGCF, including embedding propagation and message passing, while extending it for performance comparison across datasets like MovieLens and Gorwala.

<!-- ## Papar Information
- Title:  `paper name`
- Authors:  `A`,`B`,`C`
- Preprint: [https://arxiv.org/abs/xx]()
- Full-preprint: [paper position]()
- Video: [video position]() -->



## Install & Dependence
- python 
- torch
- torch_geometric
- numpy
- pandas

## Datasets

| Dataset   | #Users | #Items | #Interactions  |
|-----------|-----------------|----------|--------|
| MovieLens |  6040  |  3883  | 1000209  |
| Gorwala   |  29858 | 40981     | 1027370    |



## Run Code
- for train
  ```sh
  python main.py --batch_size 1024 --layers 16 16  --dim 16 --ks 20, 40, 60 --exp exp_0
  ```
- for test
  ```
  python test.py
  ```
important arguments for **train.py** : 
* `--batch_size` batch size of nodes
* `--epochs` run the model of how many epochs
* `--learning_rate` learning rate 
* `--dim` Nodes embedding dimension
* `--layers` Num of layers and theire dimension
* `--dataset` train witch *dataset* { movielens / gorwala }
* `--ks` @K to test with (like NDCG@k / Hit@k .. )
* `--device` device to train with { cpu / gpu }
* `--verbose` How much more infos in epoch 0:nothing , 1:loss , 2:Meterices , 3:time



## Directory Hierarchy
```
|—— Data
|    |—— movies.csv
|    |—— ratings.csv
|    |—— users.csv
|
|—— utils
|    |—— datToCsv.py
|    |—— load_dataset.py
|    |—— metrices.py
|    |—— parser.py
|    |—— evaluate.py
|  
|—— main.py
|—— model.py
```
