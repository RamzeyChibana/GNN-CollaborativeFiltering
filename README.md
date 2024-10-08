GNN CollaborativeFiltering
===
## Overview
This repository implements the Neural Graph Collaborative Filtering (NGCF) model based on the paper [Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108). NGCF uses graph neural networks (GNNs) to model user-item interactions as a graph and propagate embeddings to improve recommendations.

Our implementation reproduces the key components of NGCF, including embedding propagation and message passing, while extending it for performance comparison across datasets like MovieLens and Gorwala.


## Results
#### on Gowalla 

|    |  Hit@k  | Eecall@k | Ndcg@k  |
|-----------|---------|----------|--------|
| Mine    |  0.703 | 0.256     | 0.278    |
| Paper   |  0.774 | 0.331     | 	0.324   |



## Datasets

| Dataset   | #Users | #Items | #Interactions  |
|-----------|-----------------|----------|--------|
| MovieLens |  6040  |  3883  | 1000209  |
| Gowalla    |  29858 | 40981     | 1027370    |


## Install & Dependence
- python 
- torch == 2.4.0+cu124
- numpy==1.26.4
- pandas == 1.4.3
- torch_geometric == 2.5.3




## Run Code
- ### for train
  ```sh
  python main.py --batch_size 1024 --layers 16 16  --dim 16 --ks 20, 40, 60 --exp exp_0
  ```
important arguments for **main.py** : 
* `--batch_size` batch size of nodes
* `--epochs` run the model of how many epochs
* `--learning_rate` learning rate 
* `--dim` Nodes embedding dimension
* `--layers` Num of layers and theire dimension
* `--dataset` train witch *dataset* { movielens / gorwala }
* `--ks` @K to test with (like NDCG@k / Hit@k .. )
* `--device` device to train with { cpu / gpu }
* `--verbose` How much more infos in epoch 0:nothing , 1:loss , 2:Meterices , 3:time
* `--exp` Continue the Training of existing Experiement


 - ### for test
    ```sh
    python test.py --batch_size 1024 --ks 20, 40, 60 --exp exp_0
    ```

important arguments for **test.py** : 
* `--batch_size` batch size of nodes
* `--ks` @K to test with (like NDCG@k / Hit@k .. )
* `--exp`  Experiement you want to Test
* `--device` device to test with { cpu / gpu }


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
|—— model.py
|—— main.py
|—— test.py
```
