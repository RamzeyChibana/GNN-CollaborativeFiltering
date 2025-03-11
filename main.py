from utils.load_dataset import MovieLens,Gorwala,deepGorwala
from model import NGCF
import torch
import numpy as np
from tqdm import tqdm 
from torchsummary import summary
from utils.evaluate import test
from time import time
from utils.parser import make_parser
import os
import csv
import json
import argparse
from torch.utils.data import DataLoader


def save_args(args, filename='args.json'):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(filename='args.json'):
    with open(filename, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)




parser = make_parser()
args = parser.parse_args()
exp = args.exp
exps = os.listdir("Experiments")

if args.device=="gpu":
    device = torch.device("cuda")
else :
    device = torch.device("cpu")
num_epochs = args.epochs
verbose = args.verbose



if exp in exps:
    file = exp
    print(f"\t\tContinue Training {file}...")
    new = False
    args = load_args(f"Experiments/{file}/args.json")
    checkpoint = torch.load(f"Experiments/{file}/last_checkpoint.pth")
    

    
else :
    
    file = f"exp_{len(exps)}"
    print(f"\t\tStart New Training {file}...")
    os.mkdir(os.path.join("Experiments",file))
    save_args(args,f"Experiments/{file}/args.json")
    new = True
    
    


'''
Hyper Parameters 
'''
epsilon = 0.01
batch_size = args.batch_size
learning_rate =args.learning_rate
h_dim = args.dim
layers = args.layers
dropout = args.dropout
Ks = args.ks
lamda = args.lamda


'''
Choose Dataset to make Data generator
'''

if args.dataset == "movielens":
    data_generator = MovieLens(batch_size=batch_size)
elif args.dataset == "gorwala" :
    data_generator = Gorwala("Data\gowalla",batch_size=batch_size)
else :
    raise TypeError("Invalid dataset ..")

# Load Graph 


'''
Make NGCF Model
'''


graph = data_generator.graph
graph = graph.to(device)

model = NGCF(graph,h_dim,layers,dropout,lamda)
model=model.to(device)
optimizer = torch.optim.Adam(model.parameters(),learning_rate)


if not new :
    
    model.load_state_dict(torch.load(f"Experiments/{file}/last_weights.pt"))
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    best_ep = checkpoint["best_ndcg@k"]
    file_csv = open(f'Experiments/{file}/history.csv', 'a', newline='')
    writer = csv.writer(file_csv)
   
else :
    epoch = 0
    best_ep = -np.inf
    file_csv = open(f'Experiments/{file}/history.csv', 'w', newline='')
    writer = csv.writer(file_csv)
    metric_columns = ["Epoch", "Train Loss"]+[f"Hit@{k}" for k in Ks ]+[f"Recall@{k}" for k in Ks]+[f"Ndcg@{k}" for k in Ks]+["epoch time","load Data time","forward time","embeds time","backward time"]
    writer = csv.writer(file_csv)
    writer.writerow(metric_columns)


'''
Training 
'''

n_batch = data_generator.n_train // batch_size + 1

for epoch in range(epoch+1, epoch+1+num_epochs):
    pbar = tqdm(total=data_generator.n_train)
    pbar.set_description(f"Epoch {epoch}:")
    loss, logloss, regloss = 0, 0, 0
    t_start = time()
    load_data_time = 0
    forward_time = 0
    backward_time = 0

    for batch in range(n_batch):
        t1 = time()
        
        users, pos_items, neg_items = data_generator.sample()
        users = torch.tensor(users, dtype=torch.long, device=device)
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device)
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device)
        t2 = time()
        load_data_time += t2 - t1

        optimizer.zero_grad()
        users_embds, items_embds = model()
        t3 = time()
        forward_time += t3 - t2

        user_emb = users_embds[users, :]
        pos_emb = items_embds[pos_items, :]
        neg_emb = items_embds[neg_items, :]

        batch_loss, batch_logloss, batch_regloss = model.BprLoss(user_emb, pos_emb, neg_emb)
        batch_loss.backward()
        optimizer.step()
        t4 = time()
        backward_time += t4 - t3

        loss += batch_loss.item()
        logloss += batch_logloss.item()
        regloss += batch_regloss.item()
        pbar.update(batch_size)
    pbar.close()
    t_end = time()
    t_epoch = t_end - t_start

    result, _ = test(model, data_generator, batch_size, Ks)

    if best_ep+ epsilon <= result["NDGC@k"][0]:
        best_ep = result["NDGC@k"][0]
        print(f"\t\tNew Peak at Ndcg@{Ks[0]} : {best_ep}")
        torch.save(model.state_dict(),f"Experiments/{file}/best_weights.pt")

    # Save Infos of last epoch
    checkpoint = {
        "epoch":epoch,
        "optimizer_state":optimizer.state_dict(),
        "loss":loss/n_batch,
        "best_ndcg@k":best_ep
    }
    
    torch.save(checkpoint,f"Experiments/{file}/last_checkpoint.pth")
    torch.save(model.state_dict(),f"Experiments/{file}/last_weights.pt")

    row_csv = [epoch,float(loss/n_batch)]+result["Hit@k"].tolist()+result["Recall@k"].tolist()+result["NDGC@k"].tolist()+[t_epoch,load_data_time,forward_time,backward_time]
    writer.writerow(row_csv)

    
    time_verbose = f"load Data time {load_data_time:.2f}s | {load_data_time/t_epoch:.2f}% of epoch \nforward time {forward_time:.2f}s|{forward_time/t_epoch:.2f}% from epoch  {forward_time/t_epoch:.2f}% of epoch ,\nbackward time {backward_time:.2f}s | {backward_time/t_epoch:.2f}% of epoch" 

    loss_verbose = f"Epoch {epoch}:loss={loss/n_batch}=[{regloss/n_batch}+{logloss/n_batch}]"
    metric = []

    for i,k in enumerate(Ks):
        metric.append(f"Hit@{k}: {result['Hit@k'][i]:.3f} , Precision@{k}: {result['Percision@k'][i]:.3f} , Recall@{k}:{result['Recall@k'][i]:.3f} , NDCG@{k}: {result['NDGC@k'][i]:.3f}")
    test_verbose = "\n".join(metric)
    
    if verbose > 0 :
        print(loss_verbose)

    if verbose > 1:
        print(test_verbose)

    if verbose > 2:
        print(time_verbose)



    



        
        

