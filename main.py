from utils.load_dataset import MovieLens,Gorwala
from model import NGCF
import torch
from tqdm import tqdm 
from torchsummary import summary
from utils.test import test
from time import time





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1024
num_epochs = 100
learning_rate =1e-3
h_dim = 64
layers = [64,64,64]
Ks = [5,10,20]





# data_generator = MovieLens(batch_size=batch_size)

data_generator = Gorwala("D:\df\Master\gowalla",batch_size=batch_size)
data_generator.n_items

graph = data_generator.graph
graph = graph.to(device)

model = NGCF(graph,16,[16,16],0.3,batch_size)
model=model.to(device)


optimizer = torch.optim.Adam(model.parameters(),learning_rate)



n_batch = data_generator.n_train // batch_size + 1


# result = test(model,data_generator,batch_size,Ks)

for epoch in range(num_epochs):
    pbar = tqdm(total=data_generator.n_train)
    pbar.set_description(f"Epoch {epoch}:")
    loss,logloss , regloss = 0,0,0
    t_start= time()
    load_data_time = 0
    forward_time = 0
    backward_time = 0
    for batch in range(n_batch):
        t1 = time()
        users, pos_items, neg_items = data_generator.sample()

    
        optimizer.zero_grad()
        t2 = time()
        user_emb,pos_emb,neg_emb = model(users,pos_items,neg_items)
        t3 = time()
        # backward
        batch_loss,batch_logloss,batch_regloss = model.BprLoss(user_emb,pos_emb,neg_emb)
        batch_loss.backward() 
        optimizer.step()
        t4 = time()

        load_data_time += t2 - t1
        forward_time += t3 - t2
        backward_time +=t4 - t3

      

        loss +=batch_loss
        logloss += batch_logloss
        regloss += batch_regloss
        pbar.update(batch_size)
    pbar.close()
    t_end = time()
    t_epoch = t_end - t_start

    print(f"load Data time {load_data_time:.2f}s | {load_data_time/t_epoch:.2f}% of epoch",end=" ,")
    print(f"forward time {forward_time:.2f}s | {forward_time/t_epoch:.2f}% of epoch",end=" ,")
    print(f"backward time {backward_time:.2f}s | {backward_time/t_epoch:.2f}% of epoch")
    result = test(model,data_generator,batch_size,Ks)
    print(f"Epoch {epoch}:{loss}=[{regloss/n_batch}+{logloss/n_batch}],{result}")



        
        

