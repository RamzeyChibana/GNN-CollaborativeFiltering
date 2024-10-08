import multiprocessing.pool
import numpy as np
import heapq
# import multiprocessing
from utils.metrices import *
from utils.load_dataset import MovieLens
from time import time
import heapq
import torch
from tqdm import tqdm
# cores = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(cores)





def check_batch(batch_ratings:np.array,user_test:dict,user_train:dict,users,K):
    new_batch_ratings = np.zeros(shape=(batch_ratings.shape[0],K))
    user_pos_test = np.zeros(shape=batch_ratings.shape[0])
    for i in range(batch_ratings.shape[0]):
        user_rates = batch_ratings[i]
        # Remove train items for the rates of user and take K first elements
        test_items = user_rates[~np.isin(user_rates,user_train[users[i]])][:K]
        # test_items = heapq.nlargest(K,test_items)
        new_batch_ratings[i]=np.isin(test_items,user_test[users[i]]) 
        user_pos_test[i]=len(user_test[users[i]])
    
    return new_batch_ratings , user_pos_test




def get_performence(batch_ratings,user_pos_test,Ks):
    hit = []
    ndgc = []
    percision=[]
    recall = []
    for k in Ks :
        hit.append(Hit_at_k(batch_ratings,k))
        percision.append(Percision_at_k(batch_ratings,k))
        ndgc.append(Ndgc_at_k(batch_ratings,k))
        recall.append(Recall(batch_ratings,k,user_pos_test))

    
    return  {"Hit@k":np.array(hit),"Percision@k":np.array(percision),"NDGC@k":np.array(ndgc),"Recall@k":np.array(recall)}

       




def test(model,data:MovieLens,batch_size,Ks):
    K = max(Ks)

    result = {"Hit@k":np.zeros((len(Ks))),"Percision@k":np.zeros(len(Ks)),"NDGC@k":np.zeros(len(Ks)),"Recall@k":np.zeros(len(Ks))}
    
    users = list(data.test_set.keys())
    num_users = len(users)

   
    user_test = data.test_set
    user_train = data.train_set
    items = np.arange(data.n_items)
    users_emb , items_emb , _ = model(users,items,[]) 
    n_batch = num_users // batch_size 
    pbar = tqdm(total=num_users)
    pbar.set_description("Testing :")
    for batch in range(0,num_users,batch_size):
       
        t1 = time()

        users_batch_emb = users_emb[batch:batch+batch_size]
        users_batch = users[batch:batch+batch_size]

        batch_ratings = model.rating(users_batch_emb,items_emb)
        t2 = time()
        batch_ratings = torch.argsort(batch_ratings,dim=1,descending=True)
        
        
        t3 = time()
        batch_ratings,user_pos = check_batch(batch_ratings.detach().cpu().numpy(),user_test,user_train,users_batch,K)
        t4 = time()
        result_batch = get_performence(batch_ratings,user_pos,Ks)
        t5 = time()
        # print(f"rating :{t2-t1} ,check_batch :{t3-t2},sorting :{t4-t3},metrics :{t5-t4} ")
        result["Hit@k"]+=result_batch["Hit@k"]/num_users
        result["Percision@k"]+=result_batch["Percision@k"]/num_users
        result["NDGC@k"]+=result_batch["NDGC@k"]/num_users
        result["Recall@k"]+=result_batch["Recall@k"]/num_users
            

        pbar.update(batch_size)
    pbar.close
    
    

    
    return result







        








    












