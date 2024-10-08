import pandas as pd
import numpy as np
import torch 
from torch_geometric.data import HeteroData
from time import time
import random as rd


class MovieLens():
    def __init__(self,batch_size):
        
        t1 = time()
        ratings =pd.read_csv("Data/ratings.csv",sep="\t",encoding="latin-1")
        self.n_users = ratings["user_ID"].max()+1
        self.n_items = ratings["movie_ID"].max()+1
        user_item_src,user_item_dst = ratings["user_ID"].values,ratings["movie_ID"].values
        
        test_set = ratings.sample(n=20000,random_state=23)
        train_set = ratings.drop(test_set.index)
        self.n_train = train_set.shape[0]
        self.n_test = test_set.shape[0]
        train_set = train_set.groupby("user_ID")["movie_ID"].agg(list).to_dict()
        test_set = test_set.groupby("user_ID")["movie_ID"].agg(list).to_dict()

        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size

        self.users = np.arange(self.n_users)
        self.items = np.arange(self.n_items)

        self.graph = HeteroData()
        self.graph[("user","self_user","user")].edge_index = torch.tensor([self.users,self.users])
        self.graph[("item","self_item","item")].edge_index = torch.tensor([self.items,self.items])
        self.graph[("user","liked","item")].edge_index = torch.tensor([user_item_src,user_item_dst])
        self.graph[("item","liked_by","user")].edge_index = torch.tensor([user_item_dst,user_item_src])
        self.graph["user"].num_nodes = self.n_users
        self.graph["item"].num_nodes = self.n_items
        
        self.users= list(self.users)
        self.items= list(self.items)

        self.print_statistics()

    def print_statistics(self):
        print("n_users=%d, n_items=%d" % (self.n_users, self.n_items))
        print("n_interactions=%d" % (self.n_train + self.n_test))
        print(
            "n_train=%d, n_test=%d, sparsity=%.5f"
            % (
                self.n_train,
                self.n_test,
                (self.n_train + self.n_test) / (self.n_users * self.n_items),
            )
        )        
        
    
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [
                rd.choice(self.users) for _ in range(self.batch_size)
            ]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if (
                    neg_id not in self.train_set[u]
                    and neg_id not in neg_items
                ):
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    

class Gorwala(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + "/train.txt"
        test_file = path + "/test.txt"

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.exist_users = []

        user_item_src = []
        user_item_dst = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split(" ")
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
                    for i in l[1:]:
                        user_item_src.append(int(uid))
                        user_item_dst.append(int(i))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n")
                    try:
                        items = [int(i) for i in l.split(" ")[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        # training positive items corresponding to each user; testing positive items corresponding to each user
        self.train_set, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip("\n")
                    items = [int(i) for i in l.split(" ")]
                    uid, train_set = items[0], items[1:]
                    self.train_set[uid] = train_set

                for l in f_test.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip("\n")
                    try:
                        items = [int(i) for i in l.split(" ")]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        # construct graph from the train data and add self-loops
        user_selfs = [i for i in range(self.n_users)]
        item_selfs = [i for i in range(self.n_items)]

      

        

        self.graph = HeteroData()
        self.graph[("user", "user_self", "user")].edge_index = torch.tensor([user_selfs, user_selfs])
        self.graph[("item", "item_self", "item")].edge_index = torch.tensor([item_selfs, item_selfs])
        self.graph[("user", "liked", "item")].edge_index = torch.tensor([user_item_src, user_item_dst])
        self.graph[("item", "liked_by", "user")].edge_index = torch.tensor([user_item_dst, user_item_src])

        self.graph["user"].num_nodes = self.n_users
        self.graph["item"].num_nodes = self.n_items
        self.print_statistics()
        
    def sample(self):
        
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [
                rd.choice(self.exist_users) for _ in range(self.batch_size)
            ]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if (
                    neg_id not in self.train_set[u]
                    and neg_id not in neg_items
                ):
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    def print_statistics(self):
        print("n_users=%d, n_items=%d" % (self.n_users, self.n_items))
        print("n_interactions=%d" % (self.n_train + self.n_test))
        




