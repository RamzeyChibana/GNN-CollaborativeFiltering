import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree




class NgcfLayer(MessagePassing):
    def __init__(self,in_dim,out_dim,norm_dict,dropout_rate=0.4):
        super(NgcfLayer,self).__init__(aggr="add")
        self.W1 = nn.Linear(in_dim,out_dim)
        self.W2 = nn.Linear(in_dim,out_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout_rate)
        self.norm_dict = norm_dict

    
    def forward(self,graph:HeteroData,embedding_dict) :

        edge_types = graph.edge_types
        out_dict = dict()
        # iterate over all edge types (Hetero Graph)
        for (src_type,etype,dst_type) in edge_types:
            edge_index = graph[(src_type,etype,dst_type)].edge_index
            norm = self.norm_dict[(src_type,etype,dst_type)]
            # for self loops
            if src_type == dst_type:
                emb = embedding_dict[src_type]
                out = self.propagate(edge_index,x=emb,etype="self_loop",norm=norm)
            # for interactions between users and items 
            else :

                emb_src = embedding_dict[src_type]
                emb_dst = embedding_dict[dst_type]
                # Aggregation
                out = self.propagate(edge_index,x=(emb_src,emb_dst),etype = "cross",norm=norm)
            
            # Aggregate embeddings for every type of nodes from diffrent edge types (user<-(self_loop_user+items))
            if dst_type not in out_dict :
                out_dict[dst_type]=out
            else :
                out_dict[dst_type]+=out
            
        for node_type,h in out_dict.items():
            h=self.leaky_relu(h)
            h=self.drop(h)
            h = F.normalize(h,p=2,dim=1)
            out_dict[node_type]=h
        return out_dict
    

    def message(self,x_j,x_i,etype,norm):
        if etype == "self_loop":
            return self.W1(x_j)
        elif etype == "cross":
            return norm*self.W1(x_j)+self.W2(x_j*x_i)
        else :
            raise AttributeError(f"there is no etype called {etype}")
    

class NGCF(torch.nn.Module):
    def __init__(self,graph:HeteroData,h_dim,layer_dim,dropout,batch_size,lambd=1e-5):
        super(NgcfLayer,self).__init__()
        self.lambd = lambd
        self.num_layers = layer_dim
        self.graph = graph
        self.batch_size = batch_size

        num_users = graph["user"].num_users
        num_items = graph["item"].num_users
        norm_dict = dict()
        for (srctype,etype,dsttype) in graph.edge_types:
            if srctype != dsttype:
                edge_index = graph[(srctype,etype,dsttype)].edge_index
                src,dst = edge_index
                src_degree = degree(src,graph[srctype].num_nodes,dtype=torch.float32)[src]
                dst_degree = degree(dst,graph[dsttype].num_nodes,dtype=torch.float32)[dst]

                norm = torch.pow(src_degree*dst_degree,-0.5).unsqueeze(1)
                norm_dict[(srctype,etype,dsttype)]=norm





        initzer = nn.init.xavier_uniform_
        self.emb_dict = nn.ParameterDict({"user":initzer(nn.Parameter(torch.empty(size=(num_users,h_dim)))),"item":initzer(nn.Parameter(torch.empty(size=(num_items,h_dim))))})

        self.layers = []
        self.layers.append(NgcfLayer(h_dim,layer_dim[0],norm_dict,dropout_rate=dropout))
        for i in range(self.num_layers-1):
            self.layers.append(NgcfLayer(layer_dim[i],layer_dim[i+1],norm_dict,dropout_rate=dropout))
        
    def forward(self,users,pos_items,neg_items):
        h_dict = {"user":self.emb_dict["user"],"item":self.emb_dict["item"]}

        user_embds = []
        item_embds = []

        

        user_embds.append(h_dict["user"])
        item_embds.append(h_dict["item"])
        for layer in self.layers:
            h_dict = layer(h_dict)
            user_embds.append(h_dict["user"])
            user_embds.append(h_dict["user"])
        user_embds = torch.cat(user_embds,dim=1)
        item_embds = torch.cat(item_embds,dim=1)

        user_certain_emb = user_embds[users,:]
        pos_certain_emb = item_embds[pos_items,:]
        neg_certain_emb = item_embds[neg_items,:]

        return user_certain_emb,pos_certain_emb,neg_certain_emb
    
    def BprLoss(self,users,pos_items,neg_items):
        y_pos = torch.sum(torch.multiply(users,pos_items),dim=1)
        y_neg = torch.sum(torch.multiply(users,neg_items),dim=1)

        loss = torch.log(torch.sigmoid(y_pos-y_neg))
        ln_loss = torch.negative(torch.mean(loss))

        regularization = (torch.sum(torch.pow(users,2))  + torch.sum(torch.pow(pos_items,2))  + torch.sum(torch.pow(neg_items,2)) )/ 2
        regularization = regularization/self.batch_size

        emb_loss = self.lambd * regularization


        return ln_loss+emb_loss ,ln_loss,emb_loss








        


                
            














