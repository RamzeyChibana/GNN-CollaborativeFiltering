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

        torch.nn.init.xavier_uniform_(self.W1.weight)
        # torch.nn.init.xavier_uniform_(self.W1.bias)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        # torch.nn.init.xavier_uniform_(self.W2.bias)


    
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
        super(NGCF,self).__init__()
        self.lambd = lambd
        self.layer_dim = layer_dim
        self.num_layers = len(self.layer_dim)
        self.graph = graph
       
    
        num_users = self.graph["user"].num_nodes
        num_items = self.graph["item"].num_nodes
        norm_dict = dict()
        for (srctype,etype,dsttype) in self.graph.edge_types:
            
            edge_index = self.graph[(srctype,etype,dsttype)].edge_index
            src,dst = edge_index
          
            dst_degree = degree(dst,self.graph[dsttype].num_nodes)[dst]
            src_degree = degree(src,self.graph[srctype].num_nodes)[src]

            norm = torch.pow(src_degree*dst_degree,-0.5).unsqueeze(1)
            norm_dict[(srctype,etype,dsttype)]=norm





        initzer = nn.init.xavier_uniform_
        self.emb_dict = nn.ParameterDict({"user":initzer(nn.Parameter(torch.empty(size=(num_users,h_dim)))),"item":initzer(nn.Parameter(torch.empty(size=(num_items,h_dim))))})

        self.layers = torch.nn.ModuleList()
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
            h_dict = layer(self.graph,h_dict)
            user_embds.append(h_dict["user"])
            item_embds.append(h_dict["item"])
        user_embds = torch.cat(user_embds,dim=1)
        item_embds = torch.cat(item_embds,dim=1)
        
     
    

        user_certain_emb = user_embds[users,:]
        pos_certain_emb = item_embds[pos_items,:]
        neg_certain_emb = item_embds[neg_items,:]
        

        return user_certain_emb,pos_certain_emb,neg_certain_emb
    
    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())
    
    
    def BprLoss(self,users,pos_items,neg_items):
        y_pos = torch.sum(torch.multiply(users,pos_items),dim=1)
        y_neg = torch.sum(torch.multiply(users,neg_items),dim=1)

        loss = torch.log(torch.sigmoid(y_pos-y_neg))
        ln_loss = torch.negative(torch.mean(loss))

        regularization = (torch.sum(torch.pow(users,2))  + torch.sum(torch.pow(pos_items,2))  + torch.sum(torch.pow(neg_items,2)) )/ 2
        regularization = regularization/users.shape[0]

        emb_loss = self.lambd * regularization


        return ln_loss+emb_loss ,ln_loss,emb_loss






if __name__=="__main__":
    print("Test the model:")
    def create_sample_graph_pyg():
        # Create a PyG heterogeneous graph
        data = HeteroData()

        # Add node features for 'user' and 'item' node types
        data['user'].x = torch.randint(low=1,high=9,size=(3, 5))  # 3 'user' nodes, feature size 5
        data['item'].x = torch.randint(low=10,high=19,size=(3, 5))  # 3 'item' nodes, feature size 5

        # Add self-loop edges for 'follows' relation (user-user self-loops)

        # Add user-item interaction edges for 'likes' relation (varied interactions)
        data['user', 'likes', 'item'].edge_index = torch.tensor([[0, 0, 2, 2], [0, 1, 2, 1]])

        # Add item-user interaction edges for 'liked_by' relation (reverse of 'likes')
        # data['item', 'liked_by', 'user'].edge_index = torch.tensor([[0, 1, 2, 1], [0, 0, 2, 2]])
        data['user', 'follows', 'user'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])  # Self-loops

        return data

    graph = create_sample_graph_pyg()
    model = NGCF(graph,16,[16,16],0.3,4)

    users = np.random.randint(0,3,size=(4,1))
    pos_items = np.random.randint(0,3,size=(4,1))
    neg_items = np.random.randint(0,3,size=(4,1))

    users = torch.tensor(users)
    pos_items = torch.tensor(pos_items)
    neg_items = torch.tensor(neg_items)
    result = model(users,pos_items,neg_items)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)
    print("loss")
    print(model.BprLoss(result[0],result[1],result[2]))

        


                
            














