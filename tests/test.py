from utils.load_data import Data
import torch

device = torch.device("cuda")

batch_size = 16

data_generator = Data("D:\df\Master\gowalla",batch_size=batch_size)

graph = data_generator.g
print("--------------------Graph----------------------")
print(graph)
print("--------------------test------------------------")
n_batch = data_generator.n_train // batch_size + 1
print(n_batch)
# for batch in range(n_batch):
#     users, pos_items, neg_items = data_generator.sample()
#     print(users)
#     print(pos_items)
#     print(neg_items)
    

