from utils.evaluate import test
from utils.parser import test_parser
import numpy as np
import os 
import json 
import argparse
from model import NGCF
from utils.load_dataset import MovieLens,Gorwala
import pandas as pd
import torch




if __name__=="__main__":
    parser = test_parser()
    args = parser.parse_args()

    batch_size = args.batch_size
    Ks = args.ks
    if args.device=="gpu":
        device = torch.device("cuda")
    else :
        device = torch.device("cpu")

    exp = args.exp
    exps = os.listdir("Experiments")
    if exp not in exps :
        raise ValueError(f"There is no Experiement {exp} to test")
    
    
    def load_args(filename='args.json'):
        with open(filename, 'r') as f:
            args_dict = json.load(f)
        return argparse.Namespace(**args_dict)

    exp_args = load_args(os.path.join("Experiments",f"{exp}","args.json"))
    
    if exp_args.dataset == "movielens":
        data_generator = MovieLens(batch_size=batch_size)
    elif exp_args.dataset == "gorwala" :
        data_generator = Gorwala("Data\gowalla",batch_size=batch_size)
    else :
        raise TypeError("Invalid dataset ..")

    # model = NGCF(data_generator.graph.to(device),exp_args.dim,exp_args.layers,exp_args.dropout,exp_args.lamda).to(device)
    model = NGCF(data_generator.graph.to(device),16,[16.16],exp_args.dropout,exp_args.lamda).to(device)
    # model.load_state_dict(torch.load(os.path.join("Experiments",f"{exp}","best_weights.pt"),weights_only=True))



    result,(ratings_time,check_time,sorting_time,metrics_time,t_ep) = test(model,data_generator,batch_size,Ks)

    print(f"Time ratings :{ratings_time:.3f}/{ratings_time/t_ep:.3f}")
    print(f"Time checking :{check_time:.3f}/{check_time/t_ep:.3f}")
    print(f"Time sorting :{sorting_time:.3f}/{sorting_time/t_ep:.3f}")
    print(f"Time metrics_time :{metrics_time:.3f}/{metrics_time/t_ep:.3f}")

    result = pd.DataFrame.from_dict(result,orient="index",columns=Ks)
    print(result)

    










