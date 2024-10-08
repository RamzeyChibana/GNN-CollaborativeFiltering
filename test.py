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

    exp = args.exp
    batch_size = args.batch_size
    Ks = args.ks
    if args.device=="gpu":
        device = torch.device("cuda")
    else :
        device = torch.device("cpu")
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
        data_generator = Gorwala("D:\df\Master\gowalla",batch_size=batch_size)
    else :
        raise TypeError("Invalid dataset ..")

    model = NGCF(data_generator.graph.to(device),exp_args.dim,exp_args.layers,exp_args.dropout).to(device)
    model.load_state_dict(torch.load(os.path.join("Experiments",f"{exp}","best_weights.pt"),weights_only=True))



    result = test(model,data_generator,batch_size,Ks)

    result = pd.DataFrame.from_dict(result,orient="index",columns=Ks)
    print(result)

    










