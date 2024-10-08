import argparse
import sys





def make_parser():

    parser = argparse.ArgumentParser(description="Train Model")

    parser.add_argument("-bs","--batch_size",default=1024,type=int,help="batch size of nodes")
    parser.add_argument("-ep","--epochs",default=10,type=int,help="num of epochs for the training")
    parser.add_argument("-lr","--learning_rate",default=0.001,type=float,help="learning rate")
    parser.add_argument("-hd","--dim",default=16,type=int,help="Nodes embedding dimension")
    parser.add_argument("-l","--layers",nargs="+",type=int,default=[16,16],help="Num of layers and theire dimension")
    parser.add_argument("-dt","--dataset",default="gorwala",choices=["movielens","gorwala"],help="Dataset")
    parser.add_argument("-dp","--dropout",default=0.0,type=float,help="Dropout Rate")
    parser.add_argument("-ks","--ks",default=[3,5,10],nargs="+",type=int,help="@K to test with")
    parser.add_argument("-dv","--device",choices=["gpu","cpu"],default="gpu",help="device to train with")
    parser.add_argument("-ex","--exp",help="Continue Experiement")
    parser.add_argument("-v","--verbose",default=1,type=int,choices=[0,1,2,3],help="How much more infos in epoch 0:nothing,1:+loss,2:+Meterices,3:time")
    
    return parser

p = make_parser()
args = p.parse_args()
print(sys.argv[1])



