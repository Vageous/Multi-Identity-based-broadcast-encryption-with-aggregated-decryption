import argparse
import torch

def parser_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epoch",type=int,default=10)
    parser.add_argument("--batchsize",type=int,default=32)
    parser.add_argument("--lr",type=float,default=0.1)
    parser.add_argument("--momentum",type=float,default=0.01)
    parser.add_argument("--dataset",type=str,default="cifar10")
    parser.add_argument("--num_user",type=int,default=200)
    parser.add_argument("--local_round",type=int,default=3)
    parser.add_argument("--frac",type=int,default=1)
    parser.add_argument("--bits",type=int,default=20)
    parser.add_argument("--ID",default=["user1","user2"])
    parser.add_argument("--scale",default=10000)
    parser.add_argument("--flag",default=0)
    parser.add_argument("--layer",default="conv2.bias")
    parser.add_argument("--device",default=1)
    parser.add_argument("--rate",type=float,default=0.1)
    parser.add_argument("--droptype",type=int,default=0)
    parser.add_argument("--iid",default=0)

    args=parser.parse_args()
    return args
    
