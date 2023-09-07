import argparse
import torch

def parser_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epoch",type=int,default=40)
    parser.add_argument("--batchsize",type=int,default=32)
    parser.add_argument("--lr",type=float,default=0.1)
    parser.add_argument("--momentum",type=float,default=0.001)
    parser.add_argument("--num_user",type=int,default=20)
    parser.add_argument("--local_round",default=1)
    parser.add_argument("--frac",type=float,default=1)
    parser.add_argument("--bits",default=20)
    parser.add_argument("--ID",default=["user1","user2"])
    parser.add_argument("--scale",default=10000)
    parser.add_argument("--flag",type=int,default=1)
    parser.add_argument("--layer",default="conv.0.bias")
    parser.add_argument("--device",default=1)
    parser.add_argument("--rate",type=float,default=0.2)
    parser.add_argument("--droptype",type=int,default=0)

    args=parser.parse_args()
    return args
    
