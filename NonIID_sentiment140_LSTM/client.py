import torch
import gmpy2
import model
import init
import dataset
import sample
import transform
import numpy as np
import torch.nn as nn
from IBBE import IBBE
from config import parser_args
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

class LocalUpdate(object):
    def __init__(self,args,dataset,select_idxs,usk2,pp) -> None:
        self.num_user=args.num_user
        self.batchsize=args.batchsize
        self.lr=args.lr
        self.local_round=args.local_round
        self.layer=args.layer
        self.usk2=usk2
        self.pp=pp
        self.flag=args.flag
        self.scale=args.scale
        self.device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu")
        self.local_model=model.LSTM().to(self.device)
        self.train_set=DataLoader(sample.datasetsplit(dataset,select_idxs),batch_size=self.batchsize,shuffle=True)


    def encrypt(self,local_model):
        User=IBBE.participant(self.pp)
        for name,params in local_model.items():
            if name==self.layer:
                new_local,index=transform.encode(local_model[name],self.scale)
                local_model[name]=User.private_enc(new_local,self.usk2)
        return index

    def decrypt(self,aggre_cipher,uid,usk1,index):
        User=IBBE.participant(self.pp)
        for name,params in self.local_model.state_dict().items():
            if name==self.layer:
                plain=User.decrypt(usk1,uid,aggre_cipher[name])
                aggre_cipher[name]=transform.decode(plain,index,self.scale)
                aggre_cipher[name]=aggre_cipher[name].reshape(params.shape)
        return aggre_cipher

    def h_prime(self,onlie_user,drop_user):
        User=IBBE.participant(self.pp)
        return User.h_prime(onlie_user,drop_user)


    def train(self,global_model):
        local_model=dict()
        for name,param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer=torch.optim.Adam(self.local_model.parameters(),lr=self.lr)
        self.local_model.train()
        for i in range(self.local_round):
            for batch_idx,batch in enumerate(self.train_set):
                text=batch[0].to(self.device)
                label=batch[1].to(self.device)

                optimizer.zero_grad()
                output=self.local_model(text)
                loss=F.cross_entropy(output,label)
                loss.backward()

                optimizer.step()

        if self.flag==0:
            return self.local_model.state_dict()


# if __name__=='__main__':
#     args=parser_args()
#     kgc=IBBE.KGC(args.bits,args.num_user,args.ID)
#     pp=IBBE.params(kgc.n,kgc.N,kgc.g1,kgc.g2,kgc.g3,kgc.hash1,kgc.hash2,kgc.num,kgc.mpk1,kgc.msk1)
#     User=IBBE.participant(pp)
#     train_set,test_set=dataset.dataset_gen(args.filepath)
#     idx=sample.mnist_iid(train_set,args.num_user)
#     test=LocalUpdate(args,train_set,idx[0],kgc.usk2,pp)
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     global_model=model.LSTM().to(device)
#     local=test.train(global_model)
#     for name,params in local.items():
#         print(name)
#         print(params)
#         break