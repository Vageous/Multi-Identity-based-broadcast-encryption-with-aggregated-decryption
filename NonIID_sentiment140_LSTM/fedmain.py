import model
import torch
import client
import dataset
import central_server
import sample
import numpy as np
from IBBE import IBBE
import init
from config import parser_args
from memory_profiler import profile


def fed_main():
    args=parser_args()
    # the amount of clients
    select_users_num=int(args.frac*args.num_user)
    # the amount of dropout user
    drop_user_num=int(args.rate*select_users_num)
    kgc=IBBE.KGC(args.bits,select_users_num,args.ID)
    pp=IBBE.params(kgc.n,kgc.N,kgc.g1,kgc.g2,kgc.g3,kgc.hash1,kgc.hash2,kgc.num,kgc.mpk1,kgc.msk1)
    train_set,test_set=dataset.dataset_load(args.filepath)
    server=central_server.Server(args,train_set,test_set,pp)
    global_model=model.LSTM().to(device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu"))
    idxs=sample.mnist_iid(train_set,args.num_user)
    # init.init_network(global_model)

    for iter in range(args.epoch):
        weight_model={}
        onlie_user=[]
        # select user data for local training
        select_idxs=np.random.choice(range(args.num_user),select_users_num,replace=False)
        # drop_user list
        drop_user=np.random.choice(range(select_users_num),drop_user_num,replace=False)
        for i in range(select_users_num):
            if i not in drop_user:
                onlie_user.append(i)

        if args.flag==0:
            weight_model=init.init_param(global_model)
        else:
            weight_model=init.init_cipher_param(global_model,args)

        if args.droptype==0:
            for i in range(len(select_idxs)):
                participant=client.LocalUpdate(args,train_set,idxs[select_idxs[i]],kgc.usk2[i],pp)
                if args.flag==0:
                    local_model=participant.train(global_model)
                    server.model_aggregate(local_model,weight_model)
                else:
                    cipher_local,index=participant.train(global_model)
                    server.cipher_model_aggregate(cipher_local,weight_model)

            if args.flag==0:
                server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)
            else:
                participant.decrypt(weight_model,0,kgc.usk1[0],index)
                server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)

        elif args.droptype==1:
            for i in range(len(select_idxs)):
                participant=client.LocalUpdate(args,train_set,idxs[select_idxs[i]],kgc.usk2[i],pp)
                if args.flag==0:
                    local_model=participant.train(global_model)
                    server.model_aggregate(local_model,weight_model)
                else:
                    cipher_local,index=participant.train(global_model)
                    if i not in drop_user:
                        server.cipher_model_aggregate(cipher_local,weight_model)

            if args.flag==0:
                server.model_average(weight_model,len(onlie_user))
                global_model.load_state_dict(weight_model)
            else:
                h=participant.h_prime(onlie_user,drop_user)
                server.cipher_model_recover(weight_model,h)
                participant.decrypt(weight_model,0,kgc.usk1[0],index)
                server.model_average(weight_model,len(onlie_user))
                global_model.load_state_dict(weight_model)
        
        else:
            for i in range(len(onlie_user)):
                participant=client.LocalUpdate(args,train_set,idxs[select_idxs[onlie_user[i]]],kgc.usk2[onlie_user[i]],pp)
                if args.flag==0:
                    local_model=participant.train(global_model)
                    server.model_aggregate(local_model,weight_model)
                else:
                    cipher_local,index=participant.train(global_model)
                    server.cipher_model_aggregate(cipher_local,weight_model)

            if args.flag==0:
                server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)
            else:
                h=participant.h_prime(onlie_user,drop_user)
                server.cipher_model_recover(weight_model,h)
                participant.decrypt(weight_model,0,kgc.usk1[0],index)
                server.model_average(weight_model,len(onlie_user))
                global_model.load_state_dict(weight_model)

        acc,total_loss=server.model_test(global_model)
        print("测试集:\n")
        print("Epoch %d, acc: %f, loss: %f\n" % (iter, acc, total_loss))
        print("训练集:\n")
        acc1,total_loss1=server.model_test1(global_model)
        print("Epoch %d, acc: %f, loss: %f\n" % (iter, acc1, total_loss1))

if __name__=='__main__':
    fed_main()