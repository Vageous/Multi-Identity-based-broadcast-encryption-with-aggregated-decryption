import model
import torch
import client
import dataset
import central_server
import sample
import numpy as np
from IBBE import IBBE
import init
import time 
from option import parser_args
from memory_profiler import profile

# @profile(precision=4,stream=open('./communication/log/total.log','a+'))
def fed_main():
    args=parser_args()
    select_users_num=int(args.frac*args.num_user)
    drop_user_num=int(args.rate*select_users_num)
    kgc=IBBE.KGC(args.bits,select_users_num,args.ID)
    pp=IBBE.params(kgc.n,kgc.N,kgc.g1,kgc.g2,kgc.g3,kgc.hash1,kgc.hash2,kgc.num,kgc.mpk1,kgc.msk1)
    train_set,test_set=dataset.dataset_download()
    server=central_server.Server(args,test_set,pp)
    global_model=model.AlexNet().to(device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu"))
    idxs=sample.fashionmnist(train_set,args.num_user)
    for iter in range(args.epoch):
        time1=time.time()
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
            weight_model=init.init(global_model)
        else:
            weight_model=init.init_cipher(global_model,args)

        if args.droptype==0:
            print("No Dropout---------------")
            for i in range(len(select_idxs)):
                participant=client.LocalUpdate(args,train_set,idxs[select_idxs[i]],kgc.usk2[i],pp)
                if args.flag==0:
                    local_model=participant.train(global_model)
                    server.model_aggregate(local_model,weight_model)
                else:
                    print("Enc Model-------------")
                    cipher_local,index=participant.train(global_model)
                    server.cipher_model_aggregate(cipher_local,weight_model)

            if args.flag==0:
                server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)
            else:
                print("Dec Mode---------------")
                participant.decrypt(weight_model,0,kgc.usk1[0],index)
                server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)

        elif args.droptype==1:
            print("Droptype I----------")
            for i in range(len(select_idxs)):
                participant=client.LocalUpdate(args,train_set,idxs[select_idxs[i]],kgc.usk2[i],pp)
                if args.flag==0:
                    local_model=participant.train(global_model)
                    server.model_aggregate(local_model,weight_model)
                else:
                    print("Enc Mode-----------------")
                    cipher_local,index=participant.train(global_model)
                    if i not in drop_user:
                        server.cipher_model_aggregate(cipher_local,weight_model)

            if args.flag==0:
                server.model_average(weight_model,len(onlie_user))
                global_model.load_state_dict(weight_model)
            else:
                print("Dropout Recover Mode-----------------")
                time3=time.time()
                h=participant.h_prime(onlie_user,drop_user)
                server.cipher_model_recover(weight_model,h)
                time4=time.time()-time3
                participant.decrypt(weight_model,0,kgc.usk1[0],index)
                server.model_average(weight_model,len(onlie_user))
                global_model.load_state_dict(weight_model)
        
        else:
            print("Droptype II--------------")
            for i in range(len(onlie_user)):
                participant=client.LocalUpdate(args,train_set,idxs[select_idxs[onlie_user[i]]],kgc.usk2[onlie_user[i]],pp)
                if args.flag==0:
                    local_model=participant.train(global_model)
                    server.model_aggregate(local_model,weight_model)
                else:
                    print("Enc Mode-----------------")
                    cipher_local,index=participant.train(global_model)
                    server.cipher_model_aggregate(cipher_local,weight_model)

            if args.flag==0:
                server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)
            else:
                print("Dropout Recover Mode-----------------")
                time5=time.time()
                h=participant.h_prime(onlie_user,drop_user)
                server.cipher_model_recover(weight_model,h)
                time6=time.time()-time5
                participant.decrypt(weight_model,0,kgc.usk1[0],index)
                server.model_average(weight_model,len(onlie_user))
                global_model.load_state_dict(weight_model)

        acc,total_loss=server.model_test(global_model)
        # temp.append(acc)
        if args.droptype==0:
            with open("./result/model_acc/droptype{}.txt".format(args.droptype),'a+') as f:
                f.write("{},{},{}\n".format(iter,acc,total_loss))
            print("Epoch %d, acc: %f, loss: %f\n" % (iter, acc, total_loss))
        else:
            with open("./result/model_acc/{}droptype{}.txt".format(args.rate,args.droptype),'a+') as f:
                f.write("{},{},{}\n".format(iter,acc,total_loss))
            print("Epoch %d, acc: %f, loss: %f\n" % (iter, acc, total_loss))
        # print("Epoch %d, acc: %f, loss: %f\n" % (iter, acc, total_loss))
        time2=time.time()-time1
        # if args.droptype==0:
        #     with open("./result/total overhead/droptype{}.txt".format(args.droptype),'a+') as f:
        #         f.write("{}\n".format(time2)) 
        # else:
        #     with open("./result/total overhead/{}droptype{}.txt".format(args.rate,args.droptype),'a+') as f:
        #         f.write("{}\n".format(time2))
        # if args.droptype==1:
        #     with open("./result/recovery/{}droptype{}.txt".format(args.rate,args.droptype),'a+') as f:
        #         f.write("{}\n".format(time4))
        # else:
        #     with open("./result/recovery/{}droptype{}.txt".format(args.rate,args.droptype),'a+') as f:
        #         f.write("{}\n".format(time6))
    # # return temp

if __name__=='__main__':
# num_user=200
# select_user=10
    # filename=['nodropacc','10%droptypeacc']
    for i in range(4):
        fed_main()