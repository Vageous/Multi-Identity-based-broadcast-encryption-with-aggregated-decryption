import sys
import time
import pickle
import logging
import argparse

import numpy as np
import recovery_time as compute
from tqdm import tqdm
import random
from utils import *
from copy import deepcopy
from client import User
from server import server

def main(user_ids,t,gradients,rate,index):
    pub_key_map = {}    # the dict storing all users' public keys
    pri_key_map = {}
    with tqdm(total=len(user_ids), desc='Generating keys', unit_scale=True, unit='') as bar:
        for id in user_ids:
            pub_key, priv_key = SIG.gen(nbits=1024)
            pub_key_map[id] = pub_key
            pri_key_map[id] = priv_key
            bar.update(1)
    # advertise_keys
    c_u_pk_map={}
    c_u_sk_map={}
    s_u_pk_map={}
    s_u_sk_map={}
    for i in user_ids:
        pubkey,prikey=KA.gen()
        c_u_pk_map[i]=pubkey
        c_u_sk_map[i]=prikey
        pubkey1,prikey1=KA.gen()
        s_u_pk_map[i]=pubkey1
        s_u_sk_map[i]=prikey1

    Server=server(t)
    for i in user_ids:
        client_1=User(t,i,pub_key_map[i],pri_key_map[i],c_u_pk_map[i],s_u_pk_map[i],c_u_sk_map[i],s_u_sk_map[i])
        Server.receive_key_signature(client_1.signature_gen())
        
    # sharekeys
    # print(Server.ka_pub_keys_map)
    # ka_pub_keys_map=Server.ka_pub_keys_map

    for i in user_ids:
        client_2=User(t,i,pub_key_map[i],pri_key_map[i],c_u_pk_map[i],s_u_pk_map[i],c_u_sk_map[i],s_u_sk_map[i])
        client_2.signature_ver(Server.ka_pub_keys_map,pub_key_map)
        Server.receive_ciphertext(client_2.getshare(user_ids,Server.ka_pub_keys_map))

    # masking
    u_2=Server.u_2
    for i in u_2:
        client_i=User(t,i,pub_key_map[i],pri_key_map[i],c_u_pk_map[i],s_u_pk_map[i],c_u_sk_map[i],s_u_sk_map[i])
        Server.receive_mask_gradient(client_i.maskgradients(gradients,Server.ciphertext_map,Server.ka_pub_keys_map))
    

    online_num=int((1-rate)*len(u_2))
    u_3=random.sample(u_2,online_num)
    # print(u_2)
    # print(u_3)
    for i in u_3:
        client_i=User(t,i,pub_key_map[i],pri_key_map[i],c_u_pk_map[i],s_u_pk_map[i],c_u_sk_map[i],s_u_sk_map[i])
        Server.receive_signature(client_i.consistency_signature(u_3))

    u_4=Server.u_4
    for i in u_4:
        client_i=User(t,i,pub_key_map[i],pri_key_map[i],c_u_pk_map[i],s_u_pk_map[i],c_u_sk_map[i],s_u_sk_map[i])
        client_i.consistency_check(u_3,Server.signature_map,pub_key_map)

    for i in u_4:
        client_i=User(t,i,pub_key_map[i],pri_key_map[i],c_u_pk_map[i],s_u_pk_map[i],c_u_sk_map[i],s_u_sk_map[i])
        Server.receive_unmask(client_i.unmasking(u_2,u_3,Server.ka_pub_keys_map,Server.ciphertext_map))

    # unmasking\
    shape=gradients.shape
    # print(Server.priv_key_shares_map)
    # print(sys.getsizeof(Server.priv_key_shares_map))
    size=compute.compute_size(Server.priv_key_shares_map)
    # print(size)

    # print(Server.random_seed_shares_map)
   
    result=Server.unmasking(shape,u_2,u_3)

    # print(result[2])
    # compute.recovery_time_num(result[2],len(user_ids),len(u_2)-len(u_3),index)
    # compute.dropout_aggregation_time_num(result[3],len(user_ids),len(u_2)-len(u_3),index)
    # compute.recovery_cpu_num(size,len(user_ids),len(u_2)-len(u_3),index)

    # compute.recovery_time_gradient(result[2],shape,index)
    # compute.dropout_aggregation_time_gradient(result[2],shape,index)
    # compute.recovery_cpu_gradient(size,shape,index)

    compute.non_dropout_aggregate_time_num(result[3],len(user_ids),len(u_2)-len(u_3),index)

    # compute.nondropout_aggregation_time_gradient(result[3],shape,index)

        


# if __name__=='__main__':
#     num=[100]
#     droprate=[0.3]
#     t=2
#     shape=1000
#     gradients=np.random.random(shape)

#     for i in range(len(num)):
#         user_ids=[]
#         for j in range(num[i]):
#             user_ids.append(str(j))
#     # 梯度向量的维度
#         for k in range(len(droprate)):
#             for l in range(5):
#                 main(user_ids,t,gradients,droprate[k],l)

if __name__=='__main__':
    num=[20,40,60,80,100]
    droprate=0
    t=2
    shape=1000
    gradients=np.random.random(shape)

    for i in range(len(num)):
        user_ids=[]
        for j in range(num[i]):
            user_ids.append(str(j))
    # 梯度向量的维度
        for l in range(10):
            main(user_ids,t,gradients,droprate,l)

# if __name__=='__main__':
#     num=20
#     droprate=0.5
#     shape=[1000,2000,3000,4000,5000]
#     t=2
#     user_ids=[]
#     for i in range(num):
#         user_ids.append(str(i))

#     for i in range(len(shape)):
#         gradients = np.random.random(shape[i])
#         for j in range(10):
#             main(user_ids,t,gradients,droprate,j)

# if __name__=='__main__':
#     num=20
#     droprate=0
#     shape=[1000,2000,3000,4000,5000]
#     t=2
#     user_ids=[]
#     for i in range(num):
#         user_ids.append(str(i))

#     for i in range(len(shape)):
#         gradients = np.random.random(shape[i])
#         for j in range(10):
#             main(user_ids,t,gradients,droprate,j)