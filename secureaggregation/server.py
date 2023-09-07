import sys,random,pickle
from utils import *
from copy import deepcopy
import numpy as np
from timeit import default_timer as timer


class server(object):
    def __init__(self,t) -> None:
        self.t=t
        self.ka_pub_keys_map={}
        self.ciphertext_map={}
        self.u_2=[]
        self.u_3=[]
        self.u_4=[]
        self.u_5=[]
        self.masked_gradients_list = []
        self.verification_gradients_list = []
        self.signature_map={}
        self.priv_key_shares_map = {}        # {id: []}
        self.random_seed_shares_map = {}


     # ka_pub_keys_map = {}    # {id: {c_pk: bytes, s_pk, bytes, signature: bytes}}
    def receive_key_signature(self,result):
        msg=pickle.loads(result)
        msg_dict={'c_pk':msg[1],'s_pk':msg[2],'signature':msg[3]}
        self.ka_pub_keys_map[msg[0]]=msg_dict
    
    # ciphertext_map={}  {u:{v1: ciphertexts, v2: ciphertexts}}
    def receive_ciphertext(self,ciphertext):
        msg=pickle.loads(ciphertext)
        id=msg[0]
        self.ciphertext_map[id]=msg[1]
        self.u_2.append(id)
        return self.ciphertext_map
    
    def receive_mask_gradient(self,mask_gradient):
        msg=pickle.loads(mask_gradient)
        self.u_3.append(msg[0])
        self.masked_gradients_list.append(msg[1])
        self.verification_gradients_list.append(msg[2])

    def receive_signature(self,signatue):
        msg=pickle.loads(signatue)
        id=msg[0]
        self.u_4.append(id)
        self.signature_map[id]=msg[1]

    def receive_unmask(self,result):
        msg=pickle.loads(result)
        id = msg[0]

        # retrieve the private key shares
        for key, value in msg[1].items():
            if key not in self.priv_key_shares_map:
                self.priv_key_shares_map[key] = []
            self.priv_key_shares_map[key].append(value)

        # retrieve the ramdom seed shares
        # {'1': '2-1bba4f5f2ba20bbd1b1594e', '2': '3-997770ac158bbc9967495f', '3': '4-17749eb6570f6bd611d396f', '4': '5-551c661ecc61be28d32980', '5': '6-132eee0d827ccbef0891990', '6': '7-10c15b918337bfb83f09a1', '7': '8-ee93d64adea2c07ff4f9b1', '8': '9-1cc6651043a0dc147aae9c1', '9': 'a-aa38cbbd9578c20f60d9d2'}
        for key, value in msg[2].items():
            if key not in self.random_seed_shares_map:
                self.random_seed_shares_map[key] = []
            self.random_seed_shares_map[key].append(value)
        # print(self.random_seed_shares_map)

        self.u_5.append(id)


    def unmasking(self,shape,u_2,u_3):
        recon_random_vec_0_list = []
        recon_random_vec_1_list = []
        # 恢复掉线客户端的mask1和mask2
        time1=timer()
        for u in u_2:
            if u not in u_3:
                # the user drops out, reconstruct its private keys and then generate the corresponding random vectors
                priv_key = SS.recon(self.priv_key_shares_map[u])
                for v in u_3:
                    shared_key = KA.agree(priv_key, self.ka_pub_keys_map[v]["s_pk"])

                    random.seed(shared_key)
                    s_u_v = random.randint(0, 2**32 - 1)

                    # expand s_u_v into two random vectors
                    rs = np.random.RandomState(s_u_v | 0)
                    p_u_v_0 = rs.random(shape)
                    rs = np.random.RandomState(s_u_v | 1)
                    p_u_v_1 = rs.random(shape)

                    if int(u) > int(v):
                        recon_random_vec_0_list.append(p_u_v_0)
                        recon_random_vec_1_list.append(p_u_v_1)
                    else:
                        recon_random_vec_0_list.append(-p_u_v_0)
                        recon_random_vec_1_list.append(-p_u_v_1)

        # reconstruct private mask vectors p_u_0 and p_u_1
        recon_priv_vec_0_list = []
        recon_priv_vec_1_list = []
        # 无论掉不掉线都会有这一步的，这一步是恢复在线客户端的mask
        # 10次 |U_3|
        time2=timer()

        for u in u_3:
            random_seed = SS.recon(self.random_seed_shares_map[u])
            rs = np.random.RandomState(random_seed | 0)
            priv_mask_vec_0 = rs.random(shape)
            rs = np.random.RandomState(random_seed | 1)
            priv_mask_vec_1 = rs.random(shape)

            recon_priv_vec_0_list.append(priv_mask_vec_0)
            recon_priv_vec_1_list.append(priv_mask_vec_1)

        masked_gradients = np.sum(np.array(self.masked_gradients_list), axis=0)
        recon_priv_vec_0 = np.sum(np.array(recon_priv_vec_0_list), axis=0)
        recon_random_vec_0 = np.sum(np.array(recon_random_vec_0_list), axis=0)

        output = masked_gradients - recon_priv_vec_0 + recon_random_vec_0

        verification_gradients = np.sum(np.array(self.verification_gradients_list), axis=0)
        recon_priv_vec_1 = np.sum(np.array(recon_priv_vec_1_list), axis=0)
        recon_random_vec_1 = np.sum(np.array(recon_random_vec_1_list), axis=0)
        time3=timer()

        verification = verification_gradients - recon_priv_vec_1 + recon_random_vec_1

        return output, verification,time2-time1,time3-time1
