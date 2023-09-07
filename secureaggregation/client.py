import sys,random,pickle,logging
from utils import *
import numpy as np

class User(object):
    def __init__(self,t,id,pub_key,pri_key,c_u_pk,s_u_pk,c_u_sk,s_u_sk) -> None:
        self.t=t
        self.id=id
        self.sig=SIG
        self.ss=SS
        self.ae=AE
        self.ka=KA
        self.pub_key=pub_key
        self.pri_key=pri_key
        self.c_u_pk=c_u_pk
        self.s_u_pk=s_u_pk
        self.c_u_sk=c_u_sk
        self.s_u_sk=s_u_sk
        self.b_u=None



    def signature_gen(self):
        # self.c_u_pk,self.c_u_sk=self.ka.gen()
        # self.s_u_pk,self.s_u_sk=self.ka.gen()
        msg=pickle.dumps([self.c_u_pk,self.s_u_pk])
        signature=self.sig.sign(msg,self.pri_key)
        result=pickle.dumps([self.id,self.c_u_pk,self.s_u_pk,signature])
        return result

    # ka_pub_keys_map = {}    # {id: {c_pk: bytes, s_pk, bytes, signature: bytes}}
    # pub_key_map={id:pub_key}
    def signature_ver(self,ka_pub_keys_map,pub_key_map):
        status = True
        for key, value in ka_pub_keys_map.items():
            if key==self.id:
                continue
            else:
                msg = pickle.dumps([value["c_pk"], value["s_pk"]])

                res = SIG.verify(msg, value["signature"], pub_key_map[key])

                if res is False:
                    status = False
                    logging.error("user {}'s signature is wrong!".format(key))
        return status

    def getshare(self,u_1,ka_pub_keys_map):
        all_ciphertext={}
        b_u=random.randint(0,2**32-1)
        s_u_sk_share=self.ss.share(self.s_u_sk,self.t,len(u_1))
        b_u_share=self.ss.share(b_u,self.t,len(u_1))
        for i in range(len(u_1)):
            if u_1[i]==self.id:
                continue
            info=pickle.dumps([self.id,u_1[i],s_u_sk_share[i],b_u_share[i]])
            # print(ka_pub_keys_map[u_1[i]])
            temp=pickle.loads(info)
            share_key=self.ka.agree(self.c_u_sk,ka_pub_keys_map[u_1[i]]["c_pk"])
            ciphertext=self.ae.encrypt(share_key,share_key,info)
            all_ciphertext[u_1[i]]=ciphertext
            # print(all_ciphertext)
            # all_ciphertext={id:ciphertext}
        msg=pickle.dumps([self.id,all_ciphertext])
        return msg
    
    def maskgradients(self,gradients,ciphertext_map,ka_pub_keys_map):
        u_2=list(ciphertext_map.keys())
        b_u=random.randint(0,2**32-1)
        rs = np.random.RandomState(b_u | 0)
        priv_mask_vec_0 = rs.random(gradients.shape)
        rs = np.random.RandomState(b_u | 1)
        priv_mask_vec_1 = rs.random(gradients.shape)
        random_vec_0_list = []
        random_vec_1_list = []
        alpha = 0

        for v in u_2:
            if v == self.id:
                continue

            v_s_pk = ka_pub_keys_map[v]["s_pk"]
            shared_key = KA.agree(self.s_u_sk, v_s_pk)

            random.seed(shared_key)
            s_u_v = random.randint(0, 2**32 - 1)
            alpha = (alpha + s_u_v) % (2 ** 32)

            # expand s_u_v into two random vectors
            rs = np.random.RandomState(s_u_v | 0)
            p_u_v_0 = rs.random(gradients.shape)
            rs = np.random.RandomState(s_u_v | 1)
            p_u_v_1 = rs.random(gradients.shape)
            if int(self.id) > int(v):
                random_vec_0_list.append(p_u_v_0)
                random_vec_1_list.append(p_u_v_1)
            else:
                random_vec_0_list.append(-p_u_v_0)
                random_vec_1_list.append(-p_u_v_1)

        # expand α into two random vectors
        alpha = 10000
        rs = np.random.RandomState(alpha | 0)
        self.__a = rs.random(gradients.shape)
        rs = np.random.RandomState(alpha | 1)
        self.__b = rs.random(gradients.shape)

        verification_code = self.__a * gradients + self.__b

        masked_gradients = gradients + priv_mask_vec_0 + np.sum(np.array(random_vec_0_list), axis=0)
        verification_gradients = verification_code + priv_mask_vec_1 + np.sum(np.array(random_vec_1_list), axis=0)

        msg = pickle.dumps([self.id, masked_gradients, verification_gradients])

        return msg


    def consistency_signature(self,U_3):
        signature = self.sig.sign(pickle.dumps(U_3), self.pri_key)
        msg = pickle.dumps([self.id, signature])
        return msg
    
    def consistency_check(self,U_3,signature_map,pub_key_map):
        for key,value in signature_map.items():
            res = self.sig.verify(pickle.dumps(U_3), value, pub_key_map[key])
        return res

    
    # {u:{v1: ciphertexts, v2: ciphertexts}}
    def unmasking(self,U_2,U_3,ka_pub_keys_map,ciphertext_map):
        priv_key_shares_map = {}
        random_seed_shares_map = {}

        for v in U_2:
            if self.id == v:
                continue
            v_c_pk = ka_pub_keys_map[v]["c_pk"]
            shared_key = KA.agree(self.c_u_sk, v_c_pk)
            # print(self.ae.decrypt(shared_key, shared_key, ciphertext_map[v]))
            # print("\n")
            
            # info: [self.id,v,s_u_sk_share,b_u_share]
            info = pickle.loads(self.ae.decrypt(shared_key, shared_key, ciphertext_map[v][self.id]))
            # print(info)
            if v not in U_3:
                # send the shares of s_sk to the server
                priv_key_shares_map[v] = info[2]
            else:
                # send the shares of random seed to the server
                random_seed_shares_map[v] = info[3]
        # print(random_seed_shares_map)
        # {'1': '2-1bba4f5f2ba20bbd1b1594e', '2': '3-997770ac158bbc9967495f', '3': '4-17749eb6570f6bd611d396f', '4': '5-551c661ecc61be28d32980', '5': '6-132eee0d827ccbef0891990', '6': '7-10c15b918337bfb83f09a1', '7': '8-ee93d64adea2c07ff4f9b1', '8': '9-1cc6651043a0dc147aae9c1', '9': 'a-aa38cbbd9578c20f60d9d2'}
        msg = pickle.dumps([self.id, priv_key_shares_map, random_seed_shares_map])
        return msg

    def verify(self, output_gradients, verification_gradients, num_U_3):
        gradients_prime = self.__a * output_gradients + num_U_3 * self.__b

        return ((gradients_prime - verification_gradients) < np.full(output_gradients.shape, 1e-6)).all()