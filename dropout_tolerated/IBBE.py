from gmpy2 import mpz,invert,t_mod,gcd,powmod,is_prime,mpz_random,div,random_state
from Crypto.Util.number import getPrime
from copy import deepcopy
import random,sys
import numpy as np
from timeit import default_timer as timer
from memory_profiler import profile
import recovery_time as t

rand=random_state(random.randrange(sys.maxsize))
class KGC(object):
    def __init__(self,bits,num,ID) -> None:
        self.hash1,self.hash2,self.usk1,self.usk2,self.mpk1,self.msk1=[],[],[],[],[],[]
        self.num=num
        self.bits=bits
        self.ID=ID
        self.p_q_gen()
        self.n_gen()
        self.g_gen()
        self.pub_gen()
        self.ran_select()
        self.hash(self.ID)
        self.extract()
        self.ran_pick()
        self.usk2_gen()
        
        
    def p_q_gen(self):
        # self.p=getPrime(self.bits)
        # self.q=getPrime(self.bits)
        self.p=333872093501550099223529621903107941359
        # print(sys.getsizeof(self.p))
        self.q=181168781824753821810635617003597797899
        while True:
            if(is_prime(mpz(2)*self.p+mpz(1)) and is_prime(mpz(2)*self.q+mpz(1))):
                # print(self.p)
                # print(self.q)
                break
            else:
                self.p=getPrime(self.bits)
                self.q=getPrime(self.bits)

    def n_gen(self):
        self.n=(mpz(2)*self.p+mpz(1))*(mpz(2)*self.q+mpz(1))
        self.N=self.n*self.n
        # print(sys.getsizeof(self.N))

    def g_gen(self):
        temp=2
        temp1=powmod(temp,2,self.N)
        while True:
            if(gcd(temp1,self.N)!=1):
                temp=temp+1
                temp1=powmod(temp,2,self.N)
            else:
                break
        self.g=temp1

    def pub_gen(self):
        self.x1=mpz_random(rand,10000)
        self.x2=mpz_random(rand,10000)
        self.g1=powmod(self.g,self.x1*self.p,self.N)
        self.g2=powmod(self.g,self.x2*self.p,self.N)
        self.g3=powmod(self.g,self.p*self.p,self.N)

    def ran_select(self):
        s1=mpz_random(rand,10000)
        s2=mpz_random(rand,10000)
        self.order=self.n*self.p*self.q
        self.mod=self.n*self.q
        temp=t_mod(t_mod(self.x1*s1,self.mod)+t_mod(self.x2*s2,self.mod),self.mod)
        while True:
            if(gcd(temp,self.mod)==1):
                break
            else:
                self.s1=mpz_random(rand,10000)
                self.s2=mpz_random(rand,10000)
                temp=t_mod(t_mod(self.x1*s1,self.mod)+t_mod(self.x2*s2,self.mod),self.mod)
        return s1,s2

    def hash(self,ID):
        for i in range(self.num):
            s1,s2=self.ran_select()
            t1=mpz_random(rand,100000)
            t2=mpz_random(rand,100000)
            self.hash1.append(t_mod(t_mod(s1*self.p,self.N)+t_mod(t1*self.mod,self.N),self.N))
            self.hash2.append(t_mod(t_mod(s2*self.p,self.N)+t_mod(t2*self.mod,self.N),self.N))

    def extract(self):
        inv_p=invert(self.p,self.mod)
        for i in range(len(self.hash1)):
            h1=self.hash1[i]
            h2=self.hash2[i]
            s1=t_mod(h1*inv_p,self.mod)
            s2=t_mod(h2*inv_p,self.mod)
            temp=t_mod(t_mod(self.x1*s1,self.mod)+t_mod(self.x2*s2,self.mod),self.mod)
            self.usk1.append(invert(temp,self.mod))

    def ran_pick(self):
        j=0
        while(j<self.num):
            x=mpz_random(rand,10000)
            temp=powmod(self.g,x,self.N)
            if(gcd(temp,self.N)==1):
                self.mpk1.append(temp)
                self.msk1.append(x)
                j=j+1

    def usk2_gen(self):
        for i in range(len(self.mpk1)):
            temp1,temp2=mpz(1),mpz(1)
            for j in range(len(self.mpk1)):
                if(i<j):
                    temp1=t_mod(temp1*self.mpk1[j],self.N)
                elif(i>j):
                    temp2=t_mod(temp2*self.mpk1[j],self.N)
                else:
                    continue
            self.usk2.append(powmod(temp2*invert(temp1,self.N),self.msk1[i],self.N))


class params(object):
    def __init__(self,n,N,g1,g2,g3,hash1,hash2,num,mpk1,msk1) -> None:
        self.n=n
        self.N=N
        self.g1=g1
        self.g2=g2
        self.g3=g3
        self.hash1=hash1
        self.hash2=hash2
        self.num=num
        self.mpk1=mpk1
        self.msk1=msk1

class participant(object):
    def __init__(self,params) -> None:
        self.n=params.n
        self.N=params.N
        self.g1=params.g1
        self.g2=params.g2
        self.g3=params.g3
        self.hash1=params.hash1
        self.hahs2=params.hash2
        self.num=params.num
        self.mpk1=params.mpk1
        self.msk1=params.msk1

    def ran_select(self,r):
        temp1=powmod(self.g1,r,self.N)
        temp2=powmod(self.g2,r,self.N)
        temp3=powmod(self.g3,r,self.N)
        while True:
            if(gcd(temp1,self.N)!=1 or gcd(temp2,self.N)!=1 or gcd(temp3,self.N)!=1):
                temp1=powmod(self.g1,r,self.N)
                temp2=powmod(self.g2,r,self.N)
                temp3=powmod(self.g3,r,self.N)
            else:
                break
        return r

    # @profile(precision=4,stream=open("./communication/log/encrypt.log",'w+'))
    def private_enc(self,m,usk2):
        cipher=[]
        h1,h2=mpz(1),mpz(1)
        for i in range(len(self.hash1)):
            h1=t_mod(h1+self.hash1[i],self.N)
            h2=t_mod(h2+self.hahs2[i],self.N)
        h=t_mod(powmod(self.g1,h1,self.N)*powmod(self.g2,h2,self.N),self.N)
        for i in range(len(m)):
            temp=[]
            r=mpz_random(rand,10000)
            r=self.ran_select(r)
            temp.append(t_mod((powmod((mpz(1)+self.n),t_mod(m[i],self.n),self.N)*powmod(self.g3,r,self.N)),self.N))
            temp.append(powmod(h,r,self.N))
            temp.append(t_mod(powmod(self.g1,r,self.N)*usk2,self.N))
            temp.append(powmod(self.g2,r,self.N))
            cipher.append(temp)
        return cipher

    # @profile(precision=8,stream=open("./communication/log/decrypt.log",'w+'))
    def decrypt(self,usk1,uid,agg_cipher):
        plain=[]
        hash1=deepcopy(self.hash1)
        hash2=deepcopy(self.hahs2)
        del hash1[uid]
        del hash2[uid]
        h1,h2=mpz(1),mpz(1)
        for i in range(len(hash1)):
            h1=t_mod(h1+hash1[i],self.N)
            h2=t_mod(h2+hash2[i],self.N)
        for i in range(len(agg_cipher)):
            temp1=t_mod(powmod(invert(agg_cipher[i][2],self.N),h1,self.N)*powmod(invert(agg_cipher[i][3],self.N),h2,self.N),self.N)
            temp2=t_mod(agg_cipher[i][1]*temp1,self.N)
            temp3=powmod(temp2,usk1,self.N)
            temp4=t_mod(agg_cipher[i][0]*invert(temp3,self.N),self.N)
            plain.append(int((temp4-mpz(1))/self.n))
        return plain

    def h_prime(self,onlie_user,drop_user):
        recover_h=[]
        for i in range(len(onlie_user)):
            temp1,temp2=mpz(1),mpz(1)
            for j in range(len(drop_user)):
                if(onlie_user[i]<drop_user[j]):
                    temp1=t_mod(temp1*self.mpk1[drop_user[j]],self.N)
                else:
                    temp2=t_mod(temp2*self.mpk1[drop_user[j]],self.N)
            recover_h.append(powmod(temp1*invert(temp2,self.N),self.msk1[onlie_user[i]],self.N))
        sum_h=mpz(1)
        for i in range(len(recover_h)):
            sum_h=t_mod(sum_h*recover_h[i],self.N)
        return recover_h

class server(object):
    def __init__(self,params) -> None:
        self.N=params.N

    def aggregate(self,agg_cipher,cipher):
        for i in range(len(cipher)):
            for j in range(len(cipher[i])):
                agg_cipher[i][j]=t_mod(agg_cipher[i][j]*cipher[i][j],self.N)
        return agg_cipher

    def aggre_recover(self,h_prime,agg_cipher):
        sum_h=mpz(1)
        for i in range(len(h_prime)):
            sum_h=t_mod(sum_h*h_prime[i],self.N)

        for i in range(len(agg_cipher)):
            agg_cipher[i][2]=t_mod(agg_cipher[i][2]*sum_h,self.N)
        return agg_cipher

def test(kgc):
    print("-----------------------")
    print("prime check")
    if(is_prime(mpz(2)*kgc.p+mpz(1)) and is_prime(mpz(2)*kgc.q+mpz(1))):
        print("pass")
    else:
        print("error")
    print("-----------------------")
    print("generator check")
    if(powmod(kgc.g,kgc.order,kgc.N)==1):
        print("pass")
    else:
        print("error")
    temp=mpz(1)
    for i in range(len(kgc.usk2)):
        temp=t_mod(temp*kgc.usk2[i],kgc.N)
    print("-----------------------")
    print("second private key check")
    if(temp==1):
        print("pass")
    else:
        print("error")

def gradients_gen(shape):
    gradients=np.random.random(shape).tolist()
    for i in range(len(gradients)):
        gradients[i]=int(gradients[i]*10000)
    return gradients

def consistent_check(plaintext,new_plaintext,online_user):
    for i in range(len(plaintext)):
        if plaintext[i]*len(online_user)==new_plaintext[i]:
            continue
        else:
            print("Error")

def main(bits,num,dropout_num,id,shape,index):
    time1=timer()
    kgc=KGC(bits,num,id)
    time2=timer()
    pp=params(kgc.n,kgc.N,kgc.g1,kgc.g2,kgc.g3,kgc.hash1,kgc.hash2,kgc.num,kgc.mpk1,kgc.msk1)
    client=participant(pp)
    agg_server=server(pp)
    m=gradients_gen(shape)
    # m=[100 for i in range(10)]
    test(kgc)
    agg_cipher=[[mpz(1) for i in range(4)] for j in range(len(m))]
    drop_user=np.random.choice(range(kgc.num),dropout_num,replace=False)
    onlie_user=[]
    for i in range(kgc.num):
        if i not in drop_user:
            onlie_user.append(i)
    enc_time=0
    agg_time=0
    for j in range(len(onlie_user)):
        time3=timer()
        cipher=client.private_enc(m,kgc.usk2[onlie_user[j]])
        time4=timer()
        agg_cipher=agg_server.aggregate(agg_cipher,cipher)
        time5=timer()
        enc_time +=time4-time3
        agg_time +=time5-time4
    # time1=timer()
    time6=timer()
    h_prime=client.h_prime(onlie_user,drop_user)
    # sum_size=0
    # for i in range(len(h_prime)):
    #     sum_size += sys.getsizeof(h_prime[i])
    # time2=timer()
    new_agg_cipher=agg_server.aggre_recover(h_prime,agg_cipher)
    time7=timer()
    # time3=timer()
    time8=timer()
    for i in range(len(onlie_user)):
        plain=client.decrypt(kgc.usk1[0],0,new_agg_cipher)
    time9=timer()
    consistent_check(m,plain,onlie_user)
    # with num
    # t.recovery_cpu_num(sum_size/1024,num,dropout_num,index)
    # t.recovery_time_num(time2-time1,num,dropout_num,index)
    # t.dropout_aggregation_time_num(time3-time1,num,dropout_num,index)

    # with gradients
    # t.recovery_cpu_gradient(sum_size/1024,shape,index)
    # t.recovery_time_gradient(time2-time1,shape,index)
    # t.dropout_aggregation_time_gradient(time3-time1,shape,index)

    # t.non_dropout_aggregate_time_num(time3-time1,num,dropout_num,index)

    # t.nondropout_aggregation_time_gradient(time3-time1,shape,index)
    with open('./AD-IBBE/setup.txt','a+') as f:
        if index==0:
            f.write('total num:{},dropout num:{}\n'.format(num,dropout_num))
            f.write('{}\n'.format(time2-time1))
        else:
            f.write('{}\n'.format(time2-time1))

    with open('./AD-IBBE/enc.txt','a+') as f:
        if index==0:
            f.write('total num:{},dropout num:{}\n'.format(num,dropout_num))
            f.write('{}\n'.format(enc_time))
        else:
            f.write('{}\n'.format(enc_time))

    with open('./AD-IBBE/agg.txt','a+') as f:
        if index==0:
            f.write('total num:{},dropout num:{}\n'.format(num,dropout_num))
            f.write('{}\n'.format(agg_time+time7-time6))
        else:
            f.write('{}\n'.format(agg_time+time7-time6))

    with open('./AD-IBBE/dec.txt','a+') as f:
        if index==0:
            f.write('total num:{},dropout num:{}\n'.format(num,dropout_num))
            f.write('{}\n'.format(time9-time8))
        else:
            f.write('{}\n'.format(time9-time8))
    
   

if __name__=='__main__':
    num=[20,40,60,80,100]
    dropout_rate=[0,0.1,0.2,0.3]
    dropout_num=[]
    for i in range(len(num)):
        temp=[]
        for j in range(len(dropout_rate)):
            temp.append(int(num[i]*dropout_rate[j]))
        dropout_num.append(temp)
    # print(dropout_num)
        # the gradients shape set to 1000
        # the number of training nodes： 10，20，30，40，50
        # dropout rate: 20% 40% 60%:
    for j in range(len(dropout_num)):
        for k in range(len(dropout_num[j])):
            for i in range(10): 
                main(128,num[j],dropout_num[j][k],['user1'],(1000),i)

# if __name__=='__main__':
#     num=20
#     dropout_num=10
#     shape=[1000,2000,3000,4000,5000]
#     for i in range(len(shape)):
#         for j in range(10):
#             main(128,num,dropout_num,['user1'],shape[i],j)

# if __name__=='__main__':
#     num=[20,40,60,80,100]
#     dropout_num=0
#     for i in range(len(num)):
#         for j in range(10):
#             main(128,num[i],dropout_num,['user1'],1000,j)


# if __name__=='__main__':
#     num=20
#     dropout_num=0
#     shape=[1000,2000,3000,4000,5000]
#     for i in range(len(shape)):
#         for j in range(10):
#             main(128,num,dropout_num,['user1'],shape[i],j)