
import torch
import torch.nn as nn
from gmpy2 import mpz


# 初始化模型参数
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def init_param(global_model):
        weight_model={}
        for name,params in global_model.state_dict().items():
            weight_model[name]=torch.zeros_like(params)
        return weight_model


def init_cipher_param(global_model,args):
    weight_model={}
    for name,params in global_model.state_dict().items():
        if name==args.layer:
            len=params.numel()
            weight_model[name]=[[mpz(1) for col in range(4)] for row in range(len)]
        else:
            weight_model[name]=torch.zeros_like(params)
    return weight_model


# if __name__=='__main__':
#     lstm=model.LSTM()
#     init_network(lstm)
    # for name,params in lstm.state_dict().items():
    #     print(params)
    #     break