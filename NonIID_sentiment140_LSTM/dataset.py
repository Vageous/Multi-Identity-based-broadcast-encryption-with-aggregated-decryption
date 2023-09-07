import torch
import sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab



class textDataset(Dataset):
    def __init__(self,csv_path):
        super().__init__()
        self.install_data = pd.read_csv(csv_path, encoding='ISO-8859-1', sep=',')
        
    def __len__(self):
        return len(self.install_data)
    
    def __getitem__(self, idx):
        
        input_data = pd.DataFrame([])
        label = pd.DataFrame([])
        
        input_data = self.install_data.iloc[idx, 6]
        label = self.install_data.iloc[idx, 1]
        return label, input_data

class Predata(object):
    def __init__(self) -> None:
        # self.filepath=filepath
        self.tokenizer=get_tokenizer('basic_english')

    def split_data(self,filepath):
        df=pd.read_csv(filepath,encoding='ISO-8859-1', sep=',')
        train_data=df.sample(frac=0.8,replace=False)
        test_data=df[~df.index.isin(train_data.index)]
        train_data.to_csv("train_data.csv",index=False)
        test_data.to_csv("test_data.csv",index=False)
        
    def yield_tokens(self,data_iter):
        for _,text in data_iter:
            yield self.tokenizer(text)

    def pre_data(self,filename):
        file=['train','test']
        for id in range(len(filename)):
            datatest=textDataset(filename[id])
            vocab = build_vocab_from_iterator(self.yield_tokens(datatest), specials=["<pad>","<unk>"],min_freq=2)
            vocab.set_default_index(vocab["<unk>"])
            # 显示单词表中索引与单词的映射关系
            # print(Vocab.get_stoi(vocab))
             #词典大小：45432 test set 125899 train set
            # print(type(vocab))
            # print(type(vocab(["i","a","m"])))
            text_pipeline = lambda x : vocab(self.tokenizer(x))
            label_pipeline = lambda x : int(x)
            label_list,text_list,offsets=[],[],[0]
            for label,text in datatest:
                label_list.append(label_pipeline(label))
                # processed_text=torch.tensor(text_pipeline(text),dtype=torch.int64)
                text_list.append(text_pipeline(text))
                offsets.append(len(text_pipeline(text)))
            # max_len=max(offsets)
            for i in range(len(label_list)):
                if label_list[i]==4:
                    label_list[i]=1
            padlen=int(np.mean(offsets))
            for i in range(len(text_list)):
                if len(text_list[i])<padlen:
                    for j in range(padlen-len(text_list[i])):
                        text_list[i].append(0)
                else:
                    text_list[i]=text_list[i][:padlen]
            label_list = torch.tensor(label_list, dtype=torch.int64)
            text_list=torch.tensor(text_list,dtype=torch.int64)
            torch.save(text_list,'{}_text.pt'.format(file[id]))
            torch.save(label_list,'{}_label.pt'.format(file[id]))


# def dataset_gen(filepath):
#     data=Predata()
#     train_set=TensorDataset(data.pre_data(filepath[0])[1],data.pre_data(filepath[0])[0])
#     test_set=TensorDataset(data.pre_data(filepath[1])[1],data.pre_data(filepath[1])[0])
#     return train_set,test_set

def dataset_load(filepath):
    train_text=torch.load(filepath[0])
    print(train_text.shape)
    train_label=torch.load(filepath[1])
    test_text=torch.load(filepath[2])
    test_label=torch.load(filepath[3])
    train_set=TensorDataset(train_text,train_label)
    test_set=TensorDataset(test_text,test_label)
    return train_set,test_set

if __name__=='__main__':
    # df=pd.read_csv('./training.1600000.processed.noemoticon.csv',encoding='latin-1',sep=',')
    # df.sample(100000).to_csv('sampledata.csv')
    test=Predata()
    filepath=['train_text.pt','train_label.pt','test_text.pt','test_label.pt']
    # test.split_data('./sampledata.csv')
    # test.pre_data(["train_data.csv","test_data.csv"])
    train_set,test_set=dataset_load(filepath)
    train=DataLoader(train_set,batch_size=16,shuffle=True)
    for id,batch in enumerate(train):
        print(batch[0].shape)
        print(batch[1])
        break



    