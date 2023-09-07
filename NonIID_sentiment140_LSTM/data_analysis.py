import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt


def split_data(filepath):
    df=pd.read_csv(filepath,encoding='ISO-8859-1', sep=',')
    train_data=df.sample(frac=0.8,replace=False)
    test_data=df[~df.index.isin(train_data.index)]
    train_data.to_csv("train_data.csv",index=False)
    test_data.to_csv("test_data.csv",index=False)


def analysis(filename):
    df=pd.read_csv(filename,encoding='ISO-8859-1', sep=',',names=['label','1','2','3','4','text'])
    label_length=df.groupby('label')['label'].count()
    df['length']=df['text'].apply(lambda x:len(x))
    text_length=df.groupby('length').count()
    text_len=text_length.index.tolist()
    text_freq=text_length['text'].tolist()

    plt.bar(text_len,text_freq)
    plt.xlabel("the length of sentences")
    plt.ylabel("the freq of sentences")
    plt.savefig('./1.pdf')
    plt.close()

    sent_pentage_list=[count/sum(text_freq) for count in accumulate(text_freq)]

    plt.plot(text_len,sent_pentage_list)
    plt.savefig('./2.pdf')
    plt.close()
# 寻找分位点为quantile的句子长度
    quantile=0.99
    for length,per in zip(text_len,sent_pentage_list):
        if round(per,2)==quantile:
            index=length
            break
    print('\n分位点维%s的句子长度:%d.'%(quantile,index))


if __name__=='__main__':
    filename='/home/b1107/user/ct/code/multi-IBE/NonIID_sentiment140_LSTM/test_data.csv'
    # split_data(filename)
    filepath='./training.1600000.processed.noemoticon.csv'
    analysis(filepath)
