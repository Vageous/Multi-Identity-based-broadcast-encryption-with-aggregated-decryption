import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# python 

# plot(x,y,fmt,**kwargs)
# plot(x1,y1,fmt,x2,y2,fmt,...,**kwargs)
'''
x, y:点或线的节点,x 为 x 轴数据,y为 y 轴数据，数据可以列表或数组。
fmt:可选，定义基本格式（如颜色、标记和线条样式）。
**kwargs:可选，用在二维平面图上，设置指定属性，如标签，线的宽度等。
'''
'''
   color = ['#FF6347', '#008B8B', '#FF00FF', '#6495ED', '#FF0000', '#6A5ACD']
    linestyle = ['-', '--', '--', '--', '--', '--']
    marker = ['', '^', '*', '+', 'o', 'v']
    label=['0% Dropout I','10% Dropout I','20% Dropout I','30% Dropout I']
'''
def Figure1(mean,std,filename):
    x=np.arange(1,41)
    color = ['#E889BD','#67C2A3','#FC8C63','#8EA0C9','#CBA8A2']
    linestyle = ['-', '-', '-', '-']
    marker=['.', '.', '.', '.']
    font_size=16
    font2={
        # 'weight':'semibold',
      'size':18}
    font3={
    # 'weight':'semibold',
      'size':18}
    label=['No Dropout','20% Dropout I','40% Dropout I','60% Dropout I']
    plt.title("FEMNIST-AlexNet",fontdict=font2)
    xlabel='Training Rounds'
    ylabel='Model Accuracy(%)'    
    # legend=''
    for i in range(len(mean)):
        plt.plot(x,mean[i],linewidth = 2, color = color[i], marker=marker[i],linestyle = linestyle[i], label = label[i],markersize=8)
        plt.fill_between(x,mean[i]-std[i],mean[i]+std[i],color=color[i],alpha=0.3)
    plt.xticks(np.arange(5,45,5),size=18)
    plt.yticks(np.arange(0,110,10),size=18)
    plt.xlabel(xlabel,fontdict=font2)
    plt.ylabel(ylabel,fontdict=font2)
    plt.subplots_adjust(left=0.17, bottom=0.15)
    plt.grid(ls = '-.', lw = 0.1)
    plt.legend(loc='lower right',edgecolor='black',prop=font3)
    plt.savefig('./{}'.format(filename),dpi=600)
    plt.close()
    # plt.show()

def Figure2(mean,std,filename):
    
    x=np.arange(1,41)
    color = ['#E889BD','#67C2A3','#FC8C63','#8EA0C9','#CBA8A2']
    linestyle = ['-', '-', '-', '-']
    marker=['.', '.', '.', '.']
    font_size=16
    font1={
    'weight':'semibold',
      'size':12}
    font2={
        # 'weight':'semibold',
      'size':18}
    font3={
    # 'weight':'semibold',
      'size':18}
    label=['No Dropout','20% Dropout II','40% Dropout II','60% Dropout II']
    plt.title("FEMNIST-AlexNet",fontdict=font2)
    xlabel='Training Rounds'
    ylabel='Model Accuracy(%)'    
    # legend=''
    for i in range(len(mean)):
        plt.plot(x,mean[i],linewidth = 2, color = color[i], marker=marker[i],linestyle = linestyle[i], label = label[i],markersize=8)
        plt.fill_between(x,mean[i]-std[i],mean[i]+std[i],color=color[i],alpha=0.3)
    plt.xticks(np.arange(5,45,5),size=18)
    plt.yticks(np.arange(0,110,10),size=18)
    plt.xlabel(xlabel,fontdict=font2)
    plt.ylabel(ylabel,fontdict=font2)
    plt.grid(ls = '-.', lw = 0.1)
    plt.subplots_adjust(left=0.17, bottom=0.15)
    plt.legend(loc='lower right',edgecolor='black',prop=font3)
    plt.savefig('./{}'.format(filename),dpi=600)
    plt.close()
    # plt.show()
def deal(data):
    final_data=[]
    for i in range(40):
        temp=[]
        for j in range(5):
            temp.append(data[i+j*40])
        final_data.append(temp)
    return final_data

def loaddata(filepath,filename):
    filepath=os.path.join(filepath,filename)
    with open(filepath) as f:
        lines=f.readlines()
        length=len(lines)
        epoch,acc,loss=[],[],[]
        for line in lines:
            temp=line.strip().split(',')
            epoch.append(float(temp[0]))
            acc.append(float(temp[1]))
            loss.append(float(temp[2]))
        new_epoch=deal(acc)
        new_acc=deal(acc)
        new_loss=deal(loss)
    return epoch,new_acc,new_loss

def show():
    acc,loss,acc1,loss1=[],[],[],[]
    acc_mean,loss_mean,acc_mean1,loss_mean1,=[],[],[],[]
    acc_std,loss_std,acc_std1,loss_std1=[],[],[],[]
    filepath='./'
    filename1=['droptype0.txt','0.2droptype1.txt','0.4droptype1.txt','0.6droptype1.txt']
    filename2=['droptype0.txt','0.2droptype2.txt','0.4droptype2.txt','0.6droptype2.txt']
    # filename2=['nodropacc.txt','10%droptype2acc.txt','20%droptype2acc.txt','30%droptype2acc.txt']
    for i in range(len(filename1)):
        result=loaddata(filepath,filename1[i])
        acc.append(np.array(result[1]))
        loss.append(np.array(result[2]))

    for i in range(len(filename2)):
        result=loaddata(filepath,filename2[i])
        acc1.append(np.array(result[1]))
        loss1.append(np.array(result[2]))

    for i in range(len(acc)):
        acc_mean.append(np.mean(acc[i],axis=1))
        acc_mean1.append(np.mean(acc[i],axis=1))
        loss_mean.append(np.mean(loss[i],axis=1))
        loss_mean1.append(np.mean(loss[i],axis=1))
        acc_std.append(np.std(acc[i],axis=1))
        acc_std1.append(np.std(acc[i],axis=1))
        loss_std.append(np.std(loss[i],axis=1))
        loss_std1.append(np.std(loss[i],axis=1))

    Figure1(acc_mean,acc_std,"droptype1acc_femnist.pdf")
    Figure2(acc_mean1,acc_std1,"droptype2acc_femnist.pdf")



if __name__=='__main__':
    show()
