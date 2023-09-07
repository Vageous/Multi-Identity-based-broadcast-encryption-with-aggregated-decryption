import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def data(filepath):
    recovery_mean,recovery_std,recovery_time=[],[],[]
    with open(filepath,'r+') as f:
        lines=f.readlines()
        for line in lines:
            temp=line.strip().split()
            if len(temp)==1:
                recovery_time.append(float(temp[0]))
    new_recovery=np.array(recovery_time).reshape(15,10)
    for i in range(len(new_recovery)):
        recovery_mean.append(np.mean(new_recovery[i]))
        recovery_std.append(np.std(new_recovery[i]))
    return new_recovery,recovery_mean,recovery_std

def fig(mean,std,mean1,std1):
    new_mean=np.array(mean).reshape(5,3).T
    new_std=np.array(std).reshape(5,3).T
    new_mean1=np.array(mean1).reshape(5,3).T
    new_std1=np.array(std1).reshape(5,3).T

    x=np.arange(5)
    font_size=16
    tick_labels=['20','40','60','80','100']
    ylabels='Total Data for Recovery(KB)'
    xlabels='Number of Workers'
    #设置误差标记参数
    font={
        'weight':'semibold',
        'size':16
    }
    font1={
        'weight':'semibold',
        'size':12
    }
    '''
    #2A9D8C
    #E9C46B
    #E66F51
    '''
    color=['#2A9D8C','#E9C46B','#E66F51']
    label=["10%Dropout","20%Dropout","30%Dropout"]
    error_params1=dict(elinewidth=2,ecolor='#90C9E7',capsize=5)
    error_params2=dict(elinewidth=2,ecolor='#219EBC',capsize=5)
    error_params3=dict(elinewidth=2,ecolor='#136783',capsize=5)
    error=[error_params1,error_params2,error_params3]
    bar_width=0.3
    for i in range(len(new_mean)):
        plt.bar(x+bar_width*i,new_mean[i],bar_width,color=color[i],yerr=[(0,0,0,0,0),new_std[i]],error_kw=error[i],label=label[i])
        # plt.bar(x,new_mean1[i],bar_width,color=color[i],yerr=[(0,0,0,0,0),new_std1[i]],error_kw=error_params1)
    # plt.bar(x,mean[0],bar_width,color='#90C9E7',yerr=[(0,0,0,0,0),std[0]],error_kw=error_params1,label='60% Dropout II')
    # plt.bar(x+bar_width,mean[1],bar_width,color='#219EBC',yerr=[(0,0,0,0,0),std[1]],error_kw=error_params2,label='60% Dropout I')
    # plt.bar(x+2*bar_width,mean[2],bar_width,color='#136783',yerr=[(0,0,0,0,0),std[2]],error_kw=error_params3,label='No-Dropout')
    plt.xticks(x+bar_width,tick_labels,size=12,weight='semibold')
    plt.yticks(np.arange(0,30,5),size=12,weight='semibold')
    plt.grid(ls = '-.',axis='y', lw = 0.05)
    plt.xlabel(xlabels,fontdict=font)
    plt.ylabel(ylabels,fontdict=font)
    # plt.title("MNIST-LetNet",fontdict=font)
    plt.legend(loc='best',edgecolor='black',prop=font1)
    # plt.title("",fontdict=font2)

    plt.savefig('recovery_cpu_num.pdf',dpi=600)
    plt.close()



if __name__=='__main__':
    filename='./recovery_cpu_num.txt'
    filename1='/home/b1107/user/ct/code/multi-IBE/secureaggregation/recovery_cpu_num.txt'
    result1=data(filename)
    # print(result1[1])
    result2=data(filename1)

    # print(result2[1])
    fig(result1[1],result1[2],result2[1],result2[2])

    