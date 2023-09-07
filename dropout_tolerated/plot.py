import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def pre_data(filename):
    with open(filename,'r') as f:
        recovery_time=[]
        lines=f.readlines()
        for line in lines:
            recovery_time.append(float(line.strip().split()))
    new_recovery_time=np.array(recovery_time).reshape(3,5,10)
    mean=np.mean(new_recovery_time,axis=1)
    std=np.std(new_recovery_time,axis=1)
    return mean,std

def Figure(mean,std,filename):
    #
    matplotlib.rc('font', size=12,weight='semibold')
    x=np.arange(5)
    font_size=12
    font={
        'weight':'semibold',
        'size':16
    }
    font1={
        'weight':'semibold',
        'size':12
    }
    tick_labels=['1000','2000','3000','4000','5000']
    plt.title("MNIST-LetNet",fontdict=font)
    xlabels='Number of gradients'
    ylabels='Dropout recovery times(s)'
    #设置误差标记参数
    plt.errorbar(x,mean[0],yerr=std[0],linewidth=1.5,mec='black',color='#2A9D8C',marker='o',fmt='-', ecolor='#2A9D8C',elinewidth=2, capsize=2, capthick=1, barsabove=True,label='20% Dropout I')
    plt.errorbar(x,mean[1],yerr=std[1],linewidth=1.5,mec='black',color='#E9C46B',marker='o',fmt='-', ecolor='#E9C46B',elinewidth=2, capsize=2, capthick=1, barsabove=True,label='40% Dropout I')
    plt.errorbar(x,mean[2],yerr=std[2],linewidth=1.5,mec='black',color='#E66F51',marker='o',fmt='-', ecolor='#E66F51',elinewidth=2, capsize=2, capthick=1, barsabove=True,label='60% Dropout I')
    plt.xticks(x,tick_labels,size=12,weight='semibold')
    plt.yticks(size=12,weight='semibold')
    ax=plt.gca()
    ax.ticklabel_format(style='scientific',scilimits=(-3,1), axis='y',useMathText=True)
    plt.grid(ls = '-.', lw = 0.05)
    plt.xlabel(xlabels,fontdict=font)
    plt.ylabel(ylabels,fontdict=font)
    plt.legend(loc='best',edgecolor='black',prop=font1)
    plt.savefig("./{}.pdf".format(filename))
    plt.close()