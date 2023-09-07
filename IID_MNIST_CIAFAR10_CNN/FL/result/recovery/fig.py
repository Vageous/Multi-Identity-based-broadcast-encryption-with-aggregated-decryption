import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def loaddata(filepath,filename):
    recovery_mean,recovery_std,recovery_time=[],[],[]
    filepath=os.path.join(filepath,filename)
    with open(filepath) as f:
        lines=f.readlines()
        for line in lines:
            temp=line.strip().split()
            recovery_time.append(float(temp[0]))
    new_recovery=np.array(recovery_time).reshape(5,10)
    for i in range(len(new_recovery)):
        recovery_mean.append(np.mean(new_recovery[i]))
        recovery_std.append(np.std(new_recovery[i]))

    return recovery_mean,recovery_std

def read(filepath,filename):
    mean,std=[],[]
    for i in range(len(filename)):
        mean.append(loaddata(filepath,filename[i])[0])
        std.append(loaddata(filepath,filename[i])[1])
    return mean,std

def Figure(mean,std,legend,filename):
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

def show():
    filepath='./'
    legend1=['20% Droptype I','40% Droptype I','60% Droptype I']
    legend2=['20% Droptype II','40% Droptype II','60% Droptype II']
    filename1=['0.2droptype1.txt','0.4droptype1.txt','0.6droptype1.txt']
    filename2=['0.2droptype2.txt','0.4droptype2.txt','0.6droptype2.txt']
    mean1,std1=read(filepath,filename1)
    Figure(mean1,std1,legend1,'recovery_gradients_mnist')
    # mean2,std2=read(filepath,filename2)
    # Figure(mean2,std2,legend2,'droptype2_recovery')


if __name__=='__main__':
    show()


