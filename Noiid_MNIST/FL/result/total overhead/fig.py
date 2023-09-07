import os
import numpy as np
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
    #设置横坐标
    x=np.arange(5)
    font_size=16
    tick_labels=['20','40','60','80','100']
    xlabels='Number of traing nodes'
    ylabels='Total running times(s)'
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
    error_params1=dict(elinewidth=2,ecolor='#90C9E7',capsize=5)
    error_params2=dict(elinewidth=2,ecolor='#219EBC',capsize=5)
    error_params3=dict(elinewidth=2,ecolor='#136783',capsize=5)
    bar_width=0.3
    plt.bar(x,mean[0],bar_width,color='#90C9E7',yerr=[(0,0,0,0,0),std[0]],error_kw=error_params1,label='60% Dropout II')
    plt.bar(x+bar_width,mean[1],bar_width,color='#219EBC',yerr=[(0,0,0,0,0),std[1]],error_kw=error_params2,label='60% Dropout I')
    plt.bar(x+2*bar_width,mean[2],bar_width,color='#136783',yerr=[(0,0,0,0,0),std[2]],error_kw=error_params3,label='No-Dropout')
    plt.xticks(x+bar_width,tick_labels,size=12,weight='semibold')
    plt.yticks(np.arange(0,30,5),size=12,weight='semibold')
    plt.grid(ls = '-.',axis='y', lw = 0.05)
    plt.xlabel(xlabels,fontdict=font)
    plt.ylabel(ylabels,fontdict=font)
    plt.title("MNIST-LetNet",fontdict=font)
    plt.legend(loc='best',edgecolor='black',prop=font1)
    plt.savefig("./{}.pdf".format(filename))
    plt.close()

def show():
    filepath='./'
    legend1=['No-Droptype','60% Droptype I','60% Droptype II']
    filename1=['0.6droptype2.txt','0.6droptype1.txt','droptype0.txt']
    filename2=['0.2droptype2.txt','0.4droptype2.txt','0.6droptype2.txt']
    mean1,std1=read(filepath,filename1)
    Figure(mean1,std1,legend1,'total_mnist')
    # mean2,std2=read(filepath,filename2)
    # Figure(mean2,std2,legend2,'droptype2_recovery')


if __name__=='__main__':
    show()


