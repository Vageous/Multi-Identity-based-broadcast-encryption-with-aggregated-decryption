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
    font={
        # 'weight':'semibold',
        'size':18
    }
    font1={
         'weight':'semibold',
        'size':10.5
    }
    tick_labels=['20','40','60','80','100']
    xlabels='Number of workers'
    ylabels='Total running times(s)'
    #设置误差标记参数
    error_params1=dict(elinewidth=2,ecolor='#2A9D8C',capsize=5)
    error_params2=dict(elinewidth=2,ecolor='#E9C46B',capsize=5)
    error_params3=dict(elinewidth=2,ecolor='#E66F51',capsize=5)
    bar_width=0.3
    plt.bar(x,mean[0],bar_width,color='#2A9D8C',yerr=[(0,0,0,0,0),std[0]],error_kw=error_params1,label='30% Dropout II')
    plt.bar(x+bar_width,mean[1],bar_width,color='#E9C46B',yerr=[(0,0,0,0,0),std[1]],error_kw=error_params2,label='30% Dropout I')
    plt.bar(x+2*bar_width,mean[2],bar_width,color='#E66F51',yerr=[(0,0,0,0,0),std[2]],error_kw=error_params3,label='No-Dropout')
    plt.xticks(x+bar_width,tick_labels,size=18)
    plt.yticks(np.arange(0,65,10),size=18)
    plt.grid(ls = '-.',axis='y', lw = 0.05)
    plt.xlabel(xlabels,fontdict=font)
    plt.ylabel(ylabels,fontdict=font)
    plt.title("FEMNIST-AlexNet",fontdict=font)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.legend(loc='best',edgecolor='black',prop=font1)
    plt.savefig("./{}.pdf".format(filename))
    plt.close()

def show():
    filepath='./'
    legend1=['No-Droptype','60% Droptype I','60% Droptype II']
    filename1=['0.6droptype2.txt','0.6droptype1.txt','droptype0.txt']
    filename2=['0.2droptype2.txt','0.4droptype2.txt','0.6droptype2.txt']
    mean1,std1=read(filepath,filename1)
    Figure(mean1,std1,legend1,'total_femnist')
    # mean2,std2=read(filepath,filename2)
    # Figure(mean2,std2,legend2,'droptype2_recovery')


if __name__=='__main__':
    show()


