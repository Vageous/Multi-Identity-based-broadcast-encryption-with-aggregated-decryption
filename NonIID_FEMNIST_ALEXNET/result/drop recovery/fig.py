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
    #设置横坐标
    x=np.arange(5)
    font_size=12
    matplotlib.rc('font', size=12,weight='semibold')
    tick_labels=['20','40','60','80','100']
    xlabels='Number of traing nodes'
    ylabels='Data recovery times(s)'
    #设置误差标记参数
    '''
    #90C9E7
    #219EBC
    #136783
    '''
    '''
    #2A9D8C
    #E9C46B
    #E66F51
    '''
    font={
        'weight':'semibold',
        'size':16
    }
    font1={
        'weight':'semibold',
        'size':14
    }
    error_params1=dict(elinewidth=4,ecolor='#2A9D8C',capsize=5)
    error_params2=dict(elinewidth=4,ecolor='#E9C46B',capsize=5)
    error_params3=dict(elinewidth=4,ecolor='#E66F51',capsize=5)
    bar_width=0.3
    plt.bar(x,mean[0],bar_width,color='#2A9D8C',yerr=[(0,0,0,0,0),std[0]],error_kw=error_params1,label='20% Dropout I')
    plt.bar(x+bar_width,mean[1],bar_width,color='#E9C46B',yerr=[(0,0,0,0,0),std[1]],error_kw=error_params2,label='40% Dropout I')
    plt.bar(x+2*bar_width,mean[2],bar_width,color='#E66F51',yerr=[(0,0,0,0,0),std[2]],error_kw=error_params3,label='60% Dropout I')
    plt.xticks(x+bar_width,tick_labels,size=12,weight='semibold')
    plt.yticks(size=12,weight='semibold')
    ax=plt.gca()
    ax.ticklabel_format(style='scientific',scilimits=(-3,1), axis='y',useMathText=True)
    plt.grid(ls = '-.',axis='y', lw = 0.05)
    plt.title("FEMNIST-AlexNet",fontdict=font)
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
    Figure(mean1,std1,legend1,'recovery_num_femnist')
    # mean2,std2=read(filepath,filename2)
    # Figure(mean2,std2,legend2,'droptype2_recovery')


if __name__=='__main__':
    show()



