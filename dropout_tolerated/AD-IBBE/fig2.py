import numpy as np
import matplotlib.pyplot as plt


plt.rc('font',size=14)
def data(filepath):
    recovery_mean,recovery_std,recovery_time=[],[],[]
    with open(filepath,'r+') as f:
        lines=f.readlines()
        for line in lines:
            temp=line.strip().split()
            if len(temp)==1:
                recovery_time.append(float(temp[0]))
    new_recovery=np.array(recovery_time).reshape(20,10)
    for i in range(len(new_recovery)):
        recovery_mean.append(np.mean(new_recovery[i]))
        recovery_std.append(np.std(new_recovery[i]))
    return new_recovery,recovery_mean,recovery_std


def fig(mean,std):
    new_mean=np.array(mean).reshape(5,4).T
    new_std=np.array(std).reshape(5,4).T
    # new_mean1=np.array(mean1).reshape(5,4).T
    # new_std1=np.array(std1).reshape(5,4).T
    x=[20,40,60,80,100]
    color = ['#4E62AB','#87CFA4','#FDB96A','#469EB4']
    linestyle=['-','-','-','-']
    marker=['o', '*','x','v']
    font2={
      'size':18}
    font3={
    # 'weight':'semibold',
      'size':15}
    label=['NonDropout','10%Dropout','20%Dropout','30%Dropout']
    # label1=['10%Dropout-SA','20%Dropout-SA','30%Dropout-SA']
    # plt.title("",fontdict=font2)
    ylabel='Aggregation Time(s)'
    xlabel='Number of Workers'
    # print(new_mean)
    fig, ax = plt.subplots(1, 1)
    for i in range(len(new_mean)):
        ax.plot(x,new_mean[i],linewidth = 2.5, color = color[i],linestyle = linestyle[i],marker=marker[i], label = label[i],markersize=8)
        ax.fill_between(x,new_mean[i]-new_std[i],new_mean[i]+new_std[i],color=color[i],alpha=0.3)
    # for i in range(len(new_mean1)):
    #     ax.plot(x,new_mean1[i],linewidth = 1.5, color = color[i],linestyle = linestyle[1],marker=marker[i],label=label1[i])
    #     ax.fill_between(x,new_mean1[i]-new_std1[i],new_mean1[i]+new_std1[i],color=color[i],alpha=0.3)

    ax.set_xticks(np.arange(20,110,20))
    ax.set_yticks(np.arange(0,0.11,0.02))
    # plt.yticks(np.arange(0,,),size=12,weight='semibold')
    ax.tick_params('both',labelsize=18)
    ax.ticklabel_format(style='scientific',scilimits=(-1,1), axis='y',useMathText=True)
    ax.set_xlabel(xlabel,fontdict=font2)
    ax.set_ylabel(ylabel,fontdict=font2)
    
    ax.grid(ls = '-.', lw = 0.1)
    ax.legend(loc='best',edgecolor='black',prop=font3)
    # ax.legend(bbox_to_anchor=(0, 0), loc=0, borderaxespad=0)
    plt.subplots_adjust(left=0.18, bottom=0.15)
    plt.savefig('agg.pdf',dpi=600)
    plt.close()


if __name__=='__main__':
    filepath='./agg.txt'
    result=data(filepath)
    fig(result[1],result[2])