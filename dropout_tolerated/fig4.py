# nondropout aggregation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.rc('font',size=14)
# Recovery time num
def data(filepath):
    recovery_mean,recovery_std,recovery_time=[],[],[]
    with open(filepath,'r+') as f:
        lines=f.readlines()
        for line in lines:
            temp=line.strip().split()
            if len(temp)==1:
                recovery_time.append(float(temp[0]))
    new_recovery=np.array(recovery_time).reshape(5,10)
    for i in range(len(new_recovery)):
        recovery_mean.append(np.mean(new_recovery[i]))
        recovery_std.append(np.std(new_recovery[i]))
    return new_recovery,recovery_mean,recovery_std

def fig(mean,std,mean1,std1):
    x=[20,40,60,80,100]
    color = ['#82B0D2','#67C2A3']
    linestyle=['-','--']
    marker=['o', 'v']
    font2={
      'size':18}
    font3={
    # 'weight':'semibold',
      'size':14}
    label=['NonDropout-Ours','NonDropout-SA']
    ylabel='Non-Dropout Aggregation Time(s)'
    xlabel='Number of Workers'
    fig, ax = plt.subplots(1, 1)
    ax.plot(x,mean,linewidth = 2.5, color = color[0],linestyle = linestyle[0],marker=marker[0], label = label[0],markersize=8)
    ax.fill_between(x,mean-std,mean+std,color=color[0],alpha=0.3)
    ax.plot(x,mean1,linewidth = 2.5, color = color[1],linestyle = linestyle[1],marker=marker[1], label = label[1],markersize=8)
    ax.fill_between(x,mean1-std1,mean1+std1,color=color[1],alpha=0.3)

    axins = ax.inset_axes((0.15, 0.4, 0.4, 0.3))
    # for i in range(len(new_mean)):
    #     axins.plot(x,new_mean[i],linewidth = 1.5, color = color[i],linestyle = linestyle[0],marker=marker[i])
    axins.plot(x,mean,linewidth = 2.5, color = color[0],linestyle = linestyle[0],marker=marker[0], label = label[0],markersize=8)
    axins.fill_between(x,mean-std,mean+std,color=color[0],alpha=0.3)
    # 设置放大区间
    zone_left = 0
    zone_right = 4

# 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05  # y轴显示范围的扩展比例
    xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
    y = np.hstack((mean[zone_left:zone_right]))
    ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
    ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    axins.set_xticks([])
    # axins.set_yticks([])
    axins.ticklabel_format(style='scientific',scilimits=(-3,1), axis='y',useMathText=True)
    axins.spines['bottom'].set_color('black')
    axins.spines['top'].set_color('black') 
    # FA7F6F
    axins.spines['right'].set_color('black')
    axins.spines['left'].set_color('black')
    axins.spines['top'].set_linewidth(1.2)
    axins.spines['left'].set_linewidth(1.2)
    axins.spines['bottom'].set_linewidth(1.2)
    axins.spines['right'].set_linewidth(1.2)
    # axins.spines['right'].set_visible(False)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='black', lw=1)


    ax.set_xticks(np.arange(20,110,20))
    ax.set_yticks(np.arange(0,0.5,0.1))
    ax.tick_params('both',labelsize=18)
    ax.ticklabel_format(style='scientific',scilimits=(-3,1), axis='y',useMathText=True)
    # plt.yticks(np.arange(0,,),size=12,weight='semibold')
    ax.set_xlabel(xlabel,fontdict=font2)
    ax.set_ylabel(ylabel,fontdict=font2)
    ax.grid(ls = '-.', lw = 0.1)
    ax.legend(loc='upper center',edgecolor='black',prop=font3,ncol=2)
    plt.subplots_adjust(left=0.13, bottom=0.15)
    plt.savefig('nondropoutaggregation_num.pdf',dpi=600)
    plt.close()



if __name__=='__main__':
    filename='./non_dropout_aggregation_time_num.txt'
    filename1='/home/b1107/user/ct/code/multi-IBE/secureaggregation/non_dropout_aggregation_time_num.txt'
    result1=data(filename)
    # print(result1[1])s
    result2=data(filename1)

    # print(result2[1])
    fig(np.array(result1[1]),np.array(result1[2]),np.array(result2[1]),np.array(result2[2]))

    