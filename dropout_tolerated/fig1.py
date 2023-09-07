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
    new_recovery=np.array(recovery_time).reshape(15,10)
    for i in range(len(new_recovery)):
        recovery_mean.append(np.mean(new_recovery[i]))
        recovery_std.append(np.std(new_recovery[i]))
    return new_recovery,recovery_mean,recovery_std

def fig(mean,std,mean1,std1):
    # plt.figure(figsize=(6, 8))
    new_mean=np.array(mean).reshape(5,3).T
    new_std=np.array(std).reshape(5,3).T
    new_mean1=np.array(mean1).reshape(5,3).T
    new_std1=np.array(std1).reshape(5,3).T
    x=[20,40,60,80,100]
    color = ['#82B0D2','#67C2A3','#FFBE7A']
    linestyle=['-','--']
    marker=['o', 'v','^']
    font2={
      'size':18}
    font3={
    # 'weight':'semibold',
      'size':14}
    label=['10%Dropout-Ours','20%Dropout-Ours','30%Dropout-Ours']
    label1=['10%Dropout-SA','20%Dropout-SA','30%Dropout-SA']
    # plt.title("",fontdict=font2)
    ylabel='Time for Recovery(s)'
    xlabel='Number of Workers'
    # print(new_mean)
    fig, ax = plt.subplots(1, 1)
    for i in range(len(new_mean)):
        ax.plot(x,new_mean[i],linewidth = 2.5, color = color[i],linestyle = linestyle[0],marker=marker[i], label = label[i],markersize=8)
        ax.fill_between(x,new_mean[i]-new_std[i],new_mean[i]+new_std[i],color=color[i],alpha=0.3)
    for i in range(len(new_mean1)):
        ax.plot(x,new_mean1[i],linewidth = 2.5, color = color[i],linestyle = linestyle[1],marker=marker[i],label=label1[i],markersize=8)
        ax.fill_between(x,new_mean1[i]-new_std1[i],new_mean1[i]+new_std1[i],color=color[i],alpha=0.3)

#     axins = inset_axes((0.4, 0.1, 0.4, 0.3))
    axins = ax.inset_axes((0.75, 0.07, 0.24, 0.2))
    for i in range(len(new_mean)):
        axins.plot(x,new_mean[i],linewidth = 2.5, color = color[i],linestyle = linestyle[0],marker=marker[i])
    
    # 设置放大区间
    zone_left = 3
    zone_right = 4

# 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 1  # y轴显示范围的扩展比例
    xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
    y = np.hstack((new_mean[0][zone_left:zone_right], new_mean[1][zone_left:zone_right],new_mean[2][zone_left:zone_right]))
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
    # axins.spines[''].set_color('#FA7F6F')
    axins.spines['top'].set_linewidth(1.2)
    axins.spines['left'].set_linewidth(1.2)
    axins.spines['bottom'].set_linewidth(1.2)
    axins.spines['right'].set_linewidth(1.2)
    # axins.spines['right'].set_visible(False)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='black', lw=1)
  
  
    ax.set_xticks(np.arange(20,110,20))
    ax.set_yticks(np.arange(0,26,5))
    ax.tick_params('both',labelsize=18)
    # plt.yticks(np.arange(0,,),size=12,weight='semibold')
    ax.set_xlabel(xlabel,fontdict=font2)
    ax.set_ylabel(ylabel,fontdict=font2)
    ax.grid(ls = '-.', lw = 0.1)
    # ax.legend(bbox_to_anchor=(1, 0), loc='lower left', borderaxespad=0)
    # ax.legend(bbox_to_anchor=(0, 0), loc=0, borderaxespad=0)
    ax.legend(loc='best',edgecolor='black',prop=font3)
    plt.subplots_adjust(left=0.13, bottom=0.15)
    # plt.imshow(D, aspect='auto')
    # plt.tight_layout()
   
    plt.savefig('recovery_time_num.pdf',dpi=600)
    plt.close()



if __name__=='__main__':
    filename='./recovery_time_num.txt'
    filename1='/home/b1107/user/ct/code/multi-IBE/secureaggregation/recovery_time_num.txt'
    result1=data(filename)
    # print(result1[1])
    result2=data(filename1)

    # print(result2[1])
    fig(result1[1],result1[2],result2[1],result2[2])

    