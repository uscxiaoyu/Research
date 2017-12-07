#coding=utf-8
from __future__ import division
from random import random
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

def _diffuse_(DG, seeds, beta, gamma):  # beta为s->i的转换概率，gamma为i->r的转换概率
    if not nx.is_directed(DG):
        DG = DG.to_directed() #转化为有向图

    S_list, I_list, R_list = [], [], []  # s,i,r节点容器
    for i in DG:
        DG.node[i]['prede'] = list(DG.predecessors(i))
        DG.node[i]['state'] = 's'  # Susceptible:'s',Infectious:'i',Removed:'r'
        S_list.append(i)
        
    for i in seeds:  # S->I
        DG.node[i]['state'] = 'i'
        S_list.remove(i)
        I_list.append(i)
    
    Num_S, Num_I, Num_R = [len(S_list)], [len(I_list)], [len(R_list)]  # S,I,R初始数量
        
    k = 0
    t1 = time.clock()
    while I_list:
        for i in S_list:
            infect = beta * np.sum([1 for j in DG.node[i]['prede'] if DG.node[j]['state'] == 'i'])
            if infect > random():  # 由s转化为i
                DG.node[i]['state'] = 'i'
                S_list.remove(i)
                I_list.append(i)
                
        for i in I_list:
            if gamma > random():  # 由i转化为r
                DG.node[i]['state'] = 'r'
                I_list.remove(i)
                R_list.append(i)
        
        Num_S.append(len(S_list))
        Num_I.append(len(I_list))
        Num_R.append(len(R_list))
        k += 1

    print('%s步, 时间: %s s'%(k,time.clock()-t1))
    print('S:%s, I:%s, R:%s'%(len(S_list), len(I_list), len(R_list)))
             
    return Num_S,Num_I,Num_R

if __name__ == '__main__':
    beta = 0.1
    gamma = 0.25
    n = 10000
    k = 3
    G = nx.barabasi_albert_graph(n,k) #nx.gnm_random_graph(n,k*n)

    #找出度大节点
    degre_list = []
    for i in G.nodes():
        degre_list.append((nx.degree(G,i),i))

    degre_list = sorted(degre_list,reverse=1)
    seeds = [x[1] for x in degre_list][-5:] #选5个

    diff = _diffuse_(G,seeds,beta,gamma)

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Number of nodes')
    ax.plot(diff[0],'ko-',ms=4,label='S',alpha=0.7)
    ax.plot(diff[1],'rs-',ms=4,label='I',alpha=0.7)
    ax.plot(diff[2],'g^-',ms=4,label='R',alpha=0.7)
    ax.legend(loc='best')
    plt.show()