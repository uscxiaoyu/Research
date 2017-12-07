from __future__ import division
from random import random,shuffle
import numpy as np
import networkx as nx
import pandas as pd
import copy
import time
import matplotlib.pyplot as plt

def _syn_diffuse_(DG,seeds,gamma,delta):
    DG = DG.to_directed()
    I_list = []
    for i in DG.nodes():
        DG[i]['prede'] = DG.predecessors(i)
        DG[i]['state'] = 0                                 #Susceptible:0,Infectious:1
        
    for i in seeds:
        DG[i]['state'] = 1
    
    k = 0
    while 1:
        S_set = [i for i in DG.nodes() if DG[i]['state'] == 0]
        I_set = [i for i in DG.nodes() if DG[i]['state'] == 1]
        
        infect_list = [(i,gamma*sum([DG[j]['state'] for j in DG[i]['prede']])) for i in S_set] 
        for i,infect in infect_list:
            if infect > random():
                DG[i]['state'] = 1
        
        for i in I_set:
            if delta > random():
                DG[i]['state'] = 0
        
        Num_of_I = len([i for i in DG.nodes() if DG[i]['state']==1])
        I_list.append(Num_of_I)

        if k >= 150:
            sum_last_1 = np.mean(I_list[-101:-2])
            sum_last_2 = np.mean(I_list[-100:])
            if sum_last_1 >= sum_last_2:
                break
            
        k = k+1
            
    return I_list

def _asyn_diffuse_1(DG,seeds,gamma,delta):
    DG = DG.to_directed()
    DG_nodes = DG.nodes()
    I_list = []
    
    for i in DG.nodes():
        DG[i]['prede'] = DG.predecessors(i)
        DG[i]['state'] = 0                                 #Susceptible:0,Infectious:1
        
    for i in seeds:
        DG[i]['state'] = 1
    
    k = 0
    while 1:
        shuffle(DG_nodes)
        for i in DG_nodes:
            if DG[i]['state'] == 0:
                infect_hazard = sum([DG[j]['state'] for j in DG[i]['prede']])*gamma
                if infect_hazard > random():
                    DG[i]['state'] = 1
                        
            if DG[i]['state'] == 1:
                if delta > random():
                    DG[i]['state'] = 0
        
        Num_of_I = len([i for i in DG.nodes() if DG[i]['state']==1])
        I_list.append(Num_of_I)
        
        if k >= 150:
            sum_last_1 = np.mean(I_list[-101:-2])
            sum_last_2 = np.mean(I_list[-100:])
            if sum_last_1 >= sum_last_2:
                break
            
        k = k+1
            
    return I_list

def _asyn_diffuse_2(DG,seeds,gamma,delta,reverse=1):
    DG = DG.to_directed()
    In_degr_centr = nx.in_degree_centrality(DG)
    idn = np.argsort(In_degr_centr.values())
    num_of_nodes = len(idn)
    if reverse == 1:
        idn = [idn[num_of_nodes-i-1] for i in range(num_of_nodes)] #按照度分布由大到小排序
    
    I_list = []
    for i in DG.nodes():
        DG[i]['prede'] = DG.predecessors(i)
        DG[i]['state'] = 0                                 #Susceptible:0,Infectious:1
        
    for i in seeds:
        DG[i]['state'] = 1
    
    k = 0
    while 1:
        for i in idn:
            if DG[i]['state'] == 0:
                infect_hazard = sum([DG[j]['state'] for j in DG[i]['prede']])*gamma
                if infect_hazard > random():
                    DG[i]['state'] = 1
                        
            if DG[i]['state'] == 1:
                if delta > random():
                    DG[i]['state'] = 0
        
        Num_of_I = len([i for i in DG.nodes() if DG[i]['state']==1])
        I_list.append(Num_of_I)
        
        if k >= 150:
            sum_last_1 = np.mean(I_list[-101:-2])
            sum_last_2 = np.mean(I_list[-100:])
            if sum_last_1 >= sum_last_2:
                break
            
        k = k+1
            
    return I_list

def parallel_diffuse(DG,delta_range,gamma=0.1,num_of_rept = 4):
    f_cont = []
    for delta in delta_range:
        d_cont1,d_cont2,d_cont3,d_cont4 = [],[],[],[]
        for i in range(num_of_rept):
            seeds = np.random.choice(np.arange(n),10)
            diff1 = _syn_diffuse_(DG,seeds,gamma,delta)
            diff2 = _asyn_diffuse_1(DG,seeds,gamma,delta)
            diff3 = _asyn_diffuse_2(DG,seeds,gamma,delta,reverse=1)
            diff4 = _asyn_diffuse_2(DG,seeds,gamma,delta,reverse=0)

            d_cont1.append(np.mean(diff1[-100:]))
            d_cont2.append(np.mean(diff2[-100:]))
            d_cont3.append(np.mean(diff3[-100:]))
            d_cont4.append(np.mean(diff4[-100:]))
        
        equi_value1 = np.mean(d_cont1)
        equi_value2 = np.mean(d_cont2)
        equi_value3 = np.mean(d_cont3)
        equi_value4 = np.mean(d_cont4)
        
        f_cont.append([delta,equi_value1,equi_value2,equi_value3,equi_value4])
    
    return np.array(f_cont)


#---------------------------------------------------------------------------------
n = 1000
k = 3
gamma = 0.1

DG1 = nx.random_regular_graph(2*k,n)
DG2 = nx.gnm_random_graph(n,k*n)
DG3 = nx.barabasi_albert_graph(n,k)
DG4 = nx.watts_strogatz_graph(n,2*k,p=0)
DG5 = nx.watts_strogatz_graph(n,2*k,p=0.1)
DG6 = nx.watts_strogatz_graph(n,2*k,p=0.5)

delta_range1 = np.arange(0.001,0.7,0.01)
delta_range2 = np.arange(0.001,1,0.01)
delta_range3 = np.arange(0.001,1,0.01)
delta_range4 = np.arange(0.001,0.6,0.01)
delta_range5 = np.arange(0.001,0.6,0.01)
delta_range6 = np.arange(0.001,0.6,0.01)

DG_set = [DG1,DG2,DG3,DG4,DG5,DG6]
range_set = [delta_range1,delta_range2,delta_range3,delta_range4,delta_range5,delta_range6]
text_set = ['rr3','er3','ba','ws0','ws1','ws5']

for u in range(6):
    time1 = time.clock()
    DG = DG_set[u]
    delta_range = range_set[u]
    text = text_set[u]
    to_write = parallel_diffuse(DG,delta_range,gamma=0.1,num_of_rept = 4)
    print "Finish:",text
    print " Time elapsed:",time.clock()-time1,"s"
    np.save('/Users/xiaoyu/Documents/Dissertation/state update rule (simulation)/SIS(%s)'%text,to_write)
