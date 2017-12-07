from __future__ import division
from random import random,shuffle
import numpy as np
import networkx as nx
import copy
import time
import matplotlib.pyplot as plt

def _syn_diffuse_(DG,phi,seeds,alpha=1):
    DG = DG.to_directed()
    for i in DG.nodes():
        DG[i]['prede'] = DG.predecessors(i)
        if i in seeds:
            DG[i]['state'] = 1
        else:
            DG[i]['state'] = 0
    
    non_adopt_set = [i for i in DG.nodes() if DG[i]['state'] == 0]
    adopt_set = [i for i in DG.nodes() if DG[i]['state'] == 1]
    num_of_adopt = [len(seeds)]
    
    while 1:
        x = 0
        for i in non_adopt_set:                  
            a = len([k for k in DG[i].get('prede',[]) if DG[k]['state'] == 1])
            if a == 0:
                influ = 0
            else:
                influ = a*len(DG[i]['prede'])**(-alpha)
                
            if influ >= phi:
                x = x+1
                non_adopt_set.remove(i)
                adopt_set.append(i)
        
        for i in adopt_set:
            DG[i]['state'] = 1
        
        num_of_adopt.append(x)
        if x == 0:break
            
    return num_of_adopt

def _asyn_diffuse_1(DG,phi,seeds,alpha=1):
    DG = DG.to_directed()
    for i in DG.nodes():
        DG[i]['prede'] = DG.predecessors(i)
        if i in seeds:
            DG[i]['state']=1
        else:
            DG[i]['state']=0
    
    non_adopt_set = [i for i in DG.nodes() if DG[i]['state'] == 0]
    num_of_adopt = [len(seeds)]
    
    while 1:
        x = 0
        shuffle(non_adopt_set)
        for i in non_adopt_set:                  
            a = len([k for k in DG[i].get('prede',[]) if DG[k]['state'] == 1])
            if a == 0:
                influ = 0
            else:
                influ = a*len(DG[i]['prede'])**(-alpha)
                
            if influ >= phi:
                x = x+1
                DG[i]['state'] = 1
                non_adopt_set.remove(i)
        
        num_of_adopt.append(x)
        if x == 0:break
            
    return num_of_adopt

def _asyn_diffuse_2(DG,phi,seeds,alpha=1,reverse=1):
    DG = DG.to_directed() 
    In_degr_centr = nx.in_degree_centrality(DG)
    idn = np.argsort(In_degr_centr.values())
    num_of_nodes = len(idn)
    if reverse == 1:
        idn = [idn[num_of_nodes-i-1] for i in range(num_of_nodes)] #按照度分布由大到小排序
        
    for i in idn:
        DG[i]['prede'] = DG.predecessors(i)
        if i in seeds:
            DG[i]['state'] = 1
        else:
            DG[i]['state'] = 0
    
    non_adopt_set = [i for i in idn if DG[i]['state'] == 0]
    num_of_adopt = [len(seeds)]
    
    while 1:
        x = 0
        for i in non_adopt_set:
            a = len([k for k in DG[i].get('prede',[]) if DG[k]['state'] == 1])
            if a == 0: #there might be nodes without neighbor
                influ = 0         
            else:
                influ = a*len(DG[i]['prede'])**(-alpha)

            if influ >= phi:
                    x = x+1
                    DG[i]['state'] = 1
                    non_adopt_set.remove(i)
        
        num_of_adopt.append(x)       
        if x == 0:break
            
    return num_of_adopt

def parallel_diffuse(DG,phi_range,seeds,num_of_rept = 4):
    f_cont = []
    for phi in phi_range:
        d_cont1,d_cont2,d_cont3,d_cont4 = [],[],[],[]
        for i in range(num_of_rept):
            seeds = np.random.choice(np.arange(n),10)
            diff1 = _syn_diffuse_(DG,phi,seeds,alpha=1)
            diff2 = _asyn_diffuse_1(DG,phi,seeds,alpha=1)
            diff3 = _asyn_diffuse_2(DG,phi,seeds,alpha=1,reverse=1)
            diff4 = _asyn_diffuse_2(DG,phi,seeds,alpha=1,reverse=0)

            d_cont1.append(diff1)
            d_cont2.append(diff2)
            d_cont3.append(diff3)
            d_cont4.append(diff4)
        
        equi_value1 = np.mean(d_cont1)
        equi_value2 = np.mean(d_cont2)
        equi_value3 = np.mean(d_cont3)
        equi_value4 = np.mean(d_cont4)
        
        f_cont.append([phi,equi_value1,equi_value2,equi_value3,equi_value4])
    
    return np.array(f_cont)
#-----------------------------------------------------------------------------
n = 1000
k = 3

DG1 = nx.random_regular_graph(2*k,n)
DG2 = nx.gnm_random_graph(n,k*n)
DG3 = nx.barabasi_albert_graph(n,k)
DG4 = nx.watts_strogatz_graph(n,2*k,p=0)
DG5 = nx.watts_strogatz_graph(n,2*k,p=0.1)
DG6 = nx.watts_strogatz_graph(n,2*k,p=0.5)

phi_range1 = np.arange(0.1,0.3,0.005)
phi_range2 = np.arange(0.1,0.3,0.005)
phi_range3 = np.arange(0.1,0.3,0.005)
phi_range4 = np.arange(0.1,0.3,0.005)
phi_range5 = np.arange(0.1,0.3,0.005)
phi_range6 = np.arange(0.1,0.3,0.005)

DG_set = [DG1,DG2,DG3,DG4,DG5,DG6]
range_set = [phi_range1,phi_range2,phi_range3,phi_range4,phi_range5,phi_range6]
text_set = ['rr3','er3','ba','ws0','ws1','ws5']

for u in range(6):
    time1 = time.clock()
    DG = DG_set[u]
    phi_range = range_set[u]
    text = text_set[u]
    to_write = parallel_diffuse(DG,phi_range,seeds,num_of_rept = 5)
    print "Finish:",text
    print " Time elapsed:",time.clock()-time1,"s"
    np.save('/Users/xiaoyu/Documents/Dissertation/state update rule(simulation)/Threshold(%s)'%text,to_write)

