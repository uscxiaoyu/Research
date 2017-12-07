from __future__ import division
from random import random,shuffle
import numpy as np
import networkx as nx
import time

def _syn_diffuse_(DG,p,q,alpha=1):
    DG = DG.to_directed()
    for i in DG.nodes():
        DG.node[i]['state'] = 0
        DG.node[i]['prede'] = DG.predecessors(i)
        DG.node[i]['neigh'] = len(DG[i]['prede'])
      
    non_adopt_set = [i for i in DG.nodes()]
    num_of_adopt = []
           
    while 1:                                 
        x = 0
        adopt_set = []
        for i in non_adopt_set:
            if DG.node[i]['neigh'] == 0:
                influ = 0
            else:
                influ = len([k for k in DG.node[i].get('prede',[]) if DG.node[k]['state'] == 1])*DG.node[i]['neigh']**(-alpha)
            prob = p+q*influ
            if random() <= prob:
                adopt_set.append(i)
                non_adopt_set.remove(i)
                x = x+1
                
        num_of_adopt.append(x)       
        for i in adopt_set:
            DG.node[i]['state'] = 1
        
        if sum(num_of_adopt) >= 0.99*n:
            break
            
    return num_of_adopt

def _asyn_diffuse_1(DG,p,q,alpha=1):
    DG = DG.to_directed()       
    for i in DG.nodes():
        DG.node[i]['state'] = 0
        DG.node[i]['prede'] = DG.predecessors(i)
        DG.node[i]['neigh'] = len(DG.node[i]['prede'])
      
    non_adopt_set = [i for i in DG.nodes()]
    num_of_adopt = []
          
    while 1:                                 
        x = 0
        shuffle(non_adopt_set)
        for i in non_adopt_set:
            if DG.node[i]['neigh'] == 0:
                influ = 0
            else:
                influ = len([k for k in DG.node[i].get('prede',[]) if DG.node[k]['state'] == 1])*DG.node[i]['neigh']**(-alpha)
                
            prob = p+q*influ
            if random() <= prob:
                DG.node[i]['state'] = 1
                non_adopt_set.remove(i)
                x = x+1                
        num_of_adopt.append(x)
        
        if sum(num_of_adopt) >= 0.99*n:
            break
            
    return num_of_adopt

def _asyn_diffuse_2(DG,p,q,alpha=1,reverse=1):
    DG = DG.to_directed() 
    In_degr_centr = nx.in_degree_centrality(DG)
    idn = np.argsort(In_degr_centr.values())
    num_of_nodes = len(idn)
    if reverse == 1:
        idn = [idn[num_of_nodes-i-1] for i in range(num_of_nodes)] #按照度分布由大到小排序
        
    for i in DG.nodes():
        DG.node[i]['state'] = 0
        DG.node[i]['prede'] = DG.predecessors(i)
        DG.node[i]['neigh'] = len(DG.node[i]['prede'])
      
    non_adopt_set = [i for i in idn]
    num_of_adopt = []
          
    while 1:
        x = 0
        for i in non_adopt_set:
            if DG.node[i]['neigh'] == 0:
                influ = 0
            else:
                influ = len([k for k in DG.node[i].get('prede',[]) if DG.node[k]['state'] == 1])*DG.node[i]['neigh']**(-alpha)
            
            prob = p + q*influ
            if random() <= prob:
                DG.node[i]['state'] = 1
                non_adopt_set.remove(i)
                x = x+1  

        num_of_adopt.append(x)
        if sum(num_of_adopt) >= 0.99*n:
            break
            
    return num_of_adopt

def parallel_diffuse(DG,pq_range,num_of_rept = 4):
    f_cont = []
    for p,q in pq_range:
        d_cont1,d_cont2,d_cont3,d_cont4 = [],[],[],[]
        for i in range(num_of_rept):
            diff1 = _syn_diffuse_(DG,p,q,alpha=1)
            diff2 = _asyn_diffuse_1(DG,p,q,alpha=1)
            diff3 = _asyn_diffuse_2(DG,p,q,alpha=1,reverse=1)
            diff4 = _asyn_diffuse_2(DG,p,q,alpha=1,reverse=0)

            d_cont1.append([len(diff1),np.max(diff1),np.argmax(diff1)])
            d_cont2.append([len(diff2),np.max(diff2),np.argmax(diff2)])
            d_cont3.append([len(diff3),np.max(diff3),np.argmax(diff3)])
            d_cont4.append([len(diff4),np.max(diff4),np.argmax(diff4)])
        
        equi_value1 = np.mean(d_cont1,axis=0)
        equi_value2 = np.mean(d_cont2,axis=0)
        equi_value3 = np.mean(d_cont3,axis=0)
        equi_value4 = np.mean(d_cont4,axis=0)
        
        f_cont.append([[p,q],list(equi_value1),list(equi_value2),list(equi_value3),list(equi_value4)])
    
    return np.array(f_cont)

#-------------------------------------------------------------------------------
n = 1000
k = 3

p_range = np.linspace(0.0007,0.03,20)
q_range = np.linspace(0.38,0.53,25)
pq_range = [(i,j) for i in p_range for j in q_range]

DG_set = [DG1,DG2,DG3,DG4,DG5,DG6]
text_set = ['er3','er6','ba','ws0','ws1','ws5']

for u in range(6):
    time1 = time.clock()
    DG = DG_set[u]
    text = text_set[u]
    to_write = parallel_diffuse(DG,pq_range,num_of_rept = 4)
    print "Finish:",text
    print " Time elapsed:",time.clock()-time1,"s"
    np.save('/Users/xiaoyu/Documents/Dissertation/state update rule(simulation)/Bass(%s)'%text,to_write)

