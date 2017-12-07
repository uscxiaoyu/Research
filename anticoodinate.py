from __future__ import division
from random import random,shuffle
import numpy as np
import networkx as nx
import copy
import time
import matplotlib.pyplot as plt

def _syn_diffuse_1(DG,stg_gain,seeds,r=0.3):
    AA,AB = stg_gain[0][0],stg_gain[0][1]
    BA,BB = stg_gain[1][0],stg_gain[1][1]
    num_of_A = [len(seeds)] 
    DG = DG.to_directed()
    for i in DG.nodes():
        DG[i]['prede'] = DG.predecessors(i)
        DG[i]['neigh'] = len(DG[i]['prede'])
        if i in seeds:
            DG[i]['state'] = 0
        else:
            DG[i]['state'] = 1
            
    k = 0
    while 1:
        A_list = []
        for i in DG.nodes():
            num_B = sum([DG[j]['state'] for j in DG[i].get('prede',[])])
            num_A = DG[i]['neigh'] - num_B
            cal_A = num_A*AA + num_B*AB
            cal_B = num_A*BA + num_B*BB
            if cal_A > cal_B:
                A_list.append(i)
        
        for i in DG.nodes():
            if i in A_list:
                if r > random():
                    DG[i]['state'] = 0
            else:
                if r > random():
                    DG[i]['state'] = 1
                
        num_of_A.append(len(A_list))
        if k >= 5:
            if len(A_list) == num_of_A[-2]:break
            
        k = k+1        
    return num_of_A

def _asyn_diffuse_1(DG,stg_gain,seeds):
    AA,AB = stg_gain[0][0],stg_gain[0][1]
    BA,BB = stg_gain[1][0],stg_gain[1][1]
    
    DG = DG.to_directed()
    n = len(DG)
    for i in DG.nodes():
        if i in seeds:
            DG[i]['state'] = 0
        else:
            DG[i]['state'] = 1
        DG[i]['prede'] = DG.predecessors(i)
        DG[i]['neigh'] = len(DG[i]['prede'])
        
    A_set = [i for i in DG.nodes() if DG[i]['state'] == 0]
    num_of_A = [len(A_set)]
    
    while 1:
        R_list = [i for i in range(n)]
        shuffle(R_list)
        for i in R_list:
            num_B = sum([DG[j]['state'] for j in DG[i].get('prede',[])]) #the number of neighborhoods who took 'B'
            num_A = DG[i]['neigh'] - num_B #the number of neighborhoods who took 'A'
            cal_A = num_A*AA + num_B*AB  #if i take stratege 'A', then 'cal_A' is its gain in this period
            cal_B = num_A*BA + num_B*BB  #if i take stratege 'B', then 'cal_B' is its gain in this period
            if cal_A > cal_B:
                DG[i]['state'] = 0
            else:
                DG[i]['state'] = 1
        
        A_set = [i for i in DG.nodes() if DG[i]['state'] == 0]        
        num_of_A.append(len(A_set))    
        if len(A_set) == num_of_A[-2]:break      
    return num_of_A

def _asyn_diffuse_2(DG,stg_gain,seeds,reverse=1):
    DG = DG.to_directed()
    In_degr_centr = nx.in_degree_centrality(DG)
    idn = np.argsort(In_degr_centr.values())
    num_of_nodes = len(idn)
    if reverse == 1:
        idn = [idn[num_of_nodes-i-1] for i in range(num_of_nodes)] #按照度分布由大到小排序
        
    AA,AB = stg_gain[0][0],stg_gain[0][1]
    BA,BB = stg_gain[1][0],stg_gain[1][1]
    for i in DG.nodes():
        DG[i]['prede'] = DG.predecessors(i)
        DG[i]['neigh'] = len(DG[i]['prede'])
        if i in seeds:
            DG[i]['state'] = 0
        else:
            DG[i]['state'] = 1
            
    A_set = [i for i in DG.nodes() if DG[i]['state'] == 0]
    num_of_A = [len(A_set)]
    print len(seeds),len(A_set)
    
    while 1:
        for i in idn:
            num_B = sum([DG[j]['state'] for j in DG[i].get('prede',[])])
            num_A = DG[i]['neigh'] - num_B
            cal_A = num_A*AA + num_B*AB
            cal_B = num_A*BA + num_B*BB

            if cal_A > cal_B:
                DG[i]['state'] = 0
            else:
                DG[i]['state'] = 1
        
        A_set = [i for i in  DG.nodes() if DG[i]['state'] == 0]        
        num_of_A.append(len(A_set))
        if len(A_set) == num_of_A[-2]:break
            
    return num_of_A

def parallel_diffuse(DG,p_list,seeds,num_of_rept = 4):
    f_cont = []
    for payoff in p_list:
        stg_gain = [[0.1,payoff[0]],
                    [payoff[1],0.1]]
        
        d_cont1,d_cont2,d_cont3,d_cont4 = [],[],[],[]
        for i in range(num_of_rept):
            seeds = np.random.choice(np.arange(n),10)
            diff1 = _syn_diffuse_(DG,stg_gain,seeds)
            diff2 = _asyn_diffuse_1(DG,stg_gain,seeds)
            diff3 = _asyn_diffuse_2(DG,stg_gain,seeds,reverse=1)
            diff4 = _asyn_diffuse_2(DG,stg_gain,seeds,reverse=0)

            d_cont1.append(diff1[-1])
            d_cont2.append(diff2[-1])
            d_cont3.append(diff1[-1])
            d_cont4.append(diff1[-1])
        
        equi_value1 = np.mean(d_cont1)
        equi_value2 = np.mean(d_cont2)
        equi_value3 = np.mean(d_cont3)
        equi_value4 = np.mean(d_cont4)
        
        f_cont.append([delta,equi_value1,equi_value2,equi_value3,equi_value4])
    
    return np.array(f_cont)

#------------------------------------------------------------------------------
if __name__ == '__main__':
    a = np.arange(0.2,2,0.1)
    b = np.arange(0.2,2,0.1)
    p_list = [(i,j) for i in a for j in b]

    n = 1000
    k = 3
    gamma = 0.1

    DG1 = nx.random_regular_graph(2*k,n)
    DG2 = nx.gnm_random_graph(n,k*n)
    DG3 = nx.barabasi_albert_graph(n,k)
    DG4 = nx.watts_strogatz_graph(n,2*k,p=0)
    DG5 = nx.watts_strogatz_graph(n,2*k,p=0.1)
    DG6 = nx.watts_strogatz_graph(n,2*k,p=0.5)

    DG_set = [DG1,DG2,DG3,DG4,DG5,DG6]
    range_set = [delta_range1,delta_range2,delta_range3,delta_range4,delta_range5,delta_range6]
    tex_set = ['rr','er','ba','ws0','ws1','ws5']

    for u in range(6):
        time1 = time.clock()
        DG = DG_set[u]
        delta_range = range_set[u]
        text = text_set[u]
        to_write = parallel_diffuse(DG,p_list,seeds,num_of_rept = 4)
        print "Finish:",text
        print " Time elapsed:",time.clock()-time1,"s"
        np.save('/Users/xiaoyu/Documents/Dissertation/state update rule (simulation)/Anti(%s)'%text,to_write)
