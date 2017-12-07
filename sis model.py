from __future__ import division
from random import random,shuffle
import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt


def _diffuse_(DG, seeds, gamma, delta):
    DG = DG.to_directed()
    I_list = []
    for i in DG.nodes():
        DG[i]['prede'] = DG.predecessors(i)
        DG[i]['state'] = 0  # Susceptible:0,Infectious:1

    for i in seeds:
        DG[i]['state'] = 1

    k = 0
    while 1:
        S_set = [i for i in DG.nodes() if DG[i]['state'] == 0]
        I_set = [i for i in DG.nodes() if DG[i]['state'] == 1]

        infect_list = [(i, gamma * sum([DG[j]['state'] for j in DG[i]['prede']])) for i in S_set]
        for i, infect in infect_list:
            if infect > random():
                DG[i]['state'] = 1

        for i in I_set:
            if delta > random():
                DG[i]['state'] = 0

        Num_of_I = len([i for i in DG.nodes() if DG[i]['state'] == 1])
        I_list.append(Num_of_I)

        if k >= 150:
            sum_last_1 = np.mean(I_list[-101:-2])
            sum_last_2 = np.mean(I_list[-100:])
            if sum_last_1 >= sum_last_2:
                break

        k = k + 1

    return I_list

if __name__ == '__main__':
    gamma = 0.1
    delta = 0.3
    n = 10000
    k = 3
    DG = nx.gnm_random_graph(n,k*n)

    seeds = np.random.choice(np.arange(1,n+1),10)
    diff = _diffuse_(DG,seeds,gamma,delta)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('T')
    ax.set_ylabel('Number of the infected nodes')
    ax.plot(diff, 'ro-')
    plt.show()