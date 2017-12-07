# coding: utf-8
from __future__ import division
import networkx as nx
import numpy as np
import time

def _diffuse_1(dg, p, q, num_of_run=40):  # 计算耗时与网络规模呈线性关系
    '''
    :param dg: nx网络
    :param p: 外部影响系数
    :param q: 内部影响系数
    :param num_of_run: 运行步数
    :return: 非累计采纳率
    '''
    if not nx.is_directed(G):  # 如果不是有向图，则转化为有向图
        dg = dg.to_directed()
    non_adopt_set = np.array(dg, dtype=np.int)
    for i in dg.__iter__():
        dg.node[i]['state'] = False
    num_of_adopt = []
    for j in range(num_of_run):
        inf_set = np.array([np.sum([dg.node[k]['state'] for k in dg.predecessors(i)]) for i in non_adopt_set])
        upd_set = np.random.random(non_adopt_set.size) <= p + q * inf_set  # 决策结果
        for i in non_adopt_set[upd_set]:  # 更新节点状态
            dg.node[i]['state'] = True
        num_of_adopt.append(np.sum(upd_set))  # 添加该时间步加的采纳者数量
        non_adopt_set = non_adopt_set[np.logical_not(upd_set)]  # 更新未采纳者集合
    return num_of_adopt


def _diffuse_2(G, p, q, num_of_run=40):
    '''
    :param G: nx网络
    :param p: 外部影响系数
    :param q: 内部影响系数
    :param num_of_run: 运行步数
    :return: 非累计采纳率
    '''
    DG = G.to_directed()
    non_adopt_set = []
    for i in DG.__iter__():
        DG.node[i]['state'] = False
        non_adopt_set.append(i)

    num_of_adopt = []
    for j in range(num_of_run):
        x = 0
        np.random.shuffle(non_adopt_set)
        for i in non_adopt_set:
            inf = sum([DG.node[k]['state'] for k in DG.predecessors(i)])
            if np.random.random() <= p + q * inf:
                DG.node[i]['state'] = True
                non_adopt_set.remove(i)
                x += 1

        num_of_adopt.append(x)


    return num_of_adopt

def _diffuse_3(G, p, q, num_of_run=40):
    '''
    :param G: nx网络
    :param p: 外部影响系数
    :param q: 内部影响系数
    :param num_of_run: 运行步数
    :return: 非累计采纳率
    '''
    DG = G.to_directed()
    non_adopt_set = []
    for i in DG.__iter__():
        DG.node[i]['state'] = False
        non_adopt_set.append(i)

    num_of_adopt = []
    for j in range(num_of_run):
        c = 0
        inf_cont = [np.sum([DG.node[k]['state'] for k in DG.predecessors(x)]) for x in non_adopt_set]
        for i, x in enumerate(non_adopt_set):
            if np.random.random() <= p + q * inf_cont[i]:
                DG.node[i]['state'] = True
                non_adopt_set.remove(x)
                c += 1

        num_of_adopt.append(c)

    return num_of_adopt

if __name__ == '__main__':
    G = nx.barabasi_albert_graph(100000, 3)
    p, q = 0.01, 0.1

    t1 = time.clock()
    diff_1 = _diffuse_1(G, p, q)
    print u'耗时%.1fs'%(time.clock() - t1)

    t1 = time.clock()
    diff_2 = _diffuse_2(G, p, q)
    print u'耗时%.1fs'%(time.clock() - t1)

    t1 = time.clock()
    diff_3 = _diffuse_3(G, p, q)
    print u'耗时%.1fs' % (time.clock() - t1)


    '''
    n = 10000
    k = 3
    p_cont = [i * 0.005 + 0.001 for i in range(5)]
    # p_cont = [i*0.00293+0.0007 for i in range(11)]
    q_cont = np.linspace(0.055, 0.127, 20)

    data_cont = []
    for p in p_cont:
        time1 = time.clock()
        for q in q_cont:
            s_estim = []
            for i in range(10):
                DG = nx.watts_strogatz_graph(n, 2 * k, p=1)
                diffuse = _diffuse_(DG, p, q, num_of_run=30)
                s_estim.append(diffuse)

            s_estim = np.array(s_estim)
            s_estim_avr = np.mean(s_estim, axis=0)
            data_cont.append(np.concatenate(([p, q], s_estim_avr)))
        print 'p:', p,
        print 'Time elasped:', time.clock() - time1

    np.save("/Users/xiaoyu/Documents/Bass model estimate AB model/2 degree heterogenous/%s" % DG.name, data_cont)
    '''