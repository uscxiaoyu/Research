# coding: utf-8
from __future__ import division
import sys
sys.path.append('/Users/xiaoyu/Documents')
from bassestima import *
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import networkx as nx
from random import shuffle

# 扩散函数
def _diffuse_(DG, p, q, num_of_run=25):
    if p < 0:
        p == 1e-5
    DG = DG.to_directed()       
    for i in DG.nodes():
        DG[i]['state'] = 0
        DG[i]['prede'] = DG.predecessors(i)
      
    non_adopt_set = [i for i in DG.nodes()]
    num_of_adopt = []
    
    j = 1       
    while 1:                                 
        x = 0
        shuffle(non_adopt_set)
        for i in non_adopt_set:
            influ = len([k for k in DG[i].get('prede',[]) if DG[k]['state'] == 1])           
            prob = p + q*influ
            if np.random.random() <= prob:
                DG[i]['state'] = 1
                non_adopt_set.remove(i)
                x = x+1                
        num_of_adopt.append(x)
        
        j = j+1
        if j > num_of_run:
            break
            
    return num_of_adopt

# R2
def R_sigma(f_actual,s_actual):
    f_actual = np.array(f_actual)
    s_actual = np.array(s_actual)
    sse = np.sum(np.square(s_actual-f_actual))
    ave_y = np.mean(s_actual)
    ssl = np.sum(np.square(s_actual-ave_y))
    R_2 = (ssl-sse)/ssl
    return R_2

# 确定扩散率曲线对应的最优潜在市场规模(与现实数据拟合)
def search_M(M,s_act,x):
    indiv_diff = x[2:]
    sse = 0
    for i in range(len(s_act)):
        sse = (s_act[i]-indiv_diff[i]*M)**2 + sse
    return sse/len(s_act)


# 获取对应扩散率曲线的最优潜在市场容量
def get_solution(y,s_act,rpt_times,nwk,num_of_run):
    s_estim = []
    x0 = 10000
    for i in range(rpt_times):           
        DG = nwk
        diffuse = _diffuse_(DG,y[0],y[1],num_of_run)
        s_estim.append(diffuse)

    s_estim = np.array(s_estim)
    s_estim_avr = np.mean(s_estim,axis=0)

    x = np.concatenate(([y[0],y[1]],s_estim_avr))
    res = minimize(search_M, x0, args = (s_act,x), method='Nelder-Mead')
    s_M = res.x[0]
    M_sse = res.fun
    return M_sse,y[0],y[1],s_M,s_estim_avr


# 生成初始搜索点(p0,q0)
def generate_p0_q0(BM_solution,s_act,nwk,k,rpt_times):
    global last_diff_curves,to_fit
    num_of_run = len(s_act)
    BM_diff = estimation(BM_solution,num_of_run)
    midx = np.argmax(BM_diff)
    delta = num_of_run - midx - 1    #delta steps after peak time
    para_range = [[1e-5,0.2],[1e-3,1],[0,30000]]  
    bass_P,bass_Q,bass_M = BM_solution
    
    if len(last_diff_curves) == 0:
        p_range = np.linspace(0.4*bass_P,bass_P,num=3)
        q_range = np.linspace(0.2*bass_Q/k,0.6*bass_Q/k,num=3)
        params_cont = []       
        for p in p_range:
            for q in q_range:
                s_estim = []               
                for i in range(rpt_times):           
                    DG = nwk
                    diffuse = _diffuse_(DG,p,q,num_of_run)
                    s_estim.append(diffuse)

                s_estim = np.array(s_estim)
                s_estim_avr = np.mean(s_estim,axis=0)
                idx = np.argmax(s_estim_avr) + delta + 1 #(delta+1) steps after peak time

                Eparams,Esse,R2 = fit_it(s_estim_avr[:idx],para_range,t_n= 500,c_n= 100,s_threshold=1e-8,repetes=1)
                F_solution = [Eparams.Parameters[i] for i in range(3)]
                params_cont.append([[p,q],F_solution])
            
        to_fit['p'] = [x[0][0] for x in params_cont]
        to_fit['q'] = [x[0][1] for x in params_cont]
        to_fit['P'] = [x[1][0] for x in params_cont]
        to_fit['Q'] = [x[1][1] for x in params_cont]
        to_fit['M'] = [x[1][2] for x in params_cont]
        to_fit['m_M'] = [10000/x[1][2] for x in params_cont]

        result_p = smf.ols('p ~ P + Q - 1', data = to_fit).fit()
        param_p_P = result_p.params['P']
        param_p_Q = result_p.params['Q']
        mse_p = np.mean(np.square(np.array(to_fit['p']) - (np.array(to_fit['P'])*param_p_P + np.array(to_fit['Q'])*param_p_Q))) 

        result_q = smf.ols('q ~ P + Q - 1', data = to_fit).fit()
        param_q_P = result_p.params['P']
        param_q_Q = result_q.params['Q']
        mse_q = np.mean(np.square(np.array(to_fit['q']) - (np.array(to_fit['Q'])*param_q_Q + np.array(to_fit['P'])*param_q_P)))

        result_m = smf.ols('m_M ~ P + Q - 1', data = to_fit).fit()
        param_m_P,param_m_Q = result_m.params['P'],result_m.params['Q']
        mse_m = np.mean(np.square(np.array(to_fit['m_M']) - 
                    (np.array(to_fit['P'])*param_m_P + np.array(to_fit['Q'])*param_m_Q)))

        p0 = result_p.params['P']*bass_P + result_p.params['Q']*bass_Q
        q0 = result_q.params['Q']*bass_Q + result_q.params['P']*bass_P

        D_matrix = [[param_p_P,param_p_Q,mse_p],
                    [param_q_P,param_q_Q,mse_q],
                    [param_m_P,param_m_Q,mse_m]]
        
        print 'Initial points:',p0,q0
        
    else:
        p_cont,q_cont,P_cont,Q_cont,M_cont,m_M_cont = [],[],[],[],[],[]
        for x in last_diff_curves:
            curve_x = x[1]
            s_exp = repete_estim(curve_x,para_range,t_n=500,c_n=100,s_threshold=1e-8,repetes=1)
            p_cont.append(x[0][0]);q_cont.append(x[0][1])
            P_cont.append(s_exp[1]);Q_cont.append(s_exp[2])
            M_cont.append(s_exp[3]);m_M_cont.append(10000/s_exp[3])
            
        to_add = pd.DataFrame({'p':p_cont,'q':q_cont,'P':P_cont,'Q':Q_cont,'M':M_cont,'m_M':m_M_cont})
        to_fit = to_fit.append(to_add,ignore_index=True)
        
        print 'Length of to_fit:',len(to_fit)

        result_p = smf.ols('p ~ P + Q - 1', data = to_fit).fit()
        param_p_P = result_p.params['P']
        param_p_Q = result_p.params['Q']
        mse_p = np.mean(np.square(np.array(to_fit['p']) - (np.array(to_fit['P'])*param_p_P + np.array(to_fit['Q'])*param_p_Q))) 

        result_q = smf.ols('q ~ P + Q - 1', data = to_fit).fit()
        param_q_P = result_q.params['P']
        param_q_Q = result_q.params['Q']
        mse_q = np.mean(np.square(np.array(to_fit['q']) - (np.array(to_fit['Q'])*param_q_Q + np.array(to_fit['P'])*param_q_P)))

        result_m = smf.ols('m_M ~ P + Q - 1', data = to_fit).fit()
        param_m_P,param_m_Q = result_m.params['P'],result_m.params['Q']
        mse_m = np.mean(np.square(np.array(to_fit['m_M']) - 
                    (np.array(to_fit['P'])*param_m_P + np.array(to_fit['Q'])*param_m_Q)))

        p0 = result_p.params['P']*bass_P + result_p.params['Q']*bass_Q
        q0 = result_q.params['Q']*bass_Q + result_q.params['P']*bass_P

        D_matrix = [[param_p_P,param_p_Q,mse_p],
                    [param_q_P,param_q_Q,mse_q],
                    [param_m_P,param_m_Q,mse_m]]
        
        print 'p0 and q0:',p0,q0
        last_diff_curves = [] #delete diffusion curves produced in last run
        
    return p0,q0,D_matrix

# 寻找最优解--第一阶段（粗选）
def search_opt_solution(p0,q0,s_act,num_of_run,nwk,intv_p,intv_q,rpt_times,num_of_cond=2):
    org_p,org_q = p0,q0
    global diff_cont,diff_cont2,last_diff_curves
    diff_cont = []
    solution_cont = []

    p_q_cont = [(org_p-intv_p,org_q-intv_q),(org_p,org_q-intv_q),(org_p+intv_p,org_q+intv_q),
                (org_p-intv_p,org_q),       (org_p,org_q),       (org_p+intv_p,org_q),
                (org_p-intv_p,org_q+intv_p),(org_p,org_q+intv_p),(org_p+intv_p,org_q+intv_p)]

    for y in p_q_cont:
        solution = get_solution(y,s_act,rpt_times,nwk,num_of_run)
        solution_cont.append(solution[:4])
        diff_cont.append(solution[4])
        last_diff_curves.append([y,solution[4]])

    best_solution = sorted(solution_cont)[:num_of_cond]
    while 1:
        p_q_cont2 = []
        solution_cont2 = []
        diff_cont2 = []        
        for z in best_solution:
            temp = [(z[1]-intv_p,z[2]-intv_q),(z[1],z[2]-intv_q),(z[1]+intv_p,z[2]-intv_q),
                    (z[1]-intv_p,z[2]),       (z[1],z[2]),       (z[1]+intv_p,z[2]),
                    (z[1]-intv_p,z[2]+intv_q),(z[1],z[2]+intv_q),(z[1]+intv_p,z[2]+intv_q)]
            p_q_cont2.extend(temp)
            
        p_q_cont2 = list(set(p_q_cont2 + p_q_cont))
        for y in p_q_cont2:
            if y in p_q_cont:
                solution_cont2.append(solution_cont[p_q_cont.index(y)])
                diff_cont2.append(diff_cont[p_q_cont.index(y)])
            else:
                time1 = time.clock()
                solution = get_solution(y,s_act,rpt_times,nwk,num_of_run)
                solution_cont2.append(solution[:4])
                diff_cont2.append(solution[4])
                last_diff_curves.append([y,solution[4]]) #collect all curves and its parameters produced in estimation

        best_solution = sorted(solution_cont2)[:num_of_cond]
        opt_solution = best_solution[0]
        opt_curve = diff_cont2[solution_cont2.index(best_solution[0])]

        if len(p_q_cont2) == len(p_q_cont):
            break
        else:
            solution_cont = solution_cont2
            diff_cont = diff_cont2
            p_q_cont = p_q_cont2

    f_act = opt_solution[-1]*opt_curve
    R2 = R_sigma(f_act,s_act)
    search_times = len(p_q_cont)
    
    return opt_solution[1:],f_act,R2,search_times,p_q_cont

# 寻找最优解--第二阶段（精选）
def refined_search(p1,q1,intv_p,intv_q,s_act,rpt_times,nwk):
    diff_cont,solution_cont = [],[]
    org_p,org_q = p1,q1
    n_of_run = len(s_act)

    p_q_cont = [(org_p-intv_p/2,org_q-intv_q/2),(org_p,org_q-intv_q/2),(org_p+intv_p/2,org_q-intv_q/2),
                (org_p-intv_p/2,org_q),         (org_p,org_q),         (org_p+intv_p/2,org_q),
                (org_p-intv_p/2,org_q+intv_q/2),(org_p,org_q+intv_q/2),(org_p+intv_p/2,org_q+intv_q/2)]

    for y in p_q_cont:
        solution = get_solution(y,s_act,rpt_times,nwk,n_of_run)
        solution_cont.append(solution[:4])
        diff_cont.append(solution[4])

    opt_solution = sorted(solution_cont)[0]
    opt_curve = diff_cont[solution_cont.index(opt_solution)]
    
    p,q,M = opt_solution[1:]
    f_act = M*opt_curve
    R2 = R_sigma(f_act,s_act)
    
    return [p,q,M],f_act,R2

# 估计参数值
def ABM_estimate(s_act,nwk,k,intv_p=0.001,intv_q=0.005,rpt_times=8,cf=1):
    global last_diff_curves,to_fit
    last_diff_curves = []
    to_fit = pd.DataFrame({'p':[],'q':[],'P':[],'Q':[],'M':[],'m_M':[]})
    estimates_cont,curve_cont,R2_cont,search_cont,pq_cont = [],[],[],[],[]
    
    num_of_run = len(s_act)
    para_range = [[1e-5,0.2],[1e-3,1],[sum(s_act),5*sum(s_act)]]
    Eparams,Esse,R2 = fit_it(s_act,para_range,t_n= 500,c_n= 100,s_threshold = 1e-10,repetes = 1)
    BM_solution= [Eparams.Parameters[i] for i in range(3)]
       
    for l in range(cf):
        time1 = time.clock()
        p0,q0,D_matrix = generate_p0_q0(BM_solution,s_act,nwk,k,rpt_times)
        estimates,curve,R2,search_times,pq_range = search_opt_solution(p0,q0,s_act,num_of_run,nwk,intv_p,intv_q,rpt_times,num_of_cond=2)
        
        estimates_cont.append(estimates)
        curve_cont.append(curve)
        R2_cont.append(R2)
        search_cont.append(search_times)
        pq_cont.append(pq_range)

        print 'The',l+1,'run'
        print '  Optima:',estimates
        print '  R2:',R2
        print '  Number of candidate solutions:',search_times
        print '  Time elasped:',time.clock()-time1,'s'

    p0,q0,D_matrix = generate_p0_q0(BM_solution,s_act,nwk,k,rpt_times)   #get the final Design matrix          
    return estimates_cont,curve_cont,R2_cont,search_cont,pq_cont,D_matrix

#new algorithm
def Refined_ABM_estimate(s_act,nwk,k,intv_p=0.001,intv_q=0.005,rpt_times=8,cf=10):
    global last_diff_curves,to_fit
    last_diff_curves = []
    to_fit = pd.DataFrame({'p':[],'q':[],'P':[],'Q':[],'M':[],'m_M':[]})
    estimates_cont,curve_cont,R2_cont,search_cont = [],[],[],[]
    
    num_of_run = len(s_act)
    para_range = [[1e-5,0.2],[1e-3,1],[sum(s_act),5*sum(s_act)]]
    Eparams,Esse,R2 = fit_it(s_act,para_range,t_n= 500,c_n= 100,s_threshold = 1e-10,repetes = 1)
    BM_solution= [Eparams.Parameters[i] for i in range(3)]
       
    time1 = time.clock()
    p0,q0,D_matrix = generate_p0_q0(BM_solution,s_act,nwk,k,rpt_times)
    estimates,curve,R2,search_times,pq_range = search_opt_solution(p0,q0,s_act,num_of_run,nwk,intv_p,intv_q,rpt_times,num_of_cond=2)
    
    p0,q0,D_matrix = generate_p0_q0(BM_solution,s_act,nwk,k,rpt_times)  #get the final Design matrix

    print 'Time elasped for 1st round search:',time.clock()-time1,'s'
    print 'Number of candidates:',search_times
    print 'The 1 round search:',estimates,'R2:',R2
    
    #repeat cf times searching process
    for l in range(cf):
        time2 = time.clock()
        p1,q1 = estimates[:2]
        f_solution = refined_search(p1,q1,intv_p,intv_q,s_act,rpt_times,nwk)
        p,q,M = f_solution[0]
        f_act = f_solution[1]
        R2 = f_solution[2]
        
        estimates_cont.append([p,q,M])
        curve_cont.append(f_act)
        R2_cont.append(R2)
        
        print 'The',l+1,'round search'
        print 'Time elasped:',time.clock()-time2,'s'
        print '  Optima:',[p,q,M],'  R2:',R2
        
    return estimates_cont,curve_cont,R2_cont,search_times,pq_range,D_matrix

# ABM参数的统计推断，综合利用估计程序中的扩散数据集，以及Bass Model中的参数估计值的协方差矩阵。
#其中，All_diff_curves为ABM参数估计过程中产生的扩散曲线及对应参数；estms_cont为ABM参数估计值（重复n次）；
#curve_cont为参数估计值对应的扩散曲线；r_num为估计bass模型标准差（以ABM扩散曲线均值为被拟合数据）的次数。
def estimate_ABM_std(s_act,D_matrix,estms_cont,curve_cont,para_range,r_num=30):
    #(1) get design matrix
    param_p_P = D_matrix[0][0]
    param_p_Q = D_matrix[0][1]
    mse_p = D_matrix[0][2]
    
    param_q_P = D_matrix[1][0]
    param_q_Q = D_matrix[1][1]
    mse_q = D_matrix[1][2]
  
    param_m_P = D_matrix[2][0]
    param_m_Q = D_matrix[2][1]
    mse_m = D_matrix[2][2]
    
    #(2) compute the covariance matrix of p,q,m
    param_cont = []
    len_act = len(s_act)
    f_act = np.mean(curve_cont,axis=0)
    s_act = np.array(s_act)
    sse = np.sum(np.square(s_act-f_act))
    sigma = np.sqrt(sse/len_act)
    expri_data = np.array([np.array(f_act) + sigma*np.random.randn(len_act) for i in range(r_num)])
    for s_data in expri_data:
        s_exp = repete_estim(s_data,para_range,t_n=500,c_n=100,s_threshold=1e-8,repetes=1)
        param_cont.append(s_exp[1:4])
        
    solutions = np.array(param_cont)  
    P_seq = solutions[:,0]
    Q_seq = solutions[:,1]
    M_seq = solutions[:,2]
    
    cov_PQ = np.cov(P_seq,Q_seq)
    var_P = cov_PQ[0][0]
    var_Q = cov_PQ[1][1]
    cov_P_Q = cov_PQ[0][1]
    
    v1_p = var_P*param_p_P**2 + var_Q*param_p_Q**2 + 2*param_p_P*param_p_Q*cov_P_Q + mse_p #calculate variance brought by real-world variation
    v1_q = var_P*param_q_P**2 + var_Q*param_q_Q**2 + 2*param_q_Q*param_q_Q*cov_P_Q + mse_q
    v1_m = var_P*param_m_P**2 + var_Q*param_m_Q**2 + 2*param_m_P*param_m_Q*cov_P_Q + mse_m
    
    #(3) compute the variance from ABM itself
    estms_cont = np.array(estms_cont)
    dp_seq = estms_cont[:,0] - np.mean(estms_cont[:,0])
    dq_seq = estms_cont[:,1] - np.mean(estms_cont[:,1])
    dm_seq = estms_cont[:,2] - np.mean(estms_cont[:,2])
    t_mat = np.vstack((dp_seq,dq_seq,dm_seq))
    para_cov = np.cov(t_mat)
    
    v2_p = para_cov[0][0] #calculate variance brought by ABM itself
    v2_q = para_cov[1][1]
    v2_m = para_cov[2][2]
    
    #(4) summation of all variances
    V_p = v1_p + v2_p
    V_q = v1_q + v2_q
    V_m = v1_m + v2_m
    
    std_p = np.sqrt(V_p)
    std_q = np.sqrt(V_q)
    std_m = np.sqrt(V_m)*10000
    
    V_mat = [[v1_p,v1_q,v1_m],[v2_p,v2_q,v2_m]]
    
    return [std_p,std_q,std_m],V_mat

if __name__ == 'main':
    # 经典扩散数据集
    data_set = {'room air conditioners':(np.arange(1949,1962),[96,195,238,380,1045,1230,1267,1828,1586,1673,1800,1580,1500]),
                'color televisions':(np.arange(1963,1971),[747,1480,2646,5118,5777,5982,5962,4631]),
                'clothers dryers':(np.arange(1949,1962),[106,319,492,635,737,890,1397,1523,1294,1240,1425,1260,1236]),
                'ultrasound':(np.arange(1965,1979),[5,3,2,5,7,12,6,16,16,28,28,21,13,6]),
                'mammography':(np.arange(1965,1979),[2,2,2,3,4,9,7,16,23,24,15,6,5,1]),
                'foreign language':(np.arange(1952,1964),[1.25,0.77,0.86,0.48,1.34,3.56,3.36,6.24,5.95,6.24,4.89,0.25]),
                'accelerated program':(np.arange(1952,1964),[0.67,0.48,2.11,0.29,2.59,2.21,16.80,11.04,14.40,6.43,6.15,1.15])}

    china_set = {'color televisions':(np.arange(1997,2013),[2.6,1.2,2.11,3.79,3.6,7.33,7.18,5.29,8.42,5.68,6.57,5.49,6.48,5.42,10.72,5.15]),
                 'mobile phones':(np.arange(1997,2013),[1.7,1.6,3.84,12.36,14.5,28.89,27.18,21.33,25.6,15.88,12.3,6.84,9.02,7.82,16.39,7.39]),
                 'computers':(np.arange(1997,2013),[2.6,1.2,2.11,3.79,3.6,7.33,7.18,5.29,8.42,5.68,6.57,5.49,6.48,5.42,10.72,5.15]),
                 'conditioners':(np.arange(1992,2013),[1.19,1.14,2.67,3.09,3.52,4.68,3.71,4.48,6.32,5.0,15.3,10.69,8.01,10.87,7.12,7.29,5.2,6.56,5.23,9.93,4.81]),
                 'water heaters':(np.arange(1988,2013),[28.07,8.4,5.86,6.37,3.9,4.08,5.42,4.12,3.45,3.31,3.12,1.64,2.36,1.8,5.48,1.35,1.47,0.52,1.03,3.28,-1.4,1.72,1.26,0.62,1.25])
                 }

    # 统计推断（暴力仿真方法）
    s_act = data_set['clothers dryers'][1]
    n = 10000
    k = 3
    intv_p = 0.001
    intv_q = 0.005
    rpt_times = 8
    nwk = nx.gnm_random_graph(n,n*k)
    estimates_cont,curve_cont,R2_cont,search_times,pq_range,D_matrix = Refined_ABM_estimate(s_act,intv_p,intv_q,nwk,k,cf=2)
