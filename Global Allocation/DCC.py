# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:11:24 2017

@author: wangbin
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
'''
constant = np.array([0.3, 0.2])
alpha = np.array([0.6, 0.3])
beta = np.array([0.2, 0.5])
q_0 = 0.5
rho = 0.2
theta = 0.2
qm_0 = np.matrix([[1.0, q_0], [q_0, 1.0]])

u_0 = np.random.multivariate_normal([0.0, 0.0], qm_0)
h_0 = (constant/(1-alpha-beta))
e_0 = u_0 * h_0**0.5

temp_h_1 = h_0
temp_e_1 = e_0
temp_q_1 = qm_0
temp_u_1 = u_0
h_list = []
u_list = []
e_list = []
q_list = []
for i in range(1000):
    temp_h = constant + alpha*(temp_e_1**2.0) + beta*temp_h_1
    h_list.append(temp_h)
    temp_mean = [0.0, 0.0]
    temp_q = (1-rho-theta)*qm_0 + rho*(np.matrix(temp_u_1).T*np.matrix(temp_u_1)) + theta*temp_q_1
    temp_qs = np.matrix(np.diag(np.sqrt(np.diag(temp_q))))
    temp_r = temp_qs.I * temp_q * temp_qs.I
    temp_u = np.random.multivariate_normal(temp_mean, temp_r)
    u_list.append(temp_u)
    temp_e = temp_u*(temp_h**0.5)
    e_list.append(temp_e)
    temp_h_1 = temp_h
    temp_e_1 = temp_e
    temp_q_1 = temp_q
    temp_u_1 = temp_u
'''


def MV_Normal_Likelihood(data, Hm):
    '''
    此函数仅针对均值为0的多元正态分布
    Hm为协方差矩阵
    '''
    from numpy import log
    from numpy.linalg import det
    k = len(Hm)
    r = np.matrix(data)
    lnL = float(-0.5*(k*log(2*np.pi)+log(det(Hm))+r*Hm.I*r.T))
    return lnL

def Normal_Likelihood(data, theta):
    '''
    data是数据
    theta是参数向量，包括均值与标准差
    '''
    lnL = -0.5*np.log(2*np.pi)-0.5*np.log(theta[1]**2)-(0.5/(theta[1]**2.0))*((data-theta[0])**2.0)
    return lnL


def Garch_1_1(e_list):
    h_0 = np.std(e_list)**2.0
    e_0 = 0.0
    
    def Garch_Fun(x):
        goal = 0.0
        temp_h_1 = h_0
        temp_e_1 = e_0
        for i in range(len(e_list)):
            temp_h = x[0] + x[1]*(temp_e_1**2.0) + x[2]*temp_h_1
            temp_lnL = Normal_Likelihood(e_list[i], [0.0, temp_h**0.5])
            temp_h_1 = temp_h
            temp_e_1 = e_list[i]
            goal = goal + temp_lnL
        return -goal
    
    x0 = [0.0, 0.0, 0.0]
    options = {'disp': False, 'ftol': 1e-10}
    res = minimize(Garch_Fun, x0, method='nelder-mead', options=options)
    
    h_list = []
    e_list_new = [0.0]+e_list
    temp_h_1 = h_0
    for i in range(len(e_list)):
        temp_h = res['x'][0] + res['x'][1]*(e_list_new[i]**2.0) + res['x'][2]*temp_h_1
        h_list.append(temp_h)
        temp_h_1 = temp_h
    return res['x'], h_list
    
def DCC_1_1(edata, hdata):
    
    def para2pm(x,n):
        x = list(x)
        i_m = np.matrix(np.diag([1.0]*n))
        s = 0
        m_array = list()
        for i in range(n):
            temp_row = [0.0]*(i+1)+x[s:(s+n-i-1)]
            m_array.append(temp_row)
            s = s+n-i-1
            m = np.matrix(m_array)
        return (m+i_m).T + (m+i_m) - i_m

    def DCC_1_1_Fun(x):
        x = list(x)
        goal = 0.0
        temp_q_1 = qm_0
        temp_u_1 = u_0
        for i in range(len(uarray)):
            temp_q = (1-x[-2]-x[-1])*para2pm(x,n) + x[-2]*(np.matrix(temp_u_1).T*np.matrix(temp_u_1)) + x[-1]*temp_q_1
            temp_qs = np.matrix(np.diag(np.sqrt(np.diag(temp_q))))
            temp_r = temp_qs.I * temp_q * temp_qs.I
            temp_lnL = MV_Normal_Likelihood(uarray[i], temp_r)
            temp_u_1 = uarray[i]
            temp_q_1 = temp_q
            goal = goal + temp_lnL
        return -goal
    
    udata = edata/hdata**0.5
    qm_0 = np.matrix(udata.corr())
    n = len(qm_0)
    u_0 = np.random.multivariate_normal([0.0]*n, qm_0)
    uarray = udata.values
    x0 = [0.1] * (n*(n-1)/2+2)
    options = {'disp': False, 'ftol': 1e-10}
    res = minimize(DCC_1_1_Fun, x0, method='nelder-mead', options=options)
    return res['x']

def para2pm(x,n):
        x = list(x)
        i_m = np.matrix(np.diag([1.0]*n))
        s = 0
        m_array = list()
        for i in range(n):
            temp_row = [0.0]*(i+1)+x[s:(s+n-i-1)]
            m_array.append(temp_row)
            s = s+n-i-1
            m = np.matrix(m_array)
        return (m+i_m).T + (m+i_m) - i_m

for i in range(20):
    k = 5
    
    constant = np.array([0.3]*k)
    alpha = np.array([0.6]*k)
    beta = np.array([0.2]*k)
    q_0 = [0.3]*10
    rho = 0.2
    theta = 0.2
    qm_0 = para2pm(q_0, k)
    
    u_0 = np.random.multivariate_normal([0.0]*k, qm_0)
    h_0 = (constant/(1-alpha-beta))
    e_0 = u_0 * h_0**0.5
    
    temp_h_1 = h_0
    temp_e_1 = e_0
    temp_q_1 = qm_0
    temp_u_1 = u_0
    h_list = []
    u_list = []
    e_list = []
    q_list = []
    for i in range(150):
        temp_h = constant + alpha*(temp_e_1**2.0) + beta*temp_h_1
        h_list.append(temp_h)
        temp_mean = [0.0]*k
        temp_q = (1-rho-theta)*qm_0 + rho*(np.matrix(temp_u_1).T*np.matrix(temp_u_1)) + theta*temp_q_1
        temp_qs = np.matrix(np.diag(np.sqrt(np.diag(temp_q))))
        temp_r = temp_qs.I * temp_q * temp_qs.I
        temp_u = np.random.multivariate_normal(temp_mean, temp_r)
        u_list.append(temp_u)
        temp_e = temp_u*(temp_h**0.5)
        e_list.append(temp_e)
        temp_h_1 = temp_h
        temp_e_1 = temp_e
        temp_q_1 = temp_q
        temp_u_1 = temp_u

    hdata = pd.DataFrame(h_list)
    edata = pd.DataFrame(e_list)
    
    h_array = []
    for i in range(k):
        garch_res, temp_h_list = Garch_1_1(list(edata[i]))
        print garch_res
        h_array.append(temp_h_list)
    
    hdata = pd.DataFrame(np.array(h_array).T)
    print "DCC:"
    print DCC_1_1(edata, hdata)