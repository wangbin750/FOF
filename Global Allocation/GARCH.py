# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 14:03:38 2017

@author: wangbin
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

constant = 0.3
alpha = 0.6
beta = 0.2
h_0 = (constant/(1-alpha-beta))
e_0 = np.random.randn() * h_0**0.5
temp_h_1 = h_0
temp_e_1 = e_0
h_list = []
u_list = []
e_list = []
for i in range(1000):
    temp_h = constant + alpha*temp_h_1 + beta*(temp_e_1**2.0)
    h_list.append(temp_h)
    temp_u = np.random.randn()
    u_list.append(temp_u)
    temp_e = temp_u*(temp_h**0.5)
    e_list.append(temp_e)
    temp_h_1 = temp_h
    temp_e_1 = temp_e

pd.Series(e_list).var()
pd.Series(e_list).hist(bins=100)
pd.Series(e_list).plot()
pd.Series(e_list).kurtosis(axis=0)

def Normal_Likelihood(data, theta):
    '''
    data是数据
    theta是参数向量，包括均值与标准差
    '''
    lnL = -0.5*np.log(2*np.pi)-0.5*np.log(theta[1]**2)-(0.5/(theta[1]**2.0))*((data-theta[0])**2.0)
    return lnL
    
def normal_fun(x):
    goal = 0.0
    for i in range(len(u_list)):
        temp_log_likelihood = Normal_Likelihood(u_list[i], x)
        goal = goal + temp_log_likelihood
    return -goal

x0 = (0.0, 1.0)
options = {'disp': False, 'maxiter': 500, 'ftol': 1e-10}
res = minimize(normal_fun, x0, method='SLSQP', options=options)
res['x']

def garch_fun(x):
    goal = 0.0
    h_0 = np.std(e_list)**2.0
    e_0 = 0.0
    temp_h_1 = h_0
    temp_e_1 = e_0
    for i in range(len(e_list)):
        temp_h = x[0] + x[1]*temp_h_1 + x[2]*(temp_e_1**2.0)
        temp_lnL = Normal_Likelihood(e_list[i], [0.0, temp_h**0.5])
        temp_h_1 = temp_h
        temp_e_1 = e_list[i]
        goal = goal + temp_lnL
    return -goal

x0 = (0.0, 0.5, 0.5)
options = {'disp': True, 'xtol': 1e-10}
res = minimize(garch_fun, x0, method='nelder-mead', options=options)
res['x']

res_list = []
for j in range(20):
    constant = 0.3
    alpha = 0.6
    beta = 0.2
    h_0 = (constant/(1-alpha-beta))
    e_0 = 0.0
    temp_h_1 = h_0
    temp_e_1 = e_0
    h_list = []
    u_list = []
    e_list = []
    for i in range(1000):
        temp_h = constant + alpha*temp_h_1 + beta*(temp_e_1**2.0)
        h_list.append(temp_h)
        temp_u = np.random.randn()
        u_list.append(temp_u)
        temp_e = temp_u*(temp_h**0.5)
        e_list.append(temp_e)
        temp_h_1 = temp_h
        temp_e_1 = temp_e
    x0 = (0.3, 0.6, 0.2)
    options = {'disp': False, 'ftol': 1e-10}
    res = minimize(garch_fun, x0, method='nelder-mead', options=options)
    print res['x']
    res_list.append(res['x'])
    

