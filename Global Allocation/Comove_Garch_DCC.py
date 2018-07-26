# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import pyper as pr
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import socket

hostname = socket.gethostname()
if hostname == "DESKTOP-OGC5NH7":
    path = u"E:/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "localhost":
    path = "/Users/WangBin-Mac/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "CICCB6CR7213VFT":
    path = "F:/GitHub/FOF/Global Allocation/"

asset_col_all = ["Barclays_US_Treasury","Barclays_US_HY","SP500","MSCI_exUS","MSCI_emerging","London_gold"]
asset_col = ["Barclays_US_HY","SP500","MSCI_exUS","MSCI_emerging","London_gold"]

def multi_str(str_list):
    str_out = []
    for i in str_list:
        for j in str_list:
            temp = i + "*" + j
            str_out.append(temp)
    return str_out

asset_col_rho = multi_str(asset_col)
r = pr.R(use_pandas=True)
r("library(MTS)")

Return_frame_ori = pd.read_excel(path+"DCC_Assets_M.xlsx")
Return_frame = pd.DataFrame(Return_frame_ori[asset_col].values, columns=asset_col)
Return_frame_all = Return_frame_ori[asset_col_all]
'''
k = 3

constant = np.array([0.5]*k)
alpha = np.array([0.6]*k)
beta = np.array([0.2]*k)
rho = 0.2
theta = 0.2

b = [0.9, 0.5, -0.5]
B = np.matrix(np.diag(np.array(b)))
A = np.matrix(np.ones([k,k]))-np.matrix(np.eye(k))
I = np.matrix(np.eye(k))

q_0 = 0.5
#乘法模型
Q_0 = q_0*(B*A*B)+I
#加法模型

u_0 = np.random.multivariate_normal([0.0]*k, Q_0)
h_0 = (constant/(1-alpha-beta))
e_0 = u_0 * h_0**0.5

temp_h_1 = h_0
temp_e_1 = e_0
temp_q_1 = q_0
temp_u_1 = u_0
h_list = []
u_list = []
e_list = []
q_list = []
for i in range(10):
    temp_h = constant + alpha*(temp_e_1**2.0) + beta*temp_h_1
    h_list.append(temp_h)
    temp_mean = [0.0] * k
    temp_u = np.random.multivariate_normal(temp_mean, Q_0)
    u_list.append(temp_u)
    temp_e = temp_u*(temp_h**0.5)
    e_list.append(temp_e)
    temp_h_1 = temp_h
    temp_e_1 = temp_e
    temp_u_1 = temp_u
for i in range(300):
    temp_h = constant + alpha*(temp_e_1**2.0) + beta*temp_h_1
    h_list.append(temp_h)
    temp_mean = [0.0] * k
    temp_Psi_1 = np.matrix(pd.DataFrame(u_list[-10:]).corr())
    temp_psi_1 = (np.ones([1,k])*(B.I*(temp_Psi_1-I)*B.I)*np.ones([1,k]).T)/(k*(k-1))
    temp_q = float((1-rho-theta)*q_0 + rho*temp_psi_1 + theta*temp_q_1)
    q_list.append(temp_q)
    temp_Q = temp_q*(B*A*B)+I
    #print temp_Q
    temp_u = np.random.multivariate_normal(temp_mean, temp_Q)
    u_list.append(temp_u)
    temp_e = temp_u*(temp_h**0.5)
    e_list.append(temp_e)
    temp_h_1 = temp_h
    temp_e_1 = temp_e
    temp_q_1 = temp_q
    temp_u_1 = temp_u


#pd.DataFrame(e_list).mean()
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
    lnL = float(-0.5*(log(det(Hm))+r*Hm.I*r.T))
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
    
    x0 = [0.5, 0.5, 0.5]
    options={'disp': False, 'initial_simplex': None, 'maxiter': None, 'xatol': 0.0001, 'return_all': False, 'fatol': 0.0001, 'maxfev': None}
    res = minimize(Garch_Fun, x0, method='Nelder-Mead', options=options)
    #res = minimize(Garch_Fun, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
    '''
    minimizer_kwargs = {"method":"Nelder-Mead"}
    res = basinhopping(Garch_Fun, x0, minimizer_kwargs=minimizer_kwargs, niter=200)
    '''
    h_list = []
    e_list_new = [0.0]+e_list
    temp_h_1 = h_0
    for i in range(len(e_list)):
        temp_h = res['x'][0] + res['x'][1]*(e_list_new[i]**2.0) + res['x'][2]*temp_h_1
        h_list.append(temp_h)
        temp_h_1 = temp_h
    return res['x'], h_list
    
def CDCC_1_1(udata):

    def CDCC_1_1_Fun(x):
        x = list(x)
        b = np.array(x[3:])
        B = np.matrix(np.diag(np.array(b)))

        goal = 0.0
        temp_q_1 = x[0]/(1-x[1]-x[2])
        temp_psi_1 = 0.0
        for i in range(len(uarray)):
            temp_q = float(x[0] + x[1]*temp_psi_1 + x[2]*temp_q_1)
            #print temp_q
            temp_Q_1 = max(0,temp_q)*np.matrix(np.ones([k,k]))+B*np.matrix(np.ones([k,k]))*B
            temp_Q_d = np.matrix(np.diag(np.diag(temp_Q_1)))
            temp_Q_d2 = np.matrix(np.diag([temp_Q_1.max()]*k))
            temp_Q = temp_Q_1 - temp_Q_d + temp_Q_d2
            temp_Q_s = np.matrix(np.diag(np.sqrt(np.diag(temp_Q))))
            #print temp_Q
            temp_Q = temp_Q_s.I * temp_Q * temp_Q_s.I
            #print temp_Q
            temp_lnL = MV_Normal_Likelihood(uarray[i], temp_Q)
            temp_q_1 = temp_q
            if i >= 10:
                temp_Psi_1 = np.matrix(pd.DataFrame(uarray[i-10:i+1]).corr())
            else:
                temp_Psi_1 = np.matrix(pd.DataFrame(uarray).corr())
            temp_psi_1 = (np.ones([1,k])*(temp_Psi_1-I)*np.ones([1,k]).T)/(k*(k-1))
            goal = goal + temp_lnL
        print goal
        print x
        return -goal
    
    k = len(Q_0)
    A = np.matrix(np.ones([k,k]))-np.matrix(np.eye(k))
    I = np.matrix(np.eye(k))
    uarray = udata.values
    x0 = [0.1, 0.1, 0.1, 1.0, 2.0, 1.5, 1.0, 0.0]
    options = {'disp': False, 'ftol': 1e-10}
    cons = ({'type':'ineq', 'fun':lambda x: 1-x[1]-x[2]}) 
    res = minimize(CDCC_1_1_Fun, x0, method='nelder-mead', options=options)
    bnds = ((0,None), (0,1), (0,1), (None,None), (None,None), (None,None))
    #res = minimize(CDCC_1_1_Fun, x0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
    #res = minimize(CDCC_1_1_Fun, x0, args=(), method='SLSQP', jac=None, tol=None, constraints=cons, callback=None, options={'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1.0})
    return res['x']

'''
for i in range(20):
    k = 3

    constant = np.array([0.5]*k)
    alpha = np.array([0.6]*k)
    beta = np.array([0.2]*k)
    rho = 0.2
    theta = 0.2

    b = [2, 1.0, -1.5]
    B = np.matrix(np.diag(np.array(b)))
    A = np.matrix(np.ones([k,k]))-np.matrix(np.eye(k))
    I = np.matrix(np.eye(k))

    q_0 = 0.3
    #乘法模型
    Q_1 = 0.5*np.matrix(np.ones([k,k]))+B*np.matrix(np.ones([k,k]))*B
    Q_d = np.matrix(np.diag(np.diag(Q_1)))
    Q_d2 = np.matrix(np.diag([Q_1.max()]*k))
    Q = Q_1 - Q_d + Q_d2
    Q_s = np.matrix(np.diag(np.sqrt(np.diag(Q))))
    Q_0 = Q_s.I * Q * Q_s.I
    #加法模型

    u_0 = np.random.multivariate_normal([0.0]*k, Q_0)
    h_0 = (constant/(1-alpha-beta))
    e_0 = u_0 * h_0**0.5

    temp_h_1 = h_0
    temp_e_1 = e_0
    temp_q_1 = q_0
    temp_u_1 = u_0
    h_list = []
    u_list = []
    e_list = []
    q_list = []
    for j in range(10):
        temp_h = constant + alpha*(temp_e_1**2.0) + beta*temp_h_1
        h_list.append(temp_h)
        temp_mean = [0.0] * k
        temp_u = np.random.multivariate_normal(temp_mean, Q_0)
        u_list.append(temp_u)
        temp_e = temp_u*(temp_h**0.5)
        e_list.append(temp_e)
        temp_h_1 = temp_h
        temp_e_1 = temp_e
        temp_u_1 = temp_u      
    for ij in range(300):
        temp_h = constant + alpha*(temp_e_1**2.0) + beta*temp_h_1
        h_list.append(temp_h)
        temp_mean = [0.0] * k
        temp_Psi_1 = np.matrix(pd.DataFrame(u_list[-10:]).corr())
        temp_psi_1 = (np.ones([1,k])*(temp_Psi_1-I)*np.ones([1,k]).T)/(k*(k-1))
        temp_q = float(q_0 + rho*temp_psi_1 + theta*temp_q_1)
        q_list.append(temp_q)
        temp_Q_1 = temp_q*np.matrix(np.ones([k,k]))+B*np.matrix(np.ones([k,k]))*B
        temp_Q_d = np.matrix(np.diag(np.diag(temp_Q_1)))
        temp_Q_d2 = np.matrix(np.diag([temp_Q_1.max()]*k))
        temp_Q = temp_Q_1 - temp_Q_d + temp_Q_d2
        temp_Q_s = np.matrix(np.diag(np.sqrt(np.diag(temp_Q))))
        temp_Q = temp_Q_s.I * temp_Q * temp_Q_s.I
        #temp_Q = temp_q*(B*A*B)+I
        #print temp_Q
        temp_u = np.random.multivariate_normal(temp_mean, temp_Q)
        u_list.append(temp_u)
        temp_e = temp_u*(temp_h**0.5)
        e_list.append(temp_e)
        temp_h_1 = temp_h
        temp_e_1 = temp_e
        temp_q_1 = temp_q
        temp_u_1 = temp_u

    hdata = pd.DataFrame(h_list)
    edata = pd.DataFrame(e_list)
    #print edata.var()
    #print edata.corr()    
    
    h_array = []
    for jj in range(k):
        garch_res, temp_h_list = Garch_1_1(list(edata[jj]))
        print garch_res
        h_array.append(temp_h_list)
    hdata = pd.DataFrame(np.array(h_array).T)
    
    print "CDCC:"
    print CDCC_1_1(edata, hdata)
'''

r.assign("rframe", Return_frame)
r('''
rm1 <- dccPre(rframe, include.mean=T, p=0, cond.dist="norm")
rtn1 <- rm1$sresi
rmarVol <- rm1$marVol
rm1_est <- rm1$est
''')
marVol = r.get("rmarVol")
tn1 = r.get("rtn1")

Q_0 = np.matrix(Return_frame.corr())
std_frame = pd.DataFrame(marVol, columns=asset_col)
udata = pd.DataFrame(tn1, columns=asset_col)

print "CDCC:"
print CDCC_1_1(udata)