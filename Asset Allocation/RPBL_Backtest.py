# coding=utf-8

import pandas as pd
import numpy as np
import pyper as pr
from Allocation_Method import Risk_Parity_Weight, Combined_Return_Distribution, Max_Utility_Weight
import socket

hostname = socket.gethostname()
if hostname == "DESKTOP-OGC5NH7":
    path = u"E:/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "localhost":
    path = "/Users/WangBin-Mac/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "CICCB6CR7213VFT":
    path = "F:/GitHub/FOF/Global Allocation/"

def Performance(return_series, rf_ret):
    end_value = (return_series + 1).prod()
    annual_return = (return_series + 1).prod() ** (1/(len(return_series)/12.0)) - 1
    annual_variance = (return_series.var() * 12.0) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    win_ratio = float(sum(return_series>=0))/float(len(return_series))
    return [end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown, win_ratio]


Return_frame = pd.read_excel(path+"SBG_US_M.xlsx")
Return_frame = Return_frame.interpolate().dropna().pct_change().dropna()
asset_list = Return_frame.columns
n_frame = pd.read_excel(path+"RPBL_Backtest_nlist.xlsx")

lam = 2.5
rf_r = 0.025
bnds = [(0.0, None), (0.0, None), (0.0, None)]


r_list = []
n_list = []
for each in range(119, len(Return_frame)-1):
    temp_r_list = []
    return_frame = Return_frame[Return_frame.index[each-119]:Return_frame.index[each]]
    
    cov_mat = return_frame.cov() * 12.0
    omega = np.matrix(cov_mat.values)
    mkt_wgt = Risk_Parity_Weight(cov_mat)
    P = np.diag([1] * len(mkt_wgt))
    for ii in range(50):
        #nr = n_frame[0][each-119]
        nr  = np.random.randn()
        #nr_sum = np.sum(nr)
        #n_list.append(nr)

        for noise in [1.0, 2.0, 4.0,7.0/3.0, 9.0, 19.0, 29.0, 99.0, 199.0, 999.0]:
            if noise in [1.0, 2.0, 4.0, 19.0, 29.0, 199.0, 999.0]:
                continue
            gamma = 1.0/(1+noise)
            #nr_t = (nr_sum/999.0**0.5)*noise**0.5
            #nr_sum = np.sum(nr)
            for tau in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 100.0]:
                if tau in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 100.0]:
                    continue
                conf_list = list()
                q_list = list()
                for i in asset_list:
                    conf_temp = return_frame[i].var() * 12.0
                    q_temp = (1.0+(gamma*Return_frame[i][Return_frame.index[each+1]] + (1-gamma)*nr*conf_temp**0.5)/(gamma**2+(1-gamma)**2)**0.5)**12.0-1.0
                    conf_list.append(conf_temp)
                    q_list.append(q_temp)
                conf_mat = np.matrix(np.diag(conf_list))        
                Q = np.matrix(q_list)

                com_ret, com_cov_mat = Combined_Return_Distribution(2, cov_mat, tau, mkt_wgt, P, Q, conf_mat)

                weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)
                temp_r = np.sum(weight_bl*Return_frame.loc[Return_frame.index[each+1]])
                temp_r_list.append(temp_r)
                #print noise 
                #print tau
    r_list.append(temp_r_list)
        
    print each
    #print nr

r_frame = pd.DataFrame(r_list)
print r_frame
p_list = []
for k in r_frame.columns:
    print Performance(r_frame[k], rf_r)   
    p_list.append(Performance(r_frame[k], rf_r))

pd.DataFrame(p_list).to_excel(path+"RPBL_Backtest_P_gamma_3.xlsx")
r_frame.to_excel(path+"RPBL_Backtest_r_gamma.xlsx")
#pd.DataFrame(n_list).to_excel(path+"RPBL_Backtest_nlist_1.xlsx")
'''
std = 0.1
d_list = []
e_list = []
f_list = []
for i in range(1000):
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn(199)
    d = (a*std + b*std*199)/(1+199)**0.5
    e = (a*std + np.sum(c)*std)/(1+199)**0.5
    f = (np.sum(c)/199**0.5)*99.0**0.5
    d_list.append(d)
    e_list.append(e)
    f_list.append(f)
'''
