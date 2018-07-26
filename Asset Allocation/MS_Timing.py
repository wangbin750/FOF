# coding=utf-8
import pandas as pd
import numpy as np
import pyper as pr
import statsmodels.api as sm
#import matplotlib.pyplot as plt

from Allocation_Method import Risk_Parity_Weight, Max_Utility_Weight_new, Max_Utility_Weight, Max_Utility_Weight_MS, Max_Utility_Weight_new_MS

def Ms_Simulation(length, p=0.9, q=0.8, mean_p=0.1, mean_q=-0.1, std_p=0.05, std_q=0.15):
    temp_list = []
    indicator = 1
    for i in range(length):
        if indicator == 1:
            temp_ran = np.random.uniform(0, 1)
            if temp_ran <= p:
                temp_data = np.random.randn() * std_p + mean_p
            else:
                temp_data = np.random.randn() * std_q + mean_q
                indicator = 0
        else:
            temp_ran = np.random.uniform(0, 1)
            if temp_ran <= q:
                temp_data = np.random.randn() * std_q + mean_q
            else:
                temp_data = np.random.randn() * std_p + mean_p
                indicator = 1
        temp_list.append(temp_data)
    return temp_list


def Ms_R(return_list, regime_count):
    #return_list = list(data_M["AU9999.SGE"])
    return_frame = pd.DataFrame(np.array([return_list, [1]*len(return_list)]).T, columns=['return', 'One'])
    r("rm(list = ls())")
    r.assign("rframe", return_frame)
    r.assign("rregimecount", regime_count)
    Coef = None
    break_i = 0
    while Coef is None:
        r('''
        lm_return <- lm(formula=return~0+One, data=rframe)
        lm_mswm <- msmFit(lm_return, k=rregimecount, p=0, sw=c(T,T))
        rstd <- lm_mswm@std
        rCoef <- lm_mswm@Coef
        rtransMat <- lm_mswm@transMat
        rprob_smo <- lm_mswm@Fit@smoProb[-1,]
        #print(lm_mswm@Fit@logLikel)
        raic_mswn <- AIC(lm_mswm)
        raic_lm <- AIC(lm_return)
        ''')
        Coef = r.get("rCoef")
        break_i = break_i + 1
        if break_i >= 100:
            break
    if Coef is None:
        print "None!"

    aic_mswm = r.get("raic_mswm")
    aic_lm = r.get("raic_lm")

    if aic_mswm < aic_lm:
        std = np.round(r.get("rstd"),4)
        Coef = np.round(np.array(r.get("rCoef")[' One ']),4)
        transMat = np.round(r.get("rtransMat").T,4)
        prob_smo = np.round(r.get("rprob_smo"),4)
    else:
        std = np.array([np.std(return_list)]*regime_count)
        Coef = np.array([np.mean(return_list)]*regime_count)
        transMat = np.diag([1]*regime_count)
        prob_smo = np.array([[1.0/regime_count]*regime_count]*len(return_list))

    return std, Coef, transMat, prob_smo

def Ms_Py(return_list, regime_count):
    temp_model = sm.tsa.MarkovRegression(np.array(list(return_list)), k_regimes=regime_count, switching_variance=True)
    temp_fit = temp_model.fit()
    temp_params = temp_fit.params
    std = temp_params[(regime_count*regime_count):]
    Coef = temp_params[((regime_count-1)*regime_count):(regime_count*regime_count)]
    transMat = np.array(list(temp_params[0:((regime_count-1)*regime_count)])+ list((1-sum(np.array(temp_params[0:((regime_count-1)*regime_count)]).reshape(regime_count-1, regime_count))))).reshape(regime_count, regime_count)
    #print transMat[2,2]
    prob_smo = temp_fit.smoothed_marginal_probabilities
    #print std
    #print Coef
    return std, Coef, transMat, prob_smo

r = pr.R(use_pandas=True)
r("library(MSwM)")
#data = pd.read_excel('/Users/WangBin-Mac/FOF/Asset Allocation/History_Data_D.xlsx')

data = pd.read_excel("/Users/WangBin-Mac/FOF/Global Allocation/SBG_US_W.xlsx")
data = data.interpolate()
data_W = data.dropna().pct_change().dropna()["SP500"]
port_name = 'large'
switch_count = 2
'''
data_W = (data['stock_%s'%port_name]+1).resample("W").prod().dropna()-1
print (data_W[299:]+1).prod()
data_M = (data['stock_%s'%port_name]+1).resample("M").prod().dropna()-1
'''

return_list = []
position_list = []
date_list = []
ret_list = []
turn_over = 0
position = 0
for each in range(299,len(data_W)-1):

    temp_data = data_W[data_W.index[each-299]:data_W.index[each]]
    #temp_data = data_W[:data_W.index[each]]
    position_pre = position

    #temp_std, temp_Coef, temp_transMat, temp_prob_smo = Ms_R(temp_data, switch_count)
    temp_std, temp_Coef, temp_transMat, temp_prob_smo = Ms_Py(temp_data, switch_count)
    print temp_std
    print temp_Coef
    prob_list = []
    sr_list = []
    for i in range(len(temp_Coef)):
        sr_list.append(temp_Coef[i]/temp_std[i])

    for stat in range(switch_count):
        temp_prob = sum(temp_prob_smo[-1,:]*temp_transMat[:,stat])
        prob_list.append(temp_prob)
    stat_no = prob_list.index(max(prob_list))
    #print prob_list
    if max(prob_list) <= 0.8:
        position = 0
    else:
        if (temp_std[stat_no] >= min(temp_std)) and (sum(data_W[data_W.index[each-1]:data_W.index[each]]) > 0):
        #if sum(data_W[data_W.index[each-3]:data_W.index[each]]) > 0 and temp_std[stat_no] == max(temp_std):
        #if sr_list[stat_no] == max(sr_list):
        #if sum(data_W[data_W.index[each-3]:data_W.index[each]])>0 and temp_std[stat_no] == max(temp_std):
            position = 1
        else:
            if sum(data_W[data_W.index[each-1]:data_W.index[each]]) > 0:
                position = 1
            else:
                position = 0


    if position == position_pre:
        pass
    else:
        turn_over += 1


    temp_return = data_W[data_W.index[each+1]]*position
    return_list.append(temp_return)
    position_list.append(position)
    date_list.append(data_W.index[each+1])
    ret_list.append(data_W[data_W.index[each+1]])
    print data_W.index[each+1]
    print turn_over

pd.DataFrame(np.array([ret_list, position_list, return_list]).T, index=date_list, columns=['ret', 'position', 'return']).to_csv('/Users/WangBin-Mac/FOF/Asset Allocation/MS_T_%s.csv'%port_name)

def Performance(return_series, rf_ret):
    end_value = (return_series + 1.0).prod()
    annual_return = (return_series + 1.0).prod() ** (1/(len(return_series)/52.0)) - 1
    annual_variance = (return_series.var() * 52) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    (return_series + 1).cumprod().plot()
    return [end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown]

timing_series = pd.read_csv('/Users/WangBin-Mac/FOF/Asset Allocation/MS_T_%s.csv'%port_name)['return']
print Performance(timing_series, 0.025)

original_series = pd.read_csv('/Users/WangBin-Mac/FOF/Asset Allocation/MS_T_%s.csv'%port_name)['ret']
print Performance(original_series, 0.025)
