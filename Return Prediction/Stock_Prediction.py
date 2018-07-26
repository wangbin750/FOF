# -*- coding: utf-8 -*-
"""
@author: Wang Bin
"""

import pandas as pd
import numpy as np
from scipy import stats
#from WindPy import *
#w.start()

year_delta = 3 #估计模型使用的数据期限
outofsample_delta = 1 # 样本外检验的数据期限
#pre_data = w.wsd("000300.SH", "pb_lf,pe_ttm,dividendyield2,pct_chg", "2005-04-01", "2016-12-19", "Period=M")
#pre_data = pd.DataFrame(np.array(pre_data.Data).T, index=pd.to_datetime(map(lambda x: x.strftime('%Y%m%d'), pre_data.Times)), columns=pre_data.Fields)
#pre_data需把数据时期错开
pre_data = pd.read_excel('pre_data.xlsx')
#print pre_data.columns

pre_list = list()
for each in pre_data.index[48:]:
    para_list = list()
    year = each.year - year_delta
    month = each.month
    history_data = pre_data[str(year)+'-'+str(month): each][:-1]
    year_o = each.year - outofsample_delta
    #for DIVIDENDYIELD2
    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], history_data['PCT_CHG_1'])[0:2]
    para_list.append(['PCT_CHG_1', temp_beta, temp_alpha])


    '''
    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], np.log(history_data['DIVIDENDYIELD2']))[0:2]
    if temp_beta > 0:
        para_list.append(['DIVIDENDYIELD2',temp_beta, temp_alpha])
    else:
        pass

    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], np.log(history_data['PB_LF']))[0:2]
    if temp_beta < 0:
        para_list.append(['PB_LF', temp_beta, temp_alpha])
    else:
        pass

    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], np.log(history_data['PE_TTM']))[0:2]
    if temp_beta < 0:
        para_list.append(['PE_TTM', temp_beta, temp_alpha])
    else:
        pass
    '''
    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], history_data['PMI'])[0:2]
    if temp_beta > 0:
        para_list.append(['PMI', temp_beta, temp_alpha])
    else:
        pass

    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], np.log(history_data['CC']))[0:2]
    if temp_beta > 0:
        para_list.append(['CC', temp_beta, temp_alpha])
    else:
        pass
    '''
    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], history_data['II'])[0:2]
    if temp_beta < 0:
        para_list.append(['II', temp_beta, temp_alpha])
    else:
        pass

    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], history_data['PG'])[0:2]
    if temp_beta > 0:
        para_list.append(['PG', temp_beta, temp_alpha])
    else:
        pass
    '''
    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], np.log(history_data['AN']))[0:2]
    if temp_beta > 0:
        para_list.append(['AN', temp_beta, temp_alpha])
    else:
        pass

    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], history_data['CII'])[0:2]
    if temp_beta < 0:
        para_list.append(['CII', temp_beta, temp_alpha])
    else:
        pass

    temp_beta, temp_alpha = stats.linregress(history_data['PCT_CHG'], history_data['ML'])[0:2]
    if temp_beta < 0:
        para_list.append(['ML', temp_beta, temp_alpha])
    else:
        pass
    #print para_list

    y_hat_list = list()
    square_list = list()
    if len(para_list) >= 1:
        for i in range(len(para_list)):
            temp_square_sum = 0
            for j in pre_data[str(year_o)+'-'+str(month): each][:-1].index:
                year_temp = j.year - year_delta
                month_temp = j.month
                history_data_temp = pre_data[str(year_temp)+'-'+str(month_temp): j][:-1]
                beta_temp, alpha_temp = stats.linregress(history_data_temp['PCT_CHG'], history_data_temp[para_list[i][0]])[0:2]
                y_hat_temp = pre_data[para_list[i][0]][j:][1] * beta_temp + alpha_temp
                square_temp = (y_hat_temp - pre_data['PCT_CHG'][j:][0]) ** 2
                temp_square_sum = temp_square_sum + square_temp
            y_hat_weight = (pre_data[para_list[i][0]][each:][0] * para_list[i][1] + para_list[i][2]) * (1/temp_square_sum)
            y_hat_list.append(y_hat_weight)
            square_list.append((1/temp_square_sum))
        y_hat_comb = np.sum(y_hat_list) / np.sum(square_list)
    else:
        y_hat_comb = history_data['PCT_CHG'].mean()

    pre_list.append(y_hat_comb)

pre_results = pd.DataFrame(pre_list, index=pre_data.index[48:])
real_return = pd.DataFrame(list(pre_data['PCT_CHG'][48:]), index=pre_data.index[48:])


print np.mean((pre_results-real_return).values ** 2)
print real_return.cov()
print np.mean((pre_results-real_return).values)

pd.merge(pre_results, real_return, left_index=True, right_index=True).to_excel('pre_results_new.xlsx')
