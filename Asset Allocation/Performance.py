# -*- coding: utf-8 -*-
"""
@author: Wang Bin
"""

import pandas as pd
import numpy as np


def Performance(file_path, rf_ret):
    data = pd.read_excel(file_path)
    return_series = data['return']
    end_value = (return_series + 1).prod()
    annual_return = (return_series + 1).prod() ** (1/(len(data)/12.0)) - 1
    annual_variance = (return_series.var() * 12.0) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    return end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown

ret_list = list()
var_list = list()
sr_list = list()
mdd_list = list()
ev_list = list()
nv_frame = list()
for i in range(300):
    file_path = u'sbg_lam3_mse0.5/BL_Weights_%s.xlsx' % i
    e, a, b, c, d = Performance(file_path, 0.025)
    ret_list.append(a)
    var_list.append(b)
    sr_list.append(c)
    mdd_list.append(d)
    ev_list.append(e)
    nv_frame.append(list((pd.read_excel(file_path)['return'] + 1).cumprod()))

pd.DataFrame(np.array(nv_frame).T,index=pd.read_excel(file_path).index).plot(legend=False, color='k')
temp = [[np.mean(ev_list), np.mean(ret_list), np.mean(var_list), np.mean(sr_list), np.mean(mdd_list)],
 [np.median(ev_list), np.median(ret_list), np.median(var_list), np.median(sr_list), np.median(mdd_list)],
 [np.max(ev_list), np.max(ret_list), np.min(var_list), np.max(sr_list), np.min(mdd_list)],
 [pd.Series(ev_list).quantile(0.05), pd.Series(ret_list).quantile(0.05), pd.Series(var_list).quantile(0.05), pd.Series(sr_list).quantile(0.05), pd.Series(mdd_list).quantile(0.05)],
        [pd.Series(ev_list).quantile(0.01), pd.Series(ret_list).quantile(0.01), pd.Series(var_list).quantile(0.01), pd.Series(sr_list).quantile(0.01), pd.Series(mdd_list).quantile(0.01)],
        [np.min(ev_list), np.min(ret_list), np.max(var_list), np.min(sr_list), np.max(mdd_list)]]

#pd.DataFrame(np.array(temp)).to_excel('temp.xlsx')

test = pd.Series(ev_list)
test[test==test[(test > test.median())].min()]
file_path = u'股债_lam3/BL_Weights_234.xlsx'
nv_frame = list((pd.read_excel(file_path)['return'] + 1).cumprod())
pd.DataFrame(nv_frame,index=pd.read_excel(file_path).index).plot(legend=False, color='k')

def Performance(return_series, rf_ret):
    end_value = (return_series + 1).prod()
    annual_return = (return_series + 1).prod() ** (1/(len(return_series)/12.0)) - 1
    annual_variance = (return_series.var() * 12.0) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    return [end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown]

data = pd.read_excel(u"/Users/WangBin-Mac/FOF/Asset Allocation/History_data.xlsx")

per_list = []
for each in data.columns:
    temp_performance = Performance(data[each].dropna(), 0.03)
    per_list.append(temp_performance)

pd.DataFrame(np.array(per_list), index=data.columns).to_excel(u"/Users/WangBin-Mac/Documents/金建投资/智能投顾/建行智投项目/20170728-项目推进汇报/资产收益特征.xlsx")
