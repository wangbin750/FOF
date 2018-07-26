# -*- coding: utf-8 -*-
"""
@author: Wang Bin
"""

portfolio_name = u"001"

from Allocation_Method import Risk_Parity_Weight, Min_Variance_Weight, Combined_Return_Distribution, Max_Sharpe_Weight, Max_Utility_Weight, Inverse_Minimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

History_Data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/History_Data.xlsx")
Predict_Data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/Predict_Data.xlsx")
'''
asset_list = ["stock", "foreign", "bond_whole", "gold", "money"]
bnds = [(0.0, None), (0.0, None), (0.0, None),
        (0.0, None), (0.0, None)]
asset_level_1 = pd.Series([-0.05, -0.05, -0.01, -0.05, -0.08], index=asset_list)
asset_level_2 = pd.Series([-0.10, -0.10, -0.02, -0.10, -0.16], index=asset_list)
'''
if portfolio_name == "001":
    #极端保守
    asset_list = ["bond_whole", "money"]
    bnds = [(0.0, 0.7), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -1], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -1], index=asset_list)
    lam = 1.65
    money_weight = 1.0
    asset_drawdown = pd.Series([0.0, 0.0], index=asset_list)
    asset_position = pd.Series([1.0, 1.0], index=asset_list)
elif portfolio_name == "002":
    #极端保守
    asset_list = ["bond_whole", "money"]
    bnds = [(0.0, 0.7), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -1], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -1], index=asset_list)
    lam = 0.1
    money_weight = 1.0
    asset_drawdown = pd.Series([0.0, 0.0], index=asset_list)
    asset_position = pd.Series([1.0, 1.0], index=asset_list)
elif portfolio_name == "003":
    #保守
    asset_list = ["bond_whole", "money"]
    bnds = [(0.0, None), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -1], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -1], index=asset_list)
    lam = 0.06
    money_weight = 1.0
    asset_drawdown = pd.Series([0.0, 0.0], index=asset_list)
    asset_position = pd.Series([1.0, 1.0], index=asset_list)
elif portfolio_name == "004":
    asset_list = ["bond_whole", "money", "foreign", "stock", "gold"]
    bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -1, -0.05, -0.05, -0.05], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -1, -0.10, -0.10, -0.10], index=asset_list)
    lam = 0.775
    money_weight = 1.0
    asset_drawdown = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=asset_list)
    asset_position = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=asset_list)
elif portfolio_name == "005":
    asset_list = ["bond_whole", "money", "foreign", "stock", "gold"]
    bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -1, -0.05, -0.05, -0.05], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -1, -0.10, -0.10, -0.10], index=asset_list)
    lam = 1.1
    money_weight = 1.0
    asset_drawdown = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=asset_list)
    asset_position = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=asset_list)
elif portfolio_name == "006":
    asset_list = ["bond_whole", "money", "foreign", "stock", "gold"]
    bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -1, -0.05, -0.05, -0.05], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -1, -0.10, -0.10, -0.10], index=asset_list)
    lam = 1.1
    money_weight = 1.0
    asset_drawdown = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=asset_list)
    asset_position = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=asset_list)
elif portfolio_name == "007":
    #稳健-激进
    asset_list = ["bond_whole", "money", "foreign", "stock", "gold"]
    bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -1, -0.05, -0.05, -0.05], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -1, -0.10, -0.10, -0.10], index=asset_list)
    lam = 1.1
    money_weight = 1.0
    asset_drawdown = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=asset_list)
    asset_position = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=asset_list)
elif portfolio_name == "008":
    #稳健-激进
    asset_list = ["bond_whole", "money", "foreign", "stock", "gold"]
    bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -1, -0.05, -0.05, -0.05], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -1, -0.10, -0.10, -0.10], index=asset_list)
    lam = 1.3
    money_weight = 1.0
    asset_drawdown = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=asset_list)
    asset_position = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=asset_list)
elif portfolio_name == "009":
    #稳健-激进
    asset_list = ["bond_whole", "money", "foreign", "stock", "gold"]
    bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -1, -0.05, -0.05, -0.05], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -1, -0.10, -0.10, -0.10], index=asset_list)
    lam = 1.1
    money_weight = 1.0
    asset_drawdown = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=asset_list)
    asset_position = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=asset_list)
else:
    pass

year_delta = 5
tau = 1.0

'''
if portfolio_name == "wenjian":
    lam = 2.3 #进取-1.9 平衡-2.0 稳健-2.3
    money_weight = 0.75
elif portfolio_name == "pingheng":
    lam = 1.9
    money_weight = 0.85
elif portfolio_name == "jinqu":
    lam = 1.7
    money_weight = 0.95
else:
    raise Exception("Wrong portfolio_name!")
'''

pct_list = []
weight_list = []
date_list = []

for each_date in Predict_Data.index[72:-1]:
    last_date = History_Data.index[list(Predict_Data.index).index(each_date)-1]  # 当前月份日期
    next_date = each_date  # 下一月份日期
    if last_date.month <= 11:
        start_year = last_date.year - year_delta
        start_month = last_date.month + 1
    else:
        start_year = last_date.year - year_delta + 1
        start_month = 1

    # 基础设定
    history_data = History_Data[asset_list][
        str(start_year) + '-' + str(start_month): last_date]
    predict_data = Predict_Data[asset_list][
        str(start_year) + '-' + str(start_month): last_date]
    cov_mat = history_data[asset_list].cov() * 12.0
    for each_asset in asset_list:
        temp_drawdown = (asset_drawdown[each_asset]+1.0)*(history_data[each_asset][-1]+1.0)-1
        if temp_drawdown >= 0:
            temp_drawdown = 0
        else:
            pass
        asset_drawdown[each_asset] = temp_drawdown
    # print cov_mat
    omega = np.matrix(cov_mat.values)
    mkt_wgt = Risk_Parity_Weight(cov_mat)
    #print mkt_wgt

    P = np.diag([1] * len(mkt_wgt))

    conf_list = list()
    for each in asset_list:
        conf_temp = ((history_data[each][str(start_year) + '-' + str(start_month):] -
                      predict_data[each][str(start_year) + '-' + str(start_month):])**2).mean() * 12.0
        conf_list.append(conf_temp)
    conf_mat = np.matrix(np.diag(conf_list))

    Q = np.matrix(Predict_Data[asset_list].loc[next_date])

    com_ret, com_cov_mat = Combined_Return_Distribution(
        2, cov_mat, tau, mkt_wgt, P, Q, conf_mat)

    #print com_ret

    weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)

    for each_asset in asset_list:
        if asset_position[each_asset] == 1:
            if (asset_drawdown[each_asset] <= asset_level_1[each_asset]) and (asset_drawdown[each_asset] > asset_level_2[each_asset]):
                asset_position[each_asset] = 0.5
            elif asset_drawdown[each_asset] <= asset_level_2[each_asset]:
                asset_position[each_asset] = 0.0
            else:
                pass
        elif asset_position[each_asset] == 0.5:
            if asset_position[each_asset] <= asset_level_2[each_asset]:
                asset_position[each_asset] = 0.0
            elif (predict_data[each_asset][-1] > 0) and (history_data[each_asset][-1] > 0):
                asset_position[each_asset] = 1.0
                asset_drawdown[each_asset] = 0.0
            else:
                pass
        elif asset_position[each_asset] == 0.0:
            if (predict_data[each_asset][-1] > 0) and (history_data[each_asset][-1] > 0):
                asset_position[each_asset] = 0.5
                asset_drawdown[each_asset] = 0.0
            else:
                pass

    weight_bl = weight_bl*asset_position
    weight_bl['money'] = 1 - sum(weight_bl) + weight_bl['money']
    #print sum(weight_bl*History_Data[asset_list].loc[next_date])*money_weight + (1-money_weight)*History_Data["money"].loc[next_date]
    pct_list.append(sum(weight_bl*History_Data[asset_list].loc[next_date])*money_weight + (1-money_weight)*History_Data["money"].loc[next_date])
    weight_list.append(list(weight_bl))
    date_list.append(next_date)

pd.Series(np.array(pct_list), index=date_list).to_csv("/Users/WangBin-Mac/FOF/Robo-Advisor/backtest_%s.csv"%portfolio_name)
pd.DataFrame(np.array(weight_list), index=date_list, columns=asset_list).to_excel("/Users/WangBin-Mac/FOF/Robo-Advisor/backtest_%s_weight.xlsx"%portfolio_name)

print pd.DataFrame(np.array(weight_list), index=date_list, columns=asset_list).mean()
#print (np.array(pct_list)+1).cumprod()[-1]

def Performance(return_series, rf_ret):
    end_value = (return_series + 1).prod()
    annual_return = (return_series + 1).prod() ** (1/(len(return_series)/12.0)) - 1
    annual_variance = (return_series.var() * 12.0) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    (return_series + 1).cumprod().plot()
    return [end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown]


return_series = pd.read_csv("/Users/WangBin-Mac/FOF/Robo-Advisor/backtest_%s.csv"%portfolio_name, header=None)[1]
print Performance(return_series, 0.035)
pd.DataFrame((return_series+1).cumprod(),index=return_series.index).to_csv("/Users/WangBin-Mac/FOF/Robo-Advisor/backtest_%s.csv"%portfolio_name)

'''
file_name_list = ['wenjian', 'pingheng', 'jinqu', 'wenjian_f', 'pingheng_f', 'jinqu_f']
result_list = list()
for each_file in file_name_list:
    return_series = pd.read_csv("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_"+each_file+".csv", header=None)[1]
    return_series = [1] + list(return_series.values)
    return_series = pd.Series(return_series).pct_change().dropna()
    temp_result = Performance(return_series, 0.025)
    result_list.append(temp_result)

pd.DataFrame(np.array(result_list).T, index=['end_value', 'annual_return', 'annual_variance', 'sharpe_ratio', 'max_drawdown'], columns=['wenjian', 'pingheng', 'jinqu', 'wenjian_f', 'pingheng_f', 'jinqu_f']).to_excel("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_results.xlsx")
'''
