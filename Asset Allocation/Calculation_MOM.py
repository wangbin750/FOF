# -*- coding: utf-8 -*-
"""
@Author: Wang Bin
@Time: 2017/1/19 09:30
"""

portfolio_name = u"guangzhou"

from Allocation_Method import Risk_Parity_Weight, Combined_Return_Distribution, Max_Utility_Weight
import pandas as pd
import numpy as np

History_Data = pd.read_excel(u"/Users/WangBin-Mac/FOF/Asset Allocation/History_data.xlsx")
Predict_Data = pd.read_excel(u"//Users/WangBin-Mac/FOF/Asset Allocation/HP_Data.xlsx")
asset_list = ["stock_huge", "stock_large", "stock_small",
              "stock_US", "stock_HongKong", "bond_whole", "gold"]
bnds = [(0.0, None), (0.0, None), (0.0, None),
        (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
#asset_list = ["stock_large", "stock_small", "stock_US", "stock_HongKong", "bond_whole", "gold"]
#bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
#bnds = [(0.1, 0.6), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.0, 0.3)]
#History_Data['bond_whole'] = History_Data['bond_whole'] * 1.4
asset_list = ["stock_large", "stock_HongKong", "bond_whole"]
bnds = [(0.0, None), (0.0, None), (0.0, None)]

year_delta = 5
tau = 1.0
if portfolio_name == "wenjian":
    lam = 2.3 #进取-1.9 平衡-2.0 稳健-2.3
    money_weight = 0.75
elif portfolio_name == "pingheng":
    lam = 2.1
    money_weight = 0.85
elif portfolio_name == "guangzhou":
    lam = 2.3
    money_weight = 1.0
else:
    raise Exception("Wrong portfolio_name!")


# 日期设定
last_date = History_Data.index[-1]  # 当前月份日期
next_date = Predict_Data.index[-1]  # 下一月份日期
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

#cov_mat = history_data[asset_list].cov() * 12.0
cov_mat = pd.ewmcov(history_data, alpha=0.2).iloc[-3:] * 12.0
# print cov_mat
omega = np.matrix(cov_mat.values)
mkt_wgt = Risk_Parity_Weight(cov_mat)
print mkt_wgt
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

print com_ret

weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)

print weight_bl * money_weight
