# -*- coding: utf-8 -*-
"""
@Author: Wang Bin
@Time: 2017/1/19 09:30
"""

from Allocation_Method import Risk_Parity_Weight, Combined_Return_Distribution, Max_Utility_Weight
import pandas as pd
import numpy as np
from math import ceil


path = u"/Users/WangBin-Mac/263网盘/FOF相关程序/Asset Allocation/"
#path = u"E:\\263网盘\\FOF相关程序\\Asset Allocation\\"
#out_path = u"E:\\"
out_path = u"/Users/WangBin-Mac/"

History_Data = pd.read_excel(path + "History_data.xlsx")
Predict_Data = pd.read_excel(path + "HP_Data.xlsx")

# 日期设定
year_delta = 5
tau = 1.0
last_date = History_Data.index[-1]  # 当前月份日期
next_date = Predict_Data.index[-1]  # 下一月份日期
if last_date.month <= 11:
    start_year = last_date.year - year_delta
    start_month = last_date.month + 1
else:
    start_year = last_date.year - year_delta + 1
    start_month = 1


#asset_list = ["stock_large", "stock_small", "stock_US", "stock_HongKong", "bond_whole", "gold"]
#bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
#bnds = [(0.1, 0.6), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.0, 0.3)]
#asset_list = ["stock_large", "stock_HongKong", "bond_whole"]
#bnds = [(0.0, None), (0.0, None), (0.0, None)]

weight_frame = pd.DataFrame(index=['stock_huge','stock_large','stock_small','stock_US','stock_HongKong','bond_whole','gold','money'])

'''
稳健
'''

lam = 3.3 #进取-1.9 平衡-2.0 稳健-2.3
money_weight = 0.75
#money_weight = 1.00

asset_list = ["stock_huge", "stock_large", "stock_small", "stock_US", "stock_HongKong", "bond_whole"]
bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]

# 数据设定
history_data = History_Data[asset_list][str(start_year) + '-' + str(start_month): last_date]
predict_data = Predict_Data[asset_list][str(start_year) + '-' + str(start_month): last_date]

cov_mat = history_data[asset_list].cov() * 12.0
omega = np.matrix(cov_mat.values)
mkt_wgt = Risk_Parity_Weight(cov_mat)
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

weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)

weight_wj = weight_bl * money_weight
weight_wj['money'] = 1 - np.sum(weight_wj)

print "稳健组合："
print weight_wj.round(4)

wj_frame = pd.DataFrame(weight_wj.values, index=weight_wj.index, columns=[u'稳健组合'])
weight_frame = pd.merge(weight_frame, wj_frame, left_index=True, right_index=True, how='left')

'''
平衡
'''

lam = 1.7
money_weight = 0.85

asset_list = ["stock_huge", "stock_large", "stock_small", "stock_US", "stock_HongKong", "bond_whole", "gold"]
bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]

# 数据设定
history_data = History_Data[asset_list][str(start_year) + '-' + str(start_month): last_date]
predict_data = Predict_Data[asset_list][str(start_year) + '-' + str(start_month): last_date]

cov_mat = history_data[asset_list].cov() * 12.0
omega = np.matrix(cov_mat.values)
mkt_wgt = Risk_Parity_Weight(cov_mat)
P = np.diag([1] * len(mkt_wgt))
#print mkt_wgt

conf_list = list()
for each in asset_list:
    conf_temp = ((history_data[each][str(start_year) + '-' + str(start_month):] -
                  predict_data[each][str(start_year) + '-' + str(start_month):])**2).mean() * 12.0
    conf_list.append(conf_temp)
conf_mat = np.matrix(np.diag(conf_list))

Q = np.matrix(Predict_Data[asset_list].loc[next_date])

com_ret, com_cov_mat = Combined_Return_Distribution(
    2, cov_mat, tau, mkt_wgt, P, Q, conf_mat)

weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)

if weight_bl['gold'] <= mkt_wgt['gold']:
    pass
else:
    bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, mkt_wgt['gold'])]
    weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)

weight_ph = weight_bl * money_weight
weight_ph['money'] = 1 - np.sum(weight_ph)

print "平衡组合："
print weight_ph.round(4)

ph_frame = pd.DataFrame(weight_ph.values, index=weight_ph.index, columns=[u'平衡组合'])
weight_frame = pd.merge(weight_frame, ph_frame, left_index=True, right_index=True, how='left')


'''
进取
'''

lam = 1.5
money_weight = 0.95

asset_list = ["stock_huge", "stock_large", "stock_small", "stock_US", "stock_HongKong", "bond_whole", "gold"]
bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]

# 数据设定
history_data = History_Data[asset_list][str(start_year) + '-' + str(start_month): last_date]
predict_data = Predict_Data[asset_list][str(start_year) + '-' + str(start_month): last_date]

cov_mat = history_data[asset_list].cov() * 12.0
omega = np.matrix(cov_mat.values)
mkt_wgt = Risk_Parity_Weight(cov_mat)
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

weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)

if weight_bl['gold'] <= mkt_wgt['gold']:
    pass
else:
    bnds = [(0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None), (0.0, mkt_wgt['gold'])]
    weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)

weight_jq = weight_bl * money_weight
weight_jq['money'] = 1 - np.sum(weight_jq)

print "进取组合："
print weight_jq.round(4)

jq_frame = pd.DataFrame(weight_jq.values, index=weight_jq.index, columns=[u'进取组合'])
weight_frame = pd.merge(weight_frame, jq_frame, left_index=True, right_index=True, how='left')

weight_frame.round(4).to_excel(out_path + "weight_frame.xlsx")


passive_fund_map = {"stock_huge" : [u"50ETF"],
"stock_large" : [u"300ETF"],
"stock_small" : [u"500ETF"],
"stock_US" : [u"博时标普500ETF", u"国泰纳斯达克100ETF", u"大成标普500", u"长信美国标普100"],
"stock_HongKong" : [u"华夏恒生ETF", u"华夏沪港通恒生ETF"],
"bond_whole" : [u"海富通上证可质押城投债ETF", u"华夏亚债中国指数A", u"易方达7-10年国开行", u"国泰上证5年期国债ETF", u"银华中证中票50指数债券A", u"国泰上证5年期国债ETF", u"易方达7-10年国开行"],
"gold" : [u"华安黄金易ETF", u"易方达黄金ETF", u"国泰黄金ETF", u"博时黄金ETF"],
"money" : [u"银华日利", u"建信现金添益货币A"]}

active_fund_map = {"stock_huge" : [],
"stock_large" : [u"工银总回报", u"大成内需", u"景顺资源", u"富国天瑞强势", u"长城优选"],
"stock_small" : [u"新华中小市值", u"新华优选成长", u"华安国企改革", u"新华行业轮换"],
"stock_US" : [],
"stock_HongKong" : [u"富国中国中小盘"],
"bond_whole" : [u"大成景安短融债券A", u"大成景安短融债券E", u"中金金利A", u"华安鼎丰债券", u"江信洪福纯债", u"海富通瑞利债券"],
"gold" : [],
"money" : []}

single_uplimit_map = {"stock_huge" : 0.05,
"stock_large" : 0.05,
"stock_small" : 0.05,
"stock_US" : 0.05,
"stock_HongKong" : 0.05,
"bond_whole" : 0.1,
"gold" : 0.075,
"money" : 0.15}

passive_percent_map = {"stock_huge" : 0.5,
"stock_large" : 0.5,
"stock_small" : 0.5,
"stock_US" : 0.5,
"stock_HongKong" : 0.5,
"bond_whole" : 0.5,
"gold" : 1.0,
"money" : 1.0}


for each_portfolio in weight_frame.columns:
    asset_list = []
    fund_list = []
    weight_list = []
    temp_portfolio = weight_frame[each_portfolio].dropna()
    for each_asset in temp_portfolio.index:
        asset_weight = temp_portfolio[each_asset]
        if passive_fund_map[each_asset] == []:
            passive_weight = 0.0
        elif active_fund_map[each_asset] == []:
            passive_weight = asset_weight
        else:
            passive_weight = asset_weight * passive_percent_map[each_asset]

        passive_no = int(min(ceil(passive_weight/single_uplimit_map[each_asset]),len(passive_fund_map[each_asset])))
        temp_weight = passive_weight/passive_no
        asset_list = asset_list + [each_asset]*passive_no
        fund_list = fund_list + passive_fund_map[each_asset][0:passive_no]
        weight_list = weight_list + [temp_weight]*passive_no

        active_weight = asset_weight - passive_weight
        active_no = int(min(ceil(active_weight/single_uplimit_map[each_asset]),len(active_fund_map[each_asset])))
        temp_weight = active_weight/active_no
        asset_list = asset_list + [each_asset]*active_no
        fund_list = fund_list + active_fund_map[each_asset][0:active_no]
        weight_list = weight_list + [temp_weight]*active_no

    pd.DataFrame(np.array([asset_list,fund_list,weight_list]).T,columns=[u'资产',u'基金',u'权重']).round(4).to_excel(out_path+each_portfolio+".xlsx")
