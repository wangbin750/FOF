# coding=utf-8

import numpy as np
import pandas as pd


def constant_weight_portfolio(dataframe, weights):
    '''
    用来计算组合的历史模拟净值
    dataframe是一个数据框，记录5支产品的历史净值数据，数据来自之前发你的product_history.xlsx
    weights是一个list,记录5支产品所占的比重，比如[0.3,0.3,0.2,0.1,0.1]，元素相加为1
    '''
    dataframe = dataframe / dataframe.iloc[0]
    net_value = 1
    position = net_value * np.array(weights) / dataframe.iloc[0]
    nv_list = []
    for each_date in dataframe.index:
        if each_date[5:] in ['03/31', '06/30', '09/30', '12/31']:
            net_value = sum(dataframe.loc[each_date]*position)
            nv_list.append(net_value)
            position = net_value * np.array(weights) / dataframe.loc[each_date]
        else:
            position = position
            net_value = sum(dataframe.loc[each_date]*position)
            nv_list.append(net_value)
    return pd.Series(nv_list, index=dataframe.index)


def Performance(nv_series):
    '''
    用来计算四个指标
    输出分别是模拟历史年化业绩、模拟历史年化波动率、模拟历史业绩、95%概率亏损不超过
    '''
    return_series = nv_series.pct_change().dropna()
    annual_return = (nv_series[-1] ** (1/(len(return_series)/365.0)) - 1)*100
    annual_variance = ((return_series.var() * 365.0) ** 0.5)*100
    money_return = 10000 * annual_return / 100
    annual_VaR = -(10000 * return_series.quantile(0.05) * 365.0**0.5)
    return ["%0.2f"%annual_return, "%0.2f"%annual_variance, "%0.2f"%money_return, "%0.2f"%annual_VaR]
