# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


file_dir = u"/Users/WangBin-Mac/"

index_map = {
"money":["money_fund"],
"bond":["bond_rate", "bond_credit"],
"domestic_stock":["stock_A_large", "stock_A_small"],
"foreign_stock":["stock_HongKong", "stock_US"],
"alternative":["gold", "hedge_fund"]
}

fund_map = {
"money_fund":["000621.OF","003164.OF","000709.OF","200103.OF","002912.OF","37001B.OF"],
"bond_rate":["000174.OF","519669.OF","001776.OF","002756.OF","050027.OF","128013.OF"],
"bond_credit":["000174.OF","519669.OF","001776.OF","002756.OF","050027.OF","128013.OF"],
"stock_A_large":["159919.OF","510330.OF","159925.OF","159927.OF","510360.OF"],
"stock_A_small":["510500.OF","159922.OF","510510.OF","159935.OF","510440.OF","000478.OF"],
"stock_HongKong":["159920.OF","164705.OF","513600.OF","513660.OF"],
"stock_US":["040046.OF","159941.OF","000834.OF","513500.OF","096001.OF"],
"gold":["518800.OF","518880.OF","159934.OF"],
"hedge_fund":["000844.OF","000753.OF","001073.OF","000414.OF","000667.OF"]
}

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
            #print "oooo"
            net_value = sum(dataframe.loc[each_date]*position)
            nv_list.append(net_value)
            position = net_value * np.array(weights) / dataframe.loc[each_date]
        else:
            position = position
            net_value = sum(dataframe.loc[each_date]*position)
            nv_list.append(net_value)
    return pd.Series(nv_list, index=dataframe.index)

index_data = pd.read_excel(file_dir+"index.xlsx")
fund_data = pd.read_excel(file_dir+"fund.xlsx")

index_data_r = pd.DataFrame(index=fund_data.index)
for each in fund_map:
    index_data_r[each] = constant_weight_portfolio(fund_data[fund_map[each]], [1.0/len(fund_map[each])]*len(fund_map[each]))*index_data[each][-1]

index_data = index_data.append(index_data_r)

product_history = pd.DataFrame(index=index_data.index)
product_history['money'] = index_data['money_fund']/index_data['money_fund'][0]
product_history['bond'] = constant_weight_portfolio(index_data[index_map['bond']], [0.5, 0.5])
product_history['domestic_stock'] = constant_weight_portfolio(index_data[index_map['domestic_stock']], [0.5, 0.5])
product_history['foreign_stock'] = constant_weight_portfolio(index_data[index_map['foreign_stock']], [0.5, 0.5])
product_history['alternative'] = constant_weight_portfolio(index_data[index_map['alternative']], [0.8, 0.2])

product_history.to_excel(file_dir+"product_history.xlsx")
