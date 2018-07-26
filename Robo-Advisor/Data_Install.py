# coding=utf-8

import numpy as np
import pandas as pd
import copy
from WindPy import *
w.start()

file_dir = u"Z:\\Mac 上的 WangBin-Mac\\"
start_date = "2014-01-01"
interrupt_date_1 = "2016-12-31"
interrupt_date_2 = "2017-01-01"
end_date = "2017-09-01"

index_list = ["H11025.CSI", "038.CS", "066.CS", "000300.SH", "000905.SH", "HSI.HI", "SPX.GI", "AU9999.SGE", "885064.WI"]
index_name = ["money_fund", "bond_rate", "bond_credit", "stock_A_large", "stock_A_small", "stock_HongKong", "stock_US", "gold", "hedge_fund"]

product_index_map = {
"money":["money_fund"],
"bond":["bond_rate", "bond_credit"],
"domestic_stock":["stock_A_large", "stock_A_small"],
"foreign_stock":["stock_HongKong", "stock_US"],
"alternative":["gold", "hedge_fund"]
}

index_sub_product_map = {
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

test = []
for each in index_sub_product_map:
    test = test + index_sub_product_map[each]

index_data = w.wsd(index_list, "close", start_date, interrupt_date_1, "Days=Alldays")
print index_data.ErrorCode
index_data = pd.DataFrame(np.array(index_data.Data).T, index=index_data.Times, columns=index_name)
index_data_s = copy.deepcopy(index_data)
index_data_s.index = index_data_s.index.strftime('%Y/%m/%d')
index_data_s.to_excel(file_dir+"index.xlsx")
