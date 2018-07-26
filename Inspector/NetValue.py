# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
import datetime as dt
from WindPy import *
import xlrd
from xlwt import Workbook, easyxf
from xlutils.copy import copy
w.start()

def Portfolio_Net_Value(code_list, weight_list, previous_nv, today_date):
    pct_chg_list = list()
    yesterday_date = str(w.tdaysoffset(-1, today_date).Data[0][0])[:10]
    for each_code in code_list:
        temp_data = w.wsd(each_code, "NAV_adj", yesterday_date, today_date)
        if temp_data.ErrorCode != 0:
            print temp_data.ErrorCode
            raise Exception("error in data install!")
        else:
            pct_chg = (temp_data.Data[0][1] - temp_data.Data[0][0])/temp_data.Data[0][0]
            pct_chg_list.append(pct_chg)
    if len(pct_chg_list) == len(code_list):
        pct_chg_list = pct_chg_list + [0.0] * (len(weight_list) - len(code_list))
        nv_pct_chg = sum(np.array(pct_chg_list) * np.array(weight_list))
        nv_today = previous_nv * (1 + nv_pct_chg)
        return nv_today
    else:
        raise Exception("missing data!")

code_list_wenjian = ["163415.OF", "180031.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"159926.OF", "003358.OF", "001512.OF", "511010.OF", "161821.OF", "000022.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_wenjian = [0.54, 2.50, 4.55, 4.50, 3.75, 3.75, 5.00, 5.00, 5.00, 6.25, 5.00, 5.00,
3.72, 3.72, 10.60, 10.49]

code_list_pingheng = ["163415.OF", "180031.OF", "040035.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "159926.OF", "511010.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_pingheng = [0.47, 2.00, 1.93, 5.87, 5.50, 5.85, 5.50, 1.00, 5.75, 5.75, 5.75, 5.90, 5.90, 9.80, 9.75]

code_list_jinqu = ["163415.OF", "180031.OF", "040035.OF", "000471.OF", "513500.OF", "096001.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF", "003022.OF", "000434.OF"]

weight_list_jinqu = [0.41, 2.06, 2.06, 1.00, 4.46, 4.45, 4.45, 7.56, 7.00, 1.00, 5.51, 5.51, 4.20, 4.20, 4.20, 4.20, 10.35, 10.35]

def Nv_Series(codelist, weightlist, startdate, enddate):
    code_list = codelist
    weight_list = weightlist
    if len(code_list) != len(weight_list):
        raise Exception("codelist, weightlist do not match!")
    else:
        weight_list = np.array(weight_list)/100
        tdays_list = w.tdays(startdate, enddate)
        nv_series = pd.Series(index=tdays_list.Times)
        for each_date in nv_series.index:
            nv_series[each_date] = Portfolio_Net_Value(code_list, weight_list, 1, each_date)
        return nv_series.cumprod()

wenjian_series = Nv_Series(code_list_wenjian, weight_list_wenjian, "2017-03-01", "2017-03-8")
pingheng_series = Nv_Series(code_list_pingheng, weight_list_pingheng, "2017-03-01", "2017-03-8")
jinqu_series = Nv_Series(code_list_jinqu, weight_list_jinqu, "2017-03-01", "2017-03-8")

nv_data = pd.DataFrame(np.array([wenjian_series.values, pingheng_series.values, jinqu_series.values]).T, index=wenjian_series.index, columns=[u"稳健组合", u"平衡组合", u"进取组合"])
nv_data.to_excel(u"E:\\研究生资料\\路透实验室\\FOF投顾\\201704报告\\nv_series.xls")
'''
feb_tdays = w.tdayscount("2017-02-01", "2017-02-28").Data[0][0]

year_tdays = w.tdayscount("2017-02-01", "2018-01-31").Data[0][0]

print (nv_data[u"稳健组合"][-1]**(1.0/feb_tdays))**year_tdays
print (nv_data[u"平衡组合"][-1]**(1.0/feb_tdays))**year_tdays
print (nv_data[u"进取组合"][-1]**(1.0/feb_tdays))**year_tdays

print nv_data[u"稳健组合"][-1]**12.0
print nv_data[u"平衡组合"][-1]**12.0
print nv_data[u"进取组合"][-1]**12.0
'''
'''
(nv_series.prod() ** (21/12)) ** 12
(1.0125123 ** (21/11)) ** 12
1.0125123 ** 24
nv_series.cumprod().plot()
plt.show()
nv_series
w.tdayscount("2017-01-01", "2017-12-31")

temp_data = w.wsd("H11001.CSI", "close", "2017-01-26", "2017-02-20")

data = pd.Series(np.array(temp_data.Data[0]), index=temp_data.Times)
data

test = pd.Series([1,2,3,4])
pd.DataFrame(np.array([test.values, test.values]))
'''
