# coding=utf-8

import datetime as dt

import numpy as np
import pandas as pd
import xlrd
from xlwt import Workbook, easyxf

from WindPy import *

w.start()

out_path = u"E:\\"
#out_path = u"Z:\\Mac 上的 WangBin-Mac\\"
#交易日
pre_end_date = "2018-04-27"
start_date = "2018-05-01"
end_date = "2018-05-31"


def Portfolio_Return(codelist, weightlist, startdate, enddate):
    pct_chg_list = list()
    temp_data = w.wsd(codelist, "NAV_adj", startdate, enddate)
    if temp_data.ErrorCode != 0:
        print temp_data.ErrorCode
        raise Exception("error in data install!")
    else:
        for i in range(len(temp_data.Data)):
            pct_chg = (temp_data.Data[i][-1] - temp_data.Data[i][0])/temp_data.Data[i][0]
            pct_chg_list.append(pct_chg)
        nav_data = pd.DataFrame(np.array(temp_data.Data).T, columns=temp_data.Codes, index=temp_data.Times)
        nav_data = ((nav_data.pct_change()+1).cumprod()*weightlist).apply(sum, axis=1)+(1-sum(weightlist))

    if len(pct_chg_list) == len(codelist):
        pct_chg_list = pct_chg_list + [0.0] * (len(weightlist) - len(codelist))
        nv_pct_chg = sum(np.array(pct_chg_list) * np.array(weightlist))
        contri_list = list()
        for j in range(len(codelist)):
            contri = (pct_chg_list[j] * weightlist[j])/nv_pct_chg
            contri_list.append(contri)
        contri_series = pd.DataFrame(np.array([pct_chg_list, contri_list, weightlist]).T, index=codelist, columns=[u"涨幅", u"贡献率", u"权重"])
        return nv_pct_chg, contri_series, nav_data
    else:
        raise Exception("missing data!")

def Portfolio_Index_Return(codelist, weightlist, pre_end_date, startdate, enddate):
    pct_chg_list = list()
    temp_data = w.wsd(codelist, "close", startdate, enddate)
    temp_data_2 = w.wsd(codelist, "close", pre_end_date, enddate)
    if temp_data.ErrorCode != 0:
        print temp_data.ErrorCode
        raise Exception("error in data install!")
    else:
        for i in range(len(temp_data.Data)):
            pct_chg = (temp_data.Data[i][-1] - temp_data.Data[i][0])/temp_data.Data[i][0]
            pct_chg_list.append(pct_chg)
        index_data = pd.DataFrame(np.array(temp_data_2.Data).T, columns=temp_data_2.Codes, index=temp_data_2.Times)
    if len(pct_chg_list) == len(codelist):
        nv_pct_chg = sum(np.array(pct_chg_list) * np.array(weightlist))
        contri_list = list()
        for j in range(len(codelist)):
            contri = (pct_chg_list[j] * weightlist[j])/nv_pct_chg
            contri_list.append(contri)
        contri_series = pd.DataFrame(np.array([pct_chg_list, contri_list]).T, index=codelist, columns=[u"涨幅", u"贡献率"])
        return nv_pct_chg, contri_series, index_data
    else:
        raise Exception("missing data!")

def Return_Decomposition(indexn, fund, indexweight, fundweight, port_name):
    indexreturn, index_decom, index_data = Portfolio_Index_Return(indexn, (np.array(indexweight)/100.0), pre_end_date, start_date, end_date)
    fundreturn, fund_decom, nav_data = Portfolio_Return(fund, (np.array(fundweight)/100.0), start_date, end_date)
    index_fund_return_list = list()
    for each_index in index_decom.index:
        index_fund_return = list()
        index_fund_contri = list()
        index_fund_weight = list()
        for each_fund in fund_decom.index:
            if each_fund in index_fund_map[each_index]:
                index_fund_return.append(fund_decom[u"权重"][each_fund] * fund_decom[u"涨幅"][each_fund])
                index_fund_contri.append(fund_decom[u"贡献率"][each_fund])
                index_fund_weight.append(fund_decom[u"权重"][each_fund])
        if sum(index_fund_weight) == 0:
            index_fund_return_list.append(["--", "--"])
        else:
            index_fund_return_list.append([sum(index_fund_return)/sum(index_fund_weight), sum(index_fund_contri)])
    index_fund_decom = pd.DataFrame(np.array(index_fund_return_list), index=indexn, columns=[u"子基金组合收益率", u"子基金组合贡献率"])
    index_decom_new = pd.merge(index_decom, index_fund_decom, left_index=True, right_index=True)
    index_decom_new.index = w.wss(indexn, "sec_name").Data[0]
    index_decom_new.to_excel(out_path + "index_decom_" + port_name + ".xlsx")
    index_data.to_csv(out_path + "index.csv")

    fund_index_return = list()
    for each_fund in fund_decom.index:
        indicator = 0
        for each_index in index_decom.index:
            if each_fund in index_fund_map[each_index]:
                fund_index_return.append(index_decom[u"涨幅"][each_index])
                indicator = 1
        if indicator == 1:
            pass
        else:
            raise Exception(each_fund + " not in index_fund_map!")

    fund_index_decom = pd.DataFrame(fund_index_return, index=fund, columns=[u"指数涨跌幅"])
    fund_decom_new = pd.merge(fund_decom[[u"涨幅", u"贡献率"]], fund_index_decom, left_index=True, right_index=True)
    fund_decom_new = pd.DataFrame(fund_decom_new.values, index=w.wss(fund, "sec_name").Data[0], columns=fund_decom_new.columns)
    fund_decom_new = fund_decom_new.reindex(columns=[u"贡献率", u"涨幅", u"指数涨跌幅"])
    fund_decom_new.to_excel(out_path + "fund_decom_" + port_name + ".xlsx")
    nav_data.to_csv(out_path + "nav_" + port_name + ".csv")


index_code_list = ["000016.SH", "000300.SH", "000905.SH", "SPX.GI", "HSI.HI", "H11001.CSI", "AU9999.SGE", "H11025.CSI"]

index_fund_map = {"000016.SH":["510050.OF", "090013.OF"],
"000300.SH" : ["510300.OF", "163415.OF", "000308.OF", "519736.OF", "000577.OF", "000457.OF"],
"000905.SH" : ["510500.OF", "159915.OF", "159902.OF", "180031.OF", "000547.OF", "000524.OF", "040035.OF", "000471.OF", "519097.OF", "002214.OF"],
"SPX.GI" : ["513500.OF", "519981.OF", "096001.OF", "513100.OF"],
"HSI.HI" : ["159920.OF", "513660.OF", "100061.OF", "510900.OF", "040018.OF"],
"H11001.CSI": ["002086.OF", "000128.OF", "004614.OF", "511220.OF", "001021.OF", "003358.OF", "511010.OF", "161821.OF", "159926.OF", "001512.OF", "000022.OF", "003429.OF", "003987.OF", "003223.OF", "003327.OF", "003833.OF", "519152.OF", "003796.OF", "003795.OF"],
"AU9999.SGE" : ["518880.OF", "159937.OF", "159934.OF", "320013.OF", "518800.OF"],
"H11025.CSI" : ["003022.OF", "000434.OF", "511880.OF", "511990.OF"]}


'''
#二月组合权重
code_list_wenjian = ["180031.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"159926.OF", "003358.OF", "001512.OF", "511010.OF", "161821.OF", "000022.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_wenjian = [4.47, 3.78, 3.78, 4.33, 4.33, 5.00, 5.00, 5.00, 6.25, 5.00, 5.00,
3.72, 3.72, 10.00, 10.00]
#weight_list_wenjian = [4.47, 3.78, 3.78, 4.33, 4.33, 7.60, 7.60, 7.60, 8.88, 7.60, 7.60,
#3.72, 3.72, 10.00, 10.00]

code_list_pingheng = ["180031.OF", "040035.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "159926.OF", "511010.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_pingheng = [3.78, 3.78, 4.79, 4.79, 7.27, 7.27, 1.00, 5.75, 5.75, 5.75, 5.90, 5.90, 8.00, 8.00]
#weight_list_pingheng = [3.78, 3.78, 4.79, 4.79, 7.27, 7.27, 5.57, 10.32, 10.32, 10.32, 5.90, 5.90, 8.00, 8.00]

code_list_jinqu = ["180031.OF", "040035.OF", "000471.OF", "513500.OF", "096001.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF", "003022.OF", "000434.OF"]

weight_list_jinqu = [3.70, 3.70, 3.70, 3.95, 3.95, 3.95, 10.64, 10.64, 1.00, 5.51, 5.51, 4.20, 4.20, 4.20, 4.20, 6.50, 6.50]
#weight_list_jinqu = [3.70, 3.70, 3.70, 3.95, 3.95, 3.95, 10.64, 10.64, 5.01, 9.52, 9.52, 4.20, 4.20, 4.20, 4.20, 6.50, 6.50]


index_weight_wenjian = [0.00, 4.47, 7.56, 8.66, 31.25, 7.43, 25.00]
index_weight_pingheng = [0.00, 7.56, 9.58, 14.55, 18.25, 11.81, 20.00]
index_weight_jinqu = [0.00, 11.09, 11.83, 21.27, 12.01, 16.80, 15.00]
index_weight_wenjian = [0.00, 4.47, 7.56, 8.66, 46.88, 7.43, 25.00]
index_weight_pingheng = [0.00, 7.56, 9.58, 14.55, 36.50, 11.81, 20.00]
index_weight_jinqu = [0.00, 11.09, 11.83, 21.27, 24.02, 16.80, 15.00]
'''

'''
#三月组合权重
code_list_wenjian = ["163415.OF", "180031.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"159926.OF", "003358.OF", "001512.OF", "511010.OF", "161821.OF", "000022.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_wenjian = [0.54, 2.50, 4.55, 4.55, 3.75, 3.75, 5.00, 5.00, 5.00, 6.25, 5.00, 5.00,
3.72, 3.72, 10.60, 10.49]
weight_list_wenjian = [0.54, 2.50, 4.55, 4.55, 3.75, 3.75, 7.60, 7.60, 7.60, 8.88, 7.60, 7.60,
3.72, 3.72, 10.60, 10.49]

code_list_pingheng = ["163415.OF", "180031.OF", "040035.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "159926.OF", "511010.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_pingheng = [0.47, 2.00, 1.93, 5.87, 5.50, 5.85, 5.50, 1.00, 5.75, 5.75, 5.75, 5.90, 5.90, 9.80, 9.75]
weight_list_pingheng = [0.47, 2.00, 1.93, 5.87, 5.50, 5.85, 5.50, 5.57, 10.32, 10.32, 10.32, 5.90, 5.90, 9.80, 9.75]

code_list_jinqu = ["163415.OF", "180031.OF", "040035.OF", "000471.OF", "513500.OF", "096001.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF", "003022.OF", "000434.OF"]

weight_list_jinqu = [0.41, 2.06, 2.06, 1.00, 4.46, 4.45, 4.45, 7.56, 7.00, 1.00, 5.51, 5.51, 4.20, 4.20, 4.20, 4.20, 10.35, 10.35]
weight_list_jinqu = [0.41, 2.06, 2.06, 1.00, 4.46, 4.45, 4.45, 7.56, 7.00, 5.01, 9.52, 9.52, 4.20, 4.20, 4.20, 4.20, 10.35, 10.35]

index_weight_wenjian = [0.54, 2.50, 9.05, 7.50, 31.25, 7.43, 26.11]
index_weight_pingheng = [0.47, 3.93, 11.37, 11.35, 18.25, 11.81, 24.57]
index_weight_jinqu = [0.41, 5.12, 13.36, 14.56, 12.01, 16.80, 25.73]
index_weight_wenjian = [0.54, 2.50, 9.05, 7.50, 46.88, 7.43, 26.11]
index_weight_pingheng = [0.47, 3.93, 11.37, 11.35, 36.50, 11.81, 24.57]
index_weight_jinqu = [0.41, 5.12, 13.36, 14.56, 24.02, 16.80, 25.73]
'''
'''
#4月组合权重
code_list_wenjian = ["163415.OF", "180031.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"159926.OF", "003358.OF", "001512.OF", "511010.OF", "161821.OF", "000022.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_wenjian = [0.00, 3.45, 3.08, 3.08, 4.38, 4.38, 5.25, 5.25, 5.25, 6.25, 5.25, 5.25,
3.72, 3.72, 10.58, 10.49]
#weight_list_wenjian = [0.00, 3.45, 3.08, 3.08, 4.38, 4.38, 7.86, 7.86, 7.86, 8.85, 7.85, 7.85, 3.72, 3.72, 10.58, 10.49]

code_list_pingheng = ["163415.OF", "180031.OF", "040035.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "159926.OF", "511010.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_pingheng = [0.00, 2.67, 2.67, 3.51, 3.50, 6.74, 6.73, 2.25, 6.25, 6.25, 6.25, 5.90, 5.90, 9.05, 9.05]
#weight_list_pingheng = [0.00, 2.67, 2.67, 3.51, 3.50, 6.74, 6.73, 6.82, 10.82, 10.82, 10.82, 5.90, 5.90, 9.05, 9.05]


code_list_jinqu = ["163415.OF", "180031.OF", "040035.OF", "000471.OF", "513500.OF", "096001.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF", "003022.OF", "000434.OF"]

weight_list_jinqu = [0.00, 2.50, 2.50, 2.00, 2.60, 2.60, 2.60, 8.81, 8.81, 3.00, 7.50, 7.50, 4.20, 4.20, 4.20, 4.20, 8.00, 7.75]
#weight_list_jinqu = [0.00, 2.50, 2.50, 2.00, 2.60, 2.60, 2.60, 8.81, 8.81, 7.01, 11.51, 11.51, 4.20, 4.20, 4.20, 4.20, 8.00, 7.75]

index_weight_wenjian = [0.00, 3.45, 6.16, 8.76, 32.50, 7.44, 26.06]
index_weight_pingheng = [0.00, 5.34, 7.01, 13.47, 21.00, 11.80, 23.10]
index_weight_jinqu = [0.00, 7.00, 7.80, 17.62, 18.00, 16.80, 20.75]
#index_weight_wenjian = [0.00, 3.45, 6.16, 8.76, 48.13, 7.44, 26.06]
#index_weight_pingheng = [0.00, 5.34, 7.01, 13.47, 39.28, 11.80, 23.10]
#index_weight_jinqu = [0.00, 7.00, 7.80, 17.62, 30.03, 16.80, 20.75]
'''

'''
#5月组合权重
code_list_wenjian = ["163415.OF", "180031.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"159926.OF", "003358.OF", "001512.OF", "511010.OF", "161821.OF", "000022.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_wenjian = [0.00, 3.22, 3.03, 3.02, 3.26, 3.26, 5.25, 5.25, 5.25, 6.25, 5.25, 5.25, 3.72, 3.72, 10.94, 10.94]
#weight_list_wenjian = [0.00, 3.22, 3.03, 3.02, 3.26, 3.26, 8.15, 8.15, 8.15, 9.15, 8.15, 8.15, 3.72, 3.72, 10.94, 10.94]

code_list_pingheng = ["163415.OF", "180031.OF", "040035.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "159926.OF", "511010.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_pingheng = [0.00, 2.29, 2.29, 3.31, 3.31, 4.59, 4.58, 2.25, 6.25, 6.25, 6.25, 5.90, 5.90, 8.50, 8.50]
#weight_list_pingheng = [0.00, 2.29, 2.29, 3.31, 3.31, 4.59, 4.58, 8.52, 12.52, 12.52, 12.52, 5.90, 5.90, 8.50, 8.50]


code_list_jinqu = ["163415.OF", "180031.OF", "040035.OF", "000471.OF", "513500.OF", "096001.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF", "003022.OF", "000434.OF"]

weight_list_jinqu = [0.00, 2.80, 0.00, 2.77, 2.40, 2.40, 2.35, 5.56, 5.56, 3.00, 7.50, 7.50, 4.20, 4.20, 4.20, 4.20, 5.72, 5.72]
#weight_list_jinqu = [0.00, 2.80, 0.00, 2.77, 2.40, 2.40, 2.35, 5.56, 5.56, 11.27, 15.76, 15.76, 4.20, 4.20, 4.20, 4.20, 5.72, 5.72]

index_weight_wenjian = [0.00, 3.22, 6.05, 6.52, 32.50, 7.43, 26.88]
index_weight_pingheng = [0.00, 4.58, 6.62, 9.17, 21.00, 11.81, 21.75]
index_weight_jinqu = [0.00, 5.57, 7.15, 11.12, 18.00, 16.80, 16.44]
#index_weight_wenjian = [0.00, 3.22, 6.05, 6.52, 49.90, 7.43, 26.88]
#index_weight_pingheng = [0.00, 4.58, 6.62, 9.17, 46.08, 11.81, 21.75]
#index_weight_jinqu = [0.00, 5.57, 7.15, 11.12, 42.93, 16.80, 16.44]
'''

'''
#六月组合权重
code_list_wenjian = ["163415.OF", "180031.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"159926.OF", "003358.OF", "001512.OF", "511010.OF", "161821.OF", "000022.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_wenjian = [3.34, 0.00, 3.78, 3.78, 3.94, 3.93, 5.25, 5.25, 5.25, 6.00, 5.25, 5.25,
3.13, 3.12, 10.82, 10.80]
#weight_list_wenjian = [0.54, 2.50, 4.55, 4.55, 3.75, 3.75, 7.60, 7.60, 7.60, 8.88, 7.60, 7.60, 3.72, 3.72, 10.60, 10.49]

code_list_pingheng = ["163415.OF", "090013.OF", "180031.OF", "040035.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF", "511220.OF", "003358.OF", "159926.OF", "511010.OF", "518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_pingheng = [2.34, 2.33, 0.00, 0.00, 4.51, 4.51, 5.83, 5.83, 2.99, 6.25, 6.25, 6.25, 4.37, 4.36, 8.72, 8.72]
#weight_list_pingheng = [0.47, 2.00, 1.93, 5.87, 5.50, 5.85, 5.50, 5.57, 10.32, 10.32, 10.32, 5.90, 5.90, 9.80, 9.75]

code_list_jinqu = ["163415.OF", "090013.OF", "180031.OF", "040035.OF", "000471.OF", "513500.OF", "096001.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF", "003022.OF", "000434.OF"]

weight_list_jinqu = [2.86, 2.86, 0.00, 0.00, 0.00, 3.45, 3.45, 3.35, 7.31, 7.31, 3.78, 8.27, 8.27, 2.68, 2.68, 2.68, 2.68, 6.58, 6.58]
#weight_list_jinqu = [0.41, 2.06, 2.06, 1.00, 4.46, 4.45, 4.45, 7.56, 7.00, 5.01, 9.52, 9.52, 4.20, 4.20, 4.20, 4.20, 10.35, 10.35]

index_weight_wenjian = [3.34, 0.00, 7.55, 7.87, 32.24, 6.25, 26.63]
index_weight_pingheng = [4.67, 0.00, 9.02, 11.66, 21.74, 8.73, 22.44]
index_weight_jinqu = [5.72, 0.00, 10.15, 14.62, 20.32, 10.72, 18.16]
#index_weight_wenjian = [0.54, 2.50, 9.05, 7.50, 46.88, 7.43, 26.11]
#index_weight_pingheng = [0.47, 3.93, 11.37, 11.35, 36.50, 11.81, 24.57]
#index_weight_jinqu = [0.41, 5.12, 13.36, 14.56, 24.02, 16.80, 25.73]
'''
'''
#七月组合权重
code_list_wenjian = ["163415.OF",
"180031.OF",
"513500.OF", "513100.OF",
"159920.OF", "513660.OF", "100061.OF",
"159926.OF", "003358.OF", "001512.OF", "511010.OF", "161821.OF", "000022.OF", "003429.OF", "003987.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_wenjian = [0.00,
4.49,
3.78, 3.77,
3.80, 0.00, 3.79,
0.00, 5.25, 0.00, 5.35, 5.25, 5.25, 5.25, 5.25,
#3.13, 3.12,
1.50, 1.50,
10.86, 10.86]
#weight_list_wenjian = [0.54, 2.50, 4.55, 4.55, 3.75, 3.75, 7.60, 7.60, 7.60, 8.88, 7.60, 7.60, 3.72, 3.72, 10.60, 10.49]

code_list_pingheng = ["163415.OF", "090013.OF",
"180031.OF", "040035.OF",
"513500.OF", "513100.OF",
"159920.OF", "513660.OF", "100061.OF",
"511220.OF", "003358.OF", "159926.OF", "511010.OF", "003429.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_pingheng = [0.00, 0.00,
3.55, 3.50,
4.51, 4.51,
5.81, 0.00, 5.81,
5.06, 5.06, 0.00, 5.06, 5.06,
#4.37, 4.36,
2.10, 2.10,
9.10, 9.00]
#weight_list_pingheng = [0.47, 2.00, 1.93, 5.87, 5.50, 5.85, 5.50, 5.57, 10.32, 10.32, 10.32, 5.90, 5.90, 9.80, 9.75]

code_list_jinqu = ["163415.OF", "090013.OF",
"180031.OF", "000471.OF",
"513500.OF", "096001.OF", "513100.OF",
"159920.OF", "513660.OF", "100061.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF",
"003022.OF", "000434.OF"]

weight_list_jinqu = [0.00, 0.00,
4.66, 4.66,
3.45, 3.35, 3.35,
7.31, 0.00, 7.31,
5.80, 5.82, 5.82,
#2.68, 2.68, 2.68, 2.68,
1.50, 1.50, 1.50, 1.50,
7.65, 7.65]
#weight_list_jinqu = [0.41, 2.06, 2.06, 1.00, 4.46, 4.45, 4.45, 7.56, 7.00, 5.01, 9.52, 9.52, 4.20, 4.20, 4.20, 4.20, 10.35, 10.35]

index_weight_wenjian = [0.00, 4.49, 7.55, 7.59, 31.60, 2.00, 30.97]
index_weight_pingheng = [0.00, 7.05, 9.02, 11.62, 20.24, 3.00, 28.83]
index_weight_jinqu = [0.00, 9.32, 10.15, 14.62, 17.45, 4.00, 27.02]

index_weight_wenjian = [0.00, 4.49, 7.55, 7.59, 31.60, 6.25, 26.72]
index_weight_pingheng = [0.00, 7.05, 9.02, 11.62, 20.24, 8.73, 23.10]
index_weight_jinqu = [0.00, 9.32, 10.15, 14.62, 17.45, 10.72, 20.30]
#index_weight_wenjian = [0.54, 2.50, 9.05, 7.50, 46.88, 7.43, 26.11]
#index_weight_pingheng = [0.47, 3.93, 11.37, 11.35, 36.50, 11.81, 24.57]
#index_weight_jinqu = [0.41, 5.12, 13.36, 14.56, 24.02, 16.80, 25.73]
'''

'''
#八月组合权重
code_list_wenjian = ["163415.OF",
"180031.OF",
"513500.OF", "513100.OF",
"159920.OF", "100061.OF",
"003358.OF", "511010.OF", "161821.OF", "000022.OF", "003429.OF", "003987.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_wenjian = [8.82,
0.00,
3.78, 3.77,
3.80, 3.79,
3.27, 5.00, 5.00, 5.00, 5.00, 5.00,
#3.13, 3.12,
3.63, 3.63,
10.69, 10.69]

code_list_pingheng = ["163415.OF", "090013.OF",
"180031.OF", "040035.OF",
"513500.OF", "513100.OF",
"159920.OF", "100061.OF",
"511220.OF", "003358.OF", "511010.OF", "003429.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_pingheng = [5.00, 5.00,
1.66, 1.66,
5.70, 5.70,
5.88, 5.88,
4.74, 4.74, 4.74, 4.74,
#4.37, 4.36,
5.31, 5.31,
5.00, 5.00]

code_list_jinqu = ["163415.OF", "090013.OF",
"180031.OF", "000471.OF",
"513500.OF", "096001.OF", "513100.OF",
"159920.OF", "100061.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF",
"003022.OF", "000434.OF"]

weight_list_jinqu = [6.00, 6.00,
5.00, 4.97,
4.55, 4.50, 4.50,
9.45, 9.40,
4.99, 4.99, 4.98,
#2.68, 2.68, 2.68, 2.68,
2.68, 2.68, 2.68, 2.68,
0.00, 0.00]

#weight_list_jinqu = [0.41, 2.06, 2.06, 1.00, 4.46, 4.45, 4.45, 7.56, 7.00, 5.01, 9.52, 9.52, 4.20, 4.20, 4.20, 4.20, 10.35, 10.35]


index_weight_wenjian = [0.54, 2.50, 9.05, 7.50, 46.88, 7.43, 26.11]
index_weight_pingheng = [0.47, 3.93, 11.37, 11.35, 36.50, 11.81, 24.57]
index_weight_jinqu = [0.41, 5.12, 13.36, 14.56, 24.02, 16.80, 25.73]
'''

'''
#九月组合权重
code_list_wenjian = ["000577.OF",
"180031.OF",
"513500.OF", "513100.OF",
"159920.OF", "100061.OF",
"003358.OF", "511010.OF", "161821.OF", "000022.OF", "003429.OF", "003987.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_wenjian = [5.58,
2.31,
4.04, 4.03,
4.82,4.82,
2.92, 5.00, 5.00, 5.00, 5.00, 5.00,
#3.13, 3.12,
3.76, 3.76,
10.00, 10.00]

code_list_pingheng = ["000577.OF", "090013.OF",
"180031.OF", "040035.OF",
"513500.OF", "513100.OF",
"159920.OF", "100061.OF",
"511220.OF", "003358.OF", "511010.OF", "003429.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_pingheng = [4.27, 4.00,
1.99, 1.99,
5.20, 5.20,
7.57, 7.56,
4.23, 4.23, 4.74, 4.73,
#4.37, 4.36,
5.68, 5.68,
5.00, 5.00]

code_list_jinqu = ["000577.OF", "090013.OF",
"180031.OF", "000471.OF",
"513500.OF", "096001.OF", "513100.OF",
"159920.OF", "100061.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF",
"003022.OF", "000434.OF"]

weight_list_jinqu = [5.94, 5.94,
3.13, 3.13,
4.44, 4.44, 4.44,
11.32, 11.31,
4.06, 4.06, 4.06,
#2.68, 2.68, 2.68, 2.68,
4.14, 4.14, 4.14, 4.14,
0.00, 0.00]

index_weight_wenjian = [5.58, 2.31, 8.07, 9.64, 27.92, 7.52, 25.00]
index_weight_pingheng = [8.27, 3.97, 10.40, 15.13, 17.94, 11.36, 15.00]
index_weight_jinqu = [11.88, 6.26, 13.32, 22.63, 12.18, 16.55, 5.00]
'''

'''
#十月组合权重
code_list_wenjian = ["000577.OF",
"180031.OF",
"513500.OF", "513100.OF",
"159920.OF", "100061.OF",
"003358.OF", "511010.OF", "161821.OF", "000022.OF", "003429.OF", "003987.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_wenjian = [5.19,
2.25,
3.11, 3.10,
4.95, 4.94,
4.26, 5.00, 5.00, 5.00, 5.00, 5.00,
#3.13, 3.12,
3.79, 3.78,
10.00, 10.00]

code_list_pingheng = ["000577.OF", "090013.OF",
"180031.OF", "040035.OF",
"513500.OF", "513100.OF",
"159920.OF", "100061.OF",
"511220.OF", "003358.OF", "511010.OF", "003429.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_pingheng = [3.74, 3.74,
1.87, 1.87,
3.75, 3.75,
7.56, 7.56,
4.99, 5.00, 5.00, 5.00,
#4.37, 4.36,
5.59, 5.59,
5.00, 5.00]

code_list_jinqu = ["000577.OF", "090013.OF",
"180031.OF", "000471.OF",
"513500.OF", "096001.OF", "513100.OF",
"159920.OF", "100061.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF",
"003022.OF", "000434.OF"]

weight_list_jinqu = [5.16, 5.16,
2.81, 2.81,
2.99, 2.99, 2.99,
10.83, 10.83,
5.46, 5.46, 5.46,
#2.68, 2.68, 2.68, 2.68,
3.91, 3.92, 3.92, 3.92,
0.00, 0.00]

index_weight_wenjian = [5.19, 2.25, 6.21, 9.89, 29.26, 7.57, 25.00]
index_weight_pingheng = [7.48, 3.74, 7.51, 15.12, 19.99, 11.18, 15.00]
index_weight_jinqu = [10.33, 5.62, 8.97, 21.66, 16.38, 15.67, 5.00]
'''

'''
#十一月组合权重
code_list_wenjian = ["000577.OF",
"180031.OF", "040035.OF",
"513500.OF", "513100.OF",
"159920.OF", "100061.OF",
"003358.OF", "511010.OF", "161821.OF", "000022.OF", "003429.OF", "003987.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_wenjian = [1.58,
4.11, 4.11,
1.99, 1.99,
5.57, 5.57,
2.84, 5.00, 5.00, 5.00, 5.00, 5.00,
#3.13, 3.12,
4.16, 4.16,
10.00, 10.00]

code_list_pingheng = ["000577.OF", "090013.OF",
"180031.OF", "040035.OF",
"513500.OF", "513100.OF",
"159920.OF", "100061.OF",
"511220.OF", "003358.OF", "511010.OF", "003429.OF",
"518880.OF", "159934.OF",
"003022.OF", "000434.OF"]

weight_list_pingheng = [1.88, 0.00,
5.57, 5.57,
2.09, 2.09,
7.49, 7.49,
5.20, 5.25, 5.25, 5.25,
#4.37, 4.36,
5.46, 5.46,
5.00, 5.00]

code_list_jinqu = ["000577.OF", "090013.OF",
"180031.OF", "000471.OF",
"513500.OF", "096001.OF", "513100.OF",
"159920.OF", "100061.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF",
"003022.OF", "000434.OF"]

weight_list_jinqu = [2.28, 0.00,
8.09, 8.09,
2.01, 2.01, 0.00,
10.76, 10.76,
5.89, 6.00, 6.00,
#2.68, 2.68, 2.68, 2.68,
3.22, 4.00, 4.00, 4.00,
0.00, 0.00]

index_weight_wenjian = [1.58, 8.22, 3.98, 11.14, 27.84, 8.32, 25.00]
index_weight_pingheng = [1.88, 11.14, 4.19, 14.98, 20.95, 10.92, 15.00]
index_weight_jinqu = [2.28, 16.18, 4.02, 21.52, 17.89, 15.23, 5.00]
'''

'''
#十二月组合
code_list_wenjian = ["510300.OF", "000457.OF",
"513500.OF",
"159920.OF", "100061.OF",
"511220.OF", "511010.OF", "003223.OF", "003327.OF", "003833.OF", "519152.OF",
"518880.OF", "003022.OF"]

weight_list_wenjian = [2.72, 2.72,
4.15,
3.24, 3.24,
7.00, 7.00, 10.50, 10.50, 10.50, 10.50,
15.45, 12.50]

code_list_pingheng = ["510050.OF", "090013.OF",
"510500.OF", "159915.OF", "159902.OF", "000524.OF", "519097.OF",
"513500.OF",
"159920.OF", "100061.OF",
"511220.OF", "511010.OF", "003223.OF", "003327.OF", "003833.OF", "519152.OF",
"518880.OF", "159934.OF",
"511880.OF", "003022.OF"]

weight_list_pingheng = [1.63, 1.63,
1.86, 1.86, 1.85, 2.78, 2.78,
4.97,
4.22, 4.22,
5.96, 5.96, 8.93, 8.93, 8.93, 8.93,
4.77, 4.77,
7.50, 7.50]

code_list_jinqu = ["510050.OF", "090013.OF",
"510500.OF", "159915.OF", "159902.OF", "000524.OF", "002214.OF",
"513500.OF", "096001.OF",
"159920.OF", "513660.OF", "100061.OF",
"511220.OF", "511010.OF", "003223.OF", "003327.OF", "003833.OF", "519152.OF",
"518880.OF", "159934.OF",
"511880.OF"]

weight_list_jinqu = [0.76, 3.93,
3.05, 3.04, 3.04, 4.56, 4.56,
2.13, 2.12,
3.58, 3.58, 7.15,
5.37, 5.37, 8.06, 8.06, 8.05, 8.05,
5.33, 5.33,
5.00]

index_weight_wenjian = [0.00, 5.44, 0.00, 4.15, 6.48, 55.99, 0.00, 27.95]
index_weight_pingheng = [3.26, 0.00, 11.13, 4.97, 8.45, 47.65, 9.54, 15.00]
index_weight_jinqu = [4.69, 0.00, 18.26, 4.15, 14.30, 42.94, 10.66, 5.00]
'''

'''
#一月组合
code_list_wenjian = ["510300.OF",
"513500.OF",
"159920.OF", "510900.OF",
"004614.OF", "519152.OF", "000128.OF", "002086.OF", "003796.OF", "003795.OF",
"518880.OF", "511990.OF"]

weight_list_wenjian = [6.31,
5.06,
4.32, 4.31,
8.22, 8.21, 8.21, 8.21, 8.21, 8.21,
15.73, 15.00]

code_list_pingheng = ["510050.OF",
"510500.OF", "519097.OF", "000457.OF",
"513500.OF", "513100.OF",
"159920.OF", "513660.OF", "040018.OF",
"004614.OF", "519152.OF", "000128.OF", "002086.OF",
"518880.OF", "159934.OF",
"511880.OF", "511990.OF"]

weight_list_pingheng = [6.93,
5.03, 2.51, 2.51,
3.39, 3.39,
4.94, 4.94, 9.88,
8.00, 8.00, 8.00, 8.00,
4.73, 4.73,
7.50, 7.50]

code_list_jinqu = ["510050.OF",
                   "510500.OF", "090013.OF", "180031.OF",
                   "513500.OF", "096001.OF",
                   "159920.OF", "510900.OF", "100061.OF",
                   "004614.OF", "519152.OF", "000128.OF", "002086.OF",
                   "159937.OF", "159934.OF",
                   "511880.OF"]

weight_list_jinqu = [8.73,
6.50, 3.25, 3.25,
3.71, 3.71,
6.41, 6.41, 12.83,
7.40, 7.40, 7.40, 7.40,
5.29, 5.29,
5.00]

index_weight_wenjian = [0.00, 6.31, 0.00, 5.06, 8.63, 49.27, 0.00, 30.73]
index_weight_pingheng = [6.93, 0.00, 10.06, 6.79, 19.77, 32.00, 9.46, 15.00]
index_weight_jinqu = [8.73, 0.00, 13.01, 7.41, 25.66, 29.61, 10.58, 5.00]
'''
'''
#二月组合
code_list_wenjian = ["510050.OF",
"510500.OF",
"513500.OF",
"159920.OF", "510900.OF",
"004614.OF", "519152.OF", "000128.OF", "002086.OF", "003796.OF", "003795.OF",
"511880.OF", "511990.OF"]

weight_list_wenjian = [5.73,
5.94,
6.44,
4.00, 4.00,
8.15, 8.15, 8.15, 8.15, 8.15, 8.15,
12.50, 12.50]

code_list_pingheng = ["510050.OF",
"513500.OF", "513100.OF",
"159920.OF", "513660.OF", "040018.OF",
"004614.OF", "519152.OF", "000128.OF", "002086.OF",
"518880.OF",
"511880.OF", "511990.OF"]

weight_list_pingheng = [10.03,
5.18, 5.18,
4.81, 4.81, 9.61,
9.59, 9.59, 9.59, 9.59,
7.02,
7.50, 7.50]

code_list_jinqu = ["510050.OF",
                   "513500.OF", "096001.OF", "513100.OF",
                   "159920.OF", "510900.OF", "100061.OF",
                   "004614.OF", "519152.OF", "000128.OF", "002086.OF",
                   "159937.OF", "159934.OF",
                   "511880.OF"]

weight_list_jinqu = [14.58,
4.37, 4.37, 4.37,
7.22, 7.22, 14.44,
7.13, 7.13, 7.13, 7.13,
4.95, 4.95,
5.00]

index_weight_wenjian = [5.73, 0.00, 5.94, 6.44, 7.99, 48.90, 0.00, 25.00]
index_weight_pingheng = [10.03, 0.00, 0.00, 10.36, 19.23, 38.36, 7.02, 15.00]
index_weight_jinqu = [14.58, 0.00, 0.00, 13.11, 28.87, 28.54, 9.90, 5.00]
'''

#三月/四月组合/五月组合
code_list_wenjian = ["510050.OF",
"513500.OF",
"159920.OF", "510900.OF",
"004614.OF", "519152.OF", "000128.OF", "002086.OF", "003796.OF", "003795.OF",
"511880.OF", "511990.OF"]

weight_list_wenjian = [1.81,
7.70,
3.23, 3.25,
9.84, 9.84, 9.84, 9.84, 9.84, 9.84,
12.50, 12.50]

code_list_pingheng = ["510050.OF", "000457.OF",
"519097.OF",
"513500.OF", "513100.OF",
"159920.OF", "513660.OF", "040018.OF",
"004614.OF", "519152.OF", "000128.OF", "002086.OF",
"518880.OF",
"511880.OF", "511990.OF"]

weight_list_pingheng = [1.93, 4.22,
5.80,
4.31, 4.31,
4.25, 4.25, 8.50,
9.53, 9.53, 9.53, 9.53,
9.30,
7.50, 7.50]

code_list_jinqu = ["510050.OF", "090013.OF",
"180031.OF", "002214.OF",
"513500.OF", "096001.OF", "513100.OF",
"159920.OF", "510900.OF", "100061.OF",
"004614.OF", "519152.OF", "000128.OF", "002086.OF",
"159937.OF", "159934.OF",
"511880.OF"]

weight_list_jinqu = [10.45, 4.26,
4.14, 4.25,
3.53, 3.53, 3.53,
7.84, 7.84, 15.68,
5.91, 5.91, 5.91, 5.91,
5.20, 5.20,
5.00]

index_weight_wenjian = [1.81, 0.00, 0.00, 7.70, 6.45, 59.04, 0.00, 25.00]
#A股-超大、大、中小，美，港，债，黄金，货币
index_weight_pingheng = [6.15, 0.00, 5.80, 8.363, 17.01, 38.11, 9.31, 15.00]
index_weight_jinqu = [10.62, 0.00, 8.38, 10.59, 31.35, 23.65, 10.40, 5.00]

new_fund_list = []
for each in [code_list_wenjian, code_list_pingheng, code_list_jinqu]:
    for i in each:
        indicator = 0
        for j in index_code_list:
            if i in index_fund_map[j]:
                indicator = 1
                break
            else:
                pass
        if indicator == 0:
            new_fund_list.append(i)
        else:
            pass
if new_fund_list != []:
    print set(new_fund_list)
    raise Exception("new fund!")
else:
    pass

Return_Decomposition(index_code_list, code_list_wenjian, index_weight_wenjian, weight_list_wenjian, "wenjian")
Return_Decomposition(index_code_list, code_list_pingheng, index_weight_pingheng, weight_list_pingheng, "pingheng")
Return_Decomposition(index_code_list, code_list_jinqu, index_weight_jinqu, weight_list_jinqu, "jinqu")
