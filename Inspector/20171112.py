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
w.start()


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
        index_fund_return_list.append([sum(index_fund_return)/sum(index_fund_weight), sum(index_fund_contri)])
    index_fund_decom = pd.DataFrame(np.array(index_fund_return_list), index=indexn, columns=[u"子基金组合收益率", u"子基金组合贡献率"])
    index_decom_new = pd.merge(index_decom, index_fund_decom, left_index=True, right_index=True)
    index_decom_new.index = w.wss(indexn, "sec_name").Data[0]
    index_decom_new.to_excel(u"Z:\\Mac 上的 WangBin-Mac\\index_decom_" + port_name + ".xlsx")
    index_data.to_csv(u"Z:\\Mac 上的 WangBin-Mac\\index.csv")

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
    fund_decom_new.to_excel(u"Z:\\Mac 上的 WangBin-Mac\\fund_decom_" + port_name + ".xlsx")
    nav_data.to_csv(u"Z:\\Mac 上的 WangBin-Mac\\nav_" + port_name + ".csv")


index_code_list = ["000300.SH", "000905.SH", "SPX.GI", "HSI.HI", "H11001.CSI", "AU9999.SGE", "H11025.CSI"]

index_fund_map = {"000300.SH" : ["163415.OF", "090013.OF", "000308.OF", "519736.OF", "000577.OF"],
"000905.SH" : ["180031.OF", "000547.OF", "000524.OF", "040035.OF", "000471.OF"],
"SPX.GI" : ["513500.OF", "519981.OF", "096001.OF", "513100.OF", "100061.OF"],
"HSI.HI" : ["159920.OF", "513660.OF"],
"H11001.CSI" : ["511220.OF", "001021.OF", "003358.OF", "511010.OF", "161821.OF", "159926.OF", "001512.OF", "000022.OF", "003429.OF", "003987.OF"],
"AU9999.SGE" : ["518880.OF", "159937.OF", "159934.OF", "320013.OF", "518800.OF"],
"H11025.CSI" : ["003022.OF", "000434.OF"]}


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



pre_end_date = "2017-01-31"
start_date = "2017-02-01"
end_date = "2017-02-28"

index_weight_wenjian[6] = index_weight_wenjian[6] + index_weight_wenjian[4]/2 + index_weight_wenjian[5]
index_weight_wenjian[5] = 0.00
weight_list_wenjian[-1] = weight_list_wenjian[-1] + weight_list_wenjian[-3] + index_weight_wenjian[4]/4
weight_list_wenjian[-2] = weight_list_wenjian[-2] + weight_list_wenjian[-4] + index_weight_wenjian[4]/4
weight_list_wenjian[-3] = 0.00
weight_list_wenjian[-4] = 0.00


Return_Decomposition(index_code_list, code_list_wenjian, index_weight_wenjian, weight_list_wenjian, "wenjian"+end_date)
#Return_Decomposition(index_code_list, code_list_pingheng, index_weight_pingheng, weight_list_pingheng, "pingheng")
#Return_Decomposition(index_code_list, code_list_jinqu, index_weight_jinqu, weight_list_jinqu, "jinqu")
