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


# 获取日度数据
def Wsd_Data_Install(code, fields, startdate, enddate):
    temp_data = w.wsd(code, fields, startdate, enddate)
    if temp_data.ErrorCode == 0:
        data = pd.DataFrame(np.array(temp_data.Data).T, index=temp_data.Times, columns=temp_data.Fields)
        return data
    else:
        print "ErrodCode=%s" % temp_data.ErrorCode
        raise Exception("error in data install!")

# 获取日内数据
# interval-周期，以分钟计
def Wsi_Data_Install(code, startdate, enddate, interval):
    temp_data = w.wsi(code, "close", startdate, enddate, "BarSize=%s"%interval)
    if temp_data.ErrorCode == 0:
        data = pd.DataFrame(np.array(temp_data.Data).T, index=temp_data.Times, columns=temp_data.Fields)
        return data
    else:
        return "no wsi data"

# 计算日内波动率
def Intraday_Volatility(code, startdate, enddate, interval):
    enddate = dt.datetime.strptime(enddate, "%Y-%m-%d") + timedelta(1)
    enddate = enddate.strftime("%Y-%m-%d")
    date_set = pd.date_range(startdate, enddate, freq="D")
    data = Wsi_Data_Install(code, startdate, enddate, interval)
    if type(data) == str:
        return pd.DataFrame(columns=["intraday_std"])
    else:
        intraday_std_list = list()
        date_list = list()
        for each in range(len(date_set) - 1):
            data_temp = data.loc[date_set[each]:date_set[each+1]]
            if data_temp.empty:
                pass
            else:
                intraday_std = 100 * (data_temp.pct_change().std() * ((240.0/interval)**0.5))
                intraday_std_list.append(intraday_std)
                date_list.append(date_set[each])
        intraday_std_frame = pd.DataFrame(np.array(intraday_std_list), index=date_list, columns=["intraday_std"])
        return intraday_std_frame

test = pd.DataFrame(np.random.randn(100))
test.std()
# 计算资产组合的当日净值
# code_list只包含公募类产品，weight_list包含所有产品（含私募类产品），私募类产品的权重应在weight_list的最后
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
            pct_chg_list.append(temp_data.Data[0][0])
    if len(pct_chg_list) == len(code_list):
        pct_chg_list = pct_chg_list + [0.0] * (len(weight_list) - len(code_list))
        nv_pct_chg = np.array(pct_chg_list) * np.array(weight_list)
        nv_today = previous_nv * (1 + nv_pct_chg)
        return nv_today
    else:
        raise Exception("missing data!")

def Index_ETF_Performance(code, startdate, enddate, interval):
    wsd_data = Wsd_Data_Install(code, ["pct_chg", "swing", "amt", "free_turn"], startdate, enddate)
    today_data = list(wsd_data.iloc[-1])
    upper_quantile = list(wsd_data.iloc[:-1].quantile(0.9))
    lower_quantile = list(wsd_data.iloc[:-1].quantile(0.1))
    intraday_std_data = Intraday_Volatility(code, startdate, enddate, interval)
    if intraday_std_data.empty:
        today_data.append(np.nan)
        upper_quantile.append(np.nan)
        lower_quantile.append(np.nan)
    else:
        today_data.append(intraday_std_data.iloc[-1,0])
        upper_quantile.append(intraday_std_data.iloc[0:-1,0].quantile(0.9))
        lower_quantile.append(intraday_std_data.iloc[0:-1,0].quantile(0.1))
    data = pd.DataFrame(np.array([today_data, upper_quantile, lower_quantile]), index=[u"今日值", u"上10%分位数", u"下10%分位数"], columns=[u"涨跌幅", u"振幅", u"成交额", u"换手率", u"日内波动率"])
    return data, wsd_data

def Bondindex_Performance(code, startdate, enddate):
    startdate = str(w.tdaysoffset(-1, startdate).Data[0][0])[:10]
    wsd_data = Wsd_Data_Install(code, ["pct_chg", "amt"], startdate, enddate)
    today_data = list(wsd_data.iloc[-1])
    upper_quantile = list(wsd_data.iloc[:-1].quantile(0.9))
    lower_quantile = list(wsd_data.iloc[:-1].quantile(0.1))
    data = pd.DataFrame(np.array([today_data, upper_quantile, lower_quantile]), index=[u"今日值", u"上10%分位数", u"下10%分位数"], columns=[u"涨跌幅", u"成交额"])
    return data, wsd_data

def Publicfund_Performance(code, startdate, enddate):
    startdate = str(w.tdaysoffset(-1, startdate).Data[0][0])[:10]
    wsd_data = Wsd_Data_Install(code, "NAV_adj", startdate, enddate).pct_change()
    today_data = wsd_data.iloc[-1,0]
    upper_quantile = wsd_data.iloc[:-1,0].quantile(0.9)
    lower_quantile = wsd_data.iloc[:-1,0].quantile(0.1)
    data = pd.DataFrame(np.array([today_data, upper_quantile, lower_quantile]), index=[u"今日值", u"上10%分位数", u"下10%分位数"], columns=[u"净值涨跌幅"])
    return data, wsd_data

def Xls_Writer(ws, data, rowno, colno, field):
    if field in [u"涨跌幅", u"振幅", u"日内波动率", u"换手率"]:
        if data[u"今日值"] < data[u"下10%分位数"]:
            ws.write_merge(rowno, rowno, colno, colno+1, "%0.3f%%"%data[u"今日值"], style_low)
        elif data[u"今日值"] > data[u"上10%分位数"]:
            ws.write_merge(rowno, rowno, colno, colno+1, "%0.3f%%"%data[u"今日值"], style_high)
        else:
            ws.write_merge(rowno, rowno, colno, colno+1, "%0.3f%%"%data[u"今日值"], style_normal)
        ws.write(rowno+1, colno, "%0.3f%%"%data[u"下10%分位数"], style_quan)
        ws.write(rowno+1, colno+1, "%0.3f%%"%data[u"上10%分位数"], style_quan)
    elif field in [u"成交额"]:
        if data[u"今日值"] < data[u"下10%分位数"]:
            ws.write_merge(rowno, rowno, colno, colno+1, "%0.2f"%(data[u"今日值"]/100000000), style_low)
        elif data[u"今日值"] > data[u"上10%分位数"]:
            ws.write_merge(rowno, rowno, colno, colno+1, "%0.2f"%(data[u"今日值"]/100000000), style_high)
        else:
            ws.write_merge(rowno, rowno, colno, colno+1, "%0.2f"%(data[u"今日值"]/100000000), style_normal)
        ws.write(rowno+1, colno, "%0.2f"%(data[u"下10%分位数"]/100000000), style_quan)
        ws.write(rowno+1, colno+1, "%0.2f"%(data[u"上10%分位数"]/100000000), style_quan)
    else:
        raise Exception("not defined field!")

def Xls_Writer_pctchg(ws, data, all_data, rowno, colno, field):
    if field in [u"涨跌幅"]:
        if data[u"今日值"] < data[u"下10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_low)
        elif data[u"今日值"] > data[u"上10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_high)
        else:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_normal)
        pctchg = all_data["PCT_CHG"][:-1]
        z_score = (all_data["PCT_CHG"][-1] - np.mean(pctchg))/np.std(pctchg)
        if z_score < -2:
            ws.write(rowno, colno+1, "%0.3f"%z_score, easyxf("font: colour white;" "pattern: pattern solid, fore_colour purple;" "align: vertical center, horizontal center;" "%s" % border_style))
        elif (z_score >= -2) and (z_score < -1):
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_low)
        elif (z_score >= -1) and (z_score <= 1):
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_normal)
        elif (z_score > 1) and (z_score <= 2):
            ws.write(rowno, colno+1, "%0.3f"%z_score, easyxf("font: colour white;" "pattern: pattern solid, fore_colour orange;" "align: vertical center, horizontal center;" "%s" % border_style))
        elif z_score > 2:
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_high)
        ws.write(rowno+1, colno, "%0.3f%%"%data[u"下10%分位数"], style_quan)
        ws.write(rowno+1, colno+1, "%0.3f%%"%data[u"上10%分位数"], style_quan)
    else:
        raise Exception("not defined field!")
'''
close:收盘价
amt:交易量
pct_chg:涨跌幅
swing:振幅
free_turn:换手率（以自由流通股本计）
NAV_adj:复权净值
NAV_acc:累计净值
'''

day_shift = 120
week_shift = 52
month_shift = 36

today_date = dt.date.today().isoformat()
today_date = "2017-02-13"
#print today_date

daily_backtest_start_date = str(w.tdaysoffset(-day_shift, today_date).Data[0][0])[:10]
#print daily_backtest_start_date

'''
格式定义
'''
border_style = "border: left thin, right thin, top thin, bottom thin;"
style_normal = easyxf("font: colour white;" "pattern: pattern solid, fore_colour green;" "align: vertical center, horizontal center;" "%s" % border_style)
style_low = easyxf("font: colour white;" "pattern: pattern solid, fore_colour blue;" "align: vertical center, horizontal center;" "%s" % border_style)
style_high = easyxf("font: colour white;" "pattern: pattern solid, fore_colour red;" "align: vertical center, horizontal center;" "%s" % border_style)
style_name = easyxf("font: colour black;" "pattern: pattern solid, fore_colour white;" "align: vertical center, horizontal left;" "%s" % border_style)
style_quan = easyxf("font: colour black;" "pattern: pattern solid, fore_colour white;" "align: vertical center, horizontal center;" "%s" % border_style)
ws = Workbook()
ws_index = ws.add_sheet("Index")

stock_index_list = ["000300.SH", "000905.SH", "SPX.GI", "HSI.HI"]
bond_index_list = ["H11001.CSI"]

col_no = 1
for each in [u"涨跌幅", u"振幅", u"日内波动率", u"成交额(亿元)", u"换手率"]:
    ws_index.write_merge(0, 0, col_no, col_no+1, each, style_quan)
    col_no = col_no + 2

row_no = 1

for each_index in stock_index_list:
    temp_result, all_data = Index_ETF_Performance(each_index, daily_backtest_start_date, today_date, 5)
    ws_index.write_merge(row_no, row_no+1, 0, 0, each_index, style_name)
    col_no = 1
    temp_series = temp_result[u"涨跌幅"]
    Xls_Writer_pctchg(ws_index, temp_series, all_data, row_no, col_no, u"涨跌幅")
    col_no = col_no + 2
    for each_field in [u"振幅", u"日内波动率", u"成交额", u"换手率"]:
        temp_series = temp_result[each_field]
        Xls_Writer(ws_index, temp_series, row_no, col_no, each_field)
        col_no = col_no + 2
    row_no = row_no + 2

row_no = row_no + 1

for each_index in bond_index_list:
    temp_result, all_data = Bondindex_Performance(each_index, daily_backtest_start_date, today_date)
    ws_index.write_merge(row_no, row_no+1, 0, 0, each_index, style_name)
    col_no = 1
    for each_field in [u"涨跌幅", u"振幅", u"日内波动率", u"成交额", u"换手率"]:
        try:
            temp_series = temp_result[each_field]
            Xls_Writer(ws_index, temp_series, row_no, col_no, each_field)
        except:
            pass
        col_no = col_no + 2
    row_no = row_no + 2

ws.save("D:\\test.xls")
