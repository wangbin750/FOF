# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import datetime as dt
from WindPy import *
import xlrd
from xlwt import Workbook, easyxf
from xlutils.copy import copy
w.start()


def Wsd_Data_Install(code, fields, startdate, enddate):
    temp_data = w.wsd(code, fields, startdate, enddate)
    if temp_data.ErrorCode == 0:
        data = pd.DataFrame(np.array(temp_data.Data).T, index=temp_data.Times, columns=temp_data.Fields)
        return data
    else:
        print "ErrodCode=%s" % temp_data.ErrorCode
        raise Exception("error in data install!")

def Publicfund_Performance(code, startdate, enddate):
    startdate = str(w.tdaysoffset(-1, startdate).Data[0][0])[:10]
    wsd_data = Wsd_Data_Install(code, "NAV_adj", startdate, enddate).pct_change()
    wsd_data.columns = ["PCT_CHG"]
    wsd_data = wsd_data.dropna()
    today_data = wsd_data.iloc[-1,0] * 100
    upper_quantile = wsd_data.iloc[:-1,0].quantile(0.85) * 100
    lower_quantile = wsd_data.iloc[:-1,0].quantile(0.15) * 100
    data = pd.DataFrame(np.array([today_data, upper_quantile, lower_quantile]), index=[u"今日值", u"上10%分位数", u"下10%分位数"], columns=[u"净值涨跌幅"])
    return data, wsd_data

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

def Wsi_Data_Install(code, startdate, enddate, interval):
    temp_data = w.wsi(code, "close", startdate, enddate, "BarSize=%s"%interval)
    if temp_data.ErrorCode == 0:
        data = pd.DataFrame(np.array(temp_data.Data).T, index=temp_data.Times, columns=temp_data.Fields)
        return data
    else:
        return "no wsi data"


def Plot(data):
    plot_data = data/100 + 1
    plot_data = plot_data.cumprod()
    data.plot()
    plt.savefig("file_path"+"test.png")

def Bondindex_Performance(code, startdate, enddate):
    startdate = str(w.tdaysoffset(-1, startdate).Data[0][0])[:10]
    wsd_data = Wsd_Data_Install(code, ["pct_chg", "amt"], startdate, enddate)
    today_data = list(wsd_data.iloc[-1])
    upper_quantile = list(wsd_data.iloc[:-1].quantile(0.9))
    lower_quantile = list(wsd_data.iloc[:-1].quantile(0.1))
    data = pd.DataFrame(np.array([today_data, upper_quantile, lower_quantile]), index=[u"今日值", u"上10%分位数", u"下10%分位数"], columns=[u"涨跌幅", u"成交额"])
    return data, wsd_data

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

def Xls_Writer_pctchg(ws, data, all_data, rowno, colno, field):
    if field in [u"涨跌幅", u"净值涨跌幅", u"利率值"]:
        if data[u"今日值"] < data[u"下10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_down2)
            Plot(all_data["pct_chg"])
        elif data[u"今日值"] > data[u"上10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_up2)
        else:
            if data[u"今日值"] >= 0:
                ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_up)
            else:
                ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_down)
        try:
            pctchg = all_data["PCT_CHG"][:-1]
            z_score = (all_data["PCT_CHG"][-1] - np.mean(pctchg))/np.std(pctchg)
        except:
            try:
                pctchg = all_data["CLOSE3"][:-1]
                z_score = (all_data["CLOSE3"][-1] - np.mean(pctchg))/np.std(pctchg)
            except:
                z_score = np.nan
        if z_score < -2:
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_down2)
        elif (z_score >= -2) and (z_score < -1):
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_down)
        elif (z_score >= -1) and (z_score <= 1):
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_normal)
        elif (z_score > 1) and (z_score <= 2):
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_up)
        elif z_score > 2:
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_up2)
        ws.write(rowno+1, colno, "%0.3f%%"%data[u"下10%分位数"], style_quan)
        ws.write(rowno+1, colno+1, "%0.3f%%"%data[u"上10%分位数"], style_quan)
    else:
        raise Exception("not defined field!")

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


code_list_wenjian = ["180031.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"159926.OF", "003358.OF", "001512.OF", "511010.OF", "161821.OF", "000022.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_wenjian = [4.47, 3.78, 3.78, 4.33, 4.33, 5.00, 5.00, 5.00, 6.25, 5.00, 5.00,
3.72, 3.72, 10.00, 10.00]
weight_list_wenjian = np.array(weight_list_wenjian)/100

code_list_pingheng = ["180031.OF", "040035.OF", "513500.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "159926.OF", "511010.OF",
"518880.OF", "159934.OF", "003022.OF", "000434.OF"]

weight_list_pingheng = [3.78, 3.78, 4.79, 4.79, 7.27, 7.27, 1.00, 5.75, 5.75, 5.75, 5.90, 5.90, 8.00, 8.00]
weight_list_pingheng = np.array(weight_list_pingheng)/100

code_list_jinqu = ["180031.OF", "040035.OF", "000471.OF", "513500.OF", "096001.OF", "513100.OF", "159920.OF", "513660.OF",
"511220.OF", "003358.OF", "511010.OF",
"518880.OF", "159934.OF", "159937.OF", "518800.OF", "003022.OF", "000434.OF"]

weight_list_jinqu = [3.70, 3.70, 3.70, 3.95, 3.95, 3.95, 10.64, 10.64, 1.00, 5.51, 5.51, 4.20, 4.20, 4.20, 4.20, 6.50, 6.50]
weight_list_jinqu = np.array(weight_list_jinqu)/100

today_date = dt.date.today().isoformat()
today_date = "2017-02-21"
daily_backtest_start_date = str(w.tdaysoffset(-121, today_date).Data[0][0])[:10]
pnv_wenjian = 1.007895626
pnv_pingheng = 1.012527976
pnv_jinqu = 1.016862842
nv_wenjian = Portfolio_Net_Value(code_list_wenjian, weight_list_wenjian, pnv_wenjian, today_date)
nv_pingheng = Portfolio_Net_Value(code_list_pingheng, weight_list_pingheng, pnv_pingheng, today_date)
nv_jinqu = Portfolio_Net_Value(code_list_jinqu, weight_list_jinqu, pnv_jinqu, today_date)
file_path = "D:\\"

'''
格式定义
'''
border_style = "border: left thin, right thin, top thin, bottom thin;"
style_normal = easyxf("font: colour white;" "pattern: pattern solid, fore_colour gray40;" "align: vertical center, horizontal center;" "%s" % border_style)
style_down = easyxf("font: colour black;" "pattern: pattern solid, fore_colour light_green;" "align: vertical center, horizontal center;" "%s" % border_style)
style_up = easyxf("font: colour white;" "pattern: pattern solid, fore_colour orange;" "align: vertical center, horizontal center;" "%s" % border_style)
style_down2 = easyxf("font: colour white;" "pattern: pattern solid, fore_colour green;" "align: vertical center, horizontal center;" "%s" % border_style)
style_up2 = easyxf("font: colour white;" "pattern: pattern solid, fore_colour red;" "align: vertical center, horizontal center;" "%s" % border_style)
style_low = easyxf("font: colour white;" "pattern: pattern solid, fore_colour blue;" "align: vertical center, horizontal center;" "%s" % border_style)
style_high = easyxf("font: colour black;" "pattern: pattern solid, fore_colour yellow;" "align: vertical center, horizontal center;" "%s" % border_style)
style_name = easyxf("font: colour black;" "pattern: pattern solid, fore_colour white;" "align: vertical center, horizontal left;" "%s" % border_style)
style_quan = easyxf("font: colour black;" "pattern: pattern solid, fore_colour white;" "align: vertical center, horizontal center;" "%s" % border_style)
ws = Workbook()


ws_wenjian = ws.add_sheet(u"稳健配置")
ws_wenjian.write(0, 0, u"今日净值")
ws_wenjian.write(0, 1, "%0.5f"%nv_wenjian)
ws_wenjian.write(1, 0, u"净值涨跌幅")
ws_wenjian.write(1, 1, u"%0.3f‱"%((nv_wenjian/pnv_wenjian - 1)*10000))

ws_wenjian.write(3, 0, u"净值涨跌幅")
row_no = 4
for each_code in code_list_wenjian:
    temp_result, all_data = Publicfund_Performance(each_code, daily_backtest_start_date, today_date)
    ws_wenjian.write_merge(row_no, row_no+1, 0, 0, each_code, style_name)
    col_no = 1
    temp_series = temp_result[u"净值涨跌幅"]
    Xls_Writer_pctchg(ws_wenjian, temp_series, all_data, row_no, col_no, u"净值涨跌幅")
    row_no = row_no + 2


ws_pingheng = ws.add_sheet(u"平衡配置")
ws_pingheng.write(0, 0, u"今日净值")
ws_pingheng.write(0, 1, "%0.5f"%nv_pingheng)
ws_pingheng.write(1, 0, u"净值涨跌幅")
ws_pingheng.write(1, 1, u"%0.3f‱"%((nv_pingheng/pnv_pingheng - 1)*10000))

ws_pingheng.write(3, 0, u"净值涨跌幅")
row_no = 4
for each_code in code_list_pingheng:
    temp_result, all_data = Publicfund_Performance(each_code, daily_backtest_start_date, today_date)
    ws_pingheng.write_merge(row_no, row_no+1, 0, 0, each_code, style_name)
    col_no = 1
    temp_series = temp_result[u"净值涨跌幅"]
    Xls_Writer_pctchg(ws_pingheng, temp_series, all_data, row_no, col_no, u"净值涨跌幅")
    row_no = row_no + 2


ws_jinqu = ws.add_sheet(u"进取配置")
ws_jinqu.write(0, 0, u"今日净值")
ws_jinqu.write(0, 1, "%0.5f"%nv_jinqu)
ws_jinqu.write(1, 0, u"净值涨跌幅")
ws_jinqu.write(1, 1, u"%0.3f‱"%((nv_jinqu/pnv_jinqu - 1)*10000))

ws_jinqu.write(3, 0, u"净值涨跌幅")
row_no = 4
for each_code in code_list_jinqu:
    temp_result, all_data = Publicfund_Performance(each_code, daily_backtest_start_date, today_date)
    ws_jinqu.write_merge(row_no, row_no+1, 0, 0, each_code, style_name)
    col_no = 1
    temp_series = temp_result[u"净值涨跌幅"]
    Xls_Writer_pctchg(ws_jinqu, temp_series, all_data, row_no, col_no, u"净值涨跌幅")
    row_no = row_no + 2

ws_index = ws.add_sheet(u"大类资产")
stock_index_list = ["000300.SH", "000905.SH", "SPX.GI", "HSI.HI"]
bond_index_list = ["H11001.CSI"]

col_no = 1
for each in [u"涨跌幅", u"振幅", u"日内波动率", u"成交额(亿元)", u"换手率"]:
    ws_index.write_merge(0, 0, col_no, col_no+1, each, style_quan)
    col_no = col_no + 2

row_no = 1

for each_index in stock_index_list:
    if each_index == "SPX.GI":
        temp_date = str(w.tdaysoffset(-1, today_date).Data[0][0])[:10]
        temp_end_date = str(w.tdaysoffset(-121, temp_date).Data[0][0])[:10]
        temp_result, all_data = Index_ETF_Performance(each_index, temp_end_date, temp_date, 5)
    else:
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


ws.save("D:\\test_daily.xls")
