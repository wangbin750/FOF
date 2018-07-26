# coding=utf-8

import numpy as np
import pandas as pd
from WindPy import *
from xlwt import Workbook, easyxf
w.start()

import socket
hostname = socket.gethostname()
if hostname == "DESKTOP-OGC5NH7":
    path = u"E:/263网盘/金建投资/FOF投顾/周报/"
elif hostname == "localhost":
    path = u"/Users/WangBin-Mac/263网盘/金建投资/FOF投顾/周报/"
elif hostname == "CICCB6CR7213VFT":
    path = u"F:/263网盘/金建投资/FOF投顾/周报/"


forwardweeks = 54 #数据应包含上周数据及之前52周
enddate = "2017-12-08"

def Wsd_Data_Install_Weekly(code, fields, enddate, forwardweeks):
    temp_data = w.wsd(code, fields, "ED-%sW"%forwardweeks, enddate, "Period=W")
    if temp_data.ErrorCode == 0:
        data = pd.DataFrame(np.array(temp_data.Data).T, index=temp_data.Times, columns=temp_data.Fields)
        return data
    else:
        print "ErrodCode=%s" % temp_data.ErrorCode
        raise Exception("error in data install: " + code)

def Cal_Cdf(data, point):
    if point > data.max():
        return (point/data.max())
    elif point < data.min():
        return (point/data.min() - 1)
    else:
        percent_a = 0.5
        quan_a = data.quantile(percent_a)
        if point == quan_a:
            return percent_a
        elif point > quan_a:
            percent_b = 1
        else:
            percent_a = 0
            percent_b = 0.5
        quan_b = data.quantile(percent_b)
        diff = percent_b - percent_a
        while abs(diff) > 0.00001:
            percent_c = (percent_a + percent_b)/2
            quan_c = data.quantile(percent_c)
            if quan_c == point:
                return percetn_c
            elif quan_c > point:
                percent_b = percent_c
            else:
                percent_a = percent_c
            diff = percent_b - percent_a

        return (percent_a + percent_b)/2

def Index_Performance(code, enddate, forwardweeks):
    wsd_data = Wsd_Data_Install_Weekly(code, ["pct_chg", "swing", "amt", "free_turn"], enddate, forwardweeks)
    today_data = list(wsd_data.iloc[-1])
    upper_quantile = list(wsd_data.iloc[:-1].quantile(0.85))
    lower_quantile = list(wsd_data.iloc[:-1].quantile(0.15))
    data = pd.DataFrame(np.array([today_data, upper_quantile, lower_quantile]), index=[u"今日值", u"上10%分位数", u"下10%分位数"], columns=[u"涨跌幅", u"振幅", u"成交额", u"换手率"])
    return data, wsd_data

def Bondindex_Performance(code, enddate, forwardweeks):
    wsd_data = Wsd_Data_Install_Weekly(code, ["pct_chg", "swing", "amt"], enddate, forwardweeks)
    today_data = list(wsd_data.iloc[-1])
    upper_quantile = list(wsd_data.iloc[:-1].quantile(0.85))
    lower_quantile = list(wsd_data.iloc[:-1].quantile(0.15))
    data = pd.DataFrame(np.array([today_data, upper_quantile, lower_quantile]), index=[u"今日值", u"上10%分位数", u"下10%分位数"], columns=[u"涨跌幅", u"振幅", u"成交额"])
    return data, wsd_data

def Foreignindex_Performance(code, enddate, forwardweeks):
    wsd_data = Wsd_Data_Install_Weekly(code, ["pct_chg", "swing"], enddate, forwardweeks)
    today_data = list(wsd_data.iloc[-1])
    upper_quantile = list(wsd_data.iloc[:-1].quantile(0.85))
    lower_quantile = list(wsd_data.iloc[:-1].quantile(0.15))
    data = pd.DataFrame(np.array([today_data, upper_quantile, lower_quantile]), index=[u"今日值", u"上10%分位数", u"下10%分位数"], columns=[u"涨跌幅", u"振幅"])
    return data, wsd_data

def Interest_Performance(code, enddate, forwardweeks):
    wsd_data = Wsd_Data_Install_Weekly(code, ["close3"], enddate, forwardweeks)
    today_data = list(wsd_data.iloc[-1])
    upper_quantile = list(wsd_data.iloc[:-1].quantile(0.85))
    lower_quantile = list(wsd_data.iloc[:-1].quantile(0.15))
    data = pd.DataFrame(np.array([today_data, upper_quantile, lower_quantile]), index=[u"今日值", u"上10%分位数", u"下10%分位数"], columns=[u"利率值"])
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
    if field in [u"涨跌幅", u"利率值"]:
        if data[u"今日值"] < data[u"下10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_down2)
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

def Xls_Writer_interest(ws, data, all_data, rowno, colno, field):
    if field in [u"涨跌幅", u"利率值"]:
        if data[u"今日值"] < data[u"下10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_low)
        elif data[u"今日值"] > data[u"上10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_high)
        else:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_normal)
        try:
            pctchg = all_data["PCT_CHG"][:-1]
            z_score = Cal_Cdf(pctchg, all_data["PCT_CHG"][-1])
        except:
            try:
                pctchg = all_data["CLOSE3"][:-1]
                z_score = Cal_Cdf(pctchg, all_data["CLOSE3"][-1])
            except:
                z_score = np.nan
        if z_score < 0.1:
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_down2)
        elif (z_score >= 0.1) and (z_score < 0.3):
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_down)
        elif (z_score >= 0.3) and (z_score <= 0.7):
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_normal)
        elif (z_score > 0.7) and (z_score <= 0.9):
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_up)
        elif z_score > 0.9:
            ws.write(rowno, colno+1, "%0.3f"%z_score, style_up2)
        ws.write(rowno+1, colno, "%0.3f%%"%data[u"下10%分位数"], style_quan)
        ws.write(rowno+1, colno+1, "%0.3f%%"%data[u"上10%分位数"], style_quan)
    else:
        raise Exception("not defined field!")

def Xls_Writer_Simple(ws, data, rowno, colno, field):
    if field in [u"涨跌幅", u"振幅", u"日内波动率", u"换手率"]:
        if data[u"今日值"] < data[u"下10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_low)
        elif data[u"今日值"] > data[u"上10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_high)
        else:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_normal)
    elif field in [u"成交额"]:
        if data[u"今日值"] < data[u"下10%分位数"]:
            ws.write(rowno, colno, "%0.2f"%(data[u"今日值"]/100000000), style_low)
        elif data[u"今日值"] > data[u"上10%分位数"]:
            ws.write(rowno, colno, "%0.2f"%(data[u"今日值"]/100000000), style_high)
        else:
            ws.write(rowno, colno, "%0.2f"%(data[u"今日值"]/100000000), style_normal)
    else:
        raise Exception("not defined field!")

def Xls_Writer_pctchg_Simple(ws, data, all_data, rowno, colno, field):
    if field in [u"涨跌幅"]:
        if data[u"今日值"] < data[u"下10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_down2)
        elif data[u"今日值"] > data[u"上10%分位数"]:
            ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_up2)
        else:
            if data[u"今日值"] >= 0:
                ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_up)
            else:
                ws.write(rowno, colno, "%0.3f%%"%data[u"今日值"], style_down)
        pctchg = all_data["PCT_CHG"][:-1]
        z_score = (all_data["PCT_CHG"][-1] - np.mean(pctchg))/np.std(pctchg)
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
    else:
        raise Exception("not defined field!")



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

ws_domestic = ws.add_sheet(u"A股主要指数")
domestic_list = ["000001.SH", "399001.SZ", "399006.SZ", "000016.SH", "000300.SH",
"000905.SH", "000852.SH", "000919.SH", "000918.SH"]
domestic_name = [u"上证综指", u"深证成指", u"创业板指", u"上证50", u"沪深300",
u"中证500", u"中证1000", u"沪深300价值", u"沪深300成长"]

col_no = 1
for each in [u"涨跌幅", u"振幅", u"成交额(亿元)", u"换手率"]:
    ws_domestic.write_merge(0, 0, col_no, col_no+1, each, style_quan)
    col_no = col_no + 2

row_no = 1

for each_index in domestic_list:
    temp_result, all_data = Index_Performance(each_index, enddate, forwardweeks)
    ws_domestic.write_merge(row_no, row_no+1, 0, 0, domestic_name[domestic_list.index(each_index)], style_name)
    col_no = 1
    temp_series = temp_result[u"涨跌幅"]
    Xls_Writer_pctchg(ws_domestic, temp_series, all_data, row_no, col_no, u"涨跌幅")
    col_no = col_no + 2
    for each_field in [u"振幅",  u"成交额", u"换手率"]:
        temp_series = temp_result[each_field]
        Xls_Writer(ws_domestic, temp_series, row_no, col_no, each_field)
        col_no = col_no + 2
    row_no = row_no + 2

future_list = ["IH.CFE", "IF.CFE", "IC.CFE"]
future_name = [u"上证50期货", u"沪深300期货", u"中证500期货"]

row_no = row_no + 2
ws_domestic.write(row_no, 0, u"国内股指期货", style_name)
col_no = 1
for each in [u"涨跌幅", u"振幅", u"成交额(亿元)", u"换手率"]:
    ws_domestic.write_merge(row_no, row_no, col_no, col_no+1, each, style_quan)
    col_no = col_no + 2

row_no = row_no + 1

for each_index in future_list:
    temp_result, all_data = Index_Performance(each_index, enddate, forwardweeks)
    ws_domestic.write_merge(row_no, row_no+1, 0, 0, future_name[future_list.index(each_index)], style_name)
    col_no = 1
    temp_series = temp_result[u"涨跌幅"]
    Xls_Writer_pctchg(ws_domestic, temp_series, all_data, row_no, col_no, u"涨跌幅")
    col_no = col_no + 2
    for each_field in [u"振幅",  u"成交额", u"换手率"]:
        temp_series = temp_result[each_field]
        Xls_Writer(ws_domestic, temp_series, row_no, col_no, each_field)
        col_no = col_no + 2
    row_no = row_no + 2


ws_industry = ws.add_sheet(u"A股行业指数")
industry_list = ["801010.SI", "801020.SI", "801030.SI", "801040.SI", "801050.SI",
"801080.SI", "801110.SI", "801120.SI", "801130.SI", "801140.SI", "801150.SI",
"801160.SI", "801170.SI", "801180.SI", "801200.SI", "801210.SI", "801230.SI",
"801710.SI", "801720.SI", "801730.SI", "801740.SI", "801750.SI", "801760.SI",
"801770.SI", "801780.SI", "801790.SI", "801880.SI", "801890.SI"]
industry_name = [u"农林牧渔", u"采掘", u"化工", u"钢铁", u"有色金属", u"电子", u"家用电器",
u"食品饮料", u"纺织服装", u"轻工制造", u"医药生物", u"公用事业", u"交通运输", u"房地产",
u"商业贸易", u"休闲服务", u"综合", u"建筑材料", u"建筑装饰", u"电气设备", u"国防军工",
u"计算机", u"传媒", u"通信", u"银行", u"非银金融", u"汽车", u"机械设备"]

col_no = 1
ws_industry.write_merge(0, 0, col_no, col_no+1, u"涨跌幅", style_quan)
col_no = col_no + 2
for each in [u"振幅", u"成交额(亿元)", u"换手率"]:
    ws_industry.write(0, col_no, each, style_quan)
    col_no = col_no + 2

row_no = 1

for each_index in industry_list:
    temp_result, all_data = Index_Performance(each_index, enddate, forwardweeks)
    ws_industry.write_merge(row_no, row_no+1, 0, 0, industry_name[industry_list.index(each_index)], style_name)
    col_no = 1
    temp_series = temp_result[u"涨跌幅"]
    Xls_Writer_pctchg(ws_industry, temp_series, all_data, row_no, col_no, u"涨跌幅")
    col_no = col_no + 2
    for each_field in [u"振幅",  u"成交额", u"换手率"]:
        temp_series = temp_result[each_field]
        Xls_Writer(ws_industry, temp_series, row_no, col_no, each_field)
        col_no = col_no + 2
    row_no = row_no + 2


ws_commodity = ws.add_sheet(u"商品指数")
commodity_list = ["CCFI.WI", "NFFI.WI", "JJRI.WI", "NMBM.WI", "CIFI.WI", "CRFI.WI",
"OOFI.WI", "SOFI.WI", "APFI.WI", "B00.IPE", "CL00.NYM", "XAUCNY.IDC", "GC00.CMX", "AU9999.SGE"]
commodity_name = [u"Wind商品指数", u"Wind有色", u"Wind煤焦钢矿", u"Wind非金属建材",
u"Wind化工", u"Wind谷物", u"Wind油脂油料", u"Wind软商品", u"Wind农副产品", u"Wind布油连续",
u"NYMEX轻质原油连续", u"伦敦金", u"COMEX黄金连续", u"SGE黄金9999"]

col_no = 1
for each in [u"涨跌幅", u"振幅"]:
    ws_commodity.write_merge(0, 0, col_no, col_no+1, each, style_quan)
    col_no = col_no + 2

row_no = 1

for each_index in commodity_list:
    temp_result, all_data = Foreignindex_Performance(each_index, enddate, forwardweeks)
    ws_commodity.write_merge(row_no, row_no+1, 0, 0, commodity_name[commodity_list.index(each_index)], style_name)
    col_no = 1
    temp_series = temp_result[u"涨跌幅"]
    Xls_Writer_pctchg(ws_commodity, temp_series, all_data, row_no, col_no, u"涨跌幅")
    col_no = col_no + 2
    for each_field in [u"振幅"]:
        temp_series = temp_result[each_field]
        Xls_Writer(ws_commodity, temp_series, row_no, col_no, each_field)
        col_no = col_no + 2
    row_no = row_no + 2


ws_foreign = ws.add_sheet(u"全球重要股指")
foreign_list = ["SPX.GI", "DJI.GI", "IXIC.GI", "HSI.HI", "FTSE.GI", "GDAXI.GI", "FCHI.GI", "N225.GI"]
foreign_name = [u"标普500", u"道琼斯工业指数", u"纳斯达克指数", u"恒生指数", u"富时100", u"德国DAX", u"法国CAC40", u"日经225"]

col_no = 1
for each in [u"涨跌幅", u"振幅"]:
    ws_foreign.write_merge(0, 0, col_no, col_no+1, each, style_quan)
    col_no = col_no + 2

row_no = 1

for each_index in foreign_list:
    temp_result, all_data = Foreignindex_Performance(each_index, enddate, forwardweeks)
    ws_foreign.write_merge(row_no, row_no+1, 0, 0, foreign_name[foreign_list.index(each_index)], style_name)
    col_no = 1
    temp_series = temp_result[u"涨跌幅"]
    Xls_Writer_pctchg(ws_foreign, temp_series, all_data, row_no, col_no, u"涨跌幅")
    col_no = col_no + 2
    for each_field in [u"振幅"]:
        temp_series = temp_result[each_field]
        Xls_Writer(ws_foreign, temp_series, row_no, col_no, each_field)
        col_no = col_no + 2
    row_no = row_no + 2

row_no = row_no + 1
ws_foreign.write_merge(row_no, row_no+1, 0, 0, u"VIX指数", style_name)
col_no = 1
temp_result, all_data = Interest_Performance("VIX.GI", enddate, forwardweeks)
temp_series = temp_result[u"利率值"]
Xls_Writer_interest(ws_foreign, temp_series, all_data, row_no, col_no, u"利率值")

ws_interest = ws.add_sheet(u"利率与债券指数")
interest_list = ["DR001.IB", "DR007.IB", "SHIBORON.IR", "SHIBOR1W.IR", "CGB1Y.WI",
"CGB3Y.WI", "CGB5Y.WI", "CGB7Y.WI", "CGB10Y.WI", "LIUSDON.IR", "LIUSD1W.IR", "UST1Y.GBM",
"UST3Y.GBM", "UST5Y.GBM", "UST7Y.GBM", "UST10Y.GBM"]
interest_name = [u"银存间质押1日", u"银存间质押7日", u"SHIBOR隔夜", u"SHIBOR1周", u"1Y国债收益", u"3Y国债收益", u"5Y国债收益", u"7Y国债收益", u"10Y国债收益",
u"美元LIBOR隔夜", u"美元LIBOR1周", u"1Y美国国债收益", u"3Y美国国债收益", u"5Y美国国债收益", u"7Y美国国债收益", u"10Y美国国债收益"]

col_no = 1
for each in [u"利率值"]:
    ws_interest.write_merge(0, 0, col_no, col_no+1, each, style_quan)
    col_no = col_no + 2

row_no = 1

for each_index in interest_list:
    temp_result, all_data = Interest_Performance(each_index, enddate, forwardweeks)
    ws_interest.write_merge(row_no, row_no+1, 0, 0, interest_name[interest_list.index(each_index)], style_name)
    col_no = 1
    temp_series = temp_result[u"利率值"]
    Xls_Writer_interest(ws_interest, temp_series, all_data, row_no, col_no, u"利率值")
    row_no = row_no + 2

bond_list = ["H11006.CSI", "H11073.CSI", "TF.CFE", "T.CFE", "3YR.CBT", "TY.CBT", "US.CBT"]
bond_name = [u"中证国债", u"中证信用", u"CFFEX5年期国债期货", u"CFFEX10年期国债期货", u"CBOT3年期美国国债", u"CBOT10年期美国国债", u"CBOT30年期美国国债"]

row_no = row_no + 2
col_no = 1
for each in [u"涨跌幅", u"振幅", u"成交额"]:
    ws_interest.write_merge(row_no, row_no, col_no, col_no+1, each, style_quan)
    col_no = col_no + 2

row_no = row_no + 1

for each_index in bond_list:
    temp_result, all_data = Bondindex_Performance(each_index, enddate, forwardweeks)
    ws_interest.write_merge(row_no, row_no+1, 0, 0, bond_name[bond_list.index(each_index)], style_name)
    col_no = 1
    temp_series = temp_result[u"涨跌幅"]
    Xls_Writer_pctchg(ws_interest, temp_series, all_data, row_no, col_no, u"涨跌幅")
    col_no = col_no + 2
    for each_field in [u"振幅",  u"成交额"]:
        temp_series = temp_result[each_field]
        Xls_Writer(ws_interest, temp_series, row_no, col_no, each_field)
        col_no = col_no + 2
    row_no = row_no + 2


ws.save(path + u"Inspector_weekly_" + enddate + ".xls")
