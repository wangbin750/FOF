# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pyper as pr
from Allocation_Method import Risk_Parity_Weight, Max_Utility_Weight_new
import math
import socket

hostname = socket.gethostname()
if hostname == "DESKTOP-OGC5NH7":
    path = u"E:/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "localhost":
    path = "/Users/WangBin-Mac/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "CICCB6CR7213VFT":
    path = "F:/GitHub/FOF/Global Allocation/"


def Performance(return_series, rf_ret):
    end_value = (return_series + 1).prod()
    annual_return = (return_series + 1).prod() ** (1/(len(return_series)/12.0)) - 1
    annual_variance = (return_series.var() * 12.0) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    win_ratio = float(sum(return_series>=0))/float(len(return_series))
    return [end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown, win_ratio]

p_list = []

for j in [0, 6, 12, 24]:
    temp = pd.read_excel(path+"DCC_90_%s+REITs.xlsx"%j)
    for i in range(6):
        p_list.append(Performance(temp[i], 0.025))

pd.DataFrame(p_list).to_excel(path+"DCC_per_90+REITs.xlsx")

def Turn_over(num):
    a = int(math.floor(num/8.0))
    if a == 0:
        a = 0
    else:
        a = (2**(a-1))*8
    b = num-int(math.floor(num/8.0))*8
    wtemp = pd.read_excel(path+"DCCw_90_%s+REITs.xlsx"%a)
    temp = wtemp[[b*6, b*6+1, b*6+2, b*6+3, b*6+4, b*6+5, b*8+6, b*8+7]]
    return temp.diff().abs().sum(axis=1).mean()*12.0
