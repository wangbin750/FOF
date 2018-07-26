# -*- coding: utf-8 -*-
"""
@author: Wang Bin
"""
'''
import pandas as pd
import numpy as np

data = pd.read_csv("D:/ETF/public_funds_replication_result/Multi_Task_Lasso/funds_group.csv")
data = data['un_replicable_funds'].dropna()

total_list = []
for each in data.index:
    total_list = total_list + data[each][1:-1].split(', ')

test = pd.DataFrame(np.array([total_list, total_list]).T, columns=['a','b'])
test.groupby(['a']).count().to_excel("D:/ETF/public_funds_replication_result/Multi_Task_Lasso/funds_counts.xlsx")
'''

import os
import pymssql
conn = pymssql.connect(host = "220.248.87.222", port="12333",database = "chinatime_z", user = "chinatime_z", password = "qwertyuiop")
cur = conn.cursor()

direction='D:\\ETF'
os.chdir(direction)
from replication_function import *
w.start()

#获取ETF的wind代码
ETF=pd.read_csv(u'%s\\ticker\\ETF.csv'%(direction))
ETF_list=Get_list(ETF)
#获得所有公募基金（除掉ETF，债券基金和商品基金）的wind代码
public_funds=pd.read_csv(u'%s\\ticker\\public_funds_2.csv'%(direction))
public_funds_list=Get_list(public_funds)
#print public_funds_list
#获得所有私募基金（除掉CTA和债券基金）的wind代码
private_funds=pd.read_csv(u'%s\\ticker\\private_funds.csv'%(direction))


###############################################################################
# 样本起止日期
firstday="2004-01-01"
lastday="2017-01-01"
# fre=D,W,M,Y
fre='M'
# 可接受的没有数据的交易日所占比例不能高于n1
n1=0.2

public_funds_return=data_fetch(public_funds_list,public_funds,"NAV_adj",n1,firstday,lastday,fre)
public_funds_return.to_csv(u'%s\\data\\public_funds_return.csv'%(direction),encoding= 'gb2312')