#coding=utf-8
import pandas as pd
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
###############################################################################
'''
# 得到每一只ETF的首个交易日,去掉样本区间内没有交易数据的ETF，按照首个交易日进行排序
first_tradeday_ETF=Get_first_tradeday(ETF_list,ETF,firstday,lastday,fre)

# 得到ETF成交量volume和return
#ETF_volume_data=data_fetch(ETF_list,ETF,"volume",n1,firstday,lastday,fre)
ETF_return_data=data_fetch(ETF_list,ETF,"pct_chg",n1,firstday,lastday,fre)
ETF_return_data.to_csv(u'%s\\data\\ETF_return_datae.csv'%(direction),encoding= 'gb2312')

# 得到公募基金的return
public_funds_return=data_fetch(public_funds_list,public_funds,"NAV_adj",n1,firstday,lastday,fre)
public_funds_return.to_csv(u'%s\\data\\public_funds_return.csv'%(direction),encoding= 'gb2312')

public_funds_return=pd.read_csv(u'%s\\data\\public_funds_return.csv'%(direction),encoding= 'gb2312')[private_funds.columns]
# 得到私募基金的return
private_funds_return=data_fetch(private_funds_list,private_funds,"NAV_return",n1,firstday,lastday,fre)
'''

public_funds_return=pd.read_csv(u'%s\\data\\public_funds_return_2.csv'%(direction),encoding= 'gb2312',index_col=0)
ETF_return_data=pd.read_csv(u'%s\\data\\ETF_return_datae.csv'%(direction),encoding= 'gb2312',index_col=0)

##############################################################################
# ETF因子数据和被复制的基金数据
data=ETF_return_data
funds_data=public_funds_return
# 滚动时间窗
rolling_window=24
# 时间窗内有效数据在 n2以上
n2=0.75
# 区分可复制基金和不可复制基金的比例
ratio=0.1
# 保存路径
direction='C:\\Users\\Kitty\\Desktop\\ETF'
###############################################################################
# 利用主成分法得到可复制基金与不可复制基金的业绩
# 公募基金
replicable_perf_PCA,un_replicable_perf_PCA,funds_group_PCA,regression_r2_PCA,factor_num_PCA,r_ts,ur_ts=PCA_Replication_performance(funds_data,data,rolling_window,n2,ratio)

# 可复制基金与不可复制基金的收益表现
replicable_perf_PCA.to_csv(u'%s\\public_funds_replication_result\\PCA\\replicable_funds_performance.csv'%(direction),encoding= 'gb2312')
un_replicable_perf_PCA.to_csv(u'%s\\public_funds_replication_result\\PCA\\un_replicable_funds_performance.csv'%(direction),encoding= 'gb2312')
# 属于可复制基金与不可复制基金分别得基金编码
funds_group_PCA.to_csv(u'%s\\public_funds_replication_result\\PCA\\funds_group.csv'%(direction),encoding= 'gb2312')
# 每只基金用ETF主成分复制的最佳因子数和r2
regression_r2_PCA.to_csv(u'%s\\public_funds_replication_result\\PCA\\replication_r2.csv'%(direction),encoding= 'gb2312')
factor_num_PCA.to_csv(u'%s\\public_funds_replication_result\\PCA\\optimized_ETF_factor_number.csv'%(direction),encoding= 'gb2312')
# 时序收益率
r_ts.to_csv(u'%s\\public_funds_replication_result\\PCA\\replicable_return_timeseries.csv'%(direction),encoding= 'gb2312')
ur_ts.to_csv(u'%s\\public_funds_replication_result\\PCA\\unreplicable_return_timeseries.csv'%(direction),encoding= 'gb2312')


'''
###############################################################################
# 利用Lasso回归得到可复制基金与不可复制基金的业绩
positive=False
replicable_perf_L,un_replicable_perf_L,funds_group_L,regression_r2_L,r_ts,ur_ts=Lasso_Replication_performance(funds_data,data,rolling_window,n2,ratio,positive)

# 可复制基金与不可复制基金的收益表现
replicable_perf_L.to_csv(u'%s\\public_funds_replication_result\\Lasso\\replicable_funds_performance.csv'%(direction),encoding= 'gb2312')
un_replicable_perf_L.to_csv(u'%s\\public_funds_replication_result\\Lasso\\un_replicable_funds_performance.csv'%(direction),encoding= 'gb2312')
# 属于可复制基金与不可复制基金分别得基金编码
funds_group_L.to_csv(u'%s\\public_funds_replication_result\\Lasso\\funds_group.csv'%(direction),encoding= 'gb2312')
# 每只基金用ETF主成分复制的r2
regression_r2_L.to_csv(u'%s\\public_funds_replication_result\\Lasso\\replication_r2.csv'%(direction),encoding= 'gb2312')
'''

###############################################################################
# Multi-Task Lasso
replicable_perf_ML,un_replicable_perf_ML,funds_group_ML,regression_r2_ML,r_ts,ur_ts=MultiTaskLasso_Replication_performance(funds_data,data,rolling_window,n2,ratio)

# 可复制基金与不可复制基金的收益表现
replicable_perf_ML.to_csv(u'%s\\public_funds_replication_result\\Multi_Task_Lasso\\replicable_funds_performance.csv'%(direction),encoding= 'gb2312')
un_replicable_perf_ML.to_csv(u'%s\\public_funds_replication_result\\Multi_Task_Lasso\\un_replicable_funds_performance.csv'%(direction),encoding= 'gb2312')
# 属于可复制基金与不可复制基金分别得基金编码
funds_group_ML.to_csv(u'%s\\public_funds_replication_result\\Multi_Task_Lasso\\funds_group.csv'%(direction),encoding= 'gb2312')
# 每只基金用ETF主成分复制的r2
regression_r2_ML.to_csv(u'%s\\public_funds_replication_result\\Multi_Task_Lasso\\replication_r2.csv'%(direction),encoding= 'gb2312')

r_ts.to_csv(u'%s\\public_funds_replication_result\\Multi_Task_Lasso\\replicable_return_timeseries.csv'%(direction),encoding= 'gb2312')
ur_ts.to_csv(u'%s\\public_funds_replication_result\\Multi_Task_Lasso\\unreplicable_return_timeseries.csv'%(direction),encoding= 'gb2312')
























