# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 08:34:06 2017

@author: Kitty
"""
import pandas as pd
import numpy as np
from WindPy import *
w.start()

import os 
direction='C:\\Users\\Kitty\\Desktop\\ETF_pool'
os.chdir(direction)

# 投资人所投资产
invest_capital=1000000

# 输出的资产比例
asset_weight=pd.read_csv('input.csv')
asset_weight['amount']=invest_capital*asset_weight['weight']

balance_date='2017-06-27'
yesterday_balance_date='2017-06-26'

last_balance_date='2017-05-30'


# 取各类ETF的ticker，并变为str
def Get_str(input_df,selected_column):
    str2=''
    for i in range(0,len(input_df)-1):
        str2 += "%s," %(input_df.iloc[i][selected_column])
    str2+='%s'%(input_df.iloc[len(input_df)-1][selected_column])      
    return str2

def data_fetch(str_ticker,datatype,firstday,lastday):
    data_fetched=w.wsd(str_ticker, datatype, firstday, lastday,"unit=1")
    data=pd.DataFrame(data_fetched.Data[0],index=data_fetched.Codes,columns=[datatype])
    return data
    
def perioddata_fetch(str_ticker,datatype,firstday,lastday):
    data_fetched=w.wsd(str_ticker, datatype, firstday, lastday,"unit=1")
    data=pd.DataFrame(data_fetched.Data,index=data_fetched.Codes,columns=data_fetched.Times)
    return data
    
def tradingdata_fetch(str_ticker,datatype):
    data_fetched=w.wsq(str_ticker, datatype)
    data=pd.DataFrame(data_fetched.Data[0],index=data_fetched.Codes,columns=[datatype])
    return data
    
# 输出其排名rank
def Output_rank(input_df):
    rank=pd.DataFrame(index=input_df.index,columns=pd.DataFrame(input_df).columns)
    for i in range(1,len(input_df)+1):
        rank.iloc[i-1][pd.DataFrame(input_df).columns]=i
    return rank    
    
def Eng_to_Ch(eng_asset):
    if eng_asset=='bond':
        ch_asset=u'债券ETF'
    if eng_asset=='stock_large':
        ch_asset=u'大盘股票ETF'
    if eng_asset=='stock_small':
        ch_asset=u'小盘股票ETF'
    if eng_asset=='stock_HongKong':
        ch_asset=u'港股ETF'
    if eng_asset=='stock_US':
        ch_asset=u'美股ETF'
    if eng_asset=='gold':
        ch_asset=u'黄金ETF'
    if eng_asset=='money_fund':
        ch_asset=u'货币基金ETF'
    return ch_asset
    
###########################################################################################################################

def Get_ETF_trading_blocks(current_asset,num_ETF):
    ETF_code=pd.read_csv('%s_ETF.csv'%(current_asset),encoding= 'gb2312')
    
    str_ETF=Get_str(ETF_code,'id')
    
    # 取份额，本月平均换手率，贴水率
    sort_unit=data_fetch(str_ETF,'unit_total',balance_date,balance_date).sort_values('unit_total',ascending=False)
    #sort_volume=perioddata_fetch(str_ETF,'volume',last_balance_date,balance_date).T.mean().sort_values(ascending=False)
    sort_turn=pd.DataFrame(perioddata_fetch(str_ETF,'turn',last_balance_date,balance_date).T.mean().sort_values(ascending=False),columns=['turn'])
    sort_discount_ratio=data_fetch(str_ETF,'discount_ratio',balance_date,balance_date).sort_values('discount_ratio',ascending=True)
    
    # 分别取其排名
    rank_ETF_pool=pd.concat([Output_rank(sort_turn),Output_rank(sort_unit),Output_rank(sort_discount_ratio)],axis=1)
    # 挑出没有交易价格的ETF,如果没有实时行情，则用昨天的收盘价计算    
    allprice=tradingdata_fetch(str_ETF,'rt_last')
    yesterday_close=data_fetch(str_ETF,'close',yesterday_balance_date,yesterday_balance_date)
    noprice_code=allprice[allprice['rt_last']==0.00].index.tolist()
    allprice['rt_last'].loc[noprice_code]=yesterday_close.loc[noprice_code]['close']
    available_allprice=allprice[allprice['rt_last']!=0.00]
    

    num_available_ETF=len(available_allprice)
    
    num_ETF=min(num_available_ETF,num_ETF)
    # 选择流动性较好的前num_ETF只ETF
    rank_ETF_pool=rank_ETF_pool.loc[available_allprice.index.tolist()]
    selected=rank_ETF_pool.sort_values('turn').head(num_ETF).index.tolist()
    #str_selected=Get_str(pd.DataFrame(selected,columns=['id']),'id')
    # 实时行情
    price=allprice.loc[selected]
    # 交易的份额
    money=asset_weight[asset_weight['asset']==current_asset]['amount']
    target_trading=pd.DataFrame(index=selected,columns=['blocks','weight'])
    
    for j in selected:
        target_trading.loc[j]['blocks']=np.floor(float(money/num_ETF/price.loc[j]['rt_last']/100))
        # 计算真实的资金比例
        target_trading.loc[j]['weight']=price.loc[j]['rt_last']*target_trading.loc[j]['blocks']*100/invest_capital

    cash=money-(target_trading['blocks']*100*price['rt_last']).sum()    
    return target_trading,float(cash)
    

######################################################################################################
# 得到最终交易文件
asset_class=['bond','stock_large','stock_small','stock_HongKong','stock_US','gold']
allcode=pd.read_csv('ETF_code.csv',encoding='gb2312',index_col=0)

Final_trading=pd.DataFrame()
total_cash=0
for current_asset in asset_class:
    allocation=Get_ETF_trading_blocks(current_asset,2)[0]
    allocation['asset']=Eng_to_Ch(current_asset)
    print ('----Trading blocks for %s have been successfully calculated!----'%(current_asset))
    temp_cash=Get_ETF_trading_blocks(current_asset,2)[1]
    total_cash=total_cash+temp_cash
    Final_trading=pd.concat([Final_trading,allocation],axis=0)


Final_trading=Final_trading[Final_trading['blocks']!=0]    
Final_trading['name']=allcode.loc[Final_trading.index.tolist()]
    
# 计算现金和货基比例等    
cash_df=pd.DataFrame(index=[u'现金',u'货币型基金'],columns=['blocks','weight','asset','name'])
Final_trading=pd.concat([Final_trading,cash_df],axis=0) 
Final_trading.loc[u'现金']=[total_cash,total_cash/invest_capital,u'现金',u'现金']
money_fund_amt=invest_capital-asset_weight['amount'].sum()
Final_trading.loc[u'货币型基金']=[money_fund_amt,money_fund_amt/invest_capital,u'货币型基金',u'货币型基金']


print Final_trading


