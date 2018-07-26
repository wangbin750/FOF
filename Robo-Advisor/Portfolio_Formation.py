# -*- coding: utf-8 -*-
"""
@Author: Wang Bin
@Time: 2017/1/19 09:30
"""
import pandas as pd
import numpy as np
import os


def Risk_Parity_Weight(cov_mat):
    '''
    计算风险平价配置权重
    :param cov_mat: 资产的方差协方差矩阵-DataFrame
    :return: 配置权重-Series
    '''

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    omega = np.matrix(cov_mat.values)

    def fun(x):
        tmp = (omega * np.matrix(x).T).A1
        risk = x * tmp
        delta_risk = [sum((i - risk)**2) for i in risk]
        return sum(delta_risk)

    x0 = np.ones(omega.shape[0]) / omega.shape[0]
    bnds = tuple((0, None) for x in x0)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-20}

    res = minimize(fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    #print res['x']
    #print type(res['x'])

    weight = pd.Series(index=cov_mat.index, data=res['x'])

    return weight / weight.sum()


def Min_Variance_Weight(cov_mat):
    '''
    计算最小方差组合的配置权重
    :param cov_mat: 资产的方差协方差矩阵-DataFrame
    :return: 配置权重-Series
    '''

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    omega = np.matrix(cov_mat.values)

    def fun(x):
        return np.matrix(x) * omega * np.matrix(x).T

    x0 = np.ones(omega.shape[0]) / omega.shape[0]
    bnds = tuple((0, None) for x in x0)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-20}

    res = minimize(fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)

    weight = pd.Series(index=cov_mat.index, data=res['x'])

    return weight / weight.sum()


def Combined_Return_Distribution(lam, cov_mat, tau, mkt_wgt, P, Q, conf_mat):
    '''
    计算BL模型中给定主观观点后的修正预期收益率的均值与方差协方差矩阵
    :param lam: 风险厌恶系数-float
    :param cov_mat: 资产的方差协方差矩阵-DataFrame
    :param tau: BL模型中历史方差协方差矩阵的权重-float
    :param mkt_wgt: 资产在真实市场中的权重-DataFrame
    :param P: 观点矩阵-matrix
    :param Q: 观点收益向量-matrix
    :param conf_mat: 信心矩阵-matrix
    :return: 预期收益-DataFrame，预期方差协方差矩阵-DataFrame
    '''
    import numpy as np
    import pandas as pd

    if all(cov_mat.index == mkt_wgt.index) == False:
        mak_wgt = mkt_wgt.reindex(cov_mat.index)

    index_columns = cov_mat.index

    #print cov_mat
    equil_ret = Inverse_Minimize(mkt_wgt, cov_mat, lam)
    equil_ret = np.matrix(equil_ret).T
    #print equil_ret
    cov_mat = np.matrix(cov_mat)
    exp_cov_mat = ((tau * cov_mat).I + (P.T * conf_mat.I * P)).I
    #exp_cov_mat = cov_mat
    #print exp_cov_mat

    #print equil_ret
    #print Q
    exp_ret = exp_cov_mat * ((tau * cov_mat).I * equil_ret + P.T * conf_mat.I * Q.T)
    #exp_ret = equil_ret
    #print (tau * cov_mat).I
    #print P.T * conf_mat.I
    #print equil_ret
    #print Q.T

    exp_ret = pd.DataFrame(exp_ret, index=index_columns)
    exp_cov_mat = pd.DataFrame(exp_cov_mat, columns=index_columns, index=index_columns)
    #print exp_cov_mat
    #print exp_ret
    return exp_ret, exp_cov_mat


def Max_Sharpe_Weight(exp_ret, cov_mat, rf_ret):
    '''
    计算夏普比率最大的投资组合的权重
    :param exp_ret: 预期收益率-DataFrame
    :param cov_mat: 预期方差协方差矩阵-DataFrame
    :return: 配置权重-Series
    '''

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    omega = np.matrix(cov_mat.values)
    ret = np.matrix(exp_ret)
    #print omega
    #print exp_ret

    def fun(x):
        return -(np.matrix(x) * ret - rf_ret) / np.sqrt(np.matrix(x) * omega * np.matrix(x).T)

    x0 = np.ones(omega.shape[0]) / omega.shape[0]
    bnds = tuple((0, None) for x in x0)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    options = {'disp': False, 'maxiter': 500, 'ftol': 1e-10}

    res = minimize(fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)

    weight = pd.Series(index=cov_mat.index, data=res['x'])

    return weight / weight.sum()

def Max_Utility_Weight(exp_ret, cov_mat, lam, bnds):
    '''
    计算夏普比率最大的投资组合的权重
    :param exp_ret: 预期收益率-DataFrame
    :param cov_mat: 预期方差协方差矩阵-DataFrame
    :return: 配置权重-Series
    '''

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    omega = np.matrix(cov_mat.values)
    ret = np.matrix(exp_ret)
    #print omega
    #print exp_ret

    def fun(x):
        return -(np.matrix(x) * ret - 0.5 * lam * np.sqrt(np.matrix(x) * omega * np.matrix(x).T))

    x0 = np.ones(omega.shape[0]) / omega.shape[0]
    bnds = tuple(bnds)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    options = {'disp': False, 'maxiter': 500, 'ftol': 1e-10}

    res = minimize(fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)

    weight = pd.Series(index=cov_mat.index, data=res['x'])

    return weight / weight.sum()

def Max_Utility_Weight_MS(exp_ret_list, cov_mat_list, prob_list, lam, bnds):
    '''
    MS框架下计算夏普比率最大的投资组合的权重
    :param exp_ret: 预期收益率-DataFrame
    :param cov_mat: 预期方差协方差矩阵-DataFrame
    :return: 配置权重-Series
    '''

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    #print omega
    #print exp_ret

    def fun(x):
        gold = 0
        for i in range(len(exp_ret_list)):
            omega = np.matrix(cov_mat_list[i].values)
            ret = np.matrix(exp_ret_list[i])
            gold = gold - prob_list[i]*(np.matrix(x) * ret - 0.5 * lam * np.sqrt(np.matrix(x) * omega * np.matrix(x).T))
        return gold

    x0 = np.ones(len(exp_ret_list[0])) / len(exp_ret_list[0])
    bnds = tuple(bnds)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    options = {'disp': False, 'maxiter': 500, 'ftol': 1e-10}

    res = minimize(fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)

    weight = pd.Series(index=cov_mat_list[0].index, data=res['x'])

    return weight / weight.sum()


def Max_Utility_Weight_new(exp_ret, cov_mat, lam, bnds):
    '''
    计算夏普比率最大的投资组合的权重
    :param exp_ret: 预期收益率-DataFrame
    :param cov_mat: 预期方差协方差矩阵-DataFrame
    :return: 配置权重-Series
    '''

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    omega = np.matrix(cov_mat.values)
    ret = np.matrix(exp_ret)
    #print omega
    #print exp_ret

    def fun(x):
        return -(np.matrix(x) * ret - 0.5 * lam * (np.matrix(x) * omega * np.matrix(x).T))

    x0 = np.ones(omega.shape[0]) / omega.shape[0]
    bnds = tuple(bnds)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    options = {'disp': False, 'maxiter': 500, 'ftol': 1e-10}

    res = minimize(fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)

    weight = pd.Series(index=cov_mat.index, data=res['x'])

    return weight / weight.sum()


def Inverse_Minimize(wgt, cov_mat, lam):
    '''
    :param wgt:
    :param cov_mat:
    :param lam:
    :return:
    '''

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    omega = np.matrix(cov_mat)

    r0 = np.zeros(omega.shape[0])

    def fun(r):
        def fun_in(x):
            return -(np.matrix(x) * np.matrix(r).T - 0.5 * lam * np.sqrt(np.matrix(x) * omega * np.matrix(x).T))

        x0 = np.ones(omega.shape[0]) / omega.shape[0]
        bnds_in = tuple((0, None) for x in x0)
        cons_in = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
        options_in = {'disp': False, 'maxiter': 500, 'ftol': 1e-10}

        res_in = minimize(fun_in, x0, bounds=bnds_in, constraints=cons_in, method='SLSQP', options=options_in)
        #print res_in['x']
        #print res_in['x']
        #print np.sum((wgt - res_in['x'])**2)
        return np.sum((wgt - res_in['x'])**2)

    options = {'disp': False, 'maxiter': 500, 'ftol': 1e-8}
    #bnds = tuple((None, None) for r in r0)
    res = minimize(fun, r0, method='Nelder-Mead', options=options)
    #print "END"

    inver_ret = pd.Series(index=cov_mat.index, data=res['x'])

    return inver_ret


"""
Created on Wed Jun 21 08:34:06 2017
@author: Kitty
"""

from WindPy import *
w.start()

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

def Get_ETF_trading_blocks(current_asset,balance_date,last_balance_date,yesterday_balance_date,asset_weight,invest_capital,num_ETF):
    ETF_code=pd.read_csv('%s_ETF.csv'%(current_asset),encoding= 'gb2312')
    str_ETF=Get_str(ETF_code,'id')

    # 取份额，本月平均换手率，贴水率
    sort_unit=data_fetch(str_ETF,'unit_total',balance_date,balance_date).sort_values('unit_total',ascending=False)
    #sort_volume=perioddata_fetch(str_ETF,'volume',last_balance_date,balance_date).T.mean().sort_values(ascending=False)
    sort_turn=pd.DataFrame(perioddata_fetch(str_ETF,'turn',last_balance_date,balance_date).T.mean().sort_values(ascending=False),columns=['turn'])
    sort_discount_ratio=data_fetch(str_ETF,'discount_ratio',balance_date,balance_date).sort_values('discount_ratio',ascending=True)

    # 分别取其排名
    rank_ETF_pool=pd.concat([Output_rank(sort_turn),Output_rank(sort_unit),Output_rank(sort_discount_ratio)],axis=1)

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
def Portfolio_Form(risk_adverse):
    import pandas as pd
    import numpy as np
    import datetime as dt
    import os

    # 投资人所投资产
    invest_capital=1000000

    direction=u'Z:\Mac 上的 WangBin-Mac\FOF\Robo-Advisor'
    os.chdir(direction)

    History_Data = pd.read_excel(direction + u"\History_Data.xlsx")
    Predict_Data = pd.read_excel(direction + u"\HP_Data.xlsx")
    asset_list = ["bond", "stock_large", "stock_small",
                  "stock_HongKong", "stock_US", "gold"]
    bnds = [(0.0, None), (0.0, None), (0.0, None),
            (0.0, None), (0.0, None), (0.0, None)]
    #bnds = [(0.1, 0.6), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.0, 0.3)]

    year_delta = 5
    tau = 1.0
    lam = 2.5 - risk_adverse * 0.1
    money_weight = 0.75 + 0.02 * risk_adverse


    # 日期设定
    last_date = History_Data.index[-1]  # 当前月份日期
    next_date = Predict_Data.index[-1]  # 下一月份日期
    if last_date.month <= 11:
        start_year = last_date.year - year_delta
        start_month = last_date.month + 1
    else:
        start_year = last_date.year - year_delta + 1
        start_month = 1

    # 基础设定
    history_data = History_Data[asset_list][
        str(start_year) + '-' + str(start_month): last_date]
    predict_data = Predict_Data[asset_list][
        str(start_year) + '-' + str(start_month): last_date]
    cov_mat = history_data[asset_list].cov() * 12.0
    # print cov_mat
    omega = np.matrix(cov_mat.values)
    mkt_wgt = Risk_Parity_Weight(cov_mat)
    #print mkt_wgt
    P = np.diag([1] * len(mkt_wgt))

    conf_list = list()
    for each in asset_list:
        conf_temp = ((history_data[each][str(start_year) + '-' + str(start_month):] -
                      predict_data[each][str(start_year) + '-' + str(start_month):])**2).mean() * 12.0
        conf_list.append(conf_temp)
    conf_mat = np.matrix(np.diag(conf_list))

    Q = np.matrix(Predict_Data[asset_list].loc[next_date])


    com_ret, com_cov_mat = Combined_Return_Distribution(
        2, cov_mat, tau, mkt_wgt, P, Q, conf_mat)

    #print com_ret

    weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)

    pd.DataFrame(np.array([weight_bl.index, (weight_bl * money_weight).values]).T, columns=['asset', 'weight']).to_csv(direction + u'\input.csv')

    # 输出的资产比例
    asset_weight=pd.read_csv('input.csv')
    asset_weight['amount']=invest_capital*asset_weight['weight']

    balance_date=History_Data.index[-1]
    last_balance_date=History_Data.index[-2]
    yesterday_balance_date=dt.datetime.today()-dt.timedelta(days=1)

    asset_class=['bond','stock_large','stock_small','stock_HongKong','stock_US','gold']
    allcode=pd.read_csv('ETF_code.csv',encoding='gb2312',index_col=0)

    Final_trading=pd.DataFrame()
    total_cash=0
    for current_asset in asset_class:
        allocation=Get_ETF_trading_blocks(current_asset,balance_date,last_balance_date,yesterday_balance_date,asset_weight,invest_capital,2)[0]
        allocation['asset']=Eng_to_Ch(current_asset)
        #print ('----Trading blocks for %s have been successfully calculated!----'%(current_asset))
        temp_cash=Get_ETF_trading_blocks(current_asset,balance_date,last_balance_date,yesterday_balance_date,asset_weight,invest_capital,2)[1]
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


    return Final_trading.to_string()
