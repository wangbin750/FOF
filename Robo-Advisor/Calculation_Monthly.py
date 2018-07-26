# -*- coding: utf-8 -*-
"""
@Author: Wang Bin
@Time: 2017/1/19 09:30
"""

portfolio_name = u"wenjian"

import pandas as pd
import numpy as np

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


History_Data = pd.read_excel(u"Z:\Mac 上的 WangBin-Mac\FOF\Robo-Advisor\History_Data.xlsx")
Predict_Data = pd.read_excel(u"Z:\Mac 上的 WangBin-Mac\FOF\Robo-Advisor\HP_Data.xlsx")
asset_list = ["bond", "stock_large", "stock_small",
              "stock_HongKong", "stock_US", "gold"]
bnds = [(0.0, None), (0.0, None), (0.0, None),
        (0.0, None), (0.0, None), (0.0, None)]
#bnds = [(0.1, 0.6), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.0, 0.3)]

year_delta = 5
tau = 1.0
if portfolio_name == "wenjian":
    lam = 2.3 #进取-1.9 平衡-2.0 稳健-2.3
    money_weight = 0.75
elif portfolio_name == "pingheng":
    lam = 2.0
    money_weight = 0.8
elif portfolio_name == "jinqu":
    lam = 1.9
    money_weight = 0.85
else:
    raise Exception("Wrong portfolio_name!")


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

pd.DataFrame(np.array([weight_bl.index, (weight_bl * money_weight).values]).T, columns=['asset', 'weight']).to_csv(u'Z:\Mac 上的 WangBin-Mac\FOF\Robo-Advisor\input.csv')
