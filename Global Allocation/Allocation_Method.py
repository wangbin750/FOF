# -*- coding: utf-8 -*-
"""
@author: Wang Bin
"""

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
        goal = 0
        for i in range(len(exp_ret_list)):
            omega = np.matrix(cov_mat_list[i].values)
            ret = np.matrix(exp_ret_list[i])
            goal = goal - prob_list[i]*(np.matrix(x) * ret - 0.5 * lam * np.sqrt(np.matrix(x) * omega * np.matrix(x).T))
        return goal

    x0 = np.ones(len(exp_ret_list[0])) / len(exp_ret_list[0])
    bnds = tuple(bnds)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    options = {'disp': False, 'maxiter': 500, 'ftol': 1e-10}

    res = minimize(fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)

    weight = pd.Series(index=cov_mat_list[0].index, data=res['x'])

    return weight / weight.sum()

def Max_Utility_Weight_new_MS(exp_ret_list, cov_mat_list, prob_list, lam, bnds):
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
        goal = 0
        for i in range(len(exp_ret_list)):
            omega = np.matrix(cov_mat_list[i].values)
            ret = np.matrix(exp_ret_list[i])
            goal = goal - prob_list[i]*(np.matrix(x) * ret - 0.5 * lam * (np.matrix(x) * omega * np.matrix(x).T))
        return goal

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

def VaR_Cal(conf_level, ret_list, cov_mat, period, *weight):
    import numpy as np
    from scipy.stats import norm

    if type(ret_list) != list:
        ret = ret_list
        sigma = cov_mat
    else:
        ret = np.matrix(ret_list) * np.matrix(weight[0]).T
        omega = np.matrix(cov_mat.values)
        sigma = np.matrix(weight[0]) * omega * np.matrix(weight[0]).T

    VaR = norm.ppf(1-conf_level, ret, sigma)
    if type(VaR) == np.ndarray:
        VaR = VaR[0][0]
    else:
        pass

    return VaR

def Combined_Return_Distribution_MS(lam, exp_ret, cov_mat, tau, P, Q, conf_mat):
    '''
    计算BL模型中给定主观观点后的修正预期收益率的均值与方差协方差矩阵
    :param lam: 风险厌恶系数-float
    :param cov_mat: 资产的方差协方差矩阵-DataFrame
    :param tau: BL模型中历史方差协方差矩阵的权重-float
    :param P: 观点矩阵-matrix
    :param Q: 观点收益向量-matrix
    :param conf_mat: 信心矩阵-matrix
    :return: 预期收益-DataFrame，预期方差协方差矩阵-DataFrame
    '''
    import numpy as np
    import pandas as pd

    if all(cov_mat.index == exp_ret.index) == False:
        exp_ret = exp_ret.reindex(cov_mat.index)

    index_columns = cov_mat.index
    exp_ret = np.matrix(exp_ret)
    cov_mat = np.matrix(cov_mat)
    exp_cov_mat = ((tau * cov_mat).I + (P.T * conf_mat.I * P)).I

    exp_ret = exp_cov_mat * ((tau * cov_mat).I * exp_ret + P.T * conf_mat.I * Q.T)

    exp_ret = pd.DataFrame(exp_ret, index=index_columns)
    exp_cov_mat = pd.DataFrame(exp_cov_mat, columns=index_columns, index=index_columns)

    return exp_ret, exp_cov_mat
