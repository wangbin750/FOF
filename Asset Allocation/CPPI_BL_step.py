# coding=utf-8
# -*- coding: utf-8 -*-

from Allocation_Method import Risk_Parity_Weight, Min_Variance_Weight, Combined_Return_Distribution, Max_Sharpe_Weight, Max_Utility_Weight, Inverse_Minimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from math import floor
import multiprocessing

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

def Cushion_Cal(nv, nv_threshold, rf_ret, target_ret):
    maturity = 1
    target = nv_threshold * (1+target_ret)**maturity

    safe_asset = target/((1+rf_ret)**maturity)
    if nv < safe_asset:
        print "out!"
        cushion = 0
    else:
        cushion = nv - safe_asset
    return cushion/nv


def Backtest_CPPI_BL_step(History_Data, Predict_Data, History_Data_D, asset_list, risk_list, bnds, asset_level_1, asset_level_2, year_delta, portfolio_name, money_weight, up_per, target_ret, multiplier, max_risk_weight):

    tau = 1.0
    if portfolio_name == "wenjian":
        lam = 2.3 #进取-1.9 平衡-2.0 稳健-2.3
    elif portfolio_name == "pingheng":
        lam = 1.9
    elif portfolio_name == "jinqu":
        lam = 1.7
    else:
        raise Exception("Wrong portfolio_name!")

    pct_list = []
    weight_list = []
    date_list = []
    asset_drawdown = pd.Series([0.0]*len(asset_list), index=asset_list)
    asset_position = pd.Series([1.0]*len(asset_list), index=asset_list)
    for each_date in Predict_Data.index[60:-1]:
        last_date = History_Data.index[list(Predict_Data.index).index(each_date)-1]  # 当前月份日期
        next_date = each_date  # 下一月份日期
        if last_date.month <= 11:
            start_year = last_date.year - year_delta
            start_month = last_date.month + 1
        else:
            start_year = last_date.year - year_delta + 1
            start_month = 1

        # 基础设定
        history_data = History_Data[str(start_year) + '-' + str(start_month): last_date]
        history_data_d = History_Data_D[str(start_year) + '-' + str(start_month): last_date]
        predict_data = Predict_Data[str(start_year) + '-' + str(start_month): last_date]
        cov_mat = history_data_d[risk_list].cov() * (len(history_data_d)/year_delta)
        #cov_mat = history_data[asset_list].cov() * 12.0
        #for each_asset in risk_list:
        for each_asset in risk_list:
            temp_drawdown = (asset_drawdown[each_asset]+1.0)*(history_data[each_asset][-1]+1.0)-1
            if temp_drawdown >= 0:
                temp_drawdown = 0
            else:
                pass
            asset_drawdown[each_asset] = temp_drawdown
        # print cov_mat
        omega = np.matrix(cov_mat.values)
        mkt_wgt = Risk_Parity_Weight(cov_mat)
        #print mkt_wgt

        P = np.diag([1] * len(mkt_wgt))

        conf_list = list()
        for each in risk_list:
            conf_temp = ((history_data[each][str(start_year) + '-' + str(start_month):] -
                          predict_data[each][str(start_year) + '-' + str(start_month):])**2).mean() * 12.0
            conf_list.append(conf_temp)
        conf_mat = np.matrix(np.diag(conf_list))

        Q = np.matrix(Predict_Data[risk_list].loc[next_date])

        com_ret, com_cov_mat = Combined_Return_Distribution(
            2, cov_mat, tau, mkt_wgt, P, Q, conf_mat)

        #print com_ret

        weight_bl = Max_Utility_Weight(com_ret, com_cov_mat, lam, bnds)
        #print weight_bl


        rf_ret = history_data[list(set(asset_list)-set(risk_list))[0]].mean() * 12

        if len(pct_list) == 0:
            current_nv = 1
            nv_threshold = 1
        else:
            current_nv = (np.array(pct_list)+1).prod()
        #print "current_nv"
        #print current_nv
        if current_nv > (1+up_per)*nv_threshold:
            nv_threshold = (1+up_per)*nv_threshold
        else:
            pass

        cushion_per = Cushion_Cal(nv=current_nv, nv_threshold=nv_threshold, rf_ret=rf_ret, target_ret=target_ret)

        #print "cushion"
        #print cushion_per

        para_m = 0.01 / (-VaR_Cal(0.99, [0]*len(risk_list), history_data[risk_list].cov()*12.0, 1/12, weight_bl[risk_list]))

        #print "para_m"
        #print para_m
        #print "--------------"

        risk_weight = min(cushion_per*multiplier, max_risk_weight)

        weight_risk = weight_bl[risk_list] * risk_weight
        #weight_risk = weight_bl[risk_list]
        #print weight_risk
        weight_bl[risk_list] = weight_risk
        #weight_bl[risk_list] = [0.0]*len(risk_list)
        weight_bl[list(set(asset_list)-set(risk_list))[0]] = money_weight-sum(weight_bl[risk_list])
        #weight_bl[list(set(asset_list)-set(risk_list))[0]] = 0.0


        for each_asset in asset_list:
            if asset_position[each_asset] == 1:
                if (asset_drawdown[each_asset] <= asset_level_1[each_asset]) and (asset_drawdown[each_asset] > asset_level_2[each_asset]):
                    asset_position[each_asset] = 0.5
                elif asset_drawdown[each_asset] <= asset_level_2[each_asset]:
                    asset_position[each_asset] = 0.0
                else:
                    pass
            elif asset_position[each_asset] == 0.5:
                if asset_position[each_asset] <= asset_level_2[each_asset]:
                    asset_position[each_asset] = 0.0
                elif (predict_data[each_asset][-1] > 0) and (history_data[each_asset][-1] > 0):
                    asset_position[each_asset] = 1.0
                    asset_drawdown[each_asset] = 0.0
                else:
                    pass
            elif asset_position[each_asset] == 0.0:
                if (predict_data[each_asset][-1] > 0) and (history_data[each_asset][-1] > 0):
                    asset_position[each_asset] = 0.5
                    asset_drawdown[each_asset] = 0.0
                else:
                    pass

        weight_bl = (weight_bl*asset_position).round(2)
        #print weight_bl
        #print next_date
        #print sum(weight_bl)
        port_ret = sum(weight_bl*History_Data[asset_list].loc[next_date]) + (1.0-sum(weight_bl))*History_Data["money"].loc[next_date]

        #print sum(weight_bl*History_Data[asset_list].loc[next_date])*money_weight + money_weight*History_Data["money"].loc[next_date]
        pct_list.append(port_ret)
        weight_list.append(list(weight_bl))
        date_list.append(next_date)

    #pd.Series(np.array(pct_list), index=date_list).to_csv("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_%s.csv"%portfolio_name)
    #pd.DataFrame(np.array(weight_list), index=date_list, columns=asset_list).to_excel("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_%s_weight.xlsx"%portfolio_name)
    return pd.Series(np.array(pct_list), index=date_list)

def Performance(return_series, rf_ret):
    end_value = (return_series + 1).prod()
    annual_return = (return_series + 1).prod() ** (1/(len(return_series)/12.0)) - 1
    annual_variance = (return_series.var() * 12.0) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    return [end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown]

def Conduct(History_Data, Predict_Data, History_Data_D, asset_list, risk_list, bnds, asset_level_1, asset_level_2, year_delta, portfolio_name, money_weight, up_per, target_ret, multiplier, max_risk_weight):
    temp_series = Backtest_CPPI_BL_step(History_Data, Predict_Data, History_Data_D, asset_list, risk_list, bnds, asset_level_1, asset_level_2, year_delta, portfolio_name, money_weight, up_per, target_ret, multiplier, max_risk_weight)
    temp_performance = Performance(temp_series, 0.025)
    performance_list = temp_performance+[up_per, target_ret, multiplier, max_risk_weight]
    print [up_per, target_ret, multiplier, max_risk_weight]
    return performance_list

def test(para_1, para_2, para_3, para_4):
    return para_1+para_2+para_3+para_4
#return_series = pd.read_csv("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_wenjian.csv", header=None)[1]
#print Performance(return_series, 0.025)

'''
file_name_list = ['wenjian', 'pingheng', 'jinqu', 'wenjian_f', 'pingheng_f', 'jinqu_f']
result_list = list()
for each_file in file_name_list:
    return_series = pd.read_csv("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_"+each_file+".csv", header=None)[1]
    return_series = [1] + list(return_series.values)
    return_series = pd.Series(return_series).pct_change().dropna()
    temp_result = Performance(return_series, 0.025)
    result_list.append(temp_result)

pd.DataFrame(np.array(result_list).T, index=['end_value', 'annual_return', 'annual_variance', 'sharpe_ratio', 'max_drawdown'], columns=['wenjian', 'pingheng', 'jinqu', 'wenjian_f', 'pingheng_f', 'jinqu_f']).to_excel("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_results.xlsx")
'''
if __name__ == '__main__':
    portfolio_name = u"wenjian"

    '''
    History_Data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/History_Data.xlsx")
    Predict_Data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/Predict_Data.xlsx")
    History_Data_D = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/History_Data_D.xlsx")
    asset_list = ["bond_whole", "stock_large", "stock_small",
                  "stock_HongKong", "stock_US", "gold"]
    risk_list = ["stock_large", "stock_small",
                  "stock_HongKong", "stock_US", "gold"]
    bnds = [(0.0, None), (0.0, None), (0.0, None),
            (0.0, None), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -0.08, -0.08, -0.08, -0.08, -0.08], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -0.16, -0.16, -0.16, -0.16, -0.16], index=asset_list)
    #asset_level_1 = pd.Series([-1.0]*len(asset_list), index=asset_list)
    #asset_level_2 = pd.Series([-1.0]*len(asset_list), index=asset_list)
    #bnds = [(0.1, 0.6), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.0, 0.3)]
    '''

    #美国实际数据
    History_Data = pd.read_excel("F:\\GitHub\\FOF\\Asset Allocation\\SBG_US_M.xlsx")
    Predict_Data = pd.read_excel("F:\\GitHub\\FOF\\Asset Allocation\\SBG_US_M_P.xlsx")
    History_Data_D = pd.read_excel("F:\\GitHub\\FOF\\Asset Allocation\\SBG_US_M.xlsx")
    asset_list = [ "Barclays_US_bond", "SP500", "London_gold"]
    risk_list = ["SP500", "London_gold"]

    bnds = [(0.0, None), (0.0, None)]
    #bnds = [(0.0, None), (0.0, None), (0.0, None)]
    asset_level_1 = pd.Series([-0.01, -0.08, -0.08], index=asset_list)
    asset_level_2 = pd.Series([-0.02, -0.16, -0.16], index=asset_list)


    up_per_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    target_ret_list = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.00, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2]
    multiplier_list = [1, 2, 3, 4, 5]
    max_risk_weight_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    '''
    up_per_list = [0.1]
    target_ret_list = [-0.5]
    multiplier_list = [1]
    max_risk_weight_list = [0.1, 0.2]
    '''
    pool = multiprocessing.Pool(processes=10)
    performance_list = []
    performance_array = []
    para_list = []
    for up_per in up_per_list:
        for target_ret in target_ret_list:
            for multiplier in multiplier_list:
                for max_risk_weight in max_risk_weight_list:
                    performance_list.append(pool.apply_async(Conduct, (History_Data, Predict_Data, History_Data_D, asset_list, risk_list, bnds, asset_level_1, asset_level_2, 5, "wenjian", 1.00, up_per, target_ret, multiplier, max_risk_weight, )))

    pool.close()
    pool.join()

    for res in performance_list:
        temp = res.get()
        performance_array.append(temp)
        para_list.append("up_per=%s \ntarget_ret=%s \nmultiplier=%s \nmax_risx_weight=%s\n"%tuple(temp[5:]))

    pd.DataFrame(np.array(performance_array), columns=['nv', 'ar', 'av', 'sr', 'md', 'up_per', 'target_ret', 'multiplier', 'max_risk_weight'], index=para_list).to_excel("F:\\GitHub\\FOF\\Asset Allocation\\backtest_cppi_bl_step.xlsx")
