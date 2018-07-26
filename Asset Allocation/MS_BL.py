# coding=utf-8
import pandas as pd
import numpy as np
import pyper as pr
#import matplotlib.pyplot as plt

from Allocation_Method import Risk_Parity_Weight, Max_Utility_Weight_new, Max_Utility_Weight, Max_Utility_Weight_MS, Combined_Return_Distribution_MS

def Ms_R(return_list, regime_count):
    #return_list = list(data_M["AU9999.SGE"])
    return_frame = pd.DataFrame(np.array([return_list, [1]*len(return_list)]).T, columns=['return', 'One'])
    r("rm(list = ls())")
    r.assign("rframe", return_frame)
    r.assign("rregimecount", regime_count)
    Coef = None
    break_i = 0
    while Coef is None:
        r('''
        lm_return <- lm(formula=return~0+One, data=rframe)
        lm_mswm <- msmFit(lm_return, k=rregimecount, p=0, sw=c(T,T))
        rstd <- lm_mswm@std
        rCoef <- lm_mswm@Coef
        rtransMat <- lm_mswm@transMat
        rprob_smo <- lm_mswm@Fit@smoProb[-1,]
        #print(lm_mswm@Fit@logLikel)
        raic_mswn <- AIC(lm_mswm)
        raic_lm <- AIC(lm_return)
        ''')
        Coef = r.get("rCoef")
        break_i = break_i + 1
        if break_i > 100:
            break

    aic_mswm = r.get("raic_mswm")
    aic_lm = r.get("raic_lm")

    if aic_mswm < aic_lm:
        std = np.round(r.get("rstd"),4)
        Coef = np.round(np.array(r.get("rCoef")[' One ']),4)
        transMat = np.round(r.get("rtransMat").T,4)
        prob_smo = np.round(r.get("rprob_smo"),4)
    else:
        std = np.array([np.std(return_list)]*regime_count)
        Coef = np.array([np.mean(return_list)]*regime_count)
        transMat = np.diag([1]*regime_count)
        prob_smo = np.array([[1.0/regime_count]*regime_count]*len(return_list))

    return std, Coef, transMat, prob_smo

def Cross_Cov(return_list_1, return_list_2, Coef_1, Coef_2, prob_smo_1, prob_smo_2):
    temp_list = []
    range_i = len(Coef_1)
    range_j = len(Coef_2)
    for i in range(range_i):
        for j in range(range_j):
            temp_cov = np.sum(prob_smo_1[:,i]*prob_smo_2[:,j]*(return_list_1 - Coef_1[i])*(return_list_2 - Coef_2[j]))/np.sum(prob_smo_1[:,i]*prob_smo_2[:,j])
            temp_list.append(temp_cov)
    cov_mat = np.array(temp_list).reshape(range_i,range_j)
    return cov_mat


def Tree_Gen(switch_map, temp_columns):
    dimension_list = []
    for each in temp_columns:
        if switch_map[each] != 1:
            dimension_list.append(switch_map[each])
        else:
            dimension_list.append(2)
    #print dimension_list
    return_list = ['']
    for dimension in dimension_list:
        temp_list = []
        for i in range(dimension):
            for j in return_list:
                temp_list.append(j+str(i))
        return_list = temp_list

    return return_list


def Ms_BL(return_frame, switch_map, lam, P, Q, conf_mat):
    temp_columns = list(return_frame.columns)

    temp_Ms_list = []
    for each in temp_columns:
        if switch_map[each] != 1:
            temp_std, temp_Coef, temp_transMat, temp_prob_smo = Ms_R(list(return_frame[each]), switch_map[each])
            #print temp_Coef
            #print temp_std
        else:
            temp_std = np.array([np.std(list(return_frame[each]))]*2)
            temp_Coef = np.array([np.mean(list(return_frame[each]))]*2)
            temp_transMat = np.array([[1,0],[0,1]]).reshape(2,2)
            temp_prob_smo = np.array([[0.5]*len(return_frame[each]),[0.5]*len(return_frame[each])]).T
        temp_Ms_list.append([temp_std, temp_Coef, temp_transMat, temp_prob_smo])
    Ms_frame = pd.DataFrame(temp_Ms_list, index=temp_columns, columns=['std','Coef','transMat','prob_smo']).T

    temp_cov_list = []
    for each_i in temp_columns:
        for each_j in temp_columns[temp_columns.index(each_i)+1:]:
            temp_cov_mat = Cross_Cov(return_frame[each_i], return_frame[each_j], Ms_frame[each_i]['Coef'], Ms_frame[each_j]['Coef'], Ms_frame[each_i]['prob_smo'], Ms_frame[each_j]['prob_smo'])
            temp_cov_list.append(temp_cov_mat)
    #print temp_cov_list

    Tree = Tree_Gen(switch_map, temp_columns)
    exp_ret_list = []
    cov_mat_list = []
    mu_wgt_list = []
    for each_leaf in Tree:
        cov_mat_temp = []
        exp_ret_temp = []
        for i in range(len(temp_columns)):
            for j in range(len(temp_columns)):
                if i == j:
                    cov_mat_temp.append((Ms_frame[temp_columns[i]]['std'][int(each_leaf[i])])**2)
                    exp_ret_temp.append(Ms_frame[temp_columns[i]]['Coef'][int(each_leaf[i])])
                else:
                    if i < j :
                        location = len(temp_columns)*(i+1)-sum(range(i+2))-(len(temp_columns)-j)
                        cov_mat_temp.append(temp_cov_list[location][int(each_leaf[i]),int(each_leaf[j])])
                    else:
                        location = len(temp_columns)*(j+1)-sum(range(j+2))-(len(temp_columns)-i)
                        cov_mat_temp.append(temp_cov_list[location][int(each_leaf[j]),int(each_leaf[i])])
        exp_ret = pd.DataFrame(exp_ret_temp, index=temp_columns)*4


        cov_mat = np.array(cov_mat_temp).reshape(len(temp_columns), len(temp_columns))*4
        cov_mat = pd.DataFrame(cov_mat, columns=temp_columns, index=temp_columns)

        #这里决定是否从风险平价配置出发
        '''
        omega = np.matrix(cov_mat.values)
        mkt_wgt = Risk_Parity_Weight(cov_mat)
        '''
        com_ret, com_cov_mat = Combined_Return_Distribution_MS(2, exp_ret, cov_mat, tau, P, Q, conf_mat)
        exp_ret_list.append(com_ret)
        cov_mat_list.append(com_cov_mat)
        #mu_wgt = Max_Utility_Weight_new(exp_ret, cov_mat, lam, [(0.0,None)]*len(temp_columns))
        #mu_wgt_list.append(mu_wgt)

    prob_list =[]
    for each_leaf in Tree:
        prob_leaf = 1
        for i in range(len(temp_columns)):
            stat = int(each_leaf[i])
            trans_mat = Ms_frame[temp_columns[i]]['transMat']
            temp_prob = sum(Ms_frame[temp_columns[i]]['prob_smo'][-1,:]*trans_mat[:,stat])
            #temp_prob = Ms_frame[temp_columns[i]]['prob_smo'][-1,0]*trans_mat[0,stat] + Ms_frame[temp_columns[i]]['prob_smo'][-1,1]*trans_mat[1,stat]
            prob_leaf = prob_leaf * temp_prob
        prob_list.append(prob_leaf)
        #print prob_list
    '''
    filt_prob_list = []
    for each_prob in prob_list:
        if each_prob == max(prob_list):
            filt_prob_list.append(1.0)
        else:
            filt_prob_list.append(0.0)
    prob_list = filt_prob_list
    '''
    '''
    prob_wgt_list = []
    for i in range(len(Tree)):
        prob_wgt_list.append(mu_wgt_list[i]*prob_list[i])
    '''
    mu_wgt_ms = Max_Utility_Weight_MS(exp_ret_list, cov_mat_list, prob_list, lam, [(0.0,None)]*len(temp_columns)).round(3)
    #sum(prob_wgt_list),
    return mu_wgt_ms

'''
ratio_list = []
for ii in range(100):
    test_list_1 = Ms_Simulation(250)
    test_list_2 = Ms_Simulation(250)
    test_frame = pd.DataFrame(np.array([test_list_1,test_list_2]).T, columns=['A','B'])
    wgt = Ms_RP(test_frame, {'A':True, 'B':True})
    ratio = wgt['A']
    print ratio
    ratio_list.append(ratio)

print max(ratio_list)
print min(ratio_list)
print np.mean(ratio_list)
'''


portfolio_name = "wenjian"
History_Data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/History_Data.xlsx")
Predict_Data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/Predict_Data.xlsx")
asset_list = ["bond_whole", "stock_large", "stock_small",
              "stock_HongKong", "stock_US", "gold"]
bnds = [(0.0, None), (0.0, None), (0.0, None),
        (0.0, None), (0.0, None), (0.0, None)]
asset_level_1 = pd.Series([-0.01, -0.08, -0.08, -0.08, -0.08, -0.08], index=asset_list)
asset_level_2 = pd.Series([-0.02, -0.16, -0.16, -0.16, -0.16, -0.16], index=asset_list)
#bnds = [(0.1, 0.6), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.0, 0.3)]
switch_map = {'bond_whole':1, 'stock_large':2, 'stock_small':2, 'stock_HongKong':2, 'stock_US':2, 'gold':2}
r = pr.R(use_pandas=True)
r("library(MSwM)")

year_delta = 5
tau = 1.0
if portfolio_name == "wenjian":
    lam = 2.3 #进取-1.9 平衡-2.0 稳健-2.3
    money_weight = 0.75
elif portfolio_name == "pingheng":
    lam = 1.9
    money_weight = 0.85
elif portfolio_name == "jinqu":
    lam = 1.7
    money_weight = 0.95
else:
    raise Exception("Wrong portfolio_name!")

pct_list = []
weight_list = []
date_list = []
asset_drawdown = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=asset_list)
asset_position = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=asset_list)

for each_date in Predict_Data.index[80:-1]:
    last_date = History_Data.index[list(Predict_Data.index).index(each_date)-1]  # 当前月份日期
    next_date = each_date  # 下一月份日期
    if last_date.month <= 11:
        start_year = last_date.year - year_delta
        start_month = last_date.month + 1
    else:
        start_year = last_date.year - year_delta + 1
        start_month = 1

    # 历史收益数据与预测数据
    history_data = History_Data[asset_list][
        str(start_year) + '-' + str(start_month): last_date]
    predict_data = Predict_Data[asset_list][
        str(start_year) + '-' + str(start_month): last_date]


    #获得预测观点与信心矩阵
    P = np.diag([1] * len(asset_list))
    conf_list = list()
    for each in asset_list:
        conf_temp = ((history_data[each][str(start_year) + '-' + str(start_month):] -
                      predict_data[each][str(start_year) + '-' + str(start_month):])**2).mean() * 12.0
        conf_list.append(conf_temp)
    conf_mat = np.matrix(np.diag(conf_list))
    Q = np.matrix(Predict_Data[asset_list].loc[next_date])


    weight_bl = Ms_BL(history_data, switch_map, 2, P, Q, conf_mat)
    print weight_bl


    #计算各资产回撤
    for each_asset in asset_list:
        temp_drawdown = (asset_drawdown[each_asset]+1.0)*(history_data[each_asset][-1]+1.0)-1
        if temp_drawdown >= 0:
            temp_drawdown = 0
        else:
            pass
        asset_drawdown[each_asset] = temp_drawdown

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

    weight_bl = weight_bl*asset_position
    print sum(weight_bl*History_Data[asset_list].loc[next_date])*money_weight + money_weight*History_Data["money"].loc[next_date]
    pct_list.append(sum(weight_bl*History_Data[asset_list].loc[next_date])*money_weight + money_weight*History_Data["money"].loc[next_date])
    weight_list.append(list(weight_bl))
    date_list.append(next_date)

pd.Series(np.array(pct_list), index=date_list).to_csv("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_%s.csv"%portfolio_name)
pd.DataFrame(np.array(weight_list), index=date_list, columns=asset_list).to_excel("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_%s_weight.xlsx"%portfolio_name)
