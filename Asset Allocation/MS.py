# coding=utf-8

import pandas as pd
import numpy as np
import pyper as pr
#import matplotlib.pyplot as plt

from Allocation_Method import Risk_Parity_Weight, Max_Utility_Weight_new

#Simulation
def Ms_Simulation(length, p=0.9, q=0.8, mean_p=0.1, mean_q=-0.1, std_p=0.05, std_q=0.15):
    temp_list = []
    indicator = 1
    for i in range(length):
        if indicator == 1:
            temp_ran = np.random.uniform(0, 1)
            if temp_ran <= p:
                temp_data = np.random.randn() * std_p + mean_p
            else:
                temp_data = np.random.randn() * std_q + mean_q
                indicator = 0
        else:
            temp_ran = np.random.uniform(0, 1)
            if temp_ran <= q:
                temp_data = np.random.randn() * std_q + mean_q
            else:
                temp_data = np.random.randn() * std_p + mean_p
                indicator = 1
        temp_list.append(temp_data)
    return temp_list

def Ms_R(return_list):
    #return_list = list(data_M["AU9999.SGE"])
    return_frame = pd.DataFrame(np.array([return_list, [1]*len(return_list)]).T, columns=['return', 'One'])
    r.assign("rframe", return_frame)
    r('''
    lm_return <- lm(formula=return~0+One, data=rframe)
    lm_mswm <- msmFit(lm_return, k=2, p=0, sw=c(T,T))
    rstd <- lm_mswm@std
    rCoef <- lm_mswm@Coef
    rtransMat <- lm_mswm@transMat
    rprob_smo <- lm_mswm@Fit@smoProb[-1,]
    #print(lm_mswm@Fit@logLikel)
    raic_mswn <- AIC(lm_mswm)
    raic_lm <- AIC(lm_return)
    ''')
    aic_mswm = r.get("raic_mswm")
    aic_lm = r.get("raic_lm")
    if aic_mswm < aic_lm:
        std = np.round(r.get("rstd"),4)
        Coef = np.round(np.array(r.get("rCoef")[' One ']),4)
        transMat = np.round(r.get("rtransMat").T,4)
        prob_smo = np.round(r.get("rprob_smo"),4)
    else:
        std = np.array([np.std(return_list)]*2)
        Coef = np.array([np.mean(return_list)]*2)
        transMat = np.array([[1,0],[0,1]]).reshape(2,2)
        prob_smo = np.array([[0.5]*len(return_list),[0.5]*len(return_list)]).T

    return std, Coef, transMat, prob_smo

def Cross_Cov(return_list_1, return_list_2, Coef_1, Coef_2, prob_smo_1, prob_smo_2):
    temp_list = []
    for i in range(2):
        for j in range(2):
            temp_cov = np.sum(prob_smo_1[:,i]*prob_smo_2[:,j]*(return_list_1 - Coef_1[i])*(return_list_2 - Coef_2[j]))/np.sum(prob_smo_1[:,i]*prob_smo_2[:,j])
            temp_list.append(temp_cov)
    cov_mat = np.array(temp_list).reshape(2,2)
    return cov_mat

def Tree_Gen(length):
    temp_list = []
    for i in range(2**length):
        temp_bin = bin(i)
        temp_bin = temp_bin.replace('0b', '')
        if len(temp_bin) < length:
            temp_bin = '0'*(length-len(temp_bin)) + temp_bin
        temp_list.append(temp_bin)
    return temp_list

# intTree_Gen(4)[1][0]
def Ms_RP(return_frame, switch_map):
    temp_columns = list(return_frame.columns)

    temp_Ms_list = []
    for each in temp_columns:
        if switch_map[each]:
            temp_std, temp_Coef, temp_transMat, temp_prob_smo = Ms_R(list(return_frame[each]))
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

    Tree = Tree_Gen(len(temp_columns))
    rp_wgt_list = []
    for each_leaf in Tree:
        cov_mat_list = []
        for i in range(len(temp_columns)):
            for j in range(len(temp_columns)):
                if i == j:
                    cov_mat_list.append((Ms_frame[temp_columns[i]]['std'][int(each_leaf[i])])**2)
                else:
                    if i < j :
                        location = len(temp_columns)*(i+1)-sum(range(i+2))-(len(temp_columns)-j)
                        cov_mat_list.append(temp_cov_list[location][int(each_leaf[i]),int(each_leaf[j])])
                    else:
                        location = len(temp_columns)*(j+1)-sum(range(j+2))-(len(temp_columns)-i)
                        cov_mat_list.append(temp_cov_list[location][int(each_leaf[j]),int(each_leaf[i])])
        cov_mat = np.array(cov_mat_list).reshape(len(temp_columns), len(temp_columns))
        cov_mat = pd.DataFrame(cov_mat, columns=temp_columns, index=temp_columns)
        rp_wgt = Risk_Parity_Weight(cov_mat)
        rp_wgt_list.append(rp_wgt)

    prob_list =[]
    for each_leaf in Tree:
        prob_leaf = 1
        for i in range(len(temp_columns)):
            stat = int(each_leaf[i])
            trans_mat = Ms_frame[temp_columns[i]]['transMat']
            temp_prob = Ms_frame[temp_columns[i]]['prob_smo'][-1,0]*trans_mat[0,stat] + Ms_frame[temp_columns[i]]['prob_smo'][-1,1]*trans_mat[1,stat]
            prob_leaf = prob_leaf * temp_prob
        prob_list.append(prob_leaf)
    '''
    filt_prob_list = []
    for each_prob in prob_list:
        if each_prob == max(prob_list):
            filt_prob_list.append(1.0)
        else:
            filt_prob_list.append(0.0)
    prob_list = filt_prob_list
    '''
    prob_wgt_list = []
    for i in range(len(Tree)):
        prob_wgt_list.append(rp_wgt_list[i]*prob_list[i])

    return sum(prob_wgt_list)


def Ms_MU(return_frame, switch_map, lam):
    temp_columns = list(return_frame.columns)

    temp_Ms_list = []
    for each in temp_columns:
        if switch_map[each]:
            temp_std, temp_Coef, temp_transMat, temp_prob_smo = Ms_R(list(return_frame[each]))
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

    Tree = Tree_Gen(len(temp_columns))
    rp_wgt_list = []
    for each_leaf in Tree:
        cov_mat_list = []
        exp_ret_list = []
        for i in range(len(temp_columns)):
            for j in range(len(temp_columns)):
                if i == j:
                    cov_mat_list.append((Ms_frame[temp_columns[i]]['std'][int(each_leaf[i])])**2)
                    exp_ret_list.append(Ms_frame[temp_columns[i]]['Coef'][int(each_leaf[i])])
                else:
                    if i < j :
                        location = len(temp_columns)*(i+1)-sum(range(i+2))-(len(temp_columns)-j)
                        cov_mat_list.append(temp_cov_list[location][int(each_leaf[i]),int(each_leaf[j])])
                    else:
                        location = len(temp_columns)*(j+1)-sum(range(j+2))-(len(temp_columns)-i)
                        cov_mat_list.append(temp_cov_list[location][int(each_leaf[j]),int(each_leaf[i])])
        exp_ret = pd.DataFrame(exp_ret_list, index=temp_columns)
        cov_mat = np.array(cov_mat_list).reshape(len(temp_columns), len(temp_columns))
        cov_mat = pd.DataFrame(cov_mat, columns=temp_columns, index=temp_columns)
        rp_wgt = Max_Utility_Weight_new(exp_ret, cov_mat, lam, [(0.0,None)]*len(temp_columns))
        print exp_ret
        print cov_mat
        #print rp_wgt
        rp_wgt_list.append(rp_wgt)

    prob_list =[]
    for each_leaf in Tree:
        prob_leaf = 1
        for i in range(len(temp_columns)):
            stat = int(each_leaf[i])
            trans_mat = Ms_frame[temp_columns[i]]['transMat']
            temp_prob = Ms_frame[temp_columns[i]]['prob_smo'][-1,0]*trans_mat[0,stat] + Ms_frame[temp_columns[i]]['prob_smo'][-1,1]*trans_mat[1,stat]
            prob_leaf = prob_leaf * temp_prob
        prob_list.append(prob_leaf)
    '''
    filt_prob_list = []
    for each_prob in prob_list:
        if each_prob == max(prob_list):
            filt_prob_list.append(1.0)
        else:
            filt_prob_list.append(0.0)
    prob_list = filt_prob_list
    '''
    print rp_wgt_list
    print prob_list
    prob_wgt_list = []
    for i in range(len(Tree)):
        prob_wgt_list.append(rp_wgt_list[i]*prob_list[i])

    return sum(prob_wgt_list)

def Ms_Multi(return_frame, switch_map, lam):
    temp_columns = list(return_frame.columns)

    temp_Ms_list = []
    for each in temp_columns:
        if switch_map[each]:
            temp_std, temp_Coef, temp_transMat, temp_prob_smo = Ms_R(list(return_frame[each]))
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

    Tree = Tree_Gen(len(temp_columns))
    rp_wgt_list = []
    mu_wgt_list = []
    for each_leaf in Tree:
        cov_mat_list = []
        exp_ret_list = []
        for i in range(len(temp_columns)):
            for j in range(len(temp_columns)):
                if i == j:
                    cov_mat_list.append((Ms_frame[temp_columns[i]]['std'][int(each_leaf[i])])**2)
                    exp_ret_list.append(Ms_frame[temp_columns[i]]['Coef'][int(each_leaf[i])])
                else:
                    if i < j :
                        location = len(temp_columns)*(i+1)-sum(range(i+2))-(len(temp_columns)-j)
                        cov_mat_list.append(temp_cov_list[location][int(each_leaf[i]),int(each_leaf[j])])
                    else:
                        location = len(temp_columns)*(j+1)-sum(range(j+2))-(len(temp_columns)-i)
                        cov_mat_list.append(temp_cov_list[location][int(each_leaf[j]),int(each_leaf[i])])
        exp_ret = pd.DataFrame(exp_ret_list, index=temp_columns)
        cov_mat = np.array(cov_mat_list).reshape(len(temp_columns), len(temp_columns))
        cov_mat = pd.DataFrame(cov_mat, columns=temp_columns, index=temp_columns)
        rp_wgt = Risk_Parity_Weight(cov_mat)
        mu_wgt = Max_Utility_Weight_new(exp_ret, cov_mat, lam, [(0.0,None)]*len(temp_columns))
        #print mu_wgt
        #print "----"
        rp_wgt_list.append(rp_wgt)
        mu_wgt_list.append(mu_wgt)

    prob_list =[]
    for each_leaf in Tree:
        prob_leaf = 1
        for i in range(len(temp_columns)):
            stat = int(each_leaf[i])
            trans_mat = Ms_frame[temp_columns[i]]['transMat']
            temp_prob = Ms_frame[temp_columns[i]]['prob_smo'][-1,0]*trans_mat[0,stat] + Ms_frame[temp_columns[i]]['prob_smo'][-1,1]*trans_mat[1,stat]
            prob_leaf = prob_leaf * temp_prob
        prob_list.append(prob_leaf)
    '''
    filt_prob_list = []
    for each_prob in prob_list:
        if each_prob == max(prob_list):
            filt_prob_list.append(1.0)
        else:
            filt_prob_list.append(0.0)
    prob_list = filt_prob_list
    '''
    prob_rp_wgt_list = []
    prob_mu_wgt_list = []

    for i in range(len(Tree)):
        prob_rp_wgt_list.append(rp_wgt_list[i]*prob_list[i])
        prob_mu_wgt_list.append(mu_wgt_list[i]*prob_list[i])

    return {"rp_wgt":sum(prob_rp_wgt_list).round(3),"mu_wgt":sum(prob_mu_wgt_list).round(3)}


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

#模拟数据
'''
MS_list = []
BM_list = []
for each in range(100):
    test_list_1 = Ms_Simulation(240, 0.94798285, 0.94177644, 0.011348197, 0.0025831495, 0.02178261, 0.05045428)
    test_list_2 = Ms_Simulation(240, 0.90419533, 0.97432935, 0.021510161, 0.0016860435, 0.09058209, 0.03235686)
    test_list_3 = Ms_Simulation(240, 1, 1, 0.00616607, 0.00616607, 0.014926817, 0.014926817)
    ms_list = []
    bm_list = []
    for i in range(120):
        tl_cut_1 = test_list_1[:120+i]
        tl_cut_2 = test_list_2[:120+i]
        tl_cut_3 = test_list_3[:120+i]
        test_frame = pd.DataFrame(np.array([tl_cut_1,tl_cut_2,tl_cut_3]).T, columns=['A','B','C'])

        r = pr.R(use_pandas=True)
        r("library(MSwM)")
        wgt = Ms_MU(test_frame, {'A':True, 'B':True, 'C':False})

        wgt_rp = Max_Utility_Weight(pd.DataFrame(test_frame.mean()), test_frame.cov(), 3, [(0.0,None)]*3)
        ms_return = wgt['A']*test_list_1[120+i] + wgt['B']*test_list_2[120+i] + wgt['C']*test_list_3[120+i]
        bm_return = wgt_rp['A']*test_list_1[120+i] + wgt_rp['B']*test_list_2[120+i] + wgt_rp['C']*test_list_3[120+i]

        #test_frame = pd.DataFrame(np.array([tl_cut_1,tl_cut_2]).T, columns=['A','B'])

        #wgt = Ms_RP(test_frame, {'A':True, 'B':True})

        #wgt = pd.Series([0.6, 0.4], index=['A','B'])

        #ms_return = wgt['A']*test_list_1[250+i] + wgt['B']*test_list_2[250+i]
        #bm_return = (test_list_1[250+i] + test_list_2[250+i])/2
        #print ms_return
        #print bm_return
        ms_list.append(ms_return)
        bm_list.append(bm_return)

    MS_list.append((pd.Series(ms_list)+1).prod())
    BM_list.append((pd.Series(bm_list)+1).prod())
    print str(each)
    print (pd.Series(ms_list)+1).prod()
    print pd.Series(ms_list).std()
    print (pd.Series(bm_list)+1).prod()
    print pd.Series(bm_list).std()
    print "-----"

print np.mean(MS_list)
print np.mean(BM_list)
'''
#中国实际数据
'''
data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/stock_bond_gold_CN.xlsx")
#data_W = (data/100+1).resample("W").prod().dropna()-1
data_M = (data/100+1).resample("W").prod().dropna()-1
data_W = data_M*100
#data_W = data.pct_change().dropna()*100

ms_list = []
bm_list = []
for each in range(250,len(data_W)-1):
    #each = 95
    #data_M.index[each]
    r = pr.R(use_pandas=True)
    r("library(MSwM)")
    data_frame = data_W[:data_W.index[each]]

    wgt = Ms_RP(data_frame, {'000300.SH':True, 'AU9999.SGE':False, 'H11001.CSI':False})
    #wgt = Ms_RP(data_frame, {'SP500':True, 'London_gold':False, 'Barclays_US_bond':False})
    wgt_bm = Risk_Parity_Weight(data_frame.cov())

    #wgt = pd.Series([0.2, 0.2, 0.6], index=data_frame.columns)
    print wgt
    print data_W.loc[data_W.index[each+1]]
    ms_return = np.sum(wgt*data_W.loc[data_W.index[each+1]])/100
    bm_return = np.sum(wgt_bm*data_W.loc[data_W.index[each+1]])/100
    print data_W.index[each+1]
    #print ms_return
    #print bm_return
    ms_list.append(ms_return)
    bm_list.append(bm_return)

print (pd.Series(ms_list)+1).prod()
print np.std(pd.Series(ms_list))
print (pd.Series(bm_list)+1).prod()
print np.std(pd.Series(bm_list))
'''

#美国实际数据

data = pd.read_excel("F:\GitHub\FOF\Global Allocation\SBG_US_M.xlsx")

data = pd.read_excel("/Users/WangBin-Mac/FOF/Global Allocation/SBG_US_M.xlsx")

data = data.interpolate()
data = data.dropna().pct_change().dropna()

data


#data_W = data.pct_change().dropna()*100


rp_result_list = []
mu_result_list = []
index_list = []
r = pr.R(use_pandas=True)
r("library(MSwM)")
for each in range(119,len(data)-1):
    #each = 95
    #data_M.index[each]

    #data_frame = data[:data.index[each]]
    data_frame = data[data.index[each-119]:data.index[each]]

    #data_frame = data_frame[['SP500', 'Barclays_US_bond']]
    '''
    mu_wgt = Ms_MU(data_frame, {'SP500':True, 'Barclays_US_bond':False}, 2)
    print each
    print mu_wgt

    rp_wgt = Ms_RP(data_frame, {'SP500':True, 'Barclays_US_bond':False})
    '''

    multi_wgt = Ms_Multi(data_frame, {'SP500':True, 'London_gold':True, 'Barclays_US_bond':False}, 2)
    mu_wgt = multi_wgt["mu_wgt"]
    rp_wgt = multi_wgt["rp_wgt"]
    print mu_wgt
    print rp_wgt

    #data_frame = data[data.index[each-120]:data.index[each]]
    rp_wgt_bm = Risk_Parity_Weight(data_frame.cov()).round(3)
    mu_wgt_bm = Max_Utility_Weight_new(pd.DataFrame(data_frame.mean()), data_frame.cov(), 2, [(0.0,None)]*3).round(3)

    '''
    mu_wgt = Ms_MU(data_frame, {'SP500':True, 'London_gold':True, 'Barclays_US_bond':False})
    rp_wgt = Ms_RP(data_frame, {'SP500':True, 'London_gold':True, 'Barclays_US_bond':False})
    #data_frame = data[data.index[each-120]:data.index[each]]
    rp_wgt_bm = Risk_Parity_Weight(data_frame.cov())
    mu_wgt_bm = Max_Utility_Weight_new(pd.DataFrame(data_frame.mean()), data_frame.cov(), 2, [(0.0,None)]*3)
    '''

    mu_ms_return = np.sum(mu_wgt*data.loc[data.index[each+1]])
    mu_bm_return = np.sum(mu_wgt_bm*data.loc[data.index[each+1]])
    rp_ms_return = np.sum(rp_wgt*data.loc[data.index[each+1]])
    rp_bm_return = np.sum(rp_wgt_bm*data.loc[data.index[each+1]])


    #print bm_return
    mu_result = list(mu_wgt)+list(mu_wgt_bm)+[mu_ms_return]+[mu_bm_return]
    rp_result = list(rp_wgt)+list(rp_wgt_bm)+[rp_ms_return]+[rp_bm_return]

    mu_result_list.append(mu_result)
    rp_result_list.append(rp_result)
    index_list.append(data.index[each+1])
    print data.index[each+1]

pd.DataFrame(np.array(mu_result_list), columns=list(data_frame.columns)+["s_bm", "g_bm", "b_bm"]+["ms_return", "bm_return"], index=index_list).to_csv("MU_3.csv")
pd.DataFrame(np.array(rp_result_list), columns=list(data_frame.columns)+["s_bm", "g_bm", "b_bm"]+["ms_return", "bm_return"], index=index_list).to_csv("RP_3.csv")

'''
pd.DataFrame(np.array(mu_result_list), columns=list(data.columns)+["s_bm", "g_bm", "b_bm"]+["ms_return", "bm_return"], index=index_list).to_csv("MU_e.csv")
pd.DataFrame(np.array(rp_result_list), columns=list(data.columns)+["s_bm", "g_bm", "b_bm"]+["ms_return", "bm_return"], index=index_list).to_csv("Rp_e.csv")
'''
