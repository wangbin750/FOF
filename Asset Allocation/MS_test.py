import pandas as pd
import numpy as np
import pyper as pr

data = pd.read_excel("/Users/WangBin-Mac/FOF/Global Allocation/SBG_US_M.xlsx")
data = data.interpolate()
data_W = data.pct_change().dropna()*100
data.columns
len(return_list)
return_list = list(data_W['SP500'])[0:165]
return_list = list(data_W['SP500'])[165:329]
return_list = list(data_W['SP500'])[329:]
return_list = list(data["000300.SH"])
return_frame = pd.DataFrame(np.array([return_list, [1]*len(return_list)]).T, columns=['return', 'One'])
r = pr.R(use_pandas=True)
r("library(MSwM)")

return_frame

r("rm(list = ls())")
r.assign("rframe", return_frame)
r('''
lm_return <- lm(formula=return~0+One, data=rframe)
lm_mswm <- msmFit(lm_return, k=3, p=0, sw=c(T,T))
rstd <- lm_mswm@std
rCoef <- lm_mswm@Coef
rtransMat <- lm_mswm@transMat
rprob_smo <- lm_mswm@Fit@smoProb[-1,]
#print(lm_mswm@Fit@logLikel)
raic_mswm <- AIC(lm_mswm)
raic_lm <- AIC(lm_return)
rlogLikel <- lm_mswm@Fit@logLikel
summary(lm_mswm)
''')
r.get("rstd")
test == None
r.get("rCoef")
print r.get("rtransMat")
r.get("raic_mswm")
r.get("raic_lm")
r.get("rlogLikel")
r.get("rsummary")
pd.DataFrame(r.get("rprob_smo")).to_clipboard()
'''
round([1.123123,1.214125],4)
test = np.array([[1.1213124,2.12441241,3.125164],[1.1213124,2.12441241,3.125164]])
test = [1.1213124,2.12441241,3.125164]
a = np.round(test,4)[2]
a
test[0]
np.where(test, round(test, 4), 0)
'''

'''
$std
$model
$Coef
$seCoef
$transMat
$iniProb
$call
$k
$switch
$p
$Fit
    slot"CondMean"
    slot"error"
    slot"Likel"
    slot"margLik"
    slot"filtProb"
    slot"smoProb"
    slot"smoTransMat"
    slot"logLikel"
$class
'''


def Performance(return_series, rf_ret):
    end_value = (return_series + 1).prod()
    annual_return = (return_series + 1).prod() ** (1/(len(return_series)/12.0)) - 1
    annual_variance = (return_series.var() * 12.0) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    return [end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown]

def Comparance(file_path):
    data = pd.read_csv(file_path)
    ms_per = Performance(data["ms_return"], 0.04)
    bm_per = Performance(data["bm_return"], 0.04)
    #ms_turnover = data[["SP500", "London_gold", "Barclays_US_bond"]].diff().dropna().abs().sum(axis=1).mean()*12
    ms_turnover = data[['000300.SH', 'AU9999.SGE', 'H11001.CSI']].diff().dropna().abs().sum(axis=1).mean()*12
    bm_turnover = data[["s_bm", "g_bm", "b_bm"]].diff().dropna().abs().sum(axis=1).mean()*12
    ms_per.append(ms_turnover)
    bm_per.append(bm_turnover)
    return pd.DataFrame(np.array([ms_per, bm_per]).T, columns=[file_path[-8:-4] + "_ms", file_path[-8:-4] + "_bm"], index=['end_value', 'annual_return', 'annual_variance', 'sharpe_ratio', 'max_drawdown', 'turnover'])

def Comparance(file_path):
    data = pd.read_csv(file_path)
    ms_per = Performance(data["ms_return"], 0.04)
    bm_per = Performance(data["bm_return"], 0.04)
    ms_turnover = data[["SP500", "Barclays_US_bond"]].diff().dropna().abs().sum(axis=1).mean()*12
    bm_turnover = data[["s_bm", "b_bm"]].diff().dropna().abs().sum(axis=1).mean()*12
    ms_per.append(ms_turnover)
    bm_per.append(bm_turnover)
    return pd.DataFrame(np.array([ms_per, bm_per]).T, columns=[file_path[-8:-4] + "_ms", file_path[-8:-4] + "_bm"], index=['end_value', 'annual_return', 'annual_variance', 'sharpe_ratio', 'max_drawdown', 'turnover'])

Comparance("F:\GitHub\FOF\RP_CN.csv").to_clipboard()

def Turnover(position_data, return_data):
    position_data.columns = return_data.columns
    turnover_list = []
    for each in range(len(position_data)-1):
        current_date = position_data.index[each]
        next_date = position_data.index[each+1]
        endvalue = sum(position_data.loc[current_date] * (return_data.loc[current_date]+1))
        current_position = position_data.loc[current_date] * (return_data.loc[current_date]+1)
        next_position = endvalue * position_data.loc[next_date]
        print current_date
        current_turnover = (next_position - current_position).abs().sum()/current_position.sum()
        turnover_list.append(current_turnover)
    return np.mean(turnover_list)*12

def Comparance(file_path, return_data):
    data = pd.read_csv(file_path, parse_dates=True, index_col=['Unnamed: 0'])
    ms_per = Performance(data["ms_return"], 0.025)
    bm_per = Performance(data["bm_return"], 0.025)
    ms_turnover = Turnover(data[["000300.SH", "H11001.CSI", "AU9999.SGE"]], return_data)
    #ms_turnover = Turnover(data[["SP500", "London_gold", "Barclays_US_bond"]], return_data)
    bm_turnover = Turnover(data[["s_bm", "g_bm", "b_bm"]], return_data)
    ms_per.append(ms_turnover)
    bm_per.append(bm_turnover)
    ms_percent = float(sum(data["ms_return"]>0))/len(data["ms_return"])
    bm_percent = float(sum(data["bm_return"]>0))/len(data["bm_return"])
    ms_per.append(ms_percent)
    bm_per.append(bm_percent)
    ms_bm_percent = float(sum((data["ms_return"]-data["bm_return"])>0))/len(data["ms_return"])
    ms_per.append(ms_bm_percent)
    bm_per.append(ms_bm_percent)
    posi_percent = float(sum((data["ms_return"][data["bm_return"]>0]-data["bm_return"][data["bm_return"]>0])>0))/len(data["bm_return"][data["bm_return"]>0])
    nega_percent = float(sum((data["ms_return"][data["bm_return"]<0]-data["bm_return"][data["bm_return"]<0])>0))/len(data["bm_return"][data["bm_return"]<0])
    ms_per.append(posi_percent)
    bm_per.append(posi_percent)
    ms_per.append(nega_percent)
    bm_per.append(nega_percent)
    return pd.DataFrame(np.array([ms_per, bm_per]).T, columns=[file_path[-8:-4] + "_ms", file_path[-8:-4] + "_bm"], index=['end_value', 'annual_return', 'annual_variance', 'sharpe_ratio', 'max_drawdown', 'turnover', 'profit%', 're-profit%', 'posi%', 'nega%'])

def Comparance(file_path, return_data):
    data = pd.read_csv(file_path, parse_dates=True, index_col=['Unnamed: 0'])
    ms_per = Performance(data["ms_return"], 0.04)
    bm_per = Performance(data["bm_return"], 0.04)
    ms_turnover = Turnover(data[["SP500", "Barclays_US_bond"]], return_data[["SP500", "Barclays_US_bond"]])
    bm_turnover = Turnover(data[["s_bm", "b_bm"]], return_data[["SP500", "Barclays_US_bond"]])
    ms_per.append(ms_turnover)
    bm_per.append(bm_turnover)
    ms_percent = float(sum(data["ms_return"]>0))/len(data["ms_return"])
    bm_percent = float(sum(data["bm_return"]>0))/len(data["bm_return"])
    ms_per.append(ms_percent)
    bm_per.append(bm_percent)
    ms_bm_percent = float(sum((data["ms_return"]-data["bm_return"])>0))/len(data["ms_return"])
    ms_per.append(ms_bm_percent)
    bm_per.append(ms_bm_percent)
    posi_percent = float(sum((data["ms_return"][data["bm_return"]>0]-data["bm_return"][data["bm_return"]>0])>0))/len(data["bm_return"][data["bm_return"]>0])
    nega_percent = float(sum((data["ms_return"][data["bm_return"]<0]-data["bm_return"][data["bm_return"]<0])>0))/len(data["bm_return"][data["bm_return"]<0])
    ms_per.append(posi_percent)
    bm_per.append(posi_percent)
    ms_per.append(nega_percent)
    bm_per.append(nega_percent)
    return pd.DataFrame(np.array([ms_per, bm_per]).T, columns=[file_path[-8:-4] + "_ms", file_path[-8:-4] + "_bm"], index=['end_value', 'annual_return', 'annual_variance', 'sharpe_ratio', 'max_drawdown', 'turnover', 'profit%', 're-profit%', 'posi%', 'nega%'])

return_data = pd.read_excel("/Users/WangBin-Mac/FOF/Global Allocation/SBG_US_M.xlsx").interpolate().dropna().pct_change().dropna()

data = pd.read_excel("F:\GitHub\FOF\Asset Allocation\stock_bond_gold_CN.xlsx")
return_data = (data/100+1).resample("M").prod().dropna()-1


Comparance("/Users/WangBin-Mac/FOF/RP_6_2001.csv", return_data).to_clipboard()

data = pd.read_csv(u"/Users/WangBin-Mac/Documents/研究生文件/asset allocation regime shift/MS结果/10-00.csv", parse_dates=True, index_col=['Unnamed: 0'])

(return_data["London_gold"]*data["s_bm"] + return_data["London_gold"]*data["b_bm"]).dropna().to_clipboard()
float(sum(data["bm_return"]>0))/len(data["ms_return"])
return_data


test = np.random.randn(10000)
test_1 = np.random.randn(10000)
test_2 = np.random.randn(10000)

test1 = -0.5*test + 0.5*test_1
test2 = 0.5*test + 0.5*test_2

test = pd.DataFrame(np.array([test1, test2]).T, columns=['a', 'b'])
test3 = test[test['a']>0]
test3[test3['b']>0].corr()

test.corr()

import pandas as pd
test = pd.DataFrame([1,2,3,4])
test.to_string().replace('\n', '')
