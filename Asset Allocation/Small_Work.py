# coding=utf-8

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/History_Data.xlsx")


return_series = data["bond"]
drawdown = ((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax()
drawdown = drawdown[drawdown>0]
drawdown.hist(bins=50)
print drawdown.quantile(0.5)
plt.show()
drawdown.plot()

data = pd.read_excel("History_Data.xlsx")
return_series = data["bond"]
drawdown = ((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax()
drawdown = drawdown[drawdown>0]
drawdown.hist(bins=30)
plt.show()

return_series = data["gold"]
de_series = return_series[return_series<0]
de_list = []
cum_de = 0.0
for each in return_series:
    if each < 0:
        cum_de = (1.0+cum_de)*(1.0+each)-1
    else:
        de_list.append(cum_de)
        cum_de = 0.0

de_series = pd.Series(de_list)
(-de_series[de_series<0]).hist(bins=30)
(-de_series[de_series<0]).quantile(0.5)
plt.show()

return_series = data["stock_large"]
return_series.quantile(0.10)
return_series = data["stock_small"]
return_series.quantile(0.10)
return_series = data["stock_US"]
return_series.quantile(0.10)
return_series = data["stock_HongKong"]
return_series.quantile(0.10)
return_series = data["bond"]
return_series.quantile(0.10)
return_series = data["gold"]
return_series.quantile(0.10)

data = pd.read_excel(u"/Users/WangBin-Mac/263网盘/金建投资/FOF投顾/201712报告/收益分解/nv_index.xlsx")
data = pd.read_excel(u"E:/263网盘/金建投资/00-FOF投顾/201804报告/收益分解/nv_index.xlsx")
return_series = data[u"稳健组合"].pct_change().dropna()
Performance(return_series, 0.04)

return_series = data[u"平衡组合"].pct_change().dropna()
Performance(return_series, 0.04)

return_series = data[u"进取组合"].pct_change().dropna()
Performance(return_series, 0.04)

def Performance(return_series, rf_ret):
    end_value = (return_series + 1).prod()
    annual_return = (return_series + 1).prod() ** (1/(len(return_series)/240.0)) - 1
    annual_variance = (return_series.var() * 240.0) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    return [end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown]

from WindPy import *
w.start()
index_code_list = ["000300.SH", "000905.SH", "SPX.GI", "HSI.HI", "H11001.CSI", "AU9999.SGE", "H11025.CSI"]
temp_data = w.wsd(index_code_list, "close", "2017-02-01", "2017-03-30")
data = pd.DataFrame(np.array(temp_data.Data).T, index=temp_data.Times, columns=temp_data.Codes)
data.to_excel("D:\\FOF\\index.xlsx")

import pandas as pd
import numpy as np

data = pd.read_excel(u"/Users/WangBin-Mac/FOF/Asset Allocation/大类资产_D.xlsx")
gold_data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/gold.xlsx")
data = data/100

data = pd.merge(data, gold_data, how='left', left_index=True, right_index=True)
data.to_excel("/Users/WangBin-Mac/FOF/Asset Allocation/History_Data_D.xlsx")

if __name__ == "__main__":

    pool = multiprocessing.Pool(processes=4)
    result_list = []
    for each_i in Aminud_data.columns:
        for each_j in Aminud_data.columns:
            result_list.append(pool.apply_async(func, (each_i, each_j, )))

    pool.close()
    pool.join()

    for res in result_list:
        temp = res.get()
        Granger_result.loc[temp[1], temp[0]] = temp[2]


data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_cppi_bl_step.xlsx")
plt.scatter(data['md'], data['ar'])
plt.savefig("/Users/WangBin-Mac/FOF/Asset Allocation/md-ar.png")
plt.cla()


efficient_set = []
for index, row in data.iterrows():
    temp_ar = row.ar
    temp_av = row.av
    indicator = 0
    for each_index, each_row in data.iterrows():
        if each_row.ar > temp_ar and each_row.av < temp_av:
            break
        else:
            indicator += 1
    if indicator == len(data):
        efficient_set.append(list(row))
    else:
        pass

efficient_data = pd.DataFrame(np.array(efficient_set), columns=data.columns)
plt.scatter(efficient_data['av'], efficient_data['ar'])
plt.savefig("/Users/WangBin-Mac/FOF/Asset Allocation/md-ar2.png")
efficient_data.to_excel("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_cppi_bl_step_efficient.xlsx")

efficient_md_set = []
for index, row in data.iterrows():
    temp_ar = row.ar
    temp_md = row.md
    indicator = 0
    for each_index, each_row in data.iterrows():
        if each_row.ar > temp_ar and each_row.md < temp_md:
            break
        else:
            indicator += 1
    if indicator == len(data):
        efficient_md_set.append(list(row))
    else:
        pass

efficient_md_data = pd.DataFrame(np.array(efficient_md_set), columns=data.columns)
plt.scatter(efficient_md_data['md'], efficient_md_data['ar'])
plt.show()
efficient_md_data.to_excel("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_cppi_bl_step_efficient_md.xlsx")

import pandas as pd
import numpy as np
import statsmodels.api as sm


def effect_analysis(data, res, param_name):
    if len(res.params) == 5:
        b = res.params[param_name]
        if b > 0:
            return param_name+":positive"
        else:
            return param_name+":negative"
    elif len(res.params) == 9:
        a = res.params[param_name+'2']
        b = res.params[param_name]
        extreme_point = -b/(2*a)
        param_range = set(data[param_name])
        min_point = min(param_range)
        max_point = max(param_range)
        if extreme_point > min_point and extreme_point < max_point:
            if a*extreme_point**2+b*extreme_point < a*min_point**2+b*min_point:
                return param_name+":negative-positive - [%.3f, %.3f, %.3f]"%(min_point,extreme_point,max_point)

            else:
                return param_name+":positive-negative - [%.3f, %.3f, %.3f]"%(min_point,extreme_point,max_point)
        else:
            if a*min_point**2+b*min_point < a*max_point**2+b*max_point:
                return param_name+":positive"
            else:
                return param_name+":negative"

def ols(Y, X):
    mod = sm.OLS(Y, X)
    res = mod.fit()
    #return res.summary()
    for each in X.columns[1:5]:
        print effect_analysis(X, res, each)

def ols(Y, X):
    mod = sm.OLS(Y, X)
    res = mod.fit()
    return res.summary()

data = pd.read_excel("/Users/WangBin-Mac/FOF/Asset Allocation/backtest_cppi_bl_step.xlsx")
data = data[(data['target_ret']!=0.15)&(data['target_ret']!=0.2)]

index_list = []
ar_duplicated = data['ar'][data['ar'].duplicated()].drop_duplicates()
av_duplicated = data['av'][data['av'].duplicated()].drop_duplicates()
md_duplicated = data['md'][data['md'].duplicated()].drop_duplicates()
for index, row in data.iterrows():
    if (row['ar'] in list(ar_duplicated)) and (row['av'] in list(av_duplicated)) and (row['md'] in list(md_duplicated)):
        pass
    else:
        index_list.append(index)

data = data.loc[index_list]

for each in data.columns[5:9]:
    data[each+'2'] = data[each]**2


X1 = data[data.columns[5:9]]
X2 = data[data.columns[5:]]
X1 = sm.add_constant(X1)
X2 = sm.add_constant(X2)


ols(data['ar'], X1)
print "--------"
ols(data['av'], X1)
print "--------"
ols(data['md'], X1)
print "--------"
ols(data['sr'], X1)

ols(data['ar'], X2)
print "--------"
ols(data['av'], X2)
print "--------"
ols(data['md'], X2)
print "--------"
ols(data['sr'], X2)

print ols(data['ar'], X1)
print ols(data['ar'], data[data.columns[5:9]])
print ols(data['md'], X1)
print ols(data['md'], data[data.columns[5:9]])

test = pd.Series([1,1,1,1,2,3,4])
3 in test

len(data[data['target_ret']== 0.2])


import pandas as pd
import numpy as np

data = pd.read_excel('/Users/WangBin-Mac/FOF/Asset Allocation/History_Data_D.xlsx')
data2 = pd.read_excel('/Users/WangBin-Mac/Desktop/sz50_D.xlsx')

data = pd.merge(data, pd.DataFrame(data2['000016.SH']/100),left_index=True, right_index=True,how='left')

data.to_excel('/Users/WangBin-Mac/FOF/Asset Allocation/History_Data_D.xlsx')


import numpy as np
import statsmodels.api as sm

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

data = Ms_Simulation(1000)

ms_model = sm.tsa.MarkovRegression(np.array(data), k_regimes=2, switching_variance=True)
ms_fit = ms_model.fit()
ms_fit.smoothed_marginal_probabilities[0]
ms_fit.initial_probabilities
ms_fit.params
ms_fit.regime_transition
dir(ms_fit)
print ms_fit.summary()

data = pd.read_excel(u"/Users/WangBin-Mac/Desktop/中债分期限指数.xlsx")
data.resample("M").last().pct_change().dropna().to_excel(u"/Users/WangBin-Mac/Desktop/中债分期限指数_pct.xlsx")

from Allocation_Method import Risk_Budget_Weight

cov = pd.DataFrame([[1.0,1.0,0.0], [0.8,1.0,0.0], [0.0,0.0,1.0]])
bud_l = [1.0/3.0, 1.0/3.0, 1.0/3.0]

print Risk_Budget_Weight(cov, bud_l)
