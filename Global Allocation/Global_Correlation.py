# coding=utf-8

import pandas as pd
import numpy as np
import socket
import matplotlib.pyplot as plt
from statsmodels.multivariate.pca import PCA


hostname = socket.gethostname()
if hostname == "DESKTOP-OGC5NH7":
    path = "E:/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "localhost":
    path = "/Users/WangBin-Mac/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "CICCB6CR7213VFT":
    path = "F:/GitHub/FOF/Global Allocation/"


Barclays_US_bond = pd.read_excel(path+"Barclays_US_bond.xlsx")
BloomBerg_commodity = pd.read_excel(path+"BloomBerg_commodity.xlsx")
FTSE_global_REITs = pd.read_excel(path+"FTSE_global_REITs.xlsx")
London_gold = pd.read_excel(path+"London_gold.xlsx")
MSCI_US_REITs = pd.read_excel(path+"MSCI_US_REITs.xlsx")
MSCI_emerging = pd.read_excel(path+"MSCI_emerging.xlsx")
MSCI_global = pd.read_excel(path+"MSCI_global.xlsx")
SP500 = pd.read_excel(path+"SP500.xlsx")
MSCI_exUS = pd.read_excel(path+"MSCI_exUS.xlsx")
Barclays_US_HY = pd.read_excel(path+"Barclays_US_HY.xlsx")
#Barclays_US_HY.resample("M").last().to_excel("/Users/WangBin-Mac/FOF/Global Allocation/Barclays_US_HY.xlsx")
Barclays_US_CB = pd.read_excel(path+"Barclays_US_CB.xlsx")
#Barclays_US_CB.resample("M").last().to_excel("/Users/WangBin-Mac/FOF/Global Allocation/Barclays_US_CB.xlsx")
Barclays_US_Treasury = pd.read_excel(path+"Barclays_US_Treasury.xlsx")
#Barclays_US_Treasury.resample("M").last().to_excel("/Users/WangBin-Mac/FOF/Global Allocation/Barclays_US_Treasury.xlsx")

data = pd.merge(Barclays_US_bond, Barclays_US_Treasury, how='outer', left_index=True, right_index=True)
data = pd.merge(data, Barclays_US_CB, how='outer', left_index=True, right_index=True)
data = pd.merge(data, Barclays_US_HY, how='outer', left_index=True, right_index=True)
data = pd.merge(data, SP500, how='outer', left_index=True, right_index=True)
data = pd.merge(data, MSCI_global, how='outer', left_index=True, right_index=True)
#data = pd.merge(data, MSCI_exUS, how='outer', left_index=True, right_index=True)
data = pd.merge(data, MSCI_emerging, how='outer', left_index=True, right_index=True)
data = pd.merge(data, MSCI_US_REITs, how='outer', left_index=True, right_index=True)
data = pd.merge(data, FTSE_global_REITs, how='outer', left_index=True, right_index=True)
data = pd.merge(data, BloomBerg_commodity, how='outer', left_index=True, right_index=True)
data = pd.merge(data, London_gold, how='outer', left_index=True, right_index=True)
#data_M[['SP500', 'MSCI_global']].corr().iloc[1,0]


data.resample("M").last().pct_change().to_excel(path+"All_Assets_M.xlsx")
'''
def Rolling_Correlation(df, lags):
    corr_list = list()
    for i in range(lags-1, len(df)):
        temp_df = df[df.index[i+1-lags]:df.index[i]]
        corr_list.append(temp_df.corr().iloc[0,1])
    return pd.DataFrame(corr_list, index=df.index[lags-1:], columns=[df.columns[0]+'*'+df.columns[1]])

data_M = pd.read_excel(path+"All_Assets_M.xlsx")
data_M.corr()
correl_frame = pd.DataFrame()
for each_i in data_M.columns:
    for each_j in data_M.columns[list(data_M.columns).index(each_i)+1:]:
        temp_data = data_M[[each_i, each_j]].dropna()
        temp_corr = Rolling_Correlation(temp_data, 24)
        correl_frame = pd.merge(correl_frame, temp_corr, how='outer', left_index=True, right_index=True)

correl_frame.to_excel(path+"All_Assets_correlframe.xlsx")
correl_frame.mean()[correl_frame.mean() > 0.8]
correl_frame.plot.violin()
plt.show()
plt.boxplot(correl_frame["Barclays_US_bond*Barclays_US_HY"])



plt.figure(figsize=(33,22))
for i in range(0,11):
    for j in range(i,11):
        if i == j:
            pass
        else:
            num = (j-1)*10+i+1
            plt.subplot(10,10,num)
            plt.ylim(-1,1)
            if i != 0:
                plt.yticks([])
            else:
                pass
            if j != 10:
                plt.xticks([])
            else:
                pass
            temp_name = data_M.columns[i] + "*" + data_M.columns[j]
            plt.violinplot(np.array(correl_frame[temp_name].dropna()), showmeans=True, vert=True)
plt.show()
'''
data_M = pd.read_excel(path+"All_Assets_M.xlsx")
correl_frame = pd.read_excel(path+"All_Assets_correlframe.xlsx")
fig = plt.figure(figsize=(33,22))
for i in range(0,11):
    for j in range(0,11):
        if i == j:
            if i == 0:
                ax1 = plt.subplot(11,11,1)
                plt.ylim(-1,1) #设置纵坐标范围
                plt.xlim(0.75,1.25) #设置横坐标范围
                plt.xticks([]) #不显示横坐标
                ax1.spines["bottom"].set_color("none") #隐藏下边线
                ax1.spines["right"].set_color("none") #隐藏右边线
                plt.plot([-1,-2,-3],[-1,-2,-3]) #在坐标范围外画一个虚图
            if j == 10:
                ax2 = plt.subplot(11,11,121)
                plt.ylim(-1,1)
                plt.xlim(0.75,1.25)
                plt.yticks([])
                ax2.spines["top"].set_color("none")
                ax2.spines["left"].set_color("none")
                plt.plot([-1,-2,-3],[-1,-2,-3])
        else:
            num = j*11+i+1
            plt.subplot(11,11,num)
            plt.ylim(-1,1)
            plt.xlim(0.75,1.25)
            if i != 0:
                plt.yticks([])
            else:
                pass
            if j != 10:
                plt.xticks([])
            else:
                pass
            if i < j:
                temp_name = data_M.columns[i] + "*" + data_M.columns[j]
            else:
                temp_name = data_M.columns[j] + "*" + data_M.columns[i]
            plt.violinplot(np.array(correl_frame[temp_name].dropna()), showmeans=True, vert=True)
            plt.hlines(0, 0, 2, colors='r', linestyles='dashed')
#plt.show()
fig.savefig(path+"All_Assets_correl.png")


'''
correl_mean_frame = pd.DataFrame(index=data_M.columns, columns=data_M.columns)
for each in correl_frame.mean().index:
    name_list = each.split("*")
    correl_mean_frame[name_list[0]][name_list[1]] = correl_frame.mean()[each]

correl_min_frame = pd.DataFrame(index=data_M.columns, columns=data_M.columns)
for each in correl_frame.min().index:
    name_list = each.split("*")
    correl_min_frame[name_list[0]][name_list[1]] = correl_frame.min()[each]

correl_max_frame = pd.DataFrame(index=data_M.columns, columns=data_M.columns)
for each in correl_frame.max().index:
    name_list = each.split("*")
    correl_max_frame[name_list[0]][name_list[1]] = correl_frame.max()[each]

"ab*cd".split("*")



each = correl_frame.columns[1]
results_frame = list()
for each in correl_frame.columns[:-2]:
    temp_data = correl_frame[[each, 'corr_mean']].dropna()
    X = temp_data['corr_mean']
    X = sm.add_constant(X)
    y = temp_data[each]
    model = sm.OLS(y, X)
    results = model.fit()
    #print type(results.params)
    #print type(results.pvalues)
    #print type(results.rsquared)
    results_list = list(results.params) + list(results.pvalues) + [results.rsquared]
    results_frame.append(results_list)
pd.DataFrame(np.array(results_frame),index=correl_frame.columns[:-2],columns=['c','b','p_c','p_b','r2']).to_excel(path+'corrmean_reg_nb.xlsx')
    #if (results.params[1] < 0) or (results.pvalues[1] >= 0.05):
        #print each
data_M.columns
column_list = ['Barclays_US_bond','Barclays_US_HY','SP500','MSCI_US_REITs','MSCI_emerging','BloomBerg_comodity','London_gold']
column_list_nb = ['Barclays_US_HY','SP500','MSCI_US_REITs','MSCI_emerging','BloomBerg_comodity','London_gold'] #nb=no bond

correl_frame = pd.DataFrame()
data_m = data_M[column_list]
for each_i in data_m.columns:
    for each_j in data_m.columns[list(data_m.columns).index(each_i)+1:]:
        temp_data = data_M[[each_i, each_j]].dropna()
        temp_corr = Rolling_Correlation(temp_data, 24)
        correl_frame = pd.merge(correl_frame, temp_corr, how='outer', left_index=True, right_index=True)
correl_frame.to_excel(path+"Selected_Assets_correlframe.xlsx")

correl_frame = pd.read_excel(path+"Selected_Assets_correlframe.xlsx")
correl_mean = correl_frame.mean(axis=1)
#correl_mean.plot()
#plt.show()

correl_mean = pd.DataFrame(correl_mean, columns=['corr_mean'])
correl_frame = pd.merge(correl_frame, correl_mean, how='outer', left_index=True, right_index=True)

results_frame = list()
for each in correl_frame.columns[:-2]:
    temp_data = correl_frame[[each, 'corr_mean']].dropna()
    X = temp_data['corr_mean']
    X = sm.add_constant(X)
    y = temp_data[each]
    model = sm.OLS(y, X)
    results = model.fit()
    #print type(results.params)
    #print type(results.pvalues)
    #print type(results.rsquared)
    results_list = list(results.params) + list(results.pvalues) + [results.rsquared]
    results_frame.append(results_list)
pd.DataFrame(np.array(results_frame),index=correl_frame.columns[:-2],columns=['c','b','p_c','p_b','r2']).to_excel(path+'sel_corrmean_reg_nb.xlsx')

panel_data = list()
for each in correl_frame.columns[:-2]:
    temp_panel = pd.DataFrame(columns=['y','x']+column_list)
    temp_data = correl_frame[[each, 'corr_mean']].dropna()
    temp_panel['y'] = temp_data[each]
    temp_panel['x'] = temp_data['corr_mean']
    for each_i in column_list:
        if each_i in each.split("*"):
            temp_panel[each_i] = 1
        else:
            temp_panel[each_i] = 0
    panel_data = panel_data + list(temp_panel.values)

panel_frame = pd.DataFrame(np.array(panel_data),columns=['y','x']+column_list)
X = panel_frame[['x']+column_list]
y = panel_frame['y']
model = sm.OLS(y, X)
results = model.fit()



correl_pca = PCA(correl_frame[correl_frame.columns[:-2]].dropna())
dir(correl_pca)
correl_pca.rsquare
correl_pca.factors['comp_00']
correl_pca = pd.DataFrame(correl_pca.factors['comp_00'].values, index=correl_frame.dropna().index, columns=['corr_pca'])
correl_frame = pd.merge(correl_frame, correl_pca, how='outer', left_index=True, right_index=True)
correl_frame.corr()['corr_pca']

results_frame = list()
for each in correl_frame.columns[:-3]:
    temp_data = correl_frame[[each, 'corr_pca']].dropna()
    X = temp_data['corr_pca']
    X = sm.add_constant(X)
    y = temp_data[each]
    model = sm.OLS(y, X)
    results = model.fit()
    #print type(results.params)
    #print type(results.pvalues)
    #print type(results.rsquared)
    results_list = list(results.params) + list(results.pvalues) + [results.rsquared]
    results_frame.append(results_list)
pd.DataFrame(np.array(results_frame),index=correl_frame.columns[:-3],columns=['c','b','p_c','p_b','r2']).to_excel(path+'sel_corrpca_reg_nb.xlsx')

panel_data = list()
for each in correl_frame.columns[:-2]:
    temp_panel = pd.DataFrame(columns=['y','x']+column_list)
    temp_data = correl_frame[[each, 'corr_pca']].dropna()
    temp_panel['y'] = temp_data[each]
    temp_panel['x'] = temp_data['corr_pca']
    for each_i in column_list:
        if each_i in each.split("*"):
            temp_panel[each_i] = 1
        else:
            temp_panel[each_i] = 0
    panel_data = panel_data + list(temp_panel.values)

panel_frame = pd.DataFrame(np.array(panel_data),columns=['y','x']+column_list)
X = panel_frame[['x']+column_list]
y = panel_frame['y']
model = sm.OLS(y, X)
results = model.fit()
results.params
results.pvalues
results.rsquared

column_list_nb = ['Barclays_US_HY','SP500','MSCI_US_REITs','MSCI_emerging','BloomBerg_comodity','London_gold']
correl_frame = pd.DataFrame()
data_m_nb = data_M[column_list_nb]
for each_i in data_m_nb.columns:
    for each_j in data_m_nb.columns[list(data_m_nb.columns).index(each_i)+1:]:
        temp_data = data_m_nb[[each_i, each_j]].dropna()
        temp_corr = Rolling_Correlation(temp_data, 24)
        correl_frame = pd.merge(correl_frame, temp_corr, how='outer', left_index=True, right_index=True)
correl_frame.to_excel(path+"Selected_Assets_correlframe_nb.xlsx")

correl_mean = correl_frame.mean(axis=1)
#correl_mean.plot()
#plt.show()

correl_frame = pd.read_excel(path+"Selected_Assets_correlframe_nb.xlsx")
correl_mean = pd.DataFrame(correl_mean, columns=['corr_mean'])
correl_frame = pd.merge(correl_frame, correl_mean, how='outer', left_index=True, right_index=True)

results_frame = list()
for each in correl_frame.columns[:-2]:
    temp_data = correl_frame[[each, 'corr_mean']].dropna()
    X = temp_data['corr_mean']
    X = sm.add_constant(X)
    y = temp_data[each]
    model = sm.OLS(y, X)
    results = model.fit()
    #print type(results.params)
    #print type(results.pvalues)
    #print type(results.rsquared)
    results_list = list(results.params) + list(results.pvalues) + [results.rsquared]
    results_frame.append(results_list)
pd.DataFrame(np.array(results_frame),index=correl_frame.columns[:-2],columns=['c','b','p_c','p_b','r2']).to_excel(path+'corrmean_reg_nb.xlsx')

panel_data = list()
for each in correl_frame.columns[:-2]:
    temp_panel = pd.DataFrame(columns=['y','x']+column_list_nb)
    temp_data = correl_frame[[each, 'corr_mean']].dropna()
    temp_panel['y'] = temp_data[each]
    temp_panel['x'] = temp_data['corr_mean']
    for each_i in column_list_nb:
        if each_i in each.split("*"):
            temp_panel[each_i] = 1
        else:
            temp_panel[each_i] = 0
    panel_data = panel_data + list(temp_panel.values)

panel_frame = pd.DataFrame(np.array(panel_data),columns=['y','x']+column_list_nb)
X = panel_frame[['x']+column_list_nb]
y = panel_frame['y']
model = sm.OLS(y, X)
results = model.fit()
results.params
results.pvalues
results.rsquared


correl_pca = PCA(correl_frame[correl_frame.columns[:-2]].dropna())
dir(correl_pca)
correl_pca.rsquare
correl_pca.factors['comp_00']
correl_pca = pd.DataFrame(correl_pca.factors['comp_00'].values, index=correl_frame.dropna().index, columns=['corr_pca'])
correl_frame = pd.merge(correl_frame, correl_pca, how='outer', left_index=True, right_index=True)
correl_frame.corr()['corr_pca']

results_frame = list()
for each in correl_frame.columns[:-3]:
    temp_data = correl_frame[[each, 'corr_pca']].dropna()
    X = temp_data['corr_pca']
    X = sm.add_constant(X)
    y = temp_data[each]
    model = sm.OLS(y, X)
    results = model.fit()
    #print type(results.params)
    #print type(results.pvalues)
    #print type(results.rsquared)
    results_list = list(results.params) + list(results.pvalues) + [results.rsquared]
    results_frame.append(results_list)
pd.DataFrame(np.array(results_frame),index=correl_frame.columns[:-3],columns=['c','b','p_c','p_b','r2']).to_excel(path+'corrpca_reg_nb.xlsx')

panel_data = list()
for each in correl_frame.columns[:-2]:
    temp_panel = pd.DataFrame(columns=['y','x']+column_list_nb)
    temp_data = correl_frame[[each, 'corr_pca']].dropna()
    temp_panel['y'] = temp_data[each]
    temp_panel['x'] = temp_data['corr_pca']
    for each_i in column_list_nb:
        if each_i in each.split("*"):
            temp_panel[each_i] = 1
        else:
            temp_panel[each_i] = 0
    panel_data = panel_data + list(temp_panel.values)

panel_frame = pd.DataFrame(np.array(panel_data),columns=['y','x']+column_list_nb)
X = panel_frame[['x']+column_list_nb]
y = panel_frame['y']
model = sm.OLS(y, X)
results = model.fit()
results.params
results.pvalues
results.rsquared

def Global_Correlation(data_frame):
    cov_frame = data_frame.cov()
    cov_matrix = np.matrix(cov_frame)
    sum_var = np.sum(cov_matrix.diagonal())
    sum_cov = np.sum(cov_matrix) - sum_var
    n = len(cov_frame)
    global_correlation = (sum_cov/(n*(n-1)))/(sum_var/n)
    return global_correlation

def Corr_Mean(data_frame):
    corr_frame = data_frame.corr()
    corr_matrix = np.matrix(corr_frame)
    sum_corr = np.sum(corr_matrix)
    n = len(corr_frame)
    corr_mean = (sum_corr-n)/(n*(n-1))
    return corr_mean


data_D = data.pct_change()
data_D.to_excel(path+"All_Assets_D.xlsx")

data_D = pd.read_excel(path+"All_Assets_D.xlsx")
column_list = ['Barclays_US_bond','SP500','MSCI_US_REITs','MSCI_emerging','BloomBerg_comodity','London_gold']
column_list_nb = ['SP500','MSCI_US_REITs','MSCI_emerging','BloomBerg_comodity','London_gold']
data_d = data_D[column_list].dropna()
data_d = data_D[column_list_nb].dropna()
data_d.to_excel(path+"Selected_Assets_D.xlsx")

gc_list = list()
returns_list = list()
b = 0.13
for i in range(50,len(data_d)):
    temp_data = data_d.iloc[(i-49):(i-1),]
    weights = np.array([1.0/len(temp_data.columns)]*len(temp_data.columns))
    #gc = Global_Correlation(temp_data)
    gc = Corr_Mean(temp_data)
    orgin_return = np.sum(data_d.iloc[i,] * weights)
    if gc > 0:
        multiplier = min(0.13/gc,2)
    else:
        multiplier = 2
    new_return = multiplier*orgin_return
    returns_list.append([orgin_return, new_return, multiplier])
    gc_list.append(gc)

pd.Series(gc_list).plot()
plt.show()
np.mean(gc_list)

(pd.DataFrame(np.array(returns_list))+1).product()
min(0.2,1)
pd.DataFrame(np.array(returns_list)).to_clipboard()
pd.DataFrame(np.array(returns_list)).cov()
'''