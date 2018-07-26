# coding=utf-8
import codecs

from Portfolio_Formation import *

path = u'Z:\Mac 上的 WangBin-Mac\FOF\Robo-Advisor'

for each in range(1,11):
    portfolio_imformation = Portfolio_Form(each)
    full_path = path + "\portfolio_" + str(each) + '.txt'
    file = codecs.open(full_path, 'w', 'utf-8')
    file.write(portfolio_imformation)
    file.close()
    print('Done')

for each in range(1, 11):
    full_path = path + "\portfolio_" + str(each) + '.txt'
    file = codecs.open(full_path)
    print file.read()

full_path = '/Users/WangBin-Mac/FOF/Robo-Advisor/portfolio_1.txt'

import pandas as pd

def Performance(nv_series, rf_ret):
    end_value = nv_series[-1]
    return_series = nv_series.pct_change().dropna()
    annual_return = (return_series + 1).prod() ** (1/(len(return_series)/365.0)) - 1
    annual_variance = (return_series.var() * 365.0) ** 0.5
    sharpe_ratio = (annual_return - rf_ret)/annual_variance
    max_drawdown = max(((return_series + 1).cumprod().cummax()-(return_series + 1).cumprod())/(return_series + 1).cumprod().cummax())
    return [end_value, annual_return, annual_variance, sharpe_ratio, max_drawdown]

data = pd.read_excel("/Users/WangBin-Mac/Desktop/qyzt-8m.xlsx")

result_list = []
for each in data.columns:
    temp_series = data[each]
    temp_result = Performance(temp_series, 0.03)
    result_list.append(temp_result)

pd.DataFrame(np.array(result_list)).to_excel("/Users/WangBin-Mac/Desktop/qyzt-8m-result.xlsx")
