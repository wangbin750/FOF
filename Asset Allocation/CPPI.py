# coding=utf-8

import numpy as np
import pandas as pd

rf_ret_list = [0.0002]*242
r_ret_list = 0.0005 + np.random.randn(242)*0.01
(np.array(r_ret_list)+1).cumprod()

test = []
for i in range(200):
    r_ret_list = 0.0005 + np.random.randn(242)*0.01
    test.append((np.array(r_ret_list)+1).prod()-1)

def Cushion_Cal(nv, rf_ret, maturity, target_ret):
    final_target = 1.0 + target_ret
    safe_asset = final_target/((1+rf_ret)**maturity)
    if nv < safe_asset:
        return "out!"
    else:
        cushion = nv - safe_asset
        return cushion

target_ret = 0.00
nv = [1]
m_para = 2
for i in range(len(rf_ret_list)):
    maturity = len(rf_ret_list) - i
    cushion = Cushion_Cal(nv[i], rf_ret_list[i], maturity, target_ret)
    if cushion == "out!":
        r_weight = 0.0
        rf_weight = 1.0
    else:
        r_weight = m_para * cushion
        rf_weight = nv[i] - r_weight
    nv_new = r_weight*(1+r_ret_list[i]) + rf_weight*(1+rf_ret_list[i])
    nv.append(nv_new)

print nv[-1]

nv_list = []
for i in range(100):
    rf_ret_list = [0.0002]*242
    r_ret_list = 0.0005 + np.random.randn(242)*0.01
    target_ret = 0.00
    nv = [1]
    m_para = 5
    for i in range(len(rf_ret_list)):
        maturity = len(rf_ret_list) - i
        cushion = Cushion_Cal(nv[i], rf_ret_list[i], maturity, target_ret)
        if cushion == "out!":
            r_weight = 0.0
            rf_weight = 1.0
        else:
            r_weight = m_para * cushion
            rf_weight = nv[i] - r_weight
        nv_new = r_weight*(1+r_ret_list[i]) + rf_weight*(1+rf_ret_list[i])
        nv.append(nv_new)

    #print nv[-1]
    nv_list.append(nv[-1])
print np.array(nv_list).mean()
