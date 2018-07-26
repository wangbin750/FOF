# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pyper as pr
from Allocation_Method import Risk_Parity_Weight, Max_Utility_Weight_new
import socket

hostname = socket.gethostname()
if hostname == "DESKTOP-OGC5NH7":
    path = u"E:/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "localhost":
    path = "/Users/WangBin-Mac/263网盘/FOF相关程序/Global Allocation/"
elif hostname == "CICCB6CR7213VFT":
    path = "F:/GitHub/FOF/Global Allocation/"

asset_col_all = ["Barclays_US_Treasury","Barclays_US_HY","SP500","MSCI_exUS","MSCI_emerging","London_gold","MSCI_US_REITs","FTSE_global_REITs"]
asset_col = ["Barclays_US_HY","SP500","MSCI_exUS","MSCI_emerging","London_gold","MSCI_US_REITs","FTSE_global_REITs"]

def multi_str(str_list):
    str_out = []
    for i in str_list:
        for j in str_list:
            temp = i + "*" + j
            str_out.append(temp)
    return str_out

asset_col_rho = multi_str(asset_col)
r = pr.R(use_pandas=True)
r("library(MTS)")

Return_frame_ori = pd.read_excel(path+"DCC_Assets_M+REITs.xlsx")
Return_frame = pd.DataFrame(Return_frame_ori[asset_col].values, columns=asset_col)
Return_frame_all = Return_frame_ori[asset_col_all]

for delay in [1, -5, -11, -23]:

    r_gd_list = []
    r_g_list = []
    r_list = []
    m_gd_list = []
    m_g_list = []
    m_list = []
    w_list = []

    for each in range(89, len(Return_frame_all)-1):
        return_frame = Return_frame[Return_frame.index[each-89]:Return_frame.index[each]]
        return_frame_all = Return_frame_all[Return_frame_all.index[each-89]:Return_frame_all.index[each]]
        
        r.assign("rframe", return_frame)
        r('''
        rm1 <- dccPre(rframe, include.mean=T, p=0, cond.dist="norm")
        rtn1 <- rm1$sresi
        rmarVol <- rm1$marVol
        rm1_est <- rm1$est
        rm2 <- dccFit(rtn1, type="TseTsui", cond.dist="std", df=7, m=6)
        rm2_est <- rm2$estimates
        rm2_rho = rm2$rho.t
        ''')
        marVol = r.get("rmarVol")
        m1_est = r.get("rm1_est")
        m2_est = r.get("rm2_est")
        m2_rho = r.get("rm2_rho")

        std_frame = pd.DataFrame(marVol, columns=asset_col)
        rho_frame = pd.DataFrame(m2_rho, columns=asset_col_rho)
        garch_est = pd.DataFrame(m1_est.T, columns=asset_col, index=['w', 'a', 'b'])
        dcc_est = m2_est

        std_list_garch = []
        for k in asset_col:
            temp_h = garch_est[k]['w'] + garch_est[k]['a']*list(return_frame[k])[-1]**2.0 + garch_est[k]['b']*list(std_frame[k])[-1]**2.0
            temp_std = temp_h**0.5
            std_list_garch.append(temp_std)

        rho_matrix_dcc = pd.DataFrame(index=asset_col, columns=asset_col)
        for i in asset_col:
            for j in asset_col:
                if i == j:
                    rho_matrix_dcc[i][j] = 1.0
                else:
                    rho_bar = return_frame[[i,j]].corr()[i][j]
                    rho_1 = list(rho_frame[i+"*"+j])[-1]
                    psi = return_frame[[i,j]][-8:].corr()[i][j]
                    temp_rho = (1.0-dcc_est[0]-dcc_est[1])*rho_bar + dcc_est[0]*rho_1 + dcc_est[1]*psi
                    rho_matrix_dcc[i][j] = temp_rho

        cov_matrix_garch_dcc = np.matrix(np.diag(std_list_garch)) * np.matrix(rho_matrix_dcc) * np.matrix(np.diag(std_list_garch))
        cov_matrix_garch = np.matrix(np.diag(std_list_garch)) * np.matrix(return_frame.corr()) * np.matrix(np.diag(std_list_garch))

        cov_matrix_all_garch_dcc = return_frame_all.cov()
        cov_matrix_all_garch_dcc.loc[asset_col, asset_col] = cov_matrix_garch_dcc

        cov_matrix_all_garch = return_frame_all.cov()
        cov_matrix_all_garch.loc[asset_col, asset_col] = cov_matrix_garch
        
        rw = Risk_Parity_Weight(return_frame_all.cov()).round(3)
        rw_gd = Risk_Parity_Weight(cov_matrix_all_garch_dcc).round(3)
        rw_g = Risk_Parity_Weight(cov_matrix_all_garch).round(3)
        
        '''
        rw = Max_Utility_Weight_new(pd.DataFrame(return_frame_all.mean()), return_frame_all.cov(), 5, [(0.0,None)]*len(asset_col_all)).round(3)
        rw_gd = Max_Utility_Weight_new(pd.DataFrame(return_frame_all.mean()), cov_matrix_all_garch_dcc, 5, [(0.0,None)]*len(asset_col_all)).round(3)
        rw_g = Max_Utility_Weight_new(pd.DataFrame(return_frame_all.mean()), cov_matrix_all_garch, 5, [(0.0,None)]*len(asset_col_all)).round(3)
        '''
        mw = Max_Utility_Weight_new(pd.DataFrame(return_frame_all.mean()), return_frame_all.cov(), 2, [(0.0,None)]*len(asset_col_all)).round(3)
        mw_gd = Max_Utility_Weight_new(pd.DataFrame(return_frame_all.mean()), cov_matrix_all_garch_dcc, 2, [(0.0,None)]*len(asset_col_all)).round(3)
        mw_g = Max_Utility_Weight_new(pd.DataFrame(return_frame_all.mean()), cov_matrix_all_garch, 2, [(0.0,None)]*len(asset_col_all)).round(3)

        rr = np.sum(rw*Return_frame_all.loc[Return_frame_all.index[each+delay]])
        rr_gd = np.sum(rw_gd*Return_frame_all.loc[Return_frame_all.index[each+delay]])
        rr_g = np.sum(rw_g*Return_frame_all.loc[Return_frame_all.index[each+delay]])
        mr = np.sum(mw*Return_frame_all.loc[Return_frame_all.index[each+delay]])
        mr_gd = np.sum(mw_gd*Return_frame_all.loc[Return_frame_all.index[each+delay]])
        mr_g = np.sum(mw_g*Return_frame_all.loc[Return_frame_all.index[each+delay]])

        r_list.append(rr)
        r_gd_list.append(rr_gd)
        r_g_list.append(rr_g)
        m_list.append(mr)
        m_gd_list.append(mr_gd)
        m_g_list.append(mr_g)
        w_list.append(list(rw)+list(rw_g)+list(rw_gd)+list(mw)+list(mw_g)+list(mw_gd))
        print each

    pd.DataFrame(np.array([r_list, r_g_list, r_gd_list, m_list, m_g_list, m_gd_list]).T).to_excel(path+"DCC_90_%s+REITs.xlsx"%(1-delay))
    pd.DataFrame(np.array(w_list)).to_excel(path+"DCCw_90_%s+REITs.xlsx"%(1-delay))


