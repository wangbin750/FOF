# -*- coding: utf-8 -*-
"""
Created on Sun Jan 08 23:29:58 2017

@author: Kitty
"""

#coding=utf-8
import pandas as pd
import numpy as np
from datetime import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from pylab import *
from WindPy import *
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import r2_score

w.start()


def Get_list(ticker_frame):
    output_list=""
    count=0
    for i in ticker_frame['wind_code'].tolist():
        if count<len(ticker_frame['wind_code'])-1:
            output_list=output_list+i+","
        else:
            output_list=output_list+i
        count=count+1
    return output_list


def Get_first_tradeday(ticker_list,ticker_frame,firstday,lastday,fre):
    data_fetched=w.wsd(ticker_list,'close',firstday,lastday,"unit=1;Period=%s"%(fre))
    first_tradeday =pd.DataFrame(index=ticker_frame['wind_code'],columns=['first_tradeday'])
    for z in range(len(ticker_frame)):
        # z=0
        temp=pd.DataFrame(data_fetched.Data[z],index=data_fetched.Times,columns=['close'])
        # 取得第一个交易日
        trade_data=temp[temp['close'].isnull()==False] 
        if len(trade_data)==0:
            first_tradeday.at[ticker_frame['wind_code'].loc[z],'first_tradeday']=None
        else:
            first_tradeday.at[ticker_frame['wind_code'].loc[z],'first_tradeday']=trade_data.index[0]
    print ('First tradeday of each ETF Got!')
    return first_tradeday.sort(columns='first_tradeday').dropna()
        

# datatype="close","pct_chg","volume"
def data_fetch(ticker_list,ticker_frame,datatype,n,firstday,lastday,fre):
    data_fetched=w.wsd(ticker_list, datatype,firstday,lastday,"unit=1;Period=%s"%(fre))
    data =pd.DataFrame(index=data_fetched.Times)
    for z in range(len(ticker_frame)):
        print z
        # z=0
        temp=pd.DataFrame(data_fetched.Data[z],index=data_fetched.Times,columns=[datatype])
        # 取得开始公开交易以后的数据
        trade_data=temp[temp[datatype].isnull()==False]
        # 次日期后交易日 没有数据的比例在可接受范围内的ETF ticker
        if len(trade_data)!=0:
            if len(trade_data[trade_data[datatype].isnull()==True])<=len(trade_data)*n:
                #print len(data_fetched.Data[z])
                #print ticker_frame['wind_code'].iloc[z]
                data = pd.merge(data, pd.DataFrame(data_fetched.Data[z],columns=[ticker_frame['wind_code'].iloc[z]],index=data_fetched.Times), left_index=True, right_index=True, how='left')
    print ('%s Got!'%(datatype))
    return data
    
    
# 筛选有效数据
def Sieve(temp,n2):
    input_data=pd.DataFrame(index=temp.index)
    for z in temp.columns:
        if temp[z].sum()!=0 and len(temp[z][temp[z].fillna(0)!=0])>len(temp[z])*n2:
            input_data[z]=temp[z]
    return input_data

# 计算基金组合平均表现
def Cal_performance(return_data,ticker_list):
    funds_return=return_data[ticker_list].mean().mean()-((1+0.1)**(1/12)-1)
    funds_volatility=return_data[ticker_list].std().mean()
    funds_sharpe=((return_data[ticker_list].mean()-0.001-((1+0.1)**(1/12)-1))/return_data[ticker_list].std()).mean()
    funds_skew=return_data[ticker_list].skew().mean()
    funds_kurt=return_data[ticker_list].kurt().mean()
    return [funds_return,funds_volatility,funds_sharpe,funds_skew,funds_kurt]
    
# 得到可复制基金组合与不可复制基金组合
def PCA_Replication_performance(funds_data,data,rolling_window,n2,ratio):
    # 储存滚动时间窗内复制的最大R2和主成分数
    regression_r2=pd.DataFrame(index=data.index[rolling_window:len(data)])
    regression_factor_number=pd.DataFrame(index=data.index[rolling_window:len(data)])
    # 储存可复制基金和不可复制的基金ticker
    funds_group=pd.DataFrame(index=data.index[rolling_window:len(data)],columns=['replicable_funds','un_replicable_funds'])
    # 存储可复制基金与不可复制基金的平均收益率，波动率，夏普比率等
    replicable_funds_performance=pd.DataFrame(index=data.index[rolling_window:len(data)],columns=['return','volatility','sharpe','skewness','kurtosis'])
    un_replicable_funds_performance=pd.DataFrame(index=data.index[rolling_window:len(data)],columns=['return','volatility','sharpe','skewness','kurtosis'])
    for i in range(len(data)-rolling_window):
        # i=0
        temp_ETF=data.loc[data.index[i:i+rolling_window]]
        # 被复制的基金
        temp_funds=funds_data.loc[data.index[i:i+rolling_window]]
       
        # 筛选ETF因子/被复制的基金数据
        input_data=Sieve(temp_ETF,n2)
        input_funds=Sieve(temp_funds,n2)
        
        if input_data.sum().sum()==0:
            continue
        # 主成分个数从1-全部ETF因子循环，然后复制，选择拟合度最高的主成分个数    
        r2_factors=pd.DataFrame(index=range(1,len(input_data.columns)),columns=input_funds.columns)
        for m in range(1,len(input_data.columns)):
            # m=6
            pca=PCA(n_components=m)
            pca.fit(np.asmatrix(input_data.fillna(0)))
            # 用以训练好的数据进行主成分分析，得到主成分因子
            newdata=pd.DataFrame(pca.transform(np.asmatrix(input_data.fillna(0))))
            # 对每一只基金进行线性复制，选取r2最大的因子数
            for j in input_funds.columns:
                y=input_funds[j]    
                est=sm.OLS(y.fillna(0).values,newdata.values)
                est=est.fit()
                r2_factors.at[m,j]=est.rsquared
            print ("%s/%s PCA pass..."%(m,len(input_data.columns)-1))

        if r2_factors.sum().sum()==0:
            continue        
        # 储存最佳复制的因子数及相应的R2        
        for j in r2_factors.columns:
            # j=r2_factors.columns[0]
            a=r2_factors[j][r2_factors[j]==r2_factors[j].max()]
            opt_factors=a.index[0]
            regression_r2.at[data.index[i+rolling_window],j]=r2_factors[j].max()
            regression_factor_number.at[data.index[i+rolling_window],j]=opt_factors
        # 按照R2排序分为可复制的基金与不可复制的基金
        temp_r2=pd.DataFrame(regression_r2.loc[data.index[i+rolling_window]].values,index=regression_r2.loc[data.index[i+rolling_window]].index,columns=['r2'])
        sort_r2=temp_r2.sort(columns=['r2'])
        # save可复制和不可复制基金的ticker    
        funds_group.at[data.index[i+rolling_window],'replicable_funds']=sort_r2.index[-ratio*len(sort_r2):len(sort_r2)].tolist()
        funds_group.at[data.index[i+rolling_window],'un_replicable_funds']=sort_r2.index[0:ratio*len(sort_r2)].tolist()
        # save可复制基金和不可复制基金的performance
        replicable_funds_performance.loc[data.index[i+rolling_window]]=Cal_performance(temp_funds,funds_group.at[data.index[i+rolling_window],'replicable_funds'])
        un_replicable_funds_performance.loc[data.index[i+rolling_window]]=Cal_performance(temp_funds,funds_group.at[data.index[i+rolling_window],'un_replicable_funds'])
    
        print ("Replicable and Un-replicable funds performance for %s Got!"%(data.index[i+rolling_window]))
        
    r=replicable_funds_performance.mean()
    un_r=un_replicable_funds_performance.mean()
    return r,un_r,funds_group,regression_r2,regression_factor_number,replicable_funds_performance,un_replicable_funds_performance
    
#######################################################################################  
# PCA     

#零均值化  
def zeroMean(dataMat):        
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值  
    newData=dataMat-meanVal  
    return newData,meanVal  
  
def pca1(dataMat,n):  
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本  
      
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量  
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序  
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标  
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量  
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据  
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据  
    return pd.DataFrame(lowDDataMat),pd.DataFrame(n_eigVect.T),pd.DataFrame(reconMat)  
'''
newdata,weight,reconstructed_data=pca1(np.asmatrix(input_data.fillna(0)),6)

X = np.asmatrix(input_data.fillna(0))
pca.transform(X)
pca.inverse_transform(X)
pca.explained_variance_ratio_
pca.get_params

pd.DataFrame(pca.transform(X))
pd.DataFrame(n_eigVect)    
'''   
    
################################################################################
# 交叉验证选择惩罚系数，lasso回归选择因子，处理了因子共线性问题
# positive：True：约束系数都为正
# intercept=False，因为x以全部中心化

def Lasso_Replication_performance(funds_data,data,rolling_window,n2,ratio,positive):
    # 储存滚动时间窗内复制的最大R2和主成分数
    regression_r2=pd.DataFrame(index=data.index[rolling_window:len(data)])
    # 储存可复制基金和不可复制的基金ticker
    funds_group=pd.DataFrame(index=data.index[rolling_window:len(data)],columns=['replicable_funds','un_replicable_funds'])
    # 存储可复制基金与不可复制基金的平均收益率，波动率，夏普比率等
    replicable_funds_performance=pd.DataFrame(index=data.index[rolling_window:len(data)],columns=['return','volatility','sharpe','skewness','kurtosis'])
    un_replicable_funds_performance=pd.DataFrame(index=data.index[rolling_window:len(data)],columns=['return','volatility','sharpe','skewness','kurtosis'])

    for i in range(len(data)-rolling_window):
        # i=0
        temp_ETF=data.loc[data.index[i:i+rolling_window]]
        # 被复制的基金
        temp_funds=funds_data.loc[data.index[i:i+rolling_window]]
        # 筛选ETF因子/被复制的基金数据
        input_data=Sieve(temp_ETF,n2)
        input_funds=Sieve(temp_funds,n2)
        if input_data.sum().sum()==0:
            continue
        # 初始化一个 lasso cross-validation的regression
        if positive==True:
            lasso = linear_model.LassoLars(fit_intercept=False,positive=True)
        else:
            lasso = linear_model.LassoCV(fit_intercept=False)
        for j in input_funds.columns: 
            x=pd.DataFrame(zeroMean(np.asmatrix(input_data.fillna(0)))[0])
            y=input_funds[j].fillna(0)
            lasso.fit(x,y)
            y_pred_lasso = lasso.fit(x, y).predict(x) 
            
            #lasso.coef_
            r2_score_lasso = r2_score(y.values, y_pred_lasso)
            regression_r2.at[data.index[i+rolling_window],j]=r2_score_lasso
            
        # 按照R2排序分为可复制的基金与不可复制的基金
        temp_r2=pd.DataFrame(regression_r2.loc[data.index[i+rolling_window]].values,index=regression_r2.loc[data.index[i+rolling_window]].index,columns=['r2'])
        sort_r2=temp_r2.sort(columns=['r2'])
        # save可复制和不可复制基金的ticker    
        funds_group.at[data.index[i+rolling_window],'replicable_funds']=sort_r2.index[-ratio*len(sort_r2):len(sort_r2)].tolist()
        funds_group.at[data.index[i+rolling_window],'un_replicable_funds']=sort_r2.index[0:ratio*len(sort_r2)].tolist()
        # save可复制基金和不可复制基金的performance
        replicable_funds_performance.loc[data.index[i+rolling_window]]=Cal_performance(temp_funds,funds_group.at[data.index[i+rolling_window],'replicable_funds'])
        un_replicable_funds_performance.loc[data.index[i+rolling_window]]=Cal_performance(temp_funds,funds_group.at[data.index[i+rolling_window],'un_replicable_funds'])

        print data.index[i+rolling_window]
    r=replicable_funds_performance.mean()
    un_r=un_replicable_funds_performance.mean()        
    return r,un_r,funds_group,regression_r2,replicable_funds_performance,un_replicable_funds_performance
        

##################################################################################
# Multi-task lasso
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV

def MultiTaskLasso_Replication_performance(funds_data,data,rolling_window,n2,ratio):
    # 储存滚动时间窗内复制的最大R2和主成分数
    regression_r2=pd.DataFrame(index=data.index[rolling_window:len(data)])
    # 储存可复制基金和不可复制的基金ticker
    funds_group=pd.DataFrame(index=data.index[rolling_window:len(data)],columns=['replicable_funds','un_replicable_funds'])
    # 存储可复制基金与不可复制基金的平均收益率，波动率，夏普比率等
    replicable_funds_performance=pd.DataFrame(index=data.index[rolling_window:len(data)],columns=['return','volatility','sharpe','skewness','kurtosis'])
    un_replicable_funds_performance=pd.DataFrame(index=data.index[rolling_window:len(data)],columns=['return','volatility','sharpe','skewness','kurtosis'])

    for i in range(len(data)-rolling_window):
        # i=0
        temp_ETF=data.loc[data.index[i:i+rolling_window]]
        # 被复制的基金
        temp_funds=funds_data.loc[data.index[i:i+rolling_window]]
        # 筛选ETF因子/被复制的基金数据
        input_data=Sieve(temp_ETF,n2)
        input_funds=Sieve(temp_funds,n2)
        if input_data.sum().sum()==0:
            continue

        X = np.asmatrix(input_data.fillna(0))
        Y = np.asmatrix(input_funds.fillna(0))

        #coef_multi_task_lasso_ = MultiTaskLassoCV().fit(X, Y).coef_  
        y_pred_lasso = MultiTaskLasso().fit(X,Y).predict(X) 
           
        for j in range(len(input_funds.columns)):
            r2_score_lasso = r2_score(input_funds.fillna(0)[input_funds.columns[j]], pd.DataFrame(y_pred_lasso)[j])
            regression_r2.at[data.index[i+rolling_window],j]=r2_score_lasso
            
        # 按照R2排序分为可复制的基金与不可复制的基金
        temp_r2=pd.DataFrame(regression_r2.loc[data.index[i+rolling_window]].dropna(0).values,index=input_funds.columns,columns=['r2'])
        sort_r2=temp_r2.sort(columns=['r2'])
        # save可复制和不可复制基金的ticker    
        funds_group.at[data.index[i+rolling_window],'replicable_funds']=sort_r2.index[-ratio*len(sort_r2):len(sort_r2)].tolist()
        funds_group.at[data.index[i+rolling_window],'un_replicable_funds']=sort_r2.index[0:ratio*len(sort_r2)].tolist()
        # save可复制基金和不可复制基金的performance
        replicable_funds_performance.loc[data.index[i+rolling_window]]=Cal_performance(temp_funds,funds_group.at[data.index[i+rolling_window],'replicable_funds'])
        un_replicable_funds_performance.loc[data.index[i+rolling_window]]=Cal_performance(temp_funds,funds_group.at[data.index[i+rolling_window],'un_replicable_funds'])

        print data.index[i+rolling_window]
    r=replicable_funds_performance.mean()
    un_r=un_replicable_funds_performance.mean()        
    return r,un_r,funds_group,regression_r2,replicable_funds_performance,un_replicable_funds_performance
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    