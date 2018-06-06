#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:51:44 2017

@author: Rachel
"""

import os
os.chdir('/Users/Rachel/Dropbox/STA237 Project')
#%%
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import sqlite3 as db
import requests
import requests_cache
requests_cache.install_cache('cache')
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.tools.tools as st
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
#%%
def pair_price_data(pair, start, end):
    """
    Return the price of a pair of assets.
    """
    asset1, asset2 = pair
    
    cnx = db.connect('/Users/Rachel/Dropbox/STA237 Project/tickers_database.db')
    cur = cnx.cursor()

    sql1 = 'SELECT Date, "Adj Close" FROM Pricetable WHERE Ticker = "' + asset1 + '" AND Date >= "' + start + '" AND Date <= "' + end + '"'
    df1 = pd.read_sql(sql1, con=cnx).set_index('Date')
    df1.columns = [asset1]
    
    sql2 = 'SELECT Date, "Adj Close" FROM Pricetable WHERE Ticker = "' + asset2 + '" AND Date >= "' + start + '" AND Date <= "' + end + '"'
    df2 = pd.read_sql(sql2, con=cnx).set_index('Date')
    df2.columns = [asset2]
    
    df = pd.concat([df1, df2], axis = 1, join = 'inner')
    
    return df

def ADF(series, threshold, print_conclusion = False):
    adf = ts.adfuller(series)

    if print_conclusion:
        print('Augmented Dickey Fuller test statistic =', adf[0])
        print('Augmented Dickey Fuller p-value =', adf[1])
            
        if adf[1] > threshold:
            print('The time series is non-stationary.')
        else:
            print('The time series is stationary.')
    
    return adf[1]


def regression(y, x):
    regr = OLS(np.log(y), np.log(x)).fit() # log price 
    print('The hedge ratio is %f.' % regr.params.item())
    spread = regr.resid
    return regr.params.item(), spread

def arma_model(ts):
    aic_select = arma_order_select_ic(ts, max_ar = 4, max_ma = 4, ic = 'aic')
    result = ARMA(ts, order = aic_select.aic_min_order).fit(method = 'css')
    mu = result.params[0]/(1-result.arparams.sum())
    sigma = np.sqrt(result.sigma2)
    return mu, sigma

def BolBands(spread_t, mu, sigma, z_entry, z_exit):
    upper = mu + z_entry * sigma
    lower = mu - z_exit * sigma
    return upper, lower  

def PairsTrading(pair, test_start, test_end, method, z_entry, z_exit):
    
    data_test = pair_price_data(pair, test_start, test_end)
    train_ends = data_test.index.values.tolist()
    train_starts = [datetime.strftime(datetime.strptime(te, '%Y-%m-%d') + relativedelta(months = -12), '%Y-%m-%d') for te in train_ends]
    def Main(pair, start, end): 
        asset1, asset2 = pair
        print('Trade date is %s' % end)
        print('Obtain price data') 
        df = pair_price_data(pair, start, end)
        print('Obtain hedge ratio and spread') 
        h_r, spread = regression(df[asset1], df[asset2]) 
        print('Check stationary of the spread')
        adf_stat = ADF(spread, 0.05, print_conclusion = True)
        print('Build model of the spread')
        mu, sigma = method(spread)
        print('mu = %f, sigma = %f' % (mu, sigma))
        s_t = spread[spread.index == end].item()
        print('Current spread is %f' % s_t)
        print('Calculate Bollinger bands')
        upper_band, lower_band = BolBands(s_t, mu, sigma, z_entry, z_exit)
        return s_t, upper_band, lower_band
    
    ss = []
    us = []
    ls = []
    for ts, te in zip(train_starts, train_ends):
        s, u, l = Main(pair, ts, te)
        ss.append(s)
        us.append(u)
        ls.append(l)
        
    data_test['spread'] = ss
    data_test['upper band'] = us
    data_test['lower band'] = ls
        
    return data_test
#%%
def test(pair, test_start, test_end, method, z_entry, z_exit):
    
    data_test = pair_price_data(pair, test_start, test_end)
    train_ends = data_test.index.values.tolist()
    train_starts = [datetime.strftime(datetime.strptime(te, '%Y-%m-%d') + relativedelta(months = -12), '%Y-%m-%d') for te in train_ends]
    def Main(pair, start, end): 
        asset1, asset2 = pair
        print('Trade date is %s' % end)
        print('Obtain price data') 
        df = pair_price_data(pair, start, end)
        print('Obtain hedge ratio and spread') 
        h_r, spread = regression(df[asset1], df[asset2]) 
        print('Check stationary of the spread')
        adf_stat = ADF(spread, 0.05, print_conclusion = True)
        
    for ts, te in zip(train_starts, train_ends):
        Main(pair, ts, te)
#%%
test(['AMLP', 'USO'], '2016-11-25', '2016-12-25', arma_model, 1.5, 0.5)
#%%
pair_price_data(['AMLP', 'USO'], '2015-11-25', '2017-11-25')
#%% 
def rolling(ts, window_size, threshold = None, select_optimal_size = False):    
    n = len(ts)
    m = int(np.floor(window_size * n))
    adfs = pd.DataFrame([ADF(ts[i:(m+i)], .05) for i in range(n-m)], columns = ['ADF_pvalue'])
        
    if select_optimal_size:      
        max_pvalue = adfs.max().item()
        return max_pvalue
    else:
        pass_ratio = (adfs['ADF_pvalue'] < threshold).sum().item()/((1-window_size)*n)
        return pass_ratio
    
def select_window_size(ts):
    
    if type(ts) != np.ndarray:
        ts = np.array(ts)

    window_sizes = [.4, .5, .6, .7, .8]
    pvs = [[w_s, rolling(ts, w_s, select_optimal_size = True)] for w_s in window_sizes]
    df = pd.DataFrame(pvs, columns = ['window_size', 'max_p_value']).set_index('window_size')
    # optimal window size
    opt_m = df.idxmax().item()
    print('The optimal window size is %f' % opt_m)
    return opt_m
    
def Rolling_ADF(ts, window_size, threshold):
    if type(ts) != np.ndarray:
        ts = np.array(ts)
            
    # rolling ADF test with optimal window size
    r = rolling(ts, window_size, threshold = .1)
    print('The ratio of successful ADF tests is %f' % r)  
    return r
#%%
def Recursive_ADF(ts, threshold):
    if type(ts) != np.ndarray:
        ts = np.array(ts)
    
    n = len(ts)
    m0 = int(np.floor(.3 * n))
    df = pd.DataFrame([ADF(ts[:m0+i], .05) for i in range(n-m0)], columns = ['p_value'])
    r = (df['p_value'] < threshold).sum().item()/(n-m0)   
    print('The ratio is %f' % r)
    return r
#%%
def CIG(pair, start, end, method, threshold, window_size = None):
    """
    Input:
        method: 'rolling adf' or 'recursive adf'
    """
    print(pair)
    # get price series
    asset1, asset2 = pair
    p = pair_price_data(pair, start, end).dropna()
    
    if len(p) > 50:
        # check stationary for each series
        _ = ADF(p[asset1], threshold)
        _ = ADF(p[asset2], threshold)
        # run OLS regression on two prices series and get spread
        h_r, spread = regression(p[asset1], p[asset2])
        # check stationary of the spread
        if method == 'rolling adf':
            r = Rolling_ADF(spread, window_size, threshold = threshold)
        if method == 'recursive adf':
            r = Recursive_ADF(spread, threshold)
        #if p_v > threshold:
        #    print('The two series, %s and %s, are not co-integrated' % (asset1, asset2))
        #else:
        #    print('The two series, %s and %s, are co-integrated' % (asset1, asset2))
        #   return pair
        return asset1, asset2, r
    else:
        print('Not Found Data')
#%%
start_date, end_date = '2015-11-25', '2016-11-25'
CIG(['AMLP', 'USO'], start_date, end_date, 'recursive adf', .1)
#%%
cnx = db.connect('/Users/Rachel/Dropbox/STA237 Project/tickers_database.db')
cur = cnx.cursor()

sql1 = 'SELECT Ticker FROM ETFtable WHERE Focus = "Crude Oil" AND "Asset Class" = "Commodities"'
cmd = pd.read_sql(sql1, con=cnx)['Ticker'].values.tolist()

sql2 = 'SELECT Ticker FROM ETFtable WHERE Focus = "Energy" AND "Asset Class" = "Equity"'
eqt = pd.read_sql(sql2, con=cnx)['Ticker'].values.tolist()

cnx.close()

ticker_pairs = []
for e in eqt:
    for c in cmd:
        pair = [e, c]
        ticker_pairs.append(pair)
#%%
start_date, end_date = '2015-11-25', '2016-11-25'
result = [CIG(t_p, start_date, end_date, 'rolling adf', threshold = .4, window_size = .1) for t_p in ticker_pairs]
df = pd.DataFrame([t for t in result if t is not None ], columns = ['Asset1', 'Asset2', 'Ratio'])
df = df.sort_values('Ratio', ascending = False).reset_index(drop = True)
#%%
result = [CIG(t_p, start_date, end_date, 'recursive adf', threshold = .1) for t_p in ticker_pairs]
df = pd.DataFrame([t for t in result if t is not None ], columns = ['Asset1', 'Asset2', 'Ratio'])
df = df.sort_values('Ratio', ascending = False).reset_index(drop = True)
#%%
def OPT_ws(pair, start, end):
    print(pair)
    # get price series
    asset1, asset2 = pair
    p = pair_price_data(pair, start, end).dropna()
    
    if len(p) > 50:
        # run OLS regression on two prices series and get spread
        h_r, spread = regression(p[asset1], p[asset2])
        # check stationary of the spread
        w_s = select_window_size(spread)       
        return w_s
    else:
        print('Not Found Data')
start_date, end_date = '2015-11-25', '2016-11-25'        
result = [OPT_ws(t_p, start_date, end_date) for t_p in ticker_pairs]
sizes = [p for p in result if p is not None]
#%%
