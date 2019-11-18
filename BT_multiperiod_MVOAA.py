import timeit
start = timeit.default_timer()
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import Strategy,Portfolio
from OPT_cvxpy_optimisation import markowitz_portfolio

import extract_data_from_web as ext


class MVO(Strategy):

    def __init__(self,names,df,date_list):
        self.names = names
        self.df = df
        self.date_list = date_list

    def generate_signals(self):
        self.df = self.df.resample('BM').last()
        self.sig = pd.DataFrame(index = self.date_list)

        for index,x in enumerate(self.names):
            self.sig[x] = 1

        return self.sig


class MarketOnClosePortfolio(Portfolio):

    def __init__(self,df,names,sig,date_list,date_list_returns):
        self.df = df
        self.names = names
        self.sig = sig
        self.date_list = date_list
        self.date_list_returns = date_list_returns
        self.positions = self.generate_positions()

    def generate_positions(self):
        pos = pd.DataFrame(index=self.sig.index)
        weights = self.generate_MVO_positions()
        pos = weights.mul(self.sig.values)
        return pos

    def generate_MVO_positions(self):
        df_m = self.df.resample('BM').last()

        import datetime
        from dateutil.relativedelta import relativedelta
        riskaversion = 1.8
        constype = 3

        weights_all = []
        # cov
        for dts in self.date_list:
            dts = dts + relativedelta(months=-1)
            lb_startdate = dts + relativedelta(months=-6)
            covariances = 252 * self.df[lb_startdate:dts].pct_change(1).dropna().cov().values
            exp_rtns = 252 * self.df[lb_startdate:dts].pct_change(1).dropna().mean()
            weights, _, _  = markowitz_portfolio(exp_rtns, covariances, riskaversion, constype)
            weights_all.append(weights)

        df = pd.DataFrame(weights_all, index=date_list)
        return df

    def backtest_portfolio(self):
        self.df = self.df.resample('BM').last()
        self.df = self.df.reindex(self.date_list_returns)
        pct_chg = self.df.pct_change().dropna()

        rtns = pd.DataFrame(index=self.sig.index)
        rtns = pct_chg.mul(self.positions.values)
        rtns_comb = rtns.sum(axis=1)
        return rtns_comb



if __name__ == '__main__':

    df = pd.read_csv('all_asset_class.csv', index_col='Date', parse_dates=True)
    df.drop(['dollar', 'yc', 'senti', '7yTR', '10yTR', '30yTR'], axis=1, inplace=True)
    df.ffill(inplace=True)
    df.columns = ['Crude', 'Gold', 'DM Equity', 'EM Corp', 'EM Equity', 'TSY', '$Corp', '$HY', '$BBB']

    stocks = ['Crude', 'Gold', 'DM Equity', 'EM Corp', 'EM Equity', 'TSY', '$Corp', '$HY', '$BBB']

    df = df[stocks]
    all_list = []
    for i,x in enumerate(stocks):
        al = str(stocks[i])
        all_list.append(al)


    df_m = df.resample('BM').last()

    # Get the start date
    lookbackperiod = 6 #months
    start_date = pd.date_range(df_m.index[0], periods=lookbackperiod+2, freq='BM')[lookbackperiod+1]
    date_list = df_m[start_date:].index.tolist()

    #get separate startdate to calculate returns
    start_date = pd.date_range(df_m.index[0], periods=lookbackperiod + 1, freq='BM')[lookbackperiod]
    date_list_returns = df_m[start_date:].index.tolist()

    mv = MVO(stocks,df,date_list)
    sig = mv.generate_signals()

    port = MarketOnClosePortfolio(df,stocks,sig,date_list,date_list_returns)
    returns = port.backtest_portfolio()



    #####Claculate risk adjusted returns
    # get libor index
    lib = ext.convert_libor_to_index()
    lib = lib.reindex(returns.index).pct_change()
    lib.replace(np.nan, 0, inplace=True)

    # deduct lib from risk parity returns
    returns = pd.DataFrame(returns - lib)
    returns.columns = ['MVO']

    # get cummulative product
    cum_returns = np.cumproduct(returns + 1) - 1
    cum_returns = pd.DataFrame(cum_returns)
    cum_returns.columns = [str(all_list) + str('MVO')]

    # reindex all assets
    dfri = df.reindex(cum_returns.index)
    df_chg = dfri.resample('BM').last().pct_change()
    df_chg.replace(np.nan, 0, inplace=True)

    # adjust for libor
    for x in df_chg.columns:
        df_chg[x] -= lib

    df_chg_cum = np.cumproduct(df_chg + 1) - 1

    # combine cummulative and monthly returns
    returns = pd.concat([returns, df_chg], axis=1)
    cum_returns = pd.concat([cum_returns, df_chg_cum], axis=1)

    # calculate sharpe ratios
    vol = np.std(returns) * np.sqrt(12)
    ann_returns = returns.mean() * 12
    sr = ann_returns.divide(vol.values)
    sr.sort_values(inplace=True)
    print('\nSharpe ratios')
    print(sr)
    sr.plot(kind='barh')

    print('\nAverage annulised returns')
    ann_returns *= 100
    ann_returns.sort_values(inplace=True)
    print(ann_returns)

    # print(cum_returns)
    cum_returns.plot()
    plt.title('Allocation returns')
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    plt.show()