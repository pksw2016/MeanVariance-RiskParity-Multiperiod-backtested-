import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from scipy.optimize import minimize
TOLERANCE = 1e-10


def _allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    # It returns the risk of the weights distribution
    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) / portfolio_risk

    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return assets_risk_contribution


def _risk_budget_objective_error(weights, args):

    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We convert the weights to a matrix
    weights = np.matrix(weights)

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = _assets_risk_contribution_to_allocation_risk(weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

    # It returns the calculated error
    return error


def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x}) # i.e. lambda is >=0

    # Optimisation process in scipy
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights


def get_weights():

    #prices = pd.read_csv('spx_comp_longhist.csv', index_col='date', parse_dates=True)[['TSLA', 'AMZN', 'AAPL']]
    data = pd.read_csv('spx_comp_longhist.csv', index_col='date', parse_dates=True)
    prices = data['2000':]
    mask = prices.isnull().sum() < 10
    stock_list = mask[mask==1].index
    prices = prices[stock_list]

    stocks = ['AAPL', 'AMZN', 'MSFT']   #'AAPL', 'AMZN', 'BA', 'BAC', 'C', 'CSCO', 'CVX', 'DIS', 'HD', 'INTC',
                                        #'JNJ', 'JPM', 'KO', 'MSFT', 'ORCL', 'PEP', 'PFE', 'PG', 'T', 'UNH',
                                        #'VZ', 'WFC', 'WMT', 'XOM'
    prices = prices[stocks]
    # We calculate the covariance matrix
    #covariances = 52.0 * prices.asfreq('W-FRI').pct_change().iloc[1:, :].cov().values

    ### Covariance using other way
    covariances = 252*prices.pct_change(1).dropna().cov().values

    #Get daily returns
    hist_ret=prices.pct_change(1).dropna().mean()*252

    # The desired contribution of each asset to the portfolio risk: we want all
    # asset to contribute equally
    assets_risk_budget = [1.0 / prices.shape[1]] * prices.shape[1]

    # Initial weights: equally weighted
    init_weights = [1.0 / prices.shape[1]] * prices.shape[1]

    # Optimisation process of weights
    weights = _get_risk_parity_weights(covariances, assets_risk_budget, init_weights)

    # Convert the weights to a pandas Series
    weights = pd.Series(weights, index=prices.columns, name='weight')

    # It returns the optimised weights
    return weights,covariances,hist_ret

if __name__ == '__main__':

    final_weights,covariances,returns_hist=get_weights()

    port_ret = np.sum(final_weights * returns_hist)
    port_vol = np.sqrt(np.dot(final_weights.T,np.dot(covariances,final_weights)))


    print('The risk parity weights are ')
    print(final_weights*100)
    print('\n')
    print('Covarience matrix')
    print(covariances)
    print('\n')
    print('Historic returns are: ')
    print(returns_hist*100)
    print("portfolio return is {:0.2f}%".format(port_ret*100))
    print("portfolio vol is {:0.2f}%".format(port_vol*100))
    print("sharpe ratio is {:0.4f}".format(port_ret/port_vol))