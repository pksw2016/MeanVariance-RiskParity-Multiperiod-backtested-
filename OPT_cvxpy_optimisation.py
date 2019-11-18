import pandas as pd
import numpy as np
#import pandas_datareader.data as web
import matplotlib
matplotlib.use('qt5agg') # Remove this to compare with MacOSX backend
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as sco


import cvxpy as cvx

expected_returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.40])



def markowitz_portfolio(means, cov, risk_aversion,constrain_type):
    """Generate the optimal fully-invested portfolio for a given risk/returns tradeoff.
    """
    weights = cvx.Variable(len(means))
    expected_return = weights.T * means
    expected_vol = cvx.quad_form(weights, cov)

    utility = expected_return - risk_aversion * expected_vol
    objective = cvx.Maximize(utility)

    # individual assignment
    constraints1 = [
        cvx.sum(weights) == 1,  # fully-invested
        weights[0] >= 0.05,
        weights[1] >= 0.05,
        weights[2] >= 0.05,
        weights[3] >= 0.05,
        weights[3] <= 0.50,
        weights[4] >= 0.05,
        weights[5] >= 0.05,
        weights[4] <= 0.5,
        weights >= 0,  # long-only
    ]

    # long only
    constraints2 = [
        cvx.sum(weights) == 1,  # fully-invested
        weights >= 0,  # long-only
    ]

    # minimum allocation of 5%
    constraints3 = [
        cvx.sum(weights) == 1,  # fully-invested
        weights >= 0.05,  # long-only
    ]

    # unconstrained
    constraints4 = [   #unconstrained
        cvx.sum(weights) == 1,  # fully-invested
    ]


    if constrain_type == 1:
        cons = constraints1
    elif constrain_type == 2:
        cons = constraints2
    elif constrain_type == 3:
        cons = constraints3
    elif constrain_type == 4:
        cons = constraints4


    problem = cvx.Problem(objective, cons)
    problem.solve()

    return np.array(weights.value.flat).round(4), expected_return.value, expected_vol.value



if __name__ == '__main__':


    expected_rets = np.array([0.37, 0.28, -0.06, 0.012, 0.40, 0.034])


    cov = np.array([[0.144, 0.068, 0.062, 0.046, 0.030, 0.023],
                    [0.068, 0.253, 0.068, 0.051, 0.047, 0.028],
                    [0.062, 0.068, 0.199, 0.081, 0.052, 0.033],
                    [0.046, 0.051, 0.081, 0.091, 0.027, 0.027],
                    [0.030, 0.047, 0.052, 0.027, 0.274, 0.010],
                    [0.023, 0.028, 0.033, 0.027, 0.010, 0.045]])


    # run a loop for different values of risk aversion

    samples=50
    gamma_vals = np.logspace(-2, 3, num=samples)

    wt = []
    rt = []
    risk = []
    gamma = []
    sr = []
    constype = 4

    for x in range(samples):
        gamma_val=gamma_vals[x]
        weights, rets, var = markowitz_portfolio(expected_rets, cov, gamma_val,constype)
        sr.append(rets/np.sqrt(var))
        wt.append(weights)
        rt.append(rets)
        risk.append(np.sqrt(var))
        gamma.append(gamma_val)

    #print("Weights:", weights*100); print("Expected Return:", rets*100); print("Expected Variance:", np.sqrt(var)*100); print("sharpe:", rets/(np.sqrt(var)))

    #find max sharp ratio
    max_index=sr.index(max(sr))

    max_ret=rt[max_index]
    max_risk=risk[max_index]
    gamma_max=gamma[max_index]

    # check returns vs. sharpe ratio
    # plt.subplot(2, 1, 2)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(gamma_vals, rt, color='g')
    ax2.plot(gamma_vals, sr, color='b')

    ax1.set_xlabel('Gamma values')
    ax1.set_ylabel('Returns', color='g')
    ax2.set_ylabel('Sharpe Ratio', color='b')



    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.scatter(risk,rt,marker='o')
    plt.scatter(max_risk,max_ret,color='red')
    plt.show()

    print('The maximum risk aversion is {:0.2f}'.format(gamma_max))
    print('Optimum weights are {}'.format(wt[max_index]))