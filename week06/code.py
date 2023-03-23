###Problem1###

import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from datetime import date
import matplotlib.pyplot as plt

# Compute time until expiration
today = date(2023, 3, 3)
expiry = date(2023, 3, 17)
days_until_expiry = (expiry - today).days
time_until_expiry = days_until_expiry / 365
print("Time until Expiration: {:.4f}".format(time_until_expiry))

# Define a function to compute option prices using the Black-Scholes model

def black_scholes(underlying_price, strike_price, ttm, risk_free_rate, b, implied_volatility, option_type="call"):
    d1 = (log(underlying_price / strike_price) + (b + implied_volatility ** 2 / 2) * ttm) / (implied_volatility * sqrt(ttm))
    d2 = d1 - implied_volatility * sqrt(ttm)

    if option_type == "call":
        return underlying_price * exp((b - risk_free_rate) * ttm) * norm.cdf(d1) - strike_price * exp(-risk_free_rate * ttm) * norm.cdf(d2)
    elif option_type == "put":
        return strike_price * exp(-risk_free_rate * ttm) * norm.cdf(-d2) - underlying_price * exp((b - risk_free_rate) * ttm) * norm.cdf(-d1)
    else:
        print("Invalid option type")

# Set the initial parameters
stock_value = 165
strike_value = 165
interest_rate = 0.0425
dividend_rate = 0.0053
ivol_range = np.linspace(0.1, 0.8, 200)

call_prices = np.zeros(len(ivol_range))
put_prices = np.zeros(len(ivol_range))
call_prices_diff = np.zeros(len(ivol_range))
put_prices_diff = np.zeros(len(ivol_range))

# Compute call and put option prices for the specified implied volatilities
for idx, implied_vol in enumerate(ivol_range):
    call_prices[idx] = black_scholes(stock_value, strike_value, time_until_expiry, interest_rate, dividend_rate, implied_vol, "call")
    put_prices[idx] = black_scholes(stock_value, strike_value, time_until_expiry, interest_rate, dividend_rate, implied_vol, "put")
    call_prices_diff[idx] = black_scholes(stock_value, strike_value + 20, time_until_expiry, interest_rate, dividend_rate, implied_vol, "call")
    put_prices_diff[idx] = black_scholes(stock_value, strike_value - 20, time_until_expiry, interest_rate, dividend_rate, implied_vol, "put")

# Plot option prices for implied volatilities between 0.1 and 0.8 (Same Strike)
plt.figure()
plt.plot(ivol_range, call_prices, label="Call")
plt.plot(ivol_range, put_prices, label="Put")
plt.xlabel("Implied Volatility")
plt.ylabel("Option Price")
plt.legend()
plt.title("Option Prices for Varying Implied Volatilities (Same Strike)")
plt.grid()
plt.show()

# Plot option prices for implied volatilities between 0.1 and 0.8 (Different Strike)
plt.figure()
plt.plot(ivol_range, call_prices_diff, label="Call")
plt.plot(ivol_range, put_prices_diff, label="Put")
plt.xlabel("Implied Volatility")
plt.ylabel("Option Price")
plt.legend()
plt.title("Option Prices for Varying Implied Volatilities (Different Strike)")
plt.grid()
plt.show()



###Problem2###

import math
import sys
import pandas as pd
import numpy as np

def calculate_returns(data, method="discrete", date_col="Date"):
    assert date_col in data.columns

    price_vars = list(data.columns)
    price_vars.remove(date_col)
    price_data = data[price_vars]

    num_vars = len(price_vars)
    num_rows = len(price_data) - 1

    returns_data = np.zeros(shape=(num_rows, num_vars))
    returns_df = pd.DataFrame(returns_data)

    for i in range(num_rows):
        for j in range(num_vars):
            returns_df.iloc[i, j] = price_data.iloc[i + 1, j] / price_data.iloc[i, j]
            if method == "discrete":
                returns_df.iloc[i, j] -= 1
            elif method == "log":
                returns_df.iloc[i, j] = math.log(returns_df.iloc[i, j])
            else:
                sys.exit(1)

    date_series = data[date_col].drop(index=0)
    date_series.index -= 1

    output = pd.DataFrame({date_col: date_series})

    for i in range(num_vars):
        output[price_vars[i]] = returns_df.iloc[:, i]

    return output



import numpy as np
import pandas as pd
from math import log, sqrt, exp
from datetime import date
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns



def black_scholes(underlying_price, strike_price, ttm, risk_free_rate, b, implied_volatility, option_type="call"):
    d1 = (log(underlying_price / strike_price) + (b + implied_volatility ** 2 / 2) * ttm) / (implied_volatility * sqrt(ttm))
    d2 = d1 - implied_volatility * sqrt(ttm)

    if option_type == "call":
        return underlying_price * exp((b - risk_free_rate) * ttm) * norm.cdf(d1) - strike_price * exp(-risk_free_rate * ttm) * norm.cdf(d2)
    elif option_type == "put":
        return strike_price * exp(-risk_free_rate * ttm) * norm.cdf(-d2) - underlying_price * exp((b - risk_free_rate) * ttm) * norm.cdf(-d1)
    else:
        print("Invalid option type")


current_date = date(2023, 3, 3)
risk_free_rate = 0.0425
continuous_coupon = 0.0053
b = risk_free_rate - continuous_coupon
underlying_price = 151.03

data = pd.read_csv("AAPL_Options.csv")
expiration_dates = data["Expiration"]
option_types = data["Type"]
strike_prices = data["Strike"]
option_prices = data["Last Price"]
implied_volatilities = np.zeros(len(data))

for i in range(len(data)):
    ttm = (date(int(expiration_dates[i].split("/")[2]), int(expiration_dates[i].split("/")[0]), int(expiration_dates[i].split("/")[1])) - current_date).days / 365
    func = lambda iv: black_scholes(underlying_price, int(strike_prices[i]), ttm, risk_free_rate, b, iv, option_types[i].lower()) - float(option_prices[i])
    implied_volatilities[i] = fsolve(func, 0.5)

data["ivol"] = implied_volatilities

call_data = data.loc[data["Type"] == "Call"]
put_data = data.loc[data["Type"] == "Put"]

plt.figure()
plt.plot(strike_prices, implied_volatilities, label="All Options")
plt.plot(call_data.Strike, call_data.ivol, label="Call")
plt.plot(put_data.Strike, put_data.ivol, label="Put")
plt.xlabel("Strike Prices")
plt.ylabel("Implied Volatility")
plt.legend()
plt.show()

price_data = pd.read_csv("DailyPrices.csv")
log_returns = calculate_returns(price_data, method="log", date_col="Date")
aapl_log_returns = log_returns["AAPL"]

mean_return = np.mean(aapl_log_returns)
std_return = np.std(aapl_log_returns)
simulated_log_returns = norm(mean_return, std_return).rvs(10000)

plt.figure()
sns.kdeplot(aapl_log_returns, color="b", label='Actual Returns')
sns.kdeplot(simulated_log_returns, color="r", label='Simulated')
plt.show()
###Problem3###

import functionlib
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

### PART 1 ###
# define a function to simulate portfolio values with simulated prices
def sim_portfolio_value(portfolio, sim_p, ivol, rf, b, day_passed=0):
    sim_values = pd.DataFrame(index=portfolio.index, columns=list(range(sim_p.shape[0])))
    for i in portfolio.index:
        if portfolio["Type"][i] == "Stock":
            individual_value = sim_p
        else:
            underlying_p = sim_p
            strike = portfolio["Strike"][i]
            ttm = ((portfolio["ExpirationDate"][i] - datetime(2023,3,3)).days - day_passed) / 365
            individual_value = np.zeros(len(underlying_p))
            for z in range(len(underlying_p)):
                individual_value[z] = functionlib.gbsm(underlying_p[z], strike, ttm, rf, b, ivol[i], type=portfolio["OptionType"][i].lower())
        
        sim_values.iloc[i,:] = portfolio["Holding"][i] * individual_value
    
    sim_values['Portfolio'] = portfolio['Portfolio']
    return sim_values.groupby('Portfolio').sum()

# read data
portfolio = pd.read_csv("problem3.csv", parse_dates=["ExpirationDate"])
underlying = 151.03
rf = 0.0425
b = 0.0425 - 0.0053
ivol = np.zeros(len(portfolio.index))
for j in range(len(portfolio.index)):
    if type(portfolio["OptionType"][j]) != str:
        ivol[j] = 0
    else:
        ivol[j] = functionlib.implied_vol(underlying, portfolio["Strike"][j], (portfolio["ExpirationDate"][j] - datetime(2023,3,3)).days / 365, rf, b, portfolio["CurrentPrice"][j], type=portfolio["OptionType"][j].lower())

# apply the defined function
sim_p = np.linspace(100, 200, 50)
simulated_vals = sim_portfolio_value(portfolio, sim_p, ivol, rf, b)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
idx = 0
for portfolios, dataframes in simulated_vals.groupby('Portfolio'):
    i, j = idx // 3, idx % 3
    ax = axes[i][j]
    ax.plot(sim_p, dataframes.iloc[0, :].values)
    ax.set_title(portfolios)
    ax.set_xlabel('Underlying Price', fontsize=8)
    ax.set_ylabel('Portfolio Value', fontsize=8)
    idx += 1

### PART 2 ###
# read in prices data
prices = pd.read_csv("DailyPrices.csv")
lreturns = functionlib.return_cal(prices,method="log",datecol="Date")
aapl_lreturns = lreturns["AAPL"]
aapl_lreturns = aapl_lreturns - aapl_lreturns.mean()

# start fitting returns with AR(1) model
mod = sm.tsa.ARIMA(aapl_lreturns, order=(1, 0, 0))
results = mod.fit()
summary = results.summary()
m = float(summary.tables[1].data[1][1])
a1 = float(summary.tables[1].data[2][1])
s = math.sqrt(float(summary.tables[1].data[3][1]))
sim = pd.DataFrame(0, index=range(10000), columns=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"])
for i in range(len(sim.columns)):
    for j in range(len(sim)):
        if i == 0:
            sim.iloc[j,i] =  a1 * (aapl_lreturns.iloc[-1]) + s * np.random.normal() + m
        else:
            sim.iloc[j,i] =  a1 * (sim.iloc[j,i-1]) + s * np.random.normal() + m

# calculate prices on the 10th day from current date
ar1_sim_p = pd.DataFrame(0, index=range(10000), columns=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"])
for i in range(len(ar1_sim_p.columns)):
    if i == 0:
        ar1_sim_p.iloc[:,i] = np.exp(sim.iloc[:,i]) * underlying
    else:
        ar1_sim_p.iloc[:,i] = np.exp(sim.iloc[:,i]) * ar1_sim_p.iloc[:,i-1]
ar1_sim_10p = ar1_sim_p.iloc[:,-1]

# calculate portfolio values based on the 10th day's simulated prices from AR(1) model
ar1_sim_10port = sim_portfolio_value(portfolio, ar1_sim_10p, ivol, rf, b, day_passed=10)

# start calculating mean, VaR, ES for each portfolio
resulting_mat = pd.DataFrame(0, index=ar1_sim_10port.index.values, columns=["Mean of Portfolio Value($)", "Mean of Losses/Gains($)", "VaR($)", "ES($)", "VaR(%)", "ES(%)"])
for i in range(len(resulting_mat)):
    resulting_mat.iloc[i,0] = ar1_sim_10port.iloc[i,:].mean()
    resulting_mat.iloc[i,1] = portfolio.groupby('Portfolio').sum().iloc[i,-1] - ar1_sim_10port.iloc[i,:].mean()
    resulting_mat.iloc[i,2], resulting_mat.iloc[i,3] = portfolio.groupby('Portfolio').sum().iloc[i,-1] - functionlib.cal_ES(ar1_sim_10port.iloc[i,:],alpha=0.05)
    resulting_mat.iloc[i,4] = resulting_mat.iloc[i,2] * 100 / portfolio.groupby('Portfolio').sum().iloc[i,-1]
    resulting_mat.iloc[i,5] = resulting_mat.iloc[i,3] * 100 / portfolio.groupby('Portfolio').sum().iloc[i,-1]
resulting_mat["Current Value (on 2023/3/3)"] = portfolio.groupby('Portfolio').sum()["CurrentPrice"]
print(resulting_mat)