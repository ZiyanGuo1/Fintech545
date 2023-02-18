###Problem 1###
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

sigma = 0.5
P_0 = 100

sim_returns_cla = np.random.normal(0, sigma, 5000)
sim_prices_cla = P_0 + sim_returns_cla

std_cla = sim_prices_cla.std()
mean_cla = sim_prices_cla.mean()


theory_std_cla = sigma
theory_mean_cla = P_0


sim_returns_ari = np.random.normal(0, sigma, 5000)
sim_prices_ari = P_0 * (1 + sim_returns_ari)


std_ari = sim_prices_ari.std()
mean_ari = sim_prices_ari.mean()


theory_std_ari = P_0 * sigma
theory_mean_ari = P_0

sim_returns_geo = np.random.normal(0, sigma, 5000)
sim_prices_geo = P_0 * np.exp(sim_returns_geo)

std_geo = sim_prices_geo.std()
mean_geo = sim_prices_geo.mean()


theory_std_geo = P_0 * math.sqrt(math.exp(2 * sigma**2) - math.exp(sigma**2))
theory_mean_geo = P_0 * math.exp(0.5 * sigma**2)

fig, axs = plt.subplots(3)
sns.distplot(sim_prices_cla, ax=axs[0])
sns.distplot(sim_prices_ari, ax=axs[1])
sns.distplot(sim_prices_geo, ax=axs[2])
axs[0].set_title("Classic Brownian Motion")
axs[1].set_title("Arithmetic Return System")
axs[2].set_title("Geometric Brownian Motion")
plt.show()






###Problem2###
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from scipy.stats import t
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy import linalg
from numpy.linalg import eigh

def return_calculate(prices: pd.DataFrame, method: str = "DISCRETE", date_column: str = "date") -> pd.DataFrame:
    vars = list(prices.columns)
    n_vars = len(vars)
    vars.remove(date_column)
    if n_vars == len(vars):
        raise ValueError(f"dateColumn: {date_column} not in DataFrame: {vars}")
    n_vars = n_vars - 1

    p = prices[vars].to_numpy()
    n, m = p.shape
    p2 = np.empty((n - 1, m))
    
    for i in range(n-1):
        for j in range(m):
            p2[i,j] = p[i+1,j] / p[i,j]

    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")

    dates = prices[date_column].iloc[1:]
    out = pd.DataFrame({date_column: dates})
    for i in range(n_vars):
        out[vars[i]] = p2[:,i]
    return out

data = pd.read_csv('DailyPrices.csv')

cal = return_calculate(data, "DISCRETE", "Date")
print(cal)

META = cal["META"] - cal["META"].mean()
alpha = 0.05
sd = META.std()

#1
VaR_05_q1 = -norm.ppf(0.05, loc = 0, scale = sd)
print(VaR_05_q1)

#2
def get_weights(L, n):

    weights = (1 - L) * (L ** np.arange(n))
    return weights / weights.sum()

def weighted_variance(array, L):

    weights = get_weights(L, len(array))
    w_var = np.sum(weights * (array - array.mean()) ** 2)
    return w_var
xyz = weighted_variance(META[::-1], 0.94)
sd1 = np.sqrt(xyz)
VaR_05_q2 = -norm.ppf(0.05, loc = 0, scale = sd1)
print(VaR_05_q2)

#3
from scipy.stats import t
def cal_t(n):
    like = 1
    for i in META:
        like = like * t.pdf(i,n,scale=sd)
    return -np.log(like)

df = minimize(cal_t, 15, method = 'BFGS', options={'disp': True})

df.x

VaR_05_q3 = -t.ppf(0.05, df = 8.84783522, scale = sd)
print(VaR_05_q3)


#4
ar1_model = AutoReg(META, lags = 1).fit()

ar1_model.params

VaR_05_q4 = - (-0.000062 + 0.007233*META[248] + norm.ppf(0.05, loc = 0, scale = sd))
print(VaR_05_q4)

#5
N = 10000
x = np.random.randint(low = 1, high = 248, size = 10000)
h_return = sorted(META[x])
H_VaR_05 = - np.quantile(h_return, 0.05)
print(H_VaR_05)




###Problem3###
for i in range(1,101):
    cal.iloc[:,i] = cal.iloc[:,i] - cal.iloc[:,i].mean()

def weight(matrix, L):
    
    n = matrix.shape[0]
    weights = np.zeros(n)
    
    for i in range(1,n+1):
        weights[i-1] = (1 - L) * (L**(i-1))
        
    x = weights.sum()
        
    return weights/x


def cov(matrix, L):
    
    weights = weight(matrix, L)
    weighted = np.zeros([matrix.shape[1], matrix.shape[1]])
    
    for i in range(0,matrix.shape[1]):
        
        w = np.matrix(weights)
        x = np.asmatrix(matrix[:,i]-matrix[:,i].mean())
        w_x = np.zeros([1, x.shape[0]])
        
        for a in range(0,x.shape[0]):
            
            w_x[0,a] = w[0,a] * x[a,0]
            
        for b in range(0,matrix.shape[1]):
            
            y = np.asmatrix(matrix[:,b]-matrix[:,b].mean())
            
            weighted[i,b] += np.dot(w_x,y)
 
    return weighted


wcov = cov(np.asmatrix(cal[::-1].iloc[:,1:101]), L = 0.94)

portfolio = pd.read_csv('portfolio.csv')

#Method1
N = 10000

portfolio_a = portfolio[portfolio['Portfolio']=='A']
portfolio_b = portfolio[portfolio['Portfolio']=='B']
portfolio_c = portfolio[portfolio['Portfolio']=='C']

stock_a = portfolio_a['Stock']
stock_b = portfolio_b['Stock']
stock_c = portfolio_c['Stock']

return_a = cal[stock_a]
return_b = cal[stock_b]
return_c = cal[stock_c]

wcov_a = cov(np.asmatrix(return_a)[::-1], L = 0.94)
wcov_b = cov(np.asmatrix(return_b)[::-1], L = 0.94)
wcov_c = cov(np.asmatrix(return_c)[::-1], L = 0.94)

data_a = data[stock_a]
data_b = data[stock_b]
data_c = data[stock_c]

a = 0
b = 0
c = 0
for i in range(len(stock_a)):
    a += data_a[::-1].iloc[0,:][i] * portfolio_a['Holding'].iloc[i]
for i in range(len(stock_b)):
    b += data_b[::-1].iloc[0,:][i] * portfolio_b['Holding'].iloc[i]
for i in range(len(stock_c)):
    c += data_c[::-1].iloc[0,:][i] * portfolio_c['Holding'].iloc[i]
    
sim_return_a = np.random.multivariate_normal(np.zeros(len(stock_a)),wcov_a,size = N)
sim_price_a = (sim_return_a + 1) * np.array(data_a[::-1].iloc[0,:])
sim_PV_a = np.dot(sim_price_a, np.array(portfolio_a['Holding']))
sim_PV_a = sorted(sim_PV_a)

sim_return_b = np.random.multivariate_normal(np.zeros(len(stock_b)),wcov_b,size = N)
sim_price_b = (sim_return_b + 1) * np.array(data_b[::-1].iloc[0,:])
sim_PV_b = np.dot(sim_price_b, np.array(portfolio_b['Holding']))
sim_PV_b = sorted(sim_PV_b)

sim_return_c = np.random.multivariate_normal(np.zeros(len(stock_c)),wcov_c,size = N)
sim_price_c = (sim_return_c + 1) * np.array(data_c[::-1].iloc[0,:])
sim_PV_c = np.dot(sim_price_c, np.array(portfolio_c['Holding']))
sim_PV_c = sorted(sim_PV_c)

index_05_a = int(0.05*N-1)
index_05_b = int(0.05*N-1)
index_05_c = int(0.05*N-1)

VaR_05_a = a - sim_PV_a[index_05_a]
VaR_05_b = b - sim_PV_b[index_05_b]
VaR_05_c = c - sim_PV_c[index_05_c]

print("Portfolio A is $", VaR_05_a)
print("Portfolio B is $", VaR_05_b)
print("Portfolio C is $", VaR_05_c)
print("Portfolio Total is $", VaR_05_a + VaR_05_b +VaR_05_c)

#Method2
hs_indices_a = np.random.randint(low = 1, high = 248, size = N)
hs_indices_b = np.random.randint(low = 1, high = 248, size = N)
hs_indices_c = np.random.randint(low = 1, high = 248, size = N)

hs_return_a = np.zeros([N,len(stock_a)])
hs_return_b = np.zeros([N,len(stock_b)])
hs_return_c = np.zeros([N,len(stock_c)])

for i in range(N):
    for j in range(len(stock_a)):
        hs_return_a[i,j] = return_a.iloc[hs_indices_a[i],j]
for l in range(N):
    for m in range(len(stock_b)):
        hs_return_b[l,m] = return_b.iloc[hs_indices_b[l],m]
for o in range(N):
    for p in range(len(stock_c)):
        hs_return_c[o,p] = return_c.iloc[hs_indices_c[o],p]

hs_price_a = (hs_return_a + 1) * np.array(data_a[::-1].iloc[0,:])
hs_price_b = (hs_return_b + 1) * np.array(data_b[::-1].iloc[0,:])
hs_price_c = (hs_return_c + 1) * np.array(data_c[::-1].iloc[0,:])

hs_PV_a = np.dot(hs_price_a, np.array(portfolio_a['Holding']))
hs_PV_b = np.dot(hs_price_b, np.array(portfolio_b['Holding']))
hs_PV_c = np.dot(hs_price_c, np.array(portfolio_c['Holding']))

hs_PV_a = sorted(hs_PV_a)
hs_PV_b = sorted(hs_PV_b)
hs_PV_c = sorted(hs_PV_c)

hs_index_05_a = int(0.05*N-1)
hs_index_05_b = int(0.05*N-1)
hs_index_05_c = int(0.05*N-1)

hs_VaR_05_a = a - hs_PV_a[hs_index_05_a]
hs_VaR_05_b = b - hs_PV_b[hs_index_05_b]
hs_VaR_05_c = c - hs_PV_c[hs_index_05_c]

print("Portfolio A is $", hs_VaR_05_a)
print("Portfolio B is $", hs_VaR_05_b)
print("Portfolio C is $", hs_VaR_05_c)
print("Portfolio Total is $", hs_VaR_05_a +hs_VaR_05_a)

