####Problem 1#####
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import stats

# Test the kurtosis function for bias in small sample sizes
samples = 1000
sample_size = 100000
d = np.random.normal(0, 1, (samples, sample_size))

#calculate skewness and kurtosis
kurts = kurtosis(d, axis=1)
skew = skew(d, axis=1)

#summary statistics
print("mean kurts",np.mean(kurts))
print("var kurts",np.var(kurts))
print("std kurts",np.std(kurts))

print("mean skew",np.mean(skew))
print("var skew",np.var(skew))
print("std skew",np.std(skew))

#p-value
t, p = stats.ttest_1samp(kurts, 3)
print("p-value for kurtosis:", p)

t, p = stats.ttest_1samp(skew, 0)
print("p-value for skewness:", p)

####Problem 2####

###Part1###


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# Load the data
data = pd.read_csv("problem2.csv")

# Define the dependent and independent variables
x = data['x']
y = data['y']

# Add a column of ones to the x data for the constant term
x = sm.add_constant(x)

# Perform OLS regression
model = sm.OLS(y, x)
results = model.fit()

# Print the results
print(results.summary())

# Calculate the error vector
predictions = results.predict(x)
error = y - predictions


plt.hist(error)
plt.show()

###Part 2###

from scipy.stats import norm

# Fit the normal distribution to the data
mu, std = norm.fit(data)

# Print the estimated parameters
print("Mean:", mu)
print("Standard deviation:", std)

###Part 3###
from scipy.stats import t
from scipy.optimize import minimize

# Fix degrees of freedom to 100
df = 100

# Define the negative log-likelihood function
def neg_log_lik(params, data, df):
    loc, scale = params
    loglik = t.logpdf(data, df=df, loc=loc, scale=scale).sum()
    return -loglik

# Initial guess for the parameters
x0 = [0, 1]

# Find the MLE
res = minimize(neg_log_lik, x0, args=(data, df))

# Print the estimated parameters
print("Mean:", res.x[0])
print("Standard deviation:", res.x[1])


#####Problem 3####
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate some data for an AR(1) model
np.random.seed(1)
ar1_data = np.random.randn(100)
for i in range(1, 100):
    ar1_data[i] = 0.6 * ar1_data[i-1] + np.random.randn()

# Plot the ACF and PACF for AR(1)
plot_acf(ar1_data, lags=10)
plot_pacf(ar1_data, lags=10)

# Generate some data for an AR(2) model
np.random.seed(1)
ar2_data = np.random.randn(100)
for i in range(2, 100):
    ar2_data[i] = 0.6 * ar2_data[i-1] - 0.3 * ar2_data[i-2] + np.random.randn()

# Plot the ACF and PACF for AR(2)
plot_acf(ar2_data, lags=10)
plot_pacf(ar2_data, lags=10)

# Generate some data for an AR(3) model
np.random.seed(1)
ar3_data = np.random.randn(100)
for i in range(3, 100):
    ar3_data[i] = 0.6 * ar3_data[i-1] - 0.3 * ar3_data[i-2] + 0.2 * ar3_data[i-3] + np.random.randn()

# Plot the ACF and PACF for AR(3)
plot_acf(ar3_data, lags=10)
plot_pacf(ar3_data, lags=10)

# Generate some data for an MA(1) model
np.random.seed(1)
ma1_data = np.random.randn(100)
for i in range(1, 100):
    ma1_data[i] = 0.6 * np.random.randn() + ma1_data[i-1]

# Plot the ACF and PACF for MA(1)
plot_acf(ma1_data, lags=10)
plot_pacf(ma1_data, lags=10)

# Generate some data for an MA(2) model
np.random.seed(1)
ma2_data = np.random.randn(100)
for i in range(2, 100):
    ma2_data[i] = 0.6 * np.random.randn() + 0.3 * np.random.randn() + ma2_data[i-1] - 0.3 * ma2_data[i-2]

# Plot the ACF and PACF for MA(2)
plot_acf(ma2_data, lags=10)
plot_pacf(ma2_data, lags=10)

# Generate some data for an MA(3) model
np.random.seed(1)
ma3_data = np.random.randn(100)
for i in range(3, 100):
    ma3_data[i] = 0.6 * np.random.randn() + 0.3 * np.random.randn() + 0.2 * np.random.randn() + ma3_data[i-1] - 0.3 * ma3_data[i-2] + 0.2 * ma3_data[i-3]

# Plot the ACF and PACF for MA(3)
plot_acf(ma3_data, lags=10)
plot_pacf(ma3_data, lags=10)




