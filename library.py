import numpy as np
import pandas as pd
import scipy
import copy
from bisect import bisect_left
from scipy import stats
from scipy.optimize import minimize 
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm, t, gaussian_kde
from statsmodels.nonparametric.kernel_density import KDEMultivariate


#1. Covariance estimation techniques.
def calculate_exponential_weights(lags, lamb):
    # Calculate exponential weights given the number of lags and the decay parameter (lambda)
    weights = (1 - lamb) * np.power(lamb, np.arange(lags))
    # Normalize the weights to sum to 1
    normalized_weights = weights / np.sum(weights)
    return normalized_weights

def calculate_ewcov(data, lamb):
    # Calculate the exponentially weighted covariance matrix for a given dataset and decay parameter (lambda)
    n, p = data.shape
    weights = calculate_exponential_weights(p, lamb)
    centered_data = data - np.mean(data, axis=0)
    ewcov = np.zeros((p, p))
    for i in range(n):
        ewcov += np.outer(centered_data[i], centered_data[i]) * weights[i]
    return ewcov



#2. Non PSD fixes for correlation matrices
def is_psd(matrix):
    vals = np.linalg.eigh(matrix)[0]
    return np.all(vals >= -1e-8)

def near_psd(a, epsilon=0.0):
    # Compute the nearest positive semi-definite matrix to a given matrix a
    n = a.shape[0]
    out = copy.deepcopy(a)
    invSD = None
    # If a is a covariance matrix, calculate the corresponding correlation matrix
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    # SVD decomposition and update the eigenvalues to be positive
    u, s, v = np.linalg.svd(out)
    s = np.maximum(s, epsilon)
    # Calculate the square root of the inverse eigenvalues
    t = 1.0 / np.sqrt(s)
    t = np.diag(t)
    # Compute the nearest correlation matrix
    b = u @ t @ v
    out = b @ b.T
    # Add back the variance if a was a covariance matrix
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out

def Frobenius(input):
    result = 0
    for i in range(len(input)):
        for j in range(len(input)):
            result += input[i][j]**2
    return result
def Higham_psd(input):
    weight = np.identity(len(input))
        
    norml = np.inf
    Yk = input.copy()
    Delta_S = np.zeros_like(Yk)
    
    invSD = None
    if np.count_nonzero(np.diag(Yk) == 1.0) != input.shape[0]:
        invSD = np.diag(1 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD
    
    Y0 = Yk.copy()

    for i in range(1000):
        Rk = Yk - Delta_S
        # PS
        Xk = np.sqrt(weight)@ Rk @np.sqrt(weight)
        vals, vecs = np.linalg.eigh(Xk)
        vals = np.where(vals > 0, vals, 0)
        Xk = np.sqrt(weight)@ vecs @ np.diagflat(vals) @ vecs.T @ np.sqrt(weight)
        Delta_S = Xk - Rk
        #PU
        Yk = Xk.copy()
        np.fill_diagonal(Yk, 1)
        norm = Frobenius(Yk-Y0)
        #norm = np.linalg.norm(Yk-Y0, ord='fro')
        min_val = np.real(np.linalg.eigvals(Yk)).min()
        if abs(norm - norml) < 1e-8 and min_val > -1e-9:
            break
        else:
            norml = norm
    
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Yk = invSD @ Yk @ invSD
    return Yk


#3. Simulation Methods

def chol_psd(a):
    # Calculate the Cholesky root of a positive semi-definite matrix a
    n = a.shape[0]
    root = np.zeros((n, n))
    for j in range(n):
        s = np.dot(root[j,:j], root[j,:j])
        temp = a[j,j] - s
        if temp > 0:
            root[j,j] = np.sqrt(temp)
        else:
            root[j,j] = 0.0
        if root[j,j] == 0.0:
            root[j,j:n] = 0.0
        else:
            for i in range(j+1, n):
                s = np.dot(root[i,:j], root[j,:j])
                root[i,j] = (a[i,j] - s) / root[j,j]
    return root

def direct_simulation(cov, n_samples=25000):
    # Generate random samples using the direct simulation method with the Cholesky root
    B = chol_psd(cov)
    r = norm.rvs(size=(B.shape[1], n_samples))
    return B @ r

def pca_simulation(cov, pct_explained, n_samples=25000):
    # Generate random samples using the PCA simulation method with a specified percentage of explained variance
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    evr = sorted_eigenvalues / sorted_eigenvalues.sum()
    cumulative_evr = np.cumsum(evr)
    # Clip the eigenvalues to avoid negative values
    explained_vals = np.clip(sorted_eigenvalues, 0, np.inf)
    # Find the number of eigenvectors required to explain the desired percentage of variance
    n_components = np.searchsorted(cumulative_evr, pct_explained, side='right')
    explained_vecs = sorted_eigenvectors[:, :n_components]
    B = explained_vecs @ np.diag(np.sqrt(explained_vals[:n_components]))
    r = norm.rvs(size=(B.shape[1], n_samples))
    return B @ r

#4. VaR calculation methods (all discussed)

def calculate_var(data, mean=0, alpha=0.05):
    # Calculate the value-at-risk (VaR) of a given data set at a specified confidence level
    return mean - np.quantile(data, alpha)

def normal_var(data, mean=0, alpha=0.05, nsamples=10000):
    # Calculate the VaR of a normal distribution fitted to the data
    sigma = np.std(data, ddof=1)
    simulation_norm = norm.rvs(loc=mean, scale=sigma, size=nsamples)
    var_norm = calculate_var(simulation_norm, mean, alpha)
    return var_norm

def ewcov_normal_var(data, mean=0, alpha=0.05, nsamples=10000):
    # Calculate the VaR of a normal distribution with an exponentially weighted covariance matrix
    ew_cov = calculate_ewcov(np.matrix(data).T, 0.94)
    ew_variance = ew_cov[0, 0]
    sigma = np.sqrt(ew_variance)
    simulation_ew = norm.rvs(loc=mean, scale=sigma, size=nsamples)
    var_ew = calculate_var(simulation_ew, mean, alpha)
    return var_ew

def t_var(data, mean=0, alpha=0.05, nsamples=10000):
    # Calculate the VaR of a t distribution fitted to the data
    params = t.fit(data, method="MLE")
    df, loc, scale = params
    simulation_t = t.rvs(df, loc=loc, scale=scale, size=nsamples)
    var_t = calculate_var(simulation_t, mean, alpha)
    return var_t

def historic_var(data, mean=0, alpha=0.05):
    # Calculate the historical VaR of the data set
    return calculate_var(data, mean, alpha)

def kde_var(data, mean=0, alpha=0.05):
    # Calculate the VaR of a kernel density estimate (KDE) of the data set
    kde = gaussian_kde(data)
    quantile_func = lambda x: kde.integrate_box_1d(-np.inf, x) - alpha
    x0 = kde.resample(1)[0][0]
    var_kde = mean - scipy.optimize.fsolve(quantile_func, x0)[0]
    return var_kde

#5. ES calculation
def calculate_es(data, mean=0, alpha=0.05):
    return calculate_var(data, mean, alpha)


def pd_calculate_returns(prices, method="arithmetic"):
    # Calculate the arithmetic or logarithmic returns of a price series using pandas
    if method == "arithmetic":
        price_change_percent = prices.pct_change()
    elif method == "log":
        price_change_percent = np.log(prices / prices.shift(1))
    return price_change_percent.dropna()