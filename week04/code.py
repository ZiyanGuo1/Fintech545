###Problem 1###
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

# set up the initial price at time 0, P(t-1), to be 100, and the standard deviation, sigma, to be 0.75
P_0 = 100
sigma = 0.75

# simulate prices for Classic Brownian Motion: P(t) = P(t-1) + r(t)
sim_returns_cla = np.random.normal(0, sigma, 10000)
sim_prices_cla = P_0 + sim_returns_cla

# calculate the mean and standard deviation of the simulated prices
mean_cla = sim_prices_cla.mean()
std_cla = sim_prices_cla.std()

# calculate the theoretical mean and standard deviation of P(t) for Classic Brownian Motion
theory_mean_cla = P_0
theory_std_cla = sigma

# simulate prices for Arithmetic Return System: P(t) = P(t-1) * (1 + r(t))
sim_returns_ari = np.random.normal(0, sigma, 10000)
sim_prices_ari = P_0 * (1 + sim_returns_ari)

# calculate the mean and standard deviation of the simulated prices
mean_ari = sim_prices_ari.mean()
std_ari = sim_prices_ari.std()

# calculate the theoretical mean and standard deviation of P(t) for Arithmetic Return System
theory_mean_ari = P_0
theory_std_ari = P_0 * sigma

# simulate prices for Geometric Brownian Motion: P(t) = P(t-1) * exp(r(t))
sim_returns_geo = np.random.normal(0, sigma, 10000)
sim_prices_geo = P_0 * np.exp(sim_returns_geo)

# calculate the mean and standard deviation of the simulated prices
mean_geo = sim_prices_geo.mean()
std_geo = sim_prices_geo.std()

# calculate the theoretical mean and standard deviation of P(t) for Geometric Brownian Motion
theory_mean_geo = P_0 * math.exp(0.5 * sigma**2)
theory_std_geo = P_0 * math.sqrt(math.exp(2 * sigma**2) - math.exp(sigma**2))

# plot the histograms of the simulated prices for all three processes
fig, axs = plt.subplots(3)
sns.distplot(sim_prices_cla, ax=axs[0])
sns.distplot(sim_prices_ari, ax=axs[1])
sns.distplot(sim_prices_geo, ax=axs[2])
axs[0].set_title("Classic Brownian Motion")
axs[1].set_title("Arithmetic Return System")
axs[2].set_title("Geometric Brownian Motion")
plt.show()
