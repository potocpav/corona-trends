
import scipy.optimize
import numpy as np
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

raw = list(csv.reader(open('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')))
raw_deaths = list(csv.reader(open('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')))

by_country = lambda country: np.sum([[int(i) for i in x[5:]] for x in raw if x[1] == country], axis=0)
deaths_by_country = lambda country: np.sum([[int(i) for i in x[5:]] for x in raw_deaths if x[1] == country], axis=0)

all_countries = np.array(list(set(x[1] for x in raw[1:])))
data = np.array([by_country(c) for c in all_countries])
countries = all_countries[data[:,-1] > 500]
xs = np.arange(len(by_country("Czechia")))

month, day, year = [int(i) for i in raw[0][-1].split('/')]

czech_cases = by_country("Czechia")
korean_cases = by_country("Korea, South")
italian_cases = by_country("Italy")
us_cases = by_country("US")

# %% Logistic curve through Italy

# country = "Czechia"

# %% Loglog minutephysics plot

q = 1.125 ** 7

all_countries[(data[:,-1] > 4e3) & (data[:,-1] < 8e3) & (data[:,-1] - data[:,-7] > 1e3) & (data[:,-1] - data[:,-7] < 1e4)]


plt.figure(figsize=(12,8))

_ = plt.loglog(data[:,7:].T, (data[:,7:] - data[:,:-7]).T, color='lightgrey')
_ = plt.loglog([], [], color='lightgrey', label='Countries')
plt.loglog(czech_cases[7:].T, (czech_cases[7:] - czech_cases[:-7]).T, color='red', label='Czechia')
# plt.loglog(by_country("Norway")[7:].T, (by_country("Norway")[7:] - by_country("Norway")[:-7]).T, color='blue', label='Norway')
plt.plot([q, q * 1e6], np.array([q - 1, (q-1) * 1e6]), color='gray', linestyle='dashed', label='1/8 Spread Rate')
plt.grid('minor')
plt.xlabel('Total Cumulative Cases')
plt.title('Trends of Countries with > 500 Cases')
plt.ylabel('New Weekly Cases')
plt.legend()
plt.xlim(1e1, 1e6)
plt.ylim(1e1, 1e6)
plt.savefig('Trend2.png', dpi=90)

# %%

growth_rate = (data[:,-1] / data[:,-7]) ** (1/6)
predictions = data[:,-1,np.newaxis] * growth_rate[np.newaxis].T ** [np.arange(20)]
predictions.shape
top_ix = np.argsort(-data[:,-1])[:8]
all_countries[top_ix]
plt.xlim(50, 80)
# plt.semilogy()
plt.ylim(1e3, 5e5)
_ = plt.plot(data[top_ix].T)
_ = plt.plot(np.arange(70, 70 + 20), predictions[top_ix].T, linestyle='dotted')

# %% Fit a distribution to data from paper

weibull = lambda x, k, l: (k/l)*(x/l)**(k-1)*np.exp(-(x/l)**k)
import scipy.special

g = lambda p: [p[1] * scipy.special.gamma(1 + 1/p[0]) - 8.6, p[1] * np.log(2) ** (1/p[0]) - 6.7]
k, l = scipy.optimize.least_squares(g, [2, 7], jac='3-point').x
scipy.optimize.least_squares(g, [2, 7], jac='3-point', method='dogbox')

g([2,7])

x = np.arange(0.1, 20, 1)
pdf =  weibull(x, k, l)
np.sum(pdf * x), x[np.argmax(pdf)]
plt.plot(pdf)

f = lambda k: 6.7 * np.log(2) ** (-1/k) * scipy.special.gamma(1 + 1/k) - 8.6
plt.plot(np.linspace(1, 2, 100), f(np.linspace(1, 2, 100)))


plt.ion()
plt.plot()
plt.show()
# %%

xs = np.arange(106)

start_date = 31
natural_spread = 0.4 # daily multiplier
quarantine_date = 55
quarantine_spread = 0.15
recovery_prob = 0.8
recovery_lag = 7

new_cases = np.zeros(len(xs))
active_cases = np.zeros(len(xs))
recovered_cases = np.zeros(len(xs))
active_cases[0] = (1 + natural_spread) ** (-start_date) * 0.5
for x in xs[1:]:
    spread = natural_spread if x <= quarantine_date else quarantine_spread
    new_cases[x] = active_cases.sum() * spread
    active_cases[x] = new_cases[x]
    # Recovery
    # print(active_cases)
    recovered_cases[x] = np.sum(active_cases[:x-recovery_lag] * recovery_prob)
    active_cases[:max(0, x-recovery_lag)] = active_cases[:max(0, x-recovery_lag)] * (1 - recovery_prob)

czech_cases[1:] - czech_cases[:-1]
plt.figure(figsize=(12,8))
plt.subplot(221)
# np.append(czech_cases, [np.nan] * (len(xs) - len(czech_cases))),
plt.semilogy(xs[60:], np.column_stack([np.cumsum(new_cases), np.cumsum(recovered_cases)])[60:])
plt.subplot(222)
plt.semilogy(xs[60:], np.column_stack([new_cases])[60:])
