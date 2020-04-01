
import csv
import numpy as np
from matplotlib import pyplot as plt

raw = list(csv.reader(open('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')))
raw_deaths = list(csv.reader(open('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')))

by_country = lambda country: np.sum([[int(i) for i in x[5:]] for x in raw if x[1] == country], axis=0)
deaths_by_country = lambda country: np.sum([[int(i) for i in x[5:]] for x in raw_deaths if x[1] == country], axis=0)
country = "China"
countries = np.array(list(set(x[1] for x in raw[1:])))
data = np.array([by_country(c) for c in countries])
countries = countries[data[:,-1] > 500]
xs = np.arange(len(by_country("Czechia")))

czech_cases = by_country("Czechia")
korean_cases = by_country("Korea, South")
italian_cases = by_country("Italy")
us_cases = by_country("US")

# %% Logistic curve through Italy

import scipy.optimize
import numpy as np

countries
country = "Czechia"
cases = by_country(country)
deaths = deaths_by_country(country)

def f(x, a, c, d):
    return a / (1. + np.exp(-c * (x - d)))

x = np.arange(len(cases))
x_trend = np.arange(100)

params, covs = scipy.optimize.curve_fit(f, x, cases, p0=[1e5, 0.2, 60])
fs = np.array([f(x_trend, *p) for p in np.random.multivariate_normal(params, covs, size=10000)])
q1 = np.quantile(fs, 0.05, 0)
q2 = np.quantile(fs, 0.5, 0)
q3 = np.quantile(fs, 0.95, 0)

plt.figure(figsize=(15, 15))

ax = plt.subplot(221)
plt.title('Total confirmed cases')
plt.bar(x, cases, 1, color='grey', label=country)
plt.plot(x_trend, q2, linestyle='dashed', color='black', label='Logistic Regression')
plt.fill_between(x_trend, q1, q3, color='grey', alpha=0.3, label="0.95 Confidence")
plt.ylim(0, q3[-1] * 1.1)
plt.xlim(0, 100)
plt.xlabel('Time [days]')
plt.ylabel('Number of people [-]')
plt.hlines(q2[-1], 0, 100, color='grey', linestyle='dotted', label=f'{int(q2[-1]/1000)}k Cases')
plt.legend()

dyq1 = q1[1:] - q1[:-1]
dyq2 = q2[1:] - q2[:-1]
dyq3 = q3[1:] - q3[:-1]
plt.subplot(222)
plt.title('New confirmed cases per day')
plt.bar(x[1:], cases[1:] - cases[:-1], 1, color='grey', label=country)
plt.plot(x_trend[1:], dyq2, linestyle='dashed', color='black', label='Normal Distribution')
plt.fill_between(x_trend[1:], dyq1, dyq3, color='grey', alpha=0.3, label="0.95 Confidence")
plt.xlim(0, 100)
plt.ylabel('Number of people [-]')
plt.xlabel('Time [days]')
plt.legend()


death_rate = 0.01
confirm_rate_q1 = np.minimum(1, q1[len(x)-delay] / (deaths[-1] / 0.002))
confirm_rate_q2 = np.minimum(1, q2[len(x)-delay] / (deaths[-1] / 0.006))
confirm_rate_q3 = np.minimum(1, q3[len(x)-delay] / (deaths[-1] / 0.013))

ax = plt.subplot(223)
delay = np.arange(1, 16)
plt.title('% of Cases Confirmed')
ax.grid('major')
ax.plot(delay, confirm_rate_q2 * 100, color='black', label='Mean')
plt.fill_between(delay, confirm_rate_q1 * 100, confirm_rate_q3 * 100, color='grey', alpha=0.3, label="0.95 Confidence")
ax.set_ylabel('Infections Confirmed [%]')
ax.set_xlim(1, 14)
plt.xlabel('Death Delay from Confirmation [days]')
plt.legend()

ax = plt.subplot(224)
delay = np.arange(1, 16)
ax2 = ax.twinx()
plt.title('Total Infected & Deaths')
ax.grid('major')
ax.plot(delay, q2[-1] / confirm_rate_q2 / 1000, color='black', label='Mean')
ax.fill_between(delay, q3[-1] / confirm_rate_q1 / 1000, q1[-1] / confirm_rate_q3 / 1000, color='grey', alpha=0.3, label="0.95 Confidence")
ax2.set_ylim(ax.get_ylim()[0] / 100, ax.get_ylim()[1] / 100)
ax.set_ylabel('Number Infected up to Now [×1000]')
ax.set_xlabel('Death Delay from Confirmation [days]')
ax2.set_ylabel("Inevitable Deaths [×1000]")
ax.set_xlim(1, 14)
ax.legend()

plt.savefig(f'stats/04-01/{country}_stats.svg', dpi=90)


# %%

q = 1.125 ** 7


plt.figure(figsize=(12,8))

_ = plt.loglog(data[:,7:].T, (data[:,7:] - data[:,:-7]).T, color='lightgrey')
_ = plt.loglog([], [], color='lightgrey', label='Countries')
plt.loglog(czech_cases[7:].T, (czech_cases[7:] - czech_cases[:-7]).T, color='red', label='Czechia')
plt.plot([q, q * 1e5], np.array([q - 1, (q-1) * 1e5]), color='gray', linestyle='dashed', label='1/8 Spread Rate')
plt.grid('major')
plt.xlabel('Total Cumulative Cases')
plt.title('Trends of Countries with > 500 Cases')
plt.ylabel('New Weekly Cases')
plt.legend()
plt.xlim(1e1, 1e6)
plt.ylim(1e1, 1e6)
plt.savefig('Trend2.png', dpi=90)

# %%

plt.semilogy(by_country("China"))

plt.semilogy(xs, data.T)

czech_cases[1:] - czech_cases[:-1]
plt.plot(czech_cases)
plt.semilogy(czech_cases)

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
