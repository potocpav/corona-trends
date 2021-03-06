
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

# czech_cases = by_country("Czechia")
# korean_cases = by_country("Korea, South")
# italian_cases = by_country("Italy")
# us_cases = by_country("US")

# %% Logistic curve through Italy

# country = "Czechia"


def create_chart(country):
    cases = by_country(country)
    deaths = deaths_by_country(country)

    def f(x, a, c, d):
        return a / (1. + np.exp(-c * (x - d)))

    x = np.arange(len(cases))
    x_trend = np.arange(101)

    try:
        params, covs = scipy.optimize.curve_fit(f, x, cases, p0=[1e5, 0.2, 60])
    except RuntimeError:
        print(f"Failed curve fit for {country}.")
        return False
    fs = np.array([f(x_trend, *p) for p in np.random.multivariate_normal(params, covs, size=10000)])
    q1 = np.quantile(fs, 0.05, 0)
    q2 = np.quantile(fs, 0.5, 0)
    q3 = np.quantile(fs, 0.95, 0)

    disp_confidence = q1[-1] > 0 and q3[-1] > 0

    fig = plt.figure(figsize=(15, 15))
    plt.suptitle(country)

    ax = plt.subplot(221)
    plt.title('Total confirmed cases')
    plt.bar(x, cases / 1000, 1, color='grey', label=country)
    plt.plot(x_trend, q2 / 1000, linestyle='dashed', color='black', label='Logistic Regression')
    if disp_confidence:
        plt.fill_between(x_trend, q1 / 1000, q3 / 1000, color='grey', alpha=0.3, label="0.95 Confidence")
    plt.ylim(0, q3[-1] * 1.1 / 1000)
    plt.xlim(0, 100)
    plt.xlabel('Time [days]')
    plt.ylabel('Cases [×1000]')
    plt.hlines(q2[-1] / 1000, 0, 100, color='grey', linestyle='dotted', label=f'{int(q2[-1]/1000)}k Cases')
    plt.legend()

    dyq1 = q1[1:] - q1[:-1]
    dyq2 = q2[1:] - q2[:-1]
    dyq3 = q3[1:] - q3[:-1]
    plt.subplot(222)
    plt.title('New confirmed cases per day')
    plt.bar(x[1:], cases[1:] - cases[:-1], 1, color='grey', label=country)
    plt.plot(x_trend[1:], dyq2, linestyle='dashed', color='black', label='Normal Distribution')
    if disp_confidence:
        plt.fill_between(x_trend[1:], dyq1, dyq3, color='grey', alpha=0.3, label="0.95 Confidence")
    plt.xlim(0, 100)
    plt.ylabel('New Cases [-]')
    plt.xlabel('Time [days]')
    plt.legend()

    # Use the median prediction for all the quantiles
    delay = np.arange(1, 16)
    confirm_rate_q1 = np.clip(q2[len(x)-delay] / (deaths[-1] / 0.002), 0, 1)
    confirm_rate_q2 = np.clip(q2[len(x)-delay] / (deaths[-1] / 0.006), 0, 1)
    confirm_rate_q3 = np.clip(q2[len(x)-delay] / (deaths[-1] / 0.013), 0, 1)

    ax = plt.subplot(223)
    plt.title('% of Cases Confirmed')
    ax.grid('major')
    ax.plot(delay, confirm_rate_q2 * 100, color='black', label='Mean')
    if disp_confidence:
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
    if disp_confidence:
        ax.fill_between(delay, q2[-1] / confirm_rate_q1 / 1000, q2[-1] / confirm_rate_q3 / 1000, color='grey', alpha=0.3, label="0.95 Confidence")
    ax2.set_ylim(ax.get_ylim()[0] / 100, ax.get_ylim()[1] / 100)
    ax.set_ylabel('Number Infected up to Now [×1000]')
    ax.set_xlabel('Death Delay from Confirmation [days]')
    ax2.set_ylabel("Inevitable Deaths [×1000]")
    ax.set_xlim(1, 14)
    ax.legend()

    os.makedirs(f'stats/{month:02d}-{day:02d}', exist_ok=True)
    plt.savefig(f'stats/{month:02d}-{day:02d}/{country}.svg', dpi=90)
    plt.close(fig)
    return True

# %%

plt.ioff()
template = open('stats/index.html.in', 'r').read()
items_string = ""
for country in sorted(countries):
    print(country)
    if create_chart(country):
        items_string += f"""<li><a href="{month:02d}-{day:02d}/{country}.svg">{country}</a></li>\n"""
index = template.format(last_update=f"20{year:02d}-{month:02d}-{day:02d}", items=items_string)
with open('stats/index.html', 'w') as f:
    f.write(index)
