# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:32:34 2025

@author: Chip lab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_class import Data
from library import styles, colors
from scipy.optimize import curve_fit
from bootstrap_fit import bootstrap_fit, dist_stats

def exp_decay(x, A, tau, C):
	return A*np.exp(-x/tau) + C

file = '2025-04-10_H_e.dat'

xname = 'time'
ff = 0.88
fit_func = exp_decay

run = Data(file)

# correct the data
run.data['c9'] = run.data['c9'] * ff
run.data[xname] = run.data[xname] + 21.6-9.0 # cause we shaved off 11ms-2.0ms

# select signal and bg dfs
df = run.data.loc[run.data.VVA > 0]
bg_df = run.data.loc[run.data.VVA == 0]

# compute transfer
df['transfer'] = (df['c5'] - bg_df['c5'].mean())/df['c9']
df['N'] = (df['c5'] - bg_df['c5'].mean()) + df['c9']

# group by xname
mean = df.groupby([xname]).mean().reset_index()
sem = df.groupby([xname]).sem().reset_index().add_prefix("em_")
std = df.groupby([xname]).std().reset_index().add_prefix("e_")
avg_df = pd.concat([mean, std, sem], axis=1)

###
### plot
###
fig, axes = plt.subplots(2,2, figsize=(8,8))
axs = axes.flatten()

x = avg_df['time']
xs = np.linspace(0, x.max(), 100)

#
# c5
#

ax = axs[0]
y = avg_df['c5']
yerr = avg_df['em_c5']
ax.set(xlabel='Time (ms)', ylabel='c atoms')

popt, pcov = curve_fit(fit_func, x, y, sigma=yerr, p0=[y.max(), 50, 0])
perr = np.sqrt(np.diag(pcov))

tau = popt[1]
e_tau = perr[1]

print("tau is ", tau, e_tau)

loss_corr = (fit_func(0, *popt)-fit_func(26.6, *popt))/fit_func(0, *popt)

ax.plot(xs, fit_func(xs, *popt), '-', color=colors[0], 
		label='loss_corr={:.2f}'.format(loss_corr))
ax.errorbar(x, y, yerr=yerr, **styles[0])
ax.legend()

#
# c9
#

ax = axs[1]
y = avg_df['c9']
yerr = avg_df['em_c9']
ax.set(xlabel='Time (ms)', ylabel='a atoms')

popt, pcov = curve_fit(fit_func, x, y, sigma=yerr, p0=[y.max(), 50, 0])
perr = np.sqrt(np.diag(pcov))

ax.plot(xs, fit_func(xs, *popt), '-', color=colors[1])
ax.errorbar(x, y, yerr=yerr, **styles[1])

#
# transfer
#

ax = axs[2]
y = avg_df['transfer']
yerr = avg_df['em_transfer']
ax.set(xlabel='Time (ms)', ylabel='Transfer')
ax.errorbar(x, y, yerr=yerr, **styles[0])

ax = axs[3]
y = avg_df['N']
yerr = avg_df['em_N']
ax.set(xlabel='Time (ms)', ylabel='N')
ax.errorbar(x, y, yerr=yerr, **styles[0])

fig.tight_layout()

plt.show()

###
### Bootstrapping
###

waittime = 26.6

xx = np.array(df['time'])
yy = np.array(df['c5'])

p0 = [max(yy), 50, 0]

popts, popt_stats_dicts = bootstrap_fit(fit_func, xx, yy, p0=p0, trials=1000)

# tau stats
tau_median = popt_stats_dicts[1]['median']
tau_u = popt_stats_dicts[1]['upper']
tau_l = popt_stats_dicts[1]['lower']

# get medians for use
BS_popt = [stats_dict['median'] for stats_dict in popt_stats_dicts]

# amplitude cancels here, but offset does not. Ignore error in offset
zerotime = fit_func(0, *BS_popt)
losscorr_median = fit_func(waittime, *BS_popt)/zerotime

losscorr_u = fit_func(waittime, BS_popt[0], tau_u, BS_popt[2])/zerotime
losscorr_l = fit_func(waittime, BS_popt[0], tau_l, BS_popt[2])/zerotime

print("Bootstrap results:")
tau_eu = tau_u - tau_median
tau_el = tau_median - tau_l		
print(f'tau is {tau_median:.0f}+{tau_eu:.0f}-{tau_el:.0f}')
losscorr_eu = losscorr_u - losscorr_median
losscorr_el = losscorr_median - losscorr_l		
print(f'losscorr is {losscorr_median:.2f}+{losscorr_eu:.2f}-{losscorr_el:.2f}')

losscorr_dist = []
for popt in popts:
	zerotime_pop = fit_func(0, *popt)
	waittime_pop = fit_func(waittime, *popt)
	losscorr = waittime_pop/zerotime_pop
	losscorr_dist.append(losscorr)
    
losscorr_stats = dist_stats(losscorr_dist)
losscorr_median = losscorr_stats['median']
losscorr_u = losscorr_stats['upper']
losscorr_l = losscorr_stats['lower']

losscorr_eu = losscorr_u - losscorr_median
losscorr_el = losscorr_median - losscorr_l
print("Proper error analysis of offset param:")
print(f'losscorr is {losscorr_median:.2f}+{losscorr_eu:.2f}-{losscorr_el:.2f}')

plt.figure(figsize=(6,4))
plt.hist(losscorr_dist, bins=50)
plt.ylabel("Occurances")
plt.xlabel("loss correction")
plt.show()
