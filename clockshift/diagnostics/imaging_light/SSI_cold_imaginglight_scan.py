# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:14:49 2025

@author: Chip lab
"""
import os 
import sys
parentdir = os.path.dirname(__file__)
backonedir = os.path.dirname(parentdir)
onemore = os.path.dirname(backonedir)
if onemore not in sys.path:
	sys.path.append(onemore)
import numpy as np
from data_class import Data
from library import styles, colors
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(x, a, b):
	return a*x+b


file = '2025-04-15_F_e'
bg_file = '2025-04-15_G_e'

fig, axes = plt.subplots(1,3, figsize=(12,4))
axs = axes.flatten()

axs[0].set(xlabel="refmean", ylabel="ODmean")
axs[1].set(xlabel="refmean", ylabel="c9")
axs[2].set(xlabel="refmean", ylabel="c5 - bgc5")

idx = 0


filename = file + '.dat'
run = Data(filename)
run.group_by_mean('KAM')
df = run.avg_data


bg_filename = bg_file + '.dat'
bg_run = Data(bg_filename)
bg_run.group_by_mean('KAM')
bg_df = bg_run.avg_data

# subtract bg c5 counts
df['c5'] = df['c5'] - bg_df['c5']


fit_df = df.loc[df.refmean>1000]

x = fit_df['refmean']
y = fit_df['ODmean']

popt, pcov = curve_fit(linear, x, y)
xs = np.linspace(0, max(x), 100)

print("full ROI correction is ", linear(0, *popt)/linear(2000, *popt))

ax = axs[0]
ax.errorbar(df['refmean'], df['ODmean'], yerr=df['em_ODmean'],**styles[idx])
ax.plot(xs, linear(xs, *popt), '--', color=colors[idx])
ax.legend()

y = fit_df['c9']

popt, pcov = curve_fit(linear, x, y)
xs = np.linspace(0, max(x), 100)

print("c9 correction is ", linear(0, *popt)/linear(2000, *popt))

ax = axs[1]
ax.errorbar(df['refmean'], df['c9'], yerr=df['em_c9'],**styles[idx])
ax.plot(xs, linear(xs, *popt), '--', color=colors[idx])

y = fit_df['c5']
yerr = fit_df['em_c5']
popt, pcov = curve_fit(linear, x, y, sigma=yerr)
xs = np.linspace(0, max(x), 100)

print("c5 correction is ", linear(0, *popt)/linear(2000, *popt))

ax = axs[2]
ax.errorbar(df['refmean'], df['c5'], yerr=df['em_c5'], **styles[idx])
ax.plot(xs, linear(xs, *popt), '--', color=colors[idx])


fig.suptitle(file)
fig.tight_layout()

# 	ax = axs[0]
# 	label = 'ODmax={:.2f}'.format(ODmax)
# 	ax.plot(sub_df['refmean'], sub_df['ODmean'], **styles[i], label=label)
# 	
# 	ax = axs[1]
# 	label = 'TOF='+str(TOF)+'ms'
# 	ax.plot(sub_df['refmean'] - sub_df['atmean'], sub_df['ODmean'], **styles[i], label=label)
# 	
# 	ax = axs[2]
# 	ax.plot(sub_df['refmean'], sub_df['ODmax'], **styles[i])

# for ax in axs[:-1]:
# 	ax.legend()
# fig.tight_layout()

# plt.show()