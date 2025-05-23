# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:14:49 2025

@author: Chip lab
"""

import numpy as np
from data_class import Data
from library import styles, colors
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(x, a, b):
	return a*x+b


file = '2025-04-15_E_e_ODT1=' #"2025-04-11_J_e.dat"

ODTlist = [
	0.1,
		   0.3,
		   0.07
		   ]

fig, axes = plt.subplots(1,3, figsize=(12,4))
axs = axes.flatten()

axs[0].set(xlabel="refmean", ylabel="ODmean", ylim=[0.0025, 0.022])
axs[1].set(xlabel="refmean", ylabel="c9")
axs[2].set(xlabel="refmean", ylabel="ODmax")

for idx, i in enumerate(ODTlist): 
	filename = file + str(i) + '.dat'
	run = Data(filename)
	df = run.data
	
	fit_df = df.loc[df.refmean>1000]
	
	x = fit_df['refmean']
	y = fit_df['ODmean']
	
	popt, pcov = curve_fit(linear, x, y)
	xs = np.linspace(0, max(x), 100)
	
	print("correction is ", linear(0, *popt)/linear(2000, *popt))
	
	
	ax = axs[0]
	label = f'ODT1={i}'
	ax.plot(df['refmean'], df['ODmean'],**styles[idx],label=label)
	ax.plot(xs, linear(xs, *popt), '--', color=colors[idx])
	ax.legend()
	
	ax = axs[1]
	ax.plot(df['refmean'], df['c9'],**styles[idx])
	
	ax = axs[2]
	ax.plot(df['refmean'],df['ODmax'],**styles[idx])


fig.suptitle(file[:-8])
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