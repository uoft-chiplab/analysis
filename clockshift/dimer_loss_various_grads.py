# -*- coding: utf-8 -*-
"""
@author: coldatoms
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_class import Data
from fit_functions import Gaussian
from scipy.optimize import curve_fit
from library import plt_settings, dark_colors, light_colors, markers


files = ["2024-10-28_L_e.dat", "2024-10-28_M_e.dat"]
labels = ['jump', 'ramp']

spin_names = ['c5', 'c9']
grads = [0]

fig, axes = plt.subplots(1, 3, figsize=(10,3))
axs = axes.flatten()
xname = 'y gradient'
xlabel = xname
axs[0].set(xlabel=xlabel, ylabel='fraction95')
axs[1].set(xlabel=xlabel, ylabel='Loss')
axs[2].set(xlabel=xlabel, ylabel='c5 amp/c9 amp')
# axs[2].set(xlabel=xlabel, ylabel='max loss')
fig.suptitle("jump or ramp ac dimer spin loss comparison")

for i, file in enumerate(files):
	filename = file
	
	run = Data(filename)
	
	for spin in spin_names:
		yname = spin + '_loss'
		
		run.data[spin+'_bg'] = run.data[(run.data['VVA'] == 0)][spin].mean()
		run.data[spin+'_bg_std'] = run.data[(run.data['VVA'] == 0)][spin].std()
		run.data[spin+'_loss'] = run.data[spin+'_bg'] - run.data[spin]
		
	run.data['ratio'] = run.data['c5_loss']/run.data['c9_loss']
	run.group_by_mean('VVA')
	run.avg_data = run.avg_data.drop(run.avg_data.loc[run.avg_data.VVA == 0].index)
	
	run.avg_data['ratio'] = run.avg_data['c5_loss']/run.avg_data['c9_loss']
	run.avg_data['em_ratio'] = run.avg_data['ratio']*np.sqrt((run.avg_data['em_c5_loss']/run.avg_data['c5_loss'])**2 +\
 									  (run.avg_data['em_c9_loss']/run.avg_data['c9_loss'])**2)
		
	for j, spin in enumerate(spin_names):
		if i == 0:
			label = spin
		else:
			label = None
		
		axs[1].errorbar(i, run.avg_data[spin+'_loss'], run.avg_data['em_'+spin+'_loss'],
				  ecolor=dark_colors[j], mfc=light_colors[j], mec=dark_colors[j], 
				  marker=markers[i], label=label)
		
	axs[0].errorbar(i, run.avg_data['fraction95'], run.avg_data['em_fraction95'],
			  ecolor=dark_colors[3], mfc=light_colors[3], mec=dark_colors[3], 
			  marker=markers[i])
	
	axs[2].errorbar(i, run.avg_data['ratio'], run.avg_data['em_ratio'], ecolor=dark_colors[2],
			  mfc=light_colors[2], mec=dark_colors[2], marker=markers[i], label=labels[i])

	
axs[1].legend()
axs[2].legend()
fig.tight_layout()
plt.show()