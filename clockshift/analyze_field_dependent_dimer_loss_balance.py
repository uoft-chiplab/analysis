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

files = ["2024-10-24_D_e","2024-10-24_E_e", "2024-10-24_F_e", "2024-10-24_G_e",
		 "2024-10-24_I_e","2024-10-24_K_e", "2024-10-30_B_e"]
spin_names = ['c5', 'c9']
# spin_names = ['two2D_int1', 'two2D_int2']
fields = [201,202.5, 201.75, 202.3, 202.14,201.5, 209]
x0_guesss = [42.95, 43.3, 43.15, 43.3, 43.2, 43.1, 45]

fig, axs = plt.subplots(2, figsize=(10,6))
xname = 'field'
xlabel = xname
axs[0].set(xlabel=xlabel, ylabel='Gaussian fit amplitude')
axs[1].set(xlabel=xlabel, ylabel='c5 amp/c9 amp', ylim=[1,2])
# axs[2].set(xlabel=xlabel, ylabel='max loss')
fig.suptitle(files[0][:-4] + " ac dimer spin loss comparison")
c5_amp=0 
c9_amp = 0
for i, file in enumerate(files):
	filename = file+".dat"
	field = fields[i]
	x0_guess = x0_guesss[i]
	for spin in spin_names:
		yname = spin + '_loss'
		if spin == spin_names[0]:
			marker = markers[0]
			dark_color = dark_colors[0]
			light_color = light_colors[0]
		elif spin == spin_names[1]:
			marker = markers[1]
			dark_color = dark_colors[1]
			light_color = light_colors[1]
			
		run = Data(filename)
		bg = run.data[(run.data['VVA'] == 0)][spin].mean()
		run.data = run.data.loc[run.data['field']==field]
		run.data['loss'] = bg - run.data[spin]
		run.data = run.data.loc[run.data['freq'] > 42.0]
# 		maxloss = run.data.iloc[run.data.loss.idxmax]['loss'].mean()
# 		maxloss_std = run.data.iloc[run.data.loss.idxmax]['loss'].std()
		run.fit(Gaussian, names=['freq', 'loss'], guess=[1000, x0_guess, 0.01, 0])
		axs[0].errorbar(field,run.popt[0], run.perr[0], label=spin, ecolor=dark_color, mfc=light_color, mec=dark_color, marker=marker)
# 		axs[2].errorbar(field, maxloss, maxloss_std,label=spin, ecolor=dark_color, mfc=light_color, mec=dark_color, marker=marker )
		if spin == spin_names[0]:
			c5_amp = run.popt[0]
			c5_amp_e = run.perr[0]
		elif spin == spin_names[1]:
			c9_amp = run.popt[0]
			c9_amp_e = run.perr[0]
	
	ratio = c5_amp/c9_amp
	ratio_err = ratio*np.sqrt((c5_amp_e/c5_amp)**2 + (c9_amp_e/c9_amp)**2)
	axs[1].errorbar(field, c5_amp/c9_amp, ratio_err, ecolor = dark_colors[2],mfc=light_colors[2], mec=dark_colors[2], marker=markers[2])
		
# axs[0].legend()
fig.tight_layout()
plt.show()
	