# -*- coding: utf-8 -*-
"""
@author: coldatoms
"""
import matplotlib.pyplot as plt
import pandas as pd

from data_class import Data
from library import plt_settings, dark_colors, light_colors, markers

files = ["2024-10-22_N_e"]
spins = ["75"] 
spin_names = ['c5', 'c9']

fig, axs = plt.subplots(2, 3, figsize=(10,6))
axes = axs.flatten()
xname = 'ODT1'
xlabel = xname

axes[0].set(xlabel=xlabel, ylabel='fraction95')
axes[1].set(xlabel=xlabel, ylabel='loss')
axes[2].set(xlabel=xlabel, ylabel='c5/c9 loss ')
axes[3].set(xlabel=xlabel, ylabel='c5+c9')
axes[3].set(xlabel=xlabel, ylabel='c5+c9 loss')

fig.suptitle(files[0][:-4] + " ac dimer spin loss comparison")

i = 0
for file, in zip(files):
	marker = markers[0]
	dark_color = dark_colors[0]
	light_color = light_colors[0]
	filename = file+".dat"
	
	run = Data(filename)
	data = run.data.loc[run.data.VVA != 0].groupby(xname).mean().reset_index()
	bg = run.data.loc[run.data.VVA == 0].groupby(xname).mean().reset_index()
	
	c5_loss =  bg[spin_names[0]] - data[spin_names[0]]
	
	c9_loss = bg[spin_names[1]] - data[spin_names[1]]
	
	axes[i].plot(data[xname], data.fraction95, mfc=light_color, mec=dark_color, 
			  marker=marker, label='')
	i += 1
	axes[i].plot(data[xname], c5_loss, mfc=light_color, mec=dark_color, 
			  marker=marker, label='c5')
	axes[i].plot(data[xname], c9_loss, mfc=light_colors[1], mec=dark_colors[1], 
			  marker=markers[1], label='c9')
	i += 1
	axes[i].plot(data[xname], c5_loss/c9_loss, mfc=light_color, mec=dark_color, 
			  marker=marker)
	i += 1
	axes[i].plot(data[xname], data[spin_names[0]]+data[spin_names[1]], 
			  mfc=light_color, mec=dark_color, marker=marker)
	
	i += 1
	axes[i].plot(data[xname], c5_loss+c9_loss, 
			  mfc=light_color, mec=dark_color, marker=marker)

# axes[0].legend()
axes[1].legend()
fig.tight_layout()
plt.show()
	