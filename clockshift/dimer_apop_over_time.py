# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:44:59 2024

@author: coldatoms
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_class import Data
from fit_functions import Gaussian
from scipy.optimize import curve_fit
from library import plt_settings, dark_colors, light_colors, markers
from cycler import Cycler

styles = Cycler([{'mec':dark_color, 'color':dark_color, 'mfc':light_color, 
				  'marker':marker} for marker, dark_color, light_color in \
					 zip(markers, dark_colors, light_colors)])
	
files = ['2024-11-01_D_UHfit.dat', '2024-11-01_B_UHfit.dat']

fig, ax = plt.subplots()
ax.set(xlabel='time [ms]', ylabel='N')
# ax[1].set(xlabel='time [ms]', ylabel='Ndimer')
labels = ['bg','dimer']
for i, file in enumerate(files):
	run = Data(file)
	label=labels[i]
	run.data['abstime'] = run.data['time']+0.050
	run.group_by_mean('abstime')
	
	xname = 'abstime'
	yname = 'N'
	yerrname = 'em_' + yname
	x = run.avg_data[xname]
	y = run.avg_data[yname]
	yerr = run.avg_data[yerrname]
			
	ax.errorbar(x, y, yerr, label=label)
	
	
ax.legend()


