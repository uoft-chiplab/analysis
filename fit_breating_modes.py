# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:43:33 2024

@author: coldatoms
"""

from data_class import Data
import matplotlib.pyplot as plt
import numpy as np
from library import plt_settings, colors, markers, tintshade, tint_shade_color
# from fit_functions import TrapFreq2
from scipy.optimize import curve_fit
import pandas as pd

def TrapFreq2(data):
	"""
	Returns: A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C 
	"""
	
	param_names = ['Amplitude','Decay Time [ms]','Freq [kHz]','Center','Offset']
	guess = [200, 2, .4, 0, 550]
	
	def TrapFreq2(x, A, b, l, x0, C):
		return A*np.exp(-x/b)*(np.cos(2*np.pi*l * (x - x0))) +  C 
	return TrapFreq2, guess, param_names

def TrapFreq1(data):
	"""
	Returns: A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C 
	"""
	
	param_names = ['Amplitude', 'Position [px]','Freq [kHz]','Center']
	guess = [20, 500, .2, 0]
	
	def TrapFreq(x, A, C, l, x0):
		return A*(np.cos(2*np.pi*l * (x - x0))) +  C 
	return TrapFreq, guess, param_names

# file1 = "2024-05-13_B_e.dat"
file2 = "2024-05-13_G_e.dat"
# file3 = "2024-05-13_D_e.dat"

ms_param = 'y2 piezo'
y_name = 'G_Tx'
x_name = 'time'
fit_func = TrapFreq2

popt_list = []
perr_list = []

# dat1 = Data(file1)
# dat1.data[ms_param] = 75*dat1.data[x_name]/dat1.data[x_name]

# dat3 = Data(file3)
# dat3.data[ms_param] = 135*dat3.data[x_name]/dat3.data[x_name]

dat2 = Data(file2)
df = dat2.data
# df = pd.concat([dat1.data, dat2.data, dat3.data])
df.sort_values(ms_param, inplace=True)

drop_freqs = []
df = df.drop(df.loc[df[ms_param].isin(drop_freqs)].index.values)

for piezo in df[ms_param].unique():
	df1 = df.loc[df[ms_param]==piezo]
	xx = df1[x_name]
	yy = df1[y_name]
	func, guess, names = fit_func([x_name, y_name])
	popt, pcov = curve_fit(func, xx, yy, p0=guess)
	perr = np.sqrt(np.diag(pcov))
	popt_list.append(popt)
	perr_list.append(perr)
	
plt.rcParams.update(plt_settings)
plt.rcParams.update({"figure.figsize": [12,8],
					 "legend.fontsize": 12})
fig, axs = plt.subplots(2,2)
num = 500
		
###
### Cloud size oscillations
###
ax = axs[0,0]
xlabel = x_name
ylabel = "Delta "+y_name
ax.set(xlabel=xlabel, ylabel=ylabel)

piezos = df[ms_param].unique()
params = zip(piezos, popt_list, perr_list, colors, markers)

for piezo, popt, perr, color, marker in params:
	light_color = tint_shade_color(color, amount=1+tintshade)
	dark_color = tint_shade_color(color, amount=1-tintshade)
	
	df1 = df.loc[df[ms_param]==piezo]
	xx = df1[x_name]
	yy = df1[y_name]
	label_params = [piezo, popt[2], perr[2]*1e3, popt[1], perr[1]]
	label = ms_param+r'={}, freq={:.3f}({:.0f}) kHz, tau={:.1f}({:.1f}) ms'.format(*label_params)
	ax.plot(xx, yy-popt[-1]*np.ones(len(yy)), linestyle='', color=color, marker=marker, label=label, markeredgewidth=2,
		 markerfacecolor=light_color, markeredgecolor=dark_color)
	
	xs = np.linspace(min(xx), max(xx), num)
	ax.plot(xs, func(xs, *popt)-popt[-1], '-', color=color)

	
###
### param1
###
ax = axs[1,0]
xlabel = ms_param
ylabel = names[1]
ax.set(xlabel=xlabel, ylabel=ylabel)
color=colors[1]
light_color = tint_shade_color(color, amount=1+tintshade)
dark_color = tint_shade_color(color, amount=1-tintshade)
ax.errorbar(piezos, np.array(popt_list)[:,1], yerr=np.array(perr_list)[:,1], fmt='o', markeredgewidth=2, color=color,
		 markerfacecolor=light_color, markeredgecolor=dark_color)

###
### Freq
###
ax = axs[0,1]
xlabel = ms_param
ylabel = names[2]
ax.set(xlabel=xlabel, ylabel=ylabel)
color=colors[2]
light_color = tint_shade_color(color, amount=1+tintshade)
dark_color = tint_shade_color(color, amount=1-tintshade)
ax.errorbar(piezos, np.array(popt_list)[:,2], yerr=np.array(perr_list)[:,2], fmt='o', markeredgewidth=2, color=color,
		 markerfacecolor=light_color, markeredgecolor=dark_color)

###
### Legend in own subplot
###
lax = axs[1,1]
h, l = axs[0,0].get_legend_handles_labels() # get legend from first plot
lax.legend(h, l, borderaxespad=0)
lax.axis("off")

fig.tight_layout()
plt.show()
# fig.savefig('figures/heating_vs_time.pdf')

	
	
	