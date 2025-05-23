# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyze Rabi oscillation data to determine Rabi frequency, and check
residuals for saturation effects.
"""

from library import styles, colors
import numpy as np
import matplotlib.pyplot as plt
import os
from data_class import Data
from scipy.optimize import curve_fit
from tabulate import tabulate

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going two levels up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
data_path = os.path.join(parent_dir, 'clockshift\\data\\Rabi_freq')

# fit functions
def RabiFreq(x, A, b, x0, C):
		return A*(np.sin(2*np.pi*b/2 * (x - x0)))**2 + C
def RabiFreq_no_offset(x, A, b, x0):
		return A*(np.sin(2*np.pi*b/2 * (x - x0)))**2 - A/2

# data sets and meta data
files = [
# 	"2024-09-25_I_e.dat",  # 191 G
 	# "2024-09-25_J_e.dat",  # 191 G
 	"2024-09-25_K_e.dat",  # 209 G
 	# "2024-09-25_L_e.dat",  # 202.14 G
	]

fudge_factor = 0.9  # hmmm
x_name = 'time'  # in ms
alpha = 0.4

Zoom_Residual = True

# analysis loop for files
for i, file in enumerate(files):
	print("*----------------")
	print("Analyzing file", file)
	
	df = Data(file, path=data_path).data
	
	df['c9'] = df['c9'] * fudge_factor
	df['f95'] = df['c5']/(df['c5']+df['c9'])
	df['sum95'] = df['c5'] + df['c9']
	df['delta_sum95'] = df['sum95'] - df.sum95.mean()
	
	# plot 
	fig, axes = plt.subplots(2,2, figsize = (10,6))
	axs = axes.flatten()
	
	fig.suptitle(file)
	
	xlabel = 'Time (ms)'
	
	# plot fraction 95 vs. time
	ax = axs[0]
	ylabel = 'fraction N5/N'
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	# fit state oscillations
	x = df[x_name]
	y = df['f95']
	num = 100
	xs = np.linspace(min(x), max(x), num)
	
	# guess list of params: amp, freq, time phase, offset
	p0 = [1, 12, 0, 0]
	p_names = ['A', 'f', 'phase', 'C']
	
	print("Fitting ", ylabel)
	popt, pcov = curve_fit(RabiFreq, x, y, p0=p0)
	freq = popt[1]
	perr = np.sqrt(np.diag(pcov))
	
	parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
 								 headers=p_names)	
	print(parameter_table, "\n")
	
	ax.plot(x, y, **styles[0])
	ax.plot(xs, RabiFreq(xs, *popt), '--', color=colors[0], label='fit')
	ax.plot(xs, RabiFreq(xs, *p0), ':', color=colors[1], label='guess', alpha=alpha)
	
	ax.legend()
	
	# plot fraction 95 fit residuals vs. time
	ax = axs[2]
	ylabel = 'fraction N5/N residual'
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	r_y = y - RabiFreq(x, *popt)
	ax.plot(xs, np.zeros(num), 'k--')
	ax.plot(x, r_y, **styles[0])
	
	# guess list of params: amp, freq, time phase, offset
	p0 = [max(r_y)/2, freq, 0, 0]
	
	print("Fitting ", ylabel)
	popt, pcov = curve_fit(RabiFreq, x, r_y, p0=p0)
	perr = np.sqrt(np.diag(pcov))
	
	parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
 								 headers=p_names)	
	print(parameter_table, '\n')
	
	ax.plot(xs, RabiFreq(xs, *popt), '--', color=colors[0], label='fit')
	ax.plot(xs, RabiFreq(xs, *p0), ':', color=colors[1], label='guess', alpha=alpha)
	
	if Zoom_Residual == True:
		ax.set(ylim=[-popt[0], popt[0]])
	
	
	# plot sum 95 vs. time
	ax = axs[1]
	ylabel = r'$\Delta$ sum N5+N9'
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	y = df['delta_sum95']
	ax.plot(x, y, **styles[0])
	
	def RabiFreq_fix_freq(x, A, x0, C):
		return A*(np.sin(2*np.pi*freq/2 * (x - x0)))**2 + C
	def RabiFreq_fix_freq_no_offset(x, A, x0):
		return A*(np.sin(2*np.pi*freq/2 * (x - x0)))**2 - A/2
	
	fit_func = RabiFreq_fix_freq_no_offset
	
	# guess list of params: amp, freq, time phase
	p0 = [(max(y)-min(y)), 0,]# -(max(y)-min(y))/2]
	p_names = ['A', 'phase',]# 'C']
	
	print("Fitting ", ylabel)
	popt, pcov = curve_fit(RabiFreq_fix_freq_no_offset, x, r_y, p0=p0)
	perr = np.sqrt(np.diag(pcov))
	
	parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
 								 headers=p_names)	
	print(parameter_table, '\n')
	
	ax.plot(xs, RabiFreq_fix_freq_no_offset(xs, *popt), '--', color=colors[0], label='fit')
	ax.plot(xs, RabiFreq_fix_freq_no_offset(xs, *p0), ':', color=colors[1], label='guess', alpha=alpha)
	
	
	# plot c5 and c9  vs. time
	ax = axs[3]
	ylabel = r'$N_\sigma$'
	ax.set(xlabel=xlabel, ylabel=ylabel)
	
	# c5
	y = df['c5']
	ax.plot(x, y, **styles[0], label='N5')
	
	p0 = [max(y)-min(y), freq, 0, y.mean()]
	p_names = ['A', 'f', 'phase', 'C']
	
	print("Fitting c5")
	popt, pcov = curve_fit(RabiFreq, x, y, p0=p0)
	perr = np.sqrt(np.diag(pcov))
	
	parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
 								 headers=p_names)	
	print(parameter_table, "\n")
	
	ax.plot(xs, RabiFreq(xs, *popt), '--', color=colors[0], label='fit')
# 	ax.plot(xs, RabiFreq(xs, *p0), ':', color=colors[0], label='guess', alpha=alpha)
	
	# c9
	y = df['c9']
	ax.plot(x, y, **styles[1], label='N9')
	
	p0 = [max(y)-min(y), freq, 0, y.mean()]
	p_names = ['A', 'f', 'phase', 'C']
	
	print("Fitting c9")
	popt, pcov = curve_fit(RabiFreq, x, y, p0=p0)
	perr = np.sqrt(np.diag(pcov))
	
	parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
 								 headers=p_names)	
	print(parameter_table, "\n")
	
	ax.plot(xs, RabiFreq(xs, *popt), '--', color=colors[1], label='fit')
# 	ax.plot(xs, RabiFreq(xs, *p0), ':', color=colors[1], label='guess', alpha=alpha)
	
	ax.legend()
	
	fig.tight_layout()
	plt.show()
	
	
	
