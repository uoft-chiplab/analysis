# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:44:18 2024

@author: coldatoms
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from data_class import Data
from library import plt_settings, dark_colors, light_colors, colors, markers
from cycler import Cycler

styles = Cycler([{'mec':dark_color, 'mfc':light_color, 'marker':marker} for \
				 marker, dark_color, light_color in \
					 zip(markers, dark_colors, light_colors)])

def gaussian(x, A, x0, sigma, bg):
	return A*np.exp(-(x-x0)**2/sigma**2) + bg

def spin_map(spin):
	if spin == 'c5':
		return 'b'
	else:
		return 'a'

# files = ["2024-11-01_F_e.dat",
# 		"2024-10-01_F_e.dat",
# 		"2024-10-02_C_e.dat",
# 		"2024-10-03_C_e.dat",
# 		"2024-10-07_C_e.dat",
# 		"2024-06-12_S_e.dat"]

Save_df = True

files = ["2024-10-30_D_e.dat",
		 "2024-10-30_B_e.dat",
		 "2024-11-01_C_e.dat"]

# files = ["2024-11-01_H_e.dat"]

#fields = [204,202.14,202.14,202.14,202.14,202.14]
fields = [204,209, 202.14]
xname = 'freq'
xlabel = xname

dimer_freqs = [44.6, 43.325, 43.325, 43.225, 43.4, 43.3]
dimer_freqs = [43.797,45.441, 43.3]
guess_width = 0.03

spins = ['c5', 'c9']

ylabels = ['counts',
		   'loss',
		   'widths',
		   'total counts',
		   'count ratio',
		   'widths ratio']

Ars = []
e_Ars = []

dfs=[]

for file, field, dimer_freq in zip(files, fields, dimer_freqs):
	fig, axs = plt.subplots(3,3, figsize=(10,6))
	axes = axs.flatten()
	
	
	fig.suptitle(file[0][:-4] + "{} G a+b->ac dimer, {}".format(field, file))
	
	# load data
	run = Data(file)
	data = run.data

	# ratio of spin counts
	data['ratio_59'] = data['c5']/data['c9']
	
	# compute widths as quadrature sum of h and v
	data['c5_s'] = np.sqrt(data['two2D_sh1']*data['two2D_sv1'])
	data['c9_s'] = np.sqrt(data['two2D_sh2']*data['two2D_sv2'])
	
	data['ratio_59_s'] = data['c5_s']/data['c9_s']
	
	# fit sum of both spins to determine center and width
	x = data['freq']
	y = data['sum95']
	sum_popt, sum_pcov = curve_fit(gaussian, x, y, p0=[1e3, dimer_freq, guess_width, 30e3])
	sum_perr = np.sqrt(np.diag(sum_pcov))
	
	dimer_freq = sum_popt[1]
	xlims=[dimer_freq-guess_width*10, dimer_freq+guess_width*10]
	xs = np.linspace(*xlims, 100)
	
	# fix center and width of gaussian for individual spin fits
	center = sum_popt[1]
	e_center = sum_perr[1]
	width = sum_popt[2]
	e_width = sum_perr[2]
	
	print("Total counts width and center: ")
	print("width = {:.0f}±{:.0f} kHz".format(width*1e3, e_width*1e3))
	print("center = {:.3f}±{:.3f} MHz".format(center, e_center))
	
	def fixed_gaussian(x, A, bg):
		return gaussian(x, A, center, width, bg)
	
	# amplitudes to fill
	As = []
	e_As = []
	
	run.group_by_mean('freq')
	run.avg_data['filename']=file
	dfs.append(run.avg_data)
	for i in range(len(axes)):
		if i+1 > len(ylabels):
			break
		axes[i].set(xlabel=xlabel, ylabel=ylabels[i], xlim=xlims)
	
	for i, spin, sty in zip([0, 1], spins, styles):
		x = data['freq']
		y = data[spin]
		popt, pcov = curve_fit(fixed_gaussian, x, y, p0=[5e3, 15e3])
		perr = np.sqrt(np.diag(pcov))
		
		As.append(popt[0])
		e_As.append(perr[0])
		
		x = run.avg_data['freq']
		y = run.avg_data[spin]
		yerr = run.avg_data['em_'+spin]
	
		# counts
		axes[0].errorbar(x, y, yerr=yerr, label=spin_map(spin), **sty)
		axes[0].plot(xs, fixed_gaussian(xs, *popt), '--', color=colors[i])
		
		# loss
		axes[1].errorbar(x, popt[-1]-y, yerr=yerr, label=spin_map(spin), **sty)
		axes[1].plot(xs, -fixed_gaussian(xs, *popt[0:-1], 0), '--', color=colors[i])
		
		# widths
		y = run.avg_data[spin+'_s']
		yerr = run.avg_data['em_'+spin+'_s']
		axes[2].errorbar(x, y, yerr=yerr, label='a width', **sty)
		
		# scatter
		axes[6].plot(x, run.avg_data['em_'+spin], label=spin_map(spin), **sty)
		try:
			axes[7].hist(run.avg_data['em_'+spin],bins=10, alpha=0.5)
			axes[7].set(ylabel='count std hist')
		except ValueError:
			print('NaNs in series')
			
		try:
			axes[8].hist(run.avg_data['em_'+spin+'_s'],bins=10, alpha=0.5)
			axes[8].set(ylabel='width std hist')
		except ValueError:
			print('NaNs in series')
	sty = list(styles)[2]
	axes[3].errorbar(x, run.avg_data['sum95'], yerr=run.avg_data['em_sum95'], label='a+b', **sty)
	axes[3].plot(xs, gaussian(xs, *sum_popt), '--', color=colors[2])
	axes[4].errorbar(x, run.avg_data['ratio_59'], yerr=run.avg_data['em_ratio_59'], label='b/a', **sty)
	axes[5].errorbar(x, run.avg_data['ratio_59_s'], yerr=run.avg_data['em_ratio_59_s'], label='bw/aw', **sty)
	
	axes[0].legend()
	# axes[3].legend()
	
	plt.tight_layout()
	plt.show()
	
	print("The fit loss amplitudes are:")
	for spin, A, e_A in zip(spins, As, e_As):
		print(spin_map(spin), "loss amplitude: ", "{:.0f}±{:.0f}".format(A, e_A))
		
	Ar = As[0]/As[1]
	Ars.append(Ar)
	e_Ars.append(Ar*(np.sqrt((e_As[0]/As[0])**2 + (e_As[1]/As[1])**2)))

fig, ax = plt.subplots()
ax.set(xlabel='field',ylabel='b/a amp ratio')
ax.errorbar(fields, Ars, yerr=e_Ars)



# attempt: iterate on observable and plot results for all analyzed files
# df = pd.concat(dfs)
# for i, ylabel in enumerate(ylabels):
# 	fig, axes = plt.subplots(len(files))
# 	fig.suptitle(ylabel)
# 	for j, file in enumerate(files):
# 		data = df[df['filename']==file]
# 		axes[j].set(title=file)
# 		x=data['freq']
# 	
# 		# counts
# 		if i == 0:
# 			for spin, sty in zip(spins, styles):
# 				y = data[spin]
# 				yerr = data['em_'+spin]
# 				axes[j].errorbar(x, y, yerr=yerr, label=spin_map(spin), **sty)
# 		elif i == 2:# widths
# 			for spin, sty in zip(spins, styles):
# 				x = data['freq']
# 				y = data[spin+'_s']
# 				yerr =data['em_'+spin+'_s']
# 				axes[j].errorbar(x, y, yerr=yerr, label='a width', **sty)
# 		elif i==3: # total counts
# 			axes[j].errorbar(x, data['sum95'], yerr=data['em_sum95'], label='a+b', **sty)
# 		elif i==4: # count ratio
# 			axes[j].errorbar(x, data['ratio_59'], yerr=data['em_ratio_59'], label='b/a', **sty)
# 		elif i==5: # width ratio
# 			axes[j].errorbar(x, data['ratio_59_s'], yerr=data['em_ratio_59_s'], label='bw/aw', **sty)
# 			
# 		elif i==6:
# 			yerr = data['em_'+spin]
# 			axes[j].plot(x, yerr, label=spin_map(spin), **sty)

# 	fig.tight_layout()