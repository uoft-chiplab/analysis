# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:04:10 2024

@author: Chip Lab
"""
# paths
import os
import sys
sys.path.insert(0, 'E:\\Analysis Scripts\\analysis')
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
if root not in sys.path:
	sys.path.insert(0, root)
	
data_path = os.path.join(proj_path, 'manuscript_data')

from data_class import Data
from scipy.optimize import curve_fit
from library import generate_plt_styles, paper_settings
import numpy as np
import matplotlib.pyplot as plt
from clockshift.MonteCarloSpectraIntegration import MonteCarlo_estimate_std_from_function

def expdecay(t, A, tau, C):
	return A*np.exp(-t/tau)+C

def decay_from_one(t, tau):
	return np.exp(-t/np.abs(tau))

Save = True

# create file names
filenames = ['2024-11-21_H_e.dat',
			 '2025-03-04_U_e.dat',]

linestyles = [':','--']

ffs = [1.2, 0.88]

ToTFs = [0.58, 0.32]
e_ToTFs = [0.02, 0.01]

# plotting
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

styles = generate_plt_styles(colors, ts=0.6)
styles = [styles[1], styles[0]]
colors = [colors[1], colors[0]]

### plot settings
plt.rcdefaults()
plt.rcParams.update(paper_settings) # from library.py
font_size = paper_settings['legend.fontsize']
fig_width = 3.4 # One-column PRL figure size in inches
subplotlabel_font = 10

fig, axes = plt.subplots(2,2, figsize=(fig_width, 7/10*fig_width))
axs = axes.flatten()
if Save:
	fig_m, ax_m = plt.subplots(figsize=(fig_width, 3/5*fig_width))

# loop over files
for j, file in enumerate(filenames):
	
	ff = ffs[j]
	sty = styles[j]
	ToTF = ToTFs[j]
	e_ToTF = e_ToTFs[j]
	
	print('Analyzing ', file)
	print('ToTF = ', ToTF)
	
	label = r'$T = $' + str(ToTF) + r" $T_F$"
	label = r"{:.2f}({:.0f}) $T_F$".format(ToTF, e_ToTF*100)
	
	run = Data(path=data_path, filename=file)
	run.data['time'] = run.data['time'] + 0.025 # middle of dimer pulse (10us) to middle of pi pulse (40 us)
	run.data['c9'] = run.data['c9']*ff
	run.data['a'] = run.data['c9']
	run.data['b'] = run.data['c5']
	bg_df = run.data.loc[run.data['VVA']==0]
	bg_a = bg_df['a'].mean()
	bg_b = bg_df['b'].mean()
	bg_c5= bg_df['c5'].mean()
	bg_c9 = bg_df['c9'].mean()
	
	run.data = run.data.query('VVA > 0')
	run.data['loss_a'] = (bg_a - run.data['a'])
	run.data['loss_b'] = bg_b - run.data['b']
	run.data['loss_a'] = (bg_a - run.data['a'])/bg_a
	run.data['loss_b'] = (bg_b - run.data['b'])/bg_b 
	run.data['loss_b/loss_a'] = run.data['loss_b']/run.data['loss_a']
	run.data['loss_a/loss_b'] = run.data['loss_a']/run.data['loss_b']
	run.data['loss_c5'] = (bg_c5 - run.data['c5'])/bg_c5
	run.data['loss_c9'] = (bg_c9 - run.data['c9'])/bg_c9
	run.data['loss_c5/loss_c9'] = run.data['loss_c5']/run.data['loss_c9']
	#run.data = run.data.drop(run.data['loss_b/loss_a'].argmax()) # bad point
	#run.data = run.data[run.data['loss_c5/loss_c9'] < 3] # some really bad points
	run.data['b/a'] = run.data['b']/run.data['a']
	
	run.group_by_mean('time')
	data = run.avg_data
	x = data['time']
	y = data['fraction95']
	yerr = data['em_fraction95']
# 	popt, pcov = curve_fit(expdecay, x, y, sigma=yerr) 
# 	perr = np.sqrt(np.diag(pcov))
	xlims=[0, 5.6]
	ylims = [0.48,0.52]
	ynames=['b/a', 'loss_b', 'loss_a', 'loss_b/loss_a', ]
	
	x = data['time']
	
	# loop over axs
	for i in range(len(axs)):
		ax = axs[i]
		yname = ynames[i]
		
		ax.set(xlabel='hold time [ms]', ylabel=yname)
		
		y = data[yname]
		yerr = data['em_' + yname]
		ax.errorbar(x, y, yerr=yerr, label=label, **sty)
		if Save:
			if yname=='loss_b/loss_a':
				ax_m.set(xlabel=r'Time $t$ [ms]', ylabel=r'Loss Ratio $\delta N_2/\delta N_1$')
				y=data[yname]
				yerr=data['em_' + yname]
				ax_m.errorbar(x, y, yerr=yerr, label=label, **sty)

	# for loss_b/loss_a
	popt, pcov = curve_fit(expdecay, x, y, sigma=yerr)
	perr = np.sqrt(np.diag(pcov))
	
	print(f'{yname} fit')
	print(f'popt={popt}')
	print(f'perr={perr}')
	y0 = expdecay(0, *popt)
	y100 = expdecay(100, *popt)
	print(f'y0 = {y0}')
	print(f'y100 = {y100}')
	tau_print = r"$\tau=${:.2f}({:.0f}) ms".format(popt[1], perr[1]*100)
	print(r"Loss rate is "+tau_print)
		
	# estimate error in dimer population
	t_spec_to_img = 26.6  # ms
	dimer_pop, e_dimer_pop = MonteCarlo_estimate_std_from_function( \
		   decay_from_one, [t_spec_to_img, popt[1]], [0, perr[1]], num=100)
		
	print("Remaining dimer pop fraction before imaging {:.9f}({:.0f})".format(\
										   dimer_pop, e_dimer_pop*1e9))
	
	xmax = x.max()
	xmax = 9.025
	tt = np.linspace(x.min(), xmax, 100)
	ax.plot(tt, expdecay(tt, *popt), color=colors[j], 
		 linestyle=linestyles[j], label=tau_print)
	if Save:
		ax_m.plot(tt, expdecay(tt, *popt), #label=tau_print, 
			color=colors[j], linestyle=linestyles[j])
	
# out of loop, final plot options 
for ax in axs:
	ax.legend()
	
fig.tight_layout()
# plt.autoscale()


if Save:
	manuscript_folder = '\\\\unobtainium\\E_Carmen_Santiago\\Analysis Scripts\\analysis\\clockshift\\manuscript\\'
	data.to_pickle(manuscript_folder + 'manuscript_data\\dimer_loss_timeconstant.pkl',
			  )
	ax_m.legend(frameon=False)
	fig_m.tight_layout()
	plt.savefig(manuscript_folder + 'manuscript_figures\\dimer_loss_timeconstant.pdf')