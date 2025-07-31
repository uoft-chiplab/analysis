# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:23:06 2025

@author: Chip Lab
"""
import sys
import os
# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.dirname(proj_path))
data_path = os.path.join(proj_path, 'data/sum_rule')
if root not in sys.path:
	sys.path.append(root)
from data_class import Data
from library import styles, pi, colors
from data_class import Data
from scipy.optimize import curve_fit
from fit_functions import RabiFreq
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq

import matplotlib.pyplot as plt
import numpy as np

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

def Linear(x,m,b):
	return m*x + b

def Sin2(x, A, b, x0):
	return A*np.sin(b/2 - x0)**2

#"universal quadratic" form for transfer alpha = Z/4 Omega^2t^2
# x is Omega^2t^2
def UniversalQuad(x, Z):
	return Z/4*x


files = ['2025-07-16_I_e.dat', # 209 G pol b's into c's
		 #'2025-07-16_J_e.dat', # 209 G ab spin mix into c's
		 #'2025-07-16_K_e.dat'] # 202.14 G ab spin mix into c's,
		 '2025-07-18_I_e.dat', # 209 G ab spin mix into c's, only diff 07-16_J is date taken
		 '2025-07-18_H_e.dat',# 202.14 G ab spin mix into C's but fr this time,
		 '2025-07-17_B_e.dat'] # Rabi freq calibration b's to c's over time 
		

pulse_freqs = [48.369,
			   48.369,
			   47.2227,
			   48.369]

AC_LOSS_CORR = False
transfer_loss_strs = ['transfer','loss']
fit_func, _, _ = RabiFreq([])
ff = 0.82
RabiperVpp_47MHz_2025 = 12.13/0.452 # 2025-02-12 and slightly modified for July 2025 data
bg_cutoff = 1 # VVA
pulse_time_ms = 0.01
df_list = []
for i, file in enumerate(files):
	run = Data(file)
	
	run.data['c9'] = run.data['c9'] * ff
	if AC_LOSS_CORR and i > 0 and i < 3: # only apply ac loss to datasets that had a's present
		ac_loss = 1.16 # guess, for ab spin mix
	else:
		ac_loss = 1 # spin pol should have no loss

	if i < 3:
		bg_data = run.data[run.data['VVA'] < bg_cutoff]
		run.data = run.data[run.data['VVA'] > bg_cutoff]
		bg_c5 = bg_data['c5'].mean()
		bg_c9 = bg_data['c9'].mean()
	else : 
		bg_c5 = 0
		bg_c9 = run.data['c9'].max() # Rabi freq cal, no bg data

	run.data['c5'] = run.data['c5'] * ac_loss
	run.data['N'] = run.data['c5'] + run.data['c9']
	run.data['alpha_transfer'] = (run.data['c5'] - bg_c5) / \
		((run.data['c5']-bg_c5) + run.data['c9'])
	run.data['alpha_loss'] = (bg_c9 - run.data['c9'])/bg_c9
	
	if i < 3: # Rabi freq dataset was scaned over time not VVA, deal with separately
		run.data['OmegaR'] = Vpp_from_VVAfreq(run.data['VVA'], pulse_freqs[i]) * \
								RabiperVpp_47MHz_2025 * 2 * pi
		run.data['OmegaR2'] = run.data['OmegaR']**2
		run.data['time'] = pulse_time_ms
		run.data['OmegaRt'] = pulse_time_ms * run.data['OmegaR']
		run.data['OmegaR2t2'] = pulse_time_ms**2 * run.data['OmegaR']**2
		
		xname = 'OmegaRt'
		for string in transfer_loss_strs:
			yname = 'alpha_' + string
			guess = [1,0.15,0,0]
			run.fitnoplots(RabiFreq, [xname, yname], guess=guess)
			#xs = np.linspace(min(run.data[xname]), max(run.data[xname]), 100)
			run.data['A_' + string] = run.popt[0]
	else: 
		xname = 'pulse time (ms)'
		for string in transfer_loss_strs:
			yname = 'alpha_' + string
			guess = [1, 12, 0 , 0]
			run.fitnoplots(RabiFreq, [xname, yname], guess=guess)
			#xs = np.linspace(min(run.data[xname]), max(run.data[xname]), 100)
			
			OmegaR = run.popt[1] * 2 * pi

			run.data['OmegaR'] = OmegaR
			run.data['time'] = run.data[xname]
			run.data['OmegaR2t2'] = run.data[xname]**2 * OmegaR**2

			t_max = 1/run.popt[1]

	df_list.append(run.data.copy())
	# checking for SW on resonance -- July 2025 addition
	fig, axs = plt.subplots(2)
	for i, string in enumerate(transfer_loss_strs):
		run.data['SW'] = 4*run.data['alpha_' + string] / run.data['OmegaR2t2']
		run.group_by_mean('OmegaR2t2')

		ylabel = r'$4\alpha/\Omega_R^2t^2$'
		xlabel = r'$\Omega_R^2t^2$'
		title = file
		
		axs[i].errorbar(run.avg_data['OmegaR2t2'], run.avg_data['SW'], yerr = run.avg_data['em_SW'], marker='o')
		axs[i].plot(run.data['OmegaR2t2'], run.data['SW'], mfc='white', mec='black')
		axs[i].set(xlim=[-0.1, 20],
				ylim=[0, 2],
			ylabel = ylabel,
			xlabel = xlabel,
			title = title + ', ' + string
			)
	fig.tight_layout()
	#fig, ax = plt.subplots()
	# ax.errorbar(run.avg_data['OmegaR2t2'], run.avg_data['SW'], yerr = run.avg_data['em_SW'], markeredgewidth=1)
	# ax.plot(run.data['OmegaR2t2'], run.data['SW'], mfc='white', mec='black', markeredgewidth=1)
	# # ax.plot(subtracted_data_OmegaR2t2, subtracted_data_SW, color = 'green')
	# ax.set(xlim=[-0.1,1],
	# 	ylim = [0.1, 3],
	# 	yscale='log',
	# 		ylabel = ylabel,
	# 	xlabel = xlabel,
	# 	title = title
	# 	#ylim=[0, 1]
	# 	)
	# #ax.hlines(y=0.5, xmin=0, xmax=10, ls='--', color='red')

# plot summary
labels = ['209 G pol b to c',
		  '209 G mix b to c',
		  '202p14 G mix b to c',
		  'Rabi freq cal']

# fig, axes = plt.subplots(2,2, figsize=(10,8))
# axs = axes.flatten()
# for i, df in enumerate(df_list):
# 	ax = axs[0]
# 	ax.set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'$\alpha$', ylim=[-0.05, 1.20],
# 		xlim=[0, 15])
# 	ax.plot(df['OmegaR2t2'], df['alpha_transfer'], label=labels[i]+', transfer', **styles[i])
# 	ax.plot(df['OmegaR2t2'], df['alpha_loss'], label=labels[i]+', loss', mfc='white')
# 	ax.legend()


# 	ax = axs[1]
# 	ax.set(xlabel=r'$\alpha$', ylabel='N', xlim=[0,1])
# 	ax.plot(df['alpha_transfer'], df['N'], label=labels[i], **styles[i])

# 	ax = axs[2]
# 	ax.set(xlabel=r'$\Omega_R t$', ylabel=r'$\alpha$', 
# 		xlim=[-0.2, 2],
# 		ylim=[-0.05, 1])
# 	ax.plot(df['OmegaR'] * df['time'], df['alpha_transfer'], label=labels[i] + ', transfer', **styles[i])
# 	ax.plot(df['OmegaR2t2'], df['alpha_loss'], label=labels[i]+', loss', mfc='white')

# 	ax = axs[3]
# 	ax.set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'Transfer $\alpha$', 
# 		ylim=[-0.05, 0.4],
# 		xlim=[0, 2.5])
# 	ax.plot(df['OmegaR2t2'], df['alpha_transfer'], label=labels[i] + ', transfer', **styles[i])
# 	ax.plot(df['OmegaR2t2'], df['alpha_loss'], label=labels[i]+', loss', mfc='white')

# fig.suptitle("Resonant transfer with 10 us pulses")
# fig.tight_layout()
# plt.show()

# plot the same data but averaged
# THIS HAS WEIRD ERRORS RIGHT NOW
fig, axes = plt.subplots(1,2, figsize=(10,8))
axs = axes.flatten()
err_type = 'sem'
Z_list = []
for i, df in enumerate(df_list):
	print(i)
	ax = axs[0]
	ax.set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'Transfer $\alpha$', ylim=[-0.05, 0.1],
		xlim=[0, 1.0])
	xname = 'OmegaR2t2'
	yname = 'alpha_transfer'
	dfgrp = df.copy().groupby(xname).agg(['mean', 'std','sem']).reset_index()
	ax.errorbar(dfgrp[xname], dfgrp['alpha_transfer']['mean'], yerr=dfgrp['alpha_transfer'][err_type], label=labels[i]+', transfer', **styles[i])
	ax.errorbar(dfgrp[xname], dfgrp['alpha_loss']['mean'], yerr=dfgrp['alpha_loss'][err_type], label=labels[i]+', loss', mfc='white')


	if i == 3:
		ts = np.linspace(0, t_max, 100)
		ax.plot(ts**2*OmegaR**2, np.sin(OmegaR/2*ts)**2, '-', color=colors[i])
		ax.plot(ts**2*OmegaR**2, ts**2*OmegaR**2/4, '--', label='lin. resp.', color=colors[i])

	else:
	
		idx_cutoff = 9
		xname = 'OmegaR2t2'
		yname = 'alpha_transfer'
		guess = [1, 10]
		popt, pcov = curve_fit(UniversalQuad, dfgrp[xname][:idx_cutoff], dfgrp[yname]['mean'][:idx_cutoff],
							sigma=dfgrp[yname][err_type][:idx_cutoff])
		xs = np.linspace(0, 6, 30)
		ax.plot(xs, UniversalQuad(xs, *popt), '--', color=colors[i])
		Z_list.append(popt[0])
		
	ax.legend()

# for i, df in enumerate(df_list):
# 	ax = axs[1]
# 	ax.set(xlabel=r'$\alpha$', ylabel=r'$N_b + N_c$', xlim=[0,1])
# 	xname = 'alpha_transfer'
# 	yname = 'N'
# 	dfgrp1 = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()
# 	ax.errorbar(dfgrp1[xname], dfgrp1[yname]['mean'], yerr=dfgrp1[yname][err_type], label=labels[i], **styles[i])

# for i, df in enumerate(df_list):
# 	if i < 3:
# 		ax = axs[2]
# 		ax.set(xlabel=r'$\Omega_R t$', ylabel=r'$\alpha$', 
# 			xlim=[-0.1, 1],
# 			ylim=[-0.01, 0.1])
# 		xname = 'OmegaR'
# 		yname = 'alpha_transfer'
# 		dfgrp2 = df.copy().groupby(xname).agg(['mean', 'std','sem']).reset_index()
# 		ax.errorbar(dfgrp2[xname] * pulse_time_ms, dfgrp2[yname]['mean'], yerr=dfgrp2[yname][err_type], label=labels[i], **styles[i])

# 	ax = axs[3]
# 	ax.set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'Transfer $\alpha$', 
# 		ylim=[-0.01, 0.2],
# 		xlim=[0, 1])
# 	xname = 'OmegaR2t2'
# 	yname = 'alpha_transfer'
# 	dfgrp3 = df.copy().groupby(xname).agg(['mean', 'std','sem']).reset_index()
# 	ax.errorbar(dfgrp3[xname], dfgrp3[yname]['mean'], yerr=dfgrp3[yname][err_type], label=labels[i], **styles[i])
# 	ax.legend()

# 	ax=axs[4]
# 	ax.set(ylabel='A from sin^2(Omega*t/2) fit')
# 	if i < 3:
# 		ax.hlines(df['A'], 0, 1, label=labels[i], color=colors[i])
# 	ax.legend()

# fig.suptitle("Resonant transfer with 10 us pulses")
# fig.tight_layout()
# plt.show()

# for i, df in enumerate(df_list):
# 	if i<3:
# 		A = df['A'].values[0]
# 		print(f'df {i}: A = {A}')

# ac_loss_factor = df_list[0]['A'].values[0]/df_list[1]['A'].values[0]
# print(f'Assumed ac loss factor is {ac_loss_factor}')
# SR_factor = df_list[2]['A'].values[0]/df_list[1]['A'].values[0]
# print(f'Assumed SR factor is {SR_factor}')


# print('Fitted Z factors are:')
# print(Z_list)