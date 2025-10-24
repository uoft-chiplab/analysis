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
from library import styles, pi,h, hbar, generate_plt_styles, paper_settings
from data_class import Data
from scipy.optimize import curve_fit
from fit_functions import RabiFreq
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mycolors = ['#2827D8', '#D82827', '#3C9A34']
mystyles = generate_plt_styles(mycolors)
mpl.rcParams.update(paper_settings)


def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

def Linear(x,m,b):
	return m*x + b

# x should be Rabi*t
def mySin2(x, Z):
	return Z*np.sin(x/2)**2

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
PLOT_RABI_CAL = False
transfer_loss_strs = ['transfer','loss']
fit_func, _, _ = RabiFreq([])
ff = 0.82
RabiperVpp_47MHz_July2025 = 12.13/0.452 # 2025-02-12 and slightly modified for July 2025 data
bg_cutoff = 1 # VVA
pulse_time_ms = 0.01
df_list = []
EF = 0.019 # GUESS, MHZ
for i, file in enumerate(files):
	if not PLOT_RABI_CAL and i==3:
		continue
	run = Data(file)
	
	run.data['c9'] = run.data['c9'] * ff
	if AC_LOSS_CORR and i > 0 and i < 3: # only apply ac loss to datasets that had a's present
		ac_loss = 1.2 # based on most recent July 2025 ac loss correction for typical ToTF gas
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
								RabiperVpp_47MHz_July2025 * 2 * pi
		run.data['OmegaR2'] = run.data['OmegaR']**2
		run.data['time'] = pulse_time_ms
		run.data['OmegaRt'] = pulse_time_ms * run.data['OmegaR']
		run.data['OmegaR2t2'] = pulse_time_ms**2 * run.data['OmegaR']**2
		
		xname = 'OmegaRt'
		for string in transfer_loss_strs:
			yname = 'alpha_' + string
			guess = [1,0.15,0,0]
			popt, pcov = curve_fit(fit_func, run.data[xname], run.data[yname], p0=guess)
			run.data['A_' + string] = popt[0]
	else: 
		xname = 'pulse time (ms)'
		for string in transfer_loss_strs:
			yname = 'alpha_' + string
			guess = [1, 12, 0 , 0]
			popt, pcov = curve_fit(fit_func, run.data[xname], run.data[yname], p0=guess)
			OmegaR = popt[1] * 2 * pi

			run.data['OmegaR'] = OmegaR
			run.data['time'] = run.data[xname]
			run.data['OmegaR2t2'] = run.data[xname]**2 * OmegaR**2

			t_max = 1/popt[1]
	run.data['scaled_time'] = run.data['time']/1e3 * h*EF*1e6/hbar
	run.data['scaled_alpha_transfer'] = run.data['alpha_transfer']*(h*EF*1e6/hbar/(run.data['OmegaR']*1e3))**2
	df_list.append(run.data)
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

# plot summary of raw (non averaged) data
labels = ['209 G pol b to c',
		  '209 G mix b to c',
		  '202p14 G mix b to c',
		  'Rabi freq cal']
fig, axes = plt.subplots(3,2, figsize=(10,8))
axs = axes.flatten()
for i, df in enumerate(df_list):
	ax = axs[0]
	ax.set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'$\alpha$', ylim=[-0.05, 1.20],
		xlim=[0, 15])
	ax.plot(df['OmegaR2t2'], df['alpha_transfer'], label=labels[i]+', transfer', **mystyles[i])
	ax.plot(df['OmegaR2t2'], df['alpha_loss'], label=labels[i]+', loss', mfc='white')
	ax.legend()

	ax = axs[1]
	ax.set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'$\alpha$', 
		ylim=[-0.05, 0.5],
		xlim=[0, 3])
	ax.plot(df['OmegaR2t2'], df['alpha_transfer'], label=labels[i] + ', transfer', **mystyles[i])
	ax.plot(df['OmegaR2t2'], df['alpha_loss'], label=labels[i]+', loss', mfc='white')

	ax = axs[2]
	ax.set(xlabel=r'$\Omega_R t$', ylabel=r'$\alpha$', 
		xlim = [0,8],
		ylim=[-0.05, 1])
	ax.plot(df['OmegaR'] * df['time'], df['alpha_transfer'], label=labels[i] + ', transfer', **mystyles[i])
	ax.plot(df['OmegaR'] * df['time'], df['alpha_loss'], label=labels[i]+', loss', mfc='white')

	ax = axs[3]
	ax.set(xlabel=r'$\Omega_R t$', ylabel=r'$\alpha$', 
		xlim=[-0.2, 2],
		ylim=[-0.05, 0.5])
	ax.plot(df['OmegaR'] * df['time'], df['alpha_transfer'], label=labels[i] + ', transfer', **mystyles[i])
	ax.plot(df['OmegaR'] * df['time'], df['alpha_loss'], label=labels[i]+', loss', mfc='white')

	ax = axs[4]
	ax.set(xlabel=r'$\alpha_\mathrm{transfer}$', ylabel='N', xlim=[0,1])
	ax.plot(df['alpha_transfer'], df['N'], label=labels[i], **mystyles[i])
	ax = axs[5]
	ax.set(xlabel=r'$\alpha_\mathrm{loss}$', ylabel='N', xlim=[0,1])
	ax.plot(df['alpha_loss'], df['N'], label=labels[i], **mystyles[i])

fig.suptitle("Resonant transfer with 10 us pulses, raw")
fig.tight_layout()
plt.show()

# average data, plot and fit to linear response under a cutoff ~0.1
fig, axes = plt.subplots(2,2, figsize=(10,8))
axs = axes.flatten()
err_type = 'sem'
fit_list_tr = []
fit_list_ls = []
#### SUBPLOT SETTINGS
# plot alpha_transfer vs Rabi^2time^2
axs[0].set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'Transfer $\alpha$', ylim=[-0.03, 1],
		xlim=[-0.02, 20]
		)
# repeat but zoomed in
axs[2].set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'Transfer $\alpha$', ylim=[-0.03, 0.12],
	xlim=[-0.02, 0.6])
# plot alpha_loss vs Rabi^2time^2
axs[1].set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'Loss $\alpha$', ylim=[-0.03, 1],
		xlim=[-0.02, 20]
		)
# repeat but zoomed in
axs[3].set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'Loss $\alpha$', ylim=[-0.03, 0.12],
	xlim=[-0.02, 0.6])

# actual plotting and fitting loop
# this loop ignores the Rabi freq cal dataset because it was causing complications
#fit_func = UniversalQuad
fit_func = mySin2
for i, df in enumerate(df_list):
	if i == 3:
		continue

	# TRANSFER
	xname = 'OmegaR2t2'
	yname = 'alpha_transfer'
	dfgrp = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()
	ax=axs[0]
	ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', transfer', **mystyles[i])
	#alpha_cutoff_mask = (dfgrp[yname]['mean'] < 0.1) & (dfgrp[xname] < 3) # because alpha sometimes goes back down due to flopping, have to have second cutoff condition

	#sub_df = dfgrp[alpha_cutoff_mask]
	x_cutoff_mask = dfgrp[xname] < 18
	sub_df = dfgrp[x_cutoff_mask]
	guess = [1]
	popt, pcov = curve_fit(fit_func, np.sqrt(sub_df[xname]), sub_df[yname]['mean'],
						sigma=sub_df[yname][err_type])
	xs = np.linspace(0, sub_df[xname].max(), 100)
	ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i])
	ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])
	ax.vlines(sub_df[xname].values[-1], ymin=0, ymax=1, color=mycolors[i], ls='dotted')
	perr = np.sqrt(np.diag(pcov))
	fit_list_tr.append((popt, perr))
	
	# repeat but zoomed in
	ax=axs[2]
	ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', transfer', **mystyles[i])
	ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i])
	ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])
	ax.vlines(sub_df[xname].values[-1], ymin=0, ymax=1, color=mycolors[i], ls='dotted')

	# LOSS
	xname = 'OmegaR2t2'
	yname = 'alpha_loss'
	dfgrp = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()
	ax=axs[1]
	ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', loss', **mystyles[i])
	# because alpha sometimes goes back down due to flopping, have to have second cutoff condition
	# also, alpha_loss has poor signal and data near 0 is indistinguishable, so added a lower cutoff
	# alpha_cutoff_mask = (dfgrp[yname]['mean'] < 0.1) & \
	# 	(dfgrp[yname]['mean'] > 0.02) & \
   	# 	(dfgrp[xname] < 3) 
	alpha_cutoff_mask = (dfgrp[yname]['mean'] > 0.02) 
	sub_df = dfgrp[alpha_cutoff_mask]
	guess = [1]
	popt, pcov = curve_fit(fit_func, np.sqrt(sub_df[xname]), sub_df[yname]['mean'],
						sigma=sub_df[yname][err_type])
	xs = np.linspace(0, sub_df[xname].max(), 100)
	ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i])
	ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])
	ax.vlines(sub_df[xname].values[-1], ymin=0, ymax=1, color=mycolors[i], ls='dotted')
	ax.vlines(sub_df[xname].values[0], ymin=0, ymax=1, color=mycolors[i], ls='dotted')
	perr = np.sqrt(np.diag(pcov))
	fit_list_ls.append((popt, perr))
	
	# repeat but zoomed in
	ax=axs[3]
	ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', loss', **mystyles[i])
	ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i])
	ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])
	ax.vlines(sub_df[xname].values[-1], ymin=0, ymax=1, color=mycolors[i], ls='dotted')
	ax.vlines(sub_df[xname].values[0], ymin=0, ymax=1, color=mycolors[i], ls='dotted')

fig.suptitle('Resonant transfer with 10 us pulses, averaged')
fig.tight_layout()
plt.show()

spin_pol_209_fits_tr = fit_list_tr[0] # tuples of (popt, perr)
spin_pol_209_fits_ls = fit_list_ls[0]
spin_mix_209_fits_tr = fit_list_tr[1]
spin_mix_209_fits_ls = fit_list_ls[1]
spin_mix_202_fits_tr = fit_list_tr[2]
spin_mix_202_fits_ls = fit_list_ls[2]

print('-------------------------------------------------\nAnalyzing the transfer signal...')
# estimate ac loss factor by comparing pol and mix at 209 G
# probably not right due to different densities when noninteracting due to chem pot
ac_loss_guess = spin_pol_209_fits_tr[0][0] / spin_mix_209_fits_tr[0][0] # comparing fit amplitudes
e_ac_loss_guess = ac_loss_guess * np.sqrt((spin_pol_209_fits_tr[1][0]/spin_pol_209_fits_tr[0][0])**2 + \
										  (spin_mix_209_fits_tr[1][0]/spin_mix_209_fits_tr[0][0])**2) 
print(f'Comparing linear response between polarized 209 G and spin mix 209 G,\nthe difference in fit amplitudes possibly from ac loss is: {ac_loss_guess:.2f}+/-{e_ac_loss_guess:.2f}')

# First guess for Z; compare unitary spin mix transfer to noninteracting spin pol 
Z_guess = spin_pol_209_fits_tr[0][0] / spin_mix_202_fits_tr[0][0] # comparing fit amplitudes
e_Z_guess = Z_guess * np.sqrt((spin_pol_209_fits_tr[1][0]/spin_pol_209_fits_tr[0][0])**2 + \
										  (spin_mix_202_fits_tr[1][0]/spin_mix_202_fits_tr[0][0])**2) 
print(f'Comparing linear response between polarized 209 G and spin mix 202 G,\nthe difference in fit amplitudes is the Z factor: {Z_guess:.2f}+/-{e_Z_guess:.2f}')

# Second guess for Z; compare unitary spin mix transfer to noninteracting spin mix
Z_guess = spin_mix_209_fits_tr[0][0] / spin_mix_202_fits_tr[0][0] # comparing fit amplitudes
e_Z_guess = Z_guess * np.sqrt((spin_mix_209_fits_tr[1][0]/spin_mix_209_fits_tr[0][0])**2 + \
										  (spin_mix_202_fits_tr[1][0]/spin_mix_202_fits_tr[0][0])**2) 
print(f'Comparing linear response between spin mix 209 G and spin mix 202 G,\nthe difference in fit amplitudes is the Z factor: {Z_guess:.2f}+/-{e_Z_guess:.2f}')


print('-------------------------------------------------\nAnalyzing the loss signal...')
# estimate ac loss factor by comparing pol and mix at 209 G
# probably not right due to different densities when noninteracting due to chem pot
ac_loss_guess = spin_pol_209_fits_ls[0][0] / spin_mix_209_fits_ls[0][0] # comparing fit amplitudes
e_ac_loss_guess = ac_loss_guess * np.sqrt((spin_pol_209_fits_ls[1][0]/spin_pol_209_fits_ls[0][0])**2 + \
										  (spin_mix_209_fits_ls[1][0]/spin_mix_209_fits_ls[0][0])**2) 
print(f'Comparing linear response between polarized 209 G and spin mix 209 G,\nthe difference in fit amplitudes possibly from ac loss is: {ac_loss_guess:.2f}+/-{e_ac_loss_guess:.2f}')

# First guess for Z; compare unitary spin mix transfer to noninteracting spin pol 
Z_guess = spin_pol_209_fits_ls[0][0] / spin_mix_202_fits_ls[0][0] # comparing fit amplitudes
e_Z_guess = Z_guess * np.sqrt((spin_pol_209_fits_ls[1][0]/spin_pol_209_fits_ls[0][0])**2 + \
										  (spin_mix_202_fits_ls[1][0]/spin_mix_202_fits_ls[0][0])**2) 
print(f'Comparing linear response between polarized 209 G and spin mix 202 G,\nthe difference in fit amplitudes is the Z factor: {Z_guess:.2f}+/-{e_Z_guess:.2f}')

# Second guess for Z; compare unitary spin mix transfer to noninteracting spin mix
Z_guess = spin_mix_209_fits_ls[0][0] / spin_mix_202_fits_ls[0][0] # comparing fit amplitudes
e_Z_guess = Z_guess * np.sqrt((spin_mix_209_fits_ls[1][0]/spin_mix_209_fits_ls[0][0])**2 + \
										  (spin_mix_202_fits_ls[1][0]/spin_mix_202_fits_ls[0][0])**2) 
print(f'Comparing linear response between spin mix 209 G and spin mix 202 G,\nthe difference in fit amplitudes is the Z factor: {Z_guess:.2f}+/-{e_Z_guess:.2f}')

#################################################
##################### let's make finalized plots
##################################################

# average data, plot and fit to linear response under a cutoff ~0.1
fig, ax = plt.subplots(figsize=(3,2))
err_type = 'std'
x_max = 30

#### SUBPLOT SETTINGS
# plot alpha_transfer vs Rabi^2time^2
ax.set(xlabel=r'$\Omega_{23}^2 t^2$', ylabel=r'Transfer $\alpha_\mathrm{res}$', ylim=[-0.02, 1],
		xlim=[-0.02, x_max]
		)

# inset axis
inset_ax = fig.add_axes([0.42, 0.4, 0.2, 0.2])
inset_ax.set(xlim = [4, 17.5],
			 ylim=[0.6, 1.0],
			 xlabel = r'$\Omega_{23}^2t^2$',
			 ylabel=r'Loss $\alpha_\mathrm{res}$')

# actual plotting and fitting loop
fit_func = mySin2
for i, df in enumerate(df_list):
	if i == 3:
		continue

	# TRANSFER
	xname = 'OmegaR2t2'
	yname = 'alpha_transfer'
	dfgrp = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()
	ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], **mystyles[i])
	cutoff_mask = (dfgrp[xname] < x_max) & (dfgrp[yname]['mean']>0.01)
	sub_df = dfgrp[cutoff_mask]
	guess = [1]
	popt, pcov = curve_fit(fit_func, np.sqrt(sub_df[xname]), sub_df[yname]['mean'],
						sigma=sub_df[yname][err_type])
	perr = np.sqrt(np.diag(pcov))
	print(f'i={i}, Z={popt[0]:.2f}+/-{perr[0]:.2f}')
	xs = np.linspace(0, x_max, 100)
	ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i], label=rf'$Z=${popt[0]:.2f}({round(perr[0]*100)})')
	ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])

	# LOSS
	xname = 'OmegaR2t2'
	yname = 'alpha_loss'
	dfgrp = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()
	#ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', loss', **mystyles[i])
	cutoff_mask = (dfgrp[yname]['mean'] > 0.02) & (dfgrp[xname]<x_max)
	sub_df = dfgrp[cutoff_mask]
	guess = [1]
	popt, pcov = curve_fit(fit_func, np.sqrt(sub_df[xname]), sub_df[yname]['mean'],
						sigma=sub_df[yname][err_type])
	xs = np.linspace(0, sub_df[xname].max(), 100)
	perr = np.sqrt(np.diag(pcov))

	# repeat but zoomed in
	inset_ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', loss', **mystyles[i])
	inset_ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i])
	#inset_ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])

ax.legend(frameon=False, loc='upper right')
fig.tight_layout()
plt.savefig(r'\\UNOBTAINIUM\E_Carmen_Santiago\Analysis Scripts\analysis\clockshift\manuscript\manuscript_figures\SM_Zfactors.pdf', dpi=300)
plt.show()


# average data, plot and fit to linear response under a cutoff ~0.1
fig, ax = plt.subplots(figsize=(6,4))
err_type = 'sem'
x_max = 30

ax.set(xlabel=r'$\Omega_{23}^2 t^2$', ylabel=r'Loss $\alpha$', ylim=[-0.02, 1],
		xlim=[-0.02, x_max]
		)

# inset axis
inset_ax = fig.add_axes([0.33, 0.3, 0.3, 0.3])
inset_ax.set(xlim = [-0.01, 0.4],
			 ylim=[-0.01, 0.07],
			 xlabel = r'$\Omega_{23}^2t^2$',
			 ylabel=r'$\alpha$')


# actual plotting and fitting loop
fit_func = mySin2
for i, df in enumerate(df_list):
	if i == 3:
		continue

	# LOSS
	xname = 'OmegaR2t2'
	yname = 'alpha_loss'
	dfgrp = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()
	ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', loss', **mystyles[i])
	cutoff_mask = (dfgrp[yname]['mean'] > 0.02) & (dfgrp[xname]<x_max)
	sub_df = dfgrp[cutoff_mask]
	guess = [1]
	popt, pcov = curve_fit(fit_func, np.sqrt(sub_df[xname]), sub_df[yname]['mean'],
						sigma=sub_df[yname][err_type])
	xs = np.linspace(0, sub_df[xname].max(), 100)
	ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i])
	ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])
	perr = np.sqrt(np.diag(pcov))
	fit_list_ls.append((popt, perr))
	
	# repeat but zoomed in
	inset_ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', loss', **mystyles[i])
	inset_ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i])
	inset_ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])
	
#ax.legend()
fig.tight_layout()
plt.show()



# average data, plot and fit to linear response under a cutoff ~0.1
fig, axs = plt.subplots(1,2,figsize=(10,8))
err_type = 'sem'
x_max = 30

#### SUBPLOT SETTINGS
# plot alpha_transfer vs Rabi^2time^2
axs[0].set(xlabel=r'$\Omega_{23}^2 t^2$', ylabel=r'Transfer $\alpha$', ylim=[-0.02, 1],
		xlim=[-0.02, x_max]
		)
axs[1].set(xlabel=r'$\Omega_{23}^2 t^2$', ylabel=r'Loss $\alpha$', ylim=[-0.02, 1],
		xlim=[-0.02, x_max]
		)

# actual plotting and fitting loop
fit_func = mySin2
for i, df in enumerate(df_list):
	if i == 3:
		continue

	# TRANSFER
	xname = 'OmegaR2t2'
	yname = 'alpha_transfer'
	dfgrp = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()
	ax=axs[0]
	ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', transfer', **mystyles[i])
	cutoff_mask = (dfgrp[yname]['mean'] > 0.02) & (dfgrp[xname]<x_max)
	sub_df = dfgrp[cutoff_mask]
	guess = [1]
	popt, pcov = curve_fit(fit_func, np.sqrt(sub_df[xname]), sub_df[yname]['mean'],
						sigma=sub_df[yname][err_type])
	perr = np.sqrt(np.diag(pcov))
	xs = np.linspace(0, x_max, 100)
	ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i], label=f'Z={popt[0]:.1f}')
	ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])

	# LOSS
	xname = 'OmegaR2t2'
	yname = 'alpha_loss'
	dfgrp = df.groupby(xname).agg(['mean', 'std','sem']).reset_index()
	ax=axs[1]
	ax.errorbar(dfgrp[xname], dfgrp[yname]['mean'], yerr=dfgrp[yname][err_type], label=labels[i]+', loss', **mystyles[i])
	cutoff_mask = (dfgrp[yname]['mean'] > 0.02) & (dfgrp[xname]<x_max)
	sub_df = dfgrp[cutoff_mask]
	guess = [1]
	popt, pcov = curve_fit(fit_func, np.sqrt(sub_df[xname]), sub_df[yname]['mean'],
						sigma=sub_df[yname][err_type])
	xs = np.linspace(0, sub_df[xname].max(), 100)
	ax.plot(xs, fit_func(np.sqrt(xs), *popt), '-', color=mycolors[i])
	ax.plot(xs, UniversalQuad(xs, *popt), '--', color=mycolors[i])
	perr = np.sqrt(np.diag(pcov))
	

fig.tight_layout()
plt.show()
