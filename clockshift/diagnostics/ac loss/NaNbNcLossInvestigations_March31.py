# -*- coding: utf-8 -*-
"""
Created on 20-03-2025

@author: Chip Lab
Script to analyze a+b+c loss rates after jumping to 209 G
"""

# paths
import os
# THIS MIGHT BE COMPUTER SPECIFIC
analysis_path = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis'

import sys
for path in analysis_path:
	if path not in sys.path:
		sys.path.append(path)
data_path = analysis_path + '\\clockshift\\data\\ac_loss'
from library import pi, h, hbar, mK, a0, plt_settings, styles, colors
from data_helper import remove_indices_formatter
from save_df_to_xlsx import save_df_row_to_xlsx
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from clockshift.MonteCarloSpectraIntegration import MonteCarlo_estimate_std_from_function
from contact_correlations.UFG_analysis import calc_contact
from contact_correlations.contact_interpolation import contact_interpolation as C_interp
from scipy.optimize import curve_fit
from warnings import catch_warnings, simplefilter
from cycler import Cycler, cycler
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

files = [#'2025-03-12_E_e',
		 #'2025-03-19_I_e',
		 #'2025-03-19_K_e',
		 '2025-03-31_C_e_freq=47.2227',
		 '2025-03-31_C_e_freq=47.3227'
		 ]

TTFs = [#0.3,
		0.55,
		0.55]

dets = [0, 100]

ff = 0.88 # fudge factor

spins = ['c5','c9']


def expdecay(t, A, tau, C):
	return A*np.exp(-t/tau) + C

def log_expdecay(t, A, tau, C):
	return np.log(A) - t/tau + np.log(C)
def spin_map(spin):
	if spin == 'c5':
		return 'c'
	elif spin == 'c9':
		return 'b->a'
	elif spin == 'sum95':
		return 'b+c'

# Plotting
fig, axs = plt.subplots(len(spins))
for i, file in enumerate(files):
	run = Data(file+'.dat', path=data_path)
	run.data['c9'] = run.data['c9'] * ff
	time_elapsed = 21.54
	run.data['time'] = run.data['time'] + time_elapsed
	
	for j, spin in enumerate(spins):
		bg = run.data[run.data['VVA']==0]
		sg = run.data[run.data['VVA'] != 0]
		sg_gb = sg.groupby('time')
		# background
		if spin == 'c5':
			spin_bg = bg[spin].mean()
			spin_bg_err = bg[spin].sem()
			sg_mean = sg_gb.mean()[spin] - spin_bg
			sg_err = np.sqrt(sg_sem**2 + spin_bg_err**2)
		else:
			sg_mean=sg_gb.mean()[spin]
			sg_sem = sg_gb.sem()[spin]
			sg_err = np.sqrt(sg_sem**2)
		
		# Fit loss rates
		fit_function = log_expdecay
		popt, pcov = curve_fit(fit_function, sg_mean.index, sg_mean.values, sigma = sg_err.values, p0=[sg_mean.max()-sg_mean.min(), 100, 0])
		perr = np.sqrt(np.diag(pcov))
		print(f'{spin} Det: {dets[i]}')
		print(f'{spin} fit: {popt}, {perr}')
		print(f'Initial atom number at t=0: {fit_function(0, *popt)}')
		print(f'Atom number at t=t_start ms: {fit_function(time_elapsed, *popt)}')
		print(f'ratio={fit_function(time_elapsed, *popt)/ fit_function(0, *popt)}')
		# plot data
		label = f'det={dets[i]}, tau={popt[1]:.0f}({perr[1]:.0f}), C={popt[2]:.0f}({perr[2]:.0f})'
		axs[j].errorbar(sg_mean.index, sg_mean, yerr=sg_err, label=label, **styles[i])
		ts = np.linspace(0, sg_mean.index.max(), 100)
		axs[j].plot(ts, fit_function(ts, *popt), color=colors[i], ls='--', marker='')

for i in range(len(spins)):
	axs[i].set(
		xlabel='Time [ms]',
		ylabel=spin_map(spins[i])
	)
	axs[i].legend()
fig.suptitle(r'Atom-atom loss rates fitted to $Ae^{-t/\tau} + C$')
fig.tight_layout()
plt.show()

	