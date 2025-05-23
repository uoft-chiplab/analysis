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
data_path = analysis_path + '\\clockshift\\data'
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
		 '2025-03-19_I_e',
		 '2025-03-19_K_e',
		 ]

TTFs = [#0.3,
		0.55,
		0.297]

ff = 0.88 # fudge factor

spins = ['c5','c9']

smallbox = True

def expdecay(t, A, tau, C):
	return A*np.exp(-t/tau) + C

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
	if not smallbox:
		bg = Data(file+'_VVA=0.dat', data_path)
		sg = Data(file +'_VVA=2.5.dat', data_path)
	else:
		bg = Data(file+'_VVA=0_smallbox.dat', data_path)
		sg = Data(file +'_VVA=2.5_smallbox.dat', data_path)	
	bg.data['c9'] = bg.data['c9'] * ff
	sg.data['c9'] = sg.data['c9'] * ff
	if file == '2025-03-12_E_e':
		bg.data['time'] = bg.data['wait']
		sg.data['time'] = sg.data['wait']
	bg_gb = bg.data.groupby('time')
	sg_gb = sg.data.groupby('time')
	for j, spin in enumerate(spins):
		if spin == 'c5':
			bg_mean = bg_gb.mean()[spin]
			bg_sem = bg_gb.sem()[spin]
			sg_mean = sg_gb.mean()[spin]
			sg_sem = sg_gb.sem()[spin]
			sg_err = np.sqrt(sg_sem**2)
		elif spin == 'c9':
			bg_mean = bg_gb.mean()[spin]
			sg_mean = sg_gb.mean()[spin]
			sg_sem = sg_gb.sem()[spin]
			sg_err = np.sqrt(sg_sem**2)
		
		# Fit loss rates
		popt, pcov = curve_fit(expdecay, sg_mean.index, sg_mean.values, sigma = sg_err.values, p0=[sg_mean.max()-sg_mean.min(), 100, bg_mean.mean()])
		perr = np.sqrt(np.diag(pcov))
		print(f'{spin} TTF: {TTFs[i]}')
		print(f'{spin} fit: {popt}, {perr}')
		print(f'{spin} bg: {bg_mean.mean()}, {bg_mean.sem()}')
		# plot data
		label = f'TTF={TTFs[i]}, tau={popt[1]:.0f}({perr[1]:.0f}), C={popt[2]:.0f}({perr[2]:.0f})'
		axs[j].errorbar(sg_mean.index, sg_mean, yerr=sg_err, label=label, **styles[i])
		axs[j].plot(sg_mean.index, expdecay(sg_mean.index, *popt), color=colors[i], ls='--', marker='')
		# plot background
		axs[j].errorbar(bg_mean.index, bg_mean, yerr=bg_sem, alpha=0.1, ls='-', **styles[i])

for i in range(len(spins)):
	axs[i].set(
		xlabel='Time [ms]',
		ylabel=spin_map(spins[i])
	)
	axs[i].legend()
fig.suptitle(r'Atom-atom loss rates fitted to $Ae^{-t/\tau} + C$')
fig.tight_layout()
plt.show()

	