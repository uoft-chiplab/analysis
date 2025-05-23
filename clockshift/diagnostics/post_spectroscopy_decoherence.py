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
data_path = analysis_path + '\\clockshift\\data\\HFT_decoherence'
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

files = [#'2025-04-01_C_e',
	'2025-04-01_D_e']

ff = 0.88 # fudge factor

spins = ['c5','c9', 'sum95', 'fraction95']

time_cut = 0.6
def expdecay(t, A, tau, C):
	return A*np.exp(-t/tau) + C

def spin_map(spin):
	if spin == 'c5':
		return 'c'
	elif spin == 'c9':
		return 'b->a'
	elif spin == 'sum95':
		return 'b+c'
	elif spin=='fraction95':
		return 'c/(c+ff*b)'

# Plotting
fig_raw, axsr = plt.subplots(1,len(spins),figsize=(10,6))
fig_proc, axsp = plt.subplots(1,len(spins), figsize=(10,6))
fig_comp, axsc=plt.subplots(1,2, figsize=(10,6))
for i, file in enumerate(files):
	run = Data(file+'.dat', path=data_path)
	run.data['c9'] = run.data['c9'] * ff
	bg = run.data[(run.data['VVA']==0) & (run.data['time'] < time_cut)]
	sg = run.data[(run.data['VVA'] != 0) & (run.data['time'] < time_cut)]
	sg_gb = sg.groupby('time')
	bg_gb = bg.groupby('time')
	for j, spin in enumerate(spins):
		# raw
		bg_mean = bg_gb[spin].mean()
		bg_sem = bg_gb[spin].sem()
		sg_mean = sg_gb.mean()[spin]
		sg_sem = sg_gb.sem()[spin]
		sg_err = np.sqrt(sg_sem**2)
		# plot raw data
		axsr[j].errorbar(sg_mean.index, sg_mean, yerr=sg_sem, **styles[i], label='sg')
		axsr[j].errorbar(bg_mean.index, bg_mean, yerr=bg_sem, **styles[i+3], label='bg')
		
		# subtracted
		sg_mean = sg_mean - bg_mean
		sg_err = np.sqrt(sg_sem**2 + bg_sem**2)
		print(f'{spin} sg_err: {sg_err}')
		axsp[j].errorbar(sg_mean.index, sg_mean, yerr = sg_err, **styles[i])
		if spin == 'c9':
			axsp[0].errorbar(sg_mean.index, np.abs(sg_mean), yerr=sg_err, **styles[i+1], label='abs(c9-bg)')
		
	c5_mean = sg_gb['c5'].mean() - bg_gb['c5'].mean()
	c9_mean = sg_gb['c9'].mean() - bg_gb['c9'].mean()
	e_c5_mean = np.sqrt(sg_gb['c5'].sem()**2 + bg_gb['c5'].sem()**2)
	e_c9_mean = np.sqrt(sg_gb['c9'].sem()**2 + bg_gb['c9'].sem()**2)
	ratio = c5_mean/np.abs(c9_mean)
	e_ratio = ratio*np.sqrt((e_c5_mean/c5_mean)**2 + (e_c9_mean/c9_mean)**2)
	diff = c5_mean - np.abs(c9_mean)
	e_diff = np.sqrt(e_c5_mean**2 + e_c9_mean**2)
	axsc[0].errorbar(sg_mean.index, ratio, yerr = e_ratio, **styles[i], label='c5/c9')
	axsc[1].errorbar(sg_mean.index, diff, yerr = e_diff, **styles[i], label='c5-c9')
	axsc[0].set(xlabel='Time [ms]',ylabel='c5/c9', ylim=[0,2])
	axsc[1].set(xlabel='Time [ms]', ylabel='c5-c9')


for i in range(len(spins)):
	axsr[i].set(
		xlabel='Time [ms]',
		ylabel=spin_map(spins[i]),
		)
	axsp[i].set(
		xlabel='Time [ms]',
		ylabel=spin_map(spins[i]),
		)
	axsp[i].legend()
	
	axsr[i].legend()
fig_raw.suptitle('Raw')
fig_raw.tight_layout()
fig_proc.suptitle('Subtracted')
fig_proc.tight_layout()
fig_comp.suptitle('Comparison')
fig_comp.tight_layout()
plt.show()

	