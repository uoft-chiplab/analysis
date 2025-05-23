# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:16:45 2025
@author: Chip Lab
"""

# paths
import os
# THIS MIGHT BE COMPUTER SPECIFIC
analysis_paths = ['\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis']

import sys
for path in analysis_paths:
	if path not in sys.path:
		sys.path.append(path)

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

lineshape = 'sinc2'
spins = ['c5', 'c9', 'ratio95']

correct_spinloss = True
saturation_correction = True
bg_from_wing = True

# Omega^2 [kHz^2] 1/e saturation value fit from 09-26_F
dimer_x0 = 1200.86  # 0.6 ToTF 2025-02-13
#dimer_x0 = 3300
dimer_e_x0 = 1030.23
# Omega^2 [kHz^2] 1e saturation value fit from 09-17_C
# HFT_x0 = 805.2923

def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

def saturation_scale(x, x0):
	""" x is OmegaR^2 and x0 is fit 1/e Omega_R^2 """
	return x/x0*1/(1-np.exp(-x/x0))
	
def gaussian(x, A, x0, sigma, C):
	return A*np.exp(-(x-x0)**2/(2*sigma**2)) + C

### constants
re = 107 * a0 # ac dimer range estimate
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra
kF = 1.1e7
kappa = np.sqrt((Eb*h*10**6) *mK/hbar**2) # convert Eb back to kappa

### Vpp calibration
VpptoOmegaR47 = 12.01/0.452 # kHz/Vpp - 2025-02-12 calibration 
VpptoOmegaR43 = 14.44/0.656 *VpptoOmegaR47/(17.05/0.728) # fudged 43MHz calibration
phaseO_OmegaR = lambda VVA, freq: 2*pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)

def spin_map(spin):
	if spin == 'c5' and not correct_spinloss:
		return 'b'
	elif spin == 'c5' and correct_spinloss:
		return 'b/2'
	elif spin == 'c9':
		return 'a'
	elif spin == 'sum95':
		return 'a+b'
	elif spin == 'ratio95' or spin == 'dimer':
		return 'a/b'
	else:
		return ''
	
# okay start analyzing
files = ['2025-03-18_H_e.dat'
		]

Vpps = [
		0.29] # Vpp
trfs = [
		640] # us
res_freq = 47.2227
ff = 0.88
ToTF = 0.573
EF = 0.0191

	
fig, axs = plt.subplots(1,3, figsize=(10, 6))

for i, file in enumerate(files):
	run = Data(file)
	OmegaR = 2*pi*VpptoOmegaR43*Vpps[i] # 2 pi kHz
	print(OmegaR)
	run.data['detuning'] = (run.data['freq'] - res_freq)
	run.data['detuning_EF'] = run.data['detuning']/EF
	run.data['detuning_Hz'] = run.data['detuning']*1e6
	
	# fudge the c9 counts using ff
	run.data['c9'] = run.data['c9'] * ff
	run.data['sum95'] = run.data['c5'] + run.data['c9']
	run.data['ratio95'] = run.data['c9']/run.data['c5']
	run.data['f5'] = run.data['c5']/run.data['sum95']
	run.data['f9'] = run.data['c9']/run.data['sum95']
	if not bg_from_wing:
		bg_df = run.data[run.data['VVA']==0]
		run.data = run.data[run.data['VVA'] != 0]
	else :
		if trfs[i] == 10:	
			upper_cutoff = 43.548
			lower_cutoff = 42.948
		elif trfs[i]== 640:
			upper_cutoff = 43.28
			lower_cutoff = 43.21
		run.data = run.data[run.data['freq'] > 42.1] # kill this point first
		bg_df = run.data[(run.data['freq'] > upper_cutoff) | (run.data['freq'] < lower_cutoff)]
		run.data = run.data[(run.data['freq'] < upper_cutoff) & (run.data['freq'] > lower_cutoff)]
	if saturation_correction:
		# the fit parameters assume Omega^2 is in kHz^2... "sorry"
		if i == 0:
			sat_scale_dimer = saturation_scale(OmegaR**2/(2*np.pi)**2, 3600)
			print(sat_scale_dimer)
	else:
		sat_scale_HFT = 1
		sat_scale_dimer = 1
	
	# compute transfer for loss and ratio
	for j, (spin, sty, color) in enumerate(zip(spins, styles, colors)):
		if spin == 'c5' or spin == 'c9':
			# compute bg
			bg_counts = bg_df[spin].mean()
			e_bg_counts = bg_df[spin].sem()
			
			run.data[spin+'_alpha'] = 1-run.data[spin]/bg_counts
			
			if correct_spinloss and spin == 'c5':
				run.data[spin+'_alpha'] = run.data[spin+'_alpha']/2
				
		elif spin == 'ratio95':
			bg_f9_mean = bg_df['f9'].mean()
			bg_f5_mean = bg_df['f5'].mean()
			# point by point for signal but background is mean
			run.data[spin+'_alpha'] = (bg_f9_mean - run.data['ratio95']*bg_f5_mean)\
					/(1/2-run.data['ratio95'])
					
		# correct transfer from saturation scaling		
		run.data[spin+'_alpha']=run.data[spin+'_alpha'] *sat_scale_dimer
		run.data[spin+'_transfer'] = run.data[spin+'_alpha'] / (trfs[i]/1e6) / (OmegaR/(2*np.pi)*1e3)**2
		run.data[spin+'_scaledtransfer'] = GammaTilde(run.data[spin+'_alpha'], h*EF*1e6, OmegaR*1e3, trfs[i]/1e6)
		
		# average results
		xparam = 'detuning_Hz'
		#yparam = spin+'_scaledtransfer'
		yparam = spin+ '_transfer'
		ylabel = r'$\alpha/t/\Omega_R^2$'

		mean = run.data.groupby([xparam]).mean().reset_index()
		sem = run.data.groupby([xparam]).sem().reset_index().add_prefix("em_")
		std = run.data.groupby([xparam]).std().reset_index().add_prefix("e_")
		avg_df = pd.concat([mean, std, sem], axis=1)
		
		# fit and integrate to get spectral weight
		popt, pcov = curve_fit(gaussian, avg_df[xparam], avg_df[yparam], p0=[avg_df[yparam].max(), avg_df[xparam].mean(), avg_df[xparam].std(), avg_df[yparam].min()], sigma=avg_df['em_' + yparam])
		xs = np.linspace(min(avg_df[xparam]), max(avg_df[xparam]), 1000)
		ys = gaussian(xs, *popt)
		SW = np.abs(np.trapz(xs, ys)) - popt[-1] # subtract background
		if spin == 'ratio95':
			print(f'{spin} fit: {popt} +/- {np.sqrt(np.diag(pcov))}')
			print(f'{spin} SW: {SW}')
			print(f'{spin} amplitude: {popt[0]}')
			print(f'{spin} peak: {avg_df[yparam].max()}')
			print(f'{spin} OmegaR^2: {(OmegaR*1e3)**2}')
			print(f'sat scale dimer: {sat_scale_dimer}')
			# if i == 0:
			# 	fast_amp = popt[0]
			# 	fast_OmegaR = OmegaR
			# 	fast_t = trfs[i]/1e6
			# else: 
			# 	slow_amp = popt[0]
			# 	slow_OmegaR = OmegaR
			# 	slow_t = trfs[i]/1e6
			# 	slow_SW = SW
		# plotting
		label = f'trf={trfs[i]} us, SW = {SW:.4f}'
		axs[j].errorbar(avg_df[xparam], avg_df[yparam], avg_df['em_' + yparam],
				   **styles[i*(j+1)], label=label)
		axs[j].plot(xs, ys, ls='-', marker='', color=colors[i*(j+1)])
		axs[j].hlines(popt[-1], min(avg_df[xparam]), max(avg_df[xparam]), color=colors[i*(j+1)], ls='--')
		axs[j].set(ylim=[min(avg_df[yparam]), max(avg_df[yparam])],
			 title = spin_map(spin),
			 ylabel=ylabel,
			 xlabel=r'$\omega$')
		
		axs[j].legend()
	
fig.tight_layout()

