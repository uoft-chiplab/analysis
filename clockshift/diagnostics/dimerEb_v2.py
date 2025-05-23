# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:32:14 2025
Rough summary of dimer binding energies across field
determined experimentally, compared to theory.
Data comes from range of measurements mostly from 2023-09 to 2024-11
Measurements were originally performed to see if field affected balance
of spin loss. 
@author: coldatoms
"""

# paths

import os
import sys
# this is a hack to access modules in the parent directory
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going two levels up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
if parent_dir not in sys.path:
	sys.path.append(parent_dir)

data_path = os.path.join(parent_dir, 'analysis\\clockshift\\data\\Eb_measurements')

from library import pi, h, hbar, mK, a0, plt_settings, styles, colors, FreqMHz
from data_helper import remove_indices_formatter
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from scipy.optimize import curve_fit
from warnings import catch_warnings, simplefilter
from cycler import Cycler, cycler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

Save_results = False
spin = 'c5'

if spin == 'c5':
	spin_correction_factor = 0.5
else: 
	spin_correction_factor = 1

# fit function
def gaussian(x, A, x0, sigma, C):
	return A*np.exp(-(x-x0)**2/(2*sigma**2)) + C

### ac binding energy theory and functions
def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

re = 107*a0 # fixed, I think close to initial channel re

def EbMHz_full_sol(B, re):
	f = lambda x: 1 - 2/pi*np.arctan(pi*x*re/4) - 1/(x*a13(B))
	kappa = fsolve(f, 1e7)[0]
	Eb = -hbar**2 * kappa**2 / mK
	EbMHz = Eb / h / 1e6
	return EbMHz

def EbMHz_expansion_corr(B, re, order=1):
	if order == 1:
		kappa = 1/a13(B) * (1 + 1/2*re/a13(B))
	elif order == 2:
		kappa = 1/a13(B) * (1 + 1/2*re/a13(B) + 1/2*re**2/a13(B)**2)
	Eb = -hbar**2 * kappa**2 / mK
	EbMHz = Eb / h / 1e6
	return EbMHz

def EbMHz_naive(B):
	Eb = -hbar**2 / mK / a13(B)**2
	EbMHz = Eb / h / 1e6
	return EbMHz


####
# iterate over Eb measurements, extract and plot
### metadata
metadata_filename = 'Eb_metadata_file.xlsx'
metadata_file = os.path.join(data_path, metadata_filename)
metadata = pd.read_excel(metadata_file)
files =  metadata.loc[metadata['exclude'] != 1]['filename'].values
inset_files = ['2023-09-22_P',
			   '2024-10-30_B',
			   '2025-03-27_E']
# files = [] # for manual override
fig, axs = plt.subplots(int(np.sqrt(len(files)))+1, int(np.sqrt(len(files))+1),
						figsize=(15, 15))
ax = axs.flatten()

popts_list = []
perrs_list = []
B_list = []
files_list = []
res_freqs_list = []
peak_sctrans_list = []
e_peak_sctrans_list=[]
sat_list = []
OmegaR_list = []
trf_list = []
rsc_amp_list = []
e_rsc_amp_list = []
rsc_f_list = []
df_list = []
for i,file in enumerate(files):
	print("Analyzing file: ", file)
	meta_df = metadata.loc[metadata['filename'] == file].reset_index()
	if meta_df.empty:
		print("Dataframe is empty! Check metadata file.")
		continue
	filename = file + "_e.dat"
	ff = meta_df['ff'][0]
	trf = meta_df['trf'][0]
	VVA = meta_df['VVA'][0]
	rfsource = meta_df['PhaseOorMicrO?'][0]
	Vpp_micro = meta_df['Vpp_micro'][0]
	Bfield = meta_df['Bfield'][0]
	res_freq = FreqMHz(Bfield, 9/2, -5/2, 9/2, -7/2)
	initial_state = meta_df['initial_state'][0]
	saturated = meta_df['saturated'][0]
	if initial_state == 'bc':continue
	### Vpp calibration

	if filename[:4] == '2024' or filename[:4] == '2023':
		# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
		VpptoOmegaR47 = 17.05/0.728 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
		VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
		phaseO_OmegaR = lambda VVA, freq: 2*pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)
		
	elif filename[:4] == '2025':
		VpptoOmegaR47 = 12.01/0.452 # kHz/Vpp - 2025-02-12 calibration 
		VpptoOmegaR43 = 14.44/0.656 *VpptoOmegaR47/(17.05/0.728) # fudged 43MHz calibration
		phaseO_OmegaR = lambda VVA, freq: 2*pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)
		
	else:
		raise ValueError("filename does not start with 2024 or 2025.")
	if rfsource == 'micro':
		micrO_OmegaR = 2*pi*VpptoOmegaR43*meta_df['Vpp_micro'][0]
	else: 
		micrO_OmegaR = 0

	# create data structure
	df = Data(filename, path=data_path).data
	runfolder=filename

	df['c9'] = df['c9']*ff
	df['sum95'] = df['c5'] + df['c9']
	df['ratio95'] = df['c9']/df['c5']
	df['f5'] = df['c5']/df['sum95']
	df['f9'] = df['c9']/df['sum95'] 
	# averaged df
	gb_df = df.groupby('freq')
	mean = gb_df.mean().reset_index()
	sem = gb_df.sem().reset_index().add_prefix("em_")
	std = gb_df.std().reset_index().add_prefix("e_")
	sem.fillna(0, inplace=True)
	std.fillna(0, inplace=True)
	avg_df = pd.concat([mean, std, sem], axis=1)
	
	# compute OmegaR
	if rfsource == 'micro':
		OmegaR = micrO_OmegaR
	else:
		OmegaR = phaseO_OmegaR(VVA, df['freq'].mean()) # not sure if this function would accept list of freqs
	
	guess = [-(avg_df[spin].max()-avg_df[spin].min()), avg_df.iloc[avg_df[spin].idxmin()]['freq'], 0.05, avg_df[spin].mean()]
	popt, pcov = curve_fit(gaussian, avg_df['freq'], avg_df[spin], p0=guess)
	perr = np.sqrt(np.diag(pcov))

	# convert fit amplitude into scaled transfer
	avg_df['sctransfer'] = -(avg_df[spin]-popt[-1])*spin_correction_factor/((OmegaR*1e3)**2 * trf)
	avg_df['em_sctransfer'] = avg_df['em_'+spin]*spin_correction_factor/((OmegaR*1e3)**2 * trf)
	avg_df['f0'] = popt[1]
	avg_df['amp'] = popt[0]
	avg_df['e_amp'] = perr[0]
	avg_df['B'] = Bfield
	avg_df['rescaledf'] = avg_df['freq']/avg_df['f0']
	avg_df['rescaledamp'] = -avg_df['amp']*spin_correction_factor/((OmegaR*1e3)**2 * trf)
	if file in inset_files:
		df_list.append(avg_df)
	# add results to lists
	files_list.append(file)
	B_list.append(Bfield)
	res_freqs_list.append(res_freq)
	popts_list.append(popt)
	perrs_list.append(perr)
	sat_list.append(saturated)
	OmegaR_list.append(OmegaR)
	trf_list.append(trf)
	rsc_amp_list.append(avg_df['rescaledamp'].mean())
	e_rsc_amp_list.append(perr[0]*spin_correction_factor/((OmegaR*1e3)**2 * trf))
	rsc_f_list.append(avg_df['rescaledf'].mean())

	ax[i].errorbar(avg_df['freq'], avg_df[spin], yerr=avg_df['em_'+spin], fmt='o', label='data')
	xs = np.linspace(avg_df['freq'].min(), avg_df['freq'].max(), 100)
	ys = gaussian(xs, *popt)
	ax[i].plot(xs, ys, label='fit', ls='--', marker='')
	ax[i].set(xlabel='freq [MHz]', ylabel=spin,
		title=f'{filename}, B={Bfield}G, x0={popt[1]} MHz')

plt.show()

results_df = pd.DataFrame(
	{
		'file':files_list,
		'B':B_list,
		'res_freqs':res_freqs_list,
		'popt':popts_list,
		'perr':perrs_list,
		'sat':sat_list,
		'OmegaR':OmegaR_list,
		'trf':trf_list,
		'rsc_amp':rsc_amp_list,
		'e_rsc_amp':e_rsc_amp_list,
		'rsc_f':rsc_f_list
	}
)
results_Ebs =results_df['popt'].apply(lambda x: x[1]) - res_freqs_list
sigmas = np.abs(results_df['popt'].apply(lambda x: x[2]))
e_results_Ebs=results_df['perr'].apply(lambda x: x[1])

results_df['Eb'] = results_Ebs
results_df['e_Eb'] = e_results_Ebs
results_df['sigma']=sigmas

if Save_results:
	results_df.to_excel(os.path.join(data_path, 'Eb_results.xlsx'), index=False)

# make Eb theory lines across B
Bs = np.linspace(200, 224, 100)
Ebs_full = [EbMHz_full_sol(B, re) for B in Bs]
Ebs_o1 = EbMHz_expansion_corr(Bs, re, 1)
Ebs_o2 = EbMHz_expansion_corr(Bs, re, 2)
Ebs_naive = EbMHz_naive(Bs)

Eb_df = pd.DataFrame(
	{
		'B':Bs,
		'Ebs_full':Ebs_full,
		'Ebs_o1':Ebs_o1,
		'Ebs_o2':Ebs_o2,
		'Ebs_naive':Ebs_naive
	}
)


# ExpEb_df = pd.DataFrame(
# 	{
# 		'B':fields,
# 		'file':files,
# 		'freq':freqs,
# 		'res':ress,
# 		'Eb':Ebs,
# 		'e_Eb':e_Ebs
# 	}
# )

# if Pickle:
# 	Eb_df.to_pickle('./clockshift/analyzed_data/Ebs.pkl')
# 	ExpEb_df.to_pickle('./clockshift/analyzed_data/ExpEbs.pkl')

### PLOTTING
fig, ax = plt.subplots(figsize=(8, 6))
results_Ebs =results_df['popt'].apply(lambda x: x[1]) - res_freqs_list
sigmas = np.abs(results_df['popt'].apply(lambda x: x[2]))
e_results_Ebs=results_df['perr'].apply(lambda x: x[1])
ax.errorbar(results_df['B'], results_Ebs, sigmas, **styles[3])
ax.plot(Bs, Ebs_full, '-', color=colors[3])
#ax.plot(Bs, Ebs_o1, '-', label = "To order 1")
#ax.plot(Bs, Ebs_o2, '-', label= "To order 2")
ax.plot(Bs, Ebs_naive, '--', color=colors[-1], ls='--')
ax.legend()
ax.set(xlabel=r'$B$ [G]', ylabel=r'$\omega_d$ [MHz]')

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(results_df['B'], results_df['rsc_amp'], yerr=results_df['e_rsc_amp'], **styles[3])

### add an inset - scaked transfer comparison for selected files
left, bottom, width, height = [0.62, 0.20, 0.30, 0.28]
ax_in = fig.add_axes([left, bottom, width, height])
# test: get a long pulse 202.1 G measurement
path = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis\\clockshift\\manuscript\\manuscript_data'
file = '2025-03-19_G_e_pulsetime=0.64.dat.pkl'
test_df = pd.read_pickle(os.path.join(path, file))

# fig, ax = plt.subplots(figsize=(8,6))
# nonsat_df = results_df[results_df['sat'] == 0]
# #nonsat_df = results_df
# x = nonsat_df['Ebs']
# y = nonsat_df['rsc_amp']
# yerr= nonsat_df['e_rsc_amp']
# ax.errorbar(x,y, yerr=yerr, **styles[3])
# ax.set(
# 	#ylim = [-0.0001, 0.0006],
# 	xlim = [202, 209.1]
# )
ax.set_yscale('log')

#plot_df = results_df[results_df['sat'] == 0]
for i, df in enumerate(df_list):
	
	if i ==2:continue
	ax_in.errorbar(df['freq']/df['f0'], df['sctransfer'], yerr=df['em_sctransfer'], **styles[i], label=f'{df.B.iloc[0]} G')
#ax_in.errorbar(test_df['freq']/df_list[0]['f0'][0], test_df['c5_scaledtransfer'], yerr=test_df['em_c5_scaledtransfer'], **styles[0], label='202.14 G')
ax_in.set(yscale = 'log', xlabel=r'$|\omega/\omega_d|$', ylabel=r'$\alpha/\Omega_R^2/t_\mathrm{rf}$')
ax_in.legend(fontsize=7, loc='upper left')
fig.tight_layout()
#plt.savefig('\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis\\clockshift\\manuscript\\manuscript_figures\\dimer_Eb_v3.pdf', dpi=600)


# data binning
def bin_data(x, y, yerr, nbins, xerr=None):
	n, _ = np.histogram(x, bins=nbins)
	sy, _ = np.histogram(x, bins=nbins, weights=y/(yerr*yerr))
	syerr2, _ = np.histogram(x, bins=nbins, weights=1/(yerr*yerr))
	sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
	mean = sy / syerr2
	sem = np.sqrt(sy2/n - mean*mean)/np.sqrt(n)
	e_mean = 1/np.sqrt(syerr2)
	xbins = (_[1:] + _[:-1])/2 # mid points between bin edges
	
	# set error as yerr if n=1 for bin
	for i, num_in_bin in enumerate(n):
		if num_in_bin == 1:
			for j in range(len(y)):
				if mean[i] == y[j]:
					sem[i] += yerr[j]
					e_mean[i] = yerr[j]
					xbins[i] = x[j]
					break
		else:
			continue
		
	# average xerr
	if xerr is not None:
		sxerr, _ = np.histogram(x, bins=nbins, weights=xerr)
		mean_xerr = sxerr / n
		return xbins, mean, e_mean, mean_xerr
	
	else:
		return xbins, mean, e_mean
	# data binning


fig, ax = plt.subplots()
for i, df in enumerate(df_list):
	bins = np.arange(0.994, 1.006, 0.0008)
	df['binx'] = pd.cut(df['rescaledf'], bins=bins)
	binx = df.groupby('binx')['rescaledf'].mean()
	biny = df.groupby('binx')['rescaled_amp'].mean()
	
	#ax.errorbar(df['rescaledf'], df['amp'], yerr=df['e_amp'], **styles[i], label=f'{df.B.iloc[0]} G')
	ax.plot(binx, biny, **styles[i], label=f'{df.B.iloc[0]} G')
	#ax.errorbar(df['rescaledf'], df['sctransfer'], yerr=df['em_sctransfer'], **styles[i], label=f'{df.B.iloc[0]} G')
#ax.errorbar(test_df['freq']/df_list[0]['f0'][0], test_df['c5_scaledtransfer'], yerr=test_df['em_c5_scaledtransfer'], **styles[0], label='202.14 G')
ax.set(yscale = 'log', xlabel=r'$\omega/\omega_d$', ylabel=r'$\alpha/\Omega_R^2/t_\mathrm{rf}$')
ax.legend()
plt.show()