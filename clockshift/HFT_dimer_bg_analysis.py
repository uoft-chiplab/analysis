"""
Created by Chip lab circa 2024-2025

Analysis script for four shot scans: HFT, bg, dimer, bg.
Produces the current Fig. 3 in the clockshift manuscript.

"""

# paths
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')

from library import pi, h, hbar, mK, a0, paper_settings, generate_plt_styles
from data_helper import remove_indices_formatter
from save_df_to_xlsx import save_df_row_to_xlsx
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from clockshift.MonteCarloSpectraIntegration import MonteCarlo_estimate_std_from_function, \
								Multivariate_MonteCarlo_estimate_std_from_function
from contact_correlations.UFG_analysis import calc_contact
from contact_correlations.contact_interpolation import contact_interpolation as C_interp
from scipy.optimize import curve_fit
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

from warnings import filterwarnings	
filterwarnings('ignore')

# plotting options
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

styles = generate_plt_styles(colors, ts=0.6)

### Script options
Talk = False
Reevaluate = False
Reevaluate_last_only = False
Calc_CTheory_std = False
Plot_HFT_Data = True

# This turns on (True) and off (False) saving the data/plots 
Save = False
Tabulate_Results = True # tabulate final results only; for plotting purposes

### Analysis options
Filter_Low_Atom_Number_Shots = True
Gaussian_Cloud = False
Final_State_Correction = True
Correct_ac_Loss = True

# select spin state simaging for analysis
state_selection = '97'
spins = ['c5', 'c9', 'ratio95']

# save file path
savefilename = '4shot_analysis_results_' + state_selection + '_testing.xlsx'
savefile = os.path.join(proj_path, savefilename)

pkl_file = os.path.join(proj_path, '4shot_results_testing.pkl')

### metadata
metadata_filename = '4shot_metadata_file.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)
files =  metadata.loc[metadata['exclude'] == 0]['filename'].values

files = ["2024-10-17_S_e",
 		 "2024-10-18_H_e",
 		 "2024-10-18_O_e",
  		 "2024-11-04_M_e",
		  "2024-11-04_O_e",
		  "2024-11-04_R_e",
		  "2024-11-05_E_e",
		  "2025-02-11_M_e",
		  "2025-02-12_P_e",
		  "2025-02-12_S_e",
		  "2025-02-12_W_e",
		  "2025-02-12_Z_e",
		  "2025-02-12_ZC_e",
		  "2025-02-12_ZF_e",
		  "2025-02-18_H_e",
		  "2025-02-18_K_e",
		  "2025-02-18_O_e",
		  "2025-02-26_I_e",
		  "2025-02-27_M_e",
		  "2025-02-27_P_e",
		  "2025-03-05_K_e",
 		 ]
# files = ["2024-11-05_H_e"]

### plot settings
plt.rcdefaults()
plt.rcParams.update(paper_settings) # from library.py
font_size = paper_settings['legend.fontsize']
fig_width = 3.4 # One-column PRL figure size in inches
subplotlabel_font = 10

# plt.rcParams.update({"figure.figsize": [12,8],
# 					 "font.size": 14,
# 					 "lines.markeredgewidth": 2,
# 					 "errorbar.capsize": 0})

### Calibrations
RabiperVpp_47MHz_2024 = 17.05/0.728 # 2024-09-16
e_RabiperVpp_47MHz_2024 = 0.15

RabiperVpp_43MHz_2024 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration
e_RabiperVpp_43MHz_2024 = 0.14

RabiperVpp_47MHz_2025 = 12.01/0.452 # 2025-02-12
e_RabiperVpp_47MHz_2025 = 0.28

# Fudge the 2024 based on the 2025/2024 47MHz ratio...
RabiperVpp_43MHz_2025 = RabiperVpp_43MHz_2024 * RabiperVpp_47MHz_2025/ \
													RabiperVpp_47MHz_2024
e_RabiperVpp_43MHz_2025 = RabiperVpp_43MHz_2025 * np.sqrt(\
		  (e_RabiperVpp_43MHz_2024/RabiperVpp_43MHz_2024)**2 + \
			  (e_RabiperVpp_47MHz_2025/RabiperVpp_47MHz_2025)**2 +
			  (e_RabiperVpp_47MHz_2024/RabiperVpp_47MHz_2024)**2)
# this is about 0.35


# Omega^2 [kHz^2] 1/e saturation value fit from 09-26_F
# dimer_x0 = 1200.86  # 0.6 ToTF 2025-02-13
# dimer_x0 = 3300
# e_dimer_x0 = 100
# new after comparing, and realizing we used wrong VVA to Vpp calibration
# see rf_saturation_analysis/dimer_saturation_curves.py
dimer_x0 = 5211 
e_dimer_x0 = 216

# Omega^2 [kHz^2] 1e saturation value fit from 09-17_C
# HFT_x0 = 805.2923

# omega^2 [kHz^2] 1e saturation average fit value from various ToTF
# HFT_x0 = 848
# e_HFT_x0 = 50

# HFT_x0_cold = 737
# e_HFT_x0_cold = 35

# what if we should use loss saturation instead?
# I think this is correct CD 2025-05-02
# see rf_saturation_analysis/transfer_scaling_various_ToTF.py
HFT_x0 = 924
e_HFT_x0 = 93
HFT_x0_cold = HFT_x0
e_HFT_x0_cold = e_HFT_x0

def saturation_scale(x, x0):
	""" x is OmegaR^2 and x0 is fit 1/e Omega_R^2 """
	return x/x0*1/(1-np.exp(-x/x0))

### ac loss corrections
# these are from varying jump page results
# see diagnostics/
ToTFs = [0.26, 0.36, 0.6, 1.1]
corr_cs = [1.00, 1.15, 1.31, 1.31]
e_corr_cs = [0.05, 0.06, 0.08, 0.08]

corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(corr_cs))
e_corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(e_corr_cs))
	
### constants
re = 103 * a0 # ac dimer range estimate
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

def xstar(B, EF):
	return Eb/EF # hbar**2/mK/a13(B)**2 * (1-re/a13(Bfield))**(-1)


### common functions
def linear(x, a, b):
	return a*x + b

def slope(x,a):
	return a*x

def min_avg_max(my_array):
	return (np.min(my_array), np.mean(my_array), np.max(my_array))

def spin_map(spin):
	if spin == 'c5':
		return 'b'
	elif spin == 'c9':
		return 'a'
	elif spin == 'sum95':
		return 'a+b'
	elif spin == 'ratio95' or spin == 'dimer':
		return 'a/b'
	else:
		return ''

### transfer functions
def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

def dimer_transfer(Rab, fa, fb):
	'''computes transfer to dimer state assuming loss in b is twice the loss
		in a. Rba = Nb/Na for the dimer association shot. Nb_bg and Na_bg
		are determined from averaged bg shots.'''
	return (fa - Rab*fb)/(1/2-Rab)

def e_dimer_transfer(Rab, e_Rab, bg_fa, e_bg_fa, bg_fb, e_bg_fb):
	Rabfb = Rab*bg_fb
	e_Rabfb = Rabfb*np.sqrt((e_Rab/Rab)**2 + (e_bg_fb/bg_fb)**2)
	numer = bg_fa - Rabfb 
	e_numer = np.sqrt(e_Rabfb**2 + e_bg_fa**2)
	gamma = (bg_fa - Rabfb)/(1/2-Rab)
	e_gamma = gamma * np.sqrt((e_numer/numer)**2 + (e_Rab/Rab)**2)
	return e_gamma

def dimer_transfer_norm(fRab, bgRab, bgfa, bgfb):
	return (bgfa - fRab*bgRab*bgfb)/(1/2-fRab*bgRab)

def dimer_transfer_a(Rab, bg_fa, bg_fb):
	bgRba = bg_fb/bg_fa
	return (1-Rab*bgRba)/(1-2*Rab)

def dimer_transfer_b(Rab, bg_fa, bg_fb):
	bgRab = bg_fa/bg_fb
	return (bgRab - Rab)/(1-2*Rab)

def dimer_transfer_sum_half(Rab, bg_fa, bg_fb):
	return (dimer_transfer_a(Rab, bg_fa, bg_fb) + dimer_transfer_b(Rab, bg_fa, bg_fb))/2


# sinc^2 dimer lineshape functions
def sinc2(x, trf):
	"""sinc^2 normalized to sinc^2(0) = 1"""
	t = x*trf
	return np.piecewise(t, [t==0, t!=0], [lambda t: 1, 
					   lambda t: (np.sin(np.pi*t)/(np.pi*t))**2])

def Int2DGaussian(a, sx, sy):
	return 2*a*np.pi*sx*sy

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

### Summary plot lists
results_list = []

try:
	with open(pkl_file, 'rb') as f:
	    old_results_list = pkl.load(f)
except (OSError, IOError):
    old_results_list = []

if Reevaluate == True:
	if Reevaluate_last_only == True:
		old_results_list = old_results_list[:-1]
	else:
		old_results_list = []

### loop analysis over selected datasets
save_df_index = 0
for filename in files:
	
##############################
######### Analysis ###########
##############################
	print("----------------")
	print("Analyzing " + filename)
	
	# get metadata, skip if empty
	meta_df = metadata.loc[metadata.filename == filename].reset_index()
	if meta_df.empty:
		print("Dataframe is empty! The metadata likely needs updating." )
		continue
	
	filename = filename + ".dat"
	
	# check if we need to evalute this run
	skip_evaluation = False
	for old_results in old_results_list:
		if old_results['Run'] == filename:
			print("Already evaluated, skipping...")
			results_list.append(old_results)
			skip_evaluation = True
			break
	if skip_evaluation:
		continue
	
	###
	### INITIALIZE DATA AND METADATA STRUCTURES
	###
	
	# create data structure
	run = Data(filename, path=data_path)
	runfolder = filename 
	
	# initialize results dict to turn into df
	results = {
			'Run': filename,
			'trf_blackman': meta_df['trf_blackman'][0]*1e6,
			'trf_dimer': meta_df['trf_dimer'][0]*1e6,
			'ToTF_i': meta_df['ToTF_i'][0],
			'ToTF_f': meta_df['ToTF_f'][0],
			'ToTF_diff': meta_df['ToTF_f'][0]-meta_df['ToTF_i'][0],
			'ToTF': (meta_df['ToTF_i'][0] + meta_df['ToTF_f'][0])/2,
			'e_ToTF_if': (meta_df['ToTF_i_sem'][0] + meta_df['ToTF_f_sem'][0])/2,
			'EF': (meta_df['EF_i'][0] + meta_df['EF_f'][0])/2,
			'e_EF': np.sqrt(meta_df['EF_i_sem'][0]**2 + \
					   meta_df['EF_f_sem'][0]**2)/2,
			'N': (meta_df['N_i'][0] + meta_df['N_f'][0])/2,
			'e_N': np.sqrt(meta_df['N_i_sem'][0]**2 + \
					   meta_df['N_f_sem'][0]**2)/2,
			'barnu': meta_df['barnu'][0],
			'e_barnu': meta_df['barnu_sem'][0],
			'ff': meta_df['ff'][0],
			'pol': meta_df['pol'][0],
			'year': filename[:4]
			}
		
	# convert EF to MHz
	results['EF'] = results['EF']/1e3
	results['e_EF'] = results['e_EF']/1e3
		
	# error in fudge factor 
	results['e_ff'] = 0.02
		
	# account for change in ToTF from before and after UShots
	results['e_ToTF'] = np.sqrt(results['e_ToTF_if']**2+(results['ToTF_diff']/2*0.68)**2)
	
	# calculate other parameters
	results['x_star'] = xstar(meta_df['Bfield'][0], results['EF'])
	results['kF'] = np.sqrt(2*mK*results['EF']*h*1e6)/hbar
	
	###
	### THEORY
	###
	
	# from Tilman's unitary gas harmonic trap averaging code
	with catch_warnings(): # ignore some runtime warnings in the code
		simplefilter('ignore')
		results['C_theory'], results['Ns_theory'], results['EF_theory'], \
			results['ToTF_theory'] = calc_contact(results['ToTF'], 
										 results['EF'], results['barnu'])
	
		# sample C_theory from calibration values distributed normally to obtain std
		if Calc_CTheory_std == True:
			C_theory_mean, C_theory_std = MonteCarlo_estimate_std_from_function(calc_contact, 
					[results['ToTF'], results['EF'], results['barnu']], 
					[results['e_ToTF'], results['e_EF'], results['e_barnu']], num=200)
			print("For nominal C_theory={:.2f}".format(results['C_theory']))
			print("MC sampling of normal error gives mean={:.2f}±{:.2f}".format(
										  C_theory_mean, C_theory_std))
			results['C_theory_std'] = C_theory_std
		else:	
			results['C_theory_std'] = 0.01  # this is close to what is often obtained form the above
		
	# clock shift theory prediction from C_Theory
	results['CS_theory'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
							* results['C_theory']
	# note this is missing kF uncertainty, would have to do the above again
	results['CS_theory_std'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
							* results['C_theory_std']
							
	###
	### DATA FILTERING
	###
	
	# remove indices if requested in metadata file
	remove_indices = remove_indices_formatter(meta_df['remove_indices'][0])
	if remove_indices is not None:
		run.data.drop(remove_indices, inplace=True)
	
	# data filtering:
	if Filter_Low_Atom_Number_Shots == True:
		# filter out low atom number shots
		filter_indices = run.data.index[run.data['sum95'] < 2500].tolist()
		run.data.drop(filter_indices, inplace=True)
		
	# compute number of total data points
	num = len(run.data)
	
	###
	### VPP AND OMEGA RABI CALIBRATION
	###
	
	# check which year, taken from filename
	if filename[:4] == '2024':
		RabiperVpp47 = RabiperVpp_47MHz_2024 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
		e_RabiperVpp47 = e_RabiperVpp_47MHz_2024
		RabiperVpp43 = RabiperVpp_43MHz_2024 
		e_RabiperVpp43 = e_RabiperVpp_43MHz_2024
	elif filename[:4] == '2025':
		RabiperVpp47 = RabiperVpp_47MHz_2025 # kHz/Vpp - 2025-02-12 calibration 
		e_RabiperVpp47 = e_RabiperVpp_47MHz_2025
		RabiperVpp43 = RabiperVpp_43MHz_2025 # fudged 43MHz calibration
		e_RabiperVpp43 = e_RabiperVpp_43MHz_2025
	else:
		raise ValueError("filename does not start with 2024 or 2025.")
	
	# Rabi frequencies, phaseO is HFT, micrO is dimer
	phaseO_OmegaR = lambda VVA, freq: 2*pi*RabiperVpp47 * Vpp_from_VVAfreq(VVA, freq)
	micrO_OmegaR = 2*pi*RabiperVpp43*meta_df['Vpp_dimer'][0]
	
	###
	### PRELIMINARY CALCULATIONS	
	###
	
	# correct cloud size names
	size_names = ['two2D_sv1', 'two2D_sh1', 'two2D_sv2', 'two2D_sh2']
	new_size_names = ['c5_sv', 'c5_sh', 'c9_sv', 'c9_sh']
	for new_name, name in zip(new_size_names, size_names):
		run.data[new_name] = np.abs(run.data[name])

	# average H and V sizes
	run.data['c5_s'] = (run.data['c5_sv']+run.data['c5_sh'])/2
	run.data['c9_s'] = (run.data['c9_sv']+run.data['c9_sh'])/2
	
	# compute detuning
	if filename == "2025-02-18_H_e" or filename == "2025-02-18_K_e":
		# dumb exclusions for 02-18 testing of different dimer freqs
		exclude_test_freqs = run.data[run.data['freq']==43.218 or \
								   run.data['freq']==43.23 or \
								   run.data['freq']==43.278].index
		run.data = run.data.drop(exclude_test_freqs)
	 
	run.data['detuning'] = run.data['freq'] - meta_df['res_freq'][0] # MHz
	
	# split df into each type of measurement
	HFT_df = run.data.loc[(run.data.state==75) & (run.data.freq-meta_df['res_freq'][0]>0)]
	bg_75_df = run.data.loc[(run.data.state==75) & (run.data.freq-meta_df['res_freq'][0]<0)]
	dimer_97_df = run.data.loc[(run.data.state==97) & (run.data.VVA>0)]
	bg_97_df = run.data.loc[(run.data.state==97) & (run.data.VVA==0)]
	
	dfs = [HFT_df, bg_75_df, dimer_97_df, bg_97_df]
	dimer_dfs = [dimer_97_df, bg_97_df]
	
	# swap counts to Gaussian fit integrated counts if flag is true
	if Gaussian_Cloud == True:
		for df in dfs:
			df['c9'] = np.abs(Int2DGaussian(df['two2D_a2'], 
									 df['two2D_sh2'], df['two2D_sv2']))
		# for all dfs that have c5 signal
		for df in [HFT_df, dimer_97_df, bg_97_df]:
			df['c5'] = np.abs(Int2DGaussian(df['two2D_a1'], 
									 df['two2D_sh1'], df['two2D_sv1']))
		
		# for the HFT bg though, there is no c5 signal, so can't trust  
		# Gaussian #1 fit parameters, just set to zero
		df = bg_75_df
		df['c5'] = 0  # assuming that bg offset fit param deals with bg counts
			
	# fudge the c9 counts using ff, compute other sums and fractions
	for df in dfs:
		df['c9'] = df['c9'] * results['ff']
		df['sum95'] = df['c5'] + df['c9']
		df['ratio95'] = df['c9']/df['c5']
		df['f5'] = df['c5']/df['sum95']
		df['f9'] = df['c9']/df['sum95']
		
	# find initial state polarization a/(a+b)
	results['pol'] = bg_97_df['f9'].mean()
	results['e_pol'] = bg_97_df['f9'].sem()
			
	# compute Omega Rabi
	OmegaR_dimer = micrO_OmegaR  # in 1/ms
	e_OmegaR_dimer = OmegaR_dimer * e_RabiperVpp43/RabiperVpp43
	OmegaR_HFT = phaseO_OmegaR(HFT_df.VVA.unique()[0], HFT_df.freq.unique()[0]) * np.sqrt(0.31)
	e_OmegaR_HFT = OmegaR_HFT * e_RabiperVpp47/RabiperVpp47
	
	print("dimer VVAs", dimer_97_df.VVA.unique())
	
	###
	### SATURATION CORRECTION
	###
	
	# the fit parameters assume Omega^2 is in kHz^2... "sorry"
	OmegaR_dimer_kHz2 = OmegaR_dimer**2/(2*np.pi)**2
	e_OmegaR_dimer_kHz2 = 2*OmegaR_dimer*e_OmegaR_dimer/(2*np.pi)**2
	
	OmegaR_HFT_kHz2 = OmegaR_HFT**2/(2*np.pi)**2
	e_OmegaR_HFT_kHz2 = 2*OmegaR_HFT*e_OmegaR_HFT/(2*np.pi)**2
	
	# estimate error from saturation correction
	sat_scale_dimer, e_sat_scale_dimer = MonteCarlo_estimate_std_from_function(
		  saturation_scale, [OmegaR_dimer_kHz2, dimer_x0], 
		  [e_OmegaR_dimer_kHz2, e_dimer_x0])
	
	# for HFT, need a switch for different temperatures
	if results['ToTF'] < 0.35:
		sat_scale_HFT, e_sat_scale_HFT = MonteCarlo_estimate_std_from_function(
		  saturation_scale, [OmegaR_HFT_kHz2, HFT_x0_cold], 
		  [e_OmegaR_HFT_kHz2, e_HFT_x0_cold])
	else:
		sat_scale_HFT, e_sat_scale_HFT = MonteCarlo_estimate_std_from_function(
		  saturation_scale, [OmegaR_HFT_kHz2, HFT_x0], 
		  [e_OmegaR_HFT_kHz2, e_HFT_x0])
		
	# put results in dict
	results['OmegaR_dimer_kHz2'] = OmegaR_dimer_kHz2
	results['e_OmegaR_dimer_kHz2'] = e_OmegaR_dimer_kHz2
	results['sat_scale_dimer'] = sat_scale_dimer
	results['e_sat_scale_dimer'] = e_sat_scale_dimer
		
	results['OmegaR_HFT_kHz2'] = OmegaR_HFT_kHz2
	results['e_OmegaR_HFT_kHz2'] = e_OmegaR_HFT_kHz2
	results['sat_scale_HFT'] = sat_scale_HFT
	results['e_sat_scale_HFT'] = e_sat_scale_HFT

	###
	### HFT
	###
	
	# detuning in HFT
	HFT_detuning = HFT_df.detuning.values[0]
	
	# decide to use final state correction
	final_state_correction = (1+HFT_detuning/results['EF']/results['x_star'])
	if Final_State_Correction == False:
		final_state_correction = 1
		
	results['final_state_correction'] = final_state_correction
	
	# correct for ac loss dependent on temperature
	corr_a = 1
	corr_c = corr_c_interp(results['ToTF'])
	results['e_ac_loss'] = e_corr_c_interp(results['ToTF'])
	
	if not Correct_ac_Loss:
		corr_c = 1
		results['e_ac_loss'] = 0
		
	HFT_df.c5 = HFT_df.c5*corr_c
	HFT_df.c9 = HFT_df.c9*corr_a
	
	# calculate averages and sems
	bg_c5 = bg_75_df.c5.mean()
	e_bg_c5 = bg_75_df.c5.sem()
	bg_c9 = bg_75_df.c9.mean()
	e_bg_c9 = bg_75_df.c9.sem()
	
	results['HFT_bg_sum95'] = bg_75_df.sum95.mean()
	results['e_HFT_bg_sum95'] = bg_75_df.sum95.sem()
	results['HFT_sum95'] = HFT_df.sum95.mean()
	results['e_HFT_sum95'] = HFT_df.sum95.sem()
	
	results['HFT_c9_cloud_amp'] = HFT_df['two2D_a2'].mean()
	results['e_HFT_c9_cloud_amp'] = HFT_df['two2D_a2'].sem()
	results['HFT_c5_cloud_amp'] = HFT_df['two2D_a1'].mean()
	results['e_HFT_c5_cloud_amp'] = HFT_df['two2D_a2'].sem()
	
	results['HFT_bg_c9_cloud_amp'] = bg_75_df['two2D_a2'].mean()
	results['e_HFT_bg_c9_cloud_amp'] = bg_75_df['two2D_a2'].sem()
	results['HFT_bg_c5_cloud_amp'] = bg_75_df['two2D_a1'].mean()
	results['e_HFT_bg_c5_cloud_amp'] = bg_75_df['two2D_a2'].sem()
	
	# compute contact from HFT transfer and b atom loss
	for measure in ['transfer', 'loss_transfer']:
		if measure == 'transfer':
			HFT_df[measure] = (HFT_df.c5-bg_c5)/(HFT_df.c5-bg_c5+HFT_df.c9)
		elif measure == 'loss_transfer':
			HFT_df[measure] = np.ones(len(HFT_df.c9))-HFT_df.c9/bg_c9
			
		HFT_df[measure] = HFT_df[measure] * sat_scale_HFT
		
		HFT_df['scaled_'+measure] = GammaTilde(HFT_df[measure], 
			 h*results['EF']*1e6, OmegaR_HFT*1e3, results['trf_blackman']/1e6)
		HFT_df['C_'+measure] = 2*np.sqrt(2)*pi**2*HFT_df['scaled_'+measure]*\
								(HFT_detuning/results['EF'])**(3/2) * \
										final_state_correction
										
	# calculate contact and random error from counts
	results['C'], results['C_sem'] = (HFT_df['C_transfer'].mean(), 
											   HFT_df['C_transfer'].sem())
	
	
	
	# again for loss
	results['C_loss'], results['C_loss_sem'] = (HFT_df['C_loss_transfer'].mean(), 
												   HFT_df['C_loss_transfer'].sem())
	
	# define a function for Monte Carlo systematic error estiamtes
	c5 = HFT_df.c5.mean()  # using the mean values here for simplicity
	c9 = HFT_df.c9.mean()
	def contact(bg, EF, measure='transfer'):
		if measure == 'transfer':
			transfer = np.mean((c5-bg)/(c5-bg+c9))
		elif measure == 'loss':
			transfer = np.mean(1-c9/bg)
		
		transfer = transfer * sat_scale_HFT
		
		scaledtransfer = GammaTilde(transfer, h*EF*1e6, OmegaR_HFT*1e3, 
							  results['trf_blackman']/1e6)
		C = 2*np.sqrt(2)*pi**2*scaledtransfer*(HFT_detuning/EF)**(3/2) * \
							final_state_correction
		return C
	
	# now, to incorperate uncertainty in the bg and EF, we are going to sample
	# calculations selecting inputs in normal distributions
	inputs = [bg_c5, results['EF']]  # these will be the distribution means
	errors = [e_bg_c5, results['e_EF']]  # these will be the stds
	results['C_sys_mean'], results['C_sys_std'] = MonteCarlo_estimate_std_from_function(contact, 
								 inputs, errors, num=100, measure='transfer')
	
	# add the error form MonteCarlo bg and EF to the sem of C in quadrature
	results['e_C'] = np.sqrt(results['C_sys_std']**2 + results['C_sem']**2)
	
	# same for the loss
	inputs = [bg_75_df.c9.mean(), results['EF']]
	errors = [bg_75_df.c9.sem(), results['e_EF']]
	results['C_loss_sys_mean'], results['C_loss_sys_std'] = \
					MonteCarlo_estimate_std_from_function(contact, inputs,
								   errors, num=100, measure='loss')
	results['e_C_loss'] = np.sqrt(results['C_loss_sys_std']**2 + results['C_loss_sem']**2)
	
	if Talk == True:
		print("T/TF = {:.2f}, EF = {:.1f}kHz".format(results['ToTF'], results['EF']*1e3))
		print("HFT C = {:.3f}±{:.3f}".format(results['C'], results['e_C']))
		print("HFT loss C = {:.3f}±{:.3f}".format(results['C_loss'], results['e_C_loss']))
		
	# calculate clockshift
	results['HFT_FM'] = np.sqrt(results['x_star'])/(2**(3/2)*pi) * results['C']
	results['e_HFT_FM'] = np.sqrt(results['x_star'])/(2**(3/2)*pi) * results['e_C']
		
	###
	### PLOT HFT DATA
	###
	
	if Plot_HFT_Data:
		fig, axs = plt.subplots(3,3, figsize=[12,10])
		title = f'{filename} HFT transfer at ' + r'$T/T_F=$'+'{:.3f}±{:.3f}'.format(results['ToTF'], results['e_ToTF'])
		
		axes = axs.flatten()
		ylabels = ['c5', 'c9', 'sum95', 
				'transfer', 'scaled_transfer', 'C_transfer', 
				'loss_transfer', 'scaled_loss_transfer', 'C_loss_transfer']
		xlabel = 'cyc'
		x = HFT_df[xlabel]
		
		for i, ax in enumerate(axes):
			ax.set(xlabel=xlabel, ylabel=ylabels[i])
			ax.plot(x, HFT_df[ylabels[i]], **styles[0], label='signal')
			
		x = bg_75_df[xlabel]
		bg_ylabels = ['c5', 'c9', 'sum95']
		
		for i, ylabel in enumerate(bg_ylabels):
			ax = axes[i]
			ax.set(xlabel=xlabel, ylabel=ylabels[i])
			ax.plot(x, bg_75_df[ylabels[i]], **styles[1], label='bg')
			ax.legend()
			
		fig.suptitle(title)
		fig.tight_layout()
		plt.show()
		
	###
	### DIMER
	###
	
	dimer_df = dimer_97_df
	bg_df = bg_97_df
	
	dimer_freq = dimer_df.freq.unique()[0] - meta_df['res_freq'][0]
	
	# loop over spins
	for spin in spins:
		if spin == 'c5' or spin == 'c9':
			
			# compute averages
			counts, e_counts = (dimer_df[spin].mean(), dimer_df[spin].sem())
			bg_counts, e_bg_counts = (bg_df[spin].mean(), bg_df[spin].sem())
			
			# compute transfer from averaging whole dataset
			transfer = (1 - counts/bg_counts)
			e_transfer = counts/bg_counts * np.sqrt((e_counts/counts)**2 + \
									  (e_bg_counts/bg_counts)**2)
				
			# correct spin b by halving it
			if spin == 'c5':
				transfer = transfer/2
				e_transfer = e_transfer/2
	
			if Talk == True:
				print(spin_map(spin)+" counts = {:.2f}±{:.2f}k, ".format(counts/1e3, e_counts/1e3) + \
					"bg = {:.2f}±{:.2f}k, ".format(bg_counts/1e3, e_bg_counts/1e3) + \
						"raw transfer = {:.2f}±{:.2f}".format(bg_counts-counts, np.sqrt(e_counts**2 + e_bg_counts**2)))
				
		elif spin == 'ratio95':
			
			# compute averages
			bg_f9, e_bg_f9 = (bg_df['f9'].mean(), bg_df['f9'].sem())
			bg_f5, e_bg_f5 = (bg_df['f5'].mean(), bg_df['f5'].sem())
			
			# I want the covariance for the std error of the mean... so I think I do this
			bg_f9f5cov = np.cov(np.array(bg_df[['f9', 'f5']]).T)/np.sqrt(len(bg_df))
			# oh right, they are just perfectly anticorrelated...
			
			bgRab, e_bgRab = (bg_df['ratio95'].mean(), bg_df['ratio95'].sem())
			Rab, e_Rab = (dimer_df['ratio95'].mean(), dimer_df['ratio95'].sem())
			fRab = Rab/bgRab
			e_fRab = fRab*np.sqrt((e_Rab/Rab)**2 + (e_bgRab/bgRab)**2)
		
			# compute transfer
			transfer = dimer_transfer(Rab, bg_f9, bg_f5)
			
			# now construct covariance matrix, using the one calculated for bg f9 and f5,
			# and setting the covariance of Rab to zero
			cov_Rab_bg_f9f5 = np.array([[e_Rab**2, 0, 0],
									  [0, bg_f9f5cov[0,0], bg_f9f5cov[0,1]],
									  [0, bg_f9f5cov[1,0], bg_f9f5cov[1,1]]])
			means = [Rab, bg_f9, bg_f5]
			transfer_mean, e_transfer = Multivariate_MonteCarlo_estimate_std_from_function(dimer_transfer, 
													  means, cov_Rab_bg_f9f5)
			
		# correct transfer from fit saturation scaling
		transfer = transfer * sat_scale_dimer
		e_transfer = e_transfer * sat_scale_dimer
		
		# put in results dict
		results['transfer_'+spin] = transfer
		results['e_transfer_'+spin] = e_transfer
		
		scaledtransfer = GammaTilde(transfer, h*results['EF']*1e6, OmegaR_dimer*1e3, 
							  results['trf_dimer']/1e6)
		e_scaledtransfer = e_transfer/transfer * scaledtransfer
		
		results['scaledtransfer_'+spin] = scaledtransfer
		results['e_scaledtransfer_'+spin] = e_scaledtransfer
		
		results['SW_'+spin] = scaledtransfer/results['trf_dimer']/results['EF']
		results['e_SW_'+spin] = e_scaledtransfer/scaledtransfer*results['SW_'+spin]
		
		results['FM_'+spin] = results['SW_'+spin] * dimer_freq/results['EF']
		results['e_FM_'+spin] = np.abs(e_scaledtransfer/scaledtransfer*results['FM_'+spin])
		
		
	raw_transfer_a = bg_df['c9'].mean() - dimer_df['c9'].mean()
	raw_transfer_b = bg_df['c5'].mean() - dimer_df['c5'].mean()
	e_raw_transfer_a = np.sqrt(bg_df['c9'].sem()**2 + dimer_df['c9'].sem()**2)
	e_raw_transfer_b = np.sqrt(bg_df['c5'].sem()**2 + dimer_df['c5'].sem()**2)
	
	raw_transfer_fraction = raw_transfer_a/raw_transfer_b
	e_raw_transfer_fraction = raw_transfer_fraction * np.sqrt((e_raw_transfer_a/raw_transfer_a)**2 + \
														   (e_raw_transfer_b/raw_transfer_b)**2)
	
	if Talk == True:
		print("raw transfer a/b = {:.2f}+/-{:.2f}".format(raw_transfer_fraction, 
													e_raw_transfer_fraction))
	
	results['raw_transfer_fraction'] = raw_transfer_fraction
	results['e_raw_transfer_fraction'] = e_raw_transfer_fraction


	###
	### DIMER PLOTTING
	###
	
	fig, axs = plt.subplots(3,3, figsize=[12,10])
	axes = axs.flatten()
	j = 0
	
	title = f'{filename} dimer transfer at ' + r'$T/T_F=$'+'{:.3f}±{:.3f}'.format(results['ToTF'], results['e_ToTF']) +\
			 ' using states ' + state_selection
			 
	# prepping evaluation ranges
	xrange = 0.6
	xlow = dimer_freq - xrange
	xhigh = dimer_freq + xrange
	xnum = 1000
	xx = np.linspace(xlow, xhigh, xnum)/results['EF']
	
	# plot sinc function on averaged transfer data
	ax_fit = axes[j]
	j += 1
	ax_fit.set(xlabel=r"Detuning $\hbar\omega/E_F$",
		   ylabel=r"Scaled Transfer $\tilde\Gamma$")
	
	for i, spin in enumerate(spins):
		sty = styles[i]
		yy = results['scaledtransfer_'+spin] * sinc2(xx-dimer_freq/results['EF'], 
							results['EF']*results['trf_dimer']/2)
		ax_fit.plot(xx, yy, '--', label=spin_map(spin))
		ax_fit.errorbar([dimer_freq/results['EF'], dimer_freq/results['EF']], 
			  [0, scaledtransfer], yerr=e_scaledtransfer*np.array([1,1]), **sty)
	
	# plot transfer for each spin
	ax = axes[j]
	j += 1
	ax.set(xlabel=r"Time [min]",
		   ylabel=r"Transfer fraction $\alpha$")
	
	for i, spin in enumerate(spins):
		sty = styles[i]
		
		# compute time
		x = dimer_df.cyc*31/60
		
		# get transfer for each points
		if spin == 'ratio95':
			y = dimer_transfer(dimer_df['ratio95'], bg_f9, bg_f5)
		else:
			y = transfer = (1 - dimer_df[spin]/bg_df[spin].mean())
			
		ax.plot(x, y, **sty, label=spin_map(spin))
			
		# fit to line
		popt, pcov = curve_fit(linear, x, y)
		ax.plot(x, linear(x, *popt), '--', color=colors[i])
	
	
	# loop over spins, and plot counts vs time, and compute averages
	for i, spin, sty in zip([j, j+1, j+2], spins, styles):
		
		# plot counts vs time
		ax = axes[i]
		ax.set(xlabel="Time [min]", ylabel=spin_map(spin)+" Counts")
	
		# signal
		x = dimer_df.cyc*31/60
		y = dimer_df[spin]
		popt, pcov = curve_fit(linear, x, y)
		ax.plot(x, y, label='signal', **styles[0])
		ax.plot(x, linear(x, *popt), '--')
		# bg
		x = bg_df.cyc*31/60
		y = bg_df[spin]
		popt, pcov = curve_fit(linear, x, y)
		ax.plot(x, y, label='bg', **styles[1])
		ax.plot(x, linear(x, *popt), '-.')
	
	j += 3
		
	for i in range(j):
		axes[j].legend()
	
	# generate table
	ax_table = axes[-1]
	ax_table.axis('off')
	quantities = [r'$T/T_F$', r'$\Delta T/T_F$', r'$E_F$',
			   'sat_scale HFT', 'sat_scale dimer', r'$\Omega_R^2$ dimer', 
			   r'raw $\alpha_b$', '$N_a/(N_b+N_a)$ pol']
	values = ["{:.3f}({:.0f})".format(results['ToTF'], results['e_ToTF']*1e3),
		   "{:.3f}({:.0f})".format(results['ToTF_diff'], results['e_ToTF_if']*1e3),
		   "{:.2f}({:.0f}) kHz".format(results['EF']*1e3, results['e_EF']*1e5),
		   "{:.2f}({:.0f})".format(sat_scale_HFT, e_sat_scale_HFT*1e2),
		   "{:.2f}({:.0f})".format(sat_scale_dimer, e_sat_scale_dimer*1e2),
		   "{:.0f}({:.0f})".format(OmegaR_dimer_kHz2, e_OmegaR_dimer_kHz2),
		   "{:.3f}({:.0f})".format(results['transfer_c5']/sat_scale_dimer,
							 1e3*results['e_transfer_c5']/sat_scale_dimer),
		   
		   "{:.3f}({:.0f})".format(results['pol'], results['e_pol']*1e3)
		   ]
	table = list(zip(quantities, values))
	the_table = ax_table.table(cellText=table, loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(12)
	the_table.scale(1,1.5)
	
	fig.suptitle(title)
	fig.tight_layout()
	
	plt.show()
	
###############################
####### Saving Results ########
###############################
	
	if Save == True:
		savedf = pd.DataFrame(results, index=[save_df_index])
		save_df_index += 1
		save_df_row_to_xlsx(savedf, savefile, filename)
		
	results_list.append(results)
	
	# dump results into pickle
	with open(pkl_file, 'wb') as f:
	        pkl.dump(results_list, f)
			
###############################
###### Systematic Errors ######
###############################

# convert results into dataframe
df_total = pd.DataFrame(results_list)

# 0 is min, avg is 1, max is 2
min_avg_max_choice = 1

### EF systematics 
# barnu systematics in thermometry
e_EF_from_barnu_sys = 0.02 # error in EF from barnu is 2%

# imaging systematics in thermometry
e_EF_from_light_saturation_fudge = 0.01

e_EF_from_therm_sys = np.sqrt(e_EF_from_light_saturation_fudge**2 + \
							  e_EF_from_barnu_sys**2)

	
### Transfer rate systematics

# this is where EF errors come in
e_kF_therm_sys = e_EF_from_therm_sys/2 # C propto 1/kF = 1/sqrt(EF), check my error propagation

dimer_systematic_labels = ["saturation_correction", "Omega_R", "EF_therm"]

dimer_systematic_factors = [
	 min_avg_max(df_total['e_sat_scale_dimer']/df_total['sat_scale_dimer'])[min_avg_max_choice],
	 min_avg_max(df_total['e_OmegaR_dimer_kHz2']/df_total['OmegaR_dimer_kHz2'])[min_avg_max_choice],
# 	 e_kF_therm_sys, # don't double count, because it's in HFT systematic below
	 ]

dimer_systematics = dict(zip(dimer_systematic_labels, dimer_systematic_factors))

HFT_systematic_labels = ["saturation_correction", "Omega_R", "ac_loss", 
						 "fudge_factor", "EF_therm"]

HFT_systematic_factors = [
	 min_avg_max(df_total['e_sat_scale_HFT']/df_total['sat_scale_HFT'])[min_avg_max_choice],
	 min_avg_max(df_total['e_OmegaR_HFT_kHz2']/df_total['OmegaR_HFT_kHz2'])[min_avg_max_choice],
	 min_avg_max(df_total['e_ff'])[min_avg_max_choice],
	 e_kF_therm_sys,
	 ]

HFT_systematics = dict(zip(HFT_systematic_labels, HFT_systematic_factors))

### Temperature dependent factors

# invert C_interp
ToTF_list = np.linspace(0.2, 1.0, 100)
C_list = np.array(C_interp(ToTF_list))
ToTF_interp = lambda x: np.interp(x, C_list, ToTF_list)

# now make ac_loss correction a function of C
e_corr_c_interp_fn_C = lambda x: e_corr_c_interp(ToTF_interp(x))
		
###############################
###### Summary Plotting #######
###############################	

# plotting options
dimertype2024 = 'c5'
dimertype2025 = 'c5'
plot_options = {
				"Loss Contact": False,
				"Binned": True,
				"not Binned": False,
				"plot_fits": False,
				"CS_pred": True,
				}

true_options = []
for key, val in plot_options.items():
	if val:
		true_options.append(key)
			
title_end = ' , '.join(true_options)
if title_end != '':
	title_end = " with " + title_end
plot_title = "Four shot analysis" + title_end

# choose contact
if plot_options['Loss Contact'] == False:
	df_total['C_data'] = df_total['C']
	df_total['e_C_data'] = df_total['e_C']
else:
	df_total['C_data'] = df_total['C_loss']
	df_total['e_C_data'] = df_total['e_C_loss']
	
sum_rule = 0.5
I_d_conv = 2
# correct data for sumrule

# make spectral weight a fraction out of 1
df_total['SW_c5'] = df_total['SW_c5'] * I_d_conv
df_total['SW_c9'] = df_total['SW_c9'] * I_d_conv
df_total['e_SW_c5'] = df_total['e_SW_c5'] * I_d_conv
df_total['e_SW_c9'] = df_total['e_SW_c9'] * I_d_conv

df_total['FM_c5'] = df_total['FM_c5']
df_total['FM_c9'] = df_total['FM_c9']
df_total['e_FM_c5'] = df_total['e_FM_c5']
df_total['e_FM_c9'] = df_total['e_FM_c9']

df_total['CS_c5'] = df_total['FM_c5'] / sum_rule
df_total['CS_c9'] = df_total['FM_c9'] / sum_rule
df_total['e_CS_c5'] = df_total['e_FM_c5'] / sum_rule
df_total['e_CS_c9'] = df_total['e_FM_c9'] / sum_rule


### THEORY CALCULATIONS ####
# calculate theoretical sum rule and first moment vs contact
Bfield = 202.14
open_channel_fraction = 0.93

C = np.linspace(0, max(df_total['C_data']), 50) 
kF = np.mean(df_total['kF'])
EF = np.mean(df_total['EF'])
xstar = xstar(Bfield, EF)
a13kF = kF * a13(202.14)
kappa = np.sqrt((Eb*h*10**6) *mK/hbar**2) # convert Eb back to kappa

### ZY single-channel square well w/ effective range
# divide I_d by a13 kF,
not_small_kappa_correction = 1.08
ell_d_SqW = 1/(kappa * (1 + re/a13(Bfield))) * open_channel_fraction  * not_small_kappa_correction / a0
ell_d_SqW = 160
I_d_SqW = kF * C/pi * ell_d_SqW * a0 / a13kF
# I_d_SqW = C/a13kF * kF * 1/(pi*kappa) / (1 + re*kappa)
I_d_ZR = C/pi * open_channel_fraction

# compute clock shift
#CS_d = sum_rule*-2*kappa/(pi*kF) * (1/1+re/a13(Bfield)) * C 
CS_d_SqW = -I_d_SqW * a13kF**2 * 2 * (kappa/kF)**2 # convert I_d to CS_d to avoid rewriting sum_rule and o_c_f
# multiply FM (Eq. 7) by a13 kF
FM_d_SqW =  CS_d_SqW * sum_rule
	
### PJ CCC
# spectral weight, clockk shift, first moment
spin_me = 32/42 # spin matrix element
ell_d_CCC = spin_me * 42 * pi
I_d_CCC =  kF / a13kF / pi * ell_d_CCC * a0 * C
CS_d_CCC = -I_d_CCC *a13kF**2 /kF**2 * kappa**2 * 2
FM_d_CCC = CS_d_CCC * sum_rule

### Other analytical models for bounding
I_d_max =  kF * 1/pi * a13(Bfield) * C / a13kF  # shallow bound state 
I_d_min =  kF * 1/(pi*kappa) * 1/(1+re*kappa) * C/a13kF  # another version of square well with eff range
CS_d_max = -I_d_max * a13kF**2 /kF**2 * kappa**2
CS_d_min = -I_d_min * a13kF**2 /kF**2 * kappa**2
FM_d_max = CS_d_max * sum_rule
FM_d_min = CS_d_min * sum_rule

### Choose Model for clock shift plot
FM_d = FM_d_CCC
CS_d = CS_d_CCC

### HFT clockshifts
FM_HFT = C /(2 * pi) * kappa/kF * a13kF
CS_HFT = FM_HFT / sum_rule

### final dataset manipulations 
# calculate kF a13 for each data point
df_total['a13kF'] = np.array(df_total['kF']) * a13(202.14)
	
# split df
dfs = []
labels = ['2024', '2025']
for label in labels:
	dfs.append(df_total.loc[df_total.year==label])
	
		
### Error bands 
# add all error sources in quadrature
HFT_error_const = np.sqrt(np.sum(np.array(HFT_systematic_factors)**2))
print(f"The constant HFT systematic error band is {HFT_error_const:.2f}")

# add ToTF/C dependent factors
HFT_error_fn_C = lambda x: np.sqrt(HFT_error_const**2+e_corr_c_interp_fn_C(x)**2)

HFT_error = lambda x: np.sqrt(HFT_error_const**2+e_corr_c_interp(x)**2)

# same for dimer, but no T depedent factors
dimer_error = np.sqrt(np.sum(np.array(dimer_systematic_factors)**2))
print(f"Dimer systematic error band is {dimer_error:.2f}")

# for the spectral weight vs. contact plots, SW propto C, so any error in C
# should be propagated
SWvC_error = lambda x: np.sqrt(HFT_error_fn_C(x)**2 + dimer_error**2)
# on second thought, maybe not, since that isn't really an error in the theory
SWvC_error = lambda x: dimer_error**2

alpha = 0.3
sty_i = 0

### intitialize plots
fig, axs = plt.subplots(3,1, figsize=[fig_width, fig_width*7/5], height_ratios=[0.8,1.2,1.2])
axes = axs.flatten()

contact_label = r"Contact  $C/N k_F$"
spectral_weight_label = r"Spectral Weight  $I_d/k_Fa_{13}$"
clock_shift_label = r"Clock Shift  $\tilde\Omega k_Fa_{13}$"
temperature_label = r"Temperature  $T/T_F$"

#-- spectral weight vs. C
ax = axes[1]
ax.set(xlabel=contact_label, ylabel=spectral_weight_label, 
	   xlim=[0, C.max()+0.05], ylim=[0, 0.4])

i = 0

# theory curves
ax.plot(C, I_d_ZR, 'k:')#, label='zero range')
ax.text(0.75, 0.24, 'zero range', rotation=39, rotation_mode='anchor', size=font_size)
ax.plot(C, I_d_SqW, '-', color=colors[sty_i], label='SqW')
ax.fill_between(C, I_d_SqW*(1-SWvC_error(C)), I_d_SqW*(1+SWvC_error(C)), 
				color=colors[sty_i], alpha=alpha)
ax.plot(C, I_d_CCC, '--', color=colors[sty_i+1], label='CCC')

ax.legend(frameon=False, loc='lower right')

if Tabulate_Results == True:
	tab_theory = {
		'C':C,
		'SW_ZR': I_d_ZR,
		'SW_SqW' : I_d_SqW,
		'SW_CCC':I_d_CCC,
		'e_SW_SqW': SWvC_error(C) 
	}
	tab_theory_df = pd.DataFrame(tab_theory)
	tab_theory_df.to_csv('clockshift/tabulated_results/subplot_b_theory.csv')


# binned data
if plot_options['Binned']:
	nbins = 16
	
	x = df_total['C_data']
	xerr = df_total['e_C_data']
	y = df_total['SW_c5']/ df_total['a13kF']
	yerr = np.abs(df_total['e_SW_c5'])/ df_total['a13kF']
	
	popt_SW, pcov_SW = curve_fit(slope, x, y, sigma=yerr)
	perr_SW = np.sqrt(np.diag(pcov_SW))
	ell_d_SW_fit = popt_SW[0] * a13kF/kF/a0 *pi
	e_ell_d_SW_fit = perr_SW[0] * a13kF/kF/a0 *pi
	es_ell_d_SW_fit =  np.sqrt(HFT_error_const**2 + dimer_error**2) * ell_d_SW_fit
	
	print("Slope of Spectral Weight:")
	print("fit \ell_d = {:.0f}({:.0f})({:.0f}) a_0".format(ell_d_SW_fit, 
												e_ell_d_SW_fit, es_ell_d_SW_fit))
	
	print("CCC \ell_d = {:.0f} a_0".format(ell_d_CCC))
	print("SqW \ell_d_SqW = {:.0f} a_0".format(ell_d_SqW))
	
	
	binx, biny, binyerr, binxerr = bin_data(x, y, yerr, nbins, xerr=xerr)
	ax.errorbar(binx, biny, yerr=binyerr, xerr=binxerr, label='binned', **styles[sty_i])
	
	if Tabulate_Results == True:
		tab_data = {
			'C':binx,
			'SW': biny,
			'e_SW' : binyerr,
			'e_C' : binxerr
		}
		tab_data_df = pd.DataFrame(tab_data)
		tab_data_df.to_csv('clockshift/tabulated_results/subplot_b_data.csv')

	
	mask = ~np.isnan(biny)
	
	bpopt_SW, bpcov_SW = curve_fit(slope, binx[mask], biny[mask], sigma=binyerr[mask])
	if plot_options['plot_fits'] == True:
		ax.plot(C, slope(C, *bpopt_SW), '--', color=colors[sty_i])

if plot_options['not Binned']:
	# data
	for df, sty, label in zip(dfs, styles, labels):
		x = df['C_data']
		xerr = df['e_C_data']
		dimertype = dimertype2025
		y = df['SW_' +dimertype] / df['a13kF']
		yerr = np.abs(df['e_SW_'+dimertype]) / df['a13kF']
		ax.errorbar(x, y, yerr=yerr, xerr=xerr, label=label, **sty)
		
		i += 1


#-- Clock Shift vs. C
ax = axes[2]
ax.set(xlabel=contact_label, ylabel=clock_shift_label, xlim=[0, C.max()+0.05])

i = 0
label = 'C ratio'
	
# theory curve
ax.plot(C, CS_HFT, '-', color=colors[sty_i+1], label='HFT')
# ax.fill_between(C, FM_HFT*(1-HFT_error(C)), FM_HFT*(1+HFT_error(C)), 
# 				color=colors[sty_i+1], alpha=alpha)

CS_tot = CS_HFT+CS_d
ax.plot(C, CS_HFT+CS_d, '-', color=colors[sty_i+2], label='total')

CS_tot_err = np.sqrt((CS_d*dimer_error)**2 + (CS_HFT*HFT_error(C))**2)
# ax.fill_between(C, FM_tot-FM_tot_err, FM_tot+FM_tot_err, 
# 				color=colors[sty_i+2], alpha=alpha)
# ax.plot(C, FM_d_CCC_sc, '--', color=colors[sty_i+1], label=theory_labels[2])

ax.plot(C, CS_d, '-', color=colors[sty_i], label='dimer')
# ax.fill_between(C, FM_d*(1-dimer_error), FM_d*(1+dimer_error), 
# 				color=colors[sty_i], alpha=alpha)

if Tabulate_Results == True:
	tab_theory = {
		'C':C,
		'CS_HFT': CS_HFT,
		'CS_tot' : CS_tot,
		'CS_d':CS_d,
	}
	tab_theory_df = pd.DataFrame(tab_theory)
	tab_theory_df.to_csv('clockshift/tabulated_results/subplot_c_theory.csv')

ax.legend(frameon=False)

# binned data
if plot_options['Binned']:
	x = df_total['C_data']
	xerr = df_total['e_C_data']
	
	# FM dimer
	y = df_total['CS_c5'] * df_total['a13kF']
	yerr = np.abs(df_total['e_CS_c5']) * df_total['a13kF']
	binx, biny, binyerr, binxerr = bin_data(x, y, yerr, nbins, xerr=xerr)
	ax.errorbar(binx, biny, yerr=binyerr, xerr=binxerr, label='dimer', 
			 **styles[sty_i])
	
	d_binx = binx
	d_biny = biny.copy()
	e_d_binx = binxerr
	e_d_biny = binyerr
	
	# save these ones for later
	CS_tot = biny
	e_CS_tot = binyerr
	
	# FM HFT
	y = x / (2 * pi) * kappa/kF * a13kF / sum_rule
	yerr = xerr / (2 * pi) * kappa/kF * a13kF / sum_rule
	binx, biny, binyerr, binxerr = bin_data(x, y, yerr, nbins, xerr=xerr)
	ax.errorbar(binx, biny, yerr=binyerr, xerr=binxerr, label='HFT', 
			 **styles[sty_i+1])
	HFT_binx = binx
	HFT_biny = biny
	e_HFT_binx = binxerr
	e_HFT_biny = binyerr
	
	# FM tot
	CS_tot += biny
	e_CS_tot = np.sqrt(e_CS_tot**2 + binyerr**2)
	ax.errorbar(binx, CS_tot, yerr=e_CS_tot, xerr=binxerr,
			  label='total', **styles[sty_i+2])
	
	mask = ~np.isnan(CS_tot) # some bins are empty, so mask them
	popt_CS_tot, pcov_CS_tot = curve_fit(slope, binx[mask], CS_tot[mask], 
									  sigma=e_CS_tot[mask])
	perr_CS_tot = np.sqrt(np.diag(pcov_CS_tot))
	fit_sys_err = np.sqrt(HFT_error_const**2 + dimer_error**2)
	
	
	if Tabulate_Results == True:
		tab_data = {
			'C':binx,
			'e_C':e_d_binx,
			'CS_d': d_biny,
			'e_CS_d':e_d_biny,
			'CS_HFT': HFT_biny,
			'e_CS_HFT':e_HFT_biny ,
			'CS_tot':CS_tot,
			'e_CS_tot':e_CS_tot,
		}
		tab_data_df = pd.DataFrame(tab_data)
		tab_data_df.to_csv('clockshift/tabulated_results/subplot_c_data.csv')
	# slope of HFT line is A_HFT/pi
	# slope of dimer line is -2A_d/pi * kappa * a13
	
	print("Slope of Clock Shift fit:")
	print("fit slope = {:.3f}({:.0f})({:.0f})".format(popt_CS_tot[0], 
				1e3*perr_CS_tot[0], 1e3*np.abs(popt_CS_tot[0])*fit_sys_err))
	
	CS_firstord = -1/pi*(1-pi**2/8*re/a13(202.14))	
	print("first order re/a13 predicted slope = {:.3f}".format(CS_firstord))
	
	if plot_options['plot_fits'] == True:
		ax.plot(binx, slope(binx, *popt_CS_tot), '--', color=colors[sty_i+2])
		
	if plot_options['CS_pred'] == True:
		ax.plot(binx, slope(binx, CS_firstord), '--', color=colors[sty_i+2])

# data
if plot_options['not Binned']:
	for df, sty, label in zip(dfs, styles, labels):
		x = np.array(df['C_data'])
		xerr = np.array(df['e_C_data']) 
		dimertype = dimertype2025
		y = np.array(df['FM_' + dimertype]) * df['a13kF']
		yerr = np.array(np.abs(df['e_FM_' + dimertype])) * df['a13kF']
		
		ax.errorbar(x, y, yerr=yerr, xerr=xerr, label=label, **sty)
		ax.hlines(0, 0, x.max(), ls='dashed', color='black')
		ax.legend()
	
		i += 1

	
#-- C vs. ToTF
ax = axes[0]
ax.set(ylabel=contact_label, xlabel=temperature_label, xlim=[0.2, 0.85])

sty_i = 1

# plot trap-averaged contact
xs = np.linspace(min(df_total['ToTF'])*0.9, max(df_total['ToTF'])*1.1, 100)
ax.plot(xs, C_interp(xs), '--', color=colors[sty_i], label='trap-averaged theory')
ax.fill_between(xs, C_interp(xs)*(1-HFT_error(xs)), C_interp(xs)*(1+HFT_error(xs)), 
				color=colors[sty_i], alpha=alpha)
# ax.legend(frameon=False)



# binned data
if plot_options['Binned']:
	x = df_total['ToTF']
	xerr = df_total['e_ToTF']
	y = df_total['C_data']
	yerr = df_total['e_C_data']
	binx, biny, binyerr, binxerr = bin_data(x, y, yerr, nbins, xerr=xerr)
	ax.errorbar(binx, biny, yerr=binyerr, xerr=binxerr, label='binned', **styles[sty_i])
	
	if Tabulate_Results:
		a_binx = binx
		a_biny = biny
		a_binyerr = binyerr
		a_binxerr = binxerr
		a_ttf = xs
		a_C = C_interp(xs)
		a_HFT_error = HFT_error(xs)
		a_data = {
			'TTF':a_binx,
			'C': a_biny,
			'e_C' : a_binyerr,
			'e_TTF' : a_binxerr
		}
		a_data_df = pd.DataFrame(a_data)
		a_theory = {
			'TTF':xs,
			'C':C_interp(xs),
			'e_C':+HFT_error(xs),
			}
		a_theory_df = pd.DataFrame(a_theory)
		a_data_df.to_csv('clockshift/tabulated_results/subplot_a_data.csv')
		a_theory_df.to_csv('clockshift/tabulated_results/subplot_a_theory.csv')
	

# data
if plot_options['not Binned']:
	for df, sty, label in zip(dfs, styles, labels):
		x = df['ToTF']
		xerr = df['e_ToTF']
		y = df['C_data']
		yerr = df['e_C_data']
		ax.errorbar(x, y, yerr=yerr, xerr=xerr, **sty)

# final plot settings
# fig.suptitle(plot_title)

fig.tight_layout()  # note this is done before the labels on purpose
subplot_labels = ['(a)', '(b)', '(c)']
for n, ax in enumerate(axs):
	label = subplot_labels[n]
	ax.text(-0.18, 1.12, label, transform=ax.transAxes, size=subplotlabel_font)
	
plt.subplots_adjust(top=0.95)

fig.savefig('clockshift/manuscript/manuscript_figures/spectral_weight_.pdf')
plt.show()	
