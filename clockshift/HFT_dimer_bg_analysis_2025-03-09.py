"""
Created by Chip lab 2024-10-17

Analysis script for triple shot scans: HFT, dimer, bg
"""

# paths
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')
import sys
if root not in sys.path:
	sys.path.append(root)
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

	
### This turns on (True) and off (False) saving the data/plots 
Save = False
### script options
Debug = False
Filter = True
Talk = False
Reevaluate = False
Reevaluate_last_only = False
Calc_CTheory_std = False
# Cold = True
# cold_totf = 0.3

lineshape = 'sinc2'
ylist=[]
# select spin state simaging for analysis
state_selection = '97'

#spins = ['c5', 'c9', 'sum95']
spins = ['c5', 'c9', 'ratio95']

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
		  "2024-11-05_H_e",
		  "2025-02-11_M_e",
		  "2025-02-12_P_e",
		  "2025-02-12_S_e",
		  "2025-02-12_W_e",
		  "2025-02-12_Z_e",
		  "2025-02-12_ZC_e",
		  "2025-02-12_ZF_e",
		  "2025-02-13_E_e",
		  "2025-02-18_H_e",
		  "2025-02-18_K_e",
		  "2025-02-18_O_e",
		  "2025-02-18_R_e",
		  "2025-02-26_I_e",
		  "2025-02-27_M_e",
		  "2025-02-27_P_e",
		  "2025-03-05_K_e",
		  "2025-03-05_N_e"
 		 ]
#files = ["2025-02-12_Z_e"]

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
	
# save file path
savefilename = '4shot_analysis_results_' + state_selection + '_testing.xlsx'
savefile = os.path.join(proj_path, savefilename)

pkl_file = os.path.join(proj_path, '4shot_results_testing.pkl')

correct_spinloss = True
saturation_correction = True
local_time_analysis = False
pwave_ac_loss = True
threshold_filter = False

# Omega^2 [kHz^2] 1/e saturation value fit from 09-26_F
dimer_x0 = 1200.86  # 0.6 ToTF 2025-02-13
#dimer_x0 = 3300
dimer_e_x0 = 1030.23
# Omega^2 [kHz^2] 1e saturation value fit from 09-17_C
# HFT_x0 = 805.2923

# omega^2 [kHz^2] 1e saturation average fit value from various ToTF
HFT_x0 = 897.441
HFT_e_x0 = 35

def saturation_scale(x, x0):
	""" x is OmegaR^2 and x0 is fit 1/e Omega_R^2 """
	return x/x0*1/(1-np.exp(-x/x0))
	
### constants
re = 107 * a0 # ac dimer range estimate
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra
kF = 1.1e7
kappa = np.sqrt((Eb*h*10**6) *mK/hbar**2) # convert Eb back to kappa

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

def xstar(B, EF):
	return Eb/EF # hbar**2/mK/a13(B)**2 * (1-re/a13(Bfield))**(-1)

def linear(x, a, b):
	return a*x + b

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

def sinc2(x, trf):
	"""sinc^2 normalized to sinc^2(0) = 1"""
	t = x*trf
	return np.piecewise(t, [t==0, t!=0], [lambda t: 1, 
					   lambda t: (np.sin(np.pi*t)/(np.pi*t))**2])

### plot settings
plt.rcParams.update(plt_settings) # from library.py

plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})

### Summary plot lists
results_list = []

try:
	with open(pkl_file, 'rb') as f:
	    old_results_list = pkl.load(f)
except (OSError, IOError) as e:
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
	
	meta_df = metadata.loc[metadata.filename == filename].reset_index()
	if meta_df.empty:
		print("Dataframe is empty! The metadata likely needs updating." )
		continue
	
	filename = filename + ".dat"
	
	### Vpp calibration
	if filename[:4] == '2024':
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
			'e_ToTF':np.sqrt(meta_df['ToTF_i_sem'][0]**2 + \
							 meta_df['ToTF_f_sem'][0]**2)/2,
			'EF': (meta_df['EF_i'][0] + meta_df['EF_f'][0])/2,
			'e_EF': np.sqrt(meta_df['EF_i_sem'][0]**2 + \
					   meta_df['EF_f_sem'][0]**2)/2,
			'barnu': meta_df['barnu'][0],
			'e_barnu': meta_df['barnu_sem'][0],
			'ff': meta_df['ff'][0],
			'year': filename[:4]
			}
	
	results['x_star'] = xstar(meta_df['Bfield'][0], results['EF'])
	results['kF'] = np.sqrt(2*mK*results['EF']*h*1e6)/hbar
	
	micrO_OmegaR = 2*pi*VpptoOmegaR43*meta_df['Vpp_dimer'][0]
	
	### THEORY
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
			results['C_theory_std'] = 0.01
		
	# clock shift theory prediction from C_Theory
	results['CS_theory'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
							* results['C_theory']
	# note this is missing kF uncertainty, would have to do the above again
	results['CS_theory_std'] = 1/(pi*results['kF']*a13(meta_df['Bfield'][0])) \
							* results['C_theory_std']
							
	# correct cloud size
	size_names = ['two2D_sv1', 'two2D_sh1', 'two2D_sv2', 'two2D_sh2']
	new_size_names = ['c5_sv', 'c5_sh', 'c9_sv', 'c9_sh']
	for new_name, name in zip(new_size_names, size_names):
		run.data[new_name] = np.abs(run.data[name])

	# average H and V sizes
	run.data['c5_s'] = (run.data['c5_sv']+run.data['c5_sh'])/2
	run.data['c9_s'] = (run.data['c9_sv']+run.data['c9_sh'])/2
	
	
	### DATA FILTERING
	# remove indices if requested
	remove_indices = remove_indices_formatter(meta_df['remove_indices'][0])
	if remove_indices is not None:
		run.data.drop(remove_indices, inplace=True)
	
	# data filtering:
	if Filter == True:
		# filter out cloud fits that are too large
		filter_indices = run.data.index[run.data['sum95'] < 50].tolist()
		run.data.drop(filter_indices, inplace=True)
		
	# compute number of total data points
	num = len(run.data)
	
	### CALCULATIONS	
	# compute detuning
	if filename == "2025-02-18_H_e" or filename == "2025-02-18_K_e":
		# dumb exclusions for 02-18 testing of different dimer freqs
		#exclude_indices = run.data[(run.data['freq'] == 43.248) & (run.data['VVA']==4)].index
		exclude_test_freqs = run.data[run.data['freq']==43.218 or \
								   run.data['freq']==43.23 or \
								   run.data['freq']==43.278].index
		run.data = run.data.drop(exclude_test_freqs)
	 
	run.data['detuning'] = run.data['freq'] - meta_df['res_freq'][0] # MHz
	
	
	# fudge the c9 counts using ff
	run.data['c9'] = run.data['c9'] * results['ff']
	run.data['sum95'] = run.data['c5'] + run.data['c9']
	run.data['ratio95'] = run.data['c9']/run.data['c5']
	run.data['f5'] = run.data['c5']/run.data['sum95']
	run.data['f9'] = run.data['c9']/run.data['sum95']
	
	# split df into each type of measurement
	HFT_df = run.data.loc[(run.data.state==75) & (run.data.freq-meta_df['res_freq'][0]>0)]
	bg_75_df = run.data.loc[(run.data.state==75) & (run.data.freq-meta_df['res_freq'][0]<0)]
	dimer_75_df = run.data.loc[(run.data.state==75)]
	dimer_97_df = run.data.loc[(run.data.state==97) & (run.data.VVA>0)]
	bg_97_df = run.data.loc[(run.data.state==97) & (run.data.VVA==0)]
	bg_95_df = run.data.loc[(run.data.state==95)]
	
	# compute Omega R
	OmegaR_dimer = micrO_OmegaR  # in kHz
	OmegaR_HFT = phaseO_OmegaR(HFT_df.VVA.values[0], HFT_df.freq.values[0]) * np.sqrt(0.31)
	
	if saturation_correction:
		# the fit parameters assume Omega^2 is in kHz^2... "sorry"
		if results['ToTF'] < 0.35:
			sat_scale_HFT = saturation_scale(OmegaR_HFT**2/(2*np.pi)**2, 737)
			sat_scale_dimer = saturation_scale(OmegaR_dimer**2/(2*np.pi)**2, 3000)
		else:
			sat_scale_HFT = saturation_scale(OmegaR_HFT**2/(2*np.pi)**2, HFT_x0)
			sat_scale_dimer = saturation_scale(OmegaR_dimer**2/(2*np.pi)**2, 3600)
	else:
		sat_scale_HFT = 1
		sat_scale_dimer = 1
	
	### HFT
	# detuning in HFT
	HFT_detuning = HFT_df.detuning.values[0]
	
	c5 = np.array(HFT_df.c5)
	c9 = np.array(HFT_df.c9)
	
	# compute contact from HFT transfer
	# defined here because it's easier to propagate error
	def contact(bg, EF, measure='transfer'):
		if measure == 'transfer':
			transfer = np.mean((c5-bg)/(c5-bg+c9))
		elif measure == 'loss':
			transfer = np.mean(np.ones(len(c9))-c9/bg)
		
		transfer = transfer * sat_scale_HFT
		
		scaledtransfer = GammaTilde(transfer, h*EF*1e6, OmegaR_HFT*1e3, 
							  results['trf_blackman']/1e6)
		C = 2*np.sqrt(2)*pi**2*scaledtransfer*(HFT_detuning/EF)**(3/2) * \
					(1+HFT_detuning/EF/results['x_star'])
		return C
	
	# calculate averages and sems
	bg_c5 = bg_75_df.c5.mean()
	e_bg_c5 = bg_75_df.c5.sem()
	bg_c9 = bg_75_df.c9.mean()
	e_bg_c9 = bg_75_df.c9.sem()
	
	results['C'] = contact(bg_c5, results['EF'], measure='transfer')
	results['C_loss'] = contact(bg_c9, results['EF'], measure='loss')
	
	# now, to incorperate uncertainty in the bg and EF, we are going to sample
	# calculations selecting inputs in normal distributions
	inputs = [bg_c5, results['EF']]
	errors = [e_bg_c5, results['e_EF']]
	results['C_HFT'], results['C_HFT_std'] = MonteCarlo_estimate_std_from_function(contact, 
								 inputs, errors, num=100, measure='transfer')
	# again for loss
	inputs = [bg_75_df.c9.mean(), results['EF']]
	errors = [bg_75_df.c9.sem(), results['e_EF']]
	results['C_loss_HFT'], results['C_loss_HFT_std'] = MonteCarlo_estimate_std_from_function(contact, 
								  inputs, errors, num=100, measure='loss')
	
	if Talk == True:
		print("T/TF = {:.2f}, EF = {:.1f}kHz".format(results['ToTF'], results['EF']*1e3))
		print("raw C = {:.2f} and raw loss C = {:.2f}".format(results['C'], results['C_loss']))
		print("HFT C = {:.3f}±{:.3f}".format(results['C_HFT'], results['C_HFT_std']))
		print("HFT loss C = {:.3f}±{:.3f}".format(results['C_loss_HFT'], results['C_loss_HFT_std']))
		
		
	### Dimer
	if state_selection == '97':
		dimer_df = dimer_97_df
		bg_df = bg_97_df
	elif state_selection == '75':
		dimer_df = dimer_75_df
		bg_df = bg_75_df
	elif state_selection == '9':
		dimer_df = dimer_97_df
		bg_df = bg_95_df
	else:
		raise ValueError("State selection is incorrect, choose 97, 75, or 9")
		
	# This block tries to align the indices of bg and signal dfs
	# This allows simple series-on-series operations and computation of transfer-per-pair \
		# rather than averaging the dataset and then calculating transfer.
	if local_time_analysis:
		index_spacing = bg_df.index[0]-dimer_df.index[0] # usually this is consistent....
		bg_df.index = bg_df.index-index_spacing # align indices of bg_df and dimer_df
		# usually due to imaging errors, some shots get removed from analysis.
		# If a signal shot does not have a corresponding bg, then just get rid of both 
		bg_set = set(bg_df.index)
		dimer_set = set(dimer_df.index)
	
		unique_to_bg = bg_set - dimer_set
		unique_to_dimer = dimer_set - bg_set
		# Combine the results
		indices_to_remove = list(unique_to_bg) + list(unique_to_dimer)
		
		dimer_df = dimer_df.drop(indices_to_remove, errors='ignore')
		bg_df = bg_df.drop(indices_to_remove, errors='ignore')
	
	title = f'{filename} dimer transfer at ' + r'$T/T_F=$'+'{:.3f}±{:.3f}'.format(results['ToTF'], results['e_ToTF']) +\
			 ' using states ' + state_selection
	dimer_freq = dimer_df.freq.values[0] - meta_df['res_freq'][0]
	
	if lineshape == 'conv':
		# load lineshape
		df_ls = pd.read_pickle('./clockshift/convolutions.pkl')
		TTF = round(results['ToTF'], 1)
		if TTF == 0.2:
			TTF = 0.25
		if TTF == 0.7:
			TTF = 0.6
		TRF = results['trf_dimer']
		# input is hf/EF
		conv_lineshape = df_ls.loc[(df_ls['TTF']==TTF) & (df_ls['TRF']==TRF)]['LS'].values[0]
		
		# turn convolution lineshape into function
		def conv_func(x, A, x0):
			return A*conv_lineshape(x-x0)/conv_lineshape(0)
	
		lineshape_func = conv_func
	
	else: # default to sinc2
		def sinc2_func(x, A, x0):
			# input is hf/EF
			return A*sinc2((x-x0)*results['EF'], results['trf_dimer'])
		TTF = results['ToTF']
		lineshape_func = sinc2_func
	
	# prepping evaluation ranges
	xrange = 0.6
	xlow = dimer_freq - xrange
	xhigh = dimer_freq + xrange
	xnum = 1000
	xx = np.linspace(xlow, xhigh, xnum)/results['EF']
	
	# compute amplitude of lineshape from data
	fig, axs = plt.subplots(3,3, figsize=[12,10])
	axes = axs.flatten()
	ax_fit = axes[2]
	ax_fit.set(xlabel=r"Detuning $\hbar\omega/E_F$",
		   ylabel=r"Scaled Transfer $\tilde\Gamma$")
	
	# loop over spins, and plot counts vs time, and compute averages
	for i, spin, sty in zip([3, 4, 5], spins, styles):
		
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
		
		ax.legend()
		
	
	# loop over spins
	ax_hist = axes[6]
	for spin, sty, color in zip(spins, styles, colors):
		
		if spin == 'c5' or spin == 'c9':
			# compute averages
			counts = dimer_df[spin].mean()
			e_counts = dimer_df[spin].sem()
			bg_counts = bg_df[spin].mean()
			e_bg_counts = bg_df[spin].sem()
			# compute transfer from averaging whole dataset
			transfer = (1 - counts/bg_counts)
			e_transfer = counts/bg_counts * np.sqrt((e_counts/counts)**2 + \
									  (e_bg_counts/bg_counts)**2)
			
			# compute transfer-by-pair of bg and signal
			dimer_df[spin+'_transfer'] = 1-dimer_df[spin]/bg_df[spin]
# 			dimer_df['e_'+spin+'_transfer'] = dimer_df[spin+'_transfer']* np.sqrt((e_counts/dimer_df[spin])**2 + \
# 																	(e_bg_counts/bg_df[spin])**2)
			dimer_df['e_'+spin+'_transfer'] = dimer_df[spin+'_transfer'].sem()
				
			bg_df[spin+'_transfer'] = 0
			bg_df['e_'+spin+'_transfer'] = 0
			# correct spin b
			if correct_spinloss and spin == 'c5':
				transfer = transfer/2
				e_transfer = e_transfer/2
				dimer_df[spin+'_transfer'] = dimer_df[spin+'_transfer']/2
				dimer_df['e_'+spin+'_transfer'] = dimer_df['e_'+spin+'_transfer']/2
	
			print(spin_map(spin)+" counts = {:.2f}±{:.2f}k, ".format(counts/1e3, e_counts/1e3) + \
				"bg = {:.2f}±{:.2f}k, ".format(bg_counts/1e3, e_bg_counts/1e3) + \
					"raw transfer = {:.2f}±{:.2f}".format(bg_counts-counts, np.sqrt(e_counts**2 + e_bg_counts**2)))
				
		elif spin == 'ratio95':
			# defined ratio as ratio of fractional transfer
# 			dimer_df['fRab'] = dimer_df['ratio95']/bg_df['ratio95']
# 			dimer_df[spin+'_transfer'] = (bg_df['f9'] - dimer_df['fRab']*bg_df['ratio95']*bg_df['f5'])\
# 					/(1/2- dimer_df['fRab']*bg_df['ratio95'])
# 			bg_df[spin+'_transfer'] = (bg_df['f9'] - 1*bg_df['ratio95']*bg_df['f5'])\
# 				 	/(1/2-1*bg_df['ratio95'])
			# the "usual" way
			bg_f9_mean = bg_df['f9'].mean()
			bg_f5_mean = bg_df['f5'].mean()
			bg_f9_upp = bg_f9_mean + bg_df['f9'].sem()
			bg_f9_low = bg_f9_mean - bg_df['f9'].sem()
			bg_f5_upp = bg_f5_mean + bg_df['f5'].sem()
			bg_f5_low = bg_f5_mean - bg_df['f5'].sem()
			
			# point by point
			#dimer_df[spin+'_transfer'] = (bg_df['f9'] - dimer_df['ratio95']*bg_df['f5'])\
			#		/(1/2-dimer_df['ratio95'])
			#bg_df[spin+'_transfer'] = (bg_df['f9'] -bg_df['ratio95']*bg_df['f5'])\
 			#		/(1/2-bg_df['ratio95'])
					
			# point by point for signal but background is mean
			dimer_df[spin+'_transfer'] = (bg_f9_mean - dimer_df['ratio95']*bg_f5_mean)\
					/(1/2-dimer_df['ratio95'])
			bg_df[spin+'_transfer'] = (bg_f9_mean -bg_df['ratio95']*bg_f5_mean)\
 					/(1/2-bg_df['ratio95'])
			
 					
			# THIS IS INCORRECT BUT A PLACEHOLDER
			#dimer_df['e_'+spin+'_transfer'] = dimer_df[spin+'_transfer'].sem()
			#bg_df['e_'+spin+'_transfer'] = bg_df[spin+'_transfer'].sem()
			transfer_upp = (bg_f9_upp - dimer_df['ratio95']*bg_f5_low)\
					/(1/2-dimer_df['ratio95'])
			dimer_df['e_' + spin + '_transfer'] = np.abs( transfer_upp - dimer_df[spin+'_transfer'])
			bg_transfer_upp = (bg_f9_upp - bg_df['ratio95']*bg_f5_low)\
					/(1/2-bg_df['ratio95'])
			bg_df['e_' + spin + '_transfer'] =np.abs( bg_transfer_upp - bg_df[spin+'_transfer'])
					
			
			print(spin_map(spin)+" ratio = {:.2f}±{:.2f}, ".format(dimer_df[spin].mean(), dimer_df[spin].sem())  + \
				"bg = {:.2f}±{:.2f}".format(bg_df[spin].mean(), bg_df[spin].sem()))


		# correct transfer from fit saturation scaling
		dimer_df[spin+'_transfer'] = dimer_df[spin+'_transfer'] * sat_scale_dimer
		dimer_df['e_' + spin +'_transfer'] = dimer_df['e_' + spin +'_transfer'] * sat_scale_dimer
		bg_df[spin+'_transfer'] = bg_df[spin+'_transfer'] * sat_scale_dimer
		bg_df['e_' + spin +'_transfer'] = bg_df['e_' + spin +'_transfer'] * sat_scale_dimer
		
		mean_or_median = 'mean'
		if mean_or_median == 'mean':
			transfer = dimer_df[spin+'_transfer'].mean()
			e_transfer = dimer_df['e_' + spin +'_transfer'].mean()
		elif mean_or_median == 'median':
			transfer = dimer_df[spin+'_transfer'].median()
			e_transfer = dimer_df['e_' + spin +'_transfer'].median() 
		
		# put in results dict
		results['transfer_'+spin] = transfer
		results['e_transfer_'+spin] = e_transfer
		
		scaledtransfer = GammaTilde(transfer, h*results['EF']*1e6, OmegaR_dimer*1e3, 
							  results['trf_dimer']/1e6)
		e_scaledtransfer = e_transfer/transfer * scaledtransfer
		
		yy = lineshape_func(xx, scaledtransfer, dimer_freq/results['EF'])
		results['SW_'+spin] = np.trapz(yy, xx)
		results['e_SW_'+spin] = e_scaledtransfer/scaledtransfer*results['SW_'+spin]
		results['FM_'+spin] = np.trapz(yy*xx, xx)
		results['e_FM_'+spin] = np.abs(e_scaledtransfer/scaledtransfer*results['FM_'+spin])
		
		ax_fit.plot(xx, yy, '--', label=spin_map(spin))
		ax_fit.errorbar([dimer_freq/results['EF'], dimer_freq/results['EF']], 
			  [0, scaledtransfer], yerr=e_scaledtransfer*np.array([1,1]), **sty)
		
		ax_hist.hist(dimer_df[spin+'_transfer'], alpha = 0.5, label=spin_map(spin), color=color)
		ax_hist.vlines(transfer, ymin=0, \
				 ymax=10, ls = 'dashed', lw=3, color=color)
		
	
	raw_transfer_a = bg_df['c9'].mean() - dimer_df['c9'].mean()
	raw_transfer_b = bg_df['c5'].mean() - dimer_df['c5'].mean()
	e_raw_transfer_a = np.sqrt(bg_df['c9'].sem()**2 + dimer_df['c9'].sem()**2)
	e_raw_transfer_b = np.sqrt(bg_df['c5'].sem()**2 + dimer_df['c5'].sem()**2)
	
	raw_transfer_fraction = raw_transfer_a/raw_transfer_b
	e_raw_transfer_fraction = raw_transfer_fraction * np.sqrt((e_raw_transfer_a/raw_transfer_a)**2 + \
														   (e_raw_transfer_b/raw_transfer_b)**2)
											
	print("raw transfer a/b = {:.2f}+/-{:.2f}".format(raw_transfer_fraction, e_raw_transfer_fraction))
	
	results['raw_transfer_fraction'] = raw_transfer_fraction
	results['e_raw_transfer_fraction']=e_raw_transfer_fraction
	
	# plot ratio vs time
	spin = 'ratio95'
	ax = axes[0]
	ax.set(xlabel="Time [min]", ylabel='ratio a/b')

	# signal
	x = dimer_df.cyc*31/60
	y = dimer_df[spin]
	ymean = y.mean()
	ylist.append(ymean)
	popt, pcov = curve_fit(linear, x, y)
	ax.plot(x, y, label='signal', **styles[0])
	ax.plot(x, linear(x, *popt), '--')
	
	# bg
	x = bg_df.cyc*31/60
	y = bg_df[spin]
	popt, pcov = curve_fit(linear, x, y)
	ax.plot(x, y, label='bg', **styles[1])
	ax.plot(x, linear(x, *popt), '-.')
	
	ax.legend()
	
	# plot transfer vs time
	ax = axes[1]
	ax.set(xlabel="Time [min]", ylabel='transfer')

	# signal
	for spin, sty in zip(spins, styles):
		x = dimer_df.cyc*31/60
		y = dimer_df[spin+'_transfer']
		yerr = dimer_df['e_' + spin + '_transfer']
		popt, pcov = curve_fit(linear, x, y)
		ax.errorbar(x, y, yerr, label=spin, **sty)
		ax.plot(x, linear(x, *popt), '--')
		# bg
# 		x = bg_df.cyc*31/60
# 		y = bg_df[spin+'_transfer']
# 		popt, pcov = curve_fit(linear, x, y)
# 		ax.plot(x, y, label='bg', **sty)
# 		ax.plot(x, linear(x, *popt), '-.')
	
	ax.legend()
	ax_hist.legend()
	ax_fit.legend()
	
	# generate table
	ax_table = axes[-1]
	ax_table.axis('off')
	quantities = [r'$T/T_F$', r'$\Delta T/T_F$', r'$E_F$', 'conv $T/T_F$', 
			   'sat_scale HFT', 'sat_scale dimer']
	values = ["{:.3f}({:.0f})".format(results['ToTF'], results['e_ToTF']*1e3),
		   "{:.3f}({:.0f})".format(results['ToTF_diff'], results['e_ToTF']*1e3*2),
		   "{:.1f}({:.0f}) kHz".format(results['EF']*1e3, results['e_EF']*1e4),
		   "{:.3f}".format(TTF),
		   "{:.4f}".format(sat_scale_HFT),
		   "{:.4f}".format(sat_scale_dimer),
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
###### Summary Plotting #######
###############################	

# plotting options
dimertype2024='ratio95'
dimertype2025 ='ratio95'
plot_options = {"SR Fudge Theory":False,
				"SR Fudge Exp":False,
				"Compare SW and FM": False,
				"Loss Contact": False,
				"Plot Fits":False,
				"Open Channel Fraction": True,
				"Sum Rule 0.5": True,
				"Raw transfer fraction 0.5": False,
				"Binned":False}


true_options = []
for key, val in plot_options.items():
	if val:
		true_options.append(key)
			
title_end = ' , '.join(true_options)
if title_end != '':
	title_end = " with " + title_end
plot_title = "Four shot analysis" + title_end

# convert results into dataframe
df_total = pd.DataFrame(results_list)

# choose contact
if plot_options['Loss Contact'] == False:
	df_total['C_data'] = df_total['C_HFT']
	df_total['C_data_std'] = df_total['C_HFT_std']
else:
	df_total['C_data'] = df_total['C_loss_HFT']
	df_total['C_data_std'] = df_total['C_loss_HFT_std']
	
# contact fudge due to 0.35 sumrule
if plot_options['SR Fudge Theory'] or plot_options['SR Fudge Exp']:
	SR = 0.38 + 0.02
	SR_fudge = 0.5/SR
else:
	SR_fudge = 1

if plot_options['SR Fudge Exp']:
	df_total['C_data'] = np.array(df_total['C_data']) * SR_fudge
	df_total['C_data_std'] = np.array(df_total['C_data_std']) * SR_fudge 

if plot_options['Raw transfer fraction 0.5']:
	raw_threshold = 0.075
	df_total['rtf'] = np.where((df_total['raw_transfer_fraction'] >= (0.5-raw_threshold)) & \
					(df_total['raw_transfer_fraction'] <= (0.5 + raw_threshold)),
					1, 0)

### THEORY CALCULATIONS ####
# calculate theoretical sum rule and first moment vs contact
Bfield = 202.14

# spectral weight is defined differently between notes and lab 
sum_rule = 0.5 if plot_options['Sum Rule 0.5'] else 1
open_channel_fraction = 0.93 if plot_options["Open Channel Fraction"] else 1


C = np.linspace(0, max(df_total['C_data']), 50) 
kF = np.mean(df_total['kF'])
a13kF = kF * a13(202.14)

### ZY single-channel square well w/ effective range
# divide I_d by a13 kF,
not_small_kappa_correction = 1.08
I_d = sum_rule * kF/(pi*kappa) * 1/(1+re/a13(Bfield)) * C / a13kF * open_channel_fraction  * not_small_kappa_correction
# compute clock shift
#CS_d = sum_rule*-2*kappa/(pi*kF) * (1/1+re/a13(Bfield)) * C 
CS_d = -2*I_d *a13kF /kF**2 * kappa**2 # convert I_d to CS_d to avoid rewriting sum_rule and o_c_f
# multiply FM (Eq. 7) by a13 kF
FM_d =  CS_d * a13kF  # assumes ideal SR=0.5, factor of 2 already accounted for if sum_rule=0.5
	
### PJ CCC
# spectral weight, clock shift, first moment
I_d_CCC = sum_rule*kF* 42 *a0 * C / a13kF 
CS_d_CCC = -2*I_d_CCC *a13kF /kF**2 * kappa**2
FM_d_CCC = CS_d_CCC * a13kF # assumes ideal SR=0.5
### CCC w/ spin ME correction
spin_me_corr = 32/42
I_d_CCC_sc = sum_rule*kF* 42 *a0 * C / a13kF * spin_me_corr 
CS_d_CCC_sc = -2*I_d_CCC_sc *a13kF /kF**2 * kappa**2
FM_d_CCC_sc = CS_d_CCC_sc * a13kF # assumes ideal SR=0.5

### Other analytical models for bounding
I_d_max = sum_rule*kF* 1/pi * a13(Bfield) * C / a13kF  # shallow bound state 
I_d_min = sum_rule*kF* 1/(pi*kappa) * 1/(1+re*kappa) * C/a13kF  # another version of square well with eff range
CS_d_max = -2*I_d_max *a13kF /kF**2 * kappa**2
CS_d_min = -2*I_d_min *a13kF /kF**2 * kappa**2
FM_d_max = CS_d_max * a13kF # assumes ideal SR=0.5
FM_d_min = CS_d_min * a13kF # assumes ideal SR=0.5
# a13 times kF # redefined compared to a few lines earlier?
df_total['a13kF'] = np.array(df_total['kF']) * a13(202.14)

if plot_options['Binned']:
	#num_bins=8
	#df_total['bins'] = pd.cut(df_total['ToTF'], num_bins)
	bins = [0.2,0.3,0.4,0.5,0.6,0.8,1]
	df_total['bins'] = pd.cut(df_total['ToTF'], bins=bins)
	df_means = df_total.groupby('bins')[['C_data', 'SW_ratio95', 'SW_c5', 'SW_c9', 'FM_ratio95', 'FM_c5', 'FM_c9', 'a13kF', 'ToTF', 'ToTF_diff','raw_transfer_fraction']].mean().reset_index()
	df_errs = df_total.groupby('bins')[['C_data_std', 'e_SW_ratio95', 'e_SW_c5', 'e_SW_c9', 'e_FM_ratio95', 'e_FM_c5', 'e_FM_c9', 'e_raw_transfer_fraction']].apply(
		lambda x: np.sqrt((x**2).sum())/len(x)).reset_index()
	df_years = df_total.groupby('bins')[['year']].apply(lambda x: x).reset_index().drop('level_1', axis=1)
	df_total = pd.merge(df_means, df_errs, on='bins')
	df_total = pd.merge(df_total, df_years)
	#df_total['C_data'] = df_total['bins'].apply(lambda x: (x.left + x.right)/2).astype(float)

	

# split df
labels = ['2024', '2025']
if plot_options['Raw transfer fraction 0.5'] :
	labels = [0,1]
dfs = []

for label in labels:
	if plot_options['Raw transfer fraction 0.5']:
		dfs.append(df_total.loc[df_total.rtf==label])
	else:
		dfs.append(df_total.loc[df_total.year==label])

### intitialize plots
#fig, axs = plt.subplots(2,3, figsize=[14,7])
fig, axs=plt.subplots(1,3, figsize=[14, 7])
axes = axs.flatten()

#-- spectral weight vs. C
ax = axes[0]
ax.set(xlabel=r"Contact (kF/N)",
 		   ylabel=r"Spectral weight $I_d/k_Fa_{13}$")

i = 0

# theory curves
theory_labels = [r'1ch Sq, re', r'CCC', r'CCC w/ spin corr.', r'shallow dimer']
ax.plot(C, I_d, '-', color=colors[2], label=theory_labels[0])
#ax.plot(C, I_d_CCC, '-', color=colors[3], label=theory_labels[1])
ax.plot(C, I_d_CCC_sc, '--', color=colors[3], label=theory_labels[2])
if plot_options["SR Fudge Theory"]:
	ax.plot(C, I_d_CCC * SR_fudge, '-', color=colors[3])
	ax.fill_between(C, I_d_CCC*SR_fudge, I_d_CCC, color=colors[3], alpha=0.1)
	ax.plot(C, I_d *SR_fudge, '-', color=colors[2])
	ax.fill_between(C, I_d*SR_fudge, I_d, color=colors[2], alpha=0.1)
	ax.plot(C, I_d_CCC_sc * SR_fudge, '--', color=colors[3])
	ax.fill_between(C, I_d_CCC_sc*SR_fudge, I_d_CCC_sc, color=colors[3], alpha=0.1)

# data
for df, sty, label in zip(dfs, styles, labels):
	x = df['C_data']
	xerr = df['C_data_std']
	if df['year'].values[0] == '2024':
		dimertype= dimertype2024
	elif df['year'].values[0] == '2025':
		dimertype = dimertype2025
	y = df['SW_' +dimertype] / df['a13kF']
	yerr = np.abs(df['e_SW_'+dimertype]) / df['a13kF']
	ax.errorbar(x, y, yerr=yerr, xerr=df['C_data_std'],
		 label=label, **sty)
	
	ax.legend()
	
 	# fit to linear
	popt, pcov = curve_fit(linear, x, y, sigma=yerr)
	perr = np.sqrt(np.diag(pcov))
	xs = np.linspace(0, max(x), 100)
	if plot_options["Plot Fits"]: ax.plot(xs, linear(xs, *popt), '--', color=colors[i], label='fit') 
	print("dimer y-intercept = {:.3f}({:.0f})".format(popt[0], 1e3*perr[0]))
	 	
	 	# theory (i.e. 0 intercept)
	popt, pcov = curve_fit(lambda xx, a: linear(xx, a, 0), x, y, sigma=yerr)
	if plot_options["Plot Fits"]: ax.plot(xs, linear(xs, popt[0], 0), ':', color=colors[i], label='fix (0,0)')
	
	i += 1
	

#-- First moment vs. C
ax = axes[1]
ax.set(xlabel=r"Contact (kF/N)",
 		   ylabel=r"First moment $\tilde\Omega_d k_Fa_{13}$ ")

i = 0
label = 'C ratio'
	
# theory curve
ax.plot(C, FM_d, '-', color=colors[2], label=theory_labels[0])
#ax.plot(C, FM_d_CCC, '-', color=colors[3], label=theory_labels[1])
ax.plot(C, FM_d_CCC_sc, '--', color=colors[3], label=theory_labels[2])
if plot_options["SR Fudge Theory"]:
	ax.plot(C, FM_d *SR_fudge, '-', color=colors[2])
	ax.fill_between(C, FM_d*SR_fudge, FM_d, color=colors[2], alpha=0.1)
	ax.plot(C, FM_d_CCC * SR_fudge, '-', color=colors[3])
	ax.fill_between(C, FM_d_CCC*SR_fudge, FM_d_CCC, color=colors[3], alpha=0.1)
	ax.plot(C, FM_d_CCC_sc * SR_fudge, '--', color=colors[3])
	ax.fill_between(C, FM_d_CCC_sc*SR_fudge, FM_d_CCC_sc, color=colors[3], alpha=0.1)

# data
for df, sty, label in zip(dfs, styles, labels):
	x = np.array(df['C_data'])
	xerr = np.array(df['C_data_std']) 
	if df['year'].values[0] == '2024':
		dimertype=dimertype2024
	elif df['year'].values[0] == '2025':
		dimertype = dimertype2025
	y = np.array(df['FM_' + dimertype]) * df['a13kF']
	yerr = np.array(np.abs(df['e_FM_' + dimertype])) * df['a13kF']
	
	ax.errorbar(x, y, yerr=yerr, xerr=xerr, label=label, **sty)
	ax.legend()
	
	# fit to linear
	popt, pcov = curve_fit(linear, x, y, sigma=yerr)
	xs = np.linspace(0, max(x), 100)
	if plot_options["Plot Fits"]: ax.plot(xs, linear(xs, *popt), '--', color=colors[i])
	
	# theory (i.e. 0 intercept)
	popt, pcov = curve_fit(lambda xx, a: linear(xx, a, 0), x, y, sigma=yerr)
	if plot_options["Plot Fits"]: ax.plot(xs, linear(xs, popt[0], 0), ':', color=colors[i])

	i += 1
	
	
# Rescaling and plotting spectra weight vs. first moment
if plot_options["Compare SW and FM"]:
	ax = axes[2]
	ax.set(xlabel=r"Contact (kF/N)",
	 		   ylabel=r"$I_d$ or $\tilde\Omega_d$")
	i = 0
	for df, sty, label in zip(dfs, styles, labels):
		# spectra weight
		x = np.array(df['C_data'])
		xerr = np.array(df['C_data_std'])
		y_sw = np.array(df['SW_'+dimertype])/df['a13kF']
		yerr_sw = np.array(np.abs(df['e_SW_'+dimertype]))/df['a13kF']
		ax.errorbar(x, y_sw, yerr=yerr_sw, xerr=df['C_data_std'],
			 label=label, **sty)
		
		ax.legend()
		
		# fit to linear
		popt, pcov = curve_fit(linear, x, y_sw, sigma=yerr_sw)
		perr = np.sqrt(np.diag(pcov))
		xs = np.linspace(0, max(x), 100)
		if plot_options["Plot Fits"]: ax.plot(xs, linear(xs, *popt), '--', color=colors[i], label='fit')
		print("dimer y-intercept = {:.3f}({:.0f})".format(popt[0], 1e3*perr[0]))
		
		# theory (i.e. 0 intercept)
		popt, pcov = curve_fit(lambda xx, a: linear(xx, a, 0), x, y_sw, sigma=yerr_sw)
		if plot_options["Plot Fits"]: ax.plot(xs, linear(xs, popt[0], 0), ':', color=colors[i], label='fix (0,0)')
		A_sw = popt
		
		# now rescale first moment and plot
		y_fm = np.array(df['FM_' + dimertype]) * df['a13kF']
		yerr_fm = np.array(np.abs(df['e_FM_' + dimertype])) * df['a13kF']
		
		# fit to linear
		popt, pcov = curve_fit(linear, x, y_fm, sigma=yerr_fm)
		xs = np.linspace(0, max(x), 100)
		#ax.plot(xs, linear(xs, *popt), '--', color=colors[i])
		
		# theory (i.e. 0 intercept)
		popt, pcov = curve_fit(lambda xx, a: linear(xx, a, 0), x, y_fm, sigma=yerr_fm)
		#ax.plot(xs, linear(xs, popt[0], 0), ':', color=colors[i])
		sty = styles[2]
		ax.errorbar(x, A_sw*(y_fm/popt[0]), yerr=np.abs(A_sw*(yerr_fm/popt[0])), xerr=xerr, label=label, **sty)
		ax.legend()
		i += 1
	
	
#-- C vs. ToTF
ax = axes[2]
ax.set(ylabel=r"Contact ($k_F/N$)",
 		   xlabel=r"Temperature ($T_F$)")

i = 2
label = 'C ratio'
df = df_total
sty = styles[i]
x = df['ToTF']
# this is supposed to be the std dev of a uniform distribution of ToTFs that occur
# when the ToTF is changing linearly during the data run
xerr = np.abs(np.array(df['ToTF_diff'])*0.68) 
y = df['C_data']
yerr = df['C_data_std']
ax.errorbar(x, y, yerr=yerr, xerr=xerr, **sty)

# plot trap-averaged contact
xs = np.linspace(min(x), max(x), 100)
ax.plot(xs, C_interp(xs), ':', color=colors[i], label='Tilman Theory')
ax.legend()



# final plot settings
fig.suptitle(plot_title)
fig.tight_layout()
#plt.savefig('clockshift/df_plots/'+plot_title+'.png', dpi=300)
plt.show()	


### intitialize plots spectral weight vs. C for various analysis methods
#fig, axs = plt.subplots(2,3, figsize=[14,7])
fig, axs=plt.subplots(1,3, figsize=[14, 7])
axes = axs.flatten()


# data
for df, sty, label in zip(dfs, styles, labels):
	
	for i in range(3):
		#-- spectral weight vs. C
		if i == 0:
			dimertype= 'ratio95'
		elif i == 1:
			dimertype= 'c5'
		elif i==2:
			dimertype='c9'
		ax = axes[i]
		ax.set(xlabel=r"Contact (kF/N)",
		 		   ylabel=r"Spectral weight $I_d/k_Fa_{13}$",
					title = spin_map(dimertype))
		
		
		# theory curves
		theory_labels = [r'1ch Sq, re', r'CCC', r'CCC w/ spin corr.', r'shallow dimer']
		ax.plot(C, I_d, '-', color=colors[2], label=theory_labels[0])
		#ax.plot(C, I_d_CCC, '-', color=colors[3], label=theory_labels[1])
		ax.plot(C, I_d_CCC_sc, '--', color=colors[3], label=theory_labels[2])
		if plot_options["SR Fudge Theory"]:
			ax.plot(C, I_d_CCC * SR_fudge, '-', color=colors[3])
			ax.fill_between(C, I_d_CCC*SR_fudge, I_d_CCC, color=colors[3], alpha=0.1)
			ax.plot(C, I_d *SR_fudge, '-', color=colors[2])
			ax.fill_between(C, I_d*SR_fudge, I_d, color=colors[2], alpha=0.1)
			ax.plot(C, I_d_CCC_sc * SR_fudge, '--', color=colors[3])
			ax.fill_between(C, I_d_CCC_sc*SR_fudge, I_d_CCC_sc, color=colors[3], alpha=0.1)
	
		x = df['C_data']
		xerr = df['C_data_std']
		
		y = df['SW_' +dimertype] / df['a13kF']
		yerr = np.abs(df['e_SW_'+dimertype]) / df['a13kF']
		ax.errorbar(x, y, yerr=yerr, xerr=df['C_data_std'],
			 label=label, **sty)
		
		
	 	# fit to linear
		popt, pcov = curve_fit(linear, x, y, sigma=yerr)
		perr = np.sqrt(np.diag(pcov))
		xs = np.linspace(0, max(x), 100)
		if plot_options["Plot Fits"]: ax.plot(xs, linear(xs, *popt), '--', color=colors[i], label='fit') 
		print("dimer y-intercept = {:.3f}({:.0f})".format(popt[0], 1e3*perr[0]))
		 	
		 	# theory (i.e. 0 intercept)
		popt, pcov = curve_fit(lambda xx, a: linear(xx, a, 0), x, y, sigma=yerr)
		if plot_options["Plot Fits"]: ax.plot(xs, linear(xs, popt[0], 0), ':', color=colors[i], label='fix (0,0)')

	
fig.tight_layout()

fig, axs = plt.subplots(2)
for df, label in zip(dfs, labels):
	ax = axs[0]
	ax.hist(df['raw_transfer_fraction'], bins=30, label = label)
	ax = axs[1]
	ax.set(ylim=[0.3, 0.7],
		xlim=[0.2, 1.0],
		ylabel='a/b transfer fraction',
		xlabel='ToTF')
	ax.errorbar(df['ToTF'], df['raw_transfer_fraction'], df['e_raw_transfer_fraction'],ls='', label=label)
	ax.hlines(0.5, 0, 1, ls='dashed', color='red')

axs[1].legend()
