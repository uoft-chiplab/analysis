"""
Created by Chip lab 2024-10-17

Analysis script for triple shot scans: HFT, dimer, bg
"""

# paths
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')

from library import pi, h, hbar, mK, a0, plt_settings, markers, colors, \
					light_colors, dark_colors
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

# use **styles[i] in errorbar input to cycle through
styles = Cycler([{'color':dark_color, 'mec':dark_color, 'mfc':light_color,
					 'marker':marker} for dark_color, light_color, marker in \
						   zip(dark_colors, light_colors, markers)])
	

### This turns on (True) and off (False) saving the data/plots 
Save = True

### script options
Debug = False
Filter = True
Talk = True

Calc_CTheory_std = False

# select spin state imaging for analysis
state_selection = '97'

spins = ['c5', 'c9', 'sum95']
spins = ['c5', 'c9']

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
		 ]

def spin_map(spin):
	if spin == 'c5':
		return 'b'
	elif spin == 'c9':
		return 'a'
	elif spin == 'sum95':
		return 'a+b'
	
# save file path
savefilename = '4shot_analysis_results_' + state_selection + '.xlsx'
savefile = os.path.join(proj_path, savefilename)

### Vpp calibration
# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
VpptoOmegaR47 = 17.05/0.703 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
phaseO_OmegaR = lambda VVA, freq: 2*pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)
	
### contants
re = 107 * a0 # ac dimer range estimate
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra

def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0=224.2
	return abg*(1 - DeltaB/(B-B0))

def xstar(B, EF):
	return Eb/EF # hbar**2/mK/a13(B)**2 * (1-re/a13(Bfield))**(-1)

def linear(x, a, b):
	return a*x + b

def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

### plot settings
plt.rcParams.update(plt_settings) # from library.py

plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})

### Summary plot lists
results_list = []

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
	
	# create data structure
	filename = filename + ".dat"
	run = Data(filename, path=data_path)
	runfolder = filename 
	
	# initialize results dict to turn into df
	results = {}
	results['Run'] = filename
	results['trf_blackman'] = meta_df['trf_blackman'][0]*1e6
	results['trf_dimer'] = meta_df['trf_dimer'][0]*1e6
	results['ToTF_i'] = meta_df['ToTF_i'][0]
	results['ToTF_f'] = meta_df['ToTF_f'][0]
	results['ToTF_diff'] = meta_df['ToTF_f'][0]-meta_df['ToTF_i'][0]
	results['ToTF'] = (meta_df['ToTF_i'][0] + meta_df['ToTF_f'][0])/2
	results['e_ToTF'] = np.sqrt(meta_df['ToTF_i_sem'][0]**2 + \
							 meta_df['ToTF_f_sem'][0]**2)/2
	results['EF'] = (meta_df['EF_i'][0] + meta_df['EF_f'][0])/2
	results['e_EF'] = np.sqrt(meta_df['EF_i_sem'][0]**2 + \
						   meta_df['EF_f_sem'][0]**2)/2
	results['kF'] = np.sqrt(2*mK*results['EF']*h*1e6)/hbar
	results['barnu'] = meta_df['barnu'][0]
	results['e_barnu'] = meta_df['barnu_sem'][0]
	results['x_star'] = xstar(meta_df['Bfield'][0], results['EF'])
	
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
	run.data['detuning'] = run.data['freq'] - meta_df['res_freq'][0] # MHz
	
	# fudge the c9 counts using ff
	run.data['c9'] = run.data['c9'] * meta_df['ff'][0]
	run.data['sum95'] = run.data['c5'] + run.data['c9']
	
	# split df into each type of measurement
	HFT_df = run.data.loc[(run.data.state==75) & (run.data.freq-meta_df['res_freq'][0]>0)]
	bg_75_df = run.data.loc[(run.data.state==75) & (run.data.freq-meta_df['res_freq'][0]<0)]
	dimer_75_df = run.data.loc[(run.data.state==7)]
	dimer_97_df = run.data.loc[(run.data.state==97) & (run.data.VVA>0)]
	bg_97_df = run.data.loc[(run.data.state==97) & (run.data.VVA==0)]
	bg_95_df = run.data.loc[(run.data.state==95)]
	
	# compute Omega R
	OmegaR_dimer = micrO_OmegaR  # in kHz
	OmegaR_HFT = phaseO_OmegaR(HFT_df.VVA.values[0], HFT_df.freq.values[0]) * np.sqrt(0.31)
	
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
	
	title = f'{filename} dimer transfer at ' + r'$T/T_F=$'+'{:.3f}±{:.3f}'.format(results['ToTF'], results['e_ToTF']) +\
			 ' using states ' + state_selection
	dimer_freq = dimer_df.freq.values[0] - meta_df['res_freq'][0]
	
	# load lineshape
	df_ls = pd.read_pickle('./clockshift/convolutions.pkl')
	TTF = round(results['ToTF'], 1)
	if TTF == 0.2:
		TTF = 0.25
	if TTF == 0.7:
		TTF = 0.6
	TRF = results['trf_dimer']
	lineshape = df_ls.loc[(df_ls['TTF']==TTF) & (df_ls['TRF']==TRF)]['LS'].values[0]
	
	# turn convolution lineshape into function
	def convls(x, A, x0):
		return A*lineshape(x-x0)/lineshape(0)
	
	# prepping evaluation ranges
	xrange = 0.6
	xlow = dimer_freq - xrange
	xhigh = dimer_freq + xrange
	xnum = 1000
	xx = np.linspace(xlow, xhigh, xnum)/results['EF']
	
	# compute amplitude of lineshape from data
	fig, axs = plt.subplots(2,3, figsize=[12,7])
	axes = axs.flatten()
	ax_fit = axes[2]
	ax_fit.set(xlabel=r"Detuning $\hbar\omega/E_F$",
		   ylabel=r"Scaled Transfer $\tilde\Gamma$")
	
	for i, spin, sty in zip([0, 1], spins, styles):
		
		# plot counts vs time
		ax = axes[i]
		ax.set(xlabel="Time [min]", ylabel=spin_map(spin)+" Counts")
	
		# signal
		x = dimer_df.cyc*31/60
		y = dimer_df[spin]
		popt, pcov = curve_fit(linear, x, y)
		ax.plot(x, y, label='signal', **list(styles)[0])
		ax.plot(x, linear(x, *popt), '--')
		# bg
		x = bg_df.cyc*31/60
		y = bg_df[spin]
		popt, pcov = curve_fit(linear, x, y)
		ax.plot(x, y, label='bg', **list(styles)[1])
		ax.plot(x, linear(x, *popt), '-.')
		
		ax.legend()
		
		# plot size vs time
		ax = axes[i+3] # next row
		ax.set(xlabel="Time [min]", ylabel=spin_map(spin)+" Size [px]")
		
		# signal
		x = dimer_df.cyc*31/60
		y = dimer_df[spin+'_s']
		popt, pcov = curve_fit(linear, x, y)
		ax.plot(x, y, label='signal', **list(styles)[0])
		ax.plot(x, linear(x, *popt), '--')
		# bg
		x = bg_df.cyc*31/60
		y = bg_df[spin+'_s']
		popt, pcov = curve_fit(linear, x, y)
		ax.plot(x, y, label='bg', **list(styles)[1])
		ax.plot(x, linear(x, *popt), '--')
		
		ax.legend()
		
		# calculate averages
		counts = dimer_df[spin].mean()
		e_counts = dimer_df[spin].sem()
		bg_counts = bg_df[spin].mean()
		e_bg_counts = bg_df[spin].sem()
		cloud_size = dimer_df[spin+'_s'].mean()
		e_cloud_size = dimer_df[spin+'_s'].sem()
		bg_cloud_size = bg_df[spin+'_s'].mean()
		e_bg_cloud_size = bg_df[spin+'_s'].sem()
		
		print(spin_map(spin)+" counts = {:.2f}±{:.2f}k, ".format(counts/1e3, e_counts/1e3) + \
				"bg = {:.2f}±{:.2f}k".format(bg_counts/1e3, e_bg_counts/1e3))
		print(spin_map(spin)+" size = {:.2f}±{:.2f}, ".format(cloud_size, e_cloud_size) + \
				"bg = {:.2f}±{:.2f}".format(bg_cloud_size, e_bg_cloud_size))
			
		# compute transfer
		transfer = (1 - counts/bg_counts)
		e_transfer = counts/bg_counts * np.sqrt((e_counts/counts)**2 + \
								  (e_bg_counts/bg_counts)**2)
			
		scaledtransfer = GammaTilde(transfer, h*results['EF']*1e6, OmegaR_dimer*1e3, 
							  results['trf_dimer']/1e6)
		e_scaledtransfer = e_transfer/transfer * scaledtransfer
		
		yy = convls(xx, scaledtransfer, dimer_freq/results['EF'])
		results['SR_'+spin] = np.trapz(yy, xx)
		results['e_SR_'+spin] = e_scaledtransfer/scaledtransfer*results['SR_'+spin]
		results['FM_'+spin] = np.trapz(yy*xx, xx)
		results['e_FM_'+spin] = e_scaledtransfer/scaledtransfer*results['FM_'+spin]
		
		ax_fit.plot(xx, yy, '--', label=spin_map(spin))
		ax_fit.errorbar([dimer_freq/results['EF'], dimer_freq/results['EF']], 
			  [0, scaledtransfer], yerr=e_scaledtransfer*np.array([1,1]), **sty)
		
	ax_fit.legend()
	
	# generate table
	ax_table = axes[-1]
	ax_table.axis('off')
	quantities = [r'$T/T_F$', r'$\Delta T/T_F$', r'$E_F$', 'conv $T/T_F$']
	values = ["{:.3f}({:.0f})".format(results['ToTF'], results['e_ToTF']*1e3),
		   "{:.3f}({:.0f})".format(results['ToTF_diff'], results['e_ToTF']*1e3*2),
		   "{:.1f}({:.0f}) kHz".format(results['EF']*1e3, results['e_EF']*1e4),
		   "{:.3f}".format(TTF)
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
		
###############################
###### Summary Plotting #######
###############################	

	results_list.append(results)
	
# get all keys form reecnt results dict, make a new summary dict
keys = results.keys()
values = [[result[key] for result in results_list] for key in results.keys()]	
summary = dict(zip(keys, values))

spins = ['c9', 'c5']
 	
fig, axs = plt.subplots(2,2, figsize=[10,7])
axes = axs.flatten()

#-- Sumrule vs. C	
ax = axes[0]
ax.set(xlabel=r"Contact (kF/N)",
 		   ylabel=r"Sum Rule")

i = 0
label = 'C ratio'
for spin, sty in zip(spins, styles):
	x = summary['C_HFT']
	xerr = summary['C_HFT_std']
	y = summary['SR_'+spin]
	yerr = np.abs(summary['e_SR_'+spin])
	ax.errorbar(x, y, yerr=yerr, xerr=summary['C_HFT_std'],
		 label=spin_map(spin), **sty)
	
	ax.legend()
	
	# fit to linear
	popt, pcov = curve_fit(linear, x, y, sigma=yerr)
	xs = np.linspace(0, max(x), 100)
	ax.plot(xs, linear(xs, *popt), '--', color=colors[i], label='fit')
	# theory (i.e. 0 intercept)
	popt, pcov = curve_fit(lambda xx, a: linear(xx, a, 0), x, y, sigma=yerr)
	ax.plot(xs, linear(xs, popt[0], 0), ':', color=colors[i], label='fix (0,0)')
	
	i += 1
	

#-- First moment vs. C
ax = axes[1]
ax.set(xlabel=r"Contact (kF/N)",
 		   ylabel=r"First Moment")

i = 0
label = 'C ratio'
for spin, sty in zip(spins, styles):
	x = summary['C_HFT']
	xerr = summary['C_HFT_std']
	y = summary['FM_'+spin]
	yerr = np.abs(summary['e_FM_'+spin])
	
	ax.errorbar(x, y, yerr=yerr, xerr=xerr, label=spin_map(spin), **sty)
	
	# fit to linear
	popt, pcov = curve_fit(linear, x, y, sigma=yerr)
	xs = np.linspace(0, max(x), 100)
	ax.plot(xs, linear(xs, *popt), '--', color=colors[i])
	# theory (i.e. 0 intercept)
	popt, pcov = curve_fit(lambda xx, a: linear(xx, a, 0), x, y, sigma=yerr)
	ax.plot(xs, linear(xs, popt[0], 0), ':', color=colors[i])
	
	i += 1
	
#-- C vs. ToTF
ax = axes[2]
ax.set(ylabel=r"Contact ($k_F/N$)",
 		   xlabel=r"Temperature ($T_F$)")

i = 2
label = 'C ratio'
sty = list(styles)[i]

x = summary['ToTF']

# this is supposed to be the std dev of a uniform distribution of ToTFs that occur
# when the ToTF is changing linearly during the data run
xerr = np.abs(np.array(summary['ToTF_diff'])*0.68) 

y = summary['C_HFT']
yerr = summary['C_HFT_std']

ax.errorbar(x, y, yerr=yerr, xerr=xerr, **sty)

# plot trap-averaged contact
xs = np.linspace(min(x), max(x), 100)
ax.plot(xs, C_interp(xs), ':', color=colors[i], label='Tilman Theory')
ax.legend()

# final plot settings
fig.tight_layout()
plt.show()	
		