"""
Created by Chip lab 2024-10-17

Analysis script for triple shot scans: HFT, dimer, bg
"""

# paths
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data/full_spectrum')

from library import pi, h, hbar, mK, a0, plt_settings, styles, colors
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from scipy.optimize import curve_fit
from warnings import catch_warnings, simplefilter
from cycler import Cycler, cycler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

# select spin state simaging for analysis
state_selection = '97'

#spins = ['c5', 'c9', 'sum95']
spins = ['c5', 'c9', 'ratio95']

files = ["2025-02-14_E_e",
		 "2025-02-14_H_e"]

ToTFs = [0.631,
		 0.293]

EFs = [17.1,
	   11.2]

trfs = [0.010,
		0.010]

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

correct_spinloss = True


def dimer_transfer(Rab, fa, fb):
	'''computes transfer to dimer state assuming loss in b is twice the loss
		in a. Rba = Nb/Na for the dimer association shot. Nb_bg and Na_bg
		are determined from averaged bg shots.'''
	return (fa - Rab*fb)/(1/2-Rab)

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

for i, filename in enumerate(files):
	
##############################
######### Analysis ###########
##############################
	print("----------------")
	print("Analyzing " + filename)
	
	filename = filename + ".dat"
	
	# create data structure
	run = Data(filename, path=data_path)
	runfolder = filename 
	
	# compute number of total data points
	num = len(run.data)
	
	### CALCULATIONS	
	# compute detuning
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
	for spin, sty in zip(spins[:-1], styles):
		
		# compute averages
		counts = dimer_df[spin].mean()
		e_counts = dimer_df[spin].sem()
		bg_counts = bg_df[spin].mean()
		e_bg_counts = bg_df[spin].sem()
		
		dimer_df[spin+'_bg'] = bg_counts
		dimer_df['em_'+spin+'_bg'] = e_bg_counts
		
		print(spin_map(spin)+" counts = {:.2f}±{:.2f}k, ".format(counts/1e3, e_counts/1e3) + \
				"bg = {:.2f}±{:.2f}k".format(bg_counts/1e3, e_bg_counts/1e3))
			
		# compute transfer
		transfer = (1 - counts/bg_counts)
		e_transfer = counts/bg_counts * np.sqrt((e_counts/counts)**2 + \
								  (e_bg_counts/bg_counts)**2)
		
		# correct spin b
		if correct_spinloss:
			if spin == 'c5':
				transfer = transfer/2
				e_transfer = e_transfer/2
		
		# correct transfer from fit saturation scaling
		transfer = transfer * sat_scale_dimer
		e_transfer = e_transfer * sat_scale_dimer
		
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
		
	# compute and plot ratio transfer
	sty = styles[2]
	spin = 'ratio95'
	
	bg_f9 = bg_df['f9'].mean()
	e_bg_f9 = bg_df['f9'].sem()
	bg_f5 = bg_df['f5'].mean()
	e_bg_f5 = bg_df['f5'].sem()
	
	# compute transfer
# 	bg_f9 = 0.5
# 	bg_f5 = 0.5
	dimer_df['transfer'] = dimer_transfer(dimer_df[spin], bg_f9, bg_f5)
	# compute bg transfer where the inputed bg ratio is the mean
	bg_df['transfer'] = dimer_transfer(bg_df[spin], bg_f9, bg_f5)
	
	# correct transfer from fit saturation scaling
	dimer_df['transfer'] = dimer_df['transfer'] * sat_scale_dimer
	
	# plot ratio vs time
	ax = axes[0]
	ax.set(xlabel="Time [min]", ylabel='ratio a/b')

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
	
	# plot transfer vs time
	ax = axes[1]
	ax.set(xlabel="Time [min]", ylabel='transfer')

	# signal
	x = dimer_df.cyc*31/60
	y = dimer_df['transfer']
	popt, pcov = curve_fit(linear, x, y)
	ax.plot(x, y, label='signal', **styles[0])
	ax.plot(x, linear(x, *popt), '--')
	# bg
	x = bg_df.cyc*31/60
	y = bg_df['transfer']
	popt, pcov = curve_fit(linear, x, y)
	ax.plot(x, y, label='bg', **styles[1])
	ax.plot(x, linear(x, *popt), '-.')
	
	ax.legend()
	
	# average
	transfer = dimer_df.transfer.mean()
	# TODO fix the below to include both spins' bg error
	e_transfer = transfer*np.sqrt((dimer_df.transfer.sem()/transfer)**2 + \
					  (e_bg_f5/bg_f5)**2)
	
	# put in results dict
	results['dimer_transfer'] = transfer
	results['e_dimer_transfer'] = e_transfer
	
	scaledtransfer = GammaTilde(transfer, h*results['EF']*1e6, OmegaR_dimer*1e3, 
						  results['trf_dimer']/1e6)
	e_scaledtransfer = e_transfer/transfer * scaledtransfer
	
	yy = lineshape_func(xx, scaledtransfer, dimer_freq/results['EF'])
	results['SW_dimer'] = np.trapz(yy, xx)
	results['e_SW_dimer'] = e_scaledtransfer/scaledtransfer*results['SW_dimer']
	results['FM_dimer'] = np.trapz(yy*xx, xx)
	results['e_FM_dimer'] = np.abs(e_scaledtransfer/scaledtransfer*results['FM_dimer'])
	
	ax_fit.plot(xx, yy, '--', label=spin_map(spin))
	ax_fit.errorbar([dimer_freq/results['EF'], dimer_freq/results['EF']], 
		  [0, scaledtransfer], yerr=e_scaledtransfer*np.array([1,1]), **sty)
	
		
	# loop at cloud sizes vs. time
	for i, spin, sty in zip([6, 7], spins[:-1], styles):
		# plot size vs time
		ax = axes[i] # next row
		ax.set(xlabel="Time [min]", ylabel=spin_map(spin)+" Size [px]")
		
		# signal
		x = dimer_df.cyc*31/60
		y = dimer_df[spin+'_s']
		popt, pcov = curve_fit(linear, x, y)
		ax.plot(x, y, label='signal', **styles[0])
		ax.plot(x, linear(x, *popt), '--')
		# bg
		x = bg_df.cyc*31/60
		y = bg_df[spin+'_s']
		popt, pcov = curve_fit(linear, x, y)
		ax.plot(x, y, label='bg', **styles[1])
		ax.plot(x, linear(x, *popt), '--')
		
		ax.legend()
		
		# calculate averages
		cloud_size = dimer_df[spin+'_s'].mean()
		e_cloud_size = dimer_df[spin+'_s'].sem()
		bg_cloud_size = bg_df[spin+'_s'].mean()
		e_bg_cloud_size = bg_df[spin+'_s'].sem()
		
		print(spin_map(spin)+" size = {:.2f}±{:.2f}, ".format(cloud_size, e_cloud_size) + \
				"bg = {:.2f}±{:.2f}".format(bg_cloud_size, e_bg_cloud_size))
		
		
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
		   "{:.4f}".format(sat_scale_dimer)
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
plot_options = {"SR Fudge Theory":False,
				"SR Fudge Exp":False,
				"Compare SW and FM": False,
				"Loss Contact": False,
				"Plot Fits":False,
				"Open Channel Fraction": True,
				"Sum Rule 0.5": True}

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


# split df
labels = ['2024', '2025']
dfs = []

for label in labels:
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
ax.plot(C, I_d_CCC, '-', color=colors[3], label=theory_labels[1])
ax.plot(C, I_d_CCC_sc, '--', color=colors[3], label=theory_labels[2])
if plot_options["SR Fudge Theory"]:
	ax.plot(C, I_d_CCC * SR_fudge, '-', color=colors[3])
	ax.fill_between(C, I_d_CCC*SR_fudge, I_d_CCC, color=colors[3], alpha=0.1)
	ax.plot(C, I_d *SR_fudge, '-', color=colors[2])
	ax.fill_between(C, I_d*SR_fudge, I_d, color=colors[2], alpha=0.1)
	ax.plot(C, I_d_CCC_sc * SR_fudge, '--', color=colors[3])
	ax.fill_between(C, I_d_CCC_sc*SR_fudge, I_d_CCC_sc, color=colors[3], alpha=0.1)

#ax.plot(C, I_d_max, '-', color=colors[4], label=theory_labels[3])
#ax.plot(C, I_d_min, '-', color=colors[4], alpha=0.1)
#ax.fill_between(C, I_d_min, I_d_max, color=colors[4], alpha=0.1, label=theory_labels[3])
# data
for df, sty, label in zip(dfs, styles, labels):
	x = df['C_data']
	xerr = df['C_data_std']
	y = df['SW_dimer'] / df['a13kF']
	yerr = np.abs(df['e_SW_dimer']) / df['a13kF']
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
ax.plot(C, FM_d_CCC, '-', color=colors[3], label=theory_labels[1])
ax.plot(C, FM_d_CCC_sc, '--', color=colors[3], label=theory_labels[2])
if plot_options["SR Fudge Theory"]:
	ax.plot(C, FM_d *SR_fudge, '-', color=colors[2])
	ax.fill_between(C, FM_d*SR_fudge, FM_d, color=colors[2], alpha=0.1)
	ax.plot(C, FM_d_CCC * SR_fudge, '-', color=colors[3])
	ax.fill_between(C, FM_d_CCC*SR_fudge, FM_d_CCC, color=colors[3], alpha=0.1)
	ax.plot(C, FM_d_CCC_sc * SR_fudge, '--', color=colors[3])
	ax.fill_between(C, FM_d_CCC_sc*SR_fudge, FM_d_CCC_sc, color=colors[3], alpha=0.1)
#ax.plot(C, FM_d_max, '-', color=colors[4], label=theory_labels[3])
#ax.plot(C, FM_d_min, '-', color=colors[4], alpha=0.1)
#ax.fill_between(C, FM_d_min, FM_d_max, color=colors[4], alpha=0.1, label=theory_labels[3])
# data
for df, sty, label in zip(dfs, styles, labels):
	x = np.array(df['C_data'])
	xerr = np.array(df['C_data_std']) 
	y = np.array(df['FM_dimer']) * df['a13kF']
	yerr = np.array(np.abs(df['e_FM_dimer'])) * df['a13kF']
	
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
		y_sw = np.array(df['SW_dimer'])/df['a13kF']
		yerr_sw = np.array(np.abs(df['e_SW_dimer']))/df['a13kF']
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
		y_fm = np.array(df['FM_dimer']) * df['a13kF']
		yerr_fm = np.array(np.abs(df['e_FM_dimer'])) * df['a13kF']
		
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