# -*- coding: utf-8 -*-

"""
AC dimer association spectra analysis script.
@author: Chip Lab
"""

import os
import sys
# this is a hack to access modules in the parent directory
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
	sys.path.append(parent_dir)

from data_class import Data
from data_helper import remove_indices_formatter, bg_freq_formatter
from save_df_to_xlsx import save_df_row_to_xlsx
from scipy.optimize import curve_fit
from scipy.integrate import quad
import pandas as pd
import numpy as np
import corner
import matplotlib.pyplot as plt
from library import GammaTilde, pi, h, adjust_lightness
from clockshift.MonteCarloSpectraIntegration import DimerBootStrapFit, dist_stats
import clockshift.pwave_fd_interp as FD # FD distribution data for interpolation functions, Ben Olsen
from contact_correlations.UFG_analysis import calc_contact

## paths
proj_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(proj_path, "data")
root = os.path.dirname(proj_path)
figfolder_path = os.path.join(proj_path, 'figures')

# bootstrap iterations
BOOTSRAP_TRAIL_NUM = 1000

# print statements
Talk = True

## Diagnostic plots
plotraw = True
plotconvs = True

## Bootstrap switches
Bootstrap = True
Bootstrapplots = True
Correlations = True

# save results
Save = False

# determines whether convolved lineshape fit to data has offset parameter
fitWithOffset = False

# select spin states to analyze
spins = ['c5','c9','sum95']
spin = spins[0]

### save file name
savefile = './clockshift/acdimer_lineshape_results_' + spin + '.xlsx'

### metadata
metadata_filename = 'metadata_dimer_file.xlsx'
metadata_file = os.path.join(proj_path, metadata_filename)
metadata = pd.read_excel(metadata_file)

# if no filename selected, code will run over all files described in metadata (TO DO)

filenames = ['2024-10-02_C_e']

# if the filenames list is empty, run over all available files in metadata
if not filenames:
	filenames = metadata.filename

## plotting things
linewidth = 4

### lineshape functions
def gaussian(x, A, x0, sigma):
	return A * np.exp(-(x-x0)**2/(2*sigma**2))
def norm_gaussian(x, x0, sigma):
	return 1/(np.sqrt(2*pi*sigma**2)) * np.exp(-(x-x0)**2/(2*sigma**2))

def lsFD(x, x0, A, numDA):
	PFD = FD.distRAv[numDA] # 5 = 0.30 T/TF, 6 = 0.40 T/TF
	ls = A * PFD(np.sqrt(-x+x0))
	ls = np.nan_to_num(ls)
	return ls

def GenerateSpectraFit(Ebfix):
	def fit_func(x, A, sigma):
		x0 = Ebfix
		return A*np.sqrt(-x+x0) * np.exp((x - x0)/sigma) * np.heaviside(-x+x0,1)
	return fit_func

def bgdrift(x, A, omega, phi, C, m):
	return A*np.sin(omega*x - phi) + m*x + C

## constants
kF = 1.1e7
a0 = 5.2917721092e-11 # m
re = 107 * a0
re_i = 104 * a0
re_f = 124 * a0
re = np.sqrt(re_i*re_f) # geometric mean of initial and final

def a13(B):
	abg = 167.3*a0 
	DeltaB = 7.2
	B0=224.2
	return abg*(1 - DeltaB/(B-B0))

# using 43.2 MHz VVA calibration (very minor difference at this VVA)
plot_VVAcal = False
data_file = os.path.join(data_path, 'VVAtoVpp_square_43p2MHz.txt')
cal = pd.read_csv(data_file, sep='\t', skiprows=1, names=['VVA','Vpp'])
calInterp = lambda x: np.interp(x, cal['VVA'], cal['Vpp'])
if plot_VVAcal:
	fig_VVA, ax = plt.subplots()
	xx = np.linspace(cal.VVA.min(), cal.VVA.max(),100)
	ax.plot(xx, calInterp(xx), '--')
	ax.plot(cal.VVA, cal.Vpp, 'o')
	ax.set(xlabel='VVA', ylabel='Vpp')

# START OF ANALYSIS LOOP
##############################
######### Analysis ###########
##############################

save_df_index = 0
for filename in filenames:
	metadf = metadata.loc[metadata.filename == filename].reset_index()
	if metadf.empty:
		print("Dataframe is empty! The metadata likely needs updating." )
	
	runfolder = filename 
	xname = metadf['xname'][0]
	barnu = metadf['barnu'][0]
	ff = metadf['ff'][0]
	trf = metadf['trf'][0] #s
	EF = metadf['EF'][0] #kHz
	ToTF = metadf['ToTF'][0]
	VVA = metadf['VVA'][0] #V
	Bfield = metadf['Bfield'][0]
	res_freq = metadf['res_freq'][0]
	pulsetype = metadf['pulsetype'][0]
	gain = metadf['gain'][0]
	remove_indices = metadf['remove_indices'][0]
	track_bg = 0 # turn off this feature
	load_lineshape = metadf['load_lineshape'][0]
	Vppscope = metadf['Vpp'][0]
	
	if load_lineshape:
		df_ls = pd.read_pickle('./clockshift/convolutions.pkl')
		TTF = round(ToTF,1)
		if TTF == 0.7:
			TTF = 0.6
		TRF = trf*1e6
		lineshape = df_ls.loc[(df_ls['TTF']==TTF) & (df_ls['TRF']==TRF)]['LS'].values[0]
	
	# calculate theoretical contact from Tilman's trap averaging code
	C_theory = calc_contact(ToTF, EF*1e3, barnu)[0]
	
	# create data structure
	fn = filename + ".dat"
	run = Data(fn, path=data_path)
	
	# define a few more constants
	T = ToTF * (EF*1000)
# 	VpptoOmegaR = 17.05/0.703 # 47 MHz
	VpptoOmegaR = 14.44/0.656 # 43 MHz data [kHz] vs. scope meas V
# 	VpptoOmegaR=27.5
	if pulsetype == 'square':
		sqrtpulsearea = 1
	elif pulsetype == 'blackman':
		sqrtpulsearea = np.sqrt(0.3)
	OmegaR = 2* pi * sqrtpulsearea * VpptoOmegaR * calInterp(VVA) * gain  # 1/s
	
	# remove indices if requested
	remove_list = remove_indices_formatter(remove_indices)
	if remove_list:
		run.data.drop(remove_list, inplace=True)
	num = len(run.data[xname])
	
	### process data
	run.data['detuning'] = (run.data.freq - res_freq) # MHz
	
	# determine bg freq to be int, list or range
	bg_freq, bg_freq_type = bg_freq_formatter(metadf['bg_freq'][0])
	if bg_freq_type == 'int': # select bg at one freq
		bgdf = run.data.loc[run.data['detuning'] == bg_freq]
	elif bg_freq_type == 'list': # select bg at a list of freqs
		bgdf = run.data.loc[run.data['detuning'].isin(bg_freq)]
	elif bg_freq_type == 'range': # select freq in ranges
		bgdf = pd.concat([run.data.loc[run.data['detuning'].between(val[0], 
											  val[1])] for val in bg_freq])
	
	if not track_bg: # if we are not fudging number by evolving bg 
		bgmean = bgdf[spin].mean()
		bgerr = bgdf[spin].std()
		run.data['transfer'] = (-run.data[spin] + bgmean) / bgmean
		run.data['transferLow'] = (-run.data[spin] + (bgmean-bgerr)) / (bgmean-bgerr) 
		run.data['transferHigh'] = (-run.data[spin] + (bgmean+bgerr)) / (bgmean+bgerr) 
	else:
		# fit the bg point(s) along cyc number and use the fit to define transfer
		fig_bg, ax_bg = plt.subplots()
		ax_bg.plot(bgdf['cyc'], bgdf[spin], label='bg')
		ax_bg.set(xlabel='cycles',ylabel=spin, title='bg over exp cycles')
		degs = np.array([1])
		for deg in degs:
			popt, pcov = np.polyfit(bgdf['cyc'], bgdf[spin], deg, cov=True)
			perr = np.sqrt(np.diag(pcov))
			ax_bg.plot(bgdf['cyc'], np.polyval(popt, bgdf['cyc']), '--', 
				  label='deg={:1d}'.format(deg))
			ax_bg.plot(bgdf['cyc'], np.polyval(popt+perr, bgdf['cyc']), '-.',
				  label='hi, deg={:1d}'.format(deg))
			ax_bg.plot(bgdf['cyc'], np.polyval(popt-perr, bgdf['cyc']), '-.', 
				  label='lo, deg={:1d}'.format(deg))
		ax_bg.legend()
		
		# let's just use the 1st order polynomial for now
		pfit, pcov = np.polyfit(bgdf['cyc'], bgdf[spin],1, cov=True)
		run.data['bg'] = np.polyval(pfit, run.data['cyc'])
		ax_bg.plot(run.data['cyc'], run.data['bg'], '--')
		run.data['transfer'] = 1 - run.data[spin]/run.data['bg']
		run.data['transferLow'] = 1 - run.data[spin]/(run.data['bg'])
		run.data['transferHigh'] = 1 - run.data[spin]/(run.data['bg'])
		
	# calculate scaled detuning
	run.data['Delta'] = run.data.detuning * 1e3/EF
	
	# calculate Scaled transfer
	run.data['ScaledTransfer']= run.data.apply(lambda x: GammaTilde(x['transfer'],
									h*EF*1e3, OmegaR*1e3, trf), axis=1)
	run.data['ScaledTransferLow'] = run.data.apply(lambda x: GammaTilde(x['transferLow'],
									h*EF*1e3, OmegaR*1e3, trf), axis=1)
	run.data['ScaledTransferHigh'] = run.data.apply(lambda x: GammaTilde(x['transferHigh'],
									h*EF*1e3, OmegaR*1e3, trf), axis=1)
	run.group_by_mean('detuning') # averaging
	mti = run.avg_data['ScaledTransfer'].idxmax()
	maxt, maxterr = run.avg_data.loc[mti]['transfer'], run.avg_data.loc[mti]['em_transfer']
	maxsctrans, maxsctranserr = run.avg_data.loc[mti]['ScaledTransfer'], run.avg_data.loc[mti]['em_ScaledTransfer']
	
	# plot something
	fig_data, axs = plt.subplots(2,2, figsize=(8,8))
	ax_raw = axs[0,0]
	xlims = [run.avg_data.Delta.min()*EF,run.avg_data.Delta.max()*EF ]
	ax_raw.errorbar(run.avg_data['Delta']*EF, run.avg_data[spin], run.avg_data['em_' + spin])
	ax_raw.set(ylabel=spin, xlabel='Detuning [kHz]', xlim=xlims)
	ax_trans = axs[0,1]
	ax_trans.errorbar(run.avg_data['Delta']*EF, run.avg_data['transfer'], run.avg_data['em_transfer'])
	ax_trans.set(ylabel='transfer', xlabel='detuning [kHz]', xlim=[-4300,-3700])
	
	### plot raw data
	if plotraw:
		fig_raw, axs_raw = plt.subplots(2,2)
		xlims = [run.avg_data.Delta.min()*EF,run.avg_data.Delta.max()*EF ]
		x_plot = run.avg_data.Delta*EF
		axs_raw[0,0].errorbar(x_plot, run.avg_data.c5, run.avg_data.em_c5)
		axs_raw[0,0].set(ylabel='c5', xlim=xlims)
		axs_raw[0,1].errorbar(x_plot, run.avg_data.c9, run.avg_data.em_c9)
		axs_raw[0,1].set(ylabel='c9', xlim=xlims)
		axs_raw[1,0].errorbar(x_plot, run.avg_data.sum95, run.avg_data.em_sum95)
		axs_raw[1,0].set(ylabel='sum95', xlim=xlims)
		axs_raw[1,1].errorbar(x_plot, run.avg_data.transfer, run.avg_data.em_transfer)
		axs_raw[1,1].set(ylabel='Transfer', xlim=xlims)
		fig_raw.tight_layout()
	
	### arbitrary cutoff in case some points look strange
	cutoffLow = -6*1e3/EF # arbitrary to select all data
	cutoffHigh = -2*1e3/EF 
	run.avg_data['filter'] = np.where((run.avg_data['Delta'] > cutoffLow) & (run.avg_data['Delta']<cutoffHigh), 1, 0)

	filtdf = run.avg_data[run.avg_data['filter']==1]
	x = filtdf['Delta']
	yparam = 'transfer'
	y = filtdf[yparam]
	yerr = filtdf['em_' + yparam]
	ylo = filtdf['transferLow']
	yerrlo = filtdf['em_transferLow']
	yhi = filtdf['transferHigh']
	yerrhi = filtdf['em_transferHigh']
	
	nfiltdf = run.avg_data[run.avg_data['filter']==0]
	xnfilt = nfiltdf['Delta']
	ynfilt = nfiltdf[yparam]
	yerrnfilt = nfiltdf['em_' + yparam]
	
	### TEST: just choose Ebfix wherever the highest datapoint is, since detuning is heavily dependent on EF....
# 	Ebfix = x[np.argmax(y)] 
	Ebfix = -3.98 * 1e3/EF
	
	### prepping evaluation ranges
	xrange = np.abs(x.min() - x.max())
	xlow = Ebfix-xrange
	xhigh = Ebfix + xrange
	xnum = 1000
	xx = np.linspace(xlow, xhigh, xnum)
	xxC = np.linspace(-xrange, xrange, xnum)

	if not load_lineshape:
		def Sincw(w, t): # used Mathematica to get an exact FT
			return 0.797885 * np.sin(t/2*w)/w
		def Sincf(f,t): # kHz
			return Sincw(2*pi*f/1000, t)
		def SincD(Delta, t): # takes dimensionless detuning
			return Sincw(2*pi*EF*Delta/1000, t)	
		def Sinc2D(Delta, t): # takes dimensionless detuning
			return Sincw(2*pi*EF*Delta/1000, t)**2
		
	 	
		D = np.linspace(-10*1000/(trf*1e6) / EF, 10*1000/(trf*1e6) /EF, 1000)
	 	
		if pulsetype == 'square':
			yD = Sinc2D(D, trf*1e6)
	
	# 	norm = 397.932 # FROM MATHEMATICA
		norm = np.trapz(yD, D)
		if plotconvs:
			fig_conv, ax=plt.subplots()
			ax.plot(D, yD/norm, '-')
			ax.set(xlabel='Detuning [EF]', ylabel='Magnitude')
		print('FT[pulse] norm: ' + str(norm))
	 	
		arbscale=1
		qrange=xnum
	 	# create FD lineshapes

		ToTFround = round(ToTF, 1)
		if ToTFround == 0.3:
			FDnum = 5
		elif ToTFround == 0.4:
			FDnum = 6
		elif ToTFround == 0.5:
			FDnum = 7
		elif ToTFround == 0.6:
			FDnum = 8
	
		FDinterp = lambda x: np.interp(x, xxC, lsFD(xxC, 0, arbscale, FDnum))
		FDnorm = quad(FDinterp, -qrange, qrange, points=xxC, limit=2*xxC.shape[0])
		print('FDNORM: ' + str(FDnorm))
		def convfunc(tau, t):
			if pulsetype == 'square':
	 			return FDinterp(tau)/FDnorm[0] * (Sinc2D(t-tau, trf*1e6)/norm)
			

		def convint(t):
			# the integral converges better when ranges don't go to infinity
			sliderange=20
			qrangelow = - sliderange
			qrangehigh=  sliderange
			return quad(convfunc, qrangelow, qrangehigh, args=(t,))
	
		yyconv = []
		e_yyconv = []
		for xconv in xxC:
			a, b = convint(xconv)
			yyconv.append(a)
			e_yyconv.append(b)
		convnorm = np.trapz(yyconv,xxC)
		print('Conv norm: ' + str(convnorm))
	 	# create the convolution lineshape for current iteration

		lineshape = lambda x: np.interp(x, xxC, yyconv)
		
		# 	# show convs explicitly

		if plotconvs:
			fig_CVs, ax_CV = plt.subplots()
	# 		ax_CV.plot(xxC, FDinterp(xxC), '-')
			if pulsetype == 'square':
	 			ax_CV.plot(xxC, Sinc2D(xxC, trf*1e6)/norm, '-', label='FT')
			ax_CV.plot(xxC, yyconv, '-', label='conv')
			ax_CV.set(xlabel = 'Detuning [EF]', ylabel = 'Magnitude')
			ax_CV.legend()
		
	
# 	if fitWithOffset:
# 		guess_FDG = [0.02, -250, 0]
# 		def convls(x, A, x0, C):
#  			return A*convinterp(x-x0) + C
# 	else:
# 		def convls(x, A, x0):
# 			return A*convinterp(x-x0)
# 		guess_FDG = [0.02, -250]

	if fitWithOffset:
		guess_FDG = [0.01, Ebfix, 0]
		bounds = ([0, -600, -np.inf],[np.inf, 0, np.inf])
		def convls(x, A, x0, C):
			return A*lineshape(x-x0)+C
	else:
		def convls(x, A, x0):
			return A*lineshape(x-x0)
		guess_FDG = [200, Ebfix]
		bounds = ([0, -600],[1000, 0])
	
	# fit the lineshape onto the data
	popt, pcov = curve_fit(convls, x, y, sigma=yerr, p0=guess_FDG, bounds=bounds)
	perr = np.sqrt(np.diag(pcov))
	popthi, pcovhi = curve_fit(convls, x, yhi, sigma=yerrhi, p0=guess_FDG, bounds=bounds)
	perrhi = np.sqrt(np.diag(pcovhi))
	poptlo, pcovlo = curve_fit(convls, x, ylo, sigma=yerrlo, p0=guess_FDG, bounds=bounds)
	perrlo = np.sqrt(np.diag(pcovlo))


 	### evaluate and plot on ax_ls
	yyconvls = convls(xx, *popt)
	yyconvls_hi = convls(xx, *popthi)
	yyconvls_lo = convls(xx, *poptlo)

	ax_ls = axs[1,0]
	
	ax_ls.errorbar(x, y, yerr, marker='o', ls='', markersize =10, capsize=2, mew=2, mec=adjust_lightness('tab:gray',0.2), color='tab:gray', elinewidth=2)
# 		ax_ls.errorbar(xnfilt, ynfilt, yerrnfilt, marker='o', ls='', markersize = 12, capsize=3, mew=3, mfc='none', color='tab:gray', elinewidth=3)
	ax_ls.plot(xx, yyconvls, '-', linewidth=3, color = 'tab:green', label='bg avg')
	ax_ls.errorbar(x, yhi, yerrhi, marker='o', ls='', markersize = 6, capsize=1, mew=1, mfc='none', color='tab:blue', elinewidth=1)
	ax_ls.errorbar(x, ylo, yerrlo, marker='o', ls='', markersize = 6, capsize=1, mew=1, mfc='none', color='tab:red', elinewidth=1)
	ax_ls.plot(xx, yyconvls_hi, '--', linewidth=1, color='tab:blue', label= 'bg lo')
	ax_ls.plot(xx, yyconvls_lo, '--', linewidth=1, color='tab:red', label='bg hi')

	# integration has to deal with constant term depending on fit option
	if fitWithOffset:
		SR_convls = np.trapz(yyconvls - popt[-1], xx)
		FM_convls = np.trapz((yyconvls - popt[-1])* xx, xx)
		if not track_bg:
			SR_convls_hi = np.trapz(yyconvls_hi - popthi[-1], xx)
			FM_convls_hi = np.trapz((yyconvls_hi - popthi[-1])* xx, xx)
			SR_convls_lo = np.trapz(yyconvls_lo - poptlo[-1], xx)
			FM_convls_lo = np.trapz((yyconvls_lo - poptlo[-1])* xx, xx)
		fitdof = 3
	else:
		SR_convls = np.trapz(yyconvls, xx)
		FM_convls = np.trapz(yyconvls*xx, xx)

		SR_convls_hi = np.trapz(yyconvls_hi, xx)
		FM_convls_hi = np.trapz(yyconvls_hi* xx, xx)
		SR_convls_lo = np.trapz(yyconvls_lo, xx)
		FM_convls_lo = np.trapz(yyconvls_lo* xx, xx)
		fitdof = 2
	CS_convls = FM_convls/0.5 # assumes ideal SR
	Ctilde_convls = CS_convls * (pi*kF*a13(Bfield)) / -2
	CS_convls_hi = FM_convls_hi/0.5 # assumes ideal SR
	Ctilde_convls_hi = CS_convls_hi * (pi*kF*a13(Bfield)) / -2
	CS_convls_lo = FM_convls_lo/0.5 # assumes ideal SR
	Ctilde_convls_lo = CS_convls_lo * (pi*kF*a13(Bfield)) / -2

	### calculate residuals and chi2
	ymodel = convls(x, *popt)
	convres = y - ymodel
	
	DOF = (1/(len(y) - fitdof))
	chi2 = DOF * np.sum((y-ymodel)**2 / yerr**2)
	
	###residuals
	ax_r = axs[1,1]
	ax_r.errorbar(x,convres,yerr,  marker='o', color='b', mec=adjust_lightness('b'), capsize=2, mew=2, elinewidth=2)
	ax_r.set(xlabel='Detuning [EF]', ylabel='Residuals', title= r'$\chi^2$ = {:.1f}'.format(chi2))
	
	print('Convolution SR: {:.2} + {:.2f} - {:.2f}'.format(SR_convls,
													 np.abs(SR_convls_hi-SR_convls),
													 np.abs(SR_convls - SR_convls_lo)))
	print('Convolution FM: {:.2} + {:.2f} - {:.2f}'.format(FM_convls,
													 np.abs(FM_convls_hi-FM_convls),
													 np.abs(FM_convls - FM_convls_lo)))
	print('Convolution CS: {:.2} + {:.2f} - {:.2f}'.format(CS_convls,
													 np.abs(CS_convls_hi-CS_convls),
													 np.abs(CS_convls - CS_convls_lo)))
	print('Convolution Ctilde: {:.2} + {:.2f} - {:.2f}'.format(Ctilde_convls,
													 np.abs(Ctilde_convls_hi-Ctilde_convls),
													 np.abs(Ctilde_convls - Ctilde_convls_lo)))
	
	### final plotting touches
	ax_ls.legend()
	xadjust = 3
	ax_ls.set_xlim([xlow+xadjust, xhigh -xadjust])
	ax_r.set_xlim([xlow+xadjust, xhigh -xadjust])
# 	ax_ls.set_ylim([y.min() - 0.005,y.max() + 0.005])
	ax_ls.set_ylabel(r'Scaled transfer $\tilde{\Gamma}$ [arb.]')
	ax_ls.set_xlabel(r'Detuning from 12-resonance $\Delta$ [EF]')
	# how hard is it to put a second x-axis on this thing
	# Put MHz frequencies on upper x-axis
	f = lambda x: x * EF /1e3 
	g = lambda x: x * EF/1e3 #wtf
	ax2 = ax_ls.secondary_xaxis("top", functions=(f,g))
	ax2.set_xlabel("Detuning [MHz]")
	fig_data.suptitle(filename + ', ' + spin + ', {:.2f} G, EF={:.1f} kHz, T/TF={:.2f}, T={:.1f} kHz, Ebfix={:.3f} MHz'.format(Bfield, EF, ToTF, ToTF*EF, Ebfix*EF/1000))
	fig_data.tight_layout()

### generate summary table
	fig_table, ax_table = plt.subplots()
	ax_table.axis('off')
	ax_table.axis('tight')
	quantities = [r"file",
			   r"Observable",
			   r"ToTF",
			   r"EF [kHz]",
			   r"Max transfer",
			   r"Max scaled transfer",
			   r"SR [EF]",
			   r"FM [EF]",
			   r"CS [EF]",
			   r"Ctilde"
			   ]
	values = ["{}".format(filename),
			  "{}".format(spin),
			  "{:.2f}".format(ToTF), 
			  "{:.1f}".format(EF),
			  "{:.3f} +/- {:.3f}".format(maxt, maxterr),
			  "{:.3f} +/- {:.3f}".format(maxsctrans, maxsctranserr),
			  "{:.2f} + {:.2f} - {:.2f}".format(SR_convls,
									  np.abs(SR_convls_hi-SR_convls),
									  np.abs(SR_convls - SR_convls_lo)),
			  "{:.2f} + {:.2f} - {:.2f}".format(FM_convls,
									  np.abs(FM_convls_hi-FM_convls),
									  np.abs(FM_convls - FM_convls_lo)),
			  "{:.2f} + {:.2f} - {:.2f}".format(CS_convls,
									  np.abs(CS_convls_hi-CS_convls),
									  np.abs(CS_convls - CS_convls_lo)),
			  "{:.2f} + {:.2f} - {:.2f}".format(Ctilde_convls,
									  np.abs(Ctilde_convls_hi-Ctilde_convls),
									  np.abs(Ctilde_convls - Ctilde_convls_lo))
			  ]
	table = list(zip(quantities, values))
	
	the_table = ax_table.table(cellText=table, loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(12)
	the_table.scale(1,1.5)
	
	
	### time for clock shift analysis I guess		
	CS_pred = -2/(pi*kF*a13(Bfield))*C_theory
	print("predicted dimer clock shift [Eq. (5)]: "+ str(CS_pred))
	
	cstot_pred_zerorange = -1/(pi*kF*a13(Bfield)) * C_theory
	print("Predicted total clock shift w/o eff. range term [Eq. (1)]: "+ str(cstot_pred_zerorange))
	csHFT_pred = 1/(pi*kF*a13(Bfield)) * C_theory
	print("Predicted HFT clock shift w/o eff. range term: " + str(csHFT_pred))
	
	cstot_pred = -1/(pi*kF*a13(Bfield)) * (1- pi**2/8*re/a13(Bfield)) * C_theory
	print("Predicted total clock shift w/ eff. range term [Eq. (1)]: "+ str(cstot_pred))
	csHFT_pred_corr = 1/(pi*kF*a13(Bfield))* (1/(np.sqrt(1-re/a13(Bfield)))) * C_theory
	print("Predicted HFT clock shift w/ eff. range term: " + str(csHFT_pred_corr))
	kappa = 1.2594*1e8
	I_d = kF * C_theory / (pi * kappa) * (1/(1+re/a13(Bfield)))
	print("Predicted dimer spectral weight [Eq. 6]: " + str(I_d))
	
	correctionfactor = 1/(kappa*a13(Bfield))*(1/(1+re/a13(Bfield)))
	print("Eff. range correction: "+ str(correctionfactor))
	
	re_range = np.linspace(re_i/a0, re_f/a0, 100)
	CS_HFT_CORR = 1/(pi*kF*a13(Bfield))* (1/(np.sqrt(1-re_range*a0/a13(Bfield)))) * C_theory
	CS_TOT_CORR = -1/(pi*kF*a13(Bfield)) * (1- pi**2/8*re_range*a0/a13(Bfield)) * C_theory
	CS_DIM_CORR = CS_TOT_CORR - CS_HFT_CORR
	print("CS_HFT_CORR bounds = ({:.1f}, {:.1f})".format(min(CS_HFT_CORR), max(CS_HFT_CORR)))
	print("CS_TOT_CORR bounds = ({:.1f}, {:.1f})".format(min(CS_TOT_CORR), max(CS_TOT_CORR)))
	print("CS_DIM_CORR bounds = ({:.1f}, {:.1f})".format(min(CS_DIM_CORR), max(CS_DIM_CORR)))
	
	results = {'Run':filename,'Pulse Time (us)':trf, 'ToTF':ToTF, 'EF':EF, 
			'kF':kF, 'barnu':barnu, 'C_theory':C_theory, 'pulsetype':pulsetype, 
			'Bfield':Bfield, 'OmegaR':OmegaR, 'MaxTransfer':maxt, 
			'CS_pred':CS_pred}
	
	### generate table
	fig_table2, axs_table2 = plt.subplots(2)
	axpred = axs_table2[0]
	axpred.axis('off')
	axpred.axis('tight')
	quantities = [r"$\widetilde{C}$",
				  r"$r_e/a_0$",
				  r"$\Omega_d$ (zero range)",
				  r"$\Omega_+$ (zero range)", 
				  r"$\Omega_{tot}$ (zero range)", 
				  r"$\Omega_d$ (corr.)",
				  r"$\Omega_+$ (corr.)", 
				  r"$\Omega_{tot}$ (corr.)"]
	values = ["{:.1f}".format(C_theory),
			  "{:.1f}".format(re/a0),
		"{:.1f}".format(CS_pred), 
			  "{:.1f}".format(csHFT_pred),
			  "{:.1f}".format(cstot_pred_zerorange),
			  "{:.1f}".format(cstot_pred - csHFT_pred_corr),
			  "{:.1f}".format(csHFT_pred_corr),
			  "{:.1f}".format(cstot_pred)]
	table = list(zip(quantities, values))
	
	the_table = axpred.table(cellText=table, loc='center')
	the_table.auto_set_font_size(False)
	the_table.set_fontsize(12)
	the_table.scale(1,1.5)
	

##########################
##### Bootstrapping ######
##########################

	if Bootstrap == True:
# 		fit_func = GenerateSpectraFit(Ebfix)
		ff = convls
		
		num_iter = 1000
		conf = 68.2689  # confidence level for CI
		
		# non-averaged data
# 		x = np.array(run.data['detuning'])
		cutoffLow = -4.5*1e3/EF
		cutoffHigh = -3.5*1e3/EF
		x = np.array(run.data[(run.data['Delta'] > cutoffLow) & (run.data['Delta'] < cutoffHigh)]['Delta'])
# 		x = np.array(run.data[run.data['filter']== 1]['detuning'])
		
		xwidth = max(x) - min(x)
		int_bounds = [min(x) - 2*xwidth, max(x) + 2*xwidth]
		num = len(x)
		
# 		y = np.array(run.data['ScaledTransfer'])
		y = np.array(run.data[(run.data['Delta'] > cutoffLow) & (run.data['Delta'] < cutoffHigh)]['ScaledTransfer'])
		
		# sumrule, first moment and clockshift with analytic extension
		SR_BS_dist, FM_BS_dist, CS_BS_dist  = DimerBootStrapFit(xs=x, ys=y,
						  int_bounds=int_bounds, fit_func=ff, trialsB=BOOTSRAP_TRAIL_NUM, 
						 pGuess=guess_FDG)
			
		# calculate contact dist form CS_dist
		C_BS_dist = list(np.array(CS_BS_dist) * (pi*kF*a13(Bfield)) / -2)
			
		# list all ditributions to compure stats on
		dists = [SR_BS_dist, FM_BS_dist, CS_BS_dist, C_BS_dist]
		names = ['SR', 'FM', 'CS', 'C'] 
		
		# update results with all stats from dists
		stats_dict = {}
		for name, dist in zip(names, dists):
			for key, value in dist_stats(dist, conf).items():
				stats_dict[name+'_'+key] = value
		results.update(stats_dict)
		
		if Talk == True:
			for name in names:
				print(r"{} BS median = {:.3f}+{:.3f}-{:.3f}".format(name, 
											results[name+'_median'],
						   results[name+'_upper']-results[name+'_median'], 
						   results[name+'_median']-results[name+'_lower']))
	
		axpred.set(title='Predicted clock shifts [EF]')
		
		axexp = axs_table2[1]
		axexp.axis('off')
		axexp.axis('tight')
		quantities = [r"$\widebar{\Omega_d}$ (lineshape)",
					  r"$\widebar{\Omega_d}$ (bootstrap, ideal SR)",
					  r"Ctilde"]
					  
		# EXPERIMENTAL VALUES
		values = [
			"{:.1f} +{:.1f}-{:.1f}".format(results['CS_median'], 
					   results['CS_upper'] - results['CS_median'], 
					   results['CS_median'] - results['CS_lower']),
			"{:.1f} +{:.1f}-{:.1f}".format(results['C_median'], 
					   results['C_upper'] - results['C_median'], 
					   results['C_median'] - results['C_lower'])]
			
		table = list(zip(quantities, values))
		
		the_table = axexp.table(cellText=table, loc='center')
		the_table.auto_set_font_size(False)
		the_table.set_fontsize(12)
		the_table.scale(1,1.5)
		axexp.set(title='Experimental clock shifts [EF]')
		
		fig_table2.tight_layout()
		
		
		
	
###########################
####### Histograms ########
###########################

	if (Bootstrapplots == True and Bootstrap == True):
		plt.rcParams.update({"figure.figsize": [10,8]})
		fig_hist, axs_hist = plt.subplots(2,2)
		fig_hist.suptitle(filename)
		bins = 20
		ylabel = "Occurances"
		xlabels = ["Sum Rule", "First Moment", "Clock Shift", "Contact"]
		
		for ax, xlabel, dist, name in zip(axs_hist.flatten(), xlabels, dists, names):
			ax.set(xlabel=xlabel, ylabel=ylabel)
			ax.hist(dist, bins=bins)
			ax.axvline(x=results[name+'_lower'], color='red', alpha=0.5, linestyle='--', marker='')
			ax.axvline(x=results[name+'_upper'], color='red', alpha=0.5, linestyle='--', marker='')
			ax.axvline(x=results[name+'_median'], color='red', linestyle='--', marker='')
			ax.axvline(x=results[name+'_mean'], color='k', linestyle='--', marker='')
		fig_hist.tight_layout()	
			
#############################
####### Correlations ########
#############################
		
	if Correlations == True and Bootstrap == True:
		dists = np.vstack(dists)
		fig_cor = corner.corner(dists.T, labels=xlabels)
		
###############################
####### Saving Results ########
###############################
	
	if Save == True:
		savedf = pd.DataFrame(results, index=[save_df_index])
		save_df_index += 1
		
		save_df_row_to_xlsx(savedf, savefile, filename)
			
		figs = [fig_data, fig_table, fig_table2, fig_hist, fig_cor]
		fignames = ['_data', '_table', '_table2', '_hist', '_cor']
		figpath = os.path.join(figfolder_path, runfolder)
		os.makedirs(figpath, exist_ok=True)
		
		for fig, name in zip(figs, fignames):
			figname= filename[:-2] + name + '.pdf'
			fig.savefig(os.path.join(figpath, figname))	