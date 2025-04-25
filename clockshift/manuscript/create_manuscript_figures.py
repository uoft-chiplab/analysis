# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:34:41 2025

These are a collection of select figures for drafting the clock shift manuscript.
Typically it pulls data from manuscript_data/ and saves in manuscript_figures/

@author: coldatoms
"""
import os
import sys
# this is a hack to access modules in the parent directory
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (clockshift)
parent_dir = os.path.dirname(current_dir)
# get the parent's parent directory (analysis)
analysis_dir = os.path.dirname(parent_dir)
# Add the parent parent directory to sys.path
if analysis_dir not in sys.path:
	sys.path.append(analysis_dir)

proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'manuscript_data')

from library import pi, h, hbar, mK, a0, plt_settings, styles, colors, adjust_lightness, FreqMHz
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from scipy.optimize import curve_fit
from contact_correlations.contact_interpolation import contact_interpolation as C_interp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle as pkl

# options
Save = False
Show = True

# choose plot
Plot = 1

#region ######## FIGURE 1: PLOT DIMER AND HFT TOGETHER ON LOG SCALE, NOISE FLOOR, AND 5/2 REGION
if Plot == 1:
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	##another version
	from matplotlib.gridspec import GridSpec

	def format_axes(fig):
		for i, ax in enumerate(fig.axes):
			#ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
			ax.tick_params(labelbottom=False, labelleft=False)
	
	fig = plt.figure(layout="constrained", figsize=(12, 6))
	gs = GridSpec(2, 4, figure=fig)
	#ax1 = fig.add_subplot(gs[0, :])
	gs0=gs[0, 0:2].subgridspec(1, 2, wspace=0.05, hspace=0)
	ax1 = fig.add_subplot(gs0[0])
	ax1_2 = fig.add_subplot(gs0[1], sharey=ax1) # see: https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib for broken x-axis plotting
	plt.setp(ax1_2.get_yticklabels(), visible=False)
	ax_tr = fig.add_subplot(gs[0,2:])
	ax_bl = fig.add_subplot(gs[1, 0:2])
	#ax4 = fig.add_subplot(gs[1, 1]) # this plot used to contain a histogram of errorbar sizes
	ax_br = fig.add_subplot(gs[1, 2:])


	yparam = 'ScaledTransfer' #'ScaledTransfer' or 'transfer'
	# dimer spectrum, long pulse
	
	#file = '2024-07-17_J_e_ratio95.pkl'
	file = '2025-03-19_G_e_pulsetime=0.64.dat.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	x_dimer = data['detuning']
	y_dimer = data['c5_scaledtransfer']
	yerr_dimer = data['em_c5_scaledtransfer']
	fit = pd.read_pickle(os.path.join(data_path, 'fit_'+file))
	xs = fit['xs']/1e6
	ys=fit['ys']
	print(xs)
	#fit_Eb = fit['Eb'][0]/1000
	#fit_e_Eb = fit['e_Eb'][0]/1000

	# HFT spectrum for ax1
	file = '2024-10-08_F_e.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	data = data[data['detuning'] > -1]
	x_all = data['detuning']
	y_all = data[yparam]
	yerr_all = data['em_' +yparam]
	res_bound = 0.030
	x_res = data[data['detuning'] <= res_bound+0.02]['detuning']
	y_res = data[data['detuning'] <= res_bound+0.02][yparam]
	x_HFT = data[data['detuning'] > res_bound]['detuning']
	y_HFT = data[data['detuning'] > res_bound][yparam]


	# dimer plot (left)
	peakindex = np.where(ys==ys.max())
	xpeak = xs[peakindex]
	#filt = 0.028 # arbitrarily chosen so that the plotted lineshape doesn't have sinc^2 sidebands
	filt = 1
	xs_filt = xs[(xs > (xpeak-filt)) & (xs < (xpeak+filt))]
	ys_filt = ys[(xs > (xpeak-filt)) & (xs < (xpeak+filt))]

	ax1.plot(xs_filt, ys_filt, ls='-',  marker='', color=colors[2])
	ax1.fill_between(xs_filt, ys_filt,0, color=adjust_lightness(colors[2],2))
	ax1.plot(x_dimer, y_dimer, **styles[2])
	ax1.set(xlim=[-4.1, -3.9])
	ax1.set_yscale('log')
	ax1.set(
		xlabel=r'$\omega$ [MHz]',
		ylabel=r'$\widetilde{\Gamma}$'
	)

	# HFT plot (right)
	x_ress = np.linspace(min(x_res), max(x_res), 500)
	y_ress = np.interp(x_ress, x_res, y_res)
	x_HFTs = np.linspace(min(x_HFT), max(x_HFT),500)
	y_HFTs = np.interp(x_HFTs, x_HFT, y_HFT)
	ax1_2.plot(x_res, y_res, **styles[3])
	ax1_2.plot(x_HFT, y_HFT, **styles[2])
	ax1_2.plot(x_HFTs, y_HFTs, ls='-', marker='', color=colors[2])
	ax1_2.fill_between(x_HFTs, y_HFTs, 0, color=adjust_lightness(colors[2],2))
	ax1_2.plot(x_ress, y_ress, ls='-', marker='', color=colors[3])
	ax1_2.fill_between(x_ress, 0, y_ress, color=adjust_lightness(colors[3],1.5))

	ax1_2.set(xlim=[-0.1, 1], ylim=[10e-6, 10e-1])
	ax1_2.set_yscale('log')
	

	ax1.spines['right'].set_visible(False)
	ax1_2.spines['left'].set_visible(False)
	ax1.yaxis.tick_left()
	#ax1.tick_params(labelright='off')
	ax1_2.yaxis.tick_right()
	plt.setp(ax1_2.get_yticklabels(), visible=False)

	d = .015  # how big to make the diagonal lines in axes coordinates
	# arguments to pass plot, just so we don't keep repeating them
	kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linestyle='-', marker='')
	ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
	ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

	kwargs.update(transform=ax1_2.transAxes)  # switch to the bottom axes
	ax1_2.plot((-d, +d), (1-d, 1+d), **kwargs)
	ax1_2.plot((-d, +d), (-d, +d), **kwargs)

	# add text
	ax1.text(-4.05, 0.1e-1, 'Dimer')
	ax1_2.text(0.05, 1e-1, 'Res.')
	ax1_2.text(0.3, 10e-4, 'HFT')

	### PLOT SPACE FOR DIAGRAMS IN TOP RIGHT
	ax_tr.text(0.1, 0.5, 'SPACE FOR ENERGY DIAGRAMS')

	### DIMER ZOOM-IN BOTTOM LEFT
	# run = '2024-10-04'
	# runletter = 'H'
	# file = run+'_' + runletter+'_e_ratio95.pkl'
	# data = pd.read_pickle(os.path.join(data_path, file))
	# x_dimer = data['detuning']
	# y_dimer = data[yparam]
	# yerr_dimer = data['em_' + yparam]
	file = '2025-03-19_G_e_pulsetime=0.64.dat.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	x_dimer = data['detuning']
	y_dimer = data['c5_scaledtransfer']
	yerr_dimer = data['em_c5_scaledtransfer']
	#fit = pd.read_pickle(os.path.join(data_path, run+'_fit_ratio95.pkl'))
	fit = pd.read_pickle(os.path.join(data_path, 'fit_'+file))
	xs = fit['xs']/1e6
	ys = fit['ys']

	file2 = '2025-03-19_G_e_pulsetime=0.01.dat.pkl'
	data = pd.read_pickle(os.path.join(data_path, file2))
	x_dimer2 = data['detuning']
	y_dimer2 = data['c5_scaledtransfer']
	yerr_dimer2 = data['em_c5_scaledtransfer']
	#fit = pd.read_pickle(os.path.join(data_path, run+'_fit_ratio95.pkl'))
	fit2 = pd.read_pickle(os.path.join(data_path, 'fit_'+file2))
	xs2 = fit2['xs']/1e6
	ys2 = fit2['ys']

	#left, bottom, width, height = [0.3, 0.3, 0.2, 0.3]
	#ax_inset = fig.add_axes([left, bottom, width, height])
	ax_bl.errorbar(x_dimer, y_dimer, yerr=yerr_dimer, **styles[0], ls='', zorder=1, label=r'$t_\mathrm{rf}=640\,\mu$s') 
	ax_bl.plot(xs, ys, ls='-', marker='', color=colors[0], zorder=2) 
	ax_bl.errorbar(x_dimer2, y_dimer2, yerr=yerr_dimer2, **styles[1], ls='', zorder=1, label=r'$t_\mathrm{rf}=10\,\mu$s')
	ax_bl.plot(xs2, ys2, ls='-', marker='', color=colors[1], zorder=2) 
	ax_bl.legend()
	ax_bl.set(xlim=[-4.15, -3.8],
		xlabel=r'$\omega$ [MHz]',
		ylabel=r'$\alpha/\Omega_R^2/t_\mathrm{rf}$')

	#### ZOOM-IN HFT SPECTRUM IN BOTTOM RIGHT
	filter_by_Ut = True
	Ut = 0.180 # just an estimate
	file = '2024-09-10_L_e.pkl'
	yparam='ScaledTransfer'
	data = pd.read_pickle(os.path.join(data_path, file))
	if filter_by_Ut:
		data = data[data['detuning'] < Ut]
	x_HFT = data['detuning']
	y_HFT = data[yparam]
	yerr_HFT = data['em_' +yparam]

	file = '2024-09-10_L_e_loss.pkl'
	yparam='ScaledTransfer'
	data = pd.read_pickle(os.path.join(data_path, file))
	if filter_by_Ut:
		data = data[data['detuning'] >= Ut]
	x_loss = data['detuning']
	y_loss = data[yparam]
	yerr_loss = data['em_' +yparam]
	
		#we do some fitting
	def powerlawtail(x, A):
		xstar = 2
		return A*x**(-3/2) / (1+x/xstar)
	def tail32(x, A):
		return A*x**(-3/2)
	def tail52(x, A):
		return A*x**(-5/2)
	
	def generate_fit_func(fit_func, x, y, yerr):
		print(fit_func.__name__)
		if fit_func.__name__ == 'powerlawtail':	
			fit_range = x.between(0.040, 4)
			guess = [0.1]
		elif fit_func.__name__ == 'tail32':
			fit_range = x.between(0.040, 0.125)
			guess = [0.1]
		elif fit_func.__name__ == 'tail52':
			fit_range = x.between(0.5, 2)
			guess = [0.1]

		x_fit = x[fit_range]
		y_fit = y[fit_range]
		yerr_fit = yerr[fit_range]
		#print(x_fit)
		popt, pcov = curve_fit(fit_func, x_fit, y_fit, sigma=yerr_fit, p0=guess)
		xfit = np.linspace(x_fit.min(), x_fit.max(), 1000)
		xall = np.linspace(x.min(), x.max()+10, 1000)
		yfit = fit_func(xfit, *popt)
		yall = fit_func(xall, *popt)
		return xfit, xall, yfit, yall
	
	#noisefloor = 1e-6 # CHECK THIS (for transfer)
	noisefloor=1e-5 # for loss
	ax_br.errorbar(x_HFT, y_HFT, yerr_HFT, **styles[0], label='Transfer')
	ax_br.errorbar(x_loss, y_loss, yerr_loss, **styles[1], label='Loss')
	# fit lines
	xx, xxall, yy, yyall = generate_fit_func(tail32, x_HFT, y_HFT, yerr_HFT)
	ax_br.plot(xxall, yyall, color=colors[2], ls='-', marker='')
	xx, xxall, yy, yyall = generate_fit_func(powerlawtail, x_loss, y_loss, yerr_loss)
	ax_br.plot(xxall, yyall, color=colors[2], ls= 'dotted', marker='')
	# noise floor
	# Create a Rectangle patch
	rect = patches.Rectangle((0,0), 10, noisefloor, linewidth=2, facecolor='red', fill=True, alpha = 0.1)
	# Add the patch to the Axes
	ax_br.add_patch(rect)
	# horizontal line for noise floor?
	ax_br.plot([0, 10], [noisefloor, noisefloor], color='red', marker='', ls = '--')
	# vertical line for trap dept
	ax_br.vlines(Ut, ymin=0, ymax=1,ls='dashed', color='black')
	ax_br.set_yscale('log')
	ax_br.set_xscale('log')
	ax_br.set_xlim([0.02, 10])
	ax_br.set_ylim([1e-7, 0.1])

	ax_br.yaxis.set_visible(True)
	ax_br.set(xlabel=r'$\omega$ [MHz]',
		 ylabel = r'$\Gamma$')

	# add text

	ax_br.text(0.06, 1e-2, r'$\omega^{-3/2}$')
	ax_br.text(2, 1e-6, r'$\frac{\omega^{-3/2}}  {\frac{1}{1+\omega/\omega^*}}$')
	ax_br.text(0.23, 0.007, r'$U_t$')

	ax_br.legend()
	
	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'manuscript_figures/log_linear_spectra_v5.pdf')
		print(f'saving to {save_path}')
		plt.savefig(save_path, dpi=300)
	if Show: plt.show()

#endregion

#region ######## FIGURE 2: PLOT DIMER SPECTRAL WEIGHT AND INSET CLOCK SHIFT WITH THEORY
if Plot == 2:
	plot_options = {"SR Fudge Theory":False,
				"SR Fudge Exp":False,
				"Loss Contact": False,
				"Open Channel Fraction": True,
				"Sum Rule 0.5": True,
				"Binned": False
	}

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
	def a13(B):
		''' ac scattering length '''
		abg = 167.6*a0
		DeltaB = 7.2
		B0 = 224.2
		return abg*(1 - DeltaB/(B-B0))
	### constants
	re = 107 * a0 # ac dimer range estimate
	Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra
	kF = 1.1e7
	kappa = np.sqrt((Eb*h*10**6) *mK/hbar**2) # convert Eb back to kappa
	Bfield = 202.14
	spin = 'ratio95' # this means ratio analysis

	#file = 'spectral_weight_summary.pkl'
	file = '4shot_results_testing.pkl'
	with open(os.path.join(data_path, file),'rb') as handle:
		loaded_results = pkl.load(handle)

	summary = pd.DataFrame(loaded_results)
	### apply various options
	# choose contact
	if plot_options['Loss Contact'] == False:
		summary['C_data'] = summary['C_HFT']
		summary['C_data_std'] = summary['C_HFT_std']
	else:
		summary['C_data'] = summary['C_loss_HFT']
		summary['C_data_std'] = summary['C_loss_HFT_std']

	# contact fudge due to 0.35 sumrule
	if plot_options['SR Fudge Theory'] or plot_options['SR Fudge Exp']:
		SR = 0.38 + 0.02
		SR_fudge = 0.5/SR
	else:
		SR_fudge = 1

	if plot_options['SR Fudge Exp']:
		summary['C_data'] = np.array(summary['C_data']) * SR_fudge
		summary['C_data_std'] = np.array(summary['C_data_std']) * SR_fudge 

	sum_rule = 0.5
	open_channel_fraction = 0.93
	kF = np.mean(summary['kF'])
	a13kF = kF * a13(Bfield)
	C = np.linspace(0, max(summary['C_data']),50)
	# a13 times kF # redefined compared to a few lines earlier?
	summary['a13kF'] = np.array(summary['kF']) * a13(202.14)

	if plot_options['Binned']:
		#num_bins=8
		#summary['bins'] = pd.cut(summary['ToTF'], num_bins)
		bins = [0.2,0.3,0.4,0.5,0.6,0.8,1]
		summary['bins'] = pd.cut(summary['ToTF'], bins=bins)
		df_means = summary.groupby('bins')[['C_data', 'SW_ratio95', 'SW_c5', 'SW_c9', 'FM_ratio95', 'FM_c5', 'FM_c9', 'a13kF', 'ToTF', 'ToTF_diff','raw_transfer_fraction']].mean().reset_index()
		df_errs = summary.groupby('bins')[['C_data_std', 'e_SW_ratio95', 'e_SW_c5', 'e_SW_c9', 'e_FM_ratio95', 'e_FM_c5', 'e_FM_c9', 'e_raw_transfer_fraction']].apply(
			lambda x: np.sqrt((x**2).sum())/len(x)).reset_index()
		df_years = summary.groupby('bins')[['year']].apply(lambda x: x).reset_index().drop('level_1', axis=1)
		summary = pd.merge(df_means, df_errs, on='bins')
		summary = pd.merge(summary, df_years)

	### ZY single-channel square well w/ effective range
	# divide I_d by a13 kF,
	other_corr = 1.08 # from a not-small kappatilde
	I_d = sum_rule * kF/(pi*kappa) * 1/(1+re/a13(Bfield)) * C / a13kF * open_channel_fraction * other_corr
	# compute clock shift
	CS_d = -2*I_d *a13kF /kF**2 * kappa**2 # convert I_d to CS_d to avoid rewriting sum_rule and o_c_f
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

	### zero re naive model for bounding
	I_d_max = sum_rule*kF* 1/pi * a13(Bfield) * C / a13kF  # shallow bound state 
	CS_d_max = -2*I_d_max *a13kF /kF**2 * kappa**2
	FM_d_max = CS_d_max * a13kF # assumes ideal SR=0.5

	from matplotlib.gridspec import GridSpec

	def format_axes(fig):
		for i, ax in enumerate(fig.axes):
			#ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
			ax.tick_params(labelbottom=False, labelleft=False)
	
	fig = plt.figure(layout="constrained", figsize=(12, 6))
	gs = GridSpec(3, 3, figure=fig)
	ax1 = fig.add_subplot(gs[:, 1:3])
	ax2 = fig.add_subplot(gs[0, 0])
	ax3 = fig.add_subplot(gs[1, 0])
	ax4 = fig.add_subplot(gs[2, 0])
	format_axes(fig)

	### MAIN AREA: THEORY CURVES
	theory_labels = [r'1ch square well', r'CCC', r'CCC w/ spin corr.', r'zero range']
	ax1.plot(C, I_d, '-', color=colors[0], label=theory_labels[0])
	ax1.plot(C, I_d_CCC_sc, '-', color=colors[3], label=theory_labels[2])
	ax1.plot(C, I_d_max, '-', color='black', label=theory_labels[3])
	# MAIN AREA: EXPERIMENTAL DATA
	x = summary['C_data']
	xerr = summary['C_data_std']
	y = summary['SW_'+spin] / summary['a13kF']
	yerr = np.abs(summary['e_SW_'+spin]) / summary['a13kF']
	ax1.errorbar(x, y, yerr=yerr, xerr=summary['C_data_std'], **styles[0])
	
	ax1.legend(loc='upper left')
	ax1.set(xlabel=r"Contact [$k_F/N$]",
 		   ylabel=r"$I_d/k_Fa_{13}$",
		   ylim = [0,y.max()+0.02])
	
	# INSET: CLOCK SHIFT THEORY
	#left, bottom, width, height = [0.45, 0.65, 0.18, 0.3]
	left, bottom, width, height = [0.8, 0.15, 0.18, 0.3]
	ax_in = fig.add_axes([left, bottom, width, height])
	# Clock shift theory
	ax_in.plot(C, FM_d, '-', color=colors[0], label=theory_labels[0])
	ax_in.plot(C, FM_d_CCC_sc, '-', color=colors[3], label=theory_labels[2])
	ax_in.plot(C, FM_d_max, '-', color='black', label=theory_labels[3])

	# INSET: CLOCK SHIFT DATA
	x = summary['C_data']
	xerr = summary['C_data_std']
	y = summary['FM_'+spin] * summary['a13kF']
	yerr = np.abs(summary['e_FM_'+spin]) * summary['a13kF']
	ax_in.errorbar(x, y, yerr=yerr, xerr=summary['C_data_std'],
		 label=spin_map(spin), **styles[0])
	ax_in.set(ylabel=r"$\hbar\Omega_d  k_Fa_{13} /E_F $ ",
	ylim=[y.min()-0.1, 0.1])


	#### TOP LEFT PLOT: EXAMPLE DIMER 2SHOT
	file = '2024-11-04_M_e_2shotanalysis.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	x_dimer = data['freq'][0]/data['EF'][0]
	yparam='ScaledTransfer'
	y_dimer = data[yparam][0]
	yerr_dimer = data['e_' + yparam][0]

	xx = data['xx']
	yy= data['yy']

	ax2.errorbar([x_dimer, x_dimer], [0,y_dimer], yerr= yerr_dimer*np.array([1,1]), **styles[1])
	ax2.plot(xx, yy, ls='--', marker='', color=colors[1])
	d = np.zeros(len(yy))
	ax2.fill_between(xx, yy, where=yy>=d, color=colors[1], alpha=0.1)
	ax2.text(x_dimer.mean()-1, y_dimer.mean()-0.0009, r'$I_d$')
	xlims = [-255, -220]
	ax2.set(xlim=xlims,
		 ylabel=r'$\Gamma_d$',
		 xlabel=r'$\omega [E_F]$')

	#### MIDDLE LEFT: EXAMPLE HFT 2SHOTdata
	HFT_df = summary.loc[summary['Run']=='2024-11-04_M_e.dat']
	EF = HFT_df['EF'].values[0]
	HFT_detuning = 0.1/EF
	print(HFT_df['C_data_std'].values)
	ax3.errorbar(np.array([HFT_detuning, HFT_detuning]), [0, HFT_df['C_data']], yerr = HFT_df['C_data_std'].values*np.array([1,1]), **styles[2])
	ax3.set(ylabel=r'$\Gamma_{\mathrm{HFT}}$',
		 xlabel=r'$\omega [E_F]$',
		 ylim=[-0.1, HFT_df['C_data'].max()+0.5],
		 xlim=[1, 10])

	#file = "2024-09-24_C_e.pkl"
	file= '2024-10-08_F_e.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	data=data[data['detuning_EF']>=-0.1]
	x = data['detuning_EF']
	y = data['ScaledTransfer']
	xs = np.linspace(min(x), max(x),200)
	interp = np.interp(xs, x, y)
	arb_scale=200
	ys = arb_scale*interp
	d = np.zeros(len(ys))
	ax3.plot(xs, ys, '--', marker='', color=colors[2])
	ax3.fill_between(xs, ys, where=ys>=d, color=colors[2], alpha=0.1)
	ax3.vlines(0.200/EF, ymin=0, ymax=5,ls='dashed', color='black')
	ax3.text(6, HFT_df['C_data'].max()+0.2, r'$C \sim \Gamma_{\mathrm{HFT}} \omega^{3/2}$')

	#### BOTTOM LEFT: CONTACT VS TOTF
	x = summary['ToTF']
	# this is supposed to be the std dev of a uniform distribution of ToTFs that occur
	# when the ToTF is changing linearly during the data run
	xerr = np.abs(np.array(summary['ToTF_diff'])*0.68) 
	y = summary['C_data']
	yerr = summary['C_data_std']
	ax4.errorbar(x, y, yerr=yerr, xerr=xerr, **styles[4])
	xs = np.linspace(min(x), max(x))
	ax4.plot(xs, C_interp(xs), '--', color=colors[4])
	ax4.set(ylabel=r"Contact [$k_F/N$]",
 		   xlabel=r"Temperature [$T_F$]")

	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'manuscript_figures/spectral_weight_v2.pdf')
		print(f'saving to {save_path}')
		plt.savefig(save_path, dpi=300)
	if Show: plt.show()

#endregion

#region ################# APPENDIX: PLOT DIMER BINDING ENERGY VS FIELD
if Plot == 3:
	yparam='ScaledTransfer'

	file1 = '2024-10-24_I_e_ratio95.pkl'
	file2 = '2024-10-30_D_e_ratio95.pkl'
	file3 = '2024-10-30_B_e_ratio95.pkl'
	data1 = pd.read_pickle(os.path.join(data_path, file1))
	data2 = pd.read_pickle(os.path.join(data_path, file2))
	data3 = pd.read_pickle(os.path.join(data_path, file3))
	dfs = [data1, data2, data3]
	fields = [202.14, 204, 209]
	xparam = 'freq'

	# Eb vs field
	Ebs = pd.read_pickle(os.path.join(data_path, 'Ebs.pkl'))
	ExpEbs = pd.read_pickle(os.path.join(data_path, 'ExpEbs.pkl'))

	fig, ax = plt.subplots(figsize=(9,9))
	
	ax.plot(Ebs['B'], Ebs['Ebs_full'], ls='-', color='blue', marker='', label='T-matrix')
	ax.plot(Ebs['B'], Ebs['Ebs_naive'], ls='--', color='red', marker='',  label=r'$1/a^2$')
	ax.errorbar(ExpEbs['B'], ExpEbs['Eb'], yerr=ExpEbs['e_Eb'], **styles[7]) # TO DO: find multiple measurements for Eb at 202.14 G and get a more accurate one for the figure. Also, consider width of fit as uncertainty
	#ax2.plot([199, 225], [x_dimer, x_dimer], color=colors[1], marker='', ls = '--')
	xlabel=r'$B$ [G]'
	ylabel = r'$\omega_d$ [MHz]'
	ax.set(xlabel=xlabel, ylabel=ylabel)

	# inset axis
	left, bottom, width, height = [0.6, 0.2, 0.35, 0.35]
	ax2 = fig.add_axes([left, bottom, width, height])
	omegads = [43.24, 43.797, 45.441]

	for i, df in enumerate(dfs):
		ax2.errorbar(df[xparam]/omegads[i], df[yparam], df['em_'+yparam], **styles[i], label=fields[i])


	xlabel=r'$\omega/\omega_d$'
	ylabel= r'$\widetilde{\Gamma}$'

	ax2.set(xlabel =xlabel, ylabel=ylabel,
		 xlim=[0.996,1.004])
	ax2.set_yscale('log')
	ax2.set_xlabel(xlabel, fontsize=10)
	ax2.set_ylabel(ylabel, fontsize=10)
	ax2.yaxis.set_ticklabels([])
	ax2.xaxis.set_ticklabels(['','', '1', '' ,''])
	#ax2.legend()
	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'manuscript_figures/dimer_Eb_v2.pdf')
		print(f'saving to {save_path}')
		plt.savefig(save_path, dpi=300)
	if Show: plt.show() 
#endregion

#region ##### TESTING PLOTTING TRICKS
if Plot == 5:
	def format_axes(fig):
		for i, ax in enumerate(fig.axes):
			#ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
			ax.tick_params(labelbottom=False, labelleft=False)
	
	fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, facecolor='w', figsize=(6, 3))
	
	yparam = 'ScaledTransfer' #'ScaledTransfer' or 'transfer'
	# dimer spectrum, long pulse
	#file = '2024-07-17_J_e.pkl'
	file = '2024-07-17_J_e_ratio95.pkl'
	#file = '2024-10-04_H_e_ratio95.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	x_dimer = data['detuning']
	y_dimer = data[yparam]
	yerr_dimer = data['em_' + yparam]
	fit = pd.read_pickle(os.path.join(data_path, '2024-07-17_J_e_fit_doubletemp_ratio95.pkl'))
	fit_Eb = fit['Eb'][0]/1000
	fit_e_Eb = fit['e_Eb'][0]/1000

	# HFT spectrum for ax1
	file = '2024-10-08_F_e.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	data = data[data['detuning'] > -1]
	x_HFT = data['detuning']
	#print(x_HFT)
	y_HFT = data[yparam]
	yerr_HFT = data['em_' +yparam]

	# plot the same data on both axes
	#ax1.errorbar(x_HFT, y_HFT, yerr_HFT, **styles[2])
	#ax1.errorbar(x_dimer, y_dimer, yerr_dimer, **styles[2])
	xs = np.linspace(min(x_HFT), max(x_HFT),500)
	ys = np.interp(xs, x_HFT, y_HFT)
	ax1.fill(xs, ys, ls='-', color=colors[2])
	ax1.plot(fit['detuning']/1000, fit['transfer'], ls='-',  marker='', color=colors[2], label='fit with double temp')
	ax1.fill_between(fit['detuning']/1000, fit['transfer'],0, color=adjust_lightness(colors[2],2))
	ax1.set(xlim=[-4.1, -3.9])
	ax1.set_yscale('log')


	#ax2.errorbar(x_HFT, y_HFT, yerr_HFT, **styles[2])
	#ax2.errorbar(x_dimer, y_dimer, yerr_dimer, **styles[2])
	ax2.plot(xs, ys, ls='-', marker='', color=colors[2])
	ax2.fill_between(xs, ys, 0, color=adjust_lightness(colors[2],2))
	ax2.fill(fit['detuning']/1000, fit['transfer'], ls='-',color=colors[2], label='fit with double temp')
	ax2.set(xlim=[-0.1, 0.5])
	ax2.set_yscale('log')

	# hide the spines between ax and ax2
	ax1.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	ax1.yaxis.tick_left()
	#ax1.tick_params(labelright='off')
	ax2.yaxis.tick_right()
	#ax2.tick_params(labelleft='off')

	# This looks pretty good, and was fairly painless, but you can get that
	# cut-out diagonal lines look with just a bit more work. The important
	# thing to know here is that in axes coordinates, which are always
	# between 0-1, spine endpoints are at these locations (0, 0), (0, 1),
	# (1, 0), and (1, 1).  Thus, we just need to put the diagonals in the
	# appropriate corners of each of our axes, and so long as we use the
	# right transform and disable clipping.

	d = .0175  # how big to make the diagonal lines in axes coordinates
	# arguments to pass plot, just so we don't keep repeating them
	kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linestyle='-', marker='')
	ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
	ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
	ax2.plot((-d, +d), (-d, +d), **kwargs)

	# What's cool about this is that now if we vary the distance between
	# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
	# the diagonal lines will move accordingly, and stay right at the tips
	# of the spines they are 'breaking'

	fig.supxlabel(r'$\omega$ [MHz]')
	fig.supylabel('Transfer [arb.]')
	fig.tight_layout()

	# Make the spacing between the two axes a bit smaller
	plt.subplots_adjust(wspace=0.1)

	plt.show()