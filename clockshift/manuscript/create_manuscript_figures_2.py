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
from scipy.signal import savgol_filter
from scipy import interpolate
from contact_correlations.contact_interpolation import contact_interpolation as C_interp
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle as pkl
#plt.rcdefaults()
plt.rcParams.update({
    'font.size': 8,                   # Base font size
    'axes.labelsize': 8,             # Axis label font size
    'axes.titlesize': 8,             # Title font size (if used)
    'xtick.labelsize': 7,            # Tick label font size (x-axis)
    'ytick.labelsize': 7,            # Tick label font size (y-axis)
    'legend.fontsize': 7,            # Legend font size
    'figure.figsize': (3.4, 2.4),    # One-column PRL figure size in inches
    'figure.dpi': 300,               # Publication-ready resolution
    'lines.linewidth': 1,            # Thinner lines for compactness
    'axes.linewidth': 0.5,           # Thin axis spines
    'xtick.major.width': 0.5,        # Tick mark width
    'ytick.major.width': 0.5,
    'xtick.direction': 'in',         # Ticks pointing inward
    'ytick.direction': 'in',
    'xtick.major.size': 3,           # Shorter tick marks
    'ytick.major.size': 3,
    'font.family': 'sans-serif',         
  #  'text.usetex': True,             # Use LaTeX for typesetting
    'axes.grid': False,              # No grid for PRL figures
})

# options
Save = True
Show = True

# choose plot
Plot =3

#region ######## FIGURE 1: PLOT DIMER AND HFT TOGETHER ON LOG SCALE, NOISE FLOOR, AND 5/2 REGION
if Plot == 1:
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	##another version
	from matplotlib.gridspec import GridSpec

	def format_axes(fig):
		for i, ax in enumerate(fig.axes):
			#ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
			ax.tick_params(labelbottom=False, labelleft=False)
	
	# fig = plt.figure(layout="constrained", figsize=(8,2))
	# gs = GridSpec(1, 5, figure=fig)
	# #ax1 = fig.add_subplot(gs[0, :])
	# #gs0=gs[0, :].subgridspec(1, 2, wspace=0.03, hspace=0)
	# ax1 = fig.add_subplot(gs[0:2])
	# ax1_2 = fig.add_subplot(gs[2:4], sharey=ax1) # see: https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib for broken x-axis plotting
	# plt.setp(ax1_2.get_yticklabels(), visible=False)

	#ax_bl = fig.add_subplot(gs[1, 0])
	
	#ax_br = fig.add_subplot(gs[1, 1])

	fig = plt.figure(layout="constrained", figsize=(4, 3))
	gs = GridSpec(2, 4, figure=fig)
	#ax1 = fig.add_subplot(gs[0, :])
	gs0=gs[0, :].subgridspec(1, 2, wspace=0.05, hspace=0)
	ax1 = fig.add_subplot(gs0[0])
	ax1_2 = fig.add_subplot(gs0[1], sharey=ax1) # see: https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib for broken x-axis plotting
	plt.setp(ax1_2.get_yticklabels(), visible=False)
	#ax_bl = fig.add_subplot(gs[1, 0:2])
	#ax4 = fig.add_subplot(gs[1, 1]) # this plot used to contain a histogram of errorbar sizes
	ax_br = fig.add_subplot(gs[1, :])

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
	#file = '2024-10-08_F_e.pkl'
	file = 'HFT_2MHz_spectra.csv'
	data = pd.read_csv(os.path.join(data_path, file))
	x_name = 'detuning'
	y_name = 'loss_ScaledTransfer'
	yerr_name = 'loss_e_ScaledTransfer'
	# y_name='ScaledTransfer'
	# yerr_name='e_ScaledTransfer'
	#data = data[data[x_name] > -1]
	data[x_name] = data[x_name]/1000 # MHz
	cutoff =2.1 # cutoff because really high frequencies have bad signal and don't filter well
	data = data[data[x_name] < cutoff]
	x_all = data[x_name]
	y_all = data[y_name]
	yerr_all = data[yerr_name]
	res_bound = 0.05

	x_res = data[data[x_name] <= res_bound][x_name]
	y_res = data[data[x_name] <= res_bound][y_name]
	res_bound_adjust = 0.01
	x_HFT = data[data[x_name] > res_bound-res_bound_adjust][x_name]
	y_HFT = data[data[x_name] > res_bound-res_bound_adjust][y_name]

	#colors
	transfer_color = '#26D980'
	transfer_style = {'color':transfer_color,
				   'mec':adjust_lightness(transfer_color, 0.3),
				   'mfc':transfer_color,
				   'mew':1,
				   'marker':'^',}
	res_color = '#D9267F'
	res_style = {'color':res_color,
				   'mec':adjust_lightness(res_color, 0.3),
				   'mfc':res_color,
				   'mew':1,
				   'marker':'D',}

	# dimer plot (left)
	peakindex = np.where(ys==ys.max())
	xpeak = xs[peakindex]
	filt = 0.028 # arbitrarily chosen so that the plotted lineshape doesn't have sinc^2 sidebands
	#filt=1
	xs_filt = xs[(xs > (xpeak-filt)) & (xs < (xpeak+filt))]
	ys_filt = ys[(xs > (xpeak-filt)) & (xs < (xpeak+filt))]

	ax1.plot(xs_filt, ys_filt, ls='--',  lw= 1, marker='', color=transfer_color)
	ax1.fill_between(xs_filt, ys_filt,0, color=adjust_lightness(transfer_color,1.8))
	ax1.plot(x_dimer, y_dimer, linestyle='', **transfer_style)
	ax1.set(xlim=[-6.2, -3.8])
	ax1.set_yscale('log')
	ax1.set(
		#xlabel=r'$\omega$ [MHz]',
		ylabel=r'$\widetilde{\Gamma}$'
	)
	ax1.xaxis.set_label_coords(0.7, -0.2)
	xticks = [-6, -5, -4]
	ax1.set_xticks(xticks)
	

	# HFT plot (right)
	def transfer_function(f, a):
		# note the EFs are so similar in the datasets I've baked in the average
		# EF here to make this analysis a little easier.
		EF_avg = 19.2
		Eb=3980
		return a*f**(-3/2)/(1+f*EF_avg/Eb)  # binding energy in kHz

	x_ress = np.linspace(min(x_res), max(x_res), 30)
	y_ress = np.interp(x_ress, x_res, y_res)
	y_ress_smooth = savgol_filter(y_ress, 5, 4)
	popt, pcov = curve_fit(transfer_function, x_HFT, y_HFT)

	x_HFTs = np.linspace(min(x_HFT), max(x_HFT),30)
	y_HFTs = np.interp(x_HFTs, x_HFT, y_HFT)
	ax1_2.plot(x_HFTs, transfer_function(x_HFTs, *popt), ls='--', lw= 1, marker='', color=transfer_color)
	ax1_2.fill_between(x_HFTs, transfer_function(x_HFTs, *popt), 0, color=adjust_lightness(transfer_color,1.8))
	ax1_2.plot(x_ress, y_ress_smooth, ls='-',  lw= 1,marker='', color=res_color)
	ax1_2.fill_between(x_ress, 0, y_ress_smooth, color=adjust_lightness(res_color,1.7))
	ax1_2.plot(x_res, y_res, linestyle='', **res_style)
	ax1_2.plot(x_HFT, y_HFT, linestyle='', **transfer_style)
	ax1_2.set(xlim=[-0.2, cutoff+0.1], ylim=[0.5e-5, 7e-1])
	xticks = [0, 1, 2]
	ax1_2.set_xticks(xticks)
	#ax1_2.set(xlim=[-0.1, cutoff], ylim=[0, 0.01])
	ax1_2.set_yscale('log')


	ax1.spines['right'].set_visible(False)
	ax1_2.spines['left'].set_visible(False)
	ax1.yaxis.tick_left()
	ax1_2.yaxis.tick_right()
	plt.setp(ax1_2.get_yticklabels(), visible=False)
	ax1.minorticks_off()
	yticks=[1e-5,  1e-3, 1e-1]
	ax1.set_yticks(yticks)


	# d = .015  # how big to make the diagonal lines in axes coordinates
	# # arguments to pass plot, just so we don't keep repeating them
	# kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linestyle='-', marker='')
	# ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
	# ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

	# kwargs.update(transform=ax1_2.transAxes)  # switch to the bottom axes
	# ax1_2.plot((-d, +d), (1-d, 1+d), **kwargs)
	# ax1_2.plot((-d, +d), (-d, +d), **kwargs)

	# add text
	# ax1.text(-4.05, 0.2e-1, 'Dimer')
	# ax1_2.text(0.07, 1e-1, 'Res.')
	# ax1_2.text(0.3, 10e-4, 'HFT')



	#### ZOOM-IN HFT SPECTRUM IN BOTTOM RIGHT
	filter_by_Ut = True
	trap_depth = 200.0 # estimate
	EF_avg=19.2
	#file = '2024-09-10_L_e.pkl'
	file = 'HFT_2MHz_spectra.csv'

	data = pd.read_csv(os.path.join(data_path, file))
	x_name = 'ScaledDetuning'
	if filter_by_Ut:
		data_below = data[(data[x_name] < trap_depth/EF_avg) & (data[x_name] > 0)]
		data_above = data[data[x_name] > trap_depth/EF_avg]
		
	y_name='ScaledTransfer'
	yerr_name = 'e_ScaledTransfer'
	x = np.array(data_below[x_name])
	y = np.array(data_below[y_name])
	yerr = np.array(data_below[yerr_name])

	sty = styles[0]
	ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty, label=r'$\alpha_3 = N_3/N_\mathrm{tot}$')

	# fit to both forms of the transfer rate equation, w/wout Final State Effect
	def transfer_function(f, a):
		# note the EFs are so similar in the datasets I've baked in the average
		# EF here to make this analysis a little easier.
		EF_avg = 19.2
		Eb=3980
		return a*f**(-3/2)/(1+f*EF_avg/Eb)  # binding energy in kHz

	def transfer_function_no_FSE(f, a):
		return a*f**(-3/2)
	

	popt, pcov = curve_fit(transfer_function_no_FSE, x, y, sigma=yerr, p0=[0.05])
	perr = np.sqrt(np.diag(pcov))
	popt_2, pcov_2 = curve_fit(transfer_function, x, y, sigma=yerr, p0=[0.05])
	perr_2 = np.sqrt(np.diag(pcov_2))

	xs = np.linspace(0.5, max(x), 100)

	ax_br.plot(xs, transfer_function_no_FSE(xs, *popt), '-', color=colors[0])
	ax_br.plot(xs, transfer_function(xs, *popt_2), '--', color=colors[0])

	C_FSE = popt[0] * 2*np.sqrt(2)*np.pi**2
	e_C_FSE = perr[0] * 2*np.sqrt(2)*np.pi**2

	C = popt_2[0] * 2*np.sqrt(2)*np.pi**2
	e_C = perr_2[0] * 2*np.sqrt(2)*np.pi**2

	print("Contact from tranfser with FSE is {:.2f}({:.0f})".format(C_FSE, e_C_FSE*1e2))
	print("Contact from transfer w/out FSE is {:.2f}({:.0f})".format(C, e_C*1e2))

	# transfer above trap depth
	x = np.array(data_above[x_name])
	y = np.array(data_above[y_name])
	yerr = np.array(data_above[yerr_name])

	sty = styles[0].copy()
	sty['mfc'] = 'w'
	ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty)

	# loss
	y_name = 'loss_ScaledTransfer'
	yerr_name = 'loss_e_ScaledTransfer'
	x = np.array(data[x_name])
	y_loss = np.array(data[y_name])
	yerr_loss = np.array(data[yerr_name])

	sty = styles[1]
	ax_br.errorbar(x, y_loss, yerr=yerr_loss, linestyle= '', **sty, label=r'$\alpha_2=(N_2^{\mathrm{bg}}-N_2)/N_\mathrm{tot}$')

	df_fit = data.loc[data[x_name] > 0]
	x = df_fit[x_name]
	y = df_fit[y_name]
	yerr = df_fit[yerr_name]

	# fit to both forms of the transfer rate equation, w/wout Final State Effect
	popt, pcov = curve_fit(transfer_function_no_FSE, x, y, sigma=yerr, p0=[0.05])
	perr = np.sqrt(np.diag(pcov))
	popt_2, pcov_2 = curve_fit(transfer_function, x, y, sigma=yerr, p0=[0.05])
	perr_2 = np.sqrt(np.diag(pcov_2))

	xs = np.linspace(0.5, max(x)+500, 500)

	ax_br.plot(xs, transfer_function_no_FSE(xs, *popt), '-', color=colors[1])
	ax_br.plot(xs, transfer_function(xs, *popt_2), '--', color=colors[1])

	C_loss_FSE = popt[0] * 2*np.sqrt(2)*np.pi**2
	e_C_loss_FSE = perr[0] * 2*np.sqrt(2)*np.pi**2

	C_loss = popt_2[0] * 2*np.sqrt(2)*np.pi**2
	e_C_loss = perr_2[0] * 2*np.sqrt(2)*np.pi**2

	print("Contact from loss with FSE is {:.2f}({:.0f})".format(C_loss_FSE, e_C_loss_FSE*1e2))
	print("Contact from loss w/out FSE is {:.2f}({:.0f})".format(C_loss, e_C_loss*1e2))

	ax_br.vlines(trap_depth/EF_avg, 0, 1.0, color='k', linestyle='--') 

	# Create a Rectangle patch
	noisefloor=1e-5
	rect = patches.Rectangle((0,0), 150, noisefloor, linewidth=3, facecolor='red', fill=True, alpha = 0.1)
	# Add the patch to the Axes
	ax_br.add_patch(rect)
	# horizontal line for noise floor?
	ax_br.plot([0, 150], [noisefloor, noisefloor], color='red', marker='', ls = '--')
	# vertical line for trap dept
	ax_br.vlines(trap_depth/EF_avg, ymin=0, ymax=1.5,ls='dashed', color='black')

	ax_br.set(xlabel=r'Detuning $\tilde{\omega}\,[E_F]$',
		 ylabel = r'$\widetilde{\Gamma}_\mathrm{HFT}$',
		 yscale='log', 
		 xscale='log',
		 xlim=[x.min()-0.5, x.max()+50],
		 ylim=[10e-8, 20e-1])
	ax_br.minorticks_off()
	# add text

	#ax_br.text(1.5, 1e-1, r'$\omega^{-3/2}$')
	#ax_br.text(80, 1e-2, r'$\frac{\omega^{-3/2}}  {\frac{1}{1+\omega/\omega^*}}$')
	ax_br.text(7, 5e-2, r'$U_t$')
	#ax_br.legend(fontsize=8, loc='lower left')

	ax_br2 = ax_br.twiny()
	x_trans = x * EF_avg / 1000
	ax_br2.plot(x_trans, y, alpha=0)

	# sync axis limits
	ax_br2.set_xlim(ax_br.get_xlim()[0] * EF_avg/1000, ax_br.get_xlim()[1]*EF_avg/1000)
	ax_br2.set_xscale('log')
	
	# Set only top axis visible
	ax_br2.tick_params(
		axis='x',
		which='both',
		bottom=False,
		top=True,
		labelbottom=False,
		labeltop=True
	)
	#ax_br2.minorticks_off()
	ax_br2.set_xticklabels(['-','-','0.01','0.1','1'])
	# Hide all axis elements except top x-axis
	ax_br2.spines['bottom'].set_visible(False)
	ax_br2.spines['left'].set_visible(False)
	ax_br2.spines['right'].set_visible(False)
	ax_br2.yaxis.set_visible(False)
	ax_br2.set_xlabel(r'Detuning $\omega$ [MHz]')

	# #second axis
	# ax_br2 = ax_br.twiny()
	# ax_br2.set_xlim(ax_br.get_xlim())
	# bottom_ticks = ax_br.get_xticks()
	# #new_tick_locations = (bottom_ticks * EF_avg)/1e3
	# new_tick_locations = np.array([1, 10, 100, 1000, 10000]) 
	# #new_tick_labels = [f'{tick * EF_avg }'for tick in new_tick_locations]
	# ax_br2.set_xticks(new_tick_locations)
	# #ax_br2.set_xticklabels(new_tick_labels)
	# ax_br2.set_xscale('log')
	# ax_br2.set(
	# 	xlabel = r'Detuning $\omega $ [kHz]'
	# )
	
	
	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'manuscript_figures/log_linear_spectra_v10.pdf')
		print(f'saving to {save_path}')
		plt.savefig(save_path, dpi=1200)
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
	#spin = 'ratio95' # this means ratio analysis
	spin='c5'
	#file = 'spectral_weight_summary.pkl'
	file = '4shot_results_testing.pkl'
	with open(os.path.join(data_path, file),'rb') as handle:
		loaded_results = pkl.load(handle)

	summary = pd.DataFrame(loaded_results)
	### apply various options
	# choose contact
	if plot_options['Loss Contact'] == False:
		summary['C_data'] = summary['C_HFT']*1.29
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

	fig, (ax0, ax1) = plt.subplots(2, 1, height_ratios=[1,3])

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
	ax1.set(xlabel=r"Contact/N [$k_F$]",
 		   ylabel=r"$I_d/k_Fa_{13}$",
		   ylim = [0,y.max()+0.02])
	
	#### UPPER: CONTACT VS TOTF
	x = summary['ToTF']
	# this is supposed to be the std dev of a uniform distribution of ToTFs that occur
	# when the ToTF is changing linearly during the data run
	xerr = np.abs(np.array(summary['ToTF_diff'])*0.68) 
	y = summary['C_data']
	yerr = summary['C_data_std']
	ax0.errorbar(x, y, yerr=yerr, xerr=xerr, **styles[4])
	xs = np.linspace(min(x), max(x))
	ax0.plot(xs, C_interp(xs), '--', color=colors[4])
	ax0.set(ylabel=r"Contact [$k_F/N$]",
 		   xlabel=r"Temperature [$T_F$]")

	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'manuscript_figures/spectral_weight_v3.pdf')
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
	ExpEbs = pd.read_excel(os.path.join(data_path, 'Eb_results.xlsx'))
	ExpEbs = ExpEbs.sort_values(by='B')
	point_color = '#62de93'
	Eb_style = {'color':point_color,
				   'mec':adjust_lightness(point_color, 0.5),
				   'mfc':point_color,
				   'mew':1,
				   'marker':'.',}

	fig, axs= plt.subplots(3,1, figsize=(3.4,3.6))
	ax=axs[0]
	colornaive = '#000000'
	colorT = '#de6f62'
	ax.plot(Ebs['B'], Ebs['Ebs_naive'], ls='--', color=colornaive, marker='',  label=r'$1/a_{13}^2$')
	ax.plot(Ebs['B'], Ebs['Ebs_full'], ls='-', color=colorT, marker='', label=r'$T^{-1}=0$')
	ax.errorbar(ExpEbs['B'], ExpEbs['Eb'], yerr=ExpEbs['sigma'], linestyle='', **Eb_style) # TO DO: find multiple measurements for Eb at 202.14 G and get a more accurate one for the figure. Also, consider width of fit as uncertainty
	#2ebdff
	xlabel=r'$B$ [G]'
	ylabel = r'$\omega_d$ [MHz]'
	#ax.vlines(202.14, -5, 1)
	ax.set(xlabel=xlabel, ylabel=ylabel,
		#xlim=[Ebs['B'].min()-2, Ebs['B'].max()+2],
		#ylim = [-4.2, 0.2]
		)
	legend = ax.legend(loc='lower right', frameon=False)
	legend.get_texts()[0].set_color(colornaive)
	legend.get_texts()[1].set_color(colorT)


# inset axis
	# left, bottom, width, height = [0.7, 0.7, 0.15, 0.15]
	# ax2 = fig.add_axes([left, bottom, width, height])
	# omegads = [43.24, 43.797, 45.441]

	# for i, df in enumerate(dfs):
	# 	ax2.plot(df[xparam]/omegads[i], df[yparam], **styles[i], label=fields[i])


	# xlabel=r'$\omega/\omega_d$'
	# ylabel= r'$\widetilde{\Gamma}$'

	# ax2.set(xlabel =xlabel, ylabel=ylabel,
	# 	 xlim=[0.996,1.004])
	# ax2.set_yscale('log')
	# ax2.set_xlabel(xlabel, fontsize=5)
	# ax2.set_ylabel(ylabel, fontsize=5)
	# ax2.yaxis.set_ticklabels([])
	# ax2.xaxis.set_ticklabels(['','', '1', '' ,''])
	#ax2.legend()

	ax=axs[1]
	### DIMER ZOOM-IN BOTTOM LEFT
	file = '2025-03-19_G_e_pulsetime=0.64.dat.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	scaling = 1000
	data = data.sort_values(by='detuning')
	x_dimer = data['detuning']
	y_dimer = data['c5_scaledtransfer'] * scaling
	yerr_dimer = data['em_c5_scaledtransfer'] *scaling
	#fit = pd.read_pickle(os.path.join(data_path, run+'_fit_ratio95.pkl'))
	fit = pd.read_pickle(os.path.join(data_path, 'fit_'+file))
	xs = fit['xs']/1e6
	ys = fit['ys'] *scaling

	file2 = '2025-03-19_G_e_pulsetime=0.01.dat.pkl'
	data = pd.read_pickle(os.path.join(data_path, file2))
	data = data.sort_values(by='detuning')
	x_dimer2 = data['detuning']
	y_dimer2 = data['c5_scaledtransfer'] * scaling
	yerr_dimer2 = data['em_c5_scaledtransfer'] * scaling
	#fit = pd.read_pickle(os.path.join(data_path, run+'_fit_ratio95.pkl'))
	fit2 = pd.read_pickle(os.path.join(data_path, 'fit_'+file2))
	xs2 = fit2['xs']/1e6
	ys2 = fit2['ys'] *scaling

	color640 = '#4093ff'
	style640 = {'color':color640,
				   'mec':adjust_lightness(color640, 0.3),
				   'mfc':color640,
				   'mew':1,
				   'marker':'o',}
	color10 = '#ff5447'
	style10 = {'color':color10,
				'mec':adjust_lightness(color10, 0.3),
				'mfc':color10,
				'mew':1,
				'marker':'s',}
	ax.errorbar(x_dimer, y_dimer, yerr=yerr_dimer, **style640, ls='', zorder=1, label=r'$\sigma = 28\,\mathrm{kHz} \approx 1.4\,E_F$') 
	ax.plot(xs, ys, ls='-', marker='', color=color640, zorder=2) 
	ax.errorbar(x_dimer2, y_dimer2, yerr=yerr_dimer2, **style10, ls='', zorder=1, label=r'$\sigma = 100\,\mathrm{kHz} \approx 5 E_F$')
	ax.plot(xs2, ys2, ls='-', marker='', color=color10, zorder=2) 
	
	ax.set(xlim=[-4.21, -3.72],
		xlabel=r'$\omega$ [MHz]',
		ylabel=r'$\alpha_d/\Omega_R^2/t_\mathrm{rf} \times 10^{-3}$',
		ylim=[-2, 8.1]
	)
	xticks = [-4.2,-4.1, -4, -3.9,-3.8, -3.7]
	ax.set_xticks(xticks)

	yticks = [0, 3, 6]
	ax.set_yticks(yticks)
	legend = ax.legend(loc='upper right', frameon=False)
	legend.get_texts()[0].set_color(color640)
	legend.get_texts()[1].set_color(color10)
	ax.text(-4.19, 6.3, r'$B=202.14\,$G')
	from matplotlib import rc
	rc('text.latex', preamble = '\\usepackage{color}')
	#ax.text(-4.19, 3.8, r'$\frac{\mathcal{I}_d^{t_{\mathrm{rf}} \approx \tau_F}}{\mathcal{I}_d^{t_{\mathrm{rf}} \gg \tau_F}} = 1.23(13)$')
	ax.text(-4.19, 3.8, r'$R_{\mathcal{I}_d} = 1.23(13)$')

	ax3 = axs[2]
	file = 'veryshort_df.xlsx'
	data = pd.read_excel(os.path.join(data_path, file))
	data = data.sort_values(by='scaledtime')
	decotimecut = 1.1 # in t_F
	data['full'] = (data['scaledtime'] < decotimecut).astype(int)

	averageId = data[data['full']==1]['spectral_weight'].mean()
	stdId = data[data['full']==1]['spectral_weight'].std()

	# trying to do some manual averaging
	df = data[data['time'] == 0.003]
	t3us = df['scaledtime'].values[0]
	I3us = df['spectral_weight'].mean()
	I3us_std = np.sqrt(df['em_spectral_weight'].values[0]**2 + df['em_spectral_weight'].values[1]**2)

	df = data[data['time'] == 0.020]
	t20us = df['scaledtime'].values[0]
	I20us = df['spectral_weight'].mean()
	I20us_std = np.sqrt(df['em_spectral_weight'].values[0]**2 + df['em_spectral_weight'].values[1]**2)

	data = data[(data['time'] != 0.003) & (data['time']!=0.020)]
	lowtimedata = data[data['full']==1]
	hightimedata = data[data['full']!=1]

	my_color = '#754028'
	my_style = {'color':my_color,
				   'mec':adjust_lightness(my_color, 0.3),
				   'mfc':my_color,
				   'mew':1,
				   'marker':'o',
				   'ls':'none'}
	# low time data (somewhat arbitrary)
	x_dimer = lowtimedata['scaledtime']
	y_dimer = lowtimedata['spectral_weight'] 
	yerr_dimer = lowtimedata['em_spectral_weight'] 
	ax3.errorbar(x_dimer, y_dimer, yerr_dimer, **my_style)
	ax3.errorbar([t3us], [I3us], yerr=[I3us_std], **my_style)

	# high time data (somewhat arb)
	my_style = {'color':my_color,
				   'mec':adjust_lightness(my_color, 0.3),
				   'mfc':'none',
				   'mew':1,
				   'marker':'o',
				   'ls':'none'}
	x_dimer = hightimedata['scaledtime']
	y_dimer = hightimedata['spectral_weight'] 
	yerr_dimer = hightimedata['em_spectral_weight'] 
	ax3.errorbar(x_dimer, y_dimer, yerr_dimer, **my_style)
	ax3.errorbar([t20us], [I20us], yerr=[I20us_std], **my_style)

	# line for average of early time results
	ax3.hlines(averageId, -0.1, decotimecut, colors=my_color, ls='dashed', linewidth=1)


	ax3.set(xlabel = r'$t_\mathrm{rf}/\tau_F$',
		 ylabel = r'$\mathcal{I}_d$ response',
		 ylim = [-0, 0.015],
		 xlim = [-0.1, 5.1])
	ax3.tick_params(axis='y', labelleft=False)
	# # Smaller, separate axis
	# ax3 = fig.add_axes([0.15, -0.15,0.9, 0.05])  # Centered lower and smaller
	# ax3.axis('off')
	# table_data = [['640','1.6','18(2)', '0.056(7)'],
	# 		   ['10','100','1','0.069(7)']]
	# table_cols = [r'$t_\mathrm{rf}\,[\mu s]$', 
	# 		   r'$\Delta \omega_{\mathrm{rf}}$ [kHz]', 
	# 		   r'$\sigma/\Delta \omega_\mathrm{rf}$',
	# 		   r'$\mathcal{I}$']
	# table = ax3.table(
	# 	cellText = table_data,
	# 	colLabels = table_cols,
	# 	loc='center'
	# )
	# for(row, col), cell in table.get_celld().items():
	# 	cell.set_linewidth(0.5)
	# 	if row !=0:
	# 		cell.set_linewidth(0.5)
	# 		cell.visible_edges = 'LR'
	# 	if row == 1:
	# 		cell.get_text().set_color(color640)
	# 	if row == 2:
	# 		cell.get_text().set_color(color10)
	
	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'manuscript_figures/dimer_Eb_v6.pdf')
		print(f'saving to {save_path}')
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	if Show: plt.show() 
#endregion

# #region some other stuff
# if Plot == 5:
	
#     def expdecay(t, A, tau, C)

# 	file = 'dimer_loss_timeconstant.pkl'
# 	data = pd.read_pickle(os.path.join(data_path, file))
# 	fig, ax = plt.subplots()
	
# #endregion




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