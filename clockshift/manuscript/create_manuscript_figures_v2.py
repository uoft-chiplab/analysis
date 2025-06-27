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

from library import pi, h, hbar, mK, a0, styles, colors, adjust_lightness,  generate_plt_styles
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy import interpolate
from scipy.integrate import quad
from contact_correlations.contact_interpolation import contact_interpolation as C_interp
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle as pkl
import json

#plt.rcdefaults()
paper_settings = {
				'font.size': 8,          # Base font size
				'axes.labelsize': 8,       # Axis label font size
				'axes.titlesize': 8,       # Title font size (if used)
				'xtick.labelsize': 7,      # Tick label font size (x-axis)
				'ytick.labelsize': 7,      # Tick label font size (y-axis)
				'legend.fontsize': 7,      # Legend font size
				'figure.dpi': 300,        # Publication-ready resolution
				'lines.linewidth': 1,      # Thinner lines for compactness
				'lines.linestyle':'',
				'lines.markeredgewidth':1,
				'axes.linewidth': 0.5,      # Thin axis spines
				'xtick.major.width': 0.5,    # Tick mark width
				'ytick.major.width': 0.5,
				'xtick.minor.width':0.5,
				'ytick.minor.width':0.5,
				'xtick.minor.size':2,
				'ytick.minor.size':2,
				'xtick.direction': 'in',     # Ticks pointing inward
				'ytick.direction': 'in',
				'xtick.major.size': 3,      # Shorter tick marks
				'ytick.major.size': 3,
				'font.family': 'sans-serif',

				# 'text.usetex': True,       # Use LaTeX for typesetting, needs local LaTeX install
				'axes.grid': False,       # No grid for PRL figures}
				}
plt.rcParams.update(paper_settings)

# options
Save = True
Show = True

# plot shading
tintshade = 0.6

# choose plot
Plot =3

# data binning
def bin_data(x, y, yerr, nbins, xerr=None):

	if np.any(yerr == 0):
		avg_nonzero_yerr = np.mean(yerr[yerr>0])
		yerr[yerr==0] = avg_nonzero_yerr

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


#region ######## FIGURE 1: PLOT DIMER AND HFT TOGETHER ON LOG SCALE, NOISE FLOOR, AND 5/2 REGION
if Plot == 1:
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	##another version
	from matplotlib.gridspec import GridSpec

	def format_axes(fig):
		for i, ax in enumerate(fig.axes):
			#ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
			ax.tick_params(labelbottom=False, labelleft=False)
	
	
	#colors
	# transfer_color = '#26D980'
	# transfer_style = {'color':transfer_color,
	# 			   'mec':adjust_lightness(transfer_color, 0.3),
	# 			   'mfc':transfer_color,
	# 			   'mew':1,
	# 			   'marker':'^',}
	mysize = 5
	dimer_color = '#1b9e77'
	dimer_style = generate_plt_styles([dimer_color], ts=tintshade)[0]
	dimer_style['marker'] = 'o'
	dimer_style['markersize']=mysize
	loss_color = '#d95f02'
	loss_style = generate_plt_styles([loss_color], ts=tintshade)[0]
	loss_style['marker']='s'
	loss_style['markersize']=mysize
	# loss_style = {'color':loss_color,
	# 			   'mec':adjust_lightness(loss_color, 0.3),
	# 			   'mfc':loss_color,
	# 			   'mew':1,
	# 			   'marker':'s',}
	res_color = '#0F1AF0'
	#res_color = '#FF000C'
	res_style = generate_plt_styles([res_color], ts= 0.4)[0]
	res_style['marker']= 'D'
	res_style['markersize']=mysize


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

	
	# dimer plot (left)
	peakindex = np.where(ys==ys.max())
	xpeak = xs[peakindex]
	filt = 0.028 # arbitrarily chosen so that the plotted lineshape doesn't have sinc^2 sidebands
	#filt=1
	xs_filt = xs[(xs > (xpeak-filt)) & (xs < (xpeak+filt))]
	ys_filt = ys[(xs > (xpeak-filt)) & (xs < (xpeak+filt))]

	ax1.plot(xs_filt, ys_filt, ls='-',  lw= 1, marker='', color=dimer_color)
	ax1.fill_between(xs_filt, ys_filt,0, color=adjust_lightness(dimer_color,1.8))
	# binx, biny, binyerr, binxerr = bin_data(x_dimer, y_dimer, yerr=np.ones(len(y_dimer)), nbins=6, xerr=np.ones(len(x_dimer)))
	# ax1.plot(binx, biny, **dimer_style)
	# custom bin range
	#bin_edges = [-4.05, -4.03, -4.01, -3.99, -3.97, -3.95, -3.93, -3.91]
	bin_edges = np.arange(-4.08, -3.93, 0.015)
	bin_edges = [-4.06, -4.05, -4.04, -4.03,  -4, -3.98, -3.97,-3.96, -3.95, -3.94, -3.93]
	#bin_edges = [-4.06, -4.04, -4.03, -4.00, -3.97, -3.94]
	data['value_bin'] = pd.cut(x_dimer, bins=bin_edges)
	biny = data.groupby('value_bin')['c5_scaledtransfer'].mean()
	binx = data.groupby('value_bin')['detuning'].mean()
	ax1.plot(binx, biny, **dimer_style)
	ax1.set(xlim=[-6.2, -3.8])
	ax1.set_yscale('log')
	ax1.set(
		#xlabel=r'$\omega$ [MHz]',
		ylabel=r'$\widetilde{\Gamma}$'
	)
	ax1.xaxis.set_label_coords(0.7, -0.2)
	xticks = [-6, -5, -4]
	ax1.set_xticks(xticks)
	

	# HFT spectrum for ax1
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


	# HFT plot (right)
	def transfer_function(f, a):
		# note the EFs are so similar in the datasets I've baked in the average
		# EF here to make this analysis a little easier.
		EF_avg = 19.2
		Eb=3980
		return a*f**(-3/2)/(1+f*EF_avg/Eb)  # binding energy in kHz

	x_ress = np.linspace(min(x_res), max(x_res), 30)
	#x_ress = np.linspace(0, max(x_res),30)
	y_ress = np.interp(x_ress, x_res, y_res)
	y_ress_smooth = savgol_filter(y_ress, 5, 4)
	popt, pcov = curve_fit(transfer_function, x_HFT, y_HFT)

	x_HFTs = np.linspace(min(x_HFT), max(x_HFT),30)
	y_HFTs = np.interp(x_HFTs, x_HFT, y_HFT)
	ax1_2.plot(x_HFTs, transfer_function(x_HFTs, *popt), ls='-', lw= 1, marker='', color=loss_color)
	ax1_2.fill_between(x_HFTs, transfer_function(x_HFTs, *popt), 0, color=adjust_lightness(loss_color,2.0))

	ax1_2.plot(x_ress, y_ress_smooth, ls='-',  lw= 1,marker='', color=res_color)
	ax1_2.fill_between(x_ress, 0, y_ress_smooth, color=adjust_lightness(res_color,1.8))

	binx, biny, binyerr, binxerr = bin_data(x_res, y_res, yerr=np.ones(len(y_res)), nbins=4, xerr=np.ones(len(x_res)))
	ax1_2.plot(binx, biny, **res_style)
	# x_HFT.index = x_HFT.index - x_HFT.index[0]
	# y_HFT.index = y_HFT.index - y_HFT.index[0
	# binx, biny, binyerr, binxerr = bin_data(x_HFT, y_HFT, yerr=np.ones(len(y_HFT)), nbins=10, xerr=np.ones(len(x_HFT)))
	ax1_2.plot(x_HFT, y_HFT, **loss_style)

	# inset axis
	left, bottom, width, height = [0.78, 0.82, 0.15, 0.11]
	axi = fig.add_axes([left, bottom, width, height])
	axi.plot(x_ress, y_ress_smooth, ls='-', marker='', color=res_color)
	axi.plot(x_res, y_res, **res_style)
	axi.fill_between(x_ress, 0, y_ress_smooth, color=adjust_lightness(res_color, 1.8))
	axi.set(
		xlim=[-0.011, 0.011],
		ylim=[0, 0.35],
	)
	axi.tick_params(labelsize=6)
	axi.set_xticks([-0.01,0,0.01])
	axi.set_xticklabels(['-0.01','0','0.01'])

	ax1_2.set(xlim=[-0.2, cutoff+0.1], ylim=[0.5e-5, 7e-1])
	xticks = [0, 1, 2]
	ax1_2.set_xticks(xticks)
	#ax1_2.set(xlim=[-0.1, cutoff], ylim=[0, 0.01])
	ax1_2.set_yscale('log')


	ax1.spines['right'].set_visible(False)
	ax1_2.spines['left'].set_visible(False)
	ax1.yaxis.tick_left()
	ax1_2.yaxis.tick_right()
	#ax1_2.yick_params(axis='y', which='both', length=0)
	plt.setp(ax1_2.get_yticklabels(), visible=False)
	ax1.minorticks_off()
	yticks=[1e-5,  1e-3, 1e-1]
	ax1.set_yticks(yticks)

	#### ZOOM-IN HFT SPECTRUM IN BOTTOM RIGHT
	loss_color = '#d95f02'
	loss_style = generate_plt_styles([loss_color], ts=tintshade)[0]
	loss_style['marker']='s'

	transfer_color = '#2877dd'
	transfer_style = generate_plt_styles([transfer_color], ts=0.3)[0]
	transfer_style['marker'] = 'o'

	filter_by_Ut = False
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
	if filter_by_Ut:
		x = np.array(data_below[x_name])
		y = np.array(data_below[y_name])
		yerr = np.array(data_below[yerr_name])
		#ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty, label=r'$\alpha_3 = N_3/N_\mathrm{tot}$')
	else: 
		x= np.array(data[x_name])
		y = np.array(data[y_name])
		yerr = np.array(data[yerr_name])
		#ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty, label=r'$\alpha_3 = N_3/N_\mathrm{tot}$')

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

	#ax_br.plot(xs, transfer_function_no_FSE(xs, *popt), '-', color=colors[0])
	#ax_br.plot(xs, transfer_function(xs, *popt_2), '--', color=colors[0])

	C_FSE = popt[0] * 2*np.sqrt(2)*np.pi**2
	e_C_FSE = perr[0] * 2*np.sqrt(2)*np.pi**2

	C = popt_2[0] * 2*np.sqrt(2)*np.pi**2
	e_C = perr_2[0] * 2*np.sqrt(2)*np.pi**2

	print("Contact from tranfser with FSE is {:.2f}({:.0f})".format(C_FSE, e_C_FSE*1e2))
	print("Contact from transfer w/out FSE is {:.2f}({:.0f})".format(C, e_C*1e2))

	# transfer above trap depth
	if filter_by_Ut:
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

	sty = loss_style
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
	#ax_br.plot(xs, transfer_function(xs, *popt_2), '--', color=colors[1])

	C_loss_FSE = popt[0] * 2*np.sqrt(2)*np.pi**2
	e_C_loss_FSE = perr[0] * 2*np.sqrt(2)*np.pi**2

	C_loss = popt_2[0] * 2*np.sqrt(2)*np.pi**2
	e_C_loss = perr_2[0] * 2*np.sqrt(2)*np.pi**2

	print("Contact from loss with FSE is {:.2f}({:.0f})".format(C_loss_FSE, e_C_loss_FSE*1e2))
	print("Contact from loss w/out FSE is {:.2f}({:.0f})".format(C_loss, e_C_loss*1e2))


	data = pd.read_csv(os.path.join(data_path, file))
	x_name = 'ScaledDetuning'
	if filter_by_Ut:
		data_below = data[(data[x_name] < trap_depth/EF_avg) & (data[x_name] > 0)]
		data_above = data[data[x_name] > trap_depth/EF_avg]
		
	y_name='ScaledTransfer'
	yerr_name = 'e_ScaledTransfer'
	if filter_by_Ut:
		x = np.array(data_below[x_name])
		y = np.array(data_below[y_name])
		yerr = np.array(data_below[yerr_name])
		sty=transfer_style
		ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty, label=r'$\alpha_3 = N_3/N_\mathrm{tot}$')
	else: 
		x= np.array(data[x_name])
		y = np.array(data[y_name])
		yerr = np.array(data[yerr_name])
		sty = transfer_style
		ax_br.errorbar(x, y, yerr=yerr, linestyle='', **sty, label=r'$\alpha_3 = N_3/N_\mathrm{tot}$')


	ax_br.vlines(trap_depth/EF_avg, 0, 1.0, color='k', linestyle='--') 

	# Create a Rectangle patch
	noisefloor=1e-5
	rect = patches.Rectangle((0,0), 250, noisefloor, linewidth=3, facecolor='red', fill=True, alpha = 0.2)
	# Add the patch to the Axes
	ax_br.add_patch(rect)
	# horizontal line for noise floor?
	#ax_br.plot([0, 150], [noisefloor, noisefloor], color='red', marker='', ls = '-')
	# vertical line for trap dept
	ax_br.vlines(trap_depth/EF_avg, ymin=0, ymax=1.5,ls='dashed', color='black')

	ax_br.set(xlabel=r'Detuning $\tilde{\omega}\,[E_F]$',
		 ylabel = r'$\widetilde{\Gamma}_\mathrm{HFT}$',
		 yscale='log', 
		 xscale='log',
		 xlim=[x.min()-0.5, x.max()+50],
		 ylim=[5e-9, 20e-1])
	ax_br.minorticks_off()
	yticks = [10e-8, 10e-6, 10e-4, 10e-2]
	ax_br.set_yticks(yticks)
	# add text

	#ax_br.text(1.5, 1e-1, r'$\omega^{-3/2}$')
	#ax_br.text(80, 1e-2, r'$\frac{\omega^{-3/2}}  {\frac{1}{1+\omega/\omega^*}}$')
	ax_br.text(7.5, 5e-2, r'$U_t$')
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

	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'manuscript_figures/log_linear_spectra_v12.pdf')
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
	#file = 'Id_summary.pkl'
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
	from matplotlib.gridspec import GridSpec

	def format_axes(fig):
		for i, ax in enumerate(fig.axes):
			#ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
			ax.tick_params(labelbottom=False, labelleft=False)
	
	fig = plt.figure(layout='constrained', figsize=(3.4,2.8))
	gs = GridSpec(2, 2, figure=fig, hspace=0.1)
	
	yparam='ScaledTransfer'
	# Eb vs field
	Ebs = pd.read_pickle(os.path.join(data_path, 'Ebs.pkl'))
	ExpEbs = pd.read_excel(os.path.join(data_path, 'Eb_results.xlsx'))
	ExpEbs = ExpEbs.sort_values(by='B')
	SqW = pd.read_excel(os.path.join(data_path, 'sqw_theory_line.xlsx'))
	Tmat = pd.read_excel(os.path.join(data_path, 't_matrix_theory_line.xlsx'))
	CCC = pd.read_csv(os.path.join(data_path, 'ac_s_Eb_vs_B_220-225G.dat'), header=None, names=['B','E'], delimiter='\s')
	
	Eb_color =  '#1b9e77'
	Eb_style= {'color':Eb_color,
				   'mec':adjust_lightness(Eb_color, 0.3),
				   'mfc':Eb_color,
				   'mew':1,
				   'marker':'o',
				   'markersize':3}
	

	colornaive = '#000000'
	colorT = '#f20470'
	colorSqW = '#23d197'
	colorCC = '#f20470'
	ax = fig.add_subplot(gs[0, 0])
	ax.plot(Ebs['B'], Ebs['Ebs_naive'], ls='dotted', color=colornaive, marker='',  label=r'$1/a_{13}^2$')
	#ax.plot(Tmat['Magnetic Field (G)'], Tmat['Energy (MHz)'], color=colorT, marker='', ls='-')
	ax.plot(SqW['Magnetic Field (G)'], SqW['Energy (MHz)'], color=colorSqW, marker='', ls = '--')
	ax.plot(CCC['B'],CCC['E'], marker='', ls='-' , color = colorCC)
	binx, biny, binxerr, binyerr = bin_data(ExpEbs['B'], ExpEbs['Eb'], xerr=np.ones(len(ExpEbs['B'])), yerr= np.ones(len(ExpEbs['Eb'])), nbins=25)
	ax.plot(binx, biny, binyerr, **Eb_style)
	#2ebdff
	xlabel=r'$B$ [G]'
	ylabel = r'$\omega_d/2\pi$ (MHz)'
	#ax.vlines(202.14, -5, 1)
	ax.set(xlabel=xlabel, ylabel=ylabel,
		xlim=[199, 210],
		ylim = [-4.5, -1.5]
		)

	
	ms = 3
	color1 = "#7eb0d5"
	style1 = {'color':color1,
				   'mec':adjust_lightness(color1, 0.3),
				   'mfc':color1,
				   'mew':1,
				   'marker':'s',
				   'markersize':ms}
	color2 = "#ffb55a"
	style2 = {'color':color2,
				   'mec':adjust_lightness(color2, 0.3),
				   'mfc':color2,
				   'mew':1,
				   'marker':'p',
				   'markersize':ms}
	color3 = "#666699"
	style3 = {'color':color3,
				   'mec':adjust_lightness(color3, 0.3),
				   'mfc':color3,
				   'mew':1,
				   'marker':'d',
				   'markersize':ms} 
	mystyles = [style1, style2, style3]
	mycolors = [color1, color2, color3]
	Bfield = [202.1, 203.1, 209]

	ax = fig.add_subplot(gs[0,1])

	#ax = fig.add_axes([left, bottom, width, height])
	color640 = '#4093ff'
	style640 = {'color':color640,
				   'mec':adjust_lightness(color640, 0.3),
				   'mfc':color640,
				   'mew':1,
				   'marker':'o',
				   'markersize':3}
	color10 = '#ff5447'
	style10 = {'color':color10,
				'mec':adjust_lightness(color10, 0.3),
				'mfc':color10,
				'mew':1,
				'marker':'s',
				'markersize':3}
	#file = '2025-03-19_G_e_pulsetime=0.64.dat.pkl'
	file = '2024-07-17_J_e.dat_sat_corr.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	scaling = 1000
	data = data.sort_values(by='detuning')
	
	custom_bins = [-4.05,  -4.03, -4.02, -4.01, -4.005, -4, -3.995, -3.990, -3.985, -3.98, -3.97, -3.96, -3.94]
	x_dimer, y_dimer, yerr_dimer = bin_data(data['detuning'], data['c5_scaledtransfer']*scaling, data['em_c5_scaledtransfer']*scaling, nbins=custom_bins)
	fit = pd.read_pickle(os.path.join(data_path, 'fit_'+file))
	xs = fit['xs']/1e6
	ys = fit['ys'] *scaling
	EF_data = 0.0133
	# set offset to 0
	offs = ys.min()
	ys = ys-offs
	y_dimer = y_dimer - offs
	

	json_file = 'lineshape_2024-07-17_J_e_backup.json'

	
	with open(os.path.join(data_path, json_file)) as f:
		data_load = json.load(f)
		x_load = data_load['x']
		y_load = data_load['y']
		lineshape = interpolate.interp1d(x_load, y_load, 'linear', bounds_error=False, fill_value='extrapolate')
	
	fitWithOffset = False
	if fitWithOffset:
		guess_FDG = [0.01, -3.98/EF_data, 0]
		bounds = ([0, -600, -np.inf],[np.inf, 0, np.inf])
		def convls(x, A, x0, C):
			return A*lineshape(x-x0)+C
	else:
		def convls(x, A, x0):
			return A*lineshape(x-x0)
		guess_FDG = [0.00001, -3.98/EF_data]
		bounds = ([0, -600],[1*scaling, 0])
	
	# fit the lineshape onto the data
	popt, pcov = curve_fit(convls, x_dimer/EF_data, y_dimer, sigma=yerr_dimer, p0=guess_FDG, bounds=bounds)
	perr = np.sqrt(np.diag(pcov))

	xx = np.linspace(-4.3/EF_data, -3.7/EF_data,1000)
	yyconvls640 = convls(xx, *[popt[0], popt[1]])
	ax.errorbar(x_dimer, y_dimer, yerr = yerr_dimer, **style640, ls='',  label=r'$\sigma = 28\,\mathrm{kHz} \approx 1.4\,E_F$') 
	ax.plot(xx * EF_data, yyconvls640, marker='', ls='-', color=color640)
# axis settings
	ax.set(xlim=[-4.04, -3.96],
		 #ylim=[-0.05, 2.5],
		 #xlabel=r'$\omega$ [MHz]',
		 ylabel=r'$\widetilde{\Gamma} \; \times \; 10^3$',
		 )

	tauF1 = 1/EF_data/2/pi
	trat1 = 640/tauF1
	

	from scipy.interpolate import UnivariateSpline
	spline = UnivariateSpline(xs, ys-np.max(ys)/2, s=0)
	r1, r2 = spline.roots()
	FWHM = np.abs(r1-r2) # EF
	print(f'FWHM={FWHM} MHz, or {FWHM/EF_data} EF')
	ax.text(0.2, 0.8, r'$t_\mathrm{rf} \gg\tau_F$', color=color640, fontsize=7, transform=ax.transAxes)
	# use the maximum to estimate the spectral weight if the transfer pulse were sinc^2
	#sinc fit on 2024-07-17 gives amp=6.9222e-3 +/- 4.1596e-4 (dimer_spectra_comparison.py)
	# relative uncertainty is 6%
	GammaPeak = ys.max()/scaling
	Id640 = GammaPeak/640/EF_data*2 # factor of 2 turns SW into Id
	tcurve = np.linspace(1, 640, 100)
	Idcurve = GammaPeak/tcurve/EF_data
	print(f'640 us Id = {Id640}')

	#ax = fig.add_subplot(gs[1, 0])
	#file2 = '2025-03-19_G_e_pulsetime=0.01.dat.pkl'
	file2='2024-09-27_B_e.dat_sat_corr.pkl'
	data = pd.read_pickle(os.path.join(data_path, file2))
	data = data.sort_values(by='detuning')
	scaling = 1000

	custom_bins = [-4.25,  -4.15, -4.1, -4.05, -4.025, -4, -3.975, -3.95,  -3.9,  -3.85, -3.8, -3.75, -3.7]
	print(data['c5_scaledtransfer'])
	x_dimer2, y_dimer2, yerr_dimer2 = bin_data(data['detuning'], data['c5_scaledtransfer']*scaling, data['em_c5_scaledtransfer']*scaling, nbins=custom_bins)
	fit2 = pd.read_pickle(os.path.join(data_path, 'fit_'+file2))
	xs2 = fit2['xs']/1e6
	ys2 = fit2['ys'] *scaling
	#EF_data = 0.0199
	EF_data = 0.0182
	# set offset to 0
	offs = ys2.min()
	ys2 = ys2-offs
	y_dimer2 = y_dimer2 - offs

	ax.errorbar(x_dimer2, y_dimer2, yerr=yerr_dimer2, **style10, ls='',label=r'$\sigma = 100\,\mathrm{kHz} \approx 5 E_F$')
	ax.plot(xs2, ys2, color=color10, marker='', ls='-')
	
	# numerically figure out FWHM
	from scipy.interpolate import UnivariateSpline
	spline = UnivariateSpline(xs2, ys2-np.max(ys2)/2, s=0)
	r1, r2 = spline.roots()
	FWHM = np.abs(r1-r2) 
	print(f'FWHM={FWHM} MHz, or {FWHM/EF_data} EF')

	# use the maximum to estimate the spectral weight if the transfer pulse were sinc^2
	# from dimer_spectra_comparison.py, fit gave amp of 1.3482e-3 +/- 1.01800733e-4
	# rel unc is 7.5% ~ 8%
	GammaPeak = ys2.max()/scaling
	Id10 = GammaPeak/10/EF_data*2 # factor of 2 turns SW into Id
	print(f'10 us Id = {Id10}')

	ax.set(xlim=[-4.15, -3.85],
		xlabel=r'$\omega$ [MHz]',
		 ylabel=r'$\widetilde{\Gamma} \, \times \, 10^3$',
		#ylim=[-0.5, 2.1]
	)
	
	yticks = [0, 2, 4, 6, 8]
	yticklabels = [str(x) for x in yticks]
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticklabels)

	tauF2 = 1/EF_data/2/pi
	trat2 = 10/tauF2
	ax.text(0.05, 0.25, r'$t_\mathrm{rf} \approx \tau_F$', color=color10, fontsize=7, transform=ax.transAxes)	

	ax=fig.add_subplot(gs[1,:])
	overall_scaling = 100
	file = 'veryshort_df.xlsx'
	data = pd.read_excel(os.path.join(data_path, file))
	data = data.sort_values(by='scaledtime', ascending=True)
	
	data['scaledtime']=1/data['scaledtime'] # tau_F/t_rf
	data['Id'] = data['Id'] * overall_scaling
	data['em_Id'] = data['em_Id'] * overall_scaling
	# trying to do some manual averaging
	df = data[data['time'] == 0.003].copy()
	t3us = df['scaledtime'].values[0]
	I3us = df['Id'].mean()
	I3us_std = np.sqrt(df['em_Id'].values[0]**2 + df['em_Id'].values[1]**2)

	df = data[data['time'] == 0.020].copy()
	t20us = df['scaledtime'].values[0]
	I20us = df['Id'].mean()
	I20us_std = np.sqrt(df['em_Id'].values[0]**2 + df['em_Id'].values[1]**2)

	t640us = float(1/trat1)
	I640us = float(Id640) * overall_scaling
	I640us_std = I640us*0.06 # based on relative uncertainty of sinc2 fit in dimer_spectra_comparison

	t10us = float(1/trat2)
	I10us = float(Id10) * overall_scaling
	I10us_std = I10us*0.08 # estimated based on rel unc of sinc2 fit in dimer_spectra_comparison.py

	#data = data[(data['time'] != 0.003) & (data['time']!=0.020)]
	data =data[data['time']!=0.020]
	data= data[data['time']>0.0031]
	my_color = '#8D6E63'
	my_style = {'color':my_color,
				   'mec':adjust_lightness(my_color, 0.3),
				   'mfc':my_color,
				   'mew':1,
				   'marker':'o',
				   'markersize':5,
				   'ls':'none'}
	# low time data (somewhat arbitrary)
	x_dimer = data['scaledtime']
	y_dimer = data['Id'] 
	yerr_dimer = data['em_Id'] 
	ax.errorbar(x_dimer, y_dimer, yerr_dimer, **my_style)
	ax.errorbar([t3us], [I3us], yerr=[I3us_std], **my_style)
	ax.errorbar([t20us], [I20us], yerr=[I20us_std], **my_style)
	color640 = '#4093ff'
	style640big = {'color':color640,
				   'mec':adjust_lightness(color640, 0.3),
				   'mfc':color640,
				   'mew':1,
				   'marker':'o',
				   'markersize':5}
	color10 = '#ff5447'
	style10big = {'color':color10,
				'mec':adjust_lightness(color10, 0.3),
				'mfc':color10,
				'mew':1,
				'marker':'s',
				'markersize':5}
	ax.errorbar([t640us], [I640us], yerr=[I640us_std], **style640big)
	ax.errorbar([t10us], [I10us], yerr=[I10us_std], **style10big)

	t_add = [t20us, t640us, t10us]
	I_add = [I20us, I640us, I10us]
	Ierr_add = [I20us_std, I640us_std, I10us_std]
	t_all = pd.concat([x_dimer, pd.Series(t_add)]).sort_values(ascending=True)
	I_all = pd.concat([y_dimer, pd.Series(I_add)]).sort_values(ascending=True)
	Ierr_all = pd.concat([yerr_dimer, pd.Series(Ierr_add)]).sort_values(ascending=True)
	avg_Id = np.mean(I_all.iloc[-4:])
	e_avg_Id = np.sqrt(Ierr_all.iloc[-4]**2 + Ierr_all.iloc[-3]**2 + Ierr_all.iloc[-2]**2 + Ierr_all.iloc[-1]**2)/4
	
	xs = np.linspace(0, t_all.max(), 100)
	ax.hlines(avg_Id, 0,xs.max(), ls='--', color=my_color)
	ax.fill_between(xs, avg_Id - e_avg_Id, avg_Id + e_avg_Id, color = adjust_lightness(my_color, 1.5))
	
	scale_est = pd.read_excel(os.path.join(data_path, 'time_scaling_estimate_avg_rescaled.xlsx'))
	ax.plot(scale_est['ftimes'], scale_est['peaks']*overall_scaling,color=color640, ls='-', marker='')
	
	ax.text(0.07, 0.0155*overall_scaling, r'$I_d$', color=my_color)
	ax.set(xlabel = r'$\tau_F/t_\mathrm{rf}$',
		 #ylabel = r'$\mathcal{I}_d$ response',
		 ylabel=r'$4 \alpha / \Omega_{23}^2  t_\mathrm{rf}^2  \, \times \, 10^2$',
		 ylim = [-0.1, 2.5],
		 xlim = [-0.1,2.1],
		 xscale='linear',
		# yscale='log'
		 )
	
	xticks = [0, 0.5, 1, 1.5, 2]
	xticklabels = [str(x) for x in xticks]
	ax.set_xticks(xticks)
	ax.set_xticklabels(xticklabels)

	
	#fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'manuscript_figures/fig2_new binding energy without Tmat.pdf')
		print(f'saving to {save_path}')
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

if Plot==6:

	a0 = 5.2917721092e-11
	B097 = 202.14
	B095 = 224.2
	Delta97 = 6.9
	Delta95 = 7.2

	def a_s(B, B0, Delta, abg=167.6*a0):
		return abg*(1-Delta/(B-B0))
	
	fig, ax = plt.subplots(figsize=(5, 4))
	
	Bs = np.linspace(190, 230, 200)
	a97 = a_s(Bs, B097, Delta97)
	a95 = a_s(Bs, B095, Delta95)
	ax.plot(Bs, a97/a0, '-')
	#ax.plot(Bs, a95/a0, '-')
	ax.hlines(0, Bs.min(), Bs.max(), color='black')
	ax.set(xlim = [190, 230],
		ylim=[-500, 500],
	)
	ax.tick_params(labelsize=12)

	ax.set_xlabel('Magnetic Field [G]', fontsize=14)
	ax.set_ylabel(r'Scattering Length $[a_0]$',fontsize=14)
	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'manuscript_figures/FBresonances_only97.png')
		print(f'saving to {save_path}')
		plt.savefig(save_path, dpi=1200)
	if Show: plt.show()


if Plot == 7:
	data = pd.read_excel(os.path.join(data_path, 'cmonman.xlsx'))
	data['detuning'] = data['freq (MHz)']-47.2227
	color640 = '#4093ff'
	style640 = {'color':color640,
				   'mec':adjust_lightness(color640, 0.3),
				   'mfc':color640,
				   'mew':1,
				   'marker':'o',
				   'markersize':5}
	fig, ax = plt.subplots(figsize=(3.4, 2.8))
	spin = 'c9'
	binx, biny, binyerr = bin_data(data['detuning'], data[spin], data['em_'+spin], nbins=25)
	def gaussian(x, A, x0, sigma, C):
		return A*np.exp(-(x-x0)**2 / (2*sigma**2))+C
	
	popt, pcov = curve_fit(gaussian, data['detuning'], data[spin], p0=[-2000, -4, 0.1, 8000])
	xs = np.linspace(data['detuning'].min(), data['detuning'].max(), 200)
	ys = gaussian(xs, *popt)
	ax.errorbar(binx,biny, binyerr, **style640)
	ax.plot(xs, ys, 'r-')
	ax.set(
		xlabel='Detuning [MHz]',
		ylabel=r'$\alpha_d$'
	)

if Plot == 8:
	file = '2024-07-17_J_e.dat.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	scaling = 1
	data = data.sort_values(by='detuning')

	custom_bins = [-4.05,  -4.03, -4.02, -4.01, -4.005, -4, -3.995, -3.990, -3.985, -3.98, -3.97, -3.96,  -3.94]
	x_dimer, y_dimer, yerr_dimer = bin_data(data['detuning'], data['c5_scaledtransfer']*scaling, data['em_c5_scaledtransfer']*scaling, nbins=custom_bins)
	#arb_y_scaling = 0.00994107 / 0.0075
	#arb_y_scaling= 1
	fit = pd.read_pickle(os.path.join(data_path, 'fit_'+file))
	xs = fit['xs']/1e6
	ys = fit['ys'] *scaling
	#EF_data = 0.0199
	EF_data = 0.0133
	# set offset to 0
	offs = ys.min()
	ys = ys-offs
	y_dimer = y_dimer - offs

	json_file = 'lineshape_2024-07-17_J_e_backup.json'

	with open(os.path.join(data_path, json_file)) as f:
		data_load = json.load(f)
		x_load = data_load['x']
		y_load = data_load['y']
		lineshape = interpolate.interp1d(x_load, y_load, 'linear', bounds_error=False, fill_value='extrapolate')
	
	fitWithOffset = False
	if fitWithOffset:
		guess_FDG = [0.01, -3.98/EF_data, 0]
		bounds = ([0, -600, -np.inf],[np.inf, 0, np.inf])
		def convls(x, A, x0, C):
			return A*lineshape(x-x0)+C
	else:
		def convls(x, A, x0):
			return A*lineshape(x-x0)
		guess_FDG = [0.00001, -3.98/EF_data]
		bounds = ([0, -600],[1*scaling, 0])
	
	# fit the lineshape onto the data
	popt, pcov = curve_fit(convls, x_dimer/EF_data, y_dimer, sigma=yerr_dimer, p0=guess_FDG, bounds=bounds)
	perr = np.sqrt(np.diag(pcov))

	# plot binned data and fit lineshape
	fig, ax = plt.subplots()
	xx = np.linspace(-5/EF_data, -3/EF_data,2000)
	yyconvls640 = convls(xx, *[popt[0], popt[1]])
	ax.errorbar(x_dimer, y_dimer, yerr = yerr_dimer, **style640, ls='',  label=r'$\sigma = 28\,\mathrm{kHz} \approx 1.4\,E_F$') 

	ax.plot(xx * EF_data, yyconvls640, marker='', ls='-', color=color640)
	ax.set(ylim=[-0.001, 0.007],
		xlim=[-4.2, -3.8])


	# define a normalized sinc^2 function using dimensionelss time
	def sinc2(x, x0, T):
		T = T * EF_data # T in us, EF_data in MHz
		# if I don't piecewise this then x=x0 gives a nan
		return np.piecewise(x, [(x==x0), (x!=x0)], 
		 [lambda x: 1*T,
		lambda x: 1*T*np.sin((x-x0)*T*np.pi)**2 / ((x-x0)*T*np.pi)**2]) # the sin part normalizes to 1/T, so multiply by T to get 1
	
	tpulses = [3, 4, 5, 6, 7,8, 9,  10,11,  12.5,14,  15,17.5, 19, 20, 25, 30, 35, 40, 50, 75, 100, 250, 500, 640, 800] # us
	#tpulses = [10, 100, 1000]
	# j = 0 is for low, 1 is for mean, 2 is for high
	for j in range(0, 3):
		peaks = []
		ftimes = []
		if j != 1: continue
		if j == 0:
			amp_mod = -perr[0]
		if j==1 :
			amp_mod = 0
		if j ==2: 
			amp_mod = perr[0]
			
		yyconvls640 = convls(xx, *[popt[0] + amp_mod, popt[1]])
		arb_y_scaling = 2*0.00994107 / 0.00751
		yyconvls640 = yyconvls640 * arb_y_scaling
		norm_conv = np.trapezoid(yyconvls640, xx)
		print(f'norm of the lineshape is {norm_conv}')
		fig, axs = plt.subplots(len(tpulses))
		for i, tpulse in enumerate(tpulses):
			print(f'Beginning loop for t={tpulse} us...')
			ysinc2 = sinc2(xx, -4/EF_data, tpulse)
			norm_sinc2 = np.trapezoid(ysinc2, xx)
			print(f'norm of sinc2 is {norm_sinc2}')
			ysinc2 = ysinc2 * norm_conv
			norm_sinc2 = np.trapezoid(ysinc2, xx)
			print(f'After scaling, new norm is {norm_sinc2}')
			ax.plot(xx*EF_data, ysinc2, 'r-')

			# define convolution product function -- original lineshape slided through by sinc^2
			def convprod(tau, t):
				return convls(tau, (popt[0] + amp_mod) * arb_y_scaling, 0) * sinc2(t-tau, 0, tpulse)
			# integrate the product function
			def convintegral(t):
				sliderange=30
				qrangelow = -sliderange
				qrangehigh = sliderange
				return quad(convprod, qrangelow, qrangehigh, args=(t,))
			
			yyconv = []
			e_yyconv = []
			xxC = np.linspace(-30, 30, 1000)
			for x in xxC:
				a, b = convintegral(x)
				yyconv.append(a)
				e_yyconv.append(b)
			
			axs[i].plot((xx - np.mean(xx)), yyconvls640, 'b-', label='FD')
			axs[i].plot((xx - np.mean(xx)), ysinc2, 'r-', label=f'sinc2, T={tpulse} us')
			axs[i].plot(xxC, yyconv, 'g--', label='convolution')
			axs[i].set(ylim=[-0.001, 0.008],
			xlim=[-10, 10])
			axs[i].legend()

			# the maximum value of the convolution is the estimate of the peak transfer for a certain pulse time
			# extract them and then compare to fermi time
			peak = np.array(yyconv).max()/tpulse/EF_data
			print(peak)
			peaks.append(peak)
			ftime = (1/EF_data/2/pi)/tpulse
			ftimes.append(ftime)

	# save and output
		fig, axp = plt.subplots()
		axp.plot(ftimes, peaks)
		out_df = pd.DataFrame({
			'ftimes':ftimes,
			'peaks':peaks
		})
		if j == 0:
			out_suffix='_low'
		if j ==1:
			out_suffix = '_avg'
		if j == 2:
			out_suffix = '_high'
		print(out_df['peaks'].max())
		out_df.to_excel(os.path.join(data_path, 'time_scaling_estimate' + out_suffix+ '_rescaled.xlsx'))