# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:34:41 2025

These are a collection of select figures for drafting the clock shift manuscript.

@author: coldatoms
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

proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'analyzed_data')

from library import pi, h, hbar, mK, a0, plt_settings, styles, colors, adjust_lightness
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
import matplotlib.patches as patches
import pickle as pkl

# Import CMasher to register colormaps
import cmasher as cmr
# Access rainforest colormap through CMasher or MPL
cmap = cmr.rainforest                   # CMasher
#cmap = plt.get_cmap('cmr.rainforest')   # MPL

Save = True
Show = True

Plot = 1

#region ################# PLOT short(?) PULSE DIMER AND INSET DIMER VS. FIELD
if Plot == 1:
	yparam='ScaledTransfer'
	file = '2024-11-04_M_e_2shotanalysis.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	x_dimer = data['freq'][0]
	y_dimer = data[yparam][0]
	yerr_dimer = data['e_' + yparam][0]
	xx = data['xx']
	yy= data['yy']
	
	# Eb vs field
	Ebs = pd.read_pickle(os.path.join(data_path, 'Ebs.pkl'))
	ExpEbs = pd.read_pickle(os.path.join(data_path, 'ExpEbs.pkl'))

	fig, ax = plt.subplots(figsize=(6,3))
	# inset axis
	left, bottom, width, height = [0.69, 0.35, 0.2, 0.26]
	ax2 = fig.add_axes([left, bottom, width, height])
	ax.plot(Ebs['B'], Ebs['Ebs_full'], ls='-', color='blue', marker='', label='T-matrix')
	ax.plot(Ebs['B'], Ebs['Ebs_naive'], ls='--', color='red', marker='',  label=r'$1/a^2$')
	ax.errorbar(ExpEbs['B'], ExpEbs['Eb'], yerr=ExpEbs['e_Eb'], **styles[0], markersize=4) # TO DO: find multiple measurements for Eb at 202.14 G and get a more accurate one for the figure. Also, consider width of fit as uncertainty
	#ax2.plot([199, 225], [x_dimer, x_dimer], color=colors[1], marker='', ls = '--')
	xlabel=r'$B$ [G]'
	ylabel = r'$\omega_d$ [MHz]'
	ax.set(xlabel=xlabel, ylabel=ylabel)

	# main axis
	#ax1.errorbar(x, y, yerr, marker='o', ls='', markersize =10, capsize=2, mew=2, elinewidth=2, mec=adjust_lightness('tab:gray',0.2), color='tab:gray')
	ax2.plot(xx*data['EF'][0], yy, '--', color=colors[1])
	ax2.errorbar([x_dimer, x_dimer], [0, y_dimer], yerr=[yerr_dimer*np.array([1,1])], **styles[1])
	#ax1.axvspan(fit_Eb-fit_e_Eb, fit_Eb+fit_e_Eb, color=adjust_lightness('tab:blue', 0.2))
	xlabel=r'Detuning $\omega$ [MHz]'
	ylabel = r'Transfer $\widetilde{\Gamma}(\omega)$'
	xlims=[-4.10, -3.8]
	#ylims=[0, 0.005]
	ax2.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)
	ax2.set_xlabel(xlabel, fontsize=8)
	ax2.set_ylabel(ylabel, fontsize=8)
	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'figures/manuscript/dimer_short_pulse_v2.png')
		print(f'saving to {save_path}')
		plt.savefig(save_path)
	if Show: plt.show() 
#endregion

#region ######## PLOT HFT ON LINERA SCALE WITH SATURATION CURVES AT RES AND FAR FROM RES
if Plot == 2:

	# HFT data
	file = '2024-10-08_F_e.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	x = data['detuning_EF']
	y = data['ScaledTransfer']
	yerr = data['em_ScaledTransfer']
	
	fig, ax = plt.subplots()
	ax.errorbar(x, y, yerr, **styles[3])

	xlabel=r'Detuning $\hbar\omega/E_F$'
	ylabel = r'Transfer $\widetilde{\Gamma}(\omega)$'
	xlims = [-2, 32]
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlims)
	ax.set_yscale('linear')
	ax.set_xscale('linear')
	
	# # saturation curves
	# # save files 
	# files = [
	# 		'100kHz_saturation_curves.pkl', 
	# 		'near-res_saturation_curves.pkl',
	# 		'various_ToTF_saturation_curves.pkl',
	# 		]
	# loaded_data = []
	# # grab dictionary lists from pickle files
	# saturation_data_path = os.path.join(proj_path, 'saturation_data')
	# for i, file in enumerate(files):
	# 	with open(os.path.join(saturation_data_path, file), "rb") as input_file:
	# 		loaded_data = loaded_data + pkl.load(input_file)
		
	# # turn dictionary list into dataframe
	# df = pd.DataFrame(loaded_data)
	# # print relevant columns for data selection
	# print(df[['file', 'detuning', 'ToTF']])

	# res_df = df.loc[(df.ToTF==0.384) & (df.detuning==0)]
	# print(res_df.head())

	if Show: plt.show()
#endregion


#region ######## PLOT DIMER SPECTRAL WEIGHT AND INSET CLOCK SHIFT WITH THEORY
if Plot == 3:
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
	spin = 'dimer' # this means ratio analysis

	with open(os.path.join(data_path,'spectral_weight_summary.pkl'),'rb') as handle:
		summary = pkl.load(handle)

	sum_rule = 0.5
	open_channel_fraction = 0.93
	kF = np.mean(summary['kF'])
	a13kF = kF * a13(Bfield)
	C = np.linspace(0, max(summary['C_data']),50)
	
	### ZY single-channel square well w/ effective range
	# divide I_d by a13 kF,
	I_d = sum_rule * kF/(pi*kappa) * 1/(1+re/a13(Bfield)) * C / a13kF * open_channel_fraction 
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

	# make figure
	fig, ax = plt.subplots(figsize=(6,3))
	
	# theory curves
	theory_labels = [r'1ch square well', r'CCC', r'CCC w/ spin corr.', r'zero range']
	ax.plot(C, I_d, '-', color=colors[2], label=theory_labels[0])
	ax.plot(C, I_d_CCC_sc, '-', color=colors[3], label=theory_labels[2])
	ax.plot(C, I_d_max, '-', color=colors[4], label=theory_labels[3])
	# experimental data
	x = summary['C_data']
	xerr = summary['C_data_std']
	y = summary['SW_'+spin] / summary['a13kF']
	yerr = np.abs(summary['e_SW_'+spin]) / summary['a13kF']
	ax.errorbar(x, y, yerr=yerr, xerr=summary['C_data_std'], **styles[5])
	
	ax.legend(loc='lower right')
	ax.set(xlabel=r"Contact [$k_F/N$]",
 		   ylabel=r"$I_d/k_Fa_{13}$")
	
	# inset axis
	left, bottom, width, height = [0.22, 0.6, 0.28, 0.28]
	ax2 = fig.add_axes([left, bottom, width, height])
	# Clock shift theory
	ax2.plot(C, FM_d, '-', color=colors[2], label=theory_labels[0])
	ax2.plot(C, FM_d_CCC_sc, '-', color=colors[3], label=theory_labels[2])
	ax2.plot(C, FM_d_max, '-', color=colors[4], label=theory_labels[3])

	# clock shift experiment
	x = summary['C_data']
	xerr = summary['C_data_std']
	y = summary['FM_'+spin] * summary['a13kF']
	yerr = np.abs(summary['e_FM_'+spin]) * summary['a13kF']
	ax2.errorbar(x, y, yerr=yerr, xerr=summary['C_data_std'],
		 label=spin_map(spin), **styles[5])
	ax2.set(
 		   ylabel=r"$\hbar\Omega_d  k_Fa_{13} /E_F $ ")

	

	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'figures/manuscript/spectral_weight.png')
		print(f'saving to {save_path}')
		plt.savefig(save_path)
	if Show: plt.show()

	

#endregion

#region ######## PLOT DIMER AND HFT TOGETHER ON LOG SCALE, NOISE FLOOR, AND 5/2 REGION
if Plot == 4:
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	yparam = 'ScaledTransfer' #'ScaledTransfer' or 'transfer'
	# dimer spectrum
	#file = '2024-07-17_J_e.pkl'
	file = '2024-07-17_J_e_ratio95.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	x_dimer = data['detuning']
	y_dimer = data[yparam]
	yerr_dimer = data['em_' + yparam]
	fit = pd.read_pickle(os.path.join(data_path, '2024-07-17_J_e_fit_doubletemp_ratio95.pkl'))
	fit_Eb = fit['Eb'][0]/1000
	fit_e_Eb = fit['e_Eb'][0]/1000

	# HFT spectrum
	file = '2024-09-10_L_e.pkl'
	data = pd.read_pickle(os.path.join(data_path, file))
	x_HFT = data['detuning']
	#print(x_HFT)
	y_HFT = data[yparam]
	yerr_HFT = data['em_' +yparam]



	#we do some fitting
	def powerlawtail(x, A):
		xstar = 2
		return A*x**(-3/2) / (1+x/xstar)
	def tail32(x, A):
		return A*x**(-3/2)
	def tail52(x, A):
		return A*x**(-5/2)
	
	def generate_fit_func(fit_func):
		print(fit_func.__name__)
		if fit_func.__name__ == 'powerlawtail':	
			fit_range = x_HFT.between(0.040, 4)
			guess = [0.1]
		elif fit_func.__name__ == 'tail32':
			fit_range = x_HFT.between(0.040, 0.125)
			guess = [0.1]
		elif fit_func.__name__ == 'tail52':
			fit_range = x_HFT.between(0.5, 2)
			guess = [0.1]
		#print(fit_range)
		#print(fit_func)
		x_fit = x_HFT[fit_range]
		y_fit = y_HFT[fit_range]
		yerr_fit = yerr_HFT[fit_range]
		#print(x_fit)
		popt, pcov = curve_fit(fit_func, x_fit, y_fit, sigma=yerr_fit, p0=guess)
		xfit = np.linspace(x_fit.min(), x_fit.max(), 1000)
		xall = np.linspace(x_HFT.min(), x_HFT.max()+10, 1000)
		yfit = fit_func(xfit, *popt)
		yall = fit_func(xall, *popt)
		print(popt)
		return xfit, xall, yfit, yall
	
	noisefloor = 1e-4 # CHECK THIS

	#fig, axs = plt.subplots(1,2,figsize=(8, 6))
	fig, ax = plt.subplots(figsize=(6,3))

	ax.errorbar(x_HFT, y_HFT, yerr_HFT, **styles[2])
	# fit lines
	xx, xxall, yy, yyall = generate_fit_func(tail32)
	ax.plot(xxall, yyall, color=colors[2], ls='dotted', marker='')
	ax.plot(xx, yy, color=colors[2], ls='-', marker='')
	xx, xxall, yy, yyall = generate_fit_func(tail52)
	ax.plot(xxall, yyall, color=colors[2], ls='dotted', marker='')
	ax.plot(xx, yy, color=colors[2], ls='-', marker='')
	# xx, yy = generate_fit_func(powerlawtail)
	# ax2.plot(xx, yy, color=colors[3], ls= '--', marker='')
	# noise floor
	# Create a Rectangle patch
	rect = patches.Rectangle((0,0), 10, noisefloor, linewidth=2, facecolor='red', fill=True, alpha = 0.1)
	# Add the patch to the Axes
	ax.add_patch(rect)
	# horizontal line for noise floor?
	ax.plot([0, 10], [noisefloor, noisefloor], color='red', marker='', ls = '--')
	# vertical line for trap dept
	ax.vlines(0.200, ymin=0, ymax=1,ls='dashed', color='black')
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlim([0.02, 5])
	ax.set_ylim([1e-7, 0.05])
	#ax.spines['left'].set_visible(False)
	#ax.yaxis.set_ticks_position('right')
	ax.yaxis.set_visible(False)

	# add text
	#ax.text(0.03, 0.5e-5, 'Noise floor', color='red')
	ax.text(0.06, 1e-2, r'$\omega^{-3/2}$')
	ax.text(0.6, 1.5e-5, r'$\omega^{-5/2}$')
	ax.text(0.21, 0.01, r'$U_t$')

	divider = make_axes_locatable(ax)
	axLin = divider.append_axes("left", size=3.0, pad=0, sharey=ax)
	axLin.set_xscale('linear')
	axLin.set_xlim([-5,0])

	axLin.errorbar(x_dimer, y_dimer, yerr = yerr_dimer, **styles[2])
	axLin.plot(fit['detuning']/1000, fit['transfer'], ls='-', marker='', color=colors[2], label='fit with double temp') # NOTE: I think a simple Gaussian looks better...

	#axLin.spines['right'].set_visible(False)
	axLin.yaxis.set_ticks_position('left')
	plt.setp(axLin.get_xticklabels(), visible=True)

	# zoom in inset axis

	left, bottom, width, height = [0.35, 0.3, 0.2, 0.3]
	ax_inset = fig.add_axes([left, bottom, width, height])
	ax_inset.errorbar(x_dimer, y_dimer, yerr=yerr_dimer, **styles[2], ls='')
	ax_inset.plot(fit['detuning']/1000, fit['transfer'], ls='-', marker='', color=colors[2], label='fit with double temp') # NOTE: I think a simple Gaussian looks better...
	ax_inset.vlines(fit_Eb, -0.1, 1, ls='--', color=colors[2])
	ax_inset.set(xlim=[-4.05, -3.95], ylim=[-0.007, 0.013])
	ax_inset.xaxis.set_major_locator(plt.MaxNLocator(2))
	ax_inset.yaxis.set_major_locator(plt.MaxNLocator(2))


	# 2 shot analysis in inset axis
	# file = '2024-11-04_M_e_2shotanalysis.pkl'
	# data = pd.read_pickle(os.path.join(data_path, file))
	# x_dimer = data['freq'][0]
	# y_dimer = data[yparam][0]
	# yerr_dimer = data['e_' + yparam][0]
	# xx = data['xx']
	# yy= data['yy']
	# ax_inset.plot(xx*data['EF'][0], yy, '--', color=colors[1])
	# ax_inset.errorbar([x_dimer, x_dimer], [0, y_dimer], yerr=[yerr_dimer*np.array([1,1])], **styles[1])


	#ax2.set_xticks(ax2.get_xticks()[::2])
	# Create a Rectangle patch
	# noisefloor=0.005
	# rect = patches.Rectangle((-5,-1), 5, noisefloor+1, linewidth=1, facecolor='red', fill=True, alpha = 0.2)
	# # Add the patch to the Axes
	# axLin.add_patch(rect)

	#xlabel = r'Detuning $\omega$ [MHz]'
	#ax2.set_yscale('linear')
	#ax2.set_xscale('linear')
	#ax2.set(xlim=[-4.04, -3.96], ylim=[-0.010, 0.015], ylabel=ylabel)
	#ax2.set_xticks([10,1,0.1], labels=[r'$-10^1$', r'$-10^0$', r'$-10^{-1}$'])
	#ax2.set_xticks([3.6,4.4])

	# #ax2.set_xticks(ax2.get_xticks()[::2])
	# # Create a Rectangle patch
	# noisefloor=0.005
	# rect = patches.Rectangle((-5,-1), 2, noisefloor+1, linewidth=1, facecolor='red', fill=True, alpha = 0.2)
	# # Add the patch to the Axes
	# ax.add_patch(rect)

	# plt.xlabel(xlabel)

	#ax.plot(fit['detuning']/1000, fit['transfer'], ls='-', marker='', color=colors[1], label='fit with double temp') # NOTE: I think a simple Gaussian looks better...
	xlabel = r'Detuning $\omega$ [MHz]'
	ylabel = r'Transfer $\widetilde{\Gamma}(\omega)$'

	# ax.set_yscale('log')
	# ax.set_xscale('symlog', linthresh=(-5,0.01))
	#ax.set_xscale('linear')
	#ax.set(ylim=ylims, xlim=[-5,5])

	# # inset axis for log dimer spectrum
	# # inset axis
	# left, bottom, width, height = [0.62, 0.62, 0.28, 0.28]
	# ax2 = fig.add_axes([left, bottom, width, height])
	# ax2=axs[0]
	# ax2.errorbar(x_dimer, y_dimer, yerr = yerr_dimer, **styles[2])
	# ax2.plot(fit['detuning']/1000, fit['transfer'], ls='-', marker='', color=colors[2], label='fit with double temp') # NOTE: I think a simple Gaussian looks better...

	# xlabel = r'Detuning $\omega$ [MHz]'
	# ax2.set_yscale('linear')
	# ax2.set_xscale('linear')
	# ax2.set(xlim=[-4.04, -3.96], ylim=[-0.010, 0.015], ylabel=ylabel)
	# #ax2.set_xticks([10,1,0.1], labels=[r'$-10^1$', r'$-10^0$', r'$-10^{-1}$'])
	# #ax2.set_xticks([3.6,4.4])

	# #ax2.set_xticks(ax2.get_xticks()[::2])
	# # Create a Rectangle patch
	# noisefloor=0.005
	# rect = patches.Rectangle((-5,-1), 2, noisefloor+1, linewidth=1, facecolor='red', fill=True, alpha = 0.2)
	# # Add the patch to the Axes
	# ax2.add_patch(rect)

	# plt.xlabel(xlabel)

	ax.set_xlabel(xlabel)
	axLin.set_ylabel(ylabel)

	fig.tight_layout()
	if Save: 
		save_path = os.path.join(proj_path, 'figures/manuscript/log_linear_spectra_small.png')
		print(f'saving to {save_path}')
		plt.savefig(save_path)
	if Show: plt.show()


#endregion