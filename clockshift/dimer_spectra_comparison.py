# -*- coding: utf-8 -*-
"""
2025-03-19
@author: Chip Lab
"""
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
	sys.path.append(parent_dir)
from library import pi, h, hbar, mK, a0, plt_settings, styles, colors
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

lineshape = 'sinc2'
spins = ['c5', 'c9', 'ratio95']
spins=['c5']

correct_spinloss = True
saturation_correction = False
gaussian_cloud = True
fixed_width = True
fixed_x0 = False
free_offset = False
save=False
plot_raw=False
# Omega^2 [kHz^2] 1/e saturation value
x0_trf_10 =  3225.60  # sq_weighted avg of all x0 from fit dimer sat curves
x0_trf_640 = 31.6175  # from 2025-03-19_G

def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

def saturation_scale(x, trf):
	""" x is OmegaR^2 and x0 is fit 1/e Omega_R^2 """
	if trf == 10:
		x0 = x0_trf_10
	elif trf == 640:
		x0 = x0_trf_640
	return x/x0*1/(1-np.exp(-x/x0))
	
def gaussian(x, A, x0, sigma, C):
	return A*np.exp(-(x-x0)**2/(2*sigma**2)) + C

def sinc2(x, A, x0, T, C):
	# if I don't piecewise this then x=x0 gives a nan
	return np.piecewise(x, [(x==x0), (x!=x0)], 
	 [lambda x: A+C,  
	   lambda x: A*np.sin((x-x0)*T*np.pi)**2/((x-x0)*T*np.pi)**2 + C])

def sinc2_nobg(x, A, x0, T):
	# if I don't piecewise this then x=x0 gives a nan
	return np.piecewise(x, [(x==x0), (x!=x0)], 
	 [lambda x: A,  
	   lambda x: A*np.sin((x-x0)*T*np.pi)**2/((x-x0)*T*np.pi)**2])

def sinc2_nobg_10us(x, A, x0):
	T = 10e-6
	# if I don't piecewise this then x=x0 gives a nan
	return np.piecewise(x, [(x==x0), (x!=x0)], 
	 [lambda x: A,  
	   lambda x: A*np.sin((x-x0)*T*np.pi)**2/((x-x0)*T*np.pi)**2])

def sinc2_nobg_10us_EF(x, A, x0):
	T = 10*EF #us *2pi MHz
	# if I don't piecewise this then x=x0 gives a nan
	return np.piecewise(x, [(x==x0), (x!=x0)], 
	 [lambda x: A,  
	   lambda x: A*np.sin((x-x0)*T*np.pi)**2/((x-x0)*T*np.pi)**2])

def sinc2_10us_EF(x, A, x0, C):
	T=10*EF #us *2pi MHz
	# if I don't piecewise this then x=x0 gives a nan
	return np.piecewise(x, [(x==x0), (x!=x0)], 
	 [lambda x: A+C,  
	   lambda x: A*np.sin((x-x0)*T*np.pi)**2/((x-x0)*T*np.pi)**2 + C])
					

def sinc2_nobg_fixedx0(x, A, T):
	#x0 = -200.977119
	x0 = -3.98/EF
	# if I don't piecewise this then x=x0 gives a nan
	return np.piecewise(x, [(x==x0), (x!=x0)], 
	 [lambda x: A,  
	   lambda x: A*np.sin((x-x0)*T*np.pi)**2/((x-x0)*T*np.pi)**2])

def Int2DGaussian(a, sx, sy):
	return 2*a*np.pi*sx*sy



# data binning
binning=False
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

### constants
re = 107 * a0 # ac dimer range estimate
Eb = 3.98 # MHz # I guesstimated this from recent ac dimer spectra
kF = 1.1e7
kappa = np.sqrt((Eb*h*10**6) *mK/hbar**2) # convert Eb back to kappa

### Vpp calibration
VpptoOmegaR47 = 12.01/0.452 # kHz/Vpp - 2025-02-12 calibration 
VpptoOmegaR43 = 14.44/0.656 *VpptoOmegaR47/(17.05/0.728) # fudged 43MHz calibration
phaseO_OmegaR = lambda VVA, freq: 2*pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)

def spin_map(spin):
	if spin == 'c5' and not correct_spinloss:
		return 'b'
	elif spin == 'c5' and correct_spinloss:
		return 'b/2'
	elif spin == 'c9':
		return 'a'
	elif spin == 'sum95':
		return 'a+b'
	elif spin == 'ratio95' or spin == 'dimer':
		return 'a/b'
	else:
		return ''
	
# okay start analyzing
files = [
# 		'2025-03-13_F_e_pulsetime=0.01.dat',
# 		'2025-03-13_F_e_pulsetime=0.64.dat',
# 		'2025-03-14_F_e.dat',  # 70mG wiggle
# 		'2025-03-18_H_e.dat',  # finely scanned
		'2025-03-19_G_e_pulsetime=0.64.dat',
		'2025-03-19_G_e_pulsetime=0.01.dat',
		]

Vpps = [
# 		2.32,
# 		0.29,
# 		0.29,
# 		0.29,
		0.29,
		2.32,
		] # Vpp
trfs = [
# 		10,
# 		640,
# 		640,
# 		640,
		640,
		10,
		] # us
ToTFs = [
# 		0.476,
# 		0.476,
# 		0.485,
# 		0.573,
		0.55,
		0.55,
		]
EFs = [
# 	   0.0167,
# 	   0.0167,
# 	   0.0167,
# 	   0.0191,
	   0.0199,
	   0.0199,
	  ] # MHz

res_freq = 47.2227
dimer_freq = 43.238
ff = 0.88

plt.rcParams.update(plt_settings) # from library.py
fig, axs = plt.subplots(1,3, figsize=(12, 6))
fig.suptitle(r"Dimer spectral weight comparison for $T \approx 0.6 T_F$")


for i, file in enumerate(files):
	print()
	print("*-------------------------------------*")
	print("Analyzing file ", file)
	run = Data(file, path=parent_dir + '\\clockshift\\data')
	
	# Omega Rabi of square pulse
	OmegaR = 2*pi*VpptoOmegaR43*Vpps[i] # 2 pi kHz
	print("OmegaR = {:.3f} 1/ms".format(OmegaR))
	
	# dataset parameters
	trf = trfs[i]
	ToTF = ToTFs[i]
	Vpp = Vpps[i]
	EF = EFs[i]
	print("Vpp = {:.3f}V".format(Vpp))
	print("trf = {:.0f} us".format(trf))
	print("EF = {:.1f} us".format(EF*1e3))  # kHz
	
	# calculate detuning
	run.data['detuning'] = (run.data['freq'] - res_freq) # MHz
	run.data['detuning_EF'] = run.data['detuning']/EF # dimensionless
	run.data['detuning_Hz'] = run.data['detuning']*1e6 # Hz
	
	run.data = run.data.drop(run.data[run.data.detuning < -5].index)
	
	if gaussian_cloud == True:
		run.data['c5'] = Int2DGaussian(run.data['two2D_a1'], 
								 run.data['two2D_sh1'], run.data['two2D_sv1'])
		run.data['c9'] = Int2DGaussian(run.data['two2D_a2'], 
								 run.data['two2D_sh2'], run.data['two2D_sv2'])
	
	# fudge the c9 counts using ff
	run.data['c9'] = run.data['c9'] * ff
	
	# determine bg counts
	try:
		if 0 in run.data['VVA'].unique():  # then use the zero VVA point
			VVA_zero_exists = True
		else:
			VVA_zero_exists = False
	except KeyError:
		VVA_zero_exists = False
		
	if VVA_zero_exists:
		bg_df = run.data[run.data['VVA'] == 0]
		run.data = run.data[run.data['VVA'] != 0]
	
	# plot raw data
	if plot_raw:
		fig_raw, axs_raw = plt.subplots(1,2, figsize=(8, 4))
		fig_raw.suptitle(file)
		x = run.data.detuning
		xs = np.linspace(-4.4, -3.6, 2000)
		bgs = []
		for ax, spin, label in zip(axs_raw, ['c9', 'c5'], ['a', 'b']):
			ax.set(ylabel=label+' counts', xlabel='Detuning (MHz)')
			
			# plot data
			y = run.data[spin]
			ax.plot(x, y)
			
			# fit guess
			width_guess = min(trf, 1/EF)
			p0 = [(min(y)-max(y))/2, dimer_freq-res_freq, width_guess, (max(y)+min(y))/2]
			ax.plot(xs, sinc2(xs, *p0), 'k--', label='guess')
			print(p0)
			
			# try fit
			try:	
				popt, pcov = curve_fit(sinc2, np.array(x), y, p0=p0)
				print(popt)
				ax.plot(xs, sinc2(xs, *popt), 'r-', label='fit')
				perr = np.sqrt(np.diag(pcov))
				bg = popt[3]
			except RuntimeError:
				print("Couldn't fit bg for ", spin)
				bg = y.mean()
			bgs.append(bg)
			ax.hlines(bg, min(x), max(x), linestyle='--', color='grey', label='bg fit')
			
		if VVA_zero_exists:
			# plot these bg lines
			axs_raw[0].hlines(bg_df.c9.mean(), min(x), max(x), linestyle=':', label='VVA=0')
			axs_raw[1].hlines(bg_df.c5.mean(), min(x), max(x), linestyle=':', label='VVA=0')
			
		else:  # use data fit to determine bg_dfs
			bg_dict = {'c9':[bgs[0]], 'c5': [bgs[1]]}
			bg_df = pd.DataFrame.from_dict(bg_dict)
			
		# legends and right layout
		for ax in axs_raw:
			ax.legend()
		fig_raw.tight_layout()
		
	
	# compute other count data
	for df in [run.data, bg_df]:
		df['sum95'] = df['c5'] + df['c9']
		df['ratio95'] = df['c9']/df['c5']
		df['f5'] = df['c5']/df['sum95']
		df['f9'] = df['c9']/df['sum95']
		
	# compute saturation correction
	sat_scale_dimer = saturation_scale(OmegaR**2/(2*np.pi)**2, trf)
	print(sat_scale_dimer)
	
	# compute transfer for loss and ratio
	for j, (spin, sty, color) in enumerate(zip(spins, styles, colors)):
		if spin == 'c5' or spin == 'c9':
			# compute bg
			bg_counts = bg_df[spin].mean()
# 			e_bg_counts = bg_df[spin].sem()
			
			run.data[spin+'_alpha'] = 1-run.data[spin]/bg_counts
			
			if correct_spinloss and spin == 'c5':
				run.data[spin+'_alpha'] = run.data[spin+'_alpha']/2
				
		elif spin == 'ratio95':
			bg_f9_mean = bg_df['f9'].mean()
			bg_f5_mean = bg_df['f5'].mean()
			# point by point for signal but background is mean
			run.data[spin+'_alpha'] = (bg_f9_mean - run.data['ratio95']*bg_f5_mean)\
					/(1/2-run.data['ratio95'])
					
		# correct transfer from saturation scaling		
		run.data[spin+'_alpha']=run.data[spin+'_alpha'] *sat_scale_dimer
		run.data[spin+'_transfer'] = run.data[spin+'_alpha'] / (trf/1e6) / (OmegaR/(2*np.pi)*1e3)**2 / np.pi # 1/Hz wait why is there a pi
		run.data[spin+'_scaledtransfer'] = GammaTilde(run.data[spin+'_alpha'], h*EF*1e6, OmegaR*1e3, trf/1e6) # dimless
		
		# average results
		# make sure you pair the right dimensions together
		# xparam = 'detuning_Hz'
		# yparam = spin+ '_transfer'
		xparam='detuning_EF'
		yparam = spin+ '_scaledtransfer'
		if yparam == spin + '_transfer':
			ylabel = r'$\alpha/t/\Omega_R^2$ [1/Hz]'
		elif yparam == spin+ '_scaledtransfer':
			ylabel = r'$\widetilde{\Gamma}$'
		if xparam == 'detuning_Hz':
			xlabel = r'$\omega$ [Hz]'
		elif xparam == 'detuning_EF':
			xlabel = r'$\tilde{\omega}$'

		mean = run.data.groupby([xparam]).mean().reset_index()
		sem = run.data.groupby([xparam]).sem().reset_index().add_prefix("em_")
		std = run.data.groupby([xparam]).std().reset_index().add_prefix("e_")
		avg_df = pd.concat([mean, std, sem], axis=1)
		if trf==640:
			test_df = avg_df

		# fit and integrate to get spectral weight
		deltax = max(avg_df[xparam]) - min(avg_df[xparam])
		extend_int = 3
		xs = np.linspace(min(avg_df[xparam]) - extend_int*deltax, 
				   max(avg_df[xparam]) + extend_int*deltax, 10000)
		if xparam == 'detuning_Hz':
			p0 = [avg_df[yparam].max(), (dimer_freq-res_freq)*1e6, 
						min(trf, 1/EF)/1e6]
		elif xparam == 'detuning_EF':
			p0 = [avg_df[yparam].max(), avg_df[xparam].mean(), 
						1/trf/EF]
		
		if binning == True:
			xdata, ydata, ydataerr = bin_data(avg_df[xparam], avg_df[yparam],
						 avg_df['em_' + yparam], 12, xerr = None)
		else:
			xdata =avg_df[xparam]
			ydata = avg_df[yparam]
			ydataerr = avg_df['em_' + yparam]
		
		if trf == 640: 
			if fixed_x0:
				popt, pcov = curve_fit(sinc2_nobg_fixedx0, np.array(xdata), ydata,
							p0=[p0[0], p0[2]],
							#sigma=ydataerr,
				)
				ys = sinc2_nobg_fixedx0(xs, *popt)
				perr = np.sqrt(np.diag(pcov))
				popt=np.append(popt, popt[-1])
				perr=np.append(perr,perr[-1]) # filler value to make list length consistent
				cov02 = 0
			else:
				fitfunc = sinc2_nobg if not free_offset else sinc2
				myp0 = p0 if not free_offset else [p0[0], p0[1], p0[2], 0]
				popt, pcov = curve_fit(fitfunc, np.array(xdata), ydata, 
							#p0=p0, 
							p0=myp0,
	 						#sigma=ydataerr,
							)
				ys = fitfunc(xs, *popt)
				perr = np.sqrt(np.diag(pcov))
				cov02 = pcov[0,2]

		elif trf == 10:
			if not fixed_width:
				popt, pcov = curve_fit(sinc2_nobg, np.array(xdata), ydata, 
							p0=p0,
	 						#  sigma=ydataerr,
							)	
				ys = sinc2_nobg(xs, *popt)
				perr = np.sqrt(np.diag(pcov))
				cov02 = 0
			else:
				if xparam == 'detuning_EF':
					if free_offset:
						fitfunc = sinc2_10us_EF 
						myp0 = [p0[0],p0[1],0]
					else:
						fitfunc = sinc2_nobg_10us_EF
						myp0 = [p0[0],p0[1]]
					popt, pcov = curve_fit(fitfunc, np.array(xdata), ydata, 
									p0=myp0, 
	 							 # sigma=ydataerr,
									)
					ys = fitfunc(xs, *popt)
					perr = np.sqrt(np.diag(pcov))
					if len(myp0) < 3:
						popt = np.append(popt, trf*EF)
						perr = np.append(perr, 0.1/1e6)  # made up time unceratinty
					cov02 = 0
				if xparam == 'detuning_Hz':

					popt, pcov = curve_fit(sinc2_nobg_10us, np.array(xdata), ydata, 
									p0=[p0[0], p0[1]], 
		# 							  sigma=ydataerr,
									)
					ys = sinc2_nobg_10us(xs, *popt)
					perr = np.sqrt(np.diag(pcov))
					popt = np.append(popt, trf/1e6)
					perr = np.append(perr, 0.1/1e6)  # made up time unceratinty
					cov02 = 0
			
# 		axs[j].plot(xs, sinc2_nobg(xs, *p0), ls='-', marker='', color=colors[i])
		SW_int = np.abs(np.trapz(ys, xs))
		SW = popt[0]/popt[2]
		
		# gotta do the covariance propagation correctly
		e_SW = SW*np.sqrt((perr[0]/popt[0])**2+(perr[2]/popt[2])**2 + \
					2*perr[0]/popt[0]*perr[2]/popt[2]*np.sqrt(cov02))
		
		freq_width = 1/popt[2]/1e3  # kHz
		e_freq_width = freq_width*perr[2]/popt[2]
		if spin == 'c5':
			print(f'{spin} fit: {popt} +/- {np.sqrt(np.diag(pcov))}')
			print(f'{spin} SW integrated: {SW_int}')
			print(f'{spin} SW calculated: {SW:.3f}({1000*e_SW:.0f})')
			print(f'{spin} amplitude: {popt[0]}')
			print(f'{spin} peak: {ydata.max()}')
			print(f'{spin} OmegaR^2: {(OmegaR)**2/(2*np.pi)**2} kHz^2')
			print(f'{spin} 1/width: {popt[2]*1e6:.2f}({perr[2]*1e8:.0f}) us')
			print(f'{spin} width: {freq_width:.0f}({e_freq_width:.0f}) kHz')
			print(f'sat scale dimer: {sat_scale_dimer:.2f}')
			
		# plotting
		label = f'trf={trfs[i]} us, SW = {SW:.3f}({1000*e_SW:.0f})'
# 		label = ''
		axs[j].errorbar(xdata, ydata, ydataerr,
				   **styles[i], label=label)
		axs[j].plot(xs, ys, ls='-', marker='', color=colors[i])
# 		axs[j].hlines(popt[-1], min(xdata), max(xdata), 
# 				color=colors[i*(j+1)], ls='--')
		axs[j].set(
# 			ylim=[min(ydata), max(ydata)],
			 title = spin_map(spin),
			 ylabel=ylabel,
			 xlabel=xlabel,
			 xlim=[min(xdata), max(xdata)])
		
		axs[j].legend()
		
		if spin == 'c5' and save:
			
			
			
			save_path = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis\\clockshift\\manuscript\\manuscript_data\\'
			fit = {'xs':xs,'ys':ys}
			
			avg_df.to_pickle(save_path + file + '.pkl')
			with open(save_path+'fit_'+file+'.pkl', 'wb') as handle:
				pkl.dump(fit, handle)
	
fig.tight_layout()




	