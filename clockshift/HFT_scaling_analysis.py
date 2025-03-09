# -*- coding: utf-8 -*-
"""
2024 Nov 12
@author: Chip Lab

"""
from data_class import Data
from scipy.optimize import curve_fit
from library import plt_settings, pi, styles, colors
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(proj_path, 'saturation_data')
root = os.path.dirname(proj_path)

# plot error bands for saturation curves
fill_between = True
save = True

pkl_file = os.path.join(data_path, "100kHz_saturation_curves.pkl")

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

def satratio(x, x0):
	return x/x0*1/(1-np.exp(-x/x0))

def quotient_propagation(f, A, B, sA, sB, sAB):
	return f* (sA**2/A**2 + sB**2/B**2 - 2*sAB/A/B)**(1/2)


if __name__ == '__main__':
	
	results_list = []
	
	detunings = [
				25,
				50,
				100,
				150,
				]
	
	files = [
 			"2024-11-28_P_e_detuning=25.dat",
 			"2024-11-28_P_e_detuning=50.dat",
 			"2024-11-28_O_e.dat",
			"2024-11-28_P_e_detuning=150.dat",
			  ]
	
	fudge_factors = [
 					0.98, 
				  0.98, 
				  0.98, 
				  0.98,
				  ]
	
	ToTF = 0.616  # from the next day, but should be roughly right
	EF = 19.4  # same as above
	
	pulse_time = 0.2  # ms
	
	popts = []
	perrs = []
	popts_l = []
	perrs_l = []
	
	#### PLOTTING #####
	# initialize plots
	fig, axes = plt.subplots(2, 2, figsize=(12,10))
	axs = axes.flatten()
	
	### plot settings
	plt.rcParams.update(plt_settings) # from library.py
	plt.rcParams.update({"figure.figsize": [12,8],
						 "font.size": 14,
						 "lines.markeredgewidth": 2,
						 "errorbar.capsize": 0})
	
	fig.suptitle("200us Blackman HFT transfer saturation")
	
	### ANALYSIS ###
	xname = 'VVA'
	plot_name = "Omega Rabi Squared (1/ms^2)"
	
	
	axs[0].set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Transfer',
			   ylim=[-0.05, 0.55])
	axs[1].set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Loss',
		   ylim=[-0.05, 0.65])
	axs[2].set(xlabel='Measured transfer', 
			ylabel='Calibrated linear transfer')
	axs[3].set(xlabel='Measured loss', 
			ylabel='Calibrated linear loss')
	
	for i in range(len(detunings)):
		results = {}
		file = files[i]
		detuning = detunings[i]
		ff = fudge_factors[i]
		
		
		### Omega Rabi calibrations
		# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
		VpptoOmegaR47 = 17.05/0.728 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
		VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
		phaseO_OmegaR = lambda VVA, freq: VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)
				
		print("Analyzing", file)
		run = Data(file, path=data_path)
		run.data['ToTF'] = ToTF
		run.data['pulse_time'] = pulse_time
		run.data['EF'] = EF
		run.data['c9'] = ff * run.data['c9']
		
		# get bg data
		bg_df = run.data.loc[(run.data.VVA == 0)]
		run.data = run.data.drop(bg_df.index)
		
		# calculate bg values
		bg_c5 = bg_df.c5.mean()
		e_bg_c5 = bg_df.c5.sem()
		bg_c9 = bg_df.c9.mean()
		e_bg_c9 = bg_df.c9.sem()
		
		# calculate atom number, transfer and loss
		run.data['N'] = run.data.c5 - bg_c5 + run.data.c9
		run.data['transfer'] = (run.data.c5 - bg_c5)/(run.data.N)
		run.data['loss'] = (bg_c9 - run.data.c9)/bg_c9
		
		if detuning == 100: # for this dataset, we had to add the VVA to df
			run.data['OmegaR'] = phaseO_OmegaR(run.data.VVA, 47.3227) * np.sqrt(0.31)
		else:
			run.data['OmegaR'] = phaseO_OmegaR(run.data.VVA, run.data.freq) * np.sqrt(0.31)
		run.data['OmegaR2'] = (run.data['OmegaR'])**2
		xname = 'OmegaR2'
		
		run.group_by_mean(xname)
		
		
		### PLOTTING ###
		# transfer
		sty = styles[i]
		color = colors[i]
		label = f'det = {detuning} kHz'
		x = run.avg_data[xname]
		y = run.avg_data['transfer']
		yerr = run.avg_data['em_transfer']
		
		xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
		
		ax = axs[0]
		ax.errorbar(x, y, yerr=yerr, **sty)
		
		# fit to saturation curve
		p0 = [0.5, 10000]
		popt, pcov = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
		perr = np.sqrt(np.diag(pcov))
		ax.plot(xs, Saturation(xs, *popt), '-', label=label, color=color)
		ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), '--', color=color)
		
		
		print(r"transfer: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt[0], 
												  perr[0], popt[1], perr[1]))
		# store popt in list
		popts.append(popt)
		perrs.append(perr)
		
		# loss
		y = run.avg_data['loss']
		yerr = run.avg_data['em_loss']
		
		ax = axs[1]
		ax.errorbar(x, y, yerr=yerr, **sty)
		
		# fit to saturation curve
		p0 = [0.5, 10000]
		popt_l, pcov_l = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
		perr_l = np.sqrt(np.diag(pcov))
		ax.plot(xs, Saturation(xs, *popt_l), '-', label=label, color=color)
		ax.plot(xs, Linear(xs, popt_l[0]/popt_l[1], 0), '--', color=color)
		
		
		print(r"loss: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt_l[0], 
												  perr_l[0], popt_l[1], perr_l[1]))
		
		popts_l.append(popt_l)
		perrs_l.append(perr_l)
		
		
		# plot calibration curves
		detuning = detunings[i]
		popt = popts[i]
		popt_l = popts_l[i]
		label = f'det = {detuning} kHz'
		
		# transfer
		ax = axs[2]
		xs = np.linspace(0, popt[1], 1000)  # linspace of rf powers
		
		Gammas_Sat = Saturation(xs, *popt)
		Gammas_Lin = xs*popt[0]/popt[1]
		e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], popt[0], popt[1], perr[0], perr[1], pcov[0,1])
		
		ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label)
		if fill_between == True:
			ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
				Gammas_Lin+e_Gammas_Lin, alpha=0.5)
		ax.legend()
		
		# loss
		ax=axs[3]
		xs = np.linspace(0, popt_l[1], 1000)  # linspace of rf powers
		
		Gammas_Sat = Saturation(xs, *popt_l)
		Gammas_Lin = xs*popt_l[0]/popt_l[1]
		e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], popt[0], popt[1], perr[0], perr[1], pcov[0,1])
		
		if fill_between == True:
			ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
				Gammas_Lin+e_Gammas_Lin, alpha=0.5)
		
		ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label)
		
		# append to results
		
		keys = ['file', 'detuning', 'pulse_time', 'ToTF', 'EF', 'df', 'popt', 'pcov', 'popt_l', 'pcov_l']
		vals = [file, detuning, pulse_time, ToTF, EF, run.avg_data, popt, pcov, popt_l, pcov_l]
		
		for key, val in zip(keys, vals):
			results[key] = val
			
		results_list.append(results)
	
	# add y = x lines to calibration plots
	axs[2].plot(Gammas_Sat, Gammas_Sat, '-', color='dimgrey', zorder=1)
	axs[3].plot(Gammas_Sat, Gammas_Sat, '-', color='dimgrey', zorder=1)
	
	for ax in axs:
		ax.legend()
	fig.tight_layout()
	plt.show()
	
	
	if save == True:
		with open(pkl_file, "wb") as output_file:
			pkl.dump(results_list, output_file)
		

# if script imported
else:
	# fit results
	popts = [np.array([  0.46657157, 139.89433055]),
		   np.array([  0.49253269, 363.06301381]),
		    np.array([4.76550893e-01, 8.33509148e+02]),
			 np.array([4.31394956e-01, 1.88559341e+03])]
	popts_l = [np.array([  0.57999841, 141.11129911]),
			 np.array([  0.561563  , 300.17105107]),
			 np.array([5.06720326e-01, 9.36351736e+02]),
			 np.array([4.79542621e-01, 1.27518484e+03])]
	detunings = [
				25,
				50,
				100,
				150,
				]
	
	popt_l = np.array([0.5067, 936.3517])
	
	HFT_sat_cals = {}
	HFT_loss_sat_cals = {}
	
	# generate calibation correction functions
	for i in range(len(detunings)):
		detuning = detunings[i]
		popt = popts[i]
		popt_l = popts_l[i]
		xs = np.linspace(0, popt[1], 1000)  # linspace of rf powers
		
		Gammas_Sat = Saturation(xs, *popt)
		Gammas_Lin = xs*popt[0]/popt[1]
		def HFT_sat_correction(Gamma):
			'''Interpolates fit saturation curve of transferred fraction in HFT
			for 100kHz detuning. Returns the linear term, i.e. the unsaturated.
			transfer.'''
			return np.interp(Gamma, Gammas_Sat, Gammas_Lin)
		
		HFT_sat_cals[detuning] = HFT_sat_correction
		
		Gammas_Sat_l = Saturation(xs, *popt_l)
		Gammas_Lin_l = xs*popt_l[0]/popt_l[1]
		def HFT_loss_sat_correction(Gamma):
			'''Interpolates fit saturation curve of loss fraction in HFT
			for 100kHz detuning. Returns the linear term, i.e. the unsaturated.
			transfer.'''
			return np.interp(Gamma, Gammas_Sat_l, Gammas_Lin_l)
		
		HFT_loss_sat_cals[detuning] = HFT_loss_sat_correction
		
		


	