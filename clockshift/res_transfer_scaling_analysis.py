# -*- coding: utf-8 -*-
"""
2024 Nov 12
@author: Chip Lab

"""
from data_class import Data
from scipy.optimize import curve_fit
from library import plt_settings, styles, colors, MW_styles
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import pickle as pkl

from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(proj_path, 'saturation_data')
root = os.path.dirname(proj_path)

# plot error bands for saturation curves
fill_between = True
save = False

pkl_file = os.path.join(data_path, "near-res_saturation_curves.pkl")

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/np.abs(x0)))

def satratio(x, x0):
	return x/x0*1/(1-np.exp(-x/x0))

def error_prop_division(x, dx, y, dy):
	return np.abs((x/y)*((dx/x)**2 + (dy/y)**2)**0.5)


if __name__ == '__main__':
	
	results_list = []
	
	detunings = [-20,
				-10,
				-5,
				0,
				5,
				10,
				20
				]
	files = ["2024-12-05_L_e_detuning=-20.dat",
		  "2024-12-05_L_e_detuning=-10.dat",
		  "2024-12-05_L_e_detuning=-5.dat",
		   "2024-12-05_K_e.dat",
		  "2024-12-05_L_e_detuning=5.dat",
		  "2024-12-05_L_e_detuning=10.dat",
		  "2024-12-05_L_e_detuning=20.dat"
		  ]
	
# 	files = [
#  			# "2024-11-29_B_e_detuning=-5.dat",
# 			 "2024-12-05_K_e.dat",
#  			#"2024-11-28_P_e_detuning=0.dat",
# 			  ]
# 	files = [
#  			#"2024-11-29_B_e_detuning=-5.dat",
# 			 #"2024-11-29_B_e_detuning=0.dat",
#  			 "2024-11-28_P_e_detuning=0.dat",
# 			  "2024-11-28_P_e_detuning=-5.dat"
# 			  ]
	
	ToTF = 0.647  # from J_UShots
	pulse_time = 2  # ms
	EF = 18.7  # kHz, from J_UShots
	
	fudge_factors = np.ones(len(files))*0.98
	
	cutoffs = np.ones(len(files)) * 100
	
	popts = []
	perrs = []
	popts_l = []
	perrs_l = []
	
	#### PLOTTING #####
	# initialize plots
	fig, axes = plt.subplots(2, 2, figsize=(12,7.5), sharex=True)
	axs = axes.flatten()
	# axs[0].set(xscale="log")
	
	### plot settings
	plt.rcParams.update(plt_settings) # from library.py
	plt.rcParams.update({"figure.figsize": [12,8],
						 "font.size": 14,
						 "lines.markeredgewidth": 2,
						 "errorbar.capsize": 0})
	
	fig.suptitle("2ms Blackman resonant transfer saturation")
	
	### ANALYSIS ###
	xname = 'VVA'
	plot_name = "Omega Rabi Squared (1/ms^2)"
	
	for i in range(len(detunings)):
		results = {}
		
		file = files[i]
		detuning = detunings[i]
		ff = fudge_factors[i]
		cutoff = cutoffs[i]
		
		# paths
		proj_path = os.path.dirname(os.path.realpath(__file__))
		data_path = os.path.join(proj_path, 'saturation_data')
		root = os.path.dirname(proj_path)
		
		### Omega Rabi calibrations
		# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
		VpptoOmegaR47 = 17.05/0.703 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
		VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
		phaseO_OmegaR = lambda VVA, freq: VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)
		
		print("Analyzing", file)
		run = Data(file, path=data_path)
		run.data['c9'] = ff * run.data['c9']
		run.data['ToTF'] = ToTF
		run.data['EF'] = EF
		run.data['pulse_time'] = pulse_time
		
		# the rest used 0 VVA as bg point
		bg_df = run.data.loc[(run.data.VVA == 0)]
		run.data = run.data.drop(bg_df.index)
		
		
		bg_c5 = bg_df.c5.mean()
		e_bg_c5 = bg_df.c5.sem()
		bg_c9 = bg_df.c9.mean()
		e_bg_c9 = bg_df.c9.sem()
		
		run.data['N'] = run.data.c5 - bg_c5 + run.data.c9
		run.data['transfer'] = (run.data.c5 - bg_c5)/(run.data.N)
		run.data['loss'] = (bg_c9 - run.data.c9)/bg_c9
		
		
		run.data['OmegaR'] = phaseO_OmegaR(run.data.VVA, run.data.freq) * np.sqrt(0.31)
		# KX just playing around
		#run.data['OmegaR'] = np.sqrt(phaseO_OmegaR(run.data.VVA, run.data.freq)**2 - detuning**2) * np.sqrt(0.31)
		run.data['OmegaR2'] = (run.data['OmegaR'])**2
		xname = 'OmegaR2'
		
		run.group_by_mean(xname)
		
		# OmegaR2 greater than some cutoff determined from first pass fit
		cutoff_df = run.avg_data.loc[(run.avg_data.OmegaR2 > cutoff)]
		run.avg_data = run.avg_data.drop(cutoff_df.index)
		
		
		### PLOTTING ###
		# transfer
		sty = styles[i]
		sty_cutoff = copy.deepcopy(sty)
		sty_cutoff['mfc'] = 'w'
		color = colors[i]
		label = f'det = {detuning} kHz'
		x = run.avg_data[xname]
		y = run.avg_data['transfer']
		yerr = run.avg_data['em_transfer']
		
# 		if i != 0:
#  			j = 3
#  			k = 15
#  			x = x[j:k]
#  			y = y[j:k]
#  			yerr = yerr[j:k]
		
		xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
		
		ax = axs[0]
		ax.set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Transfer',
			   ylim=[-0.05, 0.70],
			   xlim=[0.005, 0.5])
		ax.errorbar(x, y, yerr=yerr, **sty)
		ax.errorbar(cutoff_df[xname], cutoff_df['transfer'], 
			  yerr=cutoff_df['em_transfer'], **sty_cutoff)
		
		# fit to saturation curve
		p0 = [0.4, 0.5]
# 		ax.plot(xs, Saturation(xs,*p0), '--', label=label, color='mediumvioletred')
		popt, pcov = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
		perr = np.sqrt(np.diag(pcov))
# 		label_lin = r'linear term $\Gamma(\Omega_R^2) = \Gamma_{sat} \Omega_R^2/\Omega_e^2$'
		ax.plot(xs, Saturation(xs, *popt), '-', label=label, color=color)
		# 		ax.plot(xs, Saturation(xs, *p0), ':', color=color)
		ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), '--', color=color)
		
		# plot residuals 
		ax = axs[2]
		ax.set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Transfer residuals', ylim=(-0.03, 0.06))
		ax.axhline(0, linestyle=":", color="lightgrey")
		ax.errorbar(x, y - Saturation(x, *popt), yerr=yerr, **sty, label=label)
		# ax.plot(x, y-Saturation(x, *popt), color=color, linestyle='-')
		# ax.errorbar(x, y-Linear(x, popt[0]/popt[1],0), color=color)
		
		print(r"transfer: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt[0], 
												  perr[0], popt[1], perr[1]))
		# store popt in list
		popts.append(popt)
		perrs.append(perr)
		
		# loss
		y = run.avg_data['loss']
		yerr = run.avg_data['em_loss']

# 		if i != 0:
# 			y = y[j:k]
# 			yerr = yerr[j:k]
			
		ax = axs[1]
		ax.set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Loss',
			   ylim=[-0.05, 0.75])
		ax.errorbar(x, y, yerr=yerr, **sty)
		ax.errorbar(cutoff_df[xname], cutoff_df['loss'], 
			  yerr=cutoff_df['em_loss'], **sty_cutoff)
		
		# fit to saturation curve
		p0 = [0.6, 5]
		popt_l, pcov_l = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
		perr_l = np.sqrt(np.diag(pcov))
# 		label_lin = r'loss linear term $\Gamma(\Omega_R^2) = \Gamma_{sat} \Omega_R^2/\Omega_e^2$'
		ax.plot(xs, Saturation(xs, *popt_l), '-', label=label, color=color)
# 		ax.plot(xs, Saturation(xs, *p0), ':', color=color)
		ax.plot(xs, Linear(xs, popt_l[0]/popt_l[1], 0), '--', color=color)
		
		
		print(r"loss: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt_l[0], 
												  perr_l[0], popt_l[1], perr_l[1]))
		
		popts_l.append(popt_l)
		perrs_l.append(perr_l)
		
		# plot residuals 
		ax = axs[3]
		ax.set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Loss residuals')
		ax.axhline(0, linestyle=":", color="lightgrey")
		ax.errorbar(x, y - Saturation(x, *popt_l), yerr=yerr, **sty, label=label)
		# ax.errorbar(x, y - Linear(x, popt_l[0]/popt_l[1],0), color=color)

		# append to results
		
		keys = ['file', 'detuning', 'pulse_time', 'ToTF', 'EF', 'df', 'popt', 'pcov', 'popt_l', 'pcov_l']
		vals = [file, detuning, pulse_time, ToTF, EF, run.avg_data, popt, pcov, popt_l, pcov_l]
		
		for key, val in zip(keys, vals):
			results[key] = val
			
		results_list.append(results)
	
	for ax in axs:
		ax.legend()
	fig.tight_layout()
	plt.show()
	
	# plot saturated measure vs. linear calibration
	fig, axs = plt.subplots(1, 2, figsize=(12,5))
	axs[0].set(xlabel='Measured transfer', ylabel='Calibrated linear transfer')
	axs[1].set(xlabel='Measured loss', ylabel='Calibrated linear loss')
	fig.suptitle("Calibrated saturation -> linear transfer functions")
	
	for ax in axs:
		pts = np.linspace(0, 0.5, 100)
		ax.plot(pts, pts, '--', color='dimgrey')

	for i in range(len(detunings)):
		detuning = detunings[i]
		popt = popts[i]
		popt_l = popts_l[i]
		label = f'det = {detuning} kHz'
		
		# transfer
		ax = axs[0]
		xs = np.linspace(0, popt[1], 1000)  # linspace of rf powers
		
		Gammas_Sat = Saturation(xs, *popt)
		Gammas_Lin = xs*popt[0]/popt[1]
		
		ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label)
		ax.legend()
		
		# loss
		ax = axs[1]
		xs = np.linspace(0, popt_l[1], 1000)  # linspace of rf powers
		
		Gammas_Sat = Saturation(xs, *popt_l)
		Gammas_Lin = xs*popt_l[0]/popt_l[1]
		
		ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label)
		ax.legend()
	
	fig.tight_layout()
	
	### plot A and B fit parameters and compare
	fig, axs = plt.subplots(2,3, figsize=(16,7.5))
	axs = axs.flatten()
	
	axs[0].set(xlabel="Detuning (kHz)", ylabel="A")
	axs[1].set(xlabel="Detuning (kHz)", ylabel="A/x0")
	axs[2].set(xlabel="Detuning (kHz)", ylabel="A_loss- A_trans")
	axs[3].set(xlabel="Detuning (kHz)", ylabel="A_loss/A_trans")
	axs[4].set(xlabel="Detuning (kHz)", ylabel="x0")
	axs[5].set(xlabel="Detuning (kHz)", ylabel="x0 [transfer/loss]")
	
	# styles
	style = MW_styles[0]
	style2 = MW_styles[1]
	
	# transfer
	popts = np.array(popts)
	perrs = np.array(perrs)
	
	# loss
	popts_l = np.array(popts_l)
	perrs_l = np.array(perrs_l)
	j = 0
	axs[j].errorbar(detunings, popts[:,0], yerr = perrs[:,0], label="transfer", **style)
	axs[j].errorbar(detunings, popts_l[:,0], yerr = perrs_l[:,0], label="loss", **style2)
	
	j += 1
	axs[j].errorbar(detunings, popts[:,0]/popts[:,1], 
				 yerr = error_prop_division(popts[:,0], perrs[:,0], popts[:,1], perrs[:,1]),
				 label="transfer", **style)
	axs[j].errorbar(detunings, popts_l[:,0]/popts_l[:,1], 
				 yerr = error_prop_division(popts_l[:,0], perrs_l[:,0], popts_l[:,1], perrs_l[:,1]),
				 label="loss", **style2)
	
	# plot without -20 
	i = 0
	detunings = detunings[i:]
	popts = popts[i:]
	popts_l = popts_l[i:]
	perrs = perrs[i:]
	perrs_l = perrs_l[i:]
	
	j += 1
	axs[j].errorbar(detunings, popts_l[:,0]-popts[:,0], 
				 yerr=np.sqrt(perrs[:,0]**2+perrs_l[:,0]**2), **style)
	
	j += 1
	yerrs = popts_l[:,0]/popts[:,0]*np.sqrt((perrs_l[:,0]/popts[:,0])**2 + (perrs[:,0]/popts[:,0])**2)
	axs[j].errorbar(detunings, popts_l[:,0]/popts[:,0], yerr=yerrs, **style)
	
	j += 1
	axs[j].errorbar(detunings[:], popts[:,1], yerr = perrs[:,1], label="transfer", **style)
	axs[j].errorbar(detunings[:], popts_l[:,1], yerr = perrs_l[:,1], label="loss", **style2)
	
	j += 1
	axs[j].errorbar(detunings[:], popts[:,1]/popts_l[:,1], 
				 yerr = error_prop_division(popts[:,1], perrs[:,1], popts_l[:,1], perrs_l[:,1]),
				 **style)
	axs[j].set_ylim(0,2)
	axs[j].axhline(1, linestyle=":", color="lightgrey")
	
	
	for ax_i in [0, 1, 4]:
		axs[ax_i].legend()
		
	fig.tight_layout()
	
	if save == True:
		with open(pkl_file, "wb") as output_file:
			pkl.dump(results_list, output_file)

# if script imported
else:
	# fit results
	popts = [np.array([0.22625954, 5.90819859]), 
		  np.array([ 1.09827712, 12.29783072])]
	popts_l = [np.array([0.24822713, 2.64703707]), 
			np.array([0.96862428, 8.08612931])]
	detunings = [
				-5, 0
				]
	
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
		
		


	