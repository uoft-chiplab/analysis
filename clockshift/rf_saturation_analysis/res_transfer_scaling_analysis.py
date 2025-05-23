# -*- coding: utf-8 -*-
"""
2024 Nov 12
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
from scipy.optimize import curve_fit
from library import paper_settings, styles, colors, MW_styles
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
save = True

pkl_file = os.path.join(data_path, "res_saturation_curves.pkl")

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

def satratio(x, x0):
	return x/x0*1/(1-np.exp(-x/x0))

def error_prop_division(x, dx, y, dy):
	return np.abs((x/y)*((dx/x)**2 + (dy/y)**2)**0.5)

def quotient_propagation(f, A, B, sA, sB, sAB):
	return f* (sA**2/A**2 + sB**2/B**2 - 2*sAB/A/B)**(1/2)


if __name__ == '__main__':
	
	results_list = []
	
	detunings = [
# 				-20,
# 				-10,
				-5,
# 				0,
# 				5,
# 				10,
# 				20,
# 				0,
				0,
				]
	files = [
# 		  "2024-12-05_L_e_detuning=-20.dat",
		  "2024-12-05_L_e_detuning=-10.dat",
		  "2024-12-05_L_e_detuning=-5.dat",
 		   # "2024-12-05_K_e.dat",
		  "2024-12-05_L_e_detuning=5.dat",
		  "2024-12-05_L_e_detuning=10.dat",
		  "2024-12-05_L_e_detuning=20.dat",
# 		  "2025-02-13_K_e.dat",
		  "2025-02-13_P_e.dat",
		  ]
	
#  	files = [
#  			# "2024-11-29_B_e_detuning=-5.dat",
# 			 "2024-12-05_K_e.dat",
#  			#"2024-11-28_P_e_detuning=0.dat",
#  			  ]
	files = [
 			"2024-11-29_B_e_detuning=-5.dat",
			 "2024-11-29_B_e_detuning=0.dat",
  			 # "2024-11-28_P_e_detuning=0.dat",
 			  # "2024-11-28_P_e_detuning=-5.dat"
 			  ]
	
	pulse_time = 0.2  # ms
	cutoffs = np.ones(len(files)) * 100
	
	popts = []
	perrs = []
	popts_l = []
	perrs_l = []
	
	
	### plot settings
# 	plt.rcParams.update(plt_settings) # from library.py
# 	plt.rcParams.update({"figure.figsize": [12,8],
# 						 "font.size": 14,
# 						 "lines.markeredgewidth": 2,
# 						 "errorbar.capsize": 0})
	
	plt.rcdefaults()
	plt.rcParams.update(paper_settings) # from library.py
	font_size = paper_settings['legend.fontsize']
	fig_width = 3.4 # One-column PRL figure size in inches
	subplotlabel_font = 10
	
	#### PLOTTING #####
	# initialize plots
	fig, axes = plt.subplots(3, 2, figsize=(6.4,6))
	axs = axes.flatten()
	
	fig.suptitle("2ms Blackman resonant transfer saturation")
	
	### ANALYSIS ###
	
	for i in range(len(detunings)):
		results = {}
		
		file = files[i]
		cutoff = cutoffs[i]
		detuning = detunings[i]
		
		# paths
		proj_path = os.path.dirname(os.path.realpath(__file__))
		data_path = os.path.join(proj_path, 'saturation_data')
		root = os.path.dirname(proj_path)
		
		print("Analyzing", file)
		run = Data(file, path=data_path)
		
		if file[:4] == '2024':
			EF = 18.7  # kHz, from J_UShots
			ToTF = 0.647  # from J_UShots
			ff = 0.98
			VpptoOmegaR47 = 17.05/0.703 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
			VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
		elif file == "2025-02-13_K_e.dat":
			EF = 13.9 # H_UShots
			ToTF = 0.306
			ff = 0.83
			VpptoOmegaR47 = 12.01/0.452 # kHz/Vpp - 2025-02-12 calibration 
			VpptoOmegaR43 = 14.44/0.656 *VpptoOmegaR47/(17.05/0.728) # fudged 43MHz calibration
		elif file == "2025-02-13_P_e.dat":
			EF = 18.2 # M_UShots
			ToTF = 0.595
			ff = 0.83
			VpptoOmegaR47 = 12.01/0.452 # kHz/Vpp - 2025-02-12 calibration 
			VpptoOmegaR43 = 14.44/0.656 *VpptoOmegaR47/(17.05/0.728) # fudged 43MHz calibration
		
		# calculate frequency
		run.data['freq'] = 47.2227 + detuning/1000
		
		### Omega Rabi calibrations
		# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
		phaseO_OmegaR = lambda VVA, freq: VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)	
	
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
		run.data['anomalous_loss'] = (bg_c9 + bg_c5 - run.data.c9 - run.data.c5)/bg_c9
		
		
		run.data['OmegaR'] = phaseO_OmegaR(run.data.VVA, run.data.freq) * np.sqrt(0.31)
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
		
		xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
		
		ax = axs[0]
		ax.set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel=r'Transfer $\alpha$',
			   ylim=[-0.05, 0.50],
			   xlim=[-0.5, 10])
		ax.errorbar(x, y, yerr=yerr, **sty)
		ax.errorbar(cutoff_df[xname], cutoff_df['transfer'], 
			  yerr=cutoff_df['em_transfer'], **sty_cutoff)
		
		# fit to saturation curve
		p0 = [np.max(run.data.transfer), cutoff/np.e]
		popt, pcov = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
		perr = np.sqrt(np.diag(pcov))
		x0 = popt[1]
		slope = popt[0]/popt[1]
		e_slope = quotient_propagation(slope, popt[0], popt[1], perr[0], 
								 perr[1], pcov[0,1])
		ax.plot(xs, Saturation(xs, *popt), '-', label=label, color=color)
		ax.plot(xs, Linear(xs, slope, 0), '--', color=color)
		
		# rescale x-axis
		ax = axs[0+2]
		ax.set(xlabel=r'rf power $P/P_0$', ylabel=r'Transfer $\alpha$',
			   ylim=[-0.05, 0.65],
			   xlim=[-0.5, 10])
		ax.errorbar(x/x0, y, yerr=yerr, **sty)
		ax.errorbar(cutoff_df[xname]/x0, cutoff_df['transfer'], 
			  yerr=cutoff_df['em_transfer'], **sty_cutoff)
		
		# fit to saturation curve
		ax.plot(xs/x0, Saturation(xs, *popt), '-', label=label, color=color)
		ax.plot(xs/x0, Linear(xs, slope, 0), '--', color=color)
	
		
		print(r"transfer: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt[0], 
												  perr[0], popt[1], perr[1]))
		# store popt in list
		popts.append(popt)
		perrs.append(perr)
		
		# loss
		y = run.avg_data['loss']
		yerr = run.avg_data['em_loss']
			
		ax = axs[1]
		ax.set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Loss',
			   ylim=[-0.05, 0.55], xlim=[0,5])
		ax.errorbar(x, y, yerr=yerr, **sty)
		ax.errorbar(cutoff_df[xname], cutoff_df['loss'], 
			  yerr=cutoff_df['em_loss'], **sty_cutoff)
		
		# fit to saturation curve
		p0 = [0.6, 5]
		popt_l, pcov_l = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
		perr_l = np.sqrt(np.diag(pcov))
		x0_l = popt_l[1]
		slope_l = popt_l[0]/popt_l[1]
		e_slope_l = quotient_propagation(slope_l, popt_l[0], popt_l[1], perr_l[0], 
								 perr_l[1], pcov_l[0,1])
		if e_slope_l != e_slope_l:
			print("changing error prop method becase got nan")
			e_slope_l = error_prop_division(popt[0], perr[0], popt[1], perr[1])
			
		ax.plot(xs, Saturation(xs, *popt_l), '-', label=label, color=color)
		ax.plot(xs, Linear(xs, slope_l, 0), '--', color=color)
		
		# rescale x-axis
		ax = axs[1+2]
		ax.set(xlabel=r'rf power $P/P_0$', ylabel=r'Transfer $\alpha$',
			   ylim=[-0.05, 0.75],
			   xlim=[-0.5, 10])
		ax.errorbar(x/x0_l, y, yerr=yerr, **sty)
		ax.errorbar(cutoff_df[xname]/x0_l, cutoff_df['transfer'], 
			  yerr=cutoff_df['em_transfer'], **sty_cutoff)
		
		# fit to saturation curve
		ax.plot(xs/x0_l, Saturation(xs, *popt_l), '-', label=label, color=color)
		ax.plot(xs/x0_l, Linear(xs, slope_l, 0), '--', color=color)
		
		
		print(r"loss: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt_l[0], 
												  perr_l[0], popt_l[1], perr_l[1]))
		
		popts_l.append(popt_l)
		perrs_l.append(perr_l)
	

		# plot atom number vs. transfer
		ax = axs[4]
		xlabel = 'transfer'
		ax.set(xlabel=xlabel, ylabel='N_b + N_c')
		
		x = run.avg_data['transfer']
		y = run.avg_data['N']
		yerr = run.avg_data['em_N']
		
		ax.hlines(bg_c9, min(x), max(x), linestyle='--', color=color)
		ax.errorbar(x, y, yerr=yerr, label=label, **sty)
		
		# plot atom number vs. loss
		ax = axs[5]
		xlabel = 'loss'
		ax.set(xlabel=xlabel, ylabel='Anomalous loss')
		
		x = run.avg_data['loss']
		y = run.avg_data['anomalous_loss']
		yerr = run.avg_data['em_anomalous_loss']
		
		ax.errorbar(x, y, yerr=yerr, label=label, **sty)

		# append to results
		
		keys = ['file', 'detuning', 'pulse_time', 'ToTF', 'EF', 'df', 
			  'popt', 'pcov', 'popt_l', 'pcov_l', 
			  'slope', 'e_slope', 'slope_l', 'e_slope_l']
		vals = [file, detuning, pulse_time, ToTF, EF, run.avg_data, 
			  popt, pcov, popt_l, pcov_l,
			  slope, e_slope, slope_l, e_slope_l]
		
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
		
		


	