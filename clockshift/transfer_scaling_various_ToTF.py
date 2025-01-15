# -*- coding: utf-8 -*-
"""
2024 Nov 12
@author: Chip Lab

"""
from data_class import Data
from scipy.optimize import curve_fit
from library import plt_settings, styles, colors
import numpy as np
import pandas as pd
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

pkl_file = os.path.join(data_path, "various_ToTF_saturation_curves.pkl")

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c

# def Saturation(x, A, x0, B, w):
# 	return A*(1-B*np.exp(-x/np.abs(x0))*np.cos(w*x)**2)

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/np.abs(x0)))

def satratio(x, x0):
	return x/x0*1/(1-np.exp(-x/x0))

def quotient_propagation(f, A, B, sA, sB, sAB):
	return f* (sA**2/A**2 + sB**2/B**2 - 2*sAB/A/B)**(1/2)


if __name__ == '__main__':
	
	results_list = []
	
	files = [
 			"2024-11-29_G_e.dat",
			  ]
	
	ODT1_to_ToTF_map_dict = {0.1: 0.478,
				 0.07: 0.384}
	
	def ODT1_to_ToTF_map(ODT1):
		for key, val in ODT1_to_ToTF_map_dict.items():
			if ODT1 == key:
				return val
	
	fudge_factor = 0.98
	
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
	plt.rcParams.update({"font.size": 14,
						 "lines.markeredgewidth": 2,
						 "errorbar.capsize": 0})
	
	fig.suptitle("200us Blackman resonant transfer saturation")
	
	### ANALYSIS ###
	xname = 'VVA'
	plot_name = "Omega Rabi Squared (1/ms^2)"
	
	i = 0
	for j in range(len(files)):
		file = files[j]
		ff = fudge_factor
		
		print("Analyzing", file)
		run = Data(file, path=data_path)
		
		# correct c9
		run.data['c9'] = ff * run.data['c9']
		
		run.data['pulse_time'] = pulse_time
		
		# add ToTF to data
		if file == files[0]:
			run.data['ToTF'] = run.data.ODT1.apply(ODT1_to_ToTF_map)
# 		else:  # else it was in a different run and had roughly...
# 			run.data['ToTF'] = 0.616
			
		# add detuning to data
# 		if file == files[-1]:
# 			run.data['detuning'] = 100
# 			run.data['freq'] = 47.3227
		
		for ToTF in run.data.ToTF.unique():
			
			# get df of just that ToTF
			drop_df = run.data.loc[(run.data.ToTF != ToTF)]
			ToTF_df = run.data.drop(drop_df.index)
			
			# 0 VVA as bg point, works for all detunings
			bg_df = ToTF_df.loc[(ToTF_df.VVA == 0)]
			ToTF_df = ToTF_df.drop(bg_df.index)
			
			# calculate bg stuffs
			bg_c5 = bg_df.c5.mean()
			e_bg_c5 = bg_df.c5.sem()
			bg_c9 = bg_df.c9.mean()
			e_bg_c9 = bg_df.c9.sem()
			
			for detuning in ToTF_df.detuning.unique():
				results = {}
				
				print(f"Analyzing for detuning = {detuning} and ToTF = {ToTF}")
				
				xlims = [0, 10]
				
				if detuning == 0:
					ls = '--'
					cutoff = 12
				else:
					ls = '-'
					cutoff = 10000
				
				# select detuning
				drop_df = ToTF_df.loc[(ToTF_df.detuning != detuning)]
				df = ToTF_df.drop(drop_df.index)
		
				### Omega Rabi calibrations
				# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
				VpptoOmegaR47 = 17.05/0.703 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
				VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
				phaseO_OmegaR = lambda VVA, freq: VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)
				
				# do some calculations
				df['N'] = df.c5 - bg_c5 + df.c9
				df['transfer'] = (df.c5 - bg_c5)/(df.N)
				df['loss'] = (bg_c9 - df.c9)/bg_c9
				
				df['OmegaR'] = phaseO_OmegaR(df.VVA, df.freq) * np.sqrt(0.31)
				df['OmegaR2'] = (df['OmegaR'])**2
				xname = 'OmegaR2'
				
				# OmegaR2 greater than some cutoff determined from first pass fit
				cutoff_df = df.loc[(df.OmegaR2 > cutoff)]
				df = df.drop(cutoff_df.index)
				
				# group by mean, creating avg_df
				mean = df.groupby([xname]).mean().reset_index()
				sem = df.groupby([xname]).sem().reset_index().add_prefix("em_")
				std = df.groupby([xname]).std().reset_index().add_prefix("e_")
				avg_df = pd.concat([mean, std, sem], axis=1)
		
		
		
				### PLOTTING ###
				# transfer
				sty = styles[i]
				color = colors[i]
				label = f'det = {detuning} kHz, ToTF = {ToTF}'
				x = avg_df[xname]
				y = avg_df['transfer']
				yerr = avg_df['em_transfer']
				
				xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
				
				ax = axs[0]
				ax.set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Transfer',
					   ylim=[-0.05, 0.65], 
# 					   xlim=xlims
					   )
				ax.errorbar(x, y, yerr=yerr, **sty)
				
				# fit to saturation curve
				p0 = [0.6, 5]
				popt, pcov = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
				perr = np.sqrt(np.diag(pcov))
		# 		label_lin = r'linear term $\Gamma(\Omega_R^2) = \Gamma_{sat} \Omega_R^2/\Omega_e^2$'
				ax.plot(xs, Saturation(xs, *popt), '-', label=label, color=color)
		# 		ax.plot(xs, Saturation(xs, *p0), ':', color=color)
				ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), '--', color=color)
				
				
				print(r"transfer: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt[0], 
														  perr[0], popt[1], perr[1]))
				# store popt in list
				popts.append(popt)
				perrs.append(perr)
				
				# compare measured to linear 
				ax = axs[2]
				ax.set(xlabel='Measured transfer', xlim=[-0.02, 0.3], ylim=[-0.025, 0.4],
				   ylabel='Calibrated linear transfer')
				
				xs = np.linspace(0, popt[1], 1000)  # linspace of rf powers
				Gammas_Sat = Saturation(xs, *popt)
				Gammas_Lin = xs*popt[0]/popt[1]
				e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], popt[0], popt[1], perr[0], perr[1], pcov[0,1])
				
				if fill_between == True:
					ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
						Gammas_Lin+e_Gammas_Lin, alpha=0.5)
				ax.plot(Gammas_Sat, Gammas_Lin, ls, label=label, color=color)
				
				# loss
				y = avg_df['loss']
				yerr = avg_df['em_loss']
				xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
				
				ax = axs[1]
				ax.set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Loss',
					   ylim=[-0.05, 0.65], 
# 					   xlim=xlims
					   )
				ax.errorbar(x, y, yerr=yerr, **sty)
				
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
				
				# compare measured to linear 
				ax = axs[3]
				ax.set(xlabel='Measured loss', xlim=[-0.02, 0.3], ylim=[-0.025, 0.4],
				   ylabel='Calibrated linear loss')
				
				xs = np.linspace(0, popt_l[1], 1000)  # linspace of rf powers
				Gammas_Sat = Saturation(xs, *popt_l)
				Gammas_Lin = xs*popt_l[0]/popt_l[1]
				ax.plot(Gammas_Sat, Gammas_Lin, ls, label=label)
				
				# append to results
				keys = ['file', 'detuning', 'pulse_time', 'ToTF', 'df', 'popt', 'pcov', 'popt_l', 'pcov_l']
				vals = [file, detuning, pulse_time, ToTF, avg_df, popt, pcov, popt_l, pcov_l]
				
				for key, val in zip(keys, vals):
					results[key] = val
					
				results_list.append(results)
				
				i += 1
				
	axs[2].plot(Gammas_Sat, Gammas_Sat, '-', color='dimgrey', zorder=1)
	axs[3].plot(Gammas_Sat, Gammas_Sat, '-', color='dimgrey', zorder=1)
	
	for ax in axs:
		ax.legend()
	fig.tight_layout()
	plt.show()
	
	
	if save == True:
		with open(pkl_file, "wb") as output_file:
			pkl.dump(results_list, output_file)
	