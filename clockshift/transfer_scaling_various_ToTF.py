# -*- coding: utf-8 -*-
"""
2024 Nov 12
@author: Chip Lab

Modified in 2025 to include new lower temperature data.

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
	return A*(1-np.exp(-x/x0))

def satratio(x, x0):
	return x/x0*1/(1-np.exp(-x/x0))

def quotient_propagation(f, A, B, sA, sB, sAB):
	return f* (sA**2/A**2 + sB**2/B**2 - 2*sAB/A/B)**(1/2)


if __name__ == '__main__':
	
	results_list = []
	
	files = [
 			# "2024-11-29_G_e.dat",
			  "2025-02-13_I_e.dat", # 0.3 ToTF
			  "2025-02-13_O_e.dat", # 0.6 ToTF
			  "2025-03-06_E_e.dat", # 0.95 ToTF
			  ]
	
	ODT1_to_ToTF_map_dict = {
				0.4: 0.95, # this is a lie
				0.3: 0.595,
				0.1: 0.478,
				 0.07: 0.384,
				 0.06: 0.306}
	
	def ODT1_to_ToTF_map(ODT1):
		for key, val in ODT1_to_ToTF_map_dict.items():
			if ODT1 == key:
				return val
	
	pulse_time = 0.2  # ms
	
	popts = []
	perrs = []
	popts_l = []
	perrs_l = []
	
	
	### plot settings
	plt.rcParams.update(plt_settings) # from library.py
	plt.rcParams.update({"font.size": 14,
						 "lines.markeredgewidth": 2,
						 "errorbar.capsize": 0})
	
	#### PLOTTING #####
	# initialize plots
	fig, axes = plt.subplots(2, 2, figsize=(12,8))
	axs = axes.flatten()
	
	fig.suptitle("200us Blackman resonant transfer saturation")
	
	### ANALYSIS ###
	xname = 'VVA'
	plot_name = "Omega Rabi Squared (1/ms^2)"
	
	i = 0
	for j in range(len(files)):
		file = files[j]
		
		print("Analyzing", file)
		run = Data(file, path=data_path)
		
		if file[:4] == '2024':
			fudge_factor = 0.98
			VpptoOmegaR47 = 17.05/0.728 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
			VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
		elif file[:4] == "2025":
			fudge_factor = 0.83
			VpptoOmegaR47 = 12.01/0.452 # kHz/Vpp - 2025-02-12 calibration 
			VpptoOmegaR43 = 14.44/0.656 *VpptoOmegaR47/(17.05/0.728) # fudged 43MHz calibration
			run.data['detuning'] = 100
			run.data['freq'] = 47.3227
			if file == '2025-02-13_I_e.dat':
				run.data['ODT1'] = 0.06
			elif file == "2025-02-13_O_e.dat":
				run.data['ODT1'] = 0.3
			elif file == '2025-03-06_E_e.dat':
				run.data['ODT1'] = 0.4 # this is a lie, just used to differentiate :P
		else:
			raise ValueError("File not in list")
		
		ff = fudge_factor
		
		# correct c9
		run.data['c9'] = ff * run.data['c9']
		
		run.data['pulse_time'] = pulse_time
		
		# add ToTF to data
		run.data['ToTF'] = run.data.ODT1.apply(ODT1_to_ToTF_map)
		
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
				
				if detuning != 100:
					continue
				
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
				phaseO_OmegaR = lambda VVA, freq: VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)
				
				# do some calculations
				df['N'] = df.c5 - bg_c5 + df.c9
				df['transfer'] = (df.c5 - bg_c5)/(df.N)
				df['loss'] = (bg_c9 - df.c9)/bg_c9
				df['anomalous_loss'] = (bg_c9 + bg_c5 - df.c9 - df.c5)/bg_c9
				
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
				ax.plot(xs, Linear(xs, popt_l[0]/popt_l[1], 0), '--', color=color)
				
				
				print(r"loss: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt_l[0], 
														  perr_l[0], popt_l[1], perr_l[1]))
				
				popts_l.append(popt_l)
				perrs_l.append(perr_l)
				
				
				# plot loss vs. anomalous_loss
				ax = axs[2]
				xlabel = r'Transfer $\Delta N_b/N_{b, bg}$'
				ax.set(xlabel=xlabel, ylabel=r'Loss $\Delta N/N_{bg}$')
				
				x = avg_df['loss']
				y = avg_df['anomalous_loss']
				yerr = avg_df['em_anomalous_loss']
				
		# 		ax.hlines(bg_c9, min(x), max(x), linestyle='--', color=color)
				ax.errorbar(x, y, yerr=yerr, label=label, **sty)
				
				# plot anomalous_loss vs. Nb cloud fit amplitude
				ax = axs[3]
				xlabel = r'Loss $\Delta N/N_{bg}$'
				ax.set(xlabel=xlabel, ylabel=r'$N_b$ fit amplitude')
				
				x = avg_df['anomalous_loss']
				y = avg_df['two2D_a2']#/avg_df['c9']
				yerr = avg_df['em_two2D_a2']#/avg_df['c9']
				
				ax.errorbar(x, y, yerr=yerr, label=label, **sty)
				
				
# 				# compare measured to linear 
# 				ax = axs[4]
# 				ax.set(xlabel='Measured transfer', xlim=[-0.02, 0.3], ylim=[-0.025, 0.4],
# 				   ylabel='Calibrated linear transfer')
# 				
# 				xs = np.linspace(0, popt[1], 1000)  # linspace of rf powers
# 				Gammas_Sat = Saturation(xs, *popt)
# 				Gammas_Lin = xs*popt[0]/popt[1]
# 				e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], popt[0], popt[1], perr[0], perr[1], pcov[0,1])
# 				
# 				if fill_between == True:
# 					ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
# 						Gammas_Lin+e_Gammas_Lin, alpha=0.5)
# 				ax.plot(Gammas_Sat, Gammas_Lin, ls, label=label, color=color)
# 				
# 				# compare measured to linear 
# 				ax = axs[5]
# 				ax.set(xlabel='Measured loss', xlim=[-0.02, 0.3], ylim=[-0.025, 0.4],
# 				   ylabel='Calibrated linear loss')
# 				
# 				xs = np.linspace(0, popt_l[1], 1000)  # linspace of rf powers
# 				Gammas_Sat = Saturation(xs, *popt_l)
# 				Gammas_Lin = xs*popt_l[0]/popt_l[1]
# 				ax.plot(Gammas_Sat, Gammas_Lin, ls, label=label)
				
				# append to results
				keys = ['file', 'detuning', 'pulse_time', 'ToTF', 'df', 'popt', 'pcov', 'popt_l', 'pcov_l']
				vals = [file, detuning, pulse_time, ToTF, avg_df, popt, pcov, popt_l, pcov_l]
				
				for key, val in zip(keys, vals):
					results[key] = val
					
				results_list.append(results)
				
				i += 1
				
# 	axs[4].plot(Gammas_Sat, Gammas_Sat, '-', color='dimgrey', zorder=1)
# 	axs[5].plot(Gammas_Sat, Gammas_Sat, '-', color='dimgrey', zorder=1)
	
	for ax in axs:
		ax.legend()
	fig.tight_layout()
	plt.show()
	
	
	if save == True:
		with open(pkl_file, "wb") as output_file:
			pkl.dump(results_list, output_file)
	