# -*- coding: utf-8 -*-
"""
2025 Feb 13
@author: Chip Lab


"""
from data_class import Data
from scipy.optimize import curve_fit
from library import paper_settings, styles, colors
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

# chosen average saturation Rabi
OmegaRabi2 = 3272
e_OmegaRabi2 = 136

pkl_file = os.path.join(data_path, "dimer_saturation_curves.pkl")

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c

def Offset(x, b):
	return b

# def Saturation(x, A, x0, B, w):
# 	return A*(1-B*np.exp(-x/np.abs(x0))*np.cos(w*x)**2)

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

def satratio(x, x0):
	return x/x0*1/(1-np.exp(-x/x0))

def quotient_propagation(f, A, B, sA, sB, sAB):
	return f* (sA**2/A**2 + sB**2/B**2 - 2*sAB/A/B)**(1/2)

def sum_with_coeff_propagation(A, B, a, b, sA, sB, sAB):
	return np.sqrt(a**2*sA**2 + b**2*sB**2 + 2*a*b*sAB)

def dimer_transfer(Rab, fa, fb):
	'''computes transfer to dimer state assuming loss in b is twice the loss
		in a. Rba = Nb/Na for the dimer association shot. Nb_bg and Na_bg
		are determined from averaged bg shots.'''
	return (fa - Rab*fb)/(1/2-Rab)

def sq_weighted_average(vals, errs):
	weights = 1/errs**2
	sum_weights = np.sum(weights)
	return np.sum([val*weight for val, weight in \
				zip(vals, weights)])/sum_weights
	
	

if __name__ == '__main__':
	
	spins = ['c5', 'c9']
	
	results_list = []
	
	files = [
# 			"2024-09-26_F_e.dat",
 			  "2025-02-13_J_e.dat",
 			  # "2025-02-13_N_e.dat",
 			  "2025-02-13_S_e.dat",
 			  "2025-02-14_C_e.dat",
 			  "2025-03-06_C_e.dat",
# 			  "2025-03-19_H_e.dat", # 640us
			  ]
	
	ToTFs = [
# 			0.54, # taken next day
 			0.306,
 			# 0.595,
 			0.415,
 			0.621,
 			0.95,
# 			0.55,
			]
	
	e_ToTFs = [
# 		0
		0.008,
# 		0,
		0,
		0.005,
		0,
# 		0,
		]
	
	pulse_times = [
# 				0.01,  # ms
				0.01,  
# 				0.01,  
				0.01,  
				0.01,  
				0.01,  
# 				0.640,
				]
	
	popts = []
	perrs = []
	popt_as = []
	perr_as = []
	popt_bs = []
	perr_bs = []
	popts_l = []
	perrs_l = []
	
	#### PLOTTING #####
	### plot settings
	plt.rcdefaults()
	plt.rcParams.update(paper_settings) # from library.py
	fig_width = 6.3
	
	# initialize plots
	fig, axes = plt.subplots(2, 3, figsize=(fig_width, 10/15*fig_width))
	axs = axes.flatten()
	
	fig.suptitle("10us square dimer transfer saturation")
	
	### ANALYSIS ###
	xname = 'VVA'
	plot_name = "Omega Rabi Squared (1/ms^2)"
	
	i = 0
	for j in range(len(files)):
		file = files[j]
		
		pulse_time= pulse_times[j]
		ToTF = ToTFs[j]
		e_ToTF = e_ToTFs[j]
		
		print("Analyzing", file, "at ToTF = ", ToTF)
		run = Data(file, path=data_path)
		
		df = run.data
		
		freq = 43.248
		if file == "2024-09-26_F_e.dat":
			fudge_factor = 1
		else:
			fudge_factor = 0.83
		
		ff = fudge_factor
		
		# correct c9
		df['c9'] = ff * df['c9']
		
		# do some calculations
		df['sum95'] = df['c5'] + df['c9']
		
		# ratio of atoms in each spin
		df['ratio95'] = df['c9']/df['c5']
		
		# fraction of atoms in each spin
		df['f5'] = df['c5']/df['sum95']
		df['f9'] = df['c9']/df['sum95']
		
		df['pulse_time'] = pulse_time
		df['ToTF'] = ToTF
		df['freq'] = 43.248
		
		# select background dfs
		if file == "2024-09-26_F_e.dat":
			bg_df = df.loc[df.time == 1] # this is the "bg" data
			df = df.loc[df.time == 10] # select only 10us data from this run
			VpptoOmegaR43 = 14.44/0.656 # 43MHz calibration from 2024
		else:
			# 0 VVA as bg point, works for all detunings
			bg_df = df.loc[(df.VVA == 0)]
			df = df.drop(bg_df.index)
			VpptoOmegaR47 = 12.01/0.452 # kHz/Vpp - 2025-02-12 calibration 
			VpptoOmegaR43 = 14.44/0.656 * VpptoOmegaR47/(17.05/0.728) # fudged 43MHz calibration
		
		# calculate bg stuffs
		bg_c5 = bg_df.c5.mean()
		e_bg_c5 = bg_df.c5.sem()
		bg_c9 = bg_df.c9.mean()
		e_bg_c9 = bg_df.c9.sem()
		
		df['bgc5'] = bg_c5
		df['bgc9'] = bg_c9
			
		results = {}
		
		xlims = [0, 10]
		cutoff = 10000*(0.265/0.21)**2

		### Omega Rabi calibrations
		# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
		# nuance: need to correct the Vpp_from_VVAfreq calibration for micromatic
		phaseO_OmegaR = lambda VVA, freq: VpptoOmegaR43 * Vpp_from_VVAfreq(VVA, freq)
		phaseOmicrO_corr_factor = 0.265/0.21 # micro is larger than phaseo
# 		phaseOmicrO_corr_factor = 1
		
		# background fractions
		bg_f9 = bg_df['f9'].mean()
		e_bg_f9 = bg_df['f9'].sem()
		bg_f5 = bg_df['f5'].mean()
		e_bg_f5 = bg_df['f5'].sem()
		
		for spin in spins:
			df[spin+'_transfer'] = (df['bg'+spin] - df[spin])/(df['bg'+spin])
			
		df['c5_transfer'] = df['c5_transfer']/2
		
		df['ratio_transfer'] = dimer_transfer(df['ratio95'], bg_f9, bg_f5)
		
		if file != "2025-03-19_H_e.dat":
			df['OmegaR'] = phaseO_OmegaR(df.VVA, df.freq) * phaseOmicrO_corr_factor
		else :
			df['OmegaR'] = phaseO_OmegaR(df.VVA, df.freq) * 1 # used phaseOmatic 
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
		ymax = 0.35
		ymin = -0.05
		
		ymaxs = []
		Gamma_maxs = []
		
		# transfer ratio
		sty = styles[i]
		color = colors[i]
		label = r'ToTF = {}'.format(df['ToTF'].unique()[0])
		x = avg_df[xname]
		y = avg_df['ratio_transfer']
		yerr = avg_df['em_ratio_transfer']
		
		ymaxs.append(max(y))
		
		xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
		
		ax = axs[0]
		ax.errorbar(x, y, yerr=yerr, **sty)
		
		# fit to saturation curve
		p0 = [0.2, 1800]
		popt, pcov = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
		alpha0 = popt[0]
		perr = np.sqrt(np.diag(pcov))
		ax.plot(xs, Saturation(xs, *popt), '-', label=label, color=color)
		ax.plot(xs, Linear(xs, popt[0]/popt[1], 0), '--', color=color)
		
		print(r"a/b transfer: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt[0], 
												  perr[0], popt[1], perr[1]))
		# store popt in list
		popts.append(popt)
		perrs.append(perr)
		
		# rescale x-axis
		ax = axs[0+3]
		x = avg_df[xname]/OmegaRabi2
		
		ax.errorbar(x, y, yerr=yerr, **sty)
		ax.plot(xs/OmegaRabi2, Saturation(xs, *popt), '-', 
		  label=label, color=color)
		ax.plot(xs/OmegaRabi2, Linear(xs, popt[0]/popt[1], 0), '--', 
		  color=color)
		
		
		# a transfer
		sty = styles[i]
		color = colors[i]
		label = r'ToTF = {}'.format(df['ToTF'].unique()[0])
		x = avg_df[xname]
		y = avg_df['c9_transfer']
		yerr = avg_df['em_c9_transfer']
		
		ymaxs.append(max(y))
		
		xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
		
		ax = axs[1]
		ax.errorbar(x, y, yerr=yerr, **sty)
		
		# fit to saturation curve
		p0 = [0.2, 1800]
		popt_a, pcov_a = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
		perr_a = np.sqrt(np.diag(pcov_a))
		alpha0 = popt_a[0]
		ax.plot(xs, Saturation(xs, *popt_a), '-', label=label, color=color)
		ax.plot(xs, Linear(xs, popt_a[0]/popt_a[1], 0), '--', color=color)
		
		print(r"a transfer: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt_a[0], 
												  perr_a[0], popt_a[1], perr_a[1]))
		# store popt in list
		popt_as.append(popt_a)
		perr_as.append(perr_a)
		
		# rescale x-axis
		ax = axs[1+3]
		x = avg_df[xname]/OmegaRabi2
		
		ax.errorbar(x, y, yerr=yerr, **sty)
		ax.plot(xs/OmegaRabi2, Saturation(xs, *popt_a), '-', 
		  label=label, color=color)
		ax.plot(xs/OmegaRabi2, Linear(xs, popt_a[0]/popt_a[1], 0), 
		  '--', color=color)
		
		# b transfer
		sty = styles[i]
		color = colors[i]
		label = r'ToTF = {}'.format(df['ToTF'].unique()[0])
		x = avg_df[xname]
		y = avg_df['c5_transfer']
		yerr = avg_df['em_c5_transfer']
		
		ymaxs.append(max(y))
		
		xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
		
		ax = axs[2]
		ax.errorbar(x, y, yerr=yerr, **sty)
		
		# fit to saturation curve
		p0 = [0.2, 1800]
		popt_b, pcov_b = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
		alpha0 = popt_b[0]
		perr_b = np.sqrt(np.diag(pcov_b))
		ax.plot(xs, Saturation(xs, *popt_b), '-', label=label, color=color)
		ax.plot(xs, Linear(xs, popt_b[0]/popt_b[1], 0), '--', color=color)
		
		print(r"b transfer: A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt_b[0], 
												  perr_b[0], popt_b[1], perr_b[1]))
		# store popt in list
		popt_bs.append(popt_b)
		perr_bs.append(perr_b)
		
		# rescale x-axis
		ax = axs[2+3]
		x = avg_df[xname]/OmegaRabi2
		
		ax.errorbar(x, y, yerr=yerr, **sty)
		ax.plot(xs/OmegaRabi2, Saturation(xs, *popt_b), '-', 
		  label=label, color=color)
		ax.plot(xs/OmegaRabi2, Linear(xs, popt_b[0]/popt_b[1], 0), 
		  '--', color=color)
		
		# append to results
		keys = ['file', 'pulse_time', 'ToTF', 'e_ToTF', 'df', 'popt', 'pcov', 
		  'popt_a', 'pcov_a', 'popt_b', 'pcov_b']
		vals = [file, pulse_time, ToTF, e_ToTF, avg_df, popt, pcov, 
		  popt_a, pcov_a, popt_b, pcov_b]
		
		for key, val in zip(keys, vals):
			results[key] = val
			
		results_list.append(results)
		
		i += 1
		
	# add axs labels and y=x lines
	spin_labels = ['ratio', 'a', 'b']
	ymaxs = [0.2, 0.3, 0.3]
	for j in range(len(axs)-3):
		axs[j].set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', 
		  ylabel=spin_labels[j]+r' Transfer $\alpha$', ylim=[-0.02, ymaxs[j]])
		axs[j+3].set(xlabel='rf power $P/P_0$', 
					   ylabel=spin_labels[j]+r' Transfer $\alpha$', 
					   ylim=[-0.02, ymaxs[j]*1.1])
				
	for ax in axs:
		ax.legend()
		
	fig.tight_layout()
	
	if save == True:
		with open(pkl_file, "wb") as output_file:
			pkl.dump(results_list, output_file)
	
sq_weighted_avg_x0 = sq_weighted_average(np.array(popts)[:,1], np.array(perrs)[:,1])
print("Weighted average x0 = {:.2f}".format(sq_weighted_avg_x0))

plt.figure(figsize=(fig_width, 4/6*fig_width))
plt.title(r"Dimer Saturation Rabi Frequency")
x = np.array(ToTFs)
x_avg = np.mean(x)

y = np.array(popts)[:,1]
yerr = np.array(perrs)[:,1]

popt, pcov = curve_fit(Offset, x, y)
perr = np.sqrt(pcov)

y_avg = Offset(x_avg, *popt)
e_y_avg = perr[0,0]

print("Average 1/e Omega Rabi^2 is {:.0f}+/-{:.0f} kHz^2".format(y_avg, e_y_avg))
label =  "{:.0f}+/-{:.0f}".format(y_avg, e_y_avg) + r" kHz$^2$"

plt.errorbar(x, y, yerr=yerr, **styles[0])
plt.plot(x, np.ones(len(x))*Offset(x, *popt), '--', label=label)
plt.xlabel(r"Temperature, $T$ $[T_F]$")
plt.ylabel(r"1/e Power $\Omega_R^2$ [kHz$^2$]")
plt.legend()
plt.show()

			 