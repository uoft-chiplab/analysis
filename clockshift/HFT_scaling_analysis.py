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

# double it so we don't run out
styles = styles + styles
colors = colors + colors

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(proj_path, 'saturation_data')
root = os.path.dirname(proj_path)

# plot error bands for saturation curves
fill_between = True
save = False
legends = False

pkl_file = os.path.join(data_path, "HFT_saturation_curves.pkl")

### Calibrations
RabiperVpp_47MHz_2024 = 17.05/0.728 # 2024-09-16
e_RabiperVpp_47MHz_2024 = 0.15

RabiperVpp_47MHz_2025 = 12.01/0.452 # 2025-02-12
e_RabiperVpp_47MHz_2025 = 0.28

### Fit functions
def Linear(x, m):
	return m*x

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

def satratio(x, x0):
	return x/x0*1/(1-np.exp(-x/x0))

def quotient_propagation(f, A, B, sA, sB, sAB):
	return f* (sA**2/A**2 + sB**2/B**2 - 2*sAB/A/B)**(1/2)

# columns = ["filename", "detuning", "fudge_factor", "ToTF", "EF", "pulse_time"]
	
detunings = [
			25,
			50,
			100,
			150,
			250,
			400,
			600,
			800,
			1000,
			1300,
			1600,
			2000,
			]

files = [
		"2024-11-28_P_e_detuning=25.dat",
		"2024-11-28_P_e_detuning=50.dat",
		"2024-11-28_O_e.dat",
		"2024-11-28_P_e_detuning=150.dat",
		"2025-03-27_C_e_freq=47.4727.dat",
		"2025-03-27_C_e_freq=47.6227.dat",
		"2025-03-27_C_e_freq=47.8227.dat",
		"2025-03-27_C_e_freq=48.0227.dat",
		"2025-03-27_C_e_freq=48.2227.dat",
		"2025-03-27_C_e_freq=48.5227.dat",
		"2025-03-27_C_e_freq=48.8227.dat",
		"2025-03-27_C_e_freq=49.2227.dat",
		  ]

fudge_factors = [
 					0.98, 
			  0.98, 
			  0.98, 
			  0.98,
			  0.88,
			  0.88,
			  0.88,
			  0.88,
			  0.88,
			  0.88,
			  0.88,
			  0.88,
			  ]

ToTFs = [0.616,  # from the next day, but should be roughly right
	  0.616,
	  0.616,
	  0.616,
	  0.525,
	  0.525,
	  0.525,
	  0.525,
	  0.525,
	  0.525,
	  0.525,
	  0.525,
	  ]

EFs = [19.4,  # same as above
   19.4,
   19.4,
   19.4,
   18.9,
   18.9,
   18.9,
   18.9,
   18.9,
   18.9,
   18.9,
   18.9,
   ]

pulse_times = [0.2,  # ms
			0.2,
			0.2,
			0.2,
			1.0,
			1.0,
			1.0,
			1.0,
			1.0,
			1.0,
			1.0,
			1.0,
			]
results_list = []
popts = []
perrs = []
popts_l = []
perrs_l = []

#### PLOTTING #####
# initialize plots
fig, axes = plt.subplots(2, 2, figsize=(12,10))
axs = axes.flatten()

dfig, daxes = plt.subplots(4,3, figsize=(12,10))
daxs = daxes.flatten()

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
axs[2].set(xlabel='Measured transfer', xlim=[-0.01, 0.31],
		ylabel='Calibrated linear transfer', ylim=[-0.02, 0.5])
axs[3].set(xlabel='Measured loss', xlim=[-0.02, 0.45],
		ylabel='Calibrated linear loss', ylim=[-0.02, 0.75])

for i in range(len(detunings)):
	results = {}
	file = files[i]
	detuning = detunings[i]
	ff = fudge_factors[i]
	pulse_time = pulse_times[i]
	EF = EFs[i]
	ToTF = ToTFs[i]
	
	
	### Omega Rabi calibrations
	# check which year, taken from filename
	if file[:4] == '2024':
		RabiperVpp47 = RabiperVpp_47MHz_2024 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
		e_RabiperVpp47 = e_RabiperVpp_47MHz_2024
	elif file[:4] == '2025':
		RabiperVpp47 = RabiperVpp_47MHz_2025 # kHz/Vpp - 2025-02-12 calibration 
		e_RabiperVpp47 = e_RabiperVpp_47MHz_2025
	else:
		raise ValueError("filename does not start with 2024 or 2025.")
	
	# Rabi frequencies, phaseO is HFT
	# in kHz!!!
	phaseO_OmegaR = lambda VVA, freq: RabiperVpp47 * Vpp_from_VVAfreq(VVA, freq)
			
	print()
	print("*------------------")
	print("Analyzing", file)
	run = Data(file, path=data_path)
	run.data['ToTF'] = ToTF
	run.data['pulse_time'] = pulse_time
	run.data['EF'] = EF
	run.data['c9'] = ff * run.data['c9']
	
	run.data['sum95'] = run.data.c5 + run.data.c9
	
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
	
	if detuning == 100: # for this dataset, we had to add the VVA to df
		run.data['OmegaR'] = phaseO_OmegaR(run.data.VVA, 47.3227) * np.sqrt(0.31)
	else:
		run.data['OmegaR'] = phaseO_OmegaR(run.data.VVA, run.data.freq) * np.sqrt(0.31)
	run.data['OmegaR2'] = (run.data['OmegaR'])**2
	xname = 'OmegaR2'
	
	run.group_by_mean(xname)
	
	# calculate seperately to propagate bg error
	run.avg_data['loss'] = 1 - run.avg_data.c9/bg_c9
	run.avg_data['em_loss'] = run.avg_data.c9/bg_c9*np.sqrt(\
						   (run.avg_data.em_c9/run.avg_data.c9)**2 +\
						         (e_bg_c9/bg_c9)**2)
		
	###
	### Diagnostic plotting
	###
	
	df = run.avg_data
	title = file[0:12] + ",\n"+f"{detuning:.0f} kHz, "+f"{ToTF:.2f}"+r"$T_F$"
	dax = daxs[i]
	dax.title.set_text(title)
	
	yname = 'sum95'
	
	# set axes labels on the border of the subplot array
	if i in [0, 3, 6, 9]:
		dax.set(ylabel=r"Atom number, $N$")
	if i in [9, 10, 11]:
		dax.set(xlabel=r"$\Omega_R^2$ (kHz$^2$)")
	
	dax.errorbar(df[xname], df[yname], yerr=df['em_'+yname], **styles[0], 
		  label='signal')
	dax.errorbar([0], bg_df[yname].mean(), yerr=bg_df[yname].sem(), 
		  **styles[1], label='bg')
	
	###
	### fit transfer to saturation curve
	###
	p0 = [0.5, 1000]
	x = run.avg_data[xname]
	y = run.avg_data['transfer']
	yerr = run.avg_data['em_transfer']
	fit_func = Saturation
	if len(x) > 1:
		popt, pcov = curve_fit(fit_func, x, y, p0=p0, sigma=yerr)
		perr = np.sqrt(np.diag(pcov))
		xmax = popt[1] # set to 1/e x
		slope = popt[0]/popt[1]
		e_slope = quotient_propagation(slope, popt[0], popt[1], perr[0], 
							  perr[1], pcov[0,1])
		
		if np.abs(perr[1]) > np.abs(popt[1]): # i.e. poor fit
			fit_func = Linear
			popt, pcov = curve_fit(fit_func, x, y, sigma=yerr)
			perr = np.sqrt(np.diag(pcov))
			slope = popt[0]
			e_slope = perr[0]
			xmax = max(x) # set to max if not saturating
			
	else: # if only 1 point, a little dumb.. 
		popt = [y[0]/x[0]]
		perr = [yerr[0]]
		slope = popt[0]
		e_slope = perr[0]
		fit_func = Linear

	###
	### fit loss to saturation curve
	###
	y = run.avg_data['loss']
	yerr = run.avg_data['em_loss']
	p0 = [0.5, 1000]
	fit_func_l = Saturation
	if len(x) > 1:
		popt_l, pcov_l = curve_fit(fit_func_l, x, y, p0=p0, sigma=yerr)
		perr_l = np.sqrt(np.diag(pcov_l))
		xmax_l = popt_l[1]
		slope_l = popt_l[0]/popt_l[1]
		e_slope_l = quotient_propagation(slope_l, popt_l[0], popt_l[1], 
								perr_l[0], perr_l[1], pcov_l[0,1])
		
		if np.abs(perr_l[1]) > np.abs(popt_l[1]):
			fit_func_l = Linear
			popt_l, pcov_l = curve_fit(fit_func_l, x, y, sigma=yerr)
			perr_l = np.sqrt(np.diag(pcov_l))
			xmax_l = max(x)
	else:
		popt_l = [y[0]/x[0]]
		perr_l = [yerr[0]/x[0]]
		fit_func_l = Linear
	
	###	
	### PLOTTING
	###
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
	
	ax.plot(xs, fit_func(xs, *popt), '-', label=label, color=color)
	if fit_func == Saturation:
		ax.plot(xs, Linear(xs, slope), '--', color=color)
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
	
	ax.plot(xs, fit_func_l(xs, *popt_l), '-', label=label, color=color)
	if fit_func_l == Saturation:
		ax.plot(xs, Linear(xs, slope_l), '--', color=color)
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
	
	xs = np.linspace(0, xmax, 1000)  # linspace of rf powers
	
	Gammas_Sat = fit_func(xs, *popt)
	Gammas_Lin = xs*slope
	e_Gammas_Lin = xs*e_slope
	
	ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label)
	if fill_between == True:
		ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
			Gammas_Lin+e_Gammas_Lin, alpha=0.5)
	ax.legend()
	
	# loss
	ax=axs[3]
	xs = np.linspace(0, xmax_l, 1000)  # linspace of rf powers
	
	Gammas_Sat = fit_func_l(xs, *popt_l)
	Gammas_Lin = xs*slope_l
	e_Gammas_Lin = xs*e_slope_l
	
	if fill_between == True:
		ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
			Gammas_Lin+e_Gammas_Lin, alpha=0.5)
	
	ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label)
	
	# append to results
	
	keys = ['file', 'detuning', 'pulse_time', 'ToTF', 'EF', 'df', 
	  'popt', 'pcov', 'popt_l', 'pcov_l', 'fit_func', 'fit_func_l', 
	  'slope', 'e_slope', 'slope_l', 'e_slope_l']
	vals = [file, detuning, pulse_time, ToTF, EF, run.avg_data, 
	  popt, pcov, popt_l, pcov_l, fit_func, fit_func_l,
	  slope, e_slope, slope_l, e_slope_l]
	
	for key, val in zip(keys, vals):
		results[key] = val
		
	results_list.append(results)

# add y = x lines to calibration plots
axs[2].plot(Gammas_Sat, Gammas_Sat, '-', color='dimgrey', zorder=1)
axs[3].plot(Gammas_Sat, Gammas_Sat, '-', color='dimgrey', zorder=1)

if legends == True:
	for ax in axs:
		ax.legend()
fig.tight_layout()
dfig.tight_layout()
plt.show()


if save == True:
	with open(pkl_file, "wb") as output_file:
		pkl.dump(results_list, output_file)
		
		


	