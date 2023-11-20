# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Fit bulk viscosity from wiggle heating data, scanning
- time
- freq
- amp

Relies on data_class.py, library.py

Requires tabulate. In console, execute the command:
	!pip install tabulate
"""
from data_class import Data
from library import *
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

data_folder = 'data/heating'

temp_param = "ToTFcalc"

plotting = True

test_time_plots = False
test_amp_plots = False
test_freq_plots = False

NO_OFFSET = False
EF_CUTOFF = 0

colors = ["blue", "red", 
		  "green", "orange", 
		  "purple", "black", "pink"]

def ToTFfunc(t, A, omega, zeta, EF, C):
	return 9/2*A**2*hbar*omega*omega*t*zeta/EF + C

def ToTFfunc_highfreq(t, A, omega, contact, EF, C):
	return 1/(2*np.sqrt(2)*pi)*A**2*(omega*EF/hbar)**(1/2)*t*contact + C

def ToTFfunc_tilman(omega, zeta, EF):
	return 9/2*hbar*omega**2/EF*zeta

def crappy_chi_sq(y, yfit, yerr, dof):
	return 1/dof * np.sum((np.array(y) - np.array(yfit))**2/(yerr**2))

### metadata

# scan time
C_1010 = {'filename':'2023-10-10_C_e.dat','freq':10e3,'Bamp':0.1,'B':202.1,
		  'Ni':34578,'Ti':0.528}

C_1018 = {'filename':'2023-10-18_C_e.dat','freq':30e3,'Bamp':0.1,'B':202.1,
		  'Ni':36439,'Ti':0.521}
D_1018 = {'filename':'2023-10-18_D_e.dat','freq':100e3,'Bamp':0.1,'B':202.1,
		  'Ni':36439,'Ti':0.521}
E_1018 = {'filename':'2023-10-18_E_e.dat','freq':100e3,'Bamp':0.05,'B':202.1,
		  'Ni':36439,'Ti':0.521}

G_1107 = {'filename':'2023-11-07_G_e.dat','freq':15e3,'Bamp':0.05*1.8,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

H_1107 = {'filename':'2023-11-07_H_e.dat','freq':5e3,'Bamp':0.07,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

J_1107 = {'filename':'2023-11-07_J_e.dat','freq':50e3,'Bamp':0.05*0.7,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

K_1107 = {'filename':'2023-11-07_K_e.dat','freq':150e3,'Bamp':0.05*0.54,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

R_1107 = {'filename':'2023-11-07_R_e.dat','freq':10e3,'Bamp':0.05*1.21,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

M_1107 = {'filename':'2023-11-07_M_e.dat','freq':30e3,'Bamp':0.05*1.286,'B':202.1,
		  'Ni':40307,'Ti':0.6, 'GTi':0.633}

B_1109 = {'filename':'2023-11-09_B_e.dat','freq':15e3,'Bamp':0.05*1.8,'B':203,
		  'Ni':49429,'Ti':0.48, 'GTi':0.553}

C_1109 = {'filename':'2023-11-09_C_e.dat','freq':50e3,'Bamp':0.05*(0.7/1),'B':203,
		  'Ni':49429,'Ti':0.48, 'GTi':0.553}

D_1109 = {'filename':'2023-11-09_D_e.dat','freq':15e3,'Bamp':0.05*1.8,'B':203,
		  'Ni':28230,'Ti':0.662, 'GTi':0.693}

F_1109 = {'filename':'2023-11-09_F_e.dat','freq':5e3,'Bamp':0.07,'B':203,
		  'Ni':28230,'Ti':0.662, 'GTi':0.693} 

G_1109 = {'filename':'2023-11-09_G_e.dat','freq':50e3,'Bamp':0.07,'B':203,
		  'Ni':28230,'Ti':0.662, 'GTi':0.693}

I_1109 = {'filename':'2023-11-09_I_e.dat','freq':150e3,'Bamp':0.07*0.35,'B':203,
		  'Ni':27603,'Ti':0.582, 'GTi':0.637}

fit_time = {'param':'time (ms)','xlabel':'Time (ms)','fit':ToTFfunc,
			'runs':[B_1109,C_1109,D_1109,F_1109,G_1109,I_1109]}

#B_1109,C_1109,D_1109,F_1109,G_1109,I_1109
#G_1107,J_1107,K_1107,R_1107,M_1107

# scan amp
D_1010 = {'filename':'2023-10-10_D_e.dat','freq':10e3,'time':5e-3,'B':202.1,
		  'Ni':34578,'Ti':0.528}

J_1017 = {'filename':'2023-10-17_J_e.dat','freq':30e3,'time':5e-3,'B':202.1,
		  'Ni':32517,'Ti':0.529}

F_1018 = {'filename':'2023-10-18_F_e.dat','freq':100e3,'time':2e-3,'B':202.1,
		  'Ni':36439,'Ti':0.521}

fit_amp = {'param':'amp (Vpp)','xlabel':'Amplitude (1/kFa)','fit':ToTFfunc,
		   'runs':[D_1010,J_1017,F_1018]}

# scan freq
E_1010 = {'filename':'2023-10-10_E_e.dat','Bamp':0.1,'time':5e-3,'B':202.1,
		  'Ni':34578,'Ti':0.528}

G_1017 = {'filename':'2023-10-17_G_e.dat','Bamp':0.1,'time':10e-3,'B':209,
		  'Ni':39573,'Ti':0.317}
I_1017 = {'filename':'2023-10-17_I_e.dat','Bamp':0.1,'time':5e-3,'B':202.1,
		  'Ni':32517,'Ti':0.529}
K_1017 = {'filename':'2023-10-17_K_e.dat','Bamp':0.1,'time':2e-3,'B':202.1,
		  'Ni':32517,'Ti':0.529}

H_1018 = {'filename':'2023-10-18_H_e.dat','Bamp':0.05,'time':1e-3,'B':202.1,
		  'Ni':36439,'Ti':0.521}
I_1018 = {'filename':'2023-10-18_I_e.dat','Bamp':0.1,'time':1e-3,'B':202.1,
		  'Ni':36439,'Ti':0.521}

S_1031 = {'filename':'2023-10-31_S_e.dat','Bamp':0.05,'time':1e-3,'B':202.1,
		  'Ni':34804,'Ti':0.568, 'GTi':0.604}
	  

fit_freq = {'param':'freq (kHz)','xlabel':'Wiggle Freq (kHz)','fit':ToTFfunc,
			'runs':[S_1031]}

mean_trapfreq = 2*pi*(151.6*429*442)**(1/3)
Bamp_per_Vpp = 0.1/2

### start analysis
Nrange_tolerance = 0.05
fit_types = [fit_time, fit_amp, fit_freq]
# fit_types = [fit_freq]

##################
################## TIME
##################

fit_type = fit_time
fit_type['run_results'] = []
for run in fit_type['runs']:
	# load data, passing in metadata, turning run into a class!!!
	run = Data(run['filename'], path = data_folder, metadata=run)
	
	# check if mean N matches initial 97 TShots
	run.Nmean = run.data['LiNfit'].mean()
# 	Nrange_tolerance = run.Nsd/run.Nmean
	if np.abs(run.Nmean-run.Ni) > Nrange_tolerance * run.Ni:
		run.Ndiff = np.abs(run.Nmean-run.Ni)/ run.Ni
		print("{} has N_mean > {:.0f}% different than Ni ({:.0f}%)".format(run.filename, 
									  Nrange_tolerance*100, run.Ndiff*100))
	
	run.data['kF'] = run.data["LiNfit"].apply(lambda n: \
						   FermiWavenumber(n, mean_trapfreq))
	run.data['EF'] = run.data["LiNfit"].apply(lambda n: \
						   FermiEnergy(n, mean_trapfreq))
	
	def calc_A(kF):
		try:
		    Ah = 1/(a97(run.B+run.Bamp)*kF)
		except ZeroDivisionError:
		    Ah = 0
		try: 
			Al = 1/(a97(run.B-run.Bamp)*kF)
		except ZeroDivisionError:
			Al = 0
			print("ZeroDivisionError in {}",format(run.filename))
		return np.abs(Ah - Al)/2
			
	run.data['A'] = run.data['kF'].apply(calc_A)
	run.data['DToTFcalc'] = run.data[temp_param]-run.Ti
	
	run.Amean = run.data['A'].mean()
	run.EFmean = run.data['EF'].mean()
	
	run.counts = run.data.groupby(fit_type['param']).count()
	run.group_by_mean(fit_type['param'])
	
	if NO_OFFSET == True:
		run.fit_func = lambda t, zeta: ToTFfunc(t/1000, run.Amean, 2*pi*run.freq, 
											zeta, run.EFmean, 0)
		num_params = 1
		
	else:
		run.fit_func = lambda t, zeta, C: ToTFfunc(t/1000, run.Amean, 2*pi*run.freq, 
											zeta, run.EFmean, C)
		num_params = 2
		
	run.popt, run.pcov = curve_fit(run.fit_func, run.data[fit_type['param']], 
						run.data['DToTFcalc'])
	run.err = np.sqrt(np.diag(run.pcov))
	
	run.scatter = run.avg_data['e_DToTFcalc'].mean()
	run.dof = len(run.data.index) - num_params # num data - num params in the fit
	
	run.chi_sq = crappy_chi_sq(run.data['DToTFcalc'], 
					run.fit_func(run.data[fit_type['param']], *run.popt), 
					run.scatter, run.dof)
	
	run.label = "A={:.2f}, f={:.1f} kHz, X^2={:.1f}, z={:.2E}+-{:.1E}".format(run.Amean, 
						 run.freq/1000, run.chi_sq, run.popt[0], run.err[0])
	fit_type['run_results'].append(run)

	if test_time_plots == True:
		run.plot([fit_type['param'], 'DToTFcalc'])
		num = 100
		xlist = np.linspace(run.data[fit_type['param']].min(),
					  run.data[fit_type['param']].max(), num)
		run.ax.plot(xlist, run.fit_func(xlist, *run.popt))

##################		
################## AMP
##################
fit_type = fit_amp
fit_type['run_results'] = []
for run in fit_type['runs']:
	# load data, passing in metadata, turning run into a class!!!
	run = Data(run['filename'], path = data_folder, metadata=run)
	
	# check if mean N matches initial 97 TShots
	run.Nmean = run.data['LiNfit'].mean()
# 	Nrange_tolerance = run.Nsd/run.Nmean
	if np.abs(run.Nmean-run.Ni) > Nrange_tolerance * run.Ni:
		run.Ndiff = np.abs(run.Nmean-run.Ni)/ run.Ni
		print("{} has N_mean > {:.0f}% different than Ni ({:.0f}%)".format(run.filename, 
									  Nrange_tolerance*100, run.Ndiff*100))
	
	run.data['kF'] = run.data["LiNfit"].apply(lambda n: \
						   FermiWavenumber(n, mean_trapfreq))
	run.data['EF'] = run.data["LiNfit"].apply(lambda n: \
						   FermiEnergy(n, mean_trapfreq))
	run.kFmean = run.data['kF'].mean()
			
	def calc_A(Vpp):
		try:
		    Ah = 1/(a97(run.B+Bamp_per_Vpp*Vpp)*run.kFmean)
		except ZeroDivisionError:
		    Ah = 0
		try: 
			Al = 1/(a97(run.B-Bamp_per_Vpp*Vpp) * run.kFmean)
		except ZeroDivisionError:
			Al = 0
			print("ZeroDivisionError in {}",format(run.filename))
		return np.abs(Ah - Al)/2
			
	run.data['A'] = run.data[fit_type['param']].apply(calc_A)
	run.data['DToTFcalc'] = run.data[temp_param]-run.Ti
	
	run.EFmean = run.data['EF'].mean()
	
	run.counts = run.data.groupby(fit_type['param']).count()
	run.group_by_mean(fit_type['param'])
	
	if NO_OFFSET == True:
		run.fit_func = lambda A, zeta: ToTFfunc(run.time, A, 2*pi*run.freq, 
										zeta, run.EFmean, 0)
		num_params = 1
		
	else:
		run.fit_func = lambda A, zeta, C: ToTFfunc(run.time, A, 2*pi*run.freq, 
										zeta, run.EFmean, C)
		num_params = 2
		
	run.popt, run.pcov = curve_fit(run.fit_func, run.data['A'], 
						run.data['DToTFcalc'])
	run.err = np.sqrt(np.diag(run.pcov))
	
	run.scatter = run.avg_data['e_DToTFcalc'].mean()
	run.dof = len(run.data.index) - num_params # num data - num params in the fit
	
	run.chi_sq = crappy_chi_sq(run.data['DToTFcalc'], 
					run.fit_func(run.data['A'], *run.popt), 
					run.scatter, run.dof)
	
	run.label = "t={:.0f} ms, f={:.1f} kHz, X^2={:.1f}, z={:.2E}+-{:.1E}".format(run.time*1000, 
						 run.freq/1000, run.chi_sq, run.popt[0], run.err[0])
	
	fit_type['run_results'].append(run)

	if test_amp_plots == True:
		run.plot(['A', 'DToTFcalc'])
		num = 100
		xlist = np.linspace(run.data['A'].min(),
					  run.data['A'].max(), num)
		run.ax.plot(xlist, run.fit_func(xlist, *run.popt))

###############
############### FREQ
###############
fit_type = fit_freq
fit_type['run_results'] = []
for run in fit_type['runs']:
	# load data, passing in metadata, turning run into a class!!!
	run = Data(run['filename'], path = data_folder, metadata=run)
	
	# check if mean N matches initial 97 TShots
	run.Nmean = run.data['LiNfit'].mean()
# 	Nrange_tolerance = run.Nsd/run.Nmean
	if np.abs(run.Nmean-run.Ni) > Nrange_tolerance * run.Ni:
		run.Ndiff = np.abs(run.Nmean-run.Ni)/ run.Ni
		print("{} has N_mean > {:.0f}% different than Ni ({:.0f}%)".format(run.filename, 
									  Nrange_tolerance*100, run.Ndiff*100))
	
	run.data['kF'] = run.data["LiNfit"].apply(lambda n: \
						   FermiWavenumber(n, mean_trapfreq))
	run.data['EF'] = run.data["LiNfit"].apply(lambda n: \
						   FermiEnergy(n, mean_trapfreq))
	run.kFmean = run.data['kF'].mean()
			
	Ah = 1/(a97(run.B+run.Bamp)*run.kFmean)
	Al = 1/(a97(run.B-run.Bamp)*run.kFmean)
	run.A = np.abs(Ah - Al)/2
			
	run.data['DToTFcalc'] = run.data[temp_param]-run.Ti
	run.EFmean = run.data['EF'].mean()
	
	run.data['contact'] = 2*np.sqrt(2)*pi/run.A**2/run.time \
			* np.sqrt(hbar/(2*pi*run.data['freq (kHz)']*1000)/run.EFmean) \
			* run.data['DToTFcalc']
	
	run.data = run.data.drop(run.data[run.data[fit_type['param']] < EF_CUTOFF \
								  * run.EFmean/h/1000].index)
	
	run.counts = run.data.groupby(fit_type['param']).count()
	run.group_by_mean(fit_type['param'])
	
	if NO_OFFSET == True:
		run.fit_func = lambda f, contact: ToTFfunc_highfreq(run.time, run.A, 
							 2*pi*f*1000, contact, run.EFmean, 0)
		num_params = 1
		
	else:
		run.fit_func = lambda f, contact, C: ToTFfunc_highfreq(run.time, run.A, 
							 2*pi*f*1000, contact, run.EFmean, C)
		num_params = 2
	
	run.popt, run.pcov = curve_fit(run.fit_func, run.data[fit_type['param']], 
						run.data['DToTFcalc'])
	run.err = np.sqrt(np.diag(run.pcov))
	
	run.scatter = run.avg_data['e_DToTFcalc'].mean()
	run.dof = len(run.data.index) - num_params # num data - num params in the fit
	
	run.chi_sq = crappy_chi_sq(run.data['DToTFcalc'], 
					run.fit_func(run.data[fit_type['param']], *run.popt), 
					run.scatter, run.dof)
	
	run.label = "t={:.0f} ms, A={:.2f}, X^2={:.1f}, C={:.2f}+-{:.2f}".format(run.time*1000, 
						 run.A, run.chi_sq, run.popt[0], run.err[0])
	
	fit_type['run_results'].append(run)
	
	if test_freq_plots == True:
		run.plot([fit_type['param'], 'DToTFcalc'])
		num = 100
		xlist = np.linspace(run.data[fit_type['param']].min(),
					  run.data[fit_type['param']].max(), num)
		run.ax.plot(xlist, run.fit_func(xlist, *run.popt))

############## PLOTTING ##############		
if plotting == True:
	fit_amp['param'] = 'A'
	
	plt.rcParams.update({"figure.figsize": [12,8]})
	fig, axs = plt.subplots(2,2)
	num = 100
	ylabel = "Heating (T/TF)"
	
	for fit_type, ax in zip(fit_types, axs.reshape(-1)[:-1]):
		runs = fit_type['run_results']
		xlabel = fit_type['xlabel']
# 		xmin = run.data[fit_type['param']].min()
		xmin = 0
		
		ax.set(ylabel=ylabel, xlabel=xlabel)
		for run, color in zip(runs, colors):
			ax.errorbar(np.array(run.avg_data[fit_type['param']]),np.array(run.avg_data['DToTFcalc']), 
						 yerr = np.array(run.avg_data['em_DToTFcalc']), label=run.label, 
						 color=color, capsize=2, fmt='o')
			xlist = np.linspace(xmin,run.data[fit_type['param']].max(), num)
			ax.plot(xlist, run.fit_func(xlist, *run.popt), color=color, linestyle='--')
		
		ax.legend()
	
	axs[1,0].set(ylim=[-0.05,0.7])
	### Fit freq but scaled heating
	ax = axs[1,1]
	fit_type = fit_freq
	runs = fit_type['run_results']
	xlabel = "omega/EF"
	ylabel = "Contact per atom C/N [kF]"
	xmin = 0
	
	Tilmanx, TilmanZeta = np.loadtxt("zetaomega_T0.58.txt", unpack=True)
	TilmanZeta = TilmanZeta/12	
	
	ax.set(ylabel=ylabel, xlabel=xlabel)
	for run, color in zip(runs, colors):
# 		scale = 1/run.A**2/run.time / 10000
		ax.errorbar(np.array(run.avg_data[fit_type['param']])*h*1000/run.EFmean,
			  np.array(run.avg_data['contact']), 
			  yerr = np.array(run.avg_data['em_contact']), label=run.label, 
			  color=color, capsize=2, fmt='o')
		xlist = np.linspace(xmin, run.data[fit_type['param']].max(), num)
# 		ax.plot(xlist*h*1000/run.EFmean, 
# 		  scale*run.fit_func(xlist, *run.popt), color=color, linestyle='--')
		
# 	ax.plot(Tilmanx, ToTFfunc_tilman(Tilmanx*run.EFmean/hbar, 
#  				   TilmanZeta, run.EFmean)/10000/2/pi, color="black", linestyle='--')
		
	
	ax.legend()
	fig.tight_layout()
	plt.show()
	