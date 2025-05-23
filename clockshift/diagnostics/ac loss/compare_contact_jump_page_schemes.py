# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:32:34 2025

@author: Chip lab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_class import Data
from library import styles, colors
from scipy.optimize import curve_fit

files = [
		  '2025-04-15_I_e_ODT1=0.07.dat',
		  '2025-04-15_I_e_ODT1=0.3.dat',
 		 '2025-04-15_H_e_ODT1=0.07.dat',
		  '2025-04-15_H_e_ODT1=0.3.dat',
		  '2025-04-15_K_e_ODT1=0.07.dat',
		  '2025-04-15_K_e_ODT1=0.3.dat',
		  '2025-04-17_C_e_time=3.dat',
		  '2025-04-17_C_e_time=14.dat',
		  '2025-04-17_C_e_time=25.dat',
 		 ]

times = [15.6, 15.6, 26.6, 26.6, 37.6, 37.6, 15.6, 26.6, 37.6,]
pols = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
ODTs = [0.07, 0.3, 0.07, 0.3, 0.07, 0.3]

def linear(x, a, b):
	return a*x + b

def exp_decay(x, A, B, C):
	return A*np.exp(-x/B) + C

def exp_decay_0(x, A, B):
	return A*np.exp(-x/B)

def map_droptime_to_TOTF(droptime):
	if droptime == 0.3:
		return 0.75
	elif droptime == 0.5:
		return 0.93
	elif droptime == 0.8:
		return 1.1
		 
# from 2025-04-14, i.e. today
ffs = [0.83, 0.83, 0.83, 0.83, 0.83, 0.83,
	   0.91, 0.91, 0.91,]

results_list = []


for i, file in enumerate(files):
	
	pol = pols[i]
	ff = ffs[i]
	
	run = Data(file)
	
	# correct for fudge factor
	run.data['c9'] = run.data['c9'] * ff
	
	try: 
		droptimes = run.data.droptime.unique()
		
	except AttributeError:
		droptimes = [0]
		run.data.droptime = 0
		ODT1 = ODTs[i]
		if ODT1 == 0.07:
			ToTF = 0.22
		elif ODT1 == 0.3:
			ToTF = 0.72
	
	for droptime in droptimes:
		
		results = {
				'filename': file,
				'run': file[11],
				'index': i,
				'time': times[i],
				'pol': pol,
					}
		
		# if not drop, then get temp from 
		if droptime != 0:
			ToTF = map_droptime_to_TOTF(droptime)
			
		results['ToTF'] = ToTF
			
		# select dfs
		df = run.data.loc[(run.data.VVA > 0) & (run.data.droptime == droptime)]
		bg_df = run.data.loc[(run.data.VVA == 0) & (run.data.droptime == droptime)]
			
# 		print("bg c5 counts", bg_df['c5'].mean(), bg_df['c5'].std())
		
		df['a'] = df['c9']
		df['c'] = df['c5'] - bg_df['c5'].mean()
		
		# approximately
		df['b'] = df['a']/pol - df['c']
		
		df['transfer'] = df['c']/(df['a']/pol)
		df['transfer2'] = df['c']/(bg_df['c9'].mean()/pol)
		df['N'] = df['a']
		df['Nbg'] = bg_df['c9'].mean()
		df['NoNbg'] = df['N']/df['Nbg']
			
			
		results['a'] = df['a'].mean()
		results['em_a'] = df['a'].sem()
		results['b'] = df['b'].mean()
		results['em_b'] = df['b'].sem()
		results['c'] = df['c'].mean()
		results['em_c'] = df['c'].sem()
		
		results['transfer'] = df['transfer'].mean()
		results['em_transfer'] = df['transfer'].sem()
		results['transfer2'] = df['transfer2'].mean()
		results['em_transfer2'] = df['transfer2'].sem()
		
		results['N'] = df['N'].mean()
		results['em_N'] = df['N'].sem()
		results['NoNbg'] = df['NoNbg'].mean()
		results['em_NoNbg'] = df['NoNbg'].sem()
		
		
		results_list.append(results)
	

df_total = pd.DataFrame(results_list)

ToTFs = [0.22, 0.72, 0.75, 0.93, 1.1]

dfs = [df_total.loc[df_total.ToTF == ToTF] for ToTF in ToTFs]

# plot
fig, axes = plt.subplots(2,2, figsize=(10,8))
axs = axes.flatten()

fig.suptitle(file[0:10]+" Runs H, I and K")

xlabel = 'Time after jump (ms)'
axs[0].set(xlabel=xlabel, ylabel=r'Atom number $N_c$')
axs[1].set(xlabel=xlabel, ylabel=r'Transfer $\alpha$')#, ylim=[0.085, 0.1])
axs[2].set(xlabel=xlabel, ylabel=r'Atom number $N/N_{bg}$')#, ylim=[0.085, 0.1])
	

# ToTFs = []
EFs = [11.8, 17.8, 16.7, 16.7, 16.7]
corrs = []
e_corrs = []
taus = []
e_taus = []
alpha_0s = []
for i, df in enumerate(dfs):
	
	ToTF = df.ToTF.unique()[0]
	print("ToTF = ", ToTF)
	label = "ToTF = {:.2f}".format(ToTF)
	
	# Nc
	ax = axs[0]
	
	x = df['time']
	y = df['c']
	yerr = df['em_c']
	
	ax.errorbar(x, y, yerr, **styles[i], label=label)
	
	fit_func = exp_decay_0
	p0 = [y.max(), 84]
	xs = np.linspace(0, max(x), 100)
	popt, pcov = curve_fit(fit_func, x, y, p0=p0)
	perr = np.sqrt(np.diag(pcov))

	ax.plot(xs, fit_func(xs, *popt), '--', color=colors[i])
	
	Nc0 = popt[0]
	tau = popt[1]
	e_tau = perr[1]
	corr = Nc0/fit_func(26.6, *popt)
	
	e_corr = corr * 26.6 * perr[1]/popt[1]**2
	print("tau is {:.0f}({:.0f}) ms".format(tau, e_tau))
	print("corrective factor is {:.3f}({:.0f})".format(corr, e_corr*1e3))
	
	
	ax.legend()
	
	# transfer
	ax = axs[1]
	y = df['transfer']
	yerr = df['em_transfer']
	
	ax.errorbar(x, y, yerr, **styles[i])#, label=label)
	
	fit_func = exp_decay_0
	p0 = [y.max(), 84]
	xs = np.linspace(0, max(x), 100)
	popt, pcov = curve_fit(fit_func, x, y, p0=p0)
	perr = np.sqrt(np.diag(pcov))

	ax.plot(xs, fit_func(xs, *popt), '--', color=colors[i])
	alpha_0 = fit_func(0, *popt)
	e_alpha_0 = perr[0]/popt[0] * alpha_0
	print("alpha_0 is {:.3f}({:.0f})".format(alpha_0, 1e3*e_alpha_0))
	
	# N over N bg
	ax = axs[2]
	y = df['NoNbg']
	yerr = df['em_NoNbg']
	ax.errorbar(x, y, yerr, **styles[i])
	
	corrs.append(corr)
	e_corrs.append(e_corr)
	taus.append(tau)
	e_taus.append(e_tau)
	alpha_0s.append(alpha_0)

ax = axs[3]
ax.errorbar(np.array(alpha_0s), np.array(taus), yerr=np.array(e_taus), **styles[0])
ax.set(xlabel=r'$\alpha_0$', ylabel=r'$\tau$ (ms)')

fig.tight_layout()

plt.show()

# calculate pol
bg_df['r95'] = bg_df['c9']/bg_df['c5']
pol = bg_df['r95'].mean()

# plot loss results
Ts = np.array(ToTFs) * np.array(EFs)

fig, axes = plt.subplots(1,2, figsize=(8,4))
axs = axes.flatten()

ax = axs[0]
ax.errorbar(Ts**2, taus, yerr=e_taus, **styles[0])
ax.set(xlabel=r"Temperature^2, $T^2$ (kHz^2)", ylabel=r'$\tau$ (ms)')

rescaled_taus = taus*(Ts**2 * np.array(EFs)**(3/2))
e_rescaled_taus = rescaled_taus/taus * e_taus

ax = axs[1]
ax.errorbar(ToTFs[:], rescaled_taus[:], yerr=e_rescaled_taus[:], **styles[0])
ax.set(xlabel=r'$T/T_F$', ylabel=r"$\tau(T^2 \langle n_a \rangle_{pk})$ (arb.)")


fig.tight_layout()
plt.show()
