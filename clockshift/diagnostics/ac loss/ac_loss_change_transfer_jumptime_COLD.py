# -*- coding: utf-8 -*-
"""
Created on Apr 172025

@author: Chip lab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_class import Data
from library import styles, colors
from scipy.optimize import curve_fit

files = [
# 		  '2025-04-17_D_e_time=3.dat',
# 		  '2025-04-17_D_e_time=14.dat',
# 		  '2025-04-17_D_e_time=25.dat',
# 		  '2025-04-17_D_e_time=50.dat',
  		  # '2025-04-17_F_e_time=3.dat',
		  '2025-04-17_F_e_time=14.dat',
		  '2025-04-17_F_e_time=25.dat',
		  '2025-04-17_F_e_time=50.dat',
# 		  '2025-04-17_F_e_time=100.dat',
 		 ]

def linear(x, a, b):
	return a*x + b

def exp_decay(x, A, B, C):
	return A*np.exp(-x/B) + C

def exp_decay_0(x, A, B):
	return A*np.exp(-x/B)

	 
# from 2025-04-14, i.e. today
ff = 0.91
pol = 1.0

times = [
 	
# 		 15.6, 
		 26.6, 
		 37.6, 
		 63.6,
# 		 112.6,
		 ]

results_list = []

for i, file in enumerate(files):
	run = Data(file)
	
	# correct for fudge factor
	run.data['c9'] = run.data['c9'] * ff
	
	bg_df = run.data.loc[run.data.VVA == 0]
	run.data = run.data.loc[run.data.VVA > 1.5]
	
	VVAs = run.data.VVA.unique()
	
	for VVA in run.data.VVA.unique():
		
		results = {
				'filename': file,
				'index': i,
				'time': times[i],
				'VVA': VVA
					}
			
		# select dfs
		df = run.data.loc[run.data.VVA == VVA]
			
# 		print("bg c5 counts", bg_df['c5'].mean(), bg_df['c5'].std())
	# 	bg_df['c5'] = 0
		
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

dfs = [df_total.loc[df_total.VVA == VVA] for VVA in VVAs]

# plot
fig, axes = plt.subplots(2,2, figsize=(10,8))
axs = axes.flatten()

fig.suptitle(file[0:10]+" Run F, 0.36 ToTF, vary transfered fraction")

xlabel = 'Time after jump (ms)'
axs[0].set(xlabel=xlabel, ylabel=r'Atom number $N_c$')
axs[1].set(xlabel=xlabel, ylabel=r'Transfer $\alpha$')#, ylim=[0.085, 0.1])
axs[2].set(xlabel=xlabel, ylabel=r'Atom number $N/N_{bg}$')

corrs = []
e_corrs = []
taus = []
e_taus = []
alpha_0s = []

for i, df in enumerate(dfs):
	# Nc
	ax = axs[0]
	
	x = df['time']
	y = df['c']
	yerr = df['em_c']
	
# 	label = f'VVA = {VVA}'

	ax.errorbar(x, y, yerr, **styles[i]
# 	, label=label
	)
	
	fit_func = exp_decay_0
	p0 = [y.max(), 120]
	xs = np.linspace(0, max(x), 100)
	popt, pcov = curve_fit(fit_func, x, y, p0=p0)
	perr = np.sqrt(np.diag(pcov))

	ax.plot(xs, fit_func(xs, *popt), '--', color=colors[i])
	ax.legend()
	
	tau = popt[1]
	e_tau = perr[1]
	corr = fit_func(0, *popt)/fit_func(26.6, *popt)
	e_corr = corr * 26.6 * perr[1]/popt[1]**2
	
	print("tau is {:.3f}({:.0f})".format(tau, 1e3*e_tau))
	print("corrective factor is {:.3f}({:.0f})".format(corr, e_corr*1e3))
# 	print("adjusted corrective factor is {:.3f}({:.0f})".format(corr_adj, e_corr*1e3))
	
	# tranfer
	ax = axs[1]
	y = df['transfer']
	yerr = df['em_transfer']
	ax.errorbar(x, y, yerr, **styles[i])
	
	fit_func = exp_decay_0
	p0 = [y.max(), 120]
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

# ax = axs[3]
# ax.errorbar(np.array(alpha_0s), np.array(taus)*(1-np.array(alpha_0s)), 
# 			yerr=abs(np.array(e_taus)*(1-np.array(alpha_0s))), **styles[0])
# ax.set(xlabel=r'$\alpha_0$', ylabel=r'$\tau(1-\alpha_0)$ (ms)')

fig.tight_layout()

plt.show()

# calculate mean and std
np.mean(corrs[1:-1])
np.std(corrs[1:-1])

# # plot loss results
# Ts = np.array(ToTFs) * np.array(EFs)
# scaled_taus = np.array(taus) *Ts**2 * np.array(EFs)**(2/3) * np.array(alpha_0s)
# scaled_taus2 = np.array(taus)/( Ts**2 * np.array(EFs)**(1/3))

# fig, axes = plt.subplots(2,2, figsize=(8,8))
# axs = axes.flatten()

# ax = axs[0]
# ax.plot(Ts, taus, **styles[0])
# ax.set(xlabel=r"Temperature, $T$ (kHz)", ylabel=r'$\tau$ (ms)')

# ax = axs[1]
# ax.plot(Ts**2 * np.array(EFs)**(2/3) * np.array(alpha_0s), taus, **styles[0])
# ax.set(xlabel=r"$T^2 EF^{2/3} \alpha_0$ (arb.)", ylabel=r'$\tau$ (ms)')

# ax = axs[2]
# ax.plot(Ts**2 * np.array(EFs)**(1/3), taus, **styles[0])
# ax.set(xlabel=r"$T^2 EF^{1/3}$ (arb.)", ylabel=r'$\tau$ (ms)')

# ax = axs[3]
# ax.plot(Ts, scaled_taus2, **styles[0])
# ax.set(xlabel=r"$T$", ylabel=r'$\tau / EF^{1/3} / T^2$ (arb.)', ylim=[-1,2])

# fig.tight_layout()
# plt.show()
