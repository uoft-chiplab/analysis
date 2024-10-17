# -*- coding: utf-8 -*-
"""
@author: coldatoms
"""
# paths
import os
import sys
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
if root not in sys.path:
	sys.path.insert(0, root)

from data_class import Data
from library import GammaTilde, h, pi, tint_shade_color, tintshade, \
				colors, markers, hbar, plt_settings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq

def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

# FIT IT
def Linear(x,m,b):
 	return m*x + b

def Quadratic(x, a, b, c):
 	return a*x**2 + b*x + c

### Vpp calibration
VpptoOmegaR = 27050 # Hz

file_start = "2024-09-18_D_e_freq="
freqs = [47.2427,
		 47.2727]
# 		 47.4227,
# 		 47.5227,
# 		 48.2227]

bg_freq = 46.2227
xname = 'VVA'
ff = 1.02
ODTscale = 1.5
ToTF = 0.5
EF = 23.5e3 # Hz
trf = 0.2e-3 # s
res75 = 47.2227
trap_depth = 0.3 #MHz

### compute background atom numbers
bg_file = file_start + f"{bg_freq}"+".dat"
run = Data(file)
bgc5 = run.data['c5'].mean()
bgc9 = run.data['c9'].mean()
bgtot = bgc5 + bgc9

xmin = -0.02
xmax = 0.022

cutoffs = 0.2*np.ones(len(freqs))

ylims = [-0.05, 0.3]

###### plotting ##########
### plot settings
plt.rcParams.update(plt_settings) # from library.py
plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})

fig, axs = plt.subplots(2,3)
fig.suptitle(f"{file} with {ODTscale}ODTs, trf={trf*1e6}us, EF={EF/1e3}kHz, ToTF={ToTF}")
xlabel='OmegaR2'

ynamesets = [
			 ["N"],
			 ["c5"], 
			 ["c9"], 
			 ["transfer", "loss"],

			  ["transfer", "loss"]
]
ylabels = [
		   "Total Atom Number",
 		   "C5",
 		   "C9",
		   "Transfer or Loss",
		   "Transfer or Loss"]

### loop over frequency .dat files
j = 0 # freq iter
for freq, cutoff in zip(freqs, cutoffs):
	file = file_start + f"{freq}"+".dat"
	run = Data(file)
	
	### compute transfer, Rabi freq, etc.
	# fudge c9 value based on calibration
	run.data['c9'] = run.data['c9']*ff
	num = len(run.data['c9'])
	run.data['N'] = run.data['c5']-bgc5*np.ones(num)+run.data['c9']
	run.data['transfer'] = (run.data['c5'] - bgc5*np.ones(num))/run.data['N']
	
	run.data['loss'] = (bgc9 - run.data['c9'])/bgc9
	run.data['OmegaR'] = run.data.apply(lambda x: 2*pi * np.sqrt(0.3) * VpptoOmegaR \
									 * Vpp_from_VVAfreq(x['VVA'], freq), axis=1)
	run.data['OmegaR2'] = run.data['OmegaR']**2 / 1e12 # 1/us^2
	
	run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x['transfer'],
									h*EF, x['OmegaR'], trf), axis=1)
	
	run.data['ScaledLoss'] = run.data.apply(lambda x: GammaTilde(x['loss'],
									h*EF, x['OmegaR'], trf), axis=1)
	
	run.data['detuning'] = freq-res75
	run.data['Delta'] = run.data['detuning']/(EF/1e6)
	
	run.data['Contact'] = run.data.apply(lambda x: x['Delta']**(3/2)*pi**2 * \
										 np.sqrt(2) * 2*x['ScaledTransfer'], axis=1)
		
	run.data['ContactLoss'] = run.data.apply(lambda x: x['Delta']**(3/2)*pi**2 * \
										 np.sqrt(2) * 2* x['ScaledLoss'], axis=1)
	
	run.group_by_mean(xname)
	
	##### FITTING #######
	
	cutoff = 0.5
	xvals = run.avg_data[run.avg_data['OmegaR2'] < cutoff]['OmegaR2']
	ytrans = run.avg_data[run.avg_data['OmegaR2'] < cutoff]['transfer']
	yloss = run.avg_data[run.avg_data['OmegaR2'] < cutoff]['loss']
	
	guess = [1, 1]
	poptt, pcovt = curve_fit(Linear,xvals,ytrans, p0=guess)
	poptl, pvocl = curve_fit(Linear,xvals,yloss, p0=guess)
	
	xlist = np.linspace(min(xvals),xmax, num)
	
	# plot residuals as inset
	yrestrans = ytrans - Linear(xvals, *poptt)
	yresloss = yloss - Linear(xvals, *poptl)
	
	light_color = tint_shade_color(colors[j], amount=1+tintshade)
	dark_color = tint_shade_color(colors[j], amount=1-tintshade)
	i = 0 # ax iter
	for ax, ynameset, ylabel in zip(axs.flatten(), ynamesets, ylabels):
		ax.set(xlabel=xlabel, ylabel=ylabel)
		if i == 3:
			ax.set(ylim=[-0.005, 0.5])
			ax.plot(xlist,Linear(xlist,*poptt),linestyle='-', marker='', color=colors[j])
			ax.plot(xlist, Linear(xlist, *poptl),linestyle='--',marker='', color=colors[j])
# 		if i == 4: # contact
# 			ax.set(xlim=[ ylim=ylims)
		for yname, marker in zip(ynameset, markers):
			ax.errorbar(run.avg_data['OmegaR2'], run.avg_data[yname], marker=marker,
					 yerr=run.avg_data['em_'+yname], markeredgecolor=dark_color, 
					 markerfacecolor=light_color, color=dark_color, label=yname)
		ax.legend()
		i += 1
	
	j += 1
			
	axres = axs[1,2]
	axres.plot(xvals, yrestrans)
	axres.plot(xvals, yresloss)
	
	fig.tight_layout()