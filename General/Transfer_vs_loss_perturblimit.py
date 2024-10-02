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

### Vpp calibration
VpptoOmegaR = 27050 # Hz

file = "2024-09-17_C_e.dat"
xname = 'VVA'
ff = 1.02
ODTscale = 1.5
ToTF = 0.495
EF = 23.5e3 # Hz
trf = 0.2e-3 # s
res75 = 47.2227
trap_depth = 0.3 #MHz

run = Data(file)

### compute bg c5, transfer, Rabi freq, etc.
# fudge c9 value based on calibration
run.data['c9'] = run.data['c9']*ff
bgc5 = run.data[run.data[xname] == 1]['c5'].mean()
bgc9 = run.data[run.data[xname] == 1]['c9'].mean()
bgtot = bgc5 + bgc9

run.data = run.data[run.data['VVA']!=1] # have to remove the background point now
num = len(run.data[xname])

run.data['N'] = run.data['c5']-bgc5*np.ones(num)+run.data['c9']
run.data['transfer'] = (run.data['c5'] - bgc5*np.ones(num))/run.data['N']

run.data['loss'] = (bgc9 - run.data['c9'])/bgc9
run.data['OmegaR'] = 2*pi * np.sqrt(0.3) *Vpp_from_VVAfreq(run.data['VVA'], 47.3227)*VpptoOmegaR
run.data['OmegaR2'] = run.data['OmegaR']**2 / 1e12 # 1/us^2

run.data['ScaledTransfer'] = run.data.apply(lambda x: GammaTilde(x['transfer'],
								h*EF, x['OmegaR'], trf), axis=1)

run.data['ScaledLoss'] = run.data.apply(lambda x: GammaTilde(x['loss'],
								h*EF, x['OmegaR'], trf), axis=1)

run.data['detuning'] = run.data['freq']-res75
run.data['Delta'] = run.data['detuning']/(EF/1e6)

run.data['Contact'] = run.data.apply(lambda x: x['Delta']**(3/2)*pi**2 * \
									 np.sqrt(2) * 2*x['ScaledTransfer'], axis=1)
	
run.data['ContactLoss'] = run.data.apply(lambda x: x['Delta']**(3/2)*pi**2 * \
									 np.sqrt(2) * 2* x['ScaledLoss'], axis=1)

run.group_by_mean(xname)

##### FITTING #######
# FIT IT
def Linear(x,m,b):
 	return m*x + b

def Quadratic(x, a, b, c):
 	return a*x**2 + b*x + c

cutoff = 0.2 # cut on transfer fraction
xvals = run.avg_data[run.avg_data['transfer'] < cutoff]['VVA']
ytrans = run.avg_data[run.avg_data['transfer'] < cutoff]['transfer']
yloss = run.avg_data[run.avg_data['transfer'] < cutoff]['loss']

guess = [0,0,0]
func=Quadratic
poptt, pcovt = curve_fit(func,xvals,ytrans, p0=guess)
poptl, pvocl = curve_fit(func,xvals,yloss, p0=guess)

xlist = np.linspace(min(xvals),10,1000)

# plot residuals as inset
yrestrans = ytrans - func(xvals, *poptt)
yresloss = yloss - func(xvals, *poptl)

###### plotting ##########
### plot settings
plt.rcParams.update(plt_settings) # from library.py
plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})

fig, axs = plt.subplots(2,3)
fig.suptitle(f"{file} with {ODTscale}ODTs, trf={trf*1e6}us, EF={EF/1e3}kHz, ToTF={ToTF}")
xlabel='VVA'

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

xmin = -0.02
xmax = 0.022
i = 0
for ax, ynameset, ylabel in zip(axs.flatten(), ynamesets, ylabels):
	ax.set(xlabel=xlabel, ylabel=ylabel)
	if i == 3:
		ax.plot(xlist,func(xlist,*poptt),linestyle='-', marker='')
		ax.plot(xlist, func(xlist, *poptl),linestyle='--',marker='')
	if i == 4: # contact
		ax.set(xlim=[1,4], ylim=[-0.05, cutoff])
		ax.plot(xlist,func(xlist,*poptt),linestyle='-', marker='')
		ax.plot(xlist, func(xlist, *poptl),linestyle='--',marker='')
	for yname, color, marker in zip(ynameset, colors, markers):
		light_color = tint_shade_color(color, amount=1+tintshade)
		dark_color = tint_shade_color(color, amount=1-tintshade)
		ax.errorbar(run.avg_data['VVA'], run.avg_data[yname], marker=marker,
				 yerr=run.avg_data['em_'+yname], markeredgecolor=dark_color, 
				 markerfacecolor=light_color, color=dark_color, label=yname)
	ax.legend()
	i += 1
		
axres = axs[1,2]
axres.plot(xvals, yrestrans)
axres.plot(xvals, yresloss)

fig.tight_layout()