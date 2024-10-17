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
from library import h, pi, tint_shade_color, tintshade, \
				colors, markers, hbar, plt_settings
import numpy as np
import matplotlib.pyplot as plt

from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq


def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer


# Vpp calibration
OmegaRat2p3VVA = 17.05
VppFrom2p3VVA = 0.70725
VpptoOmegaR = 24107  # Hz

file = "2024-09-24_C_e.dat"
xname = 'freq'  # is in MHz

ff = 1.03
ODTscale = 1.0
ToTF = 0.42
EF = 15.e3  # Hz
EFMHz = EF/1e6
trf = 0.4e-3  # s
FourierWidth = 2/(trf*1e6)  # MHz
res75= 47.2227
trap_depth = 0.2  # MHz

run = Data(file)
# cyccutoff = 100
# run.data = run.data[run.data['cyc']>cyccutoff]
num = len(run.data[xname])

### compute bg c5, transfer, Rabi freq, etc.
# fudge c9 value based on calibration
run.data['c9'] = run.data['c9']*ff

bgc5 = run.data[(run.data[xname]-res75)< -3*FourierWidth]['c5'].mean()
# bgc5 = 0
bgc9 = run.data[(run.data[xname]-res75) < -3*FourierWidth]['c9'].mean()

# if no bg point can use what's below
# c5res = run.data[run.data[xname] == res75]['c5'].mean()
# c9res = run.data[run.data[xname] == res75]['c9'].mean()
# bgc9 = c5res + c9res
# bgc5 = 0
bgtot = bgc5 + bgc9
# bgtot = bgc9

run.data['N'] = run.data['c5']-bgc5*np.ones(num)+run.data['c9']

# try: # sigh
# 	run.data['VVA'] = run.data['vva']
# except KeyError():
# 	run.data['vva'] = run.data['VVA']
	
run.data['transfer'] = (run.data['c5'] - bgc5*np.ones(num))/run.data['N']

run.data['loss'] = (bgc9 - run.data['c9'])/bgc9
run.data['OmegaR'] = run.data.apply(lambda x: 2*pi*np.sqrt(0.3)*VpptoOmegaR* \
						Vpp_from_VVAfreq(x['VVA'], x['freq']), axis=1)

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

run.group_by_mean('freq')


###### plotting ##########
### plot settings
plt.rcParams.update(plt_settings) # from library.py
plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})

fig, axs = plt.subplots(2,3)
fig.suptitle(f"{file} with {ODTscale}ODTs, trf={trf*1e6}us, EF={EF/1e3}kHz, ToTF={ToTF}")
xlabel = 'detuning'

ynamesets = [["N"],
			 ["c5"], 
			 ["c9"], 
			 ["transfer", "loss"],
			 ["ScaledTransfer", "ScaledLoss"], 
			 ["Contact", "ContactLoss"]]
ylabels = ["Total Atom Number",
 		   "C5",
 		   "C9",
		   "Transfer or Loss",
		   "Scaled Transfser (arb.)",
		   "Contact (N kF)"]

plotname = 'detuning'
xmin = -0.02
xmax = run.data[plotname].max()

i = 0
for ax, ynameset, ylabel in zip(axs.flatten(), ynamesets, ylabels):
	ax.set(xlabel=xlabel, ylabel=ylabel, xlim=[xmin, xmax])
	
	if i == 4:
		ax.set(xlim=[-0.025, 0.025])
	
	for yname, color, marker in zip(ynameset, colors, markers):
		light_color = tint_shade_color(color, amount=1+tintshade)
		dark_color = tint_shade_color(color, amount=1-tintshade)
		ax.errorbar(run.avg_data[plotname], run.avg_data[yname], marker=marker,
				 yerr=run.avg_data['em_'+yname], markeredgecolor=dark_color, 
				 markerfacecolor=light_color, color=dark_color, label=yname)
	ax.legend()
	i += 1
		
fig.tight_layout()

