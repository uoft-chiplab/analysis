# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 12:33:54 2025

@author: coldatoms

This code is designed to analyze data taken on 2024-07-08_H where 7-to-5 
free-to-free resonant transfer was measured using square pulses of various
pulse times. The shortest pulse (40us) is theorized to give a decent estimate
of the sum rule, which is mostly constrained to the near-resonant regime.
"""
import sys
import os
# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')
if root not in sys.path:
	sys.path.append(root)
from data_class import Data
from scipy.optimize import curve_fit
from library import plt_settings, markers, colors, tint_shade_color, tintshade, pi, GammaTilde, h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# switches
CORRECT_AC_LOSS = True
### ac loss corrections
# these are from varying jump page results
# see diagnostics/
ToTFs = [0.26, 0.36, 0.6, 1.1]
corr_cs = [1.00, 1.15, 1.31, 1.31]
e_corr_cs = [0.05, 0.06, 0.08, 0.08]

corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(corr_cs))
e_corr_c_interp = lambda x: np.interp(x, np.array(ToTFs), np.array(e_corr_cs))
	

# filename
filename = '2024-07-08_H_e.dat'
bg_filename = '2024-07-08_J_e.dat' # run H did not have a bg shot point, needed different bg run
bg_data = Data(bg_filename, path=data_path)
fig, ax = plt.subplots(2)
ax[0].plot(bg_data.data.c9)
ax[1].plot(bg_data.data.c5)
bg_c9 = bg_data.data.c9.mean()
bg_c5 = bg_data.data.c5.mean()
ff = 1.03

# estimate EF
wbar = (170*400*400)**(1/3)
N = 64415
EF = (6*N/2)**(1/3) * wbar  # Hz
ToTF = 0.6 # estimated

### Vpp calibration
VpptoOmegaR = 17.05/0.703  # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
runVpp = 0.032 # V, as stated by file name
pulsearea = 1 # square
phaseO_OmegaR = 2*pi*VpptoOmegaR*runVpp * pulsearea
OmegaR_HFT = phaseO_OmegaR

# open file and calculate parameters
run = Data(filename, path=data_path)

run.data['c9'] = run.data['c9'] 
run.data['c5'] = run.data['c5'] - bg_c5
if CORRECT_AC_LOSS:
	corr_c = corr_c_interp(ToTF)
	print(f'corr_c = {corr_c}')
	run.data['c5'] = run.data['c5']*corr_c
	run.data['e_ac_loss'] = e_corr_c_interp(ToTF)
run.data['sum95'] = run.data['c5'] + run.data['c9']
run.data['transfer'] = run.data['c5']/(run.data['c5']+run.data['c9']*ff)
run.data['scaledtransfer'] = GammaTilde(run.data['transfer'], h*EF, OmegaR_HFT*1e3, 
							  run.data['time']/1e3)
run.data['SW'] = run.data['scaledtransfer']/(run.data['time']/1e3)/EF # norm to 0.5
run.data['Ires'] = 4*run.data['transfer']/(OmegaR_HFT*1e3)**2 / (run.data['time']/1e3)**2
OmegaR2t2 = (OmegaR_HFT*1e3)**2 * (run.data.time.min()/1e3)**2
shortest_time_transfer = run.data[run.data['time'] == 0.04]['transfer'].values[0]
SW_alt = 4*shortest_time_transfer/OmegaR2t2
print(f'4alpha/OmegaR2t2 = {SW_alt}')

# averaging
run.group_by_mean('time')

# plotting
### plot settings
plt.rcParams.update(plt_settings) # from library.py
plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 20,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 2})

plot_params = ['c9','c5','sum95','transfer','scaledtransfer', 'Ires']
plot_names = [r'$N_a$', r'$N_c$', r'$N_a+N_c$', r'$\alpha$', r'$\widetilde{\Gamma}$', r'$4\alpha/\Omega_R^2t^2$']
xparam = 'time'
fig, axs = plt.subplots(2,3)
ax = axs.flatten()
for i in range(0, len(plot_params)):
	x = run.avg_data[xparam]
	y = run.avg_data[plot_params[i]]
	yerr = run.avg_data['em_' + plot_params[i]]
	ax[i].errorbar(x, y, yerr, marker=markers[i], color=colors[i])
	ax[i].set(xlabel='time [ms]',
		   ylabel=plot_names[i])
	
if CORRECT_AC_LOSS:
	title_post = ' w/ ac p-wave loss corr.'
else:
	title_post = ''
fig.suptitle(filename + title_post)
fig.tight_layout()