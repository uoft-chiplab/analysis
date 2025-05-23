# -*- coding: utf-8 -*-
"""
Created on 26/Mar/2025
This script analyzes ac p-wave loss.
@author: coldatoms
"""
# paths

import os
import sys
# this is a hack to access modules in the parent directory
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going two levels up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
if parent_dir not in sys.path:
	sys.path.append(parent_dir)

data_path = os.path.join(parent_dir, 'clockshift\\data\\ac_loss')

from library import styles, colors
from data_class import Data
from scipy.optimize import curve_fit
from bootstrap_fit import bootstrap_fit, dist_stats

import numpy as np
import matplotlib.pyplot as plt

files = ['2025-03-25_E',
		 '2025-03-25_G',
         '2025-03-26_B',
         '2025-03-26_E',
         '2025-03-26_F']

Ts = [4e-7,
	2e-7,
    4e-7,
    4e-7,
    2e-7] # K, estimates

spins = ['c5', 'c9']

t_until_midsweep = +0.020 + 2.5 + 10 + 2 + 2.5 # ms
t_after_midsweep_before_SSI = 2.5 + 2 + 5 + 0.04 + 0.01 # ms
t_total = t_until_midsweep + t_after_midsweep_before_SSI

def spin_map(spin):
    if spin == 'c5':
        return 'c'
    elif spin == 'c9':
        return  'a'
    elif spin == 'sum95':
        return 'a+c'

def expdecay(x, A, tau, C):
    return A*np.exp(-x/tau) + C

fig, axs = plt.subplots(len(spins), figsize = (6,6))
xcol = 'time'
popt_list = []
perr_list = []

for i, file in enumerate(files):
	
    if i <= 2: continue
    filename = file+'_e.dat'
    df = Data(filename, path=data_path).data
    temp = Ts[i]
	
    for j, spin in enumerate(spins):
        print("*----------------")
        print(f'Processing {file}, spin={spin}')
		
        xx = np.array(df[xcol])
        yy = np.array(df[spin])
		
        p0 = [df[spin].max()-df[spin].min(), 50, df[spin].min()]
		
        popt, pcov = curve_fit(expdecay, df[xcol], df[spin], p0=p0)
        perr = np.sqrt(np.diag(pcov))
		
        if spin == 'c5':
            popt_c5 = popt
            perr_c5 = perr
			
        elif spin == 'c9':
            popt_c9 = popt
            perr_c9 = perr
            amp_ratio = popt_c5[0]/popt_c9[0]
            e_amp_ratio = amp_ratio*np.sqrt((perr_c5[0]/popt_c5[0])**2 + (perr_c9[0]/popt_c9[0])**2)
            print(f'Ratio of c/a decay amplitudes is {amp_ratio:.2f}({e_amp_ratio:.0f})')
			
		
        xs = np.linspace(df[xcol].min(), df[xcol].max(), 300)
        ys = expdecay(xs, *popt)

        zerotime = expdecay(0, *popt)
        if spin == 'c5':
            waittime = t_total
        elif spin == 'c9':
            waittime = t_after_midsweep_before_SSI
        losscorr = expdecay(waittime, *popt)/zerotime
		
        print(f'For {file}, spin={spin}, T={temp*1e9} nK, fit params are {popt}.')
        print(f'{spin_map(spin)} spin loss after {waittime:.2f} ms is {losscorr:.2f} of initial.')
		
        label = f'T={temp*1e9} nK, amp={popt[0]:.1f}({perr[0]:.1f}), tau={popt[1]:.1f}({perr[1]:.1f})'
        axs[j].plot(df[xcol], df[spin], label=label, **styles[i])
        axs[j].plot(xs, ys, color=colors[i], linestyle='--', marker='')
		
		###
		### Bootstrapping
		###
		
        popts, popt_stats_dicts = bootstrap_fit(expdecay, xx, yy, p0=p0, trials=1000)
		
		# tau stats
        tau_median = popt_stats_dicts[1]['median']
        tau_u = popt_stats_dicts[1]['upper']
        tau_l = popt_stats_dicts[1]['lower']
		
		# get medians for use
        BS_popt = [stats_dict['median'] for stats_dict in popt_stats_dicts]
		
		# amplitude cancels here, but offset does not. Ignore error in offset
        zerotime = expdecay(0, *BS_popt)
        losscorr_median = expdecay(waittime, *BS_popt)/zerotime
		
        losscorr_u = expdecay(waittime, BS_popt[0], tau_u, BS_popt[2])/zerotime
        losscorr_l = expdecay(waittime, BS_popt[0], tau_l, BS_popt[2])/zerotime
		
        if spin=='c5':
            print("Bootstrap results:")
            tau_eu = tau_u - tau_median
            tau_el = tau_median - tau_l		
            print(f'tau is {tau_median:.0f}+{tau_eu:.0f}-{tau_el:.0f}')
            losscorr_eu = losscorr_u - losscorr_median
            losscorr_el = losscorr_median - losscorr_l		
            print(f'losscorr is {losscorr_median:.2f}+{losscorr_eu:.2f}-{losscorr_el:.2f}')
			
            losscorr_dist = []
            for popt in popts:
                zerotime_pop = expdecay(0, *popt)
                waittime_pop = expdecay(waittime, *popt)
                losscorr = waittime_pop/zerotime_pop
                losscorr_dist.append(losscorr)
                
            losscorr_stats = dist_stats(losscorr_dist)
            losscorr_median = losscorr_stats['median']
            losscorr_u = losscorr_stats['upper']
            losscorr_l = losscorr_stats['lower']

            losscorr_eu = losscorr_u - losscorr_median
            losscorr_el = losscorr_median - losscorr_l
            print("Proper error analysis of offset param:")
            print(f'losscorr is {losscorr_median:.2f}+{losscorr_eu:.2f}-{losscorr_el:.2f}')
        

# some other plotting tests
for i in range(len(axs)):
    axs[i].set(title=f'{spin_map(spins[i])} spin loss', xlabel='Time [ms]',
			   ylabel='Atom Number [arb.]')
    axs[i].legend()

fig.tight_layout()
plt.show()