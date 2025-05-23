# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:03:52 2025

@author: coldatoms
"""
import os
import sys
#analysis_path = 'E:\\Analysis Scripts\\analysis'
analysis_path = '\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Analysis Scripts\\analysis'
if analysis_path not in sys.path:
	sys.path.append(analysis_path)
from library import styles, hbar, h, pi
from data_class import Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Save = True

files = ["2025-04-23_C_e.dat",
	"2025-04-24_B_e.dat",
      "2025-04-24_C_e.dat"
]
data_path = '\\clockshift\\data\\very_short_dimer_pulse'
full_path = analysis_path + data_path
ff = 0.91
EF = 16 # kHz, estimate
sat_correction = True
VVA_to_Vpp_mapping = {10: 4.20,
						9: 3.96,
						7: 3.3,
						5.6: 2.72,
						5: 2.32,
						3.5: 1.76,
                        2.7: 1.24,
						2.2: 0.880,
                        2: 0.720,
                        1.6:0.400,
						0: 0,
						}

def VVA_to_Vpp(VVA):
	return VVA_to_Vpp_mapping[VVA]

def saturation_scale(x, t):
	""" x is OmegaR^2 and x0 is fit 1/e Omega_R^2 """
	x0 = 5211
	t0 = 0.010
	x = x*t**2
	x0 = x0*t0**2
	return x/x0*1/(1-np.exp(-x/x0))


def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

### Calibrations
RabiperVpp_47MHz_2024 = 17.05/0.728 # 2024-09-16
RabiperVpp_43MHz_2024 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration
RabiperVpp_47MHz_2025 = 12.01/0.452 # 2025-02-12
# Fudge the 2024 based on the 2025/2024 47MHz ratio...
RabiperVpp_43MHz_2025 = RabiperVpp_43MHz_2024 * RabiperVpp_47MHz_2025/ \
													RabiperVpp_47MHz_2024

### plotting
fig, axes = plt.subplots(2,3, figsize=(12,6))
axs = axes.flatten()
results_list = []
for file in files:
    run = Data(file, path=full_path)
    run.data.c9 = run.data.c9 * ff
    run.data.time = run.data.time / 1e3 # convert to ms
    run.data.loc[run.data.VVA == 0, 'time'] = 0
    #run.data = run.data.loc[(run.data['VVA']!=0) & run.data['buffer']!= 21]
    # check consistency of bg shots
    bg_3us = run.data[(run.data.VVA == 0) & (run.data.buffer == 4)]
    bg_20us = run.data[(run.data.VVA == 0) & (run.data.buffer == 21)]
    # print(f'3us bg = {bg_3us['c5'].mean():.1f} +/- {bg_3us['c5'].std():.1f}')
    # print(f'20us bg = {bg_20us['c5'].mean():.1f} +/- {bg_20us['c5'].std():.1f}')

    run.group_by_mean('VVA')

    df = run.avg_data.loc[run.avg_data.VVA > 0]
    bg_df = run.avg_data.loc[run.avg_data.VVA == 0]

    pol = (bg_df.c9/(bg_df.c5+bg_df.c9))[0]

    df = pd.concat([df, bg_df]) # add the "zero time" point

    # calculate OmegaR2
    df['file']= file
    df['Vpp'] = df.apply(lambda x: VVA_to_Vpp(x['VVA']), axis=1)
    df['OmegaR'] = 2*np.pi*RabiperVpp_43MHz_2025*df['Vpp']
    #df['OmegaR2'] = (df['OmegaR']/2/np.pi)**2

    if sat_correction:
        df['sat_correction'] = saturation_scale(df['OmegaR']**2/(2*np.pi)**2, df['time'])
    else :
        df['sat_correction']=1
    # calculate transfer fraction from b population loss, over 2
    df['alpha_b'] = (1-df.c5/bg_df.c5.mean())/2 * df['sat_correction']
    df['em_alpha_b'] = df.c5/bg_df.c5.mean()/2*np.sqrt((df.em_c5/df.c5)**2 + \
                                    (bg_df.em_c5.mean()/bg_df.c5.mean())**2) * \
                                        df['sat_correction']

    # calculate transfer rate from b transfer, in kHz/N_b
    df['Gamma_b'] = df['alpha_b']/(df['time'])
    df['em_Gamma_b'] = df['em_alpha_b']/(df['time'])

    df.loc[df.time == 0, 'Gamma_b'] = 0
    df.loc[df.time == 0, 'em_Gamma_b'] = 0

    # df['scaled_Gamma_b'] = df['Gamma_b']/df['OmegaR2']/EF*2 # since EF is in kHz, check this
    # df['em_scaled_Gamma_b'] = df['em_Gamma_b']/df['OmegaR2']/EF*2 # *2?...

    
    # note that OmegaR should be in kHz to match df['time']
    df['scaled_Gamma_b'] = df['Gamma_b']/(df['OmegaR'])**2/np.pi*(2*np.pi*EF) # since EF is in kHz, check this
    df['em_scaled_Gamma_b'] = df['em_Gamma_b']/(df['OmegaR'])**2/np.pi*(2*np.pi*EF) # *2?...

    # df['scaled_Gamma_b'] = GammaTilde(df['alpha_b'], h*EF*1e3, df['OmegaR']*1e3, df['time']/1e3)
    # df['em_scaled_Gamma_b'] = 0
    # compute spectral weight
    # df['spectral_weight'] = df['scaled_Gamma_b']/df['time']/EF
    # df['em_spectral_weight'] = df['em_scaled_Gamma_b']/df['time']/EF
    df['spectral_weight'] = df['scaled_Gamma_b']/df['time']/EF
    df['em_spectral_weight'] = df['em_scaled_Gamma_b']/df['time']/EF

    transfer = 'alpha_b'
    transfer_rate = 'Gamma_b'
    scaled_transfer_rate = 'scaled_Gamma_b'

    # manual filtering
    # if file == '2025-04-23_C_e.dat':
    #       df = df[df['time']!=0.003]
    # if file == '2025-04-24_B_e.dat':
    #       df = df[df['time'] != 0.020]

    results_list.append(df)

    time_label = r'Pulse Time $t$ (ms)'
    timesq_label = r'Pulse Time$^2$ $t^2$ (ms$^2$)'

    # atoms
    ax = axs[0]
    ax.set(xlabel=time_label, ylabel=r'Atom Number $N_\sigma$')
    ax.errorbar(df['time'], df['c9'], yerr=df['em_c9'], **styles[0], label='a')
    ax.errorbar(df['time'], df['c5'], yerr=df['em_c5'], **styles[1], label='b')
    ax.legend()

    # transfer fraction alpha
    ax = axs[1]
    ax.set(xlabel=time_label, ylabel=r'Transfer $\alpha$')
    ax.errorbar(df['time'], df[transfer], yerr=df['em_'+transfer], **styles[0])

    # transfer rate Gamma
    ax = axs[2]
    ax.set(xlabel=time_label, ylabel=r'Transfer Rate $\Gamma$ (kHz)')
    ax.errorbar(df['time'], df[transfer_rate], yerr=df['em_'+transfer_rate], 
                **styles[2])

    # scaled transfer rate Gamma Tilde
    ax = axs[3]
    ax.set(xlabel=time_label, ylabel=r'Scaled Transfer Rate  $\tilde\Gamma$')
    ax.errorbar(df['time'], df[scaled_transfer_rate], 
                yerr=df['em_'+scaled_transfer_rate], **styles[3])

    # spectral weight
    ax = axs[4]
    ax.set(xlabel=time_label, ylabel=r'Spectral Weight $I_d$',
        ylim=[-0.00, 0.014],
    )
    ax.errorbar(df['time'], df['spectral_weight'], 
                yerr=df['em_spectral_weight'], **styles[4])
    

fig.suptitle(file+', pol={:.2f}'.format(pol))
fig.tight_layout()
plt.show()

results_df = pd.concat(results_list)
results_df = results_df[results_df['time'] != 0]
print(results_df.shape)
fermitime = 1/(2*np.pi*EF)  #ms
results_df['scaledtime'] = results_df['time']/fermitime 
meanId = results_df.groupby('scaledtime')['spectral_weight'].mean()

if Save:
    fig, ax = plt.subplots(figsize=(4,3))
    ax.errorbar(results_df['scaledtime'], results_df['spectral_weight'], 
                    yerr = results_df['em_spectral_weight'], **styles[5])
    ax.set(xlabel=r'$t/\tau_F$', ylabel=r'Spectral Weight $I_d$',
            ylim=[-0.00, 0.015],
            xlim=[-0.1, 5.1])
    results_df.to_excel(os.path.join(full_path, 'results_df.xlsx'))