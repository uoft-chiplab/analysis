# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:23:06 2025

@author: Chip Lab
"""

from library import styles, pi, colors
from data_class import Data

from fit_functions import RabiFreq
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq

import matplotlib.pyplot as plt
import numpy as np

# old Rabi oscillation file for 47.2227 MHz 7 to 5
file = "2024-09-16_D_e.dat"
xname = 'pulse time'
yname = 'fraction95'
guess = [1, 16, 0, 0]

fit_func, _, _ = RabiFreq([])

ac_loss = 1.4
# fudged
ac_loss = 1.6
#fudged for KX, July 2025, while looking at square pulse run M
ac_loss = 1.31

run = Data(file)
run.data['c9'] = run.data['c9'] * 0.93
run.data['N'] = run.data['c5'] + run.data['c9']
run.data['fraction95'] = run.data['c5']/(run.data['c5'] + run.data['c9'])
run.fit(RabiFreq, [xname, yname], guess=guess)
xs = np.linspace(min(run.data[xname]), max(run.data[xname]), 100)
# run.ax.plot(xs, fit_func(xs, *guess), '--')

OmegaR = run.popt[1] * 2 * pi
run.data['OmegaR'] = OmegaR
run.data['OmegaR2t2'] = run.data[xname]**2 * OmegaR**2

t_max = 1/run.popt[1]
df_osc = run.data.loc[run.data[xname] < t_max*1e3]


# recent on peak square pulse vary OmegaR 7 to 5
file = '2025-04-16_M_e.dat'
run = Data(file)
fudge = 0.91
bg_cutoff = 1
bg_data = run.data[run.data['VVA']< bg_cutoff]
bg_c5 = bg_data['c5'].mean()
bg_c9 = bg_data['c9'].mean()
print(f'bg_c5 = {bg_c5}')
run.data = run.data[run.data['VVA']>bg_cutoff]
run.data['c9'] = run.data['c9'] * fudge
run.data['c5'] = run.data['c5'] * ac_loss
run.data['N'] = run.data['c5'] + run.data['c9']
run.data['fraction95'] = (run.data['c5'] - bg_c5)/ \
    ((run.data['c5']-bg_c5) + run.data['c9'])
RabiperVpp_47MHz_2025 = 12.01/0.452 # 2025-02-12
run.data['OmegaR'] = Vpp_from_VVAfreq(run.data['VVA'], 47.2227) * \
							RabiperVpp_47MHz_2025 * 2 * pi
run.data['OmegaR2t2'] = 0.01**2 * run.data['OmegaR']**2

# looking at raw data
fig, axs = plt.subplots(3, figsize=(10, 8))
axs[0].plot(run.data['OmegaR2t2'], run.data['c9'])
axs[0].hlines(bg_c9, xmin=0, xmax=run.data['OmegaR2t2'].max(), color='red', ls='--')
axs[0].set(xlabel = r'$\Omega_R^2t^2$',
           ylabel = r'$N_b$',
           xlim=[0, 20])
axs[1].plot(run.data['OmegaR2t2'], run.data['c5'])
axs[1].hlines(bg_c5, xmin=0, xmax=run.data['OmegaR2t2'].max(), color='red', ls='--')
axs[1].set(xlabel = r'$\Omega_R^2t^2$',
           ylabel = r'$N_c$',
           xlim=[0,20])
axs[2].plot(run.data['OmegaR2t2'], run.data['fraction95'])
axs[2].set(xlabel = r'$\Omega_R^2t^2$',
           ylabel = r'$\alpha$',
           xlim=[0,20])
fig.tight_layout()
# checking for SW on resonance -- July 2025 addition
print(run.data['VVA'])
run.data['SW'] = 4*run.data['fraction95'] / run.data['OmegaR2t2']
print(run.data['SW'])
# bg_data = run.data[run.data['VVA'] == 0]
# run.data = run.data[run.data['VVA']>0]
# subtracted_data_SW = run.data['SW'] - bg_data['SW'].mean()
# subtracted_data_OmegaR2t2 = run.data['OmegaR2t2'] - bg_data['OmegaR2t2'].mean()
run.group_by_mean('OmegaR2t2')

ylabel = r'$4\alpha/\Omega_R^2t^2$'
xlabel = r'$\Omega_R^2t^2$'
title = '2025-04-16_M 10us sq pulses on res'
fig, ax = plt.subplots()
ax.errorbar(run.avg_data['OmegaR2t2'], run.avg_data['SW'], yerr = run.avg_data['em_SW'])
ax.plot(run.data['OmegaR2t2'], run.data['SW'], mfc='white', mec='black')
#ax.plot(bg_data['OmegaR2t2'].mean(), bg_data['SW'].mean(), color = 'green')
ax.set(xlim=[-1, 20],
        ylim=[0, 1],
    ylabel = ylabel,
    xlabel = xlabel,
    title = title
       )
fig, ax = plt.subplots()
ax.errorbar(run.avg_data['OmegaR2t2'], run.avg_data['SW'], yerr = run.avg_data['em_SW'])
ax.plot(run.data['OmegaR2t2'], run.data['SW'], mfc='white', mec='black')
# ax.plot(subtracted_data_OmegaR2t2, subtracted_data_SW, color = 'green')
ax.set(xlim=[-0.1,2.5],
       ylim = [0.1, 3],
       yscale='log',
        ylabel = ylabel,
    xlabel = xlabel,
    title = title
       #ylim=[0, 1]
       )
#ax.hlines(y=0.5, xmin=0, xmax=10, ls='--', color='red')


# df_sq = run.data.loc[run.data['OmegaR2t2'] < 1e3]

# # recent on peak blackman pulse vary OmegaR 7 to 5
# file = '2025-04-16_K_e.dat'
# run = Data(file)
# run.data['c9'] = run.data['c9'] * 0.91
# run.data['c5'] = run.data['c5'] * ac_loss
# run.data['N'] = run.data['c5'] + run.data['c9']
# run.data['fraction95'] = run.data['c5']/(run.data['c5'] + run.data['c9'])
# RabiperVpp_47MHz_2025 = 12.01/0.452 # 2025-02-12

# # fudge by sqrt(0.31) for blackman square pulse area
# run.data['OmegaR'] = Vpp_from_VVAfreq(run.data['VVA'], 47.2227) * \
# 					RabiperVpp_47MHz_2025 * 2 * pi * 0.42 #* np.sqrt(0.31)
# run.data['OmegaR2t2'] = 0.01**2 * run.data['OmegaR']**2

# df_bm = run.data.loc[run.data['OmegaR2t2'] < 1e3]

# fig, axes = plt.subplots(2,2, figsize=(6.3,6))
# axs = axes.flatten()

# ax = axs[0]
# ax.set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'Transfer $\alpha$', ylim=[-0.05, 1.20],
# 	   xlim=[0, 15])
# ax.plot(df_osc['OmegaR2t2'], df_osc['fraction95'], label='Rabi osc. data', **styles[0])
# ax.plot(df_sq['OmegaR2t2'], df_sq['fraction95'], label='10us sq.', **styles[1])
# ax.plot(df_bm['OmegaR2t2'], df_bm['fraction95'], label='10us bm', **styles[2])
# ts = np.linspace(0, t_max*np.sqrt(8/10), 100)
# ax.plot(ts**2*OmegaR**2, np.sin(OmegaR/2*ts)**2, '-', color=colors[0])
# ax.plot(ts**2*OmegaR**2, ts**2*OmegaR**2/4, '--', label='lin. resp.', color=colors[1])
# ax.legend()


# ax = axs[1]
# ax.set(xlabel=r'$\alpha$', ylabel='N')
# ax.plot(df_osc['fraction95'], df_osc['N'], label='Rabi osc. data', **styles[0])
# ax.plot(df_sq['fraction95'], df_sq['N'], label='10us sq.', **styles[1])
# ax.plot(df_bm['fraction95'], df_bm['N'], label='10us bm', **styles[2])


# ax = axs[2]
# ax.set(xlabel=r'$\Omega_R t$', ylabel=r'$\alpha$', xlim=[-0.2, 2])
# ax.plot(df_osc['OmegaR'] * df_osc['pulse time'], df_osc['fraction95'], label='Rabi osc. data', **styles[0])
# ax.plot(df_sq['OmegaR'] * 0.01, df_sq['fraction95'], label='10us sq.', **styles[1])
# ax.plot(df_bm['OmegaR'] * 0.01, df_bm['fraction95'], label='10us bm', **styles[2])

# fig.suptitle("Non-interacting, resonant transfer")
# fig.tight_layout()
# plt.show()

