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
run.data['c9'] = run.data['c9'] * 0.91
run.data['c5'] = run.data['c5'] * ac_loss
run.data['N'] = run.data['c5'] + run.data['c9']
run.data['fraction95'] = run.data['c5']/(run.data['c5'] + run.data['c9'])
RabiperVpp_47MHz_2025 = 12.01/0.452 # 2025-02-12
run.data['OmegaR'] = Vpp_from_VVAfreq(run.data['VVA'], 47.2227) * \
							RabiperVpp_47MHz_2025 * 2 * pi
run.data['OmegaR2t2'] = 0.01**2 * run.data['OmegaR']**2

df_sq = run.data.loc[run.data['OmegaR2t2'] < 1e3]

# recent on peak blackman pulse vary OmegaR 7 to 5
file = '2025-04-16_K_e.dat'
run = Data(file)
run.data['c9'] = run.data['c9'] * 0.91
run.data['c5'] = run.data['c5'] * ac_loss
run.data['N'] = run.data['c5'] + run.data['c9']
run.data['fraction95'] = run.data['c5']/(run.data['c5'] + run.data['c9'])
RabiperVpp_47MHz_2025 = 12.01/0.452 # 2025-02-12

# fudge by sqrt(0.31) for blackman square pulse area
run.data['OmegaR'] = Vpp_from_VVAfreq(run.data['VVA'], 47.2227) * \
					RabiperVpp_47MHz_2025 * 2 * pi * 0.42 #* np.sqrt(0.31)
run.data['OmegaR2t2'] = 0.01**2 * run.data['OmegaR']**2

df_bm = run.data.loc[run.data['OmegaR2t2'] < 1e3]

fig, axes = plt.subplots(2,2, figsize=(6.3,6))
axs = axes.flatten()

ax = axs[0]
ax.set(xlabel=r'$\Omega_R^2 t^2$', ylabel=r'Transfer $\alpha$', ylim=[-0.05, 1.20],
	   xlim=[0, 15])
ax.plot(df_osc['OmegaR2t2'], df_osc['fraction95'], label='Rabi osc. data', **styles[0])
ax.plot(df_sq['OmegaR2t2'], df_sq['fraction95'], label='10us sq.', **styles[1])
ax.plot(df_bm['OmegaR2t2'], df_bm['fraction95'], label='10us bm', **styles[2])
ts = np.linspace(0, t_max*np.sqrt(8/10), 100)
ax.plot(ts**2*OmegaR**2, np.sin(OmegaR/2*ts)**2, '-', color=colors[0])
ax.plot(ts**2*OmegaR**2, ts**2*OmegaR**2/4, '--', label='lin. resp.', color=colors[1])
ax.legend()


ax = axs[1]
ax.set(xlabel=r'$\alpha$', ylabel='N')
ax.plot(df_osc['fraction95'], df_osc['N'], label='Rabi osc. data', **styles[0])
ax.plot(df_sq['fraction95'], df_sq['N'], label='10us sq.', **styles[1])
ax.plot(df_bm['fraction95'], df_bm['N'], label='10us bm', **styles[2])


ax = axs[2]
ax.set(xlabel=r'$\Omega_R t$', ylabel=r'$\alpha$', xlim=[-0.2, 2])
ax.plot(df_osc['OmegaR'] * df_osc['pulse time'], df_osc['fraction95'], label='Rabi osc. data', **styles[0])
ax.plot(df_sq['OmegaR'] * 0.01, df_sq['fraction95'], label='10us sq.', **styles[1])
ax.plot(df_bm['OmegaR'] * 0.01, df_bm['fraction95'], label='10us bm', **styles[2])

fig.suptitle("Non-interacting, resonant transfer")
fig.tight_layout()
plt.show()

