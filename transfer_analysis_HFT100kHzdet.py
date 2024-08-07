# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:09:31 2024

@author: coldatoms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_class import Data
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def linear(x, m, b):
	return m*x + b 

table = pd.read_csv('VVAtoVppSquare10Gain47MHz.txt', sep='\t')
table.columns=['VVA','Vpp']

VpptoOmegaR = 27.5833 # kHz
table['OmegaR'] = table['Vpp'] * VpptoOmegaR

VVAtoOmegaR = interp1d(table.VVA, table.OmegaR)
xx = np.linspace(min(table.VVA), max(table.VVA), 30)
yy = VVAtoOmegaR(xx)

fig, ax = plt.subplots()
ax.plot(table.VVA, table.OmegaR)
ax.plot(xx, yy, '--')
ax.set(xlabel='VVA', ylabel='Omega_R', title='Square pulse, 10 Gain, 47 MHz')

#### fixed time = 50 us, scan VVA (square pulse)
run = Data('2024-07-16_Q_e.dat')
# trf = 50 # us
run.data['OmegaR2'] = VVAtoOmegaR(run.data.VVA)**2
run.data['Transfer']=run.data['fraction95']
run.group_by_mean('OmegaR2')

x = run.avg_data['OmegaR2']
x2 = run.avg_data['VVA']
y= run.avg_data['Transfer']
yerr = run.avg_data['em_Transfer']

fitdata = run.avg_data[run.avg_data['Transfer']<0.26]
popt, pcov = curve_fit(linear, fitdata['OmegaR2'], fitdata['Transfer'])
xx = np.linspace(0, max(fitdata['OmegaR2']), 50)
yy = linear(xx, *popt)

# residuals
yyres = fitdata['Transfer']- linear(fitdata['OmegaR2'], *popt)


fig, ax = plt.subplots(figsize=(6, 4))
ax2 = ax.twiny()
ax.errorbar(x, y, yerr)
ax2.plot(x2, -1*np.ones(len(x2)))
ax.set(ylim=[0, max(y)+0.03], xlabel='Omega_R^2 [1/s^2]', ylabel='Transfer [arb.]')
ax2.set(xlabel='VVA')
ax.plot(xx, yy, '--', label='Free to free')
#inset
left, bottom, width, height = [0.5, 0.2, 0.2, 0.2]
ax3 = fig.add_axes([left, bottom, width, height])
ax3.errorbar(fitdata['Transfer'], yyres, fitdata['em_Transfer'])
ax.legend()

### fixed VVA = 2.1 V, scan time (square pulse)

run = Data('2024-07-16_R_e.dat')
# trf = 50 # us
run.data.VVA = 2.1
run.data['OmegaR2'] = VVAtoOmegaR(run.data.VVA)**2
run.data['Transfer']=run.data['fraction95']
run.group_by_mean('time')

x = run.avg_data['time']
y= run.avg_data['Transfer']
yerr = run.avg_data['em_Transfer']

fitdata = run.avg_data[run.avg_data['Transfer']<0.13]
popt, pcov = curve_fit(linear, fitdata['time'], fitdata['Transfer'])
xx = np.linspace(0, max(fitdata['time']), 50)
yy = linear(xx, *popt)

# residuals
yyres = fitdata['Transfer']- linear(fitdata['time'], *popt)


fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(x, y, yerr)
# ax2.plot(x2, -1*np.ones(len(x2)))
ax.set(ylim=[0, max(y)+0.03], xlabel='pulse time [ms]', ylabel='Transfer [arb.]')
# ax2.set(xlabel='VVA')
ax.plot(xx, yy, '--', label='Free to free')
#inset
left, bottom, width, height = [0.5, 0.2, 0.2, 0.2]
ax3 = fig.add_axes([left, bottom, width, height])
ax3.errorbar(fitdata['Transfer'], yyres, fitdata['em_Transfer'])
ax.legend()


