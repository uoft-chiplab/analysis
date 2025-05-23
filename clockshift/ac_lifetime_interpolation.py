# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 12:28:34 2025

@author: coldatoms
"""
import numpy as np
import matplotlib.pyplot as plt
from library import styles, colors
from scipy.optimize import curve_fit

def quadratic(x, a, b):
	return a*x**2 + b

def sat_quadratic(x, a, b):
	return a*(x**2/(1+x**2/b))

def inv_quadratic(x, a):
	return a/x**2

# def tau_func(x):
#  	return np.piecewise(x, [x<0.5, x>0.5], [lambda t: ])

data = {
		"ToTF": np.array([0.276, 0.358, 0.5771, 0.7393, 0.7545, 0.9284, 1.124]),
		"e_ToTF": np.array([0.007, 0.004, 0.004, 0.0052, 0.006, 0.019, 0.017]),
		"EF": np.array([11.8, 13.91, 17.17, 17.66, 16.72, 16.65, 16.69]),
		"e_EF": np.array([0.035, 0.06, 0.07, 0.06, 0.11, 0.16, 0.07]),
		"tau": np.array([447, 202, 78, 180, 76, 83, 130]),
		"e_tau": np.array([339, 31, 7, 20*1e1, 2, 2, 36]), 
		}

data['T'] = data['ToTF']*data['EF']
data['e_T'] = data['e_EF']*data['ToTF']  # approximately
data['scaled_tau'] = data['tau']*data['T']**2*data['EF']**(3/2)

data['loss_rate'] = 1/data['tau']
data['e_loss_rate'] = data['e_tau']/data['tau']**2

# the following is roughly e_tau * e_EF**7/2
data['e_scaled_tau'] = data['scaled_tau']*np.sqrt((7/2*data['e_EF']/data['EF'])**2 \
												  + (data['e_tau']/data['tau'])**2)

fig, axes = plt.subplots(2,2, figsize = (8,8))
axs = axes.flatten()

x = data['T']
xerr = data['e_T']
xs = np.linspace(0, max(x), 100)

ax = axs[0]
ax.set(xlabel=r"Temperature, $T$ (kHz)", ylabel=r'Lifetime, $\tau$ (ms)')

ax.errorbar(x, data['tau'], yerr=data['e_tau'], 
			 xerr=xerr, **styles[0])

# popt, pcov = curve_fit(inv_quadratic, x, data['tau'], sigma=data['e_tau'])

# ax.plot(xs[10:35], inv_quadratic(xs[10:35], *popt), ':')

ax = axs[1]
ax.set(xlabel=r"Temperature, $T$ (kHz)", ylabel=r'Scaled Lifetime, $\tau T^2 \langle n_a \rangle$ (arb.)')

y = data['scaled_tau']
yerr = data['e_scaled_tau']

popt, pcov = curve_fit(quadratic, x, y, sigma=yerr)

ax.errorbar(x, y, yerr=yerr, xerr=xerr, **styles[0])
ax.plot(xs, quadratic(xs, *popt), '--', color=colors[0])

ax = axs[2]
ax.set(xlabel=r"Temperature, $T$ (kHz)", ylabel=r'Residual Scaled Lifetime (arb.)')

y_res = data['scaled_tau'] - quadratic(x, *popt)
yerr = data['e_scaled_tau']

indices = [0,1,2,4,5]
ax.errorbar(x[indices], y_res[indices], yerr=yerr[indices], xerr=xerr[indices], **styles[0])
ax.plot(np.linspace(min(x[indices]), max(x[indices]), 100), np.zeros(100), 'k--')

ax = axs[3]
ax.set(xlabel=r'Temperature, $T$ (kHz)', ylabel=r'$1/\tau$ (kHz)')
yerr = data['e_loss_rate']/data['EF']**(3/2)
y =  data['loss_rate']/data['EF']**(3/2)

ax.errorbar(x, y, yerr=yerr,  **styles[0])

popt, pcov = curve_fit(quadratic, x[:2], y[:2], sigma=yerr[:2])
ax.plot(xs[0:35], quadratic(xs[0:35], *popt), ':')

spopt, spcov = curve_fit(sat_quadratic, x, y, sigma=yerr, p0=[4e-4/100, 100])
ax.plot(xs, sat_quadratic(xs, *spopt), '--')




fig.tight_layout()
plt.show()