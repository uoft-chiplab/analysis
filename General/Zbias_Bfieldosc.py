# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:19:19 2024

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
from fit_functions import Sinc2
from scipy.optimize import curve_fit
from library import colors, dark_colors, light_colors, markers, B_from_FreqMHz

import numpy as np
import matplotlib.pyplot as plt
from cycler import Cycler
from tabulate import tabulate

styles = Cycler([{'color':dark_color, 'mec':dark_color, 'mfc':light_color,
					 'marker':marker} for dark_color, light_color, marker in \
						   zip(dark_colors, light_colors, markers)])

plt.rcParams.update({"figure.figsize": [5,3.5]})

def sin(x, A, omega, phi, C):
	return A*np.sin(omega*x+phi) + C

# create file names
filename = "2024-11-06_F_e_delay_samples="
delay1 = np.arange(1000, 1420, 60)
delay2 = np.arange(9500, 9920, 60)
delays = list(delay1) + list(delay2)

files = [filename+str(x)+'.dat' for x in delays]

# convert delay from samples to ms
delays = np.array(delays)/100 

# freq centre guesses for fits
guesses = 45.9 * np.ones(len(files))

# guesses[8] = 45.918

fit_func = Sinc2

# .dat file column names and guess params
names = ['freq', 'fraction95']
guess = [0.17, guesses, 0.01, 0.01]

# loop over files, fitting and taking peak frequencies
freqs = []
e_freqs = []
for file, guess in zip(files, guesses):
	print("--------------------------")
	print("Fitting", file)
	guess = [0.5, guess, 0.1, 0]
	data = Data(file)
	if data.data['delay_samples'] >= 9500:
		data.data['delay_samples'] = data.data['delay_samples']-(18*500)
		
	data.fit(fit_func, names, guess=guess)
	freqs.append(data.popt[1])
	e_freqs.append(data.perr[1])
	print("freq = {:.4f}({:.0f}) MHz".format(data.popt[1], data.perr[1]*1e4)) 
	
# fit results
freqs = np.array(freqs)
e_freqs = np.array(e_freqs)
Bs = [B_from_FreqMHz(freq) for freq in freqs]
e_Bs = e_freqs/freqs*Bs

fit_func = sin
p0 = [0.02, 2*np.pi*0.25, 0, 45.90]
popt, pcov = curve_fit(fit_func, delays, Bs, sigma=e_Bs)
perr = np.sqrt(np.diag(pcov))

# plotting
sty = list(styles)[0]

xs = np.linspace(min(delays), max(delays), 1000)

plt.figure()
plt.xlabel(r'Time, $t$ [ms]')
plt.ylabel(r'Frequency, $f$ [MHz]')
plt.errorbar(delays, Bs, e_Bs, **sty)
plt.plot(xs, fit_func(xs, *popt), '-', color=colors[0])

param_names = ['A', 'omega', 'phi', 'C']
parameter_table = tabulate([['Values', *popt], ['Errors', *perr]], 
						 headers=param_names)
print(parameter_table)
plt.tight_layout()
