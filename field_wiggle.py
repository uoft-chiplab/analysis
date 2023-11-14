# -*- coding: utf-8 -*-
"""
field_wiggle.py
2023-11-14
@author: Chip Lab

Analysis field wiggle scans where the frequency of transfer is 
varied, and the delay time is varied. Files are organized by
delay time due to how MatLab outputs the .dat files.
"""

from data_class import Data
from fit_functions import Sinc2, Lorentzian, Gaussian, \
							Parabola, Sin, FixedSin
from scipy.optimize import curve_fit
from library import *
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

run = "2023-11-13_E"
wiggle_freq = 2.5

data_folder = "data\\"+run
delay_times = np.linspace(0.05, 0.57, 14)
file_prefix = run+"_e"

x_name = "freq"
y_name = "fraction95"
fit_func = Sinc2
num = 500

data_list = []
wiggle_data = []

# initalize no guesses, but fill them in if needed
guess = None
guess_list = [guess]*len(delay_times)

# manually fixed guesses
guess_list[3] = guess

### Fit freq scans
for time, guess in zip(delay_times, guess_list):
	delay_tag = "_delay={:.2f}".format(time)
	file = file_prefix+delay_tag+".dat"
	delay = Data(file, path=data_folder)
	
	fit_data = np.array(delay.data[[x_name, y_name]])
	
	func, auto_guess, fit_params = fit_func(fit_data)
	if guess is None:
		guess = auto_guess
	try:
		delay.popt, delay.pcov = curve_fit(func, fit_data[:,0],
									 fit_data[:,1], p0=guess)
		delay.perr = np.sqrt(np.diag(delay.pcov))	
	except RuntimeError: # guess params sucked, so plot and skip
		print("Unable to fit {:.2f} delay scan".format(time))
		plt.figure()
		xx = np.linspace(np.min(fit_data[:,0]),
				   np.max(fit_data[:,0]), num)
		plt.plot(xx, func(xx, *guess), "--")
		plt.plot(fit_data[:,0], fit_data[:,1], 'go')
		plt.title("Guess fit for {:.2f} delay scan".format(time))
		plt.xlabel(x_name)
		plt.ylabel(y_name)
		plt.show()
		
		data_list.append(None)
		continue
	
	delay.B = B_from_FreqMHz(delay.popt[1])
	delay.Berr = np.abs(delay.B-B_from_FreqMHz(delay.popt[1]+delay.perr[1]))
	
	wiggle_data.append(np.array([np.array(time), delay.B, delay.Berr]))
	data_list.append(delay)
	
wiggle_data = np.array(wiggle_data)

func, guess, fit_params = FixedSin(wiggle_data[:,[0,1]], wiggle_freq)
guess[1] = 2
popt, pcov = curve_fit(func, wiggle_data[:,0], wiggle_data[:,1], 
					   p0=guess)
perr = np.sqrt(np.diag(pcov))

parameter_table = tabulate([['Values', *popt], 
								 ['Errors', *perr]], 
								 headers=fit_params)
print(parameter_table)
	
plt.figure()
xx = np.linspace(np.min(wiggle_data[:,0]),
				   np.max(wiggle_data[:,0]), num)
plt.plot(xx, func(xx, *guess), "--")
plt.errorbar(wiggle_data[:,0], wiggle_data[:,1], 
			 yerr=wiggle_data[:,2], fmt='go')
plt.title("Guess fit for {:.2f} delay scan".format(time))
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.show()
	