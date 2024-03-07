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
import re
import os

run = "2024-03-05_F"
wiggle_freq = 10.0 # kHz
data_folder = "data\\FieldWiggleCal\\"+run
regex = re.compile('2024-03-05_F_e_delay from wiggle start=(\d+).dat')
x_name = "freq"
y_name = "fraction95"
fit_func = Sinc2
num = 500
data_list = []
wiggle_data = []

def FixedSinkHz(t, A, p, C):
	omega = wiggle_freq/1000.0 * 2 * np.pi # 2.5 kHz
	return A*np.sin(omega*t - p) + C


# initalize no guesses, but fill them in if needed
guess = None

### Fit freq scans
for file in os.listdir(data_folder):
	res = regex.match(file)
	if res: 
		time = int(res.group(1))

	data = Data(file, path=data_folder)
	data.fit(fit_func, names = [x_name, y_name])
	
	data.B = B_from_FreqMHz(data.popt[1])
	data.Berr = np.abs(data.B-B_from_FreqMHz(data.popt[1]+data.perr[1]))
	
	wiggle_data.append(np.array([time, data.B, data.Berr]))
	data_list.append(data)
	
wiggle_data = np.array(wiggle_data)

func=FixedSinkHz
popt, pcov = curve_fit(func, wiggle_data[:,0], wiggle_data[:,1], sigma=wiggle_data[:,2])
perr = np.sqrt(np.diag(pcov))

fit_params=['Amplitude','Phase','Offset']
parameter_table = tabulate([['Values', *popt], 
								 ['Errors', *perr]], 
								 headers=fit_params)
print("Field calibration:")
print("")
print(parameter_table)
	
plt.figure()
xx = np.linspace(0,
				   np.max(wiggle_data[:,0]), num)
plt.plot(xx, func(xx, *popt), "--")
plt.errorbar(wiggle_data[:,0], wiggle_data[:,1], 
			 yerr=wiggle_data[:,2], fmt='go')
plt.title("2024-03-05_D_" + str(wiggle_freq) + " kHz field wiggle cal")
plt.xlabel('time [us]')
plt.ylabel('field [G]')
plt.show()
	