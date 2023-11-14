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
from fit_functions import Sinc2, Lorentzian, Gaussian
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_folder = "data/2023-11-14_F"
delay_times = np.linspace(0.01, 0.77, 20)
file_prefix = "2023-11-14_F_e"

x_name = "freq"
y_name = "sum95"
fit_func = Sinc2

data_list = []

for time in delay_times:
	delay_tag = "_delay={}".format(time)
	file = file_prefix+delay_tag+".dat"
	delay = Data(file, path=data_folder)
	
	fit_data = np.array(delay.data[x_name, y_name])
	
	func, guess, fit_params = fit_func(fit_data)
	delay.popt, delay.pcov = curve_fit(func, fit_data[:,0],
									fit_data[:,1])
	delay.perr = np.sqrt(np.diag(delay.pcov))
	data_list.append(delay)
	