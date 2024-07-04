# -*- coding: utf-8 -*-
"""
2023-10-19
@author: Chip Lab

Data class for loading, fitting, and plotting .dat
Matlab output files

Relies on functions.py
"""
import os 
from glob import glob
from library import *
from fit_functions import *
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tabulate import tabulate
from data_class import * 

delay_times = np.array(np.linspace(0.005,0.565,15))
# uncert_list = []
# file = '2023-11-13_E_e.dat'

freq_list=np.array([44.8088,
					44.8097,
					44.8135,
					44.8154,
					44.819,
					44.8235,
					44.8224,
					44.8195,
					44.8162,
					44.8129,
					44.8103,
					44.8102,
					44.8141,
					44.8169,
					44.8173
					])

freq_err_list=np.array([.00062171,
					.000513825,
					.000772942,
					.000647702,
					.000627966,
					.000906767,
					.000833965,
					.000641191,
					.000570663,
					.000802226,
					.000794326,
					.00078307,
					.000501847,
					.000697719,
					.000683611
					])

B_list = np.array(list(map(B_from_FreqMHz, freq_list))).flatten()
B_list_plus = np.array(list(map(B_from_FreqMHz, freq_list+freq_err_list)))
B_list_minus = np.array(list(map(B_from_FreqMHz, freq_list-freq_err_list)))

# calculate error in B field, 
# by checking +freq and -freq error, taking the largest
B_err_list = np.array([])
for i in range(len(B_list)):
	B_err = max([np.abs(B_list[i]-B_list_plus[i]),
			  np.abs(B_list[i]-B_list_minus[i])])
	B_err_list = np.append(B_err_list, B_err)
	

fit_func, guess, params = FixedSin(np.array(list(zip(delay_times,B_list))), 2.5)
	
popt, pcov = curve_fit(fit_func, delay_times, B_list, p0=guess, sigma=B_err_list)
perr = np.sqrt(np.diag(pcov))

num = 100
x_list = np.linspace(0, 0.6, num)
	
plt.figure()
plt.errorbar(delay_times, B_list, yerr=B_err_list, fmt='o')
# plt.plot(x_list, fit_func(x_list, *guess), 'r--')
plt.plot(x_list, fit_func(x_list, *popt), 'g')
plt.xlabel("Delay Time (ms)")
plt.ylabel("Magnetic Field B (G)")
plt.ylim(202, 202.2)


# print(*popt)
# print(*perr)

# for i in range(len(delay_times)):
# 	data = Data(file).data.loc[Data.data['delay']==delay_times[i]]
# 	data.popt, data.pcov = curve_fit(fit_func, data.data['freq'], 
# 						data.data['fraction95'])
# 	data.err = np.sqrt(np.diag(data.pcov))
# 	
# 	freq_list = freq_list.append(data.popt[1])
# 	
# 	uncert_list = uncert_list.append(data.err[1])
# 	
# 	
# 	data_list = data_list.append(data)
# 	
# 	print(data_list)