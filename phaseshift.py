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

delay_times = np.array(np.linspace(0,0.77,19))
delay_times = [0,0.4,0.8,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.44,0.48,0.52,0.56,0.6]
names = ['freq','sum95']
# uncert_list = []
# file = '2023-11-13_E_e.dat'

directory = os.fsencode('E:/Analysis Scripts/analysis/data/2023-11-14_F/')

# data =  Data('2023-11-14_F_e_delay=0.25.dat', column_names=['freq','sum95'])

# print(data.fit.popt)

amp_list=np.array([])
amper_list=np.array([])
x0_list=np.array([])
x0er_list=np.array([])

for file in os.listdir(directory):
	filename = os.fsdecode(file)
# 	if filename.endswith(".dat"):
	data = Data(filename, column_names=names)
	data.fit(Lorentzian,names=names)
	amp = data.popt[0]
	amp_list = np.append(amp_list,amp)
	errors = np.sqrt(np.diag(data.pcov))
	amper = errors[0]
	amper_list = np.append(amper_list,amper)
	x0 = data.popt[1]
	x0_list = np.append(x0_list,x0)
	x0er = errors[1]
	x0er_list = np.append(x0er_list, x0er)
		


plt.figure(1)
plt.plot(delay_times,amp_list)
plt.show()


# B_list = np.array(list(map(B_from_FreqMHz, freq_list))).flatten()
# B_list_plus = np.array(list(map(B_from_FreqMHz, freq_list+freq_err_list)))
# B_list_minus = np.array(list(map(B_from_FreqMHz, freq_list-freq_err_list)))

# calculate error in B field, 
# by checking +freq and -freq error, taking the largest
# B_err_list = np.array([])
# for i in range(len(B_list)):
# 	B_err = max([np.abs(B_list[i]-B_list_plus[i]),
# 			  np.abs(B_list[i]-B_list_minus[i])])
# 	B_err_list = np.append(B_err_list, B_err)
# 	

# fit_func, guess, params = FixedSin(np.array(list(zip(delay_times,B_list))), 2.5)
# 	
# popt, pcov = curve_fit(fit_func, delay_times, B_list, p0=guess, sigma=B_err_list)
# perr = np.sqrt(np.diag(pcov))

num = 100
x_list = np.linspace(0, 0.6, num)
	
# plt.figure()
# plt.errorbar(delay_times, B_list, yerr=B_err_list, fmt='o')
# # plt.plot(x_list, fit_func(x_list, *guess), 'r--')
# plt.plot(x_list, fit_func(x_list, *popt), 'g')
# plt.xlabel("Delay Time (ms)")
# plt.ylabel("Magnetic Field B (G)")
# plt.ylim(202, 202.2)
# plt.show()

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