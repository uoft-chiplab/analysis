# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:16:48 2024

@author: coldatoms
"""
from data_class import Data
from library import *

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os


data_folder = 'data/transfer_saturation/'

VVA, Vpp = np.loadtxt("data/"+"VVAtoVpp.txt", delimiter='\t', unpack=True)

VVAtoVppdict = {}
# VpptoVAAdict = {}
for key, val in zip(VVA,Vpp):
 	VVAtoVppdict[key] = val
 	# VpptoVVAdict[val] = key

def VVAtoVpp(VVA):
 	return VVAtoVppdict[VVA]

def square(x):
 	return x*x
 
def saturation_func(x, A, P0, B):
	return A * (1 - np.exp(-x/P0)) + B

colors = ["blue", "red", 
		  "green", "orange", 
		  "purple", "black", "pink"]
detunings = [15, 30, 50, 100, 150]
file_postfixes = ["freq=46.6592",
				  "freq=46.6742",
				  "freq=46.6942",
				  "freq=46.7442",
				  "freq=46.7942"]

filename = "2024-02-06_G_e"
files = list(np.zeros(len(detunings)))
datasets = list(np.zeros(len(detunings)))

param = 'Vpp'
param2 = 'fraction95'

pGuess = [0.05, 2, 0]

func = saturation_func

for i in range(len(detunings)):
 	file = filename+"_"+file_postfixes[i]+".dat"
 	files[i] = file
 	data = Data(file, path=data_folder, average_by='VVA')
 	data.detuning = detunings[i]
 	print(data.detuning)
 	data.color = colors[i]
 	data.avg_data["Vpp"] = data.avg_data["VVA"].apply(VVAtoVpp)
 	data.avg_data['em_Vpp'] = data.avg_data["em_VVA"].apply(VVAtoVpp)
 	data.avg_data["Vpp2"] = data.avg_data["Vpp"].apply(square)
 	data.avg_data['em_Vpp2'] = data.avg_data["em_Vpp"].apply(square)
 	xx = data.avg_data["Vpp2"]
 	yy = data.avg_data[param2]
 	eyy = data.avg_data['em_'+param2]
 	data.popt, data.pcov = curve_fit(func, xx, yy, sigma = eyy, 
 								   p0=pGuess)
 	print(data.popt)
 	data.perr = np.sqrt(np.diag(data.pcov))
 	data.VppSet = np.sqrt(data.popt[1]/5)
 	datasets[i] = data
	 

param = 'Vpp2'
xlabel = '$Vpp^2$'
ylabel = 'Transfer fraction'

plt.figure()
for i, data in zip(range(len(datasets)),datasets):
	if i == 0:
# 		plt.errorbar(data.avg_data[param], data.avg_data[param2], linestyle="",
# 				 yerr = data.avg_data['em_'+param2], marker = 'o', 
# 				 color=data.color, capsize = 2)
		continue
	else:
		plt.plot(data.avg_data[param],func(data.avg_data[param], *data.popt), 
		   color=data.color, label=str(data.detuning)+" kHz")
		plt.errorbar(data.avg_data[param], data.avg_data[param2], linestyle="",
				 yerr = data.avg_data['em_'+param2], marker = 'o', 
				 color=data.color, capsize = 2)
		print(data.detuning, data.VppSet)
	
plt.legend()
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()