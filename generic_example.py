# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:45:22 2024

This file contains generic examples for plotting and fitting using the Data class.

@author: coldatoms
"""
from data_class import *

#to use data_class.py general example:
	#Data("filename").fit(fit_func=One you want, names=['x','y'])

# FB calibration
# file = '2024-03-01_J_e.dat'
# Data(file).fit(Sinc2, names= ['FB','fraction95'])

# VVA calibration
# file = '2024-03-01_K_e.dat'
# Data(file).fit(Parabola, names=['VVA','fraction95'])

# Trap freq
# file = '2024-03-06_P_e.dat'
# data=Data(file)
# data.data = data.data[data.data['time']< 6]
# data.fit(TrapFreq2, names=['time','G_ctr_y'],guess=[2,1,2,0,80])
amplitude_cutoff = 2

data = Data("2024-03-07_G_e_wiggle time=9.dat")
data.data = data.data.drop(data.data[data.data.amplitude>amplitude_cutoff].index)
data.group_by_mean("amplitude")
data.plot2(Quadratic, ["amplitude", "meanEtot_kHz"])


group = ["2024-03-06_Q_e_freq=30.dat","2024-03-06_Q_e_freq=100.dat","2024-03-06_Q_e_freq=50.dat"]

combined_amplitude = []
combined_meanEtot_kHz = []


plt.figure(figsize=(8, 6))

for idx, filename in enumerate(group):	
	data = Data(filename)
	data.data = data.data.drop(data.data[data.data.amplitude>amplitude_cutoff].index)
	data.group_by_mean("amplitude")

	
# 	data.plot2(Quadratic, ["amplitude", "meanEtot_kHz"])
	plt.plot(data.data['amplitude'], data.data['meanEtot_kHz'], 'o', label=filename)

	
# plt.plot(combined_amplitude, combined_meanEtot_kHz, 'o')
plt.title('Plots Overlaid')
plt.xlabel('Amplitude')
plt.ylabel('MeanEtot_kHz')
plt.legend()
plt.show()