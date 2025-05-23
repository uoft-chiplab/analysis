# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:22:24 2024

@author: coldatoms
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


filename = "G:\\My Drive\\Chip\\3D s-wave internal\\ac_dimer_energies.txt"

data = pd.read_csv(filename, delimiter=',', skiprows=0)

B_name = 'Field (G)'
eB_name = 'Binding Energy (MHz)'
e_eB_name = 'Error BE (MHz)'

data = data[data[B_name]<202.5]
data = data[data[B_name]>201.7]

def linear(x, a, b):
 	return a*x + b

popt, pcov = curve_fit(linear, data[B_name], data[eB_name], sigma=data[e_eB_name])

plt.figure()
plt.xlabel("Magnetic Field (G)")
plt.ylabel("Binding Energy (MHz)")
plt.errorbar(data[B_name], data[eB_name], yerr=data[e_eB_name], fmt='o', capsize=2)
plt.plot(data[B_name], linear(data[B_name], *popt))
plt.show()