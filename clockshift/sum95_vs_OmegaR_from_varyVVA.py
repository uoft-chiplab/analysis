# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:03:54 2024

@author: coldatoms
"""
from data_class import Data
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')

data_file = os.path.join(data_path, 'VVAtoVpp_square_43p2MHz.txt')

# Vpp = {}
# with open(file_path) as f:
# 	next(f)
# 	for line in f:
# 		tok = line.split()
# 		Vpp[tok[0]] = tok[1]

# Vpp = {key : float(value) for key, value in Vpp.items()}

cal = pd.read_csv(data_file, sep='\t', skiprows=1, names=['VVA','Vpp'])
calInterp = lambda x: np.interp(x, cal['VVA'], cal['Vpp'])
fig, ax = plt.subplots()
xx = np.linspace(cal.VVA.min(), cal.VVA.max(),100)
ax.plot(xx, calInterp(xx), '--')
ax.plot(cal.VVA, cal.Vpp, 'o')

def pulsearea(x):
	return np.sqrt(0.31)*(x - 25.2) + (1)*(25.2)

#pulseareanew = np.sqrt(3)
pulseareanew = 1 # square pulse

def OmegaR(VVA):
	VpptoOmegaR = 27.5833 # kHz
	return  pulseareanew*VpptoOmegaR*calInterp(VVA)
 
y_target = 10/2/np.pi  # Example target y value

# Use np.interp to find x_interp for y_target
# x_interp = np.interp(y_target, cal['Vpp'], cal['VVA'])
# print(x_interp)
# List of VVA values
#VVA_list = [float(num) for num in Data("2024-06-25_J_e.dat",average_by='VVA').avg_data['VVA']]

# Initialize an empty list to store results
# results = []

# Iterate through each VVA value
# for vva in VVA_list:
#     # Call OmegaR function and append the result to the list
#     results.append(OmegaR(vva))
	
fig, ax = plt.subplots()

# data = Data("2024-06-25_J_e.dat",average_by='VVA')
data = Data("2024-06-27_C_e.dat")
xname = 'VVA'
data.data['OmegaR'] = OmegaR(data.data[xname])
data.data['OmegaR2'] = data.data['OmegaR']**2
transfershift = 0.10
data.data['transfer'] = (data.data['sum95'].max() - data.data['sum95'])/data.data['sum95'].max() - transfershift
data.group_by_mean(xname)

cut = 13 # omegaR cutoff because of saturation
data.avg_data['cut'] = np.where(data.avg_data['OmegaR'] < 13, 1, 0)

fitdf = data.avg_data
satdf = data.avg_data
# fitdf = data.avg_data[data.avg_data.cut == 1]
# satdf = data.avg_data[data.avg_data.cut == 0]
x = fitdf['OmegaR']
y = fitdf['transfer']
yerr =fitdf['em_transfer']

xsat = satdf['OmegaR']
ysat = satdf['transfer']
yerrsat =satdf['em_transfer']

ax.errorbar(x, y, yerr)
# ax.errorbar(xsat, ysat, yerrsat, mfc='white')

def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c


guess = [100, 1]
popt, pcov = curve_fit(Linear,x,y, p0=guess)
guess = [10, 10, 0]
poptq, pcovq = curve_fit(Quadratic, x, y, p0=guess)
xlist = np.linspace(min(x),max(x),1000)

ax.plot(xlist,Linear(xlist,*popt),linestyle='-',marker='', label='linear')
ax.plot(xlist, Quadratic(xlist, *poptq), linestyle = ':', marker='', label='quadratic')
ax.set_xlabel('OmegaR [kHz]')
ax.set_ylabel('Transfer (arb)')
ax.legend()
print(f'm={popt[0]},b={popt[1]}')

# plot residuals as inset
yreslin = y - Linear(x, *popt)
yresqua = y - Quadratic(x, *poptq)

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.2, 0.6, 0.3, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.errorbar(x, yreslin, yerr, color='orange', mfc='orange',label='Linear')
ax2.errorbar(x, yresqua, yerr, color = 'green', mfc = 'green',label='Quadratic')
ax2.hlines(0, x.min(), x.max() ,ls='dashed')
ax2.set_title('Residuals')
plt.show()
