# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:03:54 2024

@author: coldatoms

plotting transfer vs OmegaR for a data set with residuals and fitting linear and 
qudratic lines
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


cal = pd.read_csv(data_file, sep='\t', skiprows=1, names=['VVA','Vpp'])
calInterp = lambda x: np.interp(x, cal['VVA'], cal['Vpp'])
fig, ax = plt.subplots()
xx = np.linspace(cal.VVA.min(), cal.VVA.max(),100)
ax.plot(xx, calInterp(xx), '--')
ax.plot(cal.VVA, cal.Vpp, 'o')

pulseareanew = 1 # square pulse

def OmegaR(VVA):
	VpptoOmegaR = 27.5833 # kHz
	return  pulseareanew*VpptoOmegaR*calInterp(VVA)

def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c
 
y_target = 10/2/np.pi  # Example target y value
	
fig, ax = plt.subplots()

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
x = fitdf['OmegaR']
y = fitdf['transfer']
yerr =fitdf['em_transfer']

xsat = satdf['OmegaR']
ysat = satdf['transfer']
yerrsat =satdf['em_transfer']

ax.errorbar(x, y, yerr)

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
