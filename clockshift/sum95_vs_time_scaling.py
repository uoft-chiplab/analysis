# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:49:08 2024

@author: coldatoms
"""
from data_class import Data
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

### filenames
dimer_vs_time_file = "2024-06-27_F_e.dat"

def pulsearea(x):
	return np.sqrt(0.31)*(x - 25.2) + (1)*(25.2)

pulseareanew = 1 # square pulse

def OmegaR(VVA):
	VpptoOmegaR = 27.5833 # kHz
	return  pulseareanew*VpptoOmegaR*calInterp(VVA)

def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')

# data file
data_file = os.path.join(data_path, 'VVAtoVpp_square_43p2MHz.txt')

# load data, plot
cal = pd.read_csv(data_file, sep='\t', skiprows=1, names=['VVA','Vpp'])
calInterp = lambda x: np.interp(x, cal['VVA'], cal['Vpp'])
fig, axes = plt.subplots(2,2)
xx = np.linspace(cal.VVA.min(), cal.VVA.max(),100)

### dimer transfer vs time
ax = axes[0,0]

ax.plot(xx, calInterp(xx), '--')
ax.plot(cal.VVA, cal.Vpp, 'o')
 
y_target = 10/2/np.pi  # Example target y value
	
fig, ax = plt.subplots()

data = Data(dimer_vs_time_file, path=data_path)
xname = 'pulse time'
data.data['OmegaR'] = OmegaR(data.data[xname])
data.data['OmegaR2'] = data.data['OmegaR']**2
transfershift=0
data.data['transfer'] = (data.data['sum95'].max() - data.data['sum95'])/data.data['sum95'].max() - transfershift
data.group_by_mean(xname)

cut = 0.07 # omegaR cutoff because of saturation
data.avg_data['cut'] = np.where(data.avg_data['pulse time'] < cut, 1, 0)

fitdf = data.avg_data[data.avg_data.cut == 1]
satdf = data.avg_data[data.avg_data.cut == 0]
x = fitdf['pulse time']
y = fitdf['sum95']
yerr =fitdf['em_sum95']

xsat = satdf['pulse time']
ysat = satdf['sum95']
yerrsat =satdf['em_sum95']

ax.errorbar(x, y, yerr, color='orange')
ax.errorbar(xsat, ysat, yerrsat, mfc='white')

guess = [100, 1]
popt, pcov = curve_fit(Linear,x,y, p0=guess)
xlist = np.linspace(min(x),max(x),1000)

ax.plot(xlist,Linear(xlist,*popt),linestyle='-',marker='', 
		label='linear', color='orange')
ax.set_xlabel('Pulse duration [ms]')
ax.set_ylabel('Atom num. [arb.]')
ax.legend()
print(f'm={popt[0]},b={popt[1]}')

# plot residuals as inset
yreslin = y - Linear(x, *popt)

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.5, 0.6, 0.3, 0.2]
ax_inset = fig.add_axes([left, bottom, width, height])
ax_inset.errorbar(x, yreslin, yerr, color='orange', mfc='orange',label='Linear')
ax_inset.hlines(0, x.min(), x.max() ,ls='dashed')
ax_inset.set_title('Residuals')
plt.show()
