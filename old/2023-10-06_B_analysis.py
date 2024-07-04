# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:55:21 2023

@author: coldatoms
"""

from analysisfunctions import * # includes numpy and constants
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os 
import scipy.optimize as curve_fit
from tabulate import tabulate # pip install tabulate
from collections import defaultdict
from data import *
from plotting import *
from generalanalysis import *
import pandas as pd

rescale=True
drop=True
residuals=True
noi = ['cyc','delay time','sum95','c9','c5']
file = glob('\\\\UNOBTAINIUM\\E_Carmen_Santiago\\Data\\2023\\*\\*\\*\\2023-10-06_B_e.dat')[0]
d=data_from_dat(file, names_of_interest=noi)
df=pd.DataFrame(d, columns=noi)
if drop:
	irmlist=[107, 108, 109, 110, 111, 184, 185, 186, 259] # drops repeated t=185.0 rows
	df.drop(index=irmlist, inplace=True)
name='c9'
################# RESCALED FITTING ##########################
if rescale:
	init_val = df['delay time'][0]
	rep_idx = df.loc[df['delay time'] == init_val].index
	rep_idx=rep_idx.append(pd.Index([df.index[-1]]))
	df['partition'] = 0
	for i in range(len(rep_idx)-1):
		df.loc[rep_idx[i]:rep_idx[i+1], 'partition']= i+1
	
	#df=df.assign(partition = lambda x: x['cyc'] // 52 +1)

	df_partitionmean = df.groupby('partition', as_index=False)[name].mean()
	df_partitionmean.rename(columns={name:name + ' mean'}, inplace=True)
	
	df_merge = pd.merge(df, df_partitionmean, on='partition')
	plotname= 'rescaled ' + name
	df_merge[plotname] = df_merge[name] / df_merge[name + ' mean']
	
	avgdf = df_merge.groupby('delay time', as_index=False).agg({plotname:['mean','sem']})
else :
	########################## TYPICAL FITTING ##################################
	plotname=name
	avgdf = df.groupby('delay time', as_index=False).agg({name:['mean','sem']})
	
x = avgdf['delay time'].values
y = avgdf[plotname, 'mean'].values
y_err = avgdf[plotname,'sem'].values

########################## FREE FIT ###########################################
guess=[1000, 0.06, 0, 21000]
popt, pcov = curve_fit.curve_fit(Sin, x, y, p0=guess, sigma=y_err,
	bounds=((0, 0.009*2*np.pi, 0, 0), (np.inf, 0.011*2*np.pi, 2*np.pi, np.inf)))
ym = Sin(np.linspace(max(x), min(x),num=200), *popt)

fig1 = plt.figure(0)
plt.title('Free Sin fit for 2023-10-06_B_e.dat')
plt.xlabel('delay time')
plt.ylabel(name)
plt.errorbar(x, y, yerr=y_err, fmt='go', capsize=4)

errors = np.sqrt(np.diag(pcov))
freq = popt[1]/2/3.14
period = 1/freq
delay = popt[2] % (3.141592654) /popt[1]
values = list([*popt, freq, period, delay])
errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
print('########## FREE FIT ##########')
print(tabulate([['Values', *values], ['Errors', *errors]], headers=['Amplitude','omega','phase','offset', 'freq', 'period', 'delay']))
#plt.plot(np.linspace(max(x),min(x),num=200),ym)

########################## FIXED FIT #########################################
guess=[1000, 0, 21000]
popt2, pcov2 = curve_fit.curve_fit(FixedSin, x, y, p0=guess, sigma=y_err,
	bounds=((0, 0, 0), (np.inf, 2*np.pi, np.inf)))
ym = FixedSin(np.linspace(min(x), max(x),num=200), *popt2)

fig2 = plt.figure(0)
plt.title('Fixed Sin fit for 2023-10-06_B_e.dat')
plt.xlabel('delay time')
plt.ylabel(name)
plt.errorbar(x, y, yerr=y_err, fmt='go', capsize=4)

errors = np.sqrt(np.diag(pcov2))
freq = 0.01
period=1/freq
delay = popt2[2] % (3.141592654) /freq
values = list([*popt2, freq, period, delay])
#errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
print('########## 10 kHz FIXED FIT ##########')
print(tabulate([['Values', *values], ['Errors', *errors]], headers=['Amplitude','phase','offset', 'freq', 'period', 'delay']))
plt.plot(np.linspace(min(x),max(x),num=200),ym, linestyle='--', color='red',label='fit')

#poptwig= [-9.98688586e-02,  5.53588954e+00,  2.02093076e+02]
poptwig= [-popt2[0], 5.53588954e+00,  popt2[2]]
ywig = FixedSin(np.linspace(min(x), max(x), num=200), *poptwig)
plt.plot(np.linspace(min(x),max(x), num=200), ywig, linestyle='-', color='blue', label='field cal')
plt.legend(loc='upper left')
plt.show()

################### RESIDUALS #############################

# y and ym arrays need to be the same length
ym2 = FixedSin(np.linspace(min(x), max(x), num=len(x)), *popt2)
yres = y - ym2
fig3 = plt.figure()
plt.title('Residuals')
plt.xlabel('delay time')
plt.ylabel(name + ' residuals')
plt.errorbar(x, yres, yerr=y_err, fmt='ko', capsize=4)
plt.show()
	
################### CHI SQUARED ###########################

ddof=7
chi2 = np.sum(yres**2 / ym2) / ddof
print('Chi2: ' + str(chi2))


