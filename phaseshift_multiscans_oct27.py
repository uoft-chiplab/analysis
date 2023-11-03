#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyse wiggle phase shift measurements from Oct 27 2023, where binding energies and loss amplitude are fit for each time point in the wiggle.

"""

from data_class import Data
from library import FreqMHz
from analysisfunctions import FixedSin5kHz, Gaussian

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


data_folder = 'data/oct27phaseshift'
df = pd.read_csv(data_folder + '/Oct27summary.csv')

def wigglefield_from_time(t, arb_scaling=1):
    omega=2*np.pi*5000
    amp=0.0695243 #+/- 0.0025798
    phase=2.17306 #+/- 0.0474719
    offset=202.072 #+/- 0.00203875
    return arb_scaling * amp * np.sin(omega * (t*10**-6) - phase) + offset

def Eb_from_field(B, whichfit='linear'):
    '''
    Eb cal from Oct 30
    Eb in MHz, field input in G
    '''
    if whichfit == 'linear':
        a=-0.05124 #+/- 0.006695
        b=14.35 #+/- 1.353
        return a*B + b

df['field']=wigglefield_from_time(df['delay + pulse'])
df['Eb']=Eb_from_field(df['field'])
df['f75'] = FreqMHz(df['field'], 9/2, -5/2, 9/2, -7/2)
df['f75-Eb']=df['f75']-df['Eb']

params=['c5','c9','sum95']
param_headers_list=[]
param_e_headers_list=[]
for param in params:
    headers = ['amp','f0','sigma','offset']
    param_headers = [param + '_' + i for i in headers]
    param_e_headers = [param + '_e_' + i for i in headers]
    param_headers_list.append(param_headers)
    param_e_headers_list.append(param_e_headers)
    df[[*param_headers]]=0.0
    df[[*param_e_headers]]=0.0

popt_list=[]
perr_list=[]
run_list=[]
FIX_EB=False
verbose=False

for irow in range(0, len(df)):
    run_letter=df.iloc[irow]['run']
    run_filename='2023-10-27_' + run_letter + '_e.dat'
    run=Data(run_filename, path=data_folder, column_names=['freq (MHz)','c5','c9','sum95'])
    #run=Data('2023-10-27_I_e.dat', path=data_folder, column_names=['freq (MHz)', 'c5','c9','sum95'])
    #print(run.data.head())
    run.field=df.loc[irow, 'field']
    run.peakfreq=df.loc[irow, 'f75-Eb']

    for i,param in enumerate(params):
    
        #print(param)
        #bounds=([-10000, 42,-0.2,0],[0, 44, 0.2, 30000])
        popt, pcov = curve_fit(Gaussian, run.data['freq (MHz)'], run.data[param],p0=[-3000, run.peakfreq, 0.05, 0])
        perr = np.sqrt(np.diag(pcov))
       
        if verbose:
            fpoints = np.linspace(run.data['freq (MHz)'].min(), run.data['freq (MHz)'].max(),200)
            fig, ax= plt.subplots(1,1)
            ax.plot(fpoints, Gaussian(fpoints, *popt), 'r-')
            ax.scatter(run.data['freq (MHz)'], run.data[param],label=run_filename)
            ax.set_ylabel(run_letter + '_' + param)
            ax.set_xlabel('freq (MHz)')
            ax.legend(loc='upper right')
       
        df.loc[irow, param_headers_list[i]]=popt
        df.loc[irow, param_e_headers_list[i]]=perr
        
        
# df['c5_Eb']=df['f75'] - df['c5_f0']
# df['c9_Eb']=df['f75'] - df['c9_f0']
# df['sum95_Eb']=df['f75'] - df['sum95_f0']
df.drop([0,3,15,16],inplace=True)
        
### Sinusoidal fit to fit results over time ###
## Amplitude 
bounds=([-np.inf, 0, -np.inf],[np.inf, np.pi, np.inf])
c5_popt, c5_pcov = curve_fit(FixedSin5kHz, df['delay + pulse'], df['c5_amp'], sigma=df['c5_e_amp'],bounds=bounds)
c5_perr = np.sqrt(np.diag(c5_pcov))
c9_popt, c9_pcov = curve_fit(FixedSin5kHz, df['delay + pulse'], df['c9_amp'], sigma=df['c9_e_amp'],bounds=bounds)
c9_perr = np.sqrt(np.diag(c9_pcov))
sum95_popt, sum95_pcov = curve_fit(FixedSin5kHz, df['delay + pulse'], df['sum95_amp'], sigma=df['sum95_e_amp'],bounds=bounds)
sum95_perr = np.sqrt(np.diag(sum95_pcov))
fit_params_list=[c5_popt, c9_popt, sum95_popt]
fit_e_params_list=[c5_perr, c9_perr, sum95_perr]

## Peak frequency
if FIX_EB == False:
    # c5_Eb_popt, c5_Eb_pcov = curve_fit(FixedSin5kHz, df['delay + pulse'], df['c5_Eb'], sigma=df['c5_e_f0'])
    # c5_Eb_perr = np.sqrt(np.diag(c5_Eb_pcov))
    # c9_Eb_popt, c9_Eb_pcov = curve_fit(FixedSin5kHz, df['delay + pulse'], df['c9_Eb'], sigma=df['c9_e_f0'])
    # c9_Eb_perr = np.sqrt(np.diag(c9_Eb_pcov))
    # sum95_Eb_popt, sum95_Eb_pcov = curve_fit(FixedSin5kHz, df['delay + pulse'], df['sum95_Eb'], sigma=df['sum95_e_f0'])
    # sum95_Eb_perr = np.sqrt(np.diag(sum95_Eb_pcov))
    # fit_Eb_params_list=[c5_Eb_popt, c9_Eb_popt, sum95_Eb_popt]
    # fit_e_Eb_params_list=[c5_Eb_perr, c9_Eb_perr, sum95_Eb_perr]
    c5_f0_popt, c5_f0_pcov = curve_fit(FixedSin5kHz, df['delay + pulse'], df['c5_f0'], sigma=df['c5_e_f0'])
    c5_f0_perr = np.sqrt(np.diag(c5_f0_pcov))
    c9_f0_popt, c9_f0_pcov = curve_fit(FixedSin5kHz, df['delay + pulse'], df['c9_f0'], sigma=df['c9_e_f0'])
    c9_f0_perr = np.sqrt(np.diag(c9_f0_pcov))
    sum95_f0_popt, sum95_f0_pcov = curve_fit(FixedSin5kHz, df['delay + pulse'], df['sum95_f0'], sigma=df['sum95_e_f0'])
    sum95_f0_perr = np.sqrt(np.diag(sum95_f0_pcov))
    fit_f0_params_list=[c5_f0_popt, c9_f0_popt, sum95_f0_popt]
    fit_e_f0_params_list=[c5_f0_perr, c9_f0_perr, sum95_f0_perr]

### Plot fit results over time ###
arb_scaling=24000
arb_offset=-3500
tpoints = np.linspace(-100,300,400)
fig, ax=plt.subplots(1,1)
ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='wiggle')
colors = ["red", 
		  "green", "orange", 
		  "purple", "black", "pink"]
linestyles=['r-','g-','y-','p-','k-','m-']
for i,param in enumerate(params):
    ax.errorbar(df['delay + pulse'], df[param+'_amp'],df[param+'_e_amp'],color=colors[i], label=param)
    ax.plot(tpoints, FixedSin5kHz(tpoints, *fit_params_list[i]), linestyles[i])

ax.legend(loc='upper right')
ax.set_xlabel('time (us)')
ax.set_ylabel('amplitude (arb.)')
ax.set_title('Phase shift from dimer scans, f0 unfixed')


### FUCK
if FIX_EB == False:
    arb_scaling=0.3
    arb_offset=-158.86
    tpoints = np.linspace(-100,300,400)
    fig, ax=plt.subplots(1,1)
    ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='wiggle')
    for i,param in enumerate(params):
        ax.errorbar(df['delay + pulse'], df[param+'_f0'],df[param+'_e_f0'],color=colors[i], label=param)
        ax.plot(tpoints, FixedSin5kHz(tpoints, *fit_f0_params_list[i]), linestyles[i])
    
    ax.legend(loc='upper right')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('f0 (MHz)')
    ax.set_title('Phase shift from dimer scans, f0 unfixed')   



        
        
               
       
        
        
   
        
  

'''
def func(x, a, b):
   return a*x*x + b

for b in xrange(10):
   popt, pcov = curve_fit(lambda x, a: func(x, a, b), x1, x2)
'''
# binding energy fits
# Linear fit
#  ------------
# Fit parameters are -0.05124, 14.35
# with errors 0.006695, 1.353
# Fit chi^2 is 0.2588
# Parabolic fit
#  ------------
# Fit parameters are -0.05124, 14.35
# with errors 0.006695, 1.353
# Fit chi^2 is 0.203

# 5 kHz wiggle cal
#   	Amplitude  	phase    	offset
# ------  -----------  ---------  ------------
# Values	0.0695243  2.17306	202.072
# Errors	0.0025798  0.0474719	0.00203875

