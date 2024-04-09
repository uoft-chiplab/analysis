#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analyse wiggle phase shift measurements from Oct 27 2023, where binding energies and loss amplitude are fit for each time point in the wiggle.

"""
import sys
# module_folder = 'E:\\Analysis Scripts\\analysis'
module_folder = '//Users//kevinxie//Documents//GitHub//analysis//phaseshift'
if module_folder not in sys.path:
	sys.path.insert(0, module_folder)
from data_class import Data
from library import FreqMHz
from fit_functions import Gaussian
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

### load data and set up
run = '2024-04-05_F'
run_fn = run + '_UHfit.dat'
run_df = pd.read_excel('phaseshift_summary.xlsx')
run_df = run_df.loc[run_df['filename'] == run_fn]
x_name = "freq"
y_name = "N"
fit_func = Gaussian
guess = [-5000, 43.23, 0.1, 35000] # A, x0, sigma, C
data_folder = 'data/'
run = Data(run_fn, path=data_folder)

### functions ###
# Fixed sinusoidal function depending on given wiggle freq
def FixedSinkHz(t, A, p, C):
	omega = run_df.freq/1000.0 * 2 * np.pi # kHz
	return A*np.sin(omega*t - p) + C

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

# plt.style.use('plottingstype.mplstyle')

### switches ###
FIX_EB=False
FIX_WIDTH=False
fixedwidth=0.034542573613013806 #taken from the median sigma of an unfixed result
save_intermediate=False
save_final=False
if FIX_EB and FIX_WIDTH:
    suffix_str = 'f0 fixed and sigma fixed'
elif FIX_EB:
    suffix_str = 'f0 fixed'
elif FIX_WIDTH:
    suffix_str = 'sigma fixed'
else:
    suffix_str = 'unfixed'
    
skip_time = [0.07, 0.09, 0.260, 0.280]
subrun_list = []
### Analysis loop
for time in run.data.time.unique():
    # not efficient but make new object and then filter
    subrun = Data(run_fn, path=data_folder)
    subrun.data = subrun.data.loc[subrun.data.time == time]
    subrun.data.time = subrun.data.time * 1000 # ms->us, won't be consistent across all data
    # if time in skip_time:
    #     continue
    subrun.fit(fit_func, names = [x_name, y_name], guess=guess, label=str(time*1000)+' us')
    
    [subrun.data['A'], subrun.data['f0'], subrun.data['sigma'], subrun.data['C']] = \
        [*subrun.popt]
        
    [subrun.data['e_A'], subrun.data['e_f0'], subrun.data['e_sigma'], subrun.data['e_C']] = \
        [*subrun.perr]
    
    subrun_list.append(subrun)
    


# df['field']=wigglefield_from_time(df['delay + pulse'])
# df['Eb']=Eb_from_field(df['field'])
# df['f75'] = FreqMHz(df['field'], 9/2, -5/2, 9/2, -7/2)
# df['f75-Eb']=df['f75']-df['Eb']

#print(df)

        
# merge fit results into original dataframe
# should have 3 rows for each run, 1 for each measurable
fit_df = pd.DataFrame(fit_dict_list)
df = pd.merge(df, fit_df, on='run',how='outer')
       
### Sinusoidal fit to fit results over time ###
if FIX_EB and FIX_WIDTH:
    params=['amp','offset']
elif FIX_EB:
    params=['amp','sigma','offset']
elif FIX_WIDTH:
    params=['amp','f0','offset']
else:
    params=['amp','f0','sigma','offset']
    
dflist=[]
for param in params:
    bounds=([-np.inf, 0, -np.inf],[np.inf, np.pi, np.inf]) # ensure phase > 0

    c5_df = df[df.meas=='c5']
    c9_df = df[df.meas=='c9']
    sum95_df = df[df.meas == 'sum95']
    
    c5_popt, c5_pcov = curve_fit(FixedSin5kHz, c5_df['delay + pulse'], c5_df[param], sigma=c5_df['e_'+param],bounds=bounds, p0=[100, 2,43])
    c5_perr = np.sqrt(np.diag(c5_pcov))
    
    c9_popt, c9_pcov = curve_fit(FixedSin5kHz, c9_df['delay + pulse'], c9_df[param], sigma=c9_df['e_' + param],bounds=bounds,p0=[100, 2,43])
    c9_perr = np.sqrt(np.diag(c9_pcov))
    
    sum95_popt, sum95_pcov = curve_fit(FixedSin5kHz, sum95_df['delay + pulse'], sum95_df[param], sigma=sum95_df['e_' + param],bounds=bounds,p0=[100, 2,43])
    sum95_perr = np.sqrt(np.diag(sum95_pcov))
    
    temp_df = pd.DataFrame([{'meas':'c5', 'param':param, 'popt':c5_popt,'perr':c5_perr},\
                           {'meas':'c9',  'param':param,'popt':c9_popt,'perr':c9_perr},\
                               {'meas':'sum95',  'param':param,'popt':sum95_popt,'perr':sum95_perr}])  
    dflist.append(temp_df)
    
# summarize phase shift results
ps_df = pd.concat(dflist)
ps_df['amp'] = ps_df.popt.map(lambda x: x[0])
ps_df['phase']=ps_df.popt.map(lambda x: x[1])
ps_df['offs'] = ps_df.popt.map(lambda x: x[2])
ps_df['e_amp'] = ps_df.perr.map(lambda x: x[0])
ps_df['e_phase']=ps_df.perr.map(lambda x: x[1])
ps_df['e_offs'] = ps_df.perr.map(lambda x: x[2])

##### Plot fit results over time #####
title_str = 'Phase shift_' + suffix_str
### AMPLITUDE
param='amp'
#arb_scaling=9000
#arb_offset=-4600
arb_scaling=24000
arb_offset=-3500
tpoints = np.linspace(-100,300,400)
fig, ax=plt.subplots(1,1)
ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='field')
colors = ["red", 
		  "green", "orange", 
		  "purple", "black", "pink"]
linestyles=['r-','g-','y-','p-','k-','m-']
meass=['sum95','c9','c5']
for i,meas in enumerate(meass):
    
    if meas == 'c5':
        legendtxt = 'b'
        plot_df = c5_df
    elif meas == 'c9':
        legendtxt = 'a'
        plot_df = c9_df
    else :
        legendtxt='a+b'
        plot_df = sum95_df
    ax.errorbar(plot_df['delay + pulse'], plot_df[param],plot_df['e_'+param],color=colors[i], label=legendtxt)
    ax.plot(tpoints, FixedSin5kHz(tpoints, *ps_df[(ps_df.meas==meas) & (ps_df.param == param)]['popt'].values[0]), linestyles[i])

ax.legend(loc='upper right')
ax.set_xlabel('time (us)')
ax.set_ylabel('amplitude (arb.)')
ax.set_title(title_str)
if save_final:
    plt.savefig(data_folder + '/results/' + title_str + '_amp.png', dpi=200)

### Plot peak frequency fit results ###
### FREQUENCY ###
if FIX_EB == False:
    param='f0'
    arb_scaling=0.3
    arb_offset=-158.86
    tpoints = np.linspace(-100,300,400)
    fig, ax=plt.subplots(1,1)
    ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='field')
    for i,meas in enumerate(meass):
        if meas == 'c5':
            legendtxt = 'b'
            plot_df = c5_df
        elif meas == 'c9':
            legendtxt = 'a'
            plot_df = c9_df
        else :
            legendtxt='a+b'
            plot_df = sum95_df
        ax.errorbar(plot_df['delay + pulse'], plot_df[param],plot_df['e_'+param],color=colors[i], label=legendtxt)
        ax.plot(tpoints, FixedSin5kHz(tpoints, *ps_df[(ps_df.meas==meas) & (ps_df.param == param)]['popt'].values[0]), linestyles[i])
    
    ax.legend(loc='upper right')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('f0 (MHz)')
    ax.set_title(title_str)
    if save_final:
        plt.savefig(data_folder + '/results/' + title_str + '_f0.png', dpi=200)
    
### SIGMA
if FIX_WIDTH==False:
    param='sigma'
    arb_scaling=0.1
    arb_offset=-202.04
    tpoints = np.linspace(-100,300,400)
    fig, ax=plt.subplots(1,1)
    ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='field')
    colors = ["red", 
    		  "green", "orange", 
    		  "purple", "black", "pink"]
    linestyles=['r-','g-','y-','p-','k-','m-']
    for i,meas in enumerate(meass):
        
        if meas == 'c5':
            legendtxt = 'b'
            plot_df = c5_df
        elif meas == 'c9':
            legendtxt = 'a'
            plot_df = c9_df
        else :
            legendtxt='a+b'
            plot_df = sum95_df
        ax.errorbar(plot_df['delay + pulse'], plot_df[param],plot_df['e_'+param],color=colors[i], label=legendtxt)
        ax.plot(tpoints, FixedSin5kHz(tpoints, *ps_df[(ps_df.meas==meas) & (ps_df.param == param)]['popt'].values[0]), linestyles[i])
    
    ax.legend(loc='upper right')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('sigma (MHz)')
    ax.set_ylim([0.02,0.07])
    ax.set_title(title_str)
    if save_final:
        plt.savefig(data_folder + '/results/' + title_str + '_sigma.png',dpi=200)


### CONSTANT
param='offset'
arb_scaling=80000
arb_offset=14000
tpoints = np.linspace(-100,300,400)
fig, ax=plt.subplots(1,1)
ax.plot(tpoints, wigglefield_from_time(tpoints, arb_scaling)+arb_offset, 'b-', label='field')
colors = ["red", 
		  "green", "orange", 
		  "purple", "black", "pink"]
linestyles=['r-','g-','y-','p-','k-','m-']
for i,meas in enumerate(meass):
    
    if meas == 'c5':
        legendtxt = 'b'
        plot_df = c5_df
    elif meas == 'c9':
        legendtxt = 'a'
        plot_df = c9_df
    else :
        legendtxt='a+b'
        plot_df = sum95_df
    ax.errorbar(plot_df['delay + pulse'], plot_df[param],plot_df['e_'+param],color=colors[i], label=legendtxt)
    ax.plot(tpoints, FixedSin5kHz(tpoints, *ps_df[(ps_df.meas==meas) & (ps_df.param == param)]['popt'].values[0]), linestyles[i])

ax.legend(loc='upper right')
ax.set_xlabel('time (us)')
ax.set_ylabel('offset (arb.)')
ax.set_title(title_str)
if save_final:
    plt.savefig(data_folder + '/results/' + title_str + '_offset.png',dpi=200)
    

### working
phi=2.61375
phi0 = 2.20464
ps = phi - phi0
EF = 16 #kHz
omega = 5
s = 0.14 # T/TF = 0.58, uniform gas
z = EF/omega*s*np.tan(ps)
print(ps)
print(z)
    
    
    
