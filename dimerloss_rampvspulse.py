# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:45:22 2024

This file contains generic examples for plotting and fitting using the Data class.

@author: coldatoms
"""
from data_class import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

B_list = [202.1, 201.9,201.7,201.7,201.9,202.1]
method_list=['ramp','ramp','ramp','pulse','pulse','pulse']

date = '2024-04-30'
letters = ['B','C','D','E','F','G']
data_files = [date + '_' + l + '_e.dat' for l in letters ]
guess = [-7000, 43.2, 0.05, 35000]

popt_list=[]
perr_list=[]
for file in data_files:
	data = Data(file)
	data.fit(Gaussian, names= ['freq','sum95'], guess= guess)
	popt_list.append(data.popt)
	perr_list.append(data.perr)
	
data_dict = {'B':B_list, 'method':method_list, 'popt':popt_list, 'perr':perr_list}	
data_df = pd.DataFrame.from_dict(data_dict)
# Extract elements of popt and perr lists into their own columns
split_df1 = pd.DataFrame(data_df['popt'].tolist(), columns=['A','x0','sigma','C'])
split_df2 = pd.DataFrame(data_df['perr'].tolist(), columns = ['e_A','e_x0','e_sigma','e_C'])
data_df = pd.concat([data_df, split_df1], axis=1)
data_df = pd.concat([data_df, split_df2], axis=1)
data_df.drop(['popt','perr'],axis=1, inplace=True)
data_df.sigma = abs(data_df.sigma)

ramp_df = data_df[data_df['method']=='ramp']
pulse_df = data_df[data_df['method']=='pulse']

fig, ax=plt.subplots(2,2)
ax[0,0].errorbar(ramp_df['B'],ramp_df['A'], ramp_df['e_A'], label='ramp', fmt = 'ro')
ax[0,0].errorbar(pulse_df['B'],pulse_df['A'], pulse_df['e_A'], label='pulse', fmt = 'bo')
ax[0,0].legend()
ax[0,0].set_ylabel('A')
ax[0,1].errorbar(ramp_df['B'],ramp_df['x0'], ramp_df['e_x0'], label='ramp', fmt = 'ro')
ax[0,1].errorbar(pulse_df['B'],pulse_df['x0'], pulse_df['e_x0'], label='pulse', fmt = 'bo')
ax[0,1].set_ylabel('x0')
ax[1,0].errorbar(ramp_df['B'],ramp_df['sigma'], ramp_df['e_sigma'], label='ramp', fmt = 'ro')
ax[1,0].errorbar(pulse_df['B'],pulse_df['sigma'], pulse_df['e_sigma'], label='pulse', fmt = 'bo')
ax[1,0].set_ylabel('sigma')
ax[1,1].errorbar(ramp_df['B'],ramp_df['C'], ramp_df['e_C'], label='ramp', fmt = 'ro')
ax[1,1].errorbar(pulse_df['B'],pulse_df['C'], pulse_df['e_C'], label='pulse', fmt = 'bo')
ax[1,1].set_ylabel('C')

fig.tight_layout()