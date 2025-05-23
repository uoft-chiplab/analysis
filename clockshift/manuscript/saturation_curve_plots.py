# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 08:33:01 2025

@author: Chip lab
"""
import numpy as np
import matplotlib.pyplot as plt
from library import paper_settings, generate_plt_styles
import pickle as pkl
import pandas as pd
import os

data_path = 'clockshift/saturation_data'

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

# plotting options
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c']
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

styles = generate_plt_styles(colors, ts=0.6)

linestyles = ['--', ':']

### plot settings
plt.rcParams.update(paper_settings) # from library.py
font_size = paper_settings['legend.fontsize']
plt.rcParams['legend.fontsize'] = 6
fig_width = 3.4 # One-column PRL figure size in inches
subplotlabel_font = 10
		
fig, axs = plt.subplots(1, 2, figsize=(fig_width, 3/5*fig_width))

axs[0].set(xlabel='rf power $P/P_0$', 
		   ylabel=r'Transfer $\alpha$',
		   ylim=[-0.02, 0.23])

# loop over dimers
dimer_file = 'dimer_saturation_curves.pkl'

with open(os.path.join(data_path, dimer_file), 'rb') as f:
	    dimer_data = pkl.load(f)
		
# select data
dimer_data = [dimer_data[0], dimer_data[2]]
		
# chosen average saturation Rabi
OmegaRabi2 = 3272
e_OmegaRabi2 = 136

for i, data in enumerate(dimer_data):
	df = data['df']
	
	popt = data['popt_b']
	data['e_ToTF'] = data['ToTF']*0.03
	
	# dimer
	sty = styles[i]
	color = colors[i]
	label = r'$T/T_F$ = {:.2f}({:.0f})'.format(data['ToTF'], 1e2*data['e_ToTF'])
	label = r'{:.2f}({:.0f}) $T_F$'.format(data['ToTF'], data['e_ToTF']*1e2)
	x = df['OmegaR2']
	y = df['c5_transfer']
	yerr = df['em_c5_transfer']
	
	xs = np.linspace(0, max(x), 1000)  # linspace of rf powers
	
	# rescale x-axis
	ax = axs[0]
	x = df['OmegaR2']/OmegaRabi2
	
	ax.errorbar(x, y, yerr=yerr, label=label, **sty)
	ax.plot(xs/OmegaRabi2, Saturation(xs, *popt), '-', color=color)
	ax.plot(xs/OmegaRabi2, Linear(xs, popt[0]/popt[1], 0), linestyles[i], 
	  color=color)
	

### HFT various ToTF, same detuning
# save files 
files = [
		 '100kHz_saturation_curves.pkl', 
		 'near-res_saturation_curves.pkl',
		 'various_ToTF_saturation_curves.pkl',
		 ]

loaded_data = []

# grab dictionary lists from pickle files
for i, file in enumerate(files):
	with open(os.path.join(data_path, file), "rb") as input_file:
		loaded_data = loaded_data + pkl.load(input_file)
		
# turn dictionary list into dataframe
df = pd.DataFrame(loaded_data)

ax = axs[1]

ax.set(xlabel='rf power $P/P_0$', 
# 		   ylabel=r'Transfer $\alpha$',
 		   ylim=[-0.04, 0.9]
		   )
sub_df = df.loc[(df.detuning == 100)]

ToTFs = sub_df.ToTF.unique()
ToTFs.sort()
ToTFs = [ToTFs[0], ToTFs[2]]

# chosen average saturation Rabi
OmegaRabi2 = np.mean([716.46, 962.81])

for i, ToTF in enumerate(ToTFs):
	sty = styles[i+2]
	color = colors[i+2]
	e_ToTF = ToTF * 0.03
	label = r'$T/T_F$ = {:.2f}({:.0f})'.format(ToTF, e_ToTF*1e2)
	label = r'{:.2f}({:.0f}) $T_F$'.format(ToTF, e_ToTF*1e2)
	
	# squeeze turns this into a series, so we can access each column without fuss
	data = sub_df.loc[sub_df.ToTF == ToTF].squeeze()
	
	xs = np.linspace(0, 4.05*OmegaRabi2, 100)
	
	popt = data.popt
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	
	# raw data plots 
	ax.plot(xs/OmegaRabi2, Gammas_Sat, '-', color=color)
	ax.plot(xs/OmegaRabi2, Gammas_Lin, linestyles[i], color=color)
	
	x = data.df['OmegaR2']
	y = data.df['transfer']
	yerr = data.df['em_transfer']
	
	ax.errorbar(x/OmegaRabi2, y, yerr=yerr, **sty, label=label)

fig.tight_layout()
subplot_labels = ['(a)', '(b)']
for n, ax in enumerate(axs):
	ax.legend(frameon=False, handletextpad=0)
	label = subplot_labels[n]
	ax.text(-0.2, 1.0, label, transform=ax.transAxes, size=subplotlabel_font)
	
plt.subplots_adjust(top=0.9)
# fig.tight_layout()
	
plt.savefig("clockshift/manuscript/manuscript_figures/dimer_and_HFT_saturation_curves.pdf")
plt.show()
