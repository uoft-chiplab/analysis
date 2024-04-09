# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Plots data fitted in heating_rates_fitting.py and theory lines from modified
Tilman code bulkvisctrap_class.py.

Relies on data_class.py, library.py, and bulkvisctrap_class.py
"""
from data_class import Data
from library import *
from bulkvisctrap_class import BulkViscTrap, BulkViscTrapToTF
from itertools import chain
from matplotlib import cm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import pickle

ARBITRARY_RESCALING = 1/0.4

removed_files = ["2024-03-26_C_UHfit.dat",
				"2024-04-03_B_UHfit.dat",
				"2024-04-03_C_UHfit.dat",
				"2024-04-03_D_UHfit.dat"]
removed_names = ["2024-04-03_F_f=1_Vpp=1.80", 
				 "2024-04-03_F_f=2_Vpp=1.80"]

data_folder = 'data\\heating'
pkl_filename = 'heating_rate_fit_results.pkl'
pkl_file = os.path.join(data_folder, pkl_filename)

BVT_pkl_filename = 'BVT.pkl'
BVT_pkl_file = os.path.join(data_folder, BVT_pkl_filename)

plotting = True
load_theory = True # load Tilman lines from previous evaluation
debugging_plots = False

# only use up to max freq
max_freq = 120
min_freq = 2

cmap = cm.get_cmap('jet')
def get_color(var, minvar, maxvar):
	if var < minvar:
		val = 0
	elif var > maxvar:
		val = 1
	else: # convert to value between 0 and 1
		val = (var-minvar)/(maxvar-minvar)
	return cmap(val)

def mutrap_est(ToTF):
	a = -70e3
	b = 36e3
	return a*ToTF + b

def calc_zeta(f, rate, Ei):
	"""This is a little weird. Ask CD."""
	return 2/9*rate/1e3/ f * Ei/f

colors = ["blue", "red", "green", "orange", 
		  "purple", "black", "brown", "teal", 
		  "olive", "pink", "blue", "red", 
		  "green", "orange", "purple", "black", 
		  "brown", "teal", "olive", "pink"]

markers = ["o", "s", "^", "D", "h", "o", "s", "^", "D", "h",
		   "o", "s", "^", "D", "h", "o", "s", "^", "D", "h"]

############ LOAD DATA ############
try: # open pkl file if it's there
	with open(pkl_file, 'rb') as f:
		lr = pickle.load(f) # load all results in file
except FileNotFoundError: # if file not there then complain
	print("Can't find data results pickle file, you silly billy.")

########### LOAD THEORY ###########
### Run tilman's code for each Theta and EF, dumping result into pickle file
if load_theory == True:
	with open(BVT_pkl_file, 'rb') as f:  
		BVTs = pickle.load(f)
else:
	BVTs = []

############## PLOTTING ##############		
# change matplotlib options
plt.rcParams.update({"figure.figsize": [12,8]})
fig, axs = plt.subplots(2,2)
num = 500
font = {'size'   : 12}
plt.rc('font', **font)
legend_font_size = 10 # makes legend smaller, so plot is visible

###
### Scaled heating rate data compared to theory (First Column)
###

# select frequencies within a band.
lr = lr[lr.freq <= max_freq]
lr = lr[lr.freq >= min_freq]

# other selection criteria
lr = lr[lr.filename.isin(removed_files) == False] # WTF pandas
lr = lr[lr.index.isin(removed_names) == False] # WTF pandas
	
ax = axs[0,0]
ylabel = "Scaled Heating Rate $\partial_t E/E/A^2$ (Hz)"
xlabel = "Drive Frequency (kHz)"
ax.set(ylabel=ylabel, xlabel=xlabel)

ax_res = axs[1,0]
ylabel = "Meas./Theory (arb.)"
ax_res.set(ylabel=ylabel, xlabel=xlabel)

ax_zeta = axs[0,1]
ylabel = "$a^2 \zeta (\omega)$ [dimless]"
xlabel = "Drive Frequency $\omega/E_F$"
ax_zeta.set(xlabel=xlabel, ylabel=ylabel, yscale='log', xscale='log')

### Heating rate measurements
loops = len(lr.filename.unique()) # to count over to pull the right theory curve if loading
for file, color, marker, i in zip(lr.filename.unique(), colors, markers, range(loops)):
	df = lr.loc[lr['filename'] == file] # get rows from lr that correspond to file
	
	df = df.loc[df['rate']-df['e_rate'] > 0]
	df = df.loc[df['rate']> 0]
	
	xx = np.array(df.freq)
	yy = np.array(df.rate)
	yerr = np.array(df.e_rate)
	Ei = np.array(df.Ei)
	EF = np.array(df.EF)
	zeta = calc_zeta(xx, yy, Ei)
	e_zeta = calc_zeta(xx, yerr, Ei) # just scales with the other params in the same way
	mean_df = df.groupby(['filename'], as_index=False).mean()
	barnu = float((mean_df.wx*mean_df.wy*mean_df.wz)**(1/3))
	EFmean = float(mean_df.EF)
	ToTFmean = float(mean_df.ToTF)
	
# 	print(ToTF, EF, barnu)
	
	label = r'[{}_{}]   $E_F={:.1f}$kHz, $T/T_F={:.2f}$'.format(df.date.values[0], 
									  df.run.values[0], EFmean, ToTFmean)
	# trap $\bar\omega={:.0f}$Hz, 
	ax.errorbar(xx, yy*ARBITRARY_RESCALING, yerr=yerr, color=color, capsize=2, 
			 fmt=marker, label=label)
	ax_zeta.errorbar(xx/EF, zeta*ARBITRARY_RESCALING, yerr=e_zeta, color=color, 
				  capsize=2, fmt=marker, label=label)

### Heating rate theory line(s)
	if load_theory == False:
		print("Computing theory curve for ToTF={:.2f}".format(ToTFmean))
		BVT = BulkViscTrapToTF(ToTFmean, EFmean*1e3, barnu, 
 			mutrap_guess=mutrap_est(ToTFmean), nu_max=max_freq*1e3)
		BVT.zeta = calc_zeta(BVT.nus/1e3, BVT.Edottraps/(2*BVT.Ns)/1e3, np.ones(len(BVT.nus)))
		BVTs.append(BVT)
	else:
		BVT = BVTs[i]
	ax.plot(BVT.nus/1e3,BVT.Edottraps/BVT.Etotal,'-', color=color)
	ax_zeta.plot(BVT.nus/1e3/EFmean, BVT.zeta,'-', color=color)

### Residual ratio
	residuals = []
	e_res = []
	for freq, rate, e_rate in zip(df.freq.values, df.rate.values, df.e_rate.values):
		for i in range(len(BVT.nus)):
			if BVT.nus[i]/1e3 == freq:
				 residuals.append(rate/(BVT.Edottraps[i]/BVT.Etotal))
				 e_res.append(e_rate/(BVT.Edottraps[i]/BVT.Etotal))
				 continue
 	
	ax_res.errorbar(xx, np.array(residuals)*ARBITRARY_RESCALING, yerr=e_res, color=color, capsize=2, fmt=marker)
	
	
C = 0.0025# arbitrary scaling
xx = np.linspace(1, 10, num)
ax_zeta.plot(xx, C*xx**(-3/2),'k--',label='$C\ \omega^{-3/2}$')

ax_zeta.errorbar(5/19, 0.04, yerr=0.01, marker='x', color='k',
		 label="Phase Shift")

###
### Legend in own subplot
###
lax = axs[1,1]
h, l = axs[0,1].get_legend_handles_labels() # get legend from first plot
lax.legend(h, l, borderaxespad=0, prop={'size':legend_font_size})
lax.axis("off")

	
fig.tight_layout()
plt.show()
# 	fig.savefig('heating_rate_data.pdf')

########### SAVE THEORY ###########
if load_theory == False:
	with open(BVT_pkl_file, 'wb') as f:  
		pickle.dump(BVTs, f)