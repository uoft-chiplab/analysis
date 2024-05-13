# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Plots data fitted in heating_rates_fitting.py and theory lines from modified
Tilman code bulkvisctrap_class.py.

Relies on data_class.py, library.py, and bulkvisctrap_class.py
"""
from data_class import Data
from library import *
from UFG_analysis import BulkViscTrap, BulkViscUniform
from itertools import chain
from matplotlib import cm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import pickle

ARBITRARY_RESCALING = 1
tintshade = 0.4

removed_files = ["2024-03-26_C_UHfit.dat",
				"2024-04-03_B_UHfit.dat",
				"2024-04-03_C_UHfit.dat",
				"2024-04-03_D_UHfit.dat",
				"2024-04-05_C_UHfit.dat",
				"2024-03-26_B_UHfit.dat",
				"2024-04-08_G_UHfit.dat",
				"2024-04-10_C_UHfit.dat",
				"2024-04-10_D_UHfit.dat"
# 				"2024-03-19_L_UHfit.dat",
# 				"2024-04-03_F_UHfit.dat"
				]

removed_names = ["2024-04-03_F_f=1_Vpp=1.80", 
				 "2024-04-03_F_f=2_Vpp=1.80",
				 "2024-04-08_G_f=20_Vpp=1.80"]

data_folder = 'data\\heating'
pkl_filename = 'heating_rate_fit_results.pkl'
pkl_file = os.path.join(data_folder, pkl_filename)

BVT_pkl_filename = 'BVT.pkl'
BVT_pkl_file = os.path.join(data_folder, BVT_pkl_filename)

TilmanPRL0p58ToTF_filename = "zetaomega_T0.58.txt"

plotting = True
plot_legend = True
load_theory = True # load Tilman lines from previous evaluation
debugging_plots = False

# only use up to max freq
nu_min = 1
nu_max = 120

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

def zeta_from_C(freq, C, EF):
	"""See OverLeaf note"""
	pi_factors = (4*pi)**(3/2)/(36*pi*(2*pi)**(3/2))
	return pi_factors * (EF/freq)**(3/2) * C

def Edot_from_C(freq, C, EF):
	"""See OverLeaf note"""
	pi_factors = (4*pi)**(3/2)/(36*pi*(2*pi)**(3/2))
	return pi_factors * (EF/freq)**(3/2) * C


colors = ["blue", "red", "green", "orange", 
		  "purple", "teal", "brown", "olive", 
		  "pink",  "black", "blue", "red", 
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
plt.rcParams.update(plt_settings)
fig, axs = plt.subplots(2,2)

for ax in axs.flatten():
	ax.tick_params(which='both', direction='in')

###
### Scaled heating rate data compared to theory (First Column)
###

# select frequencies within a band.
lr = lr[lr.freq <= max_freq]
lr = lr[lr.freq >= min_freq]

# other selection criteria
lr = lr[lr.filename.isin(removed_files) == False] # WTF pandas
lr = lr[lr.index.isin(removed_names) == False] # WTF pandas


num = int(nu_max/nu_min)
nus = np.linspace(nu_min*1e3, nu_max*1e3, num)

label_Drude = 'Drude'
label_LW = 'L-W'
label_C = 'Contact'
	
ax = axs[0,1]
ylabel = "Heating Rate $h \\langle\dot{E}\\rangle/(E_F\\,A)^2$"
xlabel = "Drive Frequency $\omega/E_F$"
ax.set(ylabel=ylabel, xlabel=xlabel)

ax_res = axs[1,0]
ylabel = "Meas./Theory"
ax_res.set(ylabel=ylabel, xlabel=xlabel)

ax_zeta = axs[0,0]
ylabel = "Contact Correlation $\\langle\\tilde{\zeta}(\omega)\\rangle $"
xlims = [0.05, 15] # omega/EF
ax_zeta.set(xlabel=xlabel, ylabel=ylabel, yscale='log', xscale='log', xlim=xlims)

if plot_legend==False:
	ax_res_zeta = axs[1,1]
	ylabel = "$a^2 \zeta$ Meas./Theory"
	xlabel = "Drive Frequency (kHz)"
# 	xlims = [0.05, 15] # omega/EF
	ax_res_zeta.set(xlabel=xlabel, ylabel=ylabel)

### Heating rate measurements
loops = len(lr.filename.unique()) # to count over to pull the right theory curve if loading
used_colors = []
Tmeans = []
for file, color, marker, i in zip(lr.filename.unique(), colors, markers, range(loops)):
	df = lr.loc[lr['filename'] == file] # get rows from lr that correspond to file
	
	df = df.loc[df['rate']-df['e_rate'] > 0]
	df = df.loc[df['rate']> 0]
	
	xx = np.array(df.freq)
	yy = np.array(df.rate)
	yerr = np.array(df.e_rate)
	Ei = np.array(df.Ei)
	EF = np.array(df.EF)
	T = np.array(df["T"]) # T is a terrible column name
	ToTF = np.array(df.ToTF)
	zetas = np.array(df.zeta)
	e_zetas = np.array(df.e_zeta) # just scales with the other params in the same way
	mean_df = df.groupby(['filename'], as_index=False).mean()
	barnu = float((mean_df.wx*mean_df.wy*mean_df.wz)**(1/3))
	EFmean = float(mean_df.EF)
	ToTFmean = float(mean_df.ToTF)
	Tmean = ToTFmean*EFmean
	
	used_colors.append(color)
	Tmeans.append(Tmean)
	
# 	print(ToTF, EF, barnu)
	
	label = r'[{}_{}]  $E_F={:.1f}$kHz, $T/T_F={:.2f}$, $\bar \nu={:.0f}Hz$'.format(df.date.values[0], 
								  df.run.values[0], EFmean, ToTFmean, barnu)
	# trap $\bar\omega={:.0f}$Hz, 
	light_color = tint_shade_color(color, amount=1+tintshade)
	dark_color = tint_shade_color(color, amount=1-tintshade)
	
	ax.errorbar(xx/EF, yy, yerr=yerr, capsize=0, fmt=marker, 
			 color=dark_color, markerfacecolor=light_color, 
			 markeredgecolor=dark_color, markeredgewidth=2)
	
	ax_zeta.errorbar(xx/EF, zetas, yerr=e_zetas, capsize=0, fmt=marker,
			  color=dark_color, markerfacecolor=light_color, 
			  markeredgecolor=dark_color, markeredgewidth=2)

### Heating rate theory line(s)
	if load_theory == False:
		print("Computing theory curve for ToTF={:.2f}".format(ToTFmean))
		params_trap = [Tmean*1e3, barnu, -3800] # give T, barnu and mutrap_guess in Hz
		BVT = BulkViscTrap(*params_trap, nus, ToTF=ToTFmean)
		BVTs.append(BVT)
	else:
		BVT = BVTs[i]
	
	# find nus for nu/EF > 1 to use for high_freq
	nu_small = 0
	for nu in nus:
		if nu < (2*BVT.T):
			nu_small += 1
	ax.plot(BVT.nus/BVT.EF,BVT.Edottraps/(2*BVT.Ns)/BVT.EF**2,
				 ':', color=color, label=label_Drude)
	ax.plot(BVT.nus[nu_small:]/BVT.EF, BVT.EdottrapsC[nu_small:]/(2*BVT.Ns)/BVT.EF**2,
		 '--', color=color, label=label_C)
	ax_zeta.plot(BVT.nus[:nu_small]/BVT.EF, BVT.zetatraps[:nu_small],':', 
			  color=color, label=label_Drude)
	ax_zeta.plot(BVT.nus[nu_small:]/BVT.EF, BVT.zetatrapsC[nu_small:],'--', 
			  color=color, label=label_C)

### Residual ratio
	residuals = []
	e_res = []
	residuals_zeta = []
	e_res_zeta = []
	for freq, rate, e_rate, zeta, e_zeta in zip(xx, yy, yerr, zetas, e_zetas):
		for j in range(len(BVT.nus)):
 			if BVT.nus[j]/1e3 == freq:
				 if BVT.nus[j] < 2*BVT.T:
	 				 residuals.append(rate/(BVT.Edottraps[j]/(2*BVT.Ns)/BVT.EF**2))
	 				 residuals_zeta.append(zeta/BVT.zetatraps[j])
	 				 e_res.append(e_rate/(BVT.Edottraps[j]/(2*BVT.Ns)/BVT.EF**2))
	 				 e_res_zeta.append(e_zeta/BVT.zetatraps[j])
				 else: 
	 				 residuals.append(rate/(BVT.EdottrapsC[j]/(2*BVT.Ns)/BVT.EF**2))
	 				 residuals_zeta.append(zeta/BVT.zetatrapsC[j])
	 				 e_res.append(e_rate/(BVT.EdottrapsC[j]/(2*BVT.Ns)/BVT.EF**2))
	 				 e_res_zeta.append(e_zeta/BVT.zetatrapsC[j])
				 continue
 	
	ax_res.errorbar(xx, np.array(residuals), yerr=e_res, capsize=0, fmt=marker, 
			  label=label, color=dark_color, markerfacecolor=light_color, 
			  markeredgecolor=dark_color, markeredgewidth=2)
	
### Legends for theory curves
	print(i)
	print(BVT.Ctrap)
	if i == 0:
		ax.legend()
		ax_zeta.legend()
		
# set dashed line to illustrate change from Drude to Contact
ymin, ymax = ax_res.get_ylim()
Tline = np.mean(Tmean)
ax_res.vlines(2*Tline, ymin, ymax, colors='grey')
ax_res.set(ylim=[ymin,ymax])

# 	if plot_legend == False:
# 		ax_res_zeta.errorbar(xx, np.array(residuals)*ARBITRARY_RESCALING, yerr=e_res, color=color, capsize=0, fmt=marker)
# 	

### Add contact line
C = 0.78
e_C = 0.01
EF = 16e3
C_theory = 1.465
ax_zeta.plot(nus[nu_small:]/EF, zeta_from_C(nus, C, EF)[nu_small:],'k--',
		 label='$\\langle\\tilde{C}\\rangle = 0.78(1)$ from HFT spectroscopy')

ax_res.plot(nus[nu_small:]/1e3, C/C_theory*np.ones(len(nus[nu_small:])),'k--',
		 label='$\\langle\\tilde{C}\\rangle = 0.78(1)$ from HFT spectroscopy')

### Add tabulated Tilman 0.58ToTF line from PRL
xtilman, ytilman = np.loadtxt(TilmanPRL0p58ToTF_filename, unpack=True, delimiter=' ')

trap_to_uni_ratio = 0.111 # determined by compared tabulated uni 
										# data and trap-averaged at one freq. 
										# This is not precise
										# 0.111?
# have to divide by twelve because Tilman scaled by 12 for some reason
# ax_zeta.plot(xtilman, ytilman/12*trap_to_uni_ratio, ':r', 
# 			 label=r'scaled L-W calculation:  $T/T_F=0.58$')

### Manually add measured phase shift result
# ax_zeta.errorbar(5/19, 0.04/ARBITRARY_RESCALING, yerr=0.01, marker='x', color='k',
#  		 label=r"Phase Shift:  $E_F \sim 19$kHz, $T/T_F \sim 0.58$")

###
### Legend in own subplot
###
if plot_legend:
	lax = axs[1,1]
	h, l = ax_res.get_legend_handles_labels() # get legend from first plot
	lax.legend(h, l, borderaxespad=0)
	lax.axis("off")

	
fig.tight_layout()
# fig.savefig("test.pdf")
plt.show()

########### SAVE THEORY ###########
if load_theory == False:
	with open(BVT_pkl_file, 'wb') as f:  
		pickle.dump(BVTs, f)