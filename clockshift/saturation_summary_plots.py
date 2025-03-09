# -*- coding: utf-8 -*-
"""
@author: Chip Lab
"""
# paths
import os
import sys
# this is a hack to access modules in the parent directory
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
	sys.path.append(parent_dir)
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'saturation_data')
figfolder_path = os.path.join(proj_path, 'figures')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import time

from library import styles, plt_settings, quotient_propagation, colors

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/np.abs(x0)))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 9999)

debug = True

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

# print relevant columns for data selection
print(df[['file', 'detuning', 'ToTF', 'popt']])

### plots
plt.rcParams.update(plt_settings)
plt.rcParams.update({
					"figure.figsize": [15,8],
					"font.size":14,
					"lines.linewidth":1.5,
					})
alpha = 0.25

fig, axes = plt.subplots(2,3)
axs = axes.flatten()

fig_raw, axes_raw = plt.subplots(2,3)
axs_raw = axes_raw.flatten()

ylims = [0.00, 0.25]
xlims = [0.00, 0.2]
axes_settings = {
				'xlabel': "Measured Transfer", 
				'ylabel': "Linear Transfer",
				'xlim': xlims,
				'ylim': ylims,
				}

axes_titles = [
				r"HFT detunings at 0.616 $T_F$",
				"HFT temperatures at 100 kHz",
				"Near res. detunings 2ms Blackman",
				"",
				"",
				"",
				]

raw_ylims = [-0.05, 0.65]

axes_raw_settings = {
				'xlabel': r'rf power $\Omega_R^2$ [kHz$^2$]', 
				'ylabel': r'Transfer $\Gamma t_{rf}/N$',
				'ylim': raw_ylims,
				}

# axes settings and titles
for i, ax in enumerate(axs):
	ax.title.set_text(axes_titles[i])
	ax.set(**axes_settings)
	# plot y=x line
	ax.plot(np.linspace(0,ylims[1],100),
		 np.linspace(0,ylims[1],100), '-', color='dimgrey')
	
# raw plots axes settings and titles
for i, ax in enumerate(axs_raw):
	if i == 3:
		axes_raw_settings['ylabel'] = r'Transfer or Loss $\Gamma t_{rf}/N$'
	ax.title.set_text(axes_titles[i])
	ax.set(**axes_raw_settings)
	# plot y=x line
# 	ax.plot(np.linspace(0,ylims[1],100),
# 		 np.linspace(0,ylims[1],100), '-', color='dimgrey')


###
### HFT various detunings, same ToTF
###
ax = axs[0]
ax_raw = axs_raw[0]
sub_df = df.loc[(df.ToTF == 0.616) & (df.detuning > 20)]

detunings = sub_df.detuning.unique()
detunings.sort()
for i, detuning in enumerate(detunings):
	sty = styles[i]
	color = colors[i]
	label = f'{detuning} kHz'
	# squeeze turns this into a series, so we can access each column without fuss
	data = sub_df.loc[sub_df.detuning == detuning].squeeze()
	
	xs = np.linspace(0, data.df.OmegaR2.max(), 1000)
	
	popt = data.popt
	pcov = data.pcov
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], 
						 popt[0], popt[1], np.sqrt(pcov[0,0]), 
						 np.sqrt(pcov[1,1]), pcov[0,1])
	
	ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label, color=color)
	ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
		Gammas_Lin+e_Gammas_Lin, color=color, alpha=alpha)
	
	# raw data plots 
	ax_raw.plot(xs, Gammas_Lin, '--', color=color)
	ax_raw.plot(xs, Gammas_Sat, '-', color=color)
	
	x = data.df['OmegaR2']
	y = data.df['transfer']
	yerr = data.df['em_transfer']
	
	ax_raw.errorbar(x, y, yerr=yerr, **sty, label=label)
	
ax.legend()
ax_raw.legend()


###
### HFT various ToTF, same detuning
###
ax = axs[1]
ax.set(xlim=[0.045, 0.055], ylim=[0.045, 0.06])
ax_raw = axs_raw[1]
sub_df = df.loc[(df.detuning == 100)]

ToTFs = sub_df.ToTF.unique()
ToTFs.sort()
for i, ToTF in enumerate(ToTFs):
	sty = styles[i]
	color = colors[i]
	label = f'{ToTF}'+ r' $T_F$'
	# squeeze turns this into a series, so we can access each column without fuss
	data = sub_df.loc[sub_df.ToTF == ToTF].squeeze()
	
	xs = np.linspace(0, data.df.OmegaR2.max(), 1000)
	
	popt = data.popt
	pcov = data.pcov
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], 
						 popt[0], popt[1], np.sqrt(pcov[0,0]), 
						 np.sqrt(pcov[1,1]), pcov[0,1])
	
	ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label, color=color)
	ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
		Gammas_Lin+e_Gammas_Lin, color=color, alpha=alpha)
	
	# raw data plots 
	ax_raw.plot(xs, Gammas_Lin, '--', color=color)
	ax_raw.plot(xs, Gammas_Sat, '-', color=color)
	
	x = data.df['OmegaR2']
	y = data.df['transfer']
	yerr = data.df['em_transfer']
	
	ax_raw.errorbar(x, y, yerr=yerr, **sty, label=label)
	
ax.legend()
ax_raw.legend()

	

###
### res various detunings
###
ax = axs[2]
ax_raw = axs_raw[2]
sub_df = df.loc[(df.pulse_time > 0.5) & (df.ToTF == 0.647)]

detunings = sub_df.detuning.unique()
detunings.sort()
for i, detuning in enumerate(detunings):
	sty = styles[i]
	color = colors[i]
	label = f'{detuning} kHz'
	# squeeze turns this into a series, so we can access each column without fuss
	data = sub_df.loc[sub_df.detuning == detuning].squeeze()
	
	xs = np.linspace(0, data.df.OmegaR2.max(), 1000)
	
	popt = data.popt
	pcov = data.pcov
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], 
						 popt[0], popt[1], np.sqrt(pcov[0,0]), 
						 np.sqrt(pcov[1,1]), pcov[0,1])
	
	ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label, color=color)
	ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
		Gammas_Lin+e_Gammas_Lin, color=color, alpha=alpha)
	
	# raw data plots 
	ax_raw.plot(xs, Gammas_Lin, '--', color=color)
	ax_raw.plot(xs, Gammas_Sat, '-', color=color)
	
	x = data.df['OmegaR2']
	y = data.df['transfer']
	yerr = data.df['em_transfer']
	
	ax_raw.errorbar(x, y, yerr=yerr, **sty, label=label)
	
ax_raw.legend()
ax.legend()


###
### HFT various detuings, same ToTF
###
ax = axs[3]
ax_raw = axs_raw[3]
sub_df = df.loc[(df.ToTF == 0.616) & ((df.detuning == 50) | (df.detuning == 100))]

# choice for loss comparison?
j = 1

detunings = sub_df.detuning.unique()
detunings.sort()
for i, detuning in enumerate(detunings):
	sty = styles[i]
	color = colors[i]
	label = f'{detuning} kHz'
	# squeeze turns this into a series, so we can access each column without fuss
	data = sub_df.loc[sub_df.detuning == detuning].squeeze()
	
	xs = np.linspace(0, data.df.OmegaR2.max(), 1000)
	
	# transfer
	popt = data.popt
	pcov = data.pcov
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], 
						 popt[0], popt[1], np.sqrt(pcov[0,0]), 
						 np.sqrt(pcov[1,1]), pcov[0,1])
	
	ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label, color=color)
	ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
		Gammas_Lin+e_Gammas_Lin, color=color, alpha=alpha)
	
	if i == j:
		sty = styles[0]
		color = colors[0]
		# raw transfer plots 
		ax_raw.plot(xs, Gammas_Lin, '--', color=color)
		ax_raw.plot(xs, Gammas_Sat, '-', color=color)
		
		x = data.df['OmegaR2']
		y = data.df['transfer']
		yerr = data.df['em_transfer']
		
		ax_raw.errorbar(x, y, yerr=yerr, **sty, label=label)
	
	# loss
	sty = styles[i]
	color = colors[i]
	popt = data.popt_l
	pcov = data.pcov_l
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], 
						 popt[0], popt[1], np.sqrt(pcov[0,0]), 
						 np.sqrt(pcov[1,1]), pcov[0,1])
	
	ax.plot(Gammas_Sat, Gammas_Lin, '--', label=label+' loss', color=color)
	ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
		Gammas_Lin+e_Gammas_Lin, color=color, alpha=alpha)
	
	if i == j:
		sty = styles[1]
		color = colors[1]
		# raw loss plots 
		ax_raw.plot(xs, Gammas_Lin, '--', color=color)
		ax_raw.plot(xs, Gammas_Sat, '-', color=color)
		
		x = data.df['OmegaR2']
		y = data.df['loss']
		yerr = data.df['em_loss']
		
		ax_raw.errorbar(x, y, yerr=yerr, **sty, label=label+' loss')
	
ax.legend()
ax_raw.legend()


###
### HFT various ToTF, same detuning
###
ax = axs[4]
ax_raw = axs_raw[4]
sub_df = df.loc[(df.detuning == 100) & ((df.ToTF == 0.616) | (df.ToTF == 0.384))]

ToTFs = sub_df.ToTF.unique()
ToTFs.sort()
for i, ToTF in enumerate(ToTFs):
	sty = styles[i]
	color = colors[i]
	label = f'{ToTF}'+ r' $T_F$'
	# squeeze turns this into a series, so we can access each column without fuss
	data = sub_df.loc[sub_df.ToTF == ToTF].squeeze()
	
	xs = np.linspace(0, data.df.OmegaR2.max(), 1000)
	
	# transfer
	popt = data.popt
	pcov = data.pcov
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], 
						 popt[0], popt[1], np.sqrt(pcov[0,0]), 
						 np.sqrt(pcov[1,1]), pcov[0,1])
	
	ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label, color=color)
	ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
		Gammas_Lin+e_Gammas_Lin, color=color, alpha=alpha)
	
	if i == 0:
		sty = styles[0]
		color = colors[0]
		# raw transfer plots 
		ax_raw.plot(xs, Gammas_Lin, '--', color=color)
		ax_raw.plot(xs, Gammas_Sat, '-', color=color)
		
		x = data.df['OmegaR2']
		y = data.df['transfer']
		yerr = data.df['em_transfer']
		
		ax_raw.errorbar(x, y, yerr=yerr, **sty, label=label)
	
	# loss
	popt = data.popt_l
	pcov = data.pcov_l
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], 
						 popt[0], popt[1], np.sqrt(pcov[0,0]), 
						 np.sqrt(pcov[1,1]), pcov[0,1])
	
	ax.plot(Gammas_Sat, Gammas_Lin, '--', label=label+' loss', color=color)
	ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
		Gammas_Lin+e_Gammas_Lin, color=color, alpha=alpha)
	
	if i == 0:
		sty = styles[1]
		color = colors[1]
		# raw loss plots 
		ax_raw.plot(xs, Gammas_Lin, '--', color=color)
		ax_raw.plot(xs, Gammas_Sat, '-', color=color)
		
		x = data.df['OmegaR2']
		y = data.df['loss']
		yerr = data.df['em_loss']
		
		ax_raw.errorbar(x, y, yerr=yerr, **sty, label=label+' loss')
	
ax.legend()
ax_raw.legend()


###
### near res various detunings
###
ax = axs[5]
ax_raw = axs_raw[5]
sub_df = df.loc[(df.ToTF == 0.647) & (df.pulse_time > 0.5) & ((df.detuning == 0) | (df.detuning == 5))]

detunings = sub_df.detuning.unique()
detunings.sort()
for i, detuning in enumerate(detunings):
	color = colors[i]
	label = f'{detuning} kHz'
	# squeeze turns this into a series, so we can access each column without fuss
	data = sub_df.loc[sub_df.detuning == detuning].squeeze()
	
	xs = np.linspace(0, data.df.OmegaR2.max(), 1000)
	
	# transfer
	popt = data.popt
	pcov = data.pcov
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], 
						 popt[0], popt[1], np.sqrt(pcov[0,0]), 
						 np.sqrt(pcov[1,1]), pcov[0,1])
	
	ax.plot(Gammas_Sat, Gammas_Lin, '-', label=label, color=color)
	ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
		Gammas_Lin+e_Gammas_Lin, color=color, alpha=alpha)
	
	if i == 0:
		sty = styles[0]
		# raw transfer plots 
		ax_raw.plot(xs, Gammas_Lin, '--', color=color)
		ax_raw.plot(xs, Gammas_Sat, '-', color=color)
		
		x = data.df['OmegaR2']
		y = data.df['transfer']
		yerr = data.df['em_transfer']
		
		ax_raw.errorbar(x, y, yerr=yerr, label=label, **sty)
	
	# loss
	popt = data.popt_l
	pcov = data.pcov_l
	Gammas_Sat = Saturation(xs, *popt)
	Gammas_Lin = xs*popt[0]/popt[1]
	e_Gammas_Lin = quotient_propagation(xs*popt[0]/popt[1], 
						 popt[0], popt[1], np.sqrt(pcov[0,0]), 
						 np.sqrt(pcov[1,1]), pcov[0,1])
	
	ax.plot(Gammas_Sat, Gammas_Lin, '--', label=label+' loss', color=color)
	ax.fill_between(Gammas_Sat, Gammas_Lin-e_Gammas_Lin, 
		Gammas_Lin+e_Gammas_Lin, color=color, alpha=alpha)
	
	if i == 0:
		sty = styles[1]
		color = colors[1]
		# raw loss plots 
		ax_raw.plot(xs, Gammas_Lin, '--', color=color)
		ax_raw.plot(xs, Gammas_Sat, '-', color=color)
		
		x = data.df['OmegaR2']
		y = data.df['loss']
		yerr = data.df['em_loss']
		
		ax_raw.errorbar(x, y, yerr=yerr, **sty, label=label+' loss')
	
ax.legend()
ax_raw.legend()


fig.tight_layout()
fig_raw.tight_layout()
plt.show()


