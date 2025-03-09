# -*- coding: utf-8 -*-
"""
@author: Chip Lab
"""
# paths
import os
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'saturation_data')
figfolder_path = os.path.join(proj_path, 'figures')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from scipy import integrate

from library import styles, plt_settings, quotient_propagation, colors, \
					GammaTilde, h

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/np.abs(x0)))


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 9999)

debug = True

# save files 
files = [
		 '100kHz_saturation_curves.pkl', 
		 'near-res_saturation_curves.pkl',
		 ]

loaded_data = []

# grab dictionary lists from pickle files
for i, file in enumerate(files):
	with open(os.path.join(data_path, file), "rb") as input_file:
		loaded_data = loaded_data + pkl.load(input_file)
		
# turn dictionary list into dataframe
df = pd.DataFrame(loaded_data)

# filter out cold data sets
df = df.loc[df.ToTF > 0.55]

# arbitrarily set OmegaR2, I don't think this matters because we will use linear transfer
OmegaR2 = 1

df['ScaledDetuning'] = df.detuning/df.EF

popts = ['popt', 'popt_l']
pcovs = ['pcov', 'pcov_l']

for popt, pcov, prefix in zip(popts, pcovs, ['', 'loss_']):
	# calculate linear transfer, at OmegaR2
	df[prefix+'Gamma_lin'] = df.apply(lambda x: OmegaR2*x[popt][0]/x[popt][1], axis=1)
	df[prefix+'e_Gamma_lin'] = df.apply(lambda x: quotient_propagation(x.Gamma_lin, 
							 x[popt][0], x[popt][1], np.sqrt(x[pcov][0,0]), 
							 np.sqrt(x[pcov][1,1]), x[pcov][0,1]), axis=1)
	
	# calculate scaled transfer, add set OmegaR2
	df[prefix+'ScaledTransfer'] = df.apply(lambda x: GammaTilde(x[prefix+'Gamma_lin'], h*x.EF, 
								 np.sqrt(OmegaR2)*2*np.pi, x.pulse_time), axis=1)
	df[prefix+'e_ScaledTransfer'] = df[prefix+'ScaledTransfer']* \
							df[prefix+'e_Gamma_lin']/df[prefix+'Gamma_lin']


# print relevant columns for data selection
print(df[['file', 'detuning', 'pulse_time', 'ToTF', 'ScaledTransfer', 'loss_ScaledTransfer', 'EF']])


### plots
plt.rcParams.update(plt_settings)
plt.rcParams.update({
					"figure.figsize": [15,8],
					"font.size": 14,
					"lines.linewidth": 1.5,
					})
alpha = 0.25

fig, axes = plt.subplots(1,2, figsize=[10, 4])
axs = axes.flatten()

axs[0].set(xlabel="Detuning [EF]", ylabel=r"Scaled Transfer, $\tilde\Gamma$")
axs[1].set(xlabel="Detuning [EF]", ylabel=r"Transfer/Loss")

###
### full spectra transfer and loss
###
ax = axs[0]
x_name = 'ScaledDetuning'

df = df.sort_values(x_name)

# transfer
y_name = 'ScaledTransfer'
yerr_name = 'e_ScaledTransfer'
x = np.array(df[x_name])
y = np.array(df[y_name])
yerr = np.array(df[yerr_name])

sty = styles[0]
color = colors[0]
ax.errorbar(x, y, yerr=yerr, **sty, label='transfer')
ax.plot(x, np.interp(x, x, y), '-', color=colors[0])

# loss
y_name = 'loss_ScaledTransfer'
yerr_name = 'loss_e_ScaledTransfer'
y_loss = np.array(df[y_name])
yerr_loss = np.array(df[yerr_name])

sty = styles[1]
color = colors[1]
ax.errorbar(x, y_loss, yerr=yerr_loss, **sty, label='loss')
ax.plot(x, np.interp(x, x, y_loss), '-', color=colors[1])

ax.legend()

###
### integrate data
###

SW = integrate.trapz(y, x=x)
SW_loss = integrate.trapz(y_loss, x=x)

SW = integrate.quad(lambda d: np.interp(d, x, y), min(x), max(x))
SW_loss = integrate.quad(lambda d: np.interp(d, x, y_loss), min(x), max(x))

# MC error of integral
num = 1000
dist = []
i = 0
while i < num:
	dist.append(integrate.trapz([np.random.normal(val, err) for val, err \
				 in zip(y, yerr)], x=x))
	i += 1
SW_mean = np.array(dist).mean()
SW_std = np.array(dist).std()

print("Spectral weight for transfer is {:.2f}({:.0f})".format(SW_mean, SW_std*1e2))

# MC error of integral
num = 1000
dist = []
i = 0
while i < num:
	dist.append(integrate.trapz([np.random.normal(val, err) for val, err \
				 in zip(y_loss, yerr_loss)], x=x))
	i += 1
SW_loss_mean = np.array(dist).mean()
SW_loss_std = np.array(dist).std()

print("Spectral weight for loss is {:.2f}({:.0f})".format(SW_loss_mean, SW_loss_std*1e2))

###
### full spectra transfer/loss
###
ax = axs[1]
x_name = 'ScaledDetuning'

# transfer
y_ratio = y/y_loss
yerr_ratio = y_ratio*np.sqrt((yerr/y)**2 + (yerr_loss/y_loss)**2)

sty = styles[2]
color = colors[2]
ax.errorbar(x, y_ratio, yerr=yerr_ratio, **sty)

fig.tight_layout()
plt.show()


