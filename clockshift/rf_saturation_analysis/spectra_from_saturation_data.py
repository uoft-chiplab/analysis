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
import sys
if root not in sys.path:
	sys.path.append(root)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pickle as pkl

from scipy import integrate
from scipy.optimize import curve_fit

from library import styles, paper_settings, colors, GammaTilde, h, BlackmanFourier2, pi

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/np.abs(x0)))

def Linear(x, m):
	return m*x

### constants
Eb = 3980 # kHz # I guesstimated this from recent ac dimer spectra

trap_depth = 200 * 1.0 # should actually be 1.5 I think.
EF_avg = 19.2

def xstar(B, EF):
	return Eb/EF # hbar**2/mK/a13(B)**2 * (1-re/a13(Bfield))**(-1)

def transfer_function(f, a):
	# note the EFs are so similar in the datasets I've baked in the average
	# EF here to make this analysis a little easier.
	return a*f**(-3/2)/(1+f*EF_avg/Eb)  # binding energy in kHz


def transfer_function_no_FSE(f, a):
	return a*f**(-3/2)


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 9999)

debug = True
Save=True
# save files 
files = [
		 'HFT_saturation_curves.pkl', 
		 'res_saturation_curves.pkl',
		 ]

loaded_data = []

# grab dictionary lists from pickle files
for i, file in enumerate(files):
	with open(os.path.join(data_path, file), "rb") as input_file:
		loaded_data = loaded_data + pkl.load(input_file)
		
# turn dictionary list into dataframe
df = pd.DataFrame(loaded_data)

# filter out cold data sets
df = df.loc[df.ToTF > 0.52]

# arbitrarily set OmegaR2, I don't think this matters because we will use linear transfer
OmegaR2 = 1

df['ScaledDetuning'] = df.detuning/df.EF

slopes = ['slope', 'slope_l']
e_slopes = ['e_slope', 'e_slope_l']

for slope, e_slope, prefix in zip(slopes, e_slopes, ['', 'loss_']):
	
	# calculate linear transfer, at OmegaR2
	df[prefix+'Gamma_lin'] = OmegaR2*df[slope]
	df[prefix+'e_Gamma_lin'] = OmegaR2*df[e_slope]
	
	# calculate scaled transfer, add set OmegaR2
	df[prefix+'ScaledTransfer'] = df.apply(lambda x: GammaTilde(x[prefix+'Gamma_lin'], h*x.EF, 
								 np.sqrt(OmegaR2)*2*np.pi, x.pulse_time), axis=1)
	df[prefix+'e_ScaledTransfer'] = df[prefix+'ScaledTransfer']* \
							df[prefix+'e_Gamma_lin']/df[prefix+'Gamma_lin']
							
	final_state_correction = (1+df['detuning']/Eb)
	df[prefix+'Contact'] = 2*np.sqrt(2)*np.pi**2*df[prefix+'ScaledTransfer']*\
							(df['detuning']/df['EF'])**(3/2) * \
									final_state_correction
	df[prefix+'e_Contact'] = df[prefix+'Contact']* df[prefix+'e_ScaledTransfer']/\
								df[prefix+'ScaledTransfer']
						

# print relevant columns for data selection
print(df[['detuning', 'pulse_time', 'ToTF', 'ScaledTransfer', 'loss_ScaledTransfer', 'EF', 'e_slope_l']])


### plots
plt.rcdefaults()
plt.rcParams.update(paper_settings)
# plt.rcParams.update({
# 					"figure.figsize": [15,8],
# 					"font.size": 14,
# 					"lines.linewidth": 1.5,
# 					})
alpha = 0.25

fig, axes = plt.subplots(2,2, figsize=[6.8, 6])
axs = axes.flatten()

axs[0].set(xlabel=r"Detuning [$E_F$]", ylabel=r"Scaled Transfer, $\tilde\Gamma$", 
		   xlim=[-1, 1])
axs[1].set(xlabel=r"Detuning [$E_F$]", ylabel=r"$\alpha_c/\alpha_b$", 
		   xlim=[-2, 50], ylim=[0.0, 1.05])
axs[2].set(xlabel=r"Detuning [$E_F$]", ylabel=r"Scaled Transfer, $\tilde\Gamma$",
		   xscale='log', yscale='log')
axs[3].set(xlabel=r"Detuning [$E_F$]", ylabel=r"Contact, $\tilde C$",
		   ylim=[-0.5,1.5], xlim=[-2, 50])

###
### full spectra transfer and loss
###
ax = axs[0]
x_name = 'ScaledDetuning'

df = df.sort_values(x_name)

# blackman Fourier
xs = np.linspace(-1, 1, 100)
# ax.plot(xs, 10*BlackmanFourier2(2*pi*xs * 15/0.5), ':')
# ax.plot(xs, 10*BlackmanFourier2(xs * 15/0.5-2.4), ':')
ax.plot(xs, 16*BlackmanFourier2(2*pi*xs * 19/5), ':')

# SW_FT2 = integrate.trapezoid(10*BlackmanFourier2(2*pi*xs * 15/0.5), x=xs)
SW_FT2 = integrate.trapezoid(16*BlackmanFourier2(2*pi*xs * 19/5), x=xs)

# transfer
y_name = 'ScaledTransfer'
yerr_name = 'e_ScaledTransfer'
x = np.array(df[x_name])
y = np.array(df[y_name])
yerr = np.array(df[yerr_name])

sty = styles[0]
color = colors[0]
ax.errorbar(x, y, yerr=yerr, **sty, label=r'$\alpha_c=N_c/N$')
ax.plot(x, np.interp(x, x, y), '-', color=colors[0])

# loss
y_name = 'loss_ScaledTransfer'
yerr_name = 'loss_e_ScaledTransfer'
y_loss = np.array(df[y_name])
yerr_loss = np.array(df[yerr_name])

sty = styles[1]
color = colors[1]
ax.errorbar(x, y_loss, yerr=yerr_loss, **sty, label=r'$\alpha_b=(N_b^{bg}-N_b)/N$')
ax.plot(x, np.interp(x, x, y_loss), '-', color=colors[1])
#ax.set(yscale='log')
ax.legend()

###
### full spectra transfer/loss
###
ax = axs[1]
x_name = 'ScaledDetuning'

# transfer
y_ratio = y/y_loss
yerr_ratio = np.abs(y_ratio*np.sqrt((yerr/y)**2 + (yerr_loss/y_loss)**2))

sty = styles[2]
color = colors[2]
ax.errorbar(x, y_ratio, yerr=yerr_ratio, **sty)
ax.vlines(trap_depth/EF_avg, 0, 1.05, color='k', linestyle='--') 
handle = Line2D([0], [0], color='k', linestyle='--', marker='', label='trap depth')

ax.legend(handles=[handle])

###
### loglog full spectra transfer and loss
###
ax = axs[2]
x_name = 'ScaledDetuning'

df_below_trap_depth = df.loc[(df[x_name] < trap_depth/EF_avg) & \
							 (df[x_name] > 0)]
df_above_trap_depth = df.loc[df[x_name] > trap_depth/EF_avg]

# transfer below trap depth
y_name = 'ScaledTransfer'
yerr_name = 'e_ScaledTransfer'

x = np.array(df_below_trap_depth[x_name])
y = np.array(df_below_trap_depth[y_name])
yerr = np.array(df_below_trap_depth[yerr_name])

sty = styles[0]
ax.errorbar(x, y, yerr=yerr, **sty, label=r'$\alpha_c=N_c/N$')

# fit to both forms of the transfer rate equation, w/wout Final State Effect
popt, pcov = curve_fit(transfer_function_no_FSE, x, y, sigma=yerr, p0=[0.05])
perr = np.sqrt(np.diag(pcov))
popt_2, pcov_2 = curve_fit(transfer_function, x, y, sigma=yerr, p0=[0.05])
perr_2 = np.sqrt(np.diag(pcov_2))

xs = np.linspace(0.5, max(x), 100)

ax.plot(xs, transfer_function_no_FSE(xs, *popt), '-', color=colors[0])
ax.plot(xs, transfer_function(xs, *popt_2), '--', color=colors[0])

C_FSE = popt[0] * 2*np.sqrt(2)*np.pi**2
e_C_FSE = perr[0] * 2*np.sqrt(2)*np.pi**2

C = popt_2[0] * 2*np.sqrt(2)*np.pi**2
e_C = perr_2[0] * 2*np.sqrt(2)*np.pi**2

print("Contact from tranfser with FSE is {:.2f}({:.0f})".format(C_FSE, e_C_FSE*1e2))
print("Contact from transfer w/out FSE is {:.2f}({:.0f})".format(C, e_C*1e2))

# transfer above trap depth
x = np.array(df_above_trap_depth[x_name])
y = np.array(df_above_trap_depth[y_name])
yerr = np.array(df_above_trap_depth[yerr_name])

sty = styles[0].copy()
sty['mfc'] = 'w'
ax.errorbar(x, y, yerr=yerr, **sty)

# loss
y_name = 'loss_ScaledTransfer'
yerr_name = 'loss_e_ScaledTransfer'
x = np.array(df[x_name])
y_loss = np.array(df[y_name])
yerr_loss = np.array(df[yerr_name])

sty = styles[1]
ax.errorbar(x, y_loss, yerr=yerr_loss, **sty, label=r'$\alpha_b=(N_b^{bg}-N_b)/N$')

df_fit = df.loc[df[x_name] > 0]
x = df_fit[x_name]
y = df_fit[y_name]
yerr = df_fit[yerr_name]

# fit to both forms of the transfer rate equation, w/wout Final State Effect
popt, pcov = curve_fit(transfer_function_no_FSE, x, y, sigma=yerr, p0=[0.05])
perr = np.sqrt(np.diag(pcov))
popt_2, pcov_2 = curve_fit(transfer_function, x, y, sigma=yerr, p0=[0.05])
perr_2 = np.sqrt(np.diag(pcov_2))

xs = np.linspace(0.5, max(x), 100)

ax.plot(xs, transfer_function_no_FSE(xs, *popt), '-', color=colors[1], label=r'with FSC')
ax.plot(xs, transfer_function(xs, *popt_2), '--', color=colors[1], label=r'no FSC')

C_loss_FSE = popt[0] * 2*np.sqrt(2)*np.pi**2
e_C_loss_FSE = perr[0] * 2*np.sqrt(2)*np.pi**2

C_loss = popt_2[0] * 2*np.sqrt(2)*np.pi**2
e_C_loss = perr_2[0] * 2*np.sqrt(2)*np.pi**2

print("Contact from loss with FSE is {:.2f}({:.0f})".format(C_loss_FSE, e_C_loss_FSE*1e2))
print("Contact from loss w/out FSE is {:.2f}({:.0f})".format(C_loss, e_C_loss*1e2))

ax.vlines(trap_depth/EF_avg, 0, 1.0, color='k', linestyle='--') 
ax.legend()

if Save:
	df.to_csv('\\\\unobtainium\\E_Carmen_Santiago\\Analysis Scripts\\analysis\\clockshift\\manuscript\\manuscript_data\\HFT_2MHz_spectra.csv',
			  )
###
###  Contact
###
ax = axs[3]
x_name = 'ScaledDetuning'

df = df.sort_values(x_name)

# transfer
y_name = 'Contact'
yerr_name = 'e_Contact'
x = np.array(df[x_name])
y = np.array(df[y_name])
yerr = np.array(df[yerr_name])

sty = styles[0]
color = colors[0]
ax.errorbar(x, y, yerr=yerr, **sty, label=r'$\alpha_c=N_c/N$')

# loss
y_name = 'loss_Contact'
yerr_name = 'loss_e_Contact'
y_loss = np.array(df[y_name])
yerr_loss = np.array(df[yerr_name])

sty = styles[1]
color = colors[1]
ax.errorbar(x, y_loss, yerr=yerr_loss, **sty, label=r'$\alpha_b=(N_b^{bg}-N_b)/N$')
ax.plot(x, np.interp(x, x, y_loss), '-', color=colors[1])

ax.vlines(trap_depth/EF_avg, 0, 1.0, color='k', linestyle='--') 

ax.legend()
fig.tight_layout()
plt.show()


###
### integrate data
###
# select integral df below trap depth
df_int = df.loc[df[x_name] < trap_depth/EF_avg]

# reselect data
x = np.array(df_int[x_name])
y = np.array(df_int['ScaledTransfer'])
yerr = np.array(df_int['e_ScaledTransfer'])

y_loss = np.array(df_int['loss_ScaledTransfer'])
yerr_loss = np.array(df_int['loss_e_ScaledTransfer'])

SW = integrate.trapezoid(y, x=x)
SW_loss = integrate.trapezoid(y_loss, x=x)

limit = 200
SW = integrate.quad(lambda d: np.interp(d, x, y), min(x), max(x), limit=limit)
SW_loss = integrate.quad(lambda d: np.interp(d, x, y_loss), min(x), max(x), limit=limit)

# MC error of integral
num = 1000
dist = []
i = 0
while i < num:
	dist.append(integrate.trapezoid([np.random.normal(val, err) for val, err \
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
	dist.append(integrate.trapezoid([np.random.normal(val, err) for val, err \
				 in zip(y_loss, yerr_loss)], x=x))
	i += 1
SW_loss_mean = np.array(dist).mean()
SW_loss_std = np.array(dist).std()

print("Spectral weight for loss is {:.2f}({:.0f})".format(SW_loss_mean, SW_loss_std*1e2))