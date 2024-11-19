# -*- coding: utf-8 -*-
"""
2024 Nov 12
@author: Chip Lab

"""
from data_class import Data
from scipy.optimize import curve_fit
from library import plt_settings, markers, tint_shade_color, tintshade, pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c

def Saturation(x, A, x0):
	return A*(1-np.exp(-x/x0))

def satratio(x, x0):
	return x/x0*1/(1-np.exp(-x/x0))

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)

### filenames
file = "2024-09-17_C_e.dat"

### Omega Rabi calibrations
# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
VpptoOmegaR47 = 17.05/0.703 # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
VpptoOmegaR43 = 14.44/0.656 # kHz/Vpp - 2024-09-25 calibration 
phaseO_OmegaR = lambda VVA, freq: 2*pi*VpptoOmegaR47 * Vpp_from_VVAfreq(VVA, freq)

# ff from Sept 17th
ff = 0.93

#### PLOTTING #####
# initialize plots
fig, ax = plt.subplots(figsize=(6,5))

### plot settings
plt.rcParams.update(plt_settings) # from library.py
color = '#1f77b4' # default matplotlib color (that blueish color)
marker = markers[0]
light_color = tint_shade_color(color, amount=1+tintshade)
dark_color = tint_shade_color(color, amount=1-tintshade)
plt.rcParams.update({"figure.figsize": [12,8],
					 "font.size": 14,
					 "lines.markeredgecolor": dark_color,
					 "lines.markerfacecolor": light_color,
					 "lines.color": dark_color,
					 "lines.markeredgewidth": 2,
					 "errorbar.capsize": 0})

### ANALYSIS ###
xname = 'VVA'
plot_name = "Omega Rabi Squared (1/ms^2)"
bg_freq = 46.323

print("Analyzing", file)
run = Data(file)

run.data['c9'] = ff * run.data['c9']

bg_df = run.data.loc[(run.data.freq==bg_freq)]
run.data = run.data.drop(bg_df.index)

bg_c5 = bg_df.c5.mean()
e_bg_c5 = bg_df.c5.sem()
bg_c9 = bg_df.c9.mean()
e_bg_c9 = bg_df.c9.sem()

run.data['N'] = run.data.c5 - bg_c5 + run.data.c9*ff
run.data['transfer'] = (run.data.c5 - bg_c5)/(run.data.N)


run.data['OmegaR'] = phaseO_OmegaR(run.data.VVA, run.data.freq) * np.sqrt(0.31)
run.data['OmegaR2'] = (run.data['OmegaR'])**2
xname = 'OmegaR2'

run.group_by_mean(xname)

### PLOTTING ###
x = run.avg_data[xname]
y = run.avg_data['transfer']
yerr = run.avg_data['em_transfer']

ax.set(xlabel=r'rf power $\Omega_R^2$ (kHz$^2$)', ylabel='Transfer',
	   ylim=[-0.05, 0.5])
ax.errorbar(x, y, yerr=yerr)

# fit to saturation curve
p0 = [0.5, 10000]
popt, pcov = curve_fit(Saturation, x, y, p0=p0, sigma=yerr)
perr = np.sqrt(np.diag(pcov))

ax.plot(x, Saturation(x, *popt), '--')
ax.plot(x, Linear(x, popt[0]/popt[1], 0), '--')

print(r"A = {:.4f} ± {:.4f}, x_0 = {:.4f} ± {:.4f}".format(popt[0], 
										  perr[0], popt[1], perr[1]))


# run.avg_data['cut'] = np.where(run.avg_data[xname] < cut, 1, 0)
# fitdf = run.avg_data[run.avg_run.cut == 1]
# satdf = run.avg_data[run.avg_run.cut == 0]

# x = fitdf[xname]
# y = fitdf['transfer']
# yerr = fitdf['em_transfer']

# xsat = satdf[xname]
# ysat = satdf['transfer']
# yerrsat =satdf['em_transfer']

# ax.errorbar(x, y, yerr, marker=marker)
# ax.errorbar(xsat, ysat, yerrsat)

# popt, pcov = curve_fit(Linear,x,y)
# perr = np.sqrt(np.diag(pcov))
# xlist = np.linspace(min(x),max(x),1000)

# ax.plot(xlist,Linear(xlist,*popt), linestyle='-', marker='', label=label)
# ax.set(xlabel=plot_name, ylabel="Transfer")
# plot_pos = ax.get_position().get_points() # get plot position
# ax_width = plot_pos[1,0] - plot_pos[0,0] 
# ax_height = plot_pos[1,1] - plot_pos[0,1] 
# inset_pos = [plot_pos[0,0] + ax_width/1.8, plot_pos[0,1], 
# 		  ax_width/2.3, ax_height/2.3]
# inset_poses.append(inset_pos)

# ax.legend()
# print(r'm={:.2f}$\pm${:.2f},b={:.3f}$\pm${:.3f}'.format(popt[0],perr[0],popt[1],perr[1]))

# # plot residuals as inset
# yreslin = y - Linear(x, *popt)

# ax_inset = fig.add_axes([plot_pos[0,0], plot_pos[0,1], ax_width/2, ax_height/2])
# ax_inset.errorbar(x, yreslin, yerr)
# ax_inset.hlines(0, x.min(), x.max() ,ls='dashed')
# 	
# inset_poses = np.array(inset_poses)
# inset_poses[0:2,:] += np.array([0,ax_height/4,0,0])
# inset_poses[[1,3],:] += np.array([ax_width/6, 0, 0, 0])
# 	
# for ax, inset_pos in zip(fig.axes[4:], inset_poses):
# 	ax.set_position(inset_pos)
# 	
# fig.tight_layout()
# plt.show()
