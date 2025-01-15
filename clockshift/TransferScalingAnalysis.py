# -*- coding: utf-8 -*-
"""
2024 July 09

@author: Chip Lab

This code is designed to analyze data taken on 2024-06-27 and 2024-07-08
to study the scaling of resonant transfer from interacting ab to ac dimer 
and ac free, respectively.

"""
from data_class import Data
from scipy.optimize import curve_fit
from library import plt_settings, markers, tint_shade_color, tintshade
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

### Fit functions
def Linear(x,m,b):
	return m*x + b

def Quadratic(x, a, b, c):
	return a*x**2 + b*x + c

# paths
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')

### filenames
dimer_vs_time_file = "2024-06-27_F_e.dat"
dimer_vs_Rabi_file = "2024-06-27_C_e.dat"
free_vs_time_file = "2024-07-08_H_e.dat"
free_vs_Rabi_file = "2024-07-08_I_e.dat"

### Omega Rabi calibrations
# dimer freq VVA cal file
cal_file_43MHz = os.path.join(data_path, 'VVAtoVpp_square_43p2MHz.txt')
cal_43MHz = pd.read_csv(cal_file_43MHz, sep='\t', skiprows=1, names=['VVA','Vpp'])
calibration43MHz = lambda x: np.interp(x, cal_43MHz['VVA'], cal_43MHz['Vpp'])

# free to free VVA cal file
# cal_file_47MHz = os.path.join(root, 'VVAtoVpp.txt')
# cal_47MHz = pd.read_csv(cal_file_47MHz, sep='\t', skiprows=1, names=['VVA','Vpp'])
# calibration47MHz = lambda x: np.interp(x, cal_47MHz['VVA'], cal_47MHz['Vpp'])

### Vpp calibration
# VpptoOmegaR = 27.5833 # kHz/Vpp, older calibration
VpptoOmegaR = 17.05/0.703  # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
OmegaR_from_VVAfreq = lambda Vpp, freq: VpptoOmegaR * Vpp_from_VVAfreq(Vpp, freq)

pulsearea_square = 1 # square pulse

# gain calibration for 47MHz
gain_list = [-9, -8, -7, -6, -5, -4, -2, 2, 6, 10]
Vpp_list = [3.6, 15.6, 23.6, 32.4, 41, 50, 67.2, 99.2, 134, 166]
gain_popt, pcov = curve_fit(Linear, gain_list, Vpp_list)
def gain_calibration(gain):
	return Linear(gain, *gain_popt)

def OmegaR(VVA, calibration):
	VpptoOmegaR = 17.05/0.703  # 27.5833 # kHz
	return  pulsearea_square*VpptoOmegaR*calibration(VVA)

# background from July 8th
bg_file = "2024-07-08_J_e.dat"
bg_data = Data(bg_file, path=data_path)
bg_c9 = bg_data.data.c9.mean()
bg_c5 = bg_data.data.c5.mean()
ff = 1.03

#### PLOTTING #####
# initialize plots
fig, axes = plt.subplots(2,2)

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
files = [dimer_vs_time_file, dimer_vs_Rabi_file, 
		 free_vs_time_file, free_vs_Rabi_file]
xnames = ['pulse time', 'VVA', 
		  'time', 'gain']
plot_names = ['Pulse Time (ms)', "Omega Rabi Squared (1/us^2)",
			  'Pulse Time (ms)', "Omega Rabi Squared (1/ms^2)"]
cutoffs = [0.07, np.infty, 
		   0.27, np.infty]
cals = [calibration43MHz, calibration43MHz, 
		lambda x: OmegaR_from_VVAfreq(x, 47.2227), 
		lambda x: OmegaR_from_VVAfreq(x, 47.2227)]
labels = ["Dimer", "Dimer", 
		  "Free to free", "Free to free"]
inset_poses = []

for file, xname, cal, cut, ax, plot_name, label in zip(files, xnames, cals, 
						   cutoffs, axes.flatten(), plot_names, labels):
	print("Analyzing", xname, "vs transfer from ", file)
	data = Data(file, path=data_path)
	
	if label == 'Dimer':
		transfershift = 0
		data.data['transfer'] = (data.data['sum95'].max() - \
			   data.data['sum95'])/data.data['sum95'].max() - transfershift
			
	else:
		data.data['N'] = data.data.c5 - bg_c5 + data.data.c9*ff
		data.data['transfer'] = (data.data.c5 - bg_c5)/(data.data.N)
	
	if xname == 'VVA':
		data.data['OmegaR'] = OmegaR(data.data['VVA'], cal)
		data.data['OmegaR2'] = (2*np.pi*data.data['OmegaR']/1000)**2
		xname = 'OmegaR2'
		
	elif xname == 'gain':
		VVA = 1.5
		data.data['OmegaR'] = gain_calibration(data.data['gain'])/\
						gain_calibration(10)*OmegaR(VVA, cal)
		data.data['OmegaR2'] = (2*np.pi*data.data['OmegaR'])**2
		xname = 'OmegaR2'
	
	data.group_by_mean(xname)
	
	
	data.avg_data['cut'] = np.where(data.avg_data[xname] < cut, 1, 0)
	fitdf = data.avg_data[data.avg_data.cut == 1]
	satdf = data.avg_data[data.avg_data.cut == 0]
	
	x = fitdf[xname]
	y = fitdf['transfer']
	yerr = fitdf['em_transfer']
	
	xsat = satdf[xname]
	ysat = satdf['transfer']
	yerrsat =satdf['em_transfer']
	
	ax.errorbar(x, y, yerr, marker=marker)
	ax.errorbar(xsat, ysat, yerrsat)
	
	popt, pcov = curve_fit(Linear,x,y)
	perr = np.sqrt(np.diag(pcov))
	xlist = np.linspace(min(x),max(x),1000)
	
	ax.plot(xlist,Linear(xlist,*popt), linestyle='-', marker='', label=label)
	ax.set(xlabel=plot_name, ylabel="Transfer")
	plot_pos = ax.get_position().get_points() # get plot position
	ax_width = plot_pos[1,0] - plot_pos[0,0] 
	ax_height = plot_pos[1,1] - plot_pos[0,1] 
	inset_pos = [plot_pos[0,0] + ax_width/1.8, plot_pos[0,1], 
			  ax_width/2.3, ax_height/2.3]
	inset_poses.append(inset_pos)
	
	ax.legend()
	print(r'm={:.2f}$\pm${:.2f},b={:.3f}$\pm${:.3f}'.format(popt[0],perr[0],popt[1],perr[1]))
	
	# plot residuals as inset
	yreslin = y - Linear(x, *popt)

	ax_inset = fig.add_axes([plot_pos[0,0], plot_pos[0,1], ax_width/2, ax_height/2])
	ax_inset.errorbar(x, yreslin, yerr)
	ax_inset.hlines(0, x.min(), x.max() ,ls='dashed')
	
inset_poses = np.array(inset_poses)
inset_poses[0:2,:] += np.array([0,ax_height/4,0,0])
inset_poses[[1,3],:] += np.array([ax_width/6, 0, 0, 0])
	
for ax, inset_pos in zip(fig.axes[4:], inset_poses):
	ax.set_position(inset_pos)
	
fig.tight_layout()
plt.show()
