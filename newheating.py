# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Fit bulk viscosity from wiggle heating data, scanning
- time
- freq
- amp

Relies on data_class.py, library.py

Requires tabulate. In console, execute the command:
	!pip install tabulate
"""
from data_class import Data
from library import *
from scipy.optimize import curve_fit
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

data_folder = 'data/heating'
drive = '\\\\UNOBTAINIUM\\E_Carmen_Santiago' 

temp_param = "G_ToTFcalc"

plotting = True

test_time_plots = False
test_amp_plots = False
test_freq_plots = False
plot_legend = False

NO_OFFSET = False
EF_CUTOFF = 0

colors = ["blue", "red", 
		  "green", "orange", 
		  "purple", "black", "pink"]

def ToTFfunc(t, A, omega, zeta, EF, C):
	return 9/2*A**2*hbar*omega*omega*t*zeta/EF + C

def ToTFfunc_highfreq(t, A, omega, contact, EF, C):
	return 1/(2*np.sqrt(2)*pi)*A**2*(omega*EF/hbar)**(1/2)*t*contact + C

def ToTFfunc_tilman(omega, zeta, EF):
	return 9/2*hbar*omega**2/EF*zeta

def crappy_chi_sq(y, yfit, yerr, dof):
	return 1/dof * np.sum((np.array(y) - np.array(yfit))**2/(yerr**2))

### metadata

# 202.1 Feb 09 varying freq using Kevin's code 

F_0209_f5 = {'filename':'2024-02-09_F_e_freq=5','freq':5e3,'Bamp':0.54*1.8,'B':202.1,
		  'Ni':48997,'Ti':0.388, 'GTi':0.276}

F_0209_f10 = {'filename':'2024-02-09_F_e_freq=10','freq':10e3,'Bamp':0.54*1.8,'B':202.1,
		  'Ni':48997,'Ti':0.388, 'GTi':0.276}

F_0209_f30 = {'filename':'2024-02-09_F_e_freq=30','freq':30e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':48997,'Ti':0.388, 'GTi':0.276}

F_0209_f50 = {'filename':'2024-02-09_F_e_freq=50','freq':50e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':48997,'Ti':0.388, 'GTi':0.276}

F_0209_f150 = {'filename':'2024-02-09_F_e_freq=150','freq':150e3,'Bamp':0.054*1.8,'B':202.1,
		  'Ni':48997,'Ti':0.388, 'GTi':0.276}

Feb09_runs = [F_0209_f5,F_0209_f10,F_0209_f30,F_0209_f50,F_0209_f150]
Feb09label = "Feb09_ToTF1p4_202p1G"

wx = 151.6#*np.sqrt(1.5)
wy = 429#*np.sqrt(2)
wz = 442#*np.sqrt(2)
mean_trapfreq = 2*pi*(wx*wy*wz)**(1/3)
Bamp_per_Vpp = 0.07/1.8

class Data:
	def __init__(self, filename, path=None, column_names=None, 
			  exclude_list=None, average_by=None, metadata=None):
		self.filename = filename
		if metadata is not None:
			self.__dict__.update(metadata)  # Store all the extra variables
		
		file = glob(drive + '\\Analysis Scripts\\analysis\\data\\heating\\' + filename)[0] # EXTREMELY greedy ; for Fermium
			
		self.data = pd.read_table(file, delimiter=',') # making dataframe of chosen data
		
		if column_names:
			self.data = self.data[column_names]
		if exclude_list:
			self.exclude(exclude_list)
		if average_by:
			self.group_by_mean(average_by)

	# exclude list of points
	def exclude(self, exclude_list):
		self.data.drop(exclude_list)
		
	# group by scan name, compute mean 
	def group_by_mean(self, scan_name):
		mean = self.data.groupby([scan_name]).mean().reset_index()
		sem = self.data.groupby([scan_name]).sem().reset_index().add_prefix("em_")
		std = self.data.groupby([scan_name]).std().reset_index().add_prefix("e_")
		self.avg_data = pd.concat([mean, std, sem], axis=1)

				
# plot raw data or average
	def plot(self, names, label=None, axes_labels=None):
		self.fig = plt.figure()
		self.ax = plt.subplot()
		if label==None:
			label = self.filename

		if hasattr(self, 'avg_data'): # check for averaging
			self.ax.errorbar(self.avg_data[f"{names[0]}"], self.avg_data[f"{names[1]}"], 
				yerr=self.avg_data[f"em_{names[1]}"], capsize=2, marker='o', ls='',
				label=label)
		else:
			self.ax.plot(self.data[f"{names[0]}"], self.data[f"{names[1]}"], 'o',
				label = label)
			
		if axes_labels == None:
			axes_labels = [f"{names[0]}", f"{names[1]}"]
			
		self.ax.set_xlabel(axes_labels[0])
		self.ax.set_ylabel(axes_labels[1])
		self.ax.legend()
		