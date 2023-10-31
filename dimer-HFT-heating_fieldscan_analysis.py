# -*- coding: utf-8 -*-
"""
@author: Chip Lab

Analysis of 3D s-wave contact vs. field using
- ac dimer association
- high-frequency tail (HFT) transfer
- heating rate

Relies on data_class.py, library.py

Requires tabulate. In console, execute the command:
	!pip install tabulate
"""
from data_class import Data
from library import *
from scipy import interpolate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

DIMERFACTOR = 1

data_folder = 'data'
plot_raw_data = False
plotting = True

# metadata
HFT_metadata = {'ff':0.56,'loss':0.52,'TShotsN':78269,'c5bg_mean':0,
				'mean_trapfreq':2*pi*(151.6*429*442)**(1/3),
				'OmegaR':99.3E3*np.sqrt(0.3)*2*pi,'detuning':150e3,'trf':30E-6,
				'trap_depth':4.7*20,'prefactor':2*np.sqrt(2)*pi**2}

dimer_metadata = {'ff':0.73,'TShotsN':94853,
				'mean_trapfreq':2*pi*(151.6*429*442)**(1/3),
				'OmegaR':99.30*2*pi*1e3*1.52/3.6,'trf':20E-6,
				'trap_depth':4.7*20}

heating_metadata = {'omega':2*pi*100e3,'time':1e-3,'A':0.1,
					'mean_trapfreq':2*pi*(151.6*429*442)**(1/3)}

# load data, and combine both sets of HFT
HFT_filename = '2023-09-18_E_e.dat'
HFT_filename2 = '2023-09-18_E2_e.dat'
dimer_filename = '2023-09-28_L_e.dat'
dimer_bg_filename = '2023-10-26_C_e.dat'
heating_filename = '2023-10-19_E_e.dat'
heating_bg_filename = '2023-10-19_C_e.dat'

HFT = Data(HFT_filename, path = data_folder, metadata=HFT_metadata)
HFT.data = pd.concat([HFT.data, Data(HFT_filename2, path = data_folder).data])
dimer = Data(dimer_filename, path = data_folder, metadata=dimer_metadata)
dimer_bg = Data(dimer_bg_filename, path = data_folder)
heating = Data(heating_filename, path = data_folder, metadata = heating_metadata)
heating_bg = Data(heating_bg_filename, path = data_folder)

# plot raw data
if plot_raw_data == True:
	HFT.plot(['field', 'c5'])
	dimer.plot(['field', 'sum95'])
	dimer_bg.plot(['field', 'sum95'])
	heating.plot(['Field','ToTFcalc'])
	heating_bg.plot(['Field','ToTFcalc'])
	
dimer_bg.data["c5pc9"] = (dimer_bg.data["c5"] + dimer.ff*dimer_bg.data["c9"])

dimer_bg.group_by_mean('field')
heating_bg.group_by_mean('Field')
loss97i = interpolate.interp1d(np.array(dimer_bg.avg_data["field"]),
							  np.array(dimer_bg.avg_data["c5pc9"]),
							  fill_value='extrapolate')

dimer_bg.bg_c5pc9 = dimer_bg.avg_data["c5pc9"].loc[15:20].mean()
loss97 = lambda B: loss97i(B)/dimer_bg.bg_c5pc9
	
# calculate HFT transfer
HFT.c9bg_mean = HFT.data["c9"].mean()
HFT.data["transfer"] = (HFT.data["c5"] - HFT.c5bg_mean)/ \
			(HFT.data["c5"] - HFT.c5bg_mean + HFT.ff*HFT.data["c9"])
HFT.data["total_atoms"] = (HFT.data["c5"] - HFT.c5bg_mean + \
							   HFT.ff*HFT.data["c9"])/(HFT.ff*HFT.c9bg_mean) \
								* HFT.TShotsN #* loss
								
HFT.data["kF"] = HFT.data["total_atoms"].apply(lambda n: \
							   FermiWavenumber(n/2, HFT.mean_trapfreq))
HFT.data["EF"] = HFT.data["total_atoms"].apply(lambda n: \
							   FermiEnergy(n/2, HFT.mean_trapfreq))
	
HFT.data["scaled_transfer"] = HFT.data["EF"]/(hbar * pi * HFT.OmegaR**2 * HFT.trf) \
									* HFT.data["transfer"]
HFT.data["CokFN"] = (HFT.detuning*h/HFT.data["EF"])**(3/2) * HFT.prefactor \
									* HFT.data["scaled_transfer"]

# calculate dimer transfer
dimer.c9bg_mean = 1.67e4
dimer.c5bg_mean = 1.36e4
dimer.data['total_atoms'] = (dimer.data['c5'] + dimer.ff*dimer.data['c9'])/ \
	(dimer.c5bg_mean + dimer.ff*dimer.c9bg_mean) * dimer.TShotsN #* loss

dimer.data['transfer_c5'] = 1 - (dimer.data['c5']/dimer.c5bg_mean)
dimer.data['transfer_c9'] = 1 - (dimer.data['c9']/dimer.c9bg_mean)
dimer.data['transfer'] = 1 - (dimer.data['c5'] + dimer.ff*dimer.data['c9'])/ \
				((dimer.c5bg_mean + dimer.ff*dimer.c9bg_mean)*dimer.data['field'].apply(loss97))
				
dimer.EF = FermiEnergy(dimer.TShotsN/2, dimer.mean_trapfreq)
dimer.kF = FermiWavenumber(dimer.TShotsN/2, dimer.mean_trapfreq)

dimer.data["scaled_transfer"] = dimer.EF/(hbar * pi * dimer.OmegaR**2 * dimer.trf) \
									* dimer.data["transfer"]
									
# calculate heating parameters
heating.EF = FermiEnergy(heating.data["LiNfit"].mean(), heating.mean_trapfreq)
									
def contact_dimer_transfer(scaled_transfer):
	aac = np.sqrt(hbar**2/(2*mK*4e6*h)) # assuming 4MHz dimer energy
	kFaac = aac * dimer.kF # 0.071, slightly larger than 0.066 for dimer spectra
	omega_bar_per_scaled_transfer = 1810 # assuming -5.9 EF omega bar for 0.00325 scaled transfer
	return pi*kFaac*omega_bar_per_scaled_transfer*scaled_transfer

dimer.data["CokFN"] = dimer.data["scaled_transfer"].apply(contact_dimer_transfer)

# group data by mean
HFT.group_by_mean('field')
dimer.group_by_mean('field')
heating.group_by_mean('Field')

# subtract heating rate bg, combine uncertainties
heating.avg_data['DToTFcalc'] = heating.avg_data['ToTFcalc'] - \
		heating_bg.avg_data['ToTFcalc']
heating.avg_data['e_DToTFcalc'] = np.sqrt(heating.avg_data['e_ToTFcalc'].pow(2) + \
		heating_bg.avg_data['e_ToTFcalc'].pow(2))
heating.avg_data['em_DToTFcalc'] = np.sqrt(heating.avg_data['em_ToTFcalc'].pow(2) + \
		heating_bg.avg_data['em_ToTFcalc'].pow(2))
	
# calculate contact
heating.delta = heating.omega * hbar/heating.EF
heating.avg_data['CokFN'] = np.sqrt(2/heating.delta)/heating.A**2 * \
		h/heating.time * heating.avg_data['DToTFcalc']/heating.EF
heating.avg_data['e_CokFN'] = np.sqrt(2/heating.delta)/heating.A**2 * \
		h/heating.time* heating.avg_data['e_DToTFcalc']/heating.EF
heating.avg_data['em_CokFN'] = np.sqrt(2/heating.delta)/heating.A**2 * \
		h/heating.time * heating.avg_data['em_DToTFcalc']/heating.EF

if plotting == True:
	plt.rcParams.update({"figure.figsize": [12,8]})
	fig, axs = plt.subplots(2, 2)
	xlabel = "Field (G)"
######### PLOTTING ARBITRARY SCALE ############
	ylabel = " \"Contact \" (arb.)"
	axs[0,0].set(ylabel=ylabel)
	
	heating_scale = 1/heating.avg_data['DToTFcalc'].max()
	dimer_scale = 1/dimer.avg_data['transfer'].max()
	HFT_scale = 1/HFT.avg_data['transfer'].max()
	
	axs[0,0].errorbar(np.array(HFT.avg_data['field']),HFT_scale*np.array(HFT.avg_data['transfer']), 
				 yerr = HFT_scale*np.array(HFT.avg_data['em_transfer']), fmt='go', label="high-freq transfer",
				 capsize=2)
	axs[0,0].errorbar(np.array(dimer.avg_data['field']),dimer_scale*np.array(dimer.avg_data['transfer']), 
				 yerr = dimer_scale*np.array(dimer.avg_data['em_transfer']), fmt='bo', 
				 label="dimer transfer", capsize=2)
	axs[0,0].errorbar(np.array(heating.avg_data['Field']),heating_scale*np.array(heating.avg_data['DToTFcalc']), 
				 yerr = heating_scale*np.array(heating.avg_data['em_DToTFcalc']), fmt ='ro', 
				 label="heating rate", capsize=2)
	
	axs[0,0].legend()
	
######### PLOTTING SCALED TRANSFER ############
	ylabel = "Transfer"
	axs[1,0].set(ylabel=ylabel,xlabel=xlabel)
	
	axs[1,0].errorbar(np.array(HFT.avg_data['field']),np.array(HFT.avg_data['transfer']), 
				 yerr = np.array(HFT.avg_data['em_transfer']), fmt='go', label="high-freq transfer",
				 capsize=2)
	axs[1,0].errorbar(np.array(dimer.avg_data['field']), np.array(dimer.avg_data['transfer']), 
				 yerr = np.array(dimer.avg_data['em_transfer']), fmt='bo', 
				 label="dimer transfer", capsize=2)
	
	axs[1,0].legend()

######### PLOTTING SCALED TRANSFER ############
	ylabel = "Scaled Transfer"
	axs[0,1].set(ylabel=ylabel)
	axs[0,1].errorbar(np.array(HFT.avg_data['field']),np.array(HFT.avg_data['scaled_transfer']), 
				 yerr = np.array(HFT.avg_data['em_scaled_transfer']), fmt='go', label="high-freq transfer",
				 capsize=2)
	axs[0,1].errorbar(np.array(dimer.avg_data['field']),np.array(dimer.avg_data['scaled_transfer']), 
				 yerr = np.array(dimer.avg_data['em_scaled_transfer']), fmt='bo', 
				 label="dimer transfer", capsize=2)
	axs[0,1].legend()

######### PLOTTING CONTACT ############
	ylabel = "Contact per atom (kF)"
	axs[1,1].set(ylabel=ylabel, xlabel=xlabel)
	axs[1,1].errorbar(np.array(HFT.avg_data['field']),np.array(HFT.avg_data['CokFN']), 
				 yerr = np.array(HFT.avg_data['em_CokFN']), fmt='go', label="high-freq transfer",
				 capsize=2)
	axs[1,1].errorbar(np.array(dimer.avg_data['field']),np.array(dimer.avg_data['CokFN']), 
				 yerr = np.array(dimer.avg_data['em_CokFN']), fmt='bo', 
				 label="dimer transfer", capsize=2)
	axs[1,1].errorbar(np.array(heating.avg_data['Field']),np.array(heating.avg_data['CokFN']), 
				 yerr = np.array(heating.avg_data['em_CokFN']), fmt ='ro', 
				 label="heating rate", capsize=2)
	
	axs[1,1].legend()
	plt.show()
		
	
	
	
		
	





