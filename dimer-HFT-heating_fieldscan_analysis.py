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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

DIMERFACTOR = 1

data_folder = 'data'
plot_raw_data = False

# metadata
HFT_metadata = {'ff':0.56,'loss':0.52,'TShotsN':78269,'c5bg_mean':0,
				'mean_trapfreq':2*pi*(151.6*429*442)**(1/3),
				'OmegaR':99.3E3*np.sqrt(0.3)*2*pi,'detuning':150e3,'trf':30E-6,
				'trap_depth':4.7*20,'prefactor':2*np.sqrt(2)*pi**2}

dimer_metadata = {'ff':0.73,'TShotsN':94853,
				'mean_trapfreq':2*pi*(151.6*429*442)**(1/3),
				'OmegaR':99.30*2*pi*1e3*1.52/3.6,'trf':20E-6,
				'trap_depth':4.7*20}

# load data, and combine both sets of HFT
HFT_filename = os.path.join(data_folder,'2023-09-18_E_e.dat')
HFT_filename2 = os.path.join(data_folder,'2023-09-18_E2_e.dat')
dimer_filename = os.path.join(data_folder,'2023-09-28_L_e.dat')
heating_filename = os.path.join(data_folder,'2023-10-19_E_e.dat')
heating_bg_filename = os.path.join(data_folder,'2023-10-19_C_e.dat')

HFT = Data(HFT_filename, metadata=HFT_metadata)
HFT.data = pd.concat([HFT.data, Data(HFT_filename2).data])
dimer = Data(dimer_filename, metadata=dimer_metadata)
heating = Data(heating_filename)
heating_bg = Data(heating_bg_filename)

# plot raw data
if plot_raw_data == True:
	HFT.plot(['field', 'c5'])
	dimer.plot(['field', 'sum95'])
	heating.plot(['Field','ToTFcalc'])
	heating_bg.plot(['Field','ToTFcalc'])
	
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
dimer.c9bg_mean = 1.6e4
dimer.c5bg_mean = 1.3e4
dimer.data['total_atoms'] = (dimer.data['c5'] + dimer.ff*dimer.data['c9'])/ \
	(dimer.c5bg_mean + dimer.ff*dimer.c9bg_mean) * dimer.TShotsN #* loss

dimer.data['transfer_c5'] = 1 - (dimer.data['c5']/dimer.c5bg_mean)
dimer.data['transfer_c9'] = 1 - (dimer.data['c9']/dimer.c9bg_mean)
dimer.data['transfer'] = 1 - (dimer.data['c5'] + dimer.ff*dimer.data['c9'])/ \
				(dimer.c5bg_mean + dimer.ff*dimer.c9bg_mean)
				
dimer.EF = FermiEnergy(dimer.TShotsN/2, dimer.mean_trapfreq)

dimer.data["scaled_transfer"] = dimer.EF/(hbar * pi * dimer.OmegaR**2 * dimer.trf) \
									* dimer.data["transfer"]
dimer.data["CokFN"] = DIMERFACTOR * dimer.data["scaled_transfer"]

# group data by mean
HFT.group_by_mean('field')
dimer.group_by_mean('field')
heating.group_by_mean('Field')
heating_bg.group_by_mean('Field')

# subtract heating rate bg, combine uncertainties
heating.avg_data['DToTFcalc'] = heating.avg_data['ToTFcalc'] - \
		heating_bg.avg_data['ToTFcalc']
heating.avg_data['e_DToTFcalc'] = np.sqrt(heating.avg_data['e_ToTFcalc'].pow(2) + \
		heating_bg.avg_data['e_ToTFcalc'].pow(2))
heating.avg_data['em_DToTFcalc'] = np.sqrt(heating.avg_data['em_ToTFcalc'].pow(2) + \
		heating_bg.avg_data['em_ToTFcalc'].pow(2))
	
######### PLOTTING ARBITRARY SCALE ############
plt.figure(0)
plt.xlabel("Field (G)")
plt.ylabel(" \" Contact \" (arb.)")

heating_scale = 1.4
dimer_scale = 0.7

plt.errorbar(np.array(HFT.avg_data['field']),np.array(HFT.avg_data['transfer']), 
			 yerr = np.array(HFT.avg_data['em_transfer']), fmt='go', label="high-freq transfer",
			 capsize=2)
plt.errorbar(np.array(dimer.avg_data['field']),dimer_scale*np.array(dimer.avg_data['transfer']), 
			 yerr = dimer_scale*np.array(dimer.avg_data['em_transfer']), fmt='bo', 
			 label="dimer transfer", capsize=2)
plt.errorbar(np.array(heating.avg_data['Field']),heating_scale*np.array(heating.avg_data['DToTFcalc']), 
			 yerr = heating_scale*np.array(heating.avg_data['em_DToTFcalc']), fmt ='ro', 
			 label="heating rate", capsize=2)

plt.legend()
plt.show()

######### PLOTTING CONTACT ############
plt.figure(1)
plt.xlabel("Field (G)")
plt.ylabel(" Contact per atom (kF)")

heating_scale = 1.4
dimer_scale = 0.7

plt.errorbar(np.array(HFT.avg_data['field']),np.array(HFT.avg_data['CokFN']), 
			 yerr = np.array(HFT.avg_data['em_transfer']), fmt='go', label="high-freq transfer",
			 capsize=2)
plt.errorbar(np.array(dimer.avg_data['field']),dimer_scale*np.array(dimer.avg_data['CokFN']), 
			 yerr = dimer_scale*np.array(dimer.avg_data['em_transfer']), fmt='bo', 
			 label="dimer transfer", capsize=2)
plt.errorbar(np.array(heating.avg_data['Field']),heating_scale*np.array(heating.avg_data['DToTFcalc']), 
			 yerr = heating_scale*np.array(heating.avg_data['em_DToTFcalc']), fmt ='ro', 
			 label="heating rate", capsize=2)

plt.legend()
plt.show()
	



	
	





