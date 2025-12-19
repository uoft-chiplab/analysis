# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:05:56 2024

@author: coldatoms
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
file_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
	plotting = True
else:
	plotting = False

# VVA to Vpp PHASEO
VVAtoVpp = 'VVAtoVpp_47MHz_squarePhaseO_4GSps_scope.txt'
print(f"PhaseO rf calibration using {VVAtoVpp}.")
file_path = os.path.join(file_dir, VVAtoVpp)
xname = 'VVA'
yname = 'Vpp'
cal_VVA_PO = pd.read_csv(file_path, sep='\t', skiprows=1, names=[xname,yname])
cal_x_PO = cal_VVA_PO['VVA']
cal_y_PO = cal_VVA_PO['Vpp']
calInterpVVA_PO = lambda x: np.interp(x, cal_x_PO, cal_y_PO)
if plotting == True:
	fig, ax = plt.subplots()
	xx = np.linspace(cal_VVA_PO[xname].min(), cal_VVA_PO[xname].max(),100)
	ax.plot(xx, calInterpVVA_PO(xx), '--')
	ax.plot(cal_VVA_PO[xname], cal_VVA_PO[yname], 'o')

# VVA to Vpp MICRO
VVAtoVpp = 'VVAtoVppMicro_43MHz.txt'
print(f"MicrO rf calibration using {VVAtoVpp}.")
file_path = os.path.join(file_dir, VVAtoVpp)
xname = 'VVA'
yname = 'Vpp'
cal_VVA_MO = pd.read_csv(file_path, sep='\t', skiprows=1, names=[xname,yname])
cal_x_MO = cal_VVA_MO['VVA']
cal_y_MO = cal_VVA_MO['Vpp']
calInterpVVA_MO = lambda x: np.interp(x, cal_x_MO, cal_y_MO)
if plotting == True:
	fig, ax = plt.subplots()
	xx = np.linspace(cal_VVA_MO[xname].min(), cal_VVA_MO[xname].max(),100)
	ax.plot(xx, calInterpVVA_MO(xx), '--')
	ax.plot(cal_VVA_MO[xname], cal_VVA_MO[yname], 'o')

# Freq to Vpp
FreqtoVpp = 'phaseofreq_to_Vpp_VVA_2p3_square.txt'
file_path = os.path.join(file_dir, FreqtoVpp)
xname = 'Freq'
yname = 'Vpp'
cal = pd.read_csv(file_path, sep='\t', skiprows=2, names=[xname,yname])
# normalize by 47 MHz value
cal['relVpp'] = cal['Vpp']/cal.loc[cal['Freq'] == 47]['Vpp'].values[0]
calInterpFreq = lambda x: np.interp(x, cal[xname], cal['relVpp'])
if plotting == True:
	fig, ax = plt.subplots()
	xx = np.linspace(cal[xname].min(), cal[xname].max(),100)
	ax.set(xlabel="Freq (MHz)", ylabel="Attenuation")
	ax.plot(xx, calInterpFreq(xx), '--')
	ax.plot(cal[xname], cal['relVpp'], 'o')

def Vpp_from_VVAfreq(VVA, freq, rfsource="phaseo"):
	''' Returns ... '''
	if rfsource == "phaseo":
		Vpp = calInterpVVA_PO(VVA)*calInterpFreq(freq)
	elif rfsource == "micro":
		Vpp = calInterpVVA_MO(VVA) * calInterpFreq(freq)
	return Vpp
