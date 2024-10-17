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

# VVA to Vpp
VVAtoVpp = 'VVAtoVpp_47MHz_squarePhaseO_4GSps_scope.txt'
file_path = os.path.join(file_dir, VVAtoVpp)
xname = 'VVA'
yname = 'Vpp'
cal_VVA = pd.read_csv(file_path, sep='\t', skiprows=1, names=[xname,yname])
cal_x = cal_VVA['VVA']
cal_y = cal_VVA['Vpp']
calInterpVVA = lambda x: np.interp(x, cal_x, cal_y)
if plotting == True:
	fig, ax = plt.subplots()
	xx = np.linspace(cal_VVA[xname].min(), cal_VVA[xname].max(),100)
	ax.plot(xx, calInterpVVA(xx), '--')
	ax.plot(cal_VVA[xname], cal_VVA[yname], 'o')

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

def Vpp_from_VVAfreq(VVA, freq):
	''' Returns ... '''
	Vpp = calInterpVVA(VVA)*calInterpFreq(freq)
	return Vpp

