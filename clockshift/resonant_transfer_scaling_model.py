# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:19:33 2024

@author: Chip Lab

Estimating 7-5 free-to-free transfer saturation curve during
HFT scans for use in rescaling transfer near free-to-free feature
"""

import numpy as np
import matplotlib.pyplot as plt
from data_class import Data
from scipy.optimize import curve_fit
from library import plt_settings, styles, pi, colors
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
import pandas as pd

trf = 0.2  # ms
f75 = 47.2227  # MHz, frequency from 7 to 5

def Coherent_Transfer(OmegaR, delta, trf):
	OmegaR_eff = np.sqrt(delta**2 + OmegaR**2)
	return OmegaR**2/(delta**2 + OmegaR**2) * np.sin(2*pi*OmegaR_eff*trf/2)**2

def Saturation(OmegaR2, A, x0):
	return A*(1-np.exp(-OmegaR2/x0))

popt = np.array([1.3913, 3.50702])
xs = np.linspace(0, 5, 1000)  # linspace of rf powers
Gammas_Sat = Saturation(xs, *popt)
Gammas_Lin = xs*popt[0]/popt[1]

def res_sat_correction(Gamma):
	'''Interpolates saturation curve of transferred fraction for resonant
	7 to 5. Returns the linear term, i.e. the unsaturated transfer.'''
	return np.interp(Gamma, Gammas_Sat, Gammas_Lin)

if __name__ == '__main__':
	### Vpp calibration
	VpptoOmegaR = 17.05/0.703  # kHz/Vpp - 2024-09-16 calibration with 4GS/s scope measure of Vpp
	OmegaR_from_VVAfreq = lambda VVA, freq: VpptoOmegaR * Vpp_from_VVAfreq(VVA, freq)
	
	Omega_R = lambda x: OmegaR_from_VVAfreq(x, f75) * np.sqrt(0.31)
	
	detunings = [-7, -5, -3, -1, 0]
	
	# plotting with omega_R
	fig, axes = plt.subplots(1,2, figsize=(10,5))
	fig.suptitle("Coherent transfer for $t_{rf}=$"+str(int(trf*1e3))+"$\mu$s Blackman 7to5 free to free")
	axs = axes.flatten()
	
	VVAs = np.linspace(1.2, 2.0, 10000)
	OmegaRs = Omega_R(VVAs)
	
	# set axes options
	axs[0].set(xlabel=r"$\Omega_R$ [kHz]", ylabel="Transfer"
			)
	axs[1].set(xlabel=r"$\Omega_R^2$ [kHz$^2$]", ylabel="Transfer", 
# 		   ylim=(-0.05, 0.05),
		   )
	
	for i, detuning in enumerate(detunings):
		Transfer = Coherent_Transfer(OmegaRs, detuning, trf*np.ones(len(OmegaRs)))
		color = colors[i]
		
		label = f'{detuning} kHz'
		
		ax = axs[0]
		ax.plot(OmegaRs, Transfer, '-', label=label, color=color)
		
		ax = axs[1]
		ax.plot(OmegaRs**2, Transfer, '-', label=label, color=color)
	
# 	popt, pcov = curve_fit(Saturation, OmegaRs**2, Transfer)
# 	label_sat = r'saturating fit $\Gamma(\Omega_R^2) = \Gamma_{sat}(1-e^{-\Omega_R^2/\Omega_e^2})$'
# 	label_lin = r'linear term $\Gamma(\Omega_R^2) = \Gamma_{sat} \Omega_R^2/\Omega_e^2$'
# 	ax.plot(OmegaRs**2, Saturation(OmegaRs**2, *popt), '--', label=label_sat)
# 	ax.plot(OmegaRs**2, popt[0]/popt[1]*OmegaRs**2, '--', label=label_lin)
	ax.legend()
	
	fig.tight_layout()
	plt.show()
	
	
	# plotting with VVA
	fig, axes = plt.subplots(1,2, figsize=(10,5))
	fig.suptitle("Coherent transfer for $t_{rf}=$"+str(int(trf*1e3))+"$\mu$s Blackman 7to5 free to free")
	axs = axes.flatten()
	
	VVAs = np.linspace(1.1, 1.4, 20)
	OmegaRs = Omega_R(VVAs)
	Transfer = Coherent_Transfer(OmegaRs, 0, trf*np.ones(len(OmegaRs)))
	
	ax = axs[0]
	ax.plot(VVAs, Transfer)
	ax.set(xlabel=r"VVA", ylabel="Transfer")
	ax = axs[1]
	ax.plot(VVAs**2, Transfer)
	ax.set(xlabel=r"VVA$^2$", ylabel="Transfer")
	
	popt, pcov = curve_fit(Saturation, OmegaRs**2, Transfer)
	label_sat = r'saturating fit $\Gamma(\Omega_R^2) = \Gamma_{sat}(1-e^{-\Omega_R^2/\Omega_e^2})$'
	label_lin = r'linear term $\Gamma(\Omega_R^2) = \Gamma_{sat} \Omega_R^2/\Omega_e^2$'
	ax.plot(VVAs**2, Saturation(OmegaRs**2, *popt), '--', label=label_sat)
	ax.plot(VVAs**2, popt[0]/popt[1]*OmegaRs**2, '--', label=label_lin)
	ax.legend()
	
	fig.tight_layout()
	plt.show()
	
	plt.figure(figsize=(6,4))
	plt.plot(VVAs**2, popt[0]/popt[1]*OmegaRs**2/Saturation(OmegaRs**2, *popt))
	plt.xlabel("VVA$^2$")
	plt.ylabel("Correction factor")
	plt.show()


