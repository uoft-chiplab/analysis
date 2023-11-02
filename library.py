# -*- coding: utf-8 -*-
"""
2023-09-25
@author: Chip Lab

Functions to call in analysis scripts
"""
from scipy.constants import pi, hbar, h
from scipy.integrate import trapz, simps, cumtrapz
import numpy as np

uatom = 1.660538921E-27
a0 = 5.2917721092E-11
uB = 9.27400915E-24
gS = 2.0023193043622
gJ = gS
mK = 39.96399848 * uatom
ahf = -h * 285.7308E6 # For groundstate 
gI = 0.000176490 # total nuclear g-factor

def EhfFieldInTesla(B, F, mF):
	term1 = -ahf/4 + gI * uB * mF * B
	term2 = (2*(gJ - gI)*uB *B /ahf/9)
	term3 = (-1)**(F-1/2) *9 *ahf/4 *np.sqrt(1+4*mF/9 * term2 + term2**2)
	return term1 + term3
	
def Ehf(B, F, mF):
	return EhfFieldInTesla(1E-4 *B, F, mF)

def FreqMHz(B, F1, mF1, F2, mF2):
  return 1E-6 *( Ehf(B, F1, mF1) - Ehf(B, F2, mF2))/h

def Gaussian(x, A, x0, sigma, C):
	return A * np.exp(-(x-x0)**2/(2*sigma**2)) + C

def Lorentzian(x, A, x0, sigma, C):
	return A /((x-x0)**2 + (sigma/2)**2) + C

def FermiEnergy(n, w):
	return hbar * w * (6 * n)**(1/3)

def FermiWavenumber(n, w):
	return np.sqrt(2*mK*FermiEnergy(n, w))/hbar

def GammaTilde(transfer, EF, OmegaR, trf):
	return EF/(hbar * pi * OmegaR**2 * trf) * transfer

def ScaleTransfer(detuning, transfer, EF, OmegaR, trf):
	"""
	detuning [kHz]
	transfer is the transferred fraction of atoms
	OmegaR in [1/s]
	EF in [kHz]
	trf should be in [s]
	
	You can pass in OmegaR and EF as floats or arrays (and it will scale 
	appropriately assuming they are the same length as data and in the same 
	order).
	
	FIX THIS
	"""
	return 1

def SumRule(data):
	"""
	integrated with simpsons rule
	"""
	return [np.trapz(data[:,1], x=data[:,0]), 
		 cumtrapz(data[:,1], x=data[:,0])[-1],
		 simps(data[:,1], x=data[:,0])]

def FirstMoment(data):
	"""
	integrated with simpsons rule
	"""
	return [np.trapz(data[:,1]*data[:,0], x=data[:,0]), 
		 cumtrapz(data[:,1]*data[:,0], x=data[:,0])[-1],
		 simps(data[:,1]*data[:,0], x=data[:,0])]

def tail3Dswave(w, C, gamma):
	return C*w**gamma

def guessACdimer(field):
	return -0.1145*field + 27.13 # MHz

def a97(B, B0=202.14, B0zero=209.07, abg=167.6*a0): 
	return abg * (1 - (B0zero - B0)/(B - B0));

