# -*- coding: utf-8 -*-
"""
2023-09-29
@author: Chip Lab

Fitting functions for general analysis scripts 
"""

import numpy as np
import math

def Gaussian(x, A, x0, sigma, C):
	return A * np.exp(-(x-x0)**2/(2*sigma**2)) + C

def Lorentzian(x, A, b, x0, sigma, C):
	return (A*b**2) /((x-x0)**2 + (sigma)**2) + C

def Sin(x, A, omega, p, C):
	return A*np.sin(omega*x - p) + C

def Cos(x, A, omega, p, C):
	return A*np.cos(omega*x - p) + C

def Sinc(x, A, x0, sigma, C):
	return A*(np.sinc((x-x0) / sigma)) + C # normalized sinc (has pi included)

def Sinc2(x, A, x0, sigma, C):
	return A*(np.sinc((x-x0) / sigma))**2 + C

def TrapFreq(x, A, b, l, x0, C, D):
	return A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C + D*x

def TrapFreq2(x, A, b, l, x0, C):
	return A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C 

def RabiFreq(x, A, b, x0, C):
	return A*(np.sin(b/2 * x - x0))**2 + C

def Parabola(x, A, x0, C):
	return A*(x - x0)**2 + C

def Linear(x, m, b):
	return m*x + b 

def Expontial(x, A, sigma):
	return A*np.exp(-x/sigma)

def RabiLine(x, b, l, m, A, s, j, k, p):
	return (b**2 / (l**2 + (x - m)**2 ) ) * (A * np.sin(np.sqrt(s**2 + (x - j)**2 ) * k)**2 + p )

def ErfcFit(x, A, x0, b, C):
	return A * math.erfc((x - x0) / b ) + C