# -*- coding: utf-8 -*-
"""
2023-10-05
@author: Chip Lab

Plotting functions for general analysis scripts 
"""

from analysisfunctions import * # includes numpy and constants
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import scipy.optimize as curve_fit
from tabulate import tabulate # pip install tabulate
from data import *

# plots 

def plots(filename, names=['delaytime','field'], guess=None, fittype='Cos'):
	fitdata = data(filename, names)
	plt.title(f"Cos fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2], fitdata[3], 'go')
	if guess is None:	
		guess = [-0.2, 0, 10, 202]
	popt, pcov = curve_fit.curve_fit(fittype, fitdata[2], fitdata[3],p0=guess)
	ym = fittype(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	
	return fig1


	errors = np.sqrt(np.diag(pcov))
	freq = popt[1]/2/3.14
	period = 1/freq
	delay = popt[2] % (3.141592654) /popt[1]
	values = list([*popt, freq, period, delay])
	errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
	print(tabulate([['Values', *values], ['Errors', *errors]], headers=['Amplitude','omega','phase','offset', 'freq', 'period', 'delay']))
	
	
# residuals 

def residuals(filename, names):
	fitdata = data(filename, names)
	guess = [-0.2, 0, 10, 202]
	popt, pcov = curve_fit.curve_fit(Cos, fitdata[2], fitdata[3],p0=guess)
	residuals = fitdata[3] - Cos(fitdata[2],*popt)
	fig2 = plt.figure(1)
	plt.plot(fitdata[2],fitdata[3]*0,'-')
	plt.plot(fitdata[2], residuals, 'g+')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel +" Residuals")
	return fig2

