# -*- coding: utf-8 -*-
"""
2023-10-05
@author: Chip Lab

Plotting functions for general analysis scripts 
"""

from data import *

# residuals 

def residuals(filename):
	fitdata = data(filename)
	guess = [-0.2, 0, 10, 202]
	popt, pcov = curve_fit.curve_fit(Cos, fitdata[2], fitdata[3],p0=guess)
	residuals = fitdata[3] - Cos(fitdata[2],*popt)
	fig2 = plt.figure(1)
	plt.plot(fitdata[2],fitdata[3]*0,'-')
	plt.plot(fitdata[2], residuals, 'g+')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel +" Residuals")
	return fig2

