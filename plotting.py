# -*- coding: utf-8 -*-
"""
2023-10-05
@author: Chip Lab

Plotting functions for general analysis scripts 
"""

from analysisfunctions import * # includes numpy and constants
import matplotlib.pyplot as plt
import scipy.optimize as curve_fit
from tabulate import tabulate # pip install tabulate
from data import *
import pandas as pd
from scipy.stats import chisquare

# All of the functions you can fit to 

def fitting_type(filename, datatype='raw', names=['amp (Vpp)','ToTFcalc'], avg=False,  fittype='Sin', guess=None):
# 	if avg is True:
# 		fitdata = avgdata_data(filename, datatype, names)
# 	else:
	if datatype == 'raw':
		fitdata = data(filename, datatype,  names)
	elif datatype == 'exclude':
		fitdata = data_exclude(filename, datatype, names)
	elif datatype == 'exclude multiple points':
		fitdata = data_exclude_points(filename, datatype, names)
	elif datatype == 'avg':
		fitdata = avgdata_data(filename, datatype, names)	
	else:
		fitdata = 'nothing'
	if fittype == 'Cos':
		if guess is None:	
			guess = [-0.2, 0, 10, 202]
		popt, pcov = curve_fit.curve_fit(Cos, fitdata[2], fitdata[3],p0=guess)
		ym = Cos(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Cos(fitdata[2],*popt)
		headers = ['Amplitude','freq','phase','offset', 'period', 'delay']
	if fittype == 'Sin':
		if guess is None:	
			guess = [(max(fitdata[3])-min(fitdata[3])),0.05,-2,21]
		popt, pcov = curve_fit.curve_fit(Sin, fitdata[2], fitdata[3],p0=guess)
		ym = Sin(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Sin(fitdata[2],*popt)
		headers = ['Amplitude', 'freq','phase','offset', 'period', 'delay']
	if fittype == 'Gaussian':
		if guess is None:	
			guess = [-(max(fitdata[3])-min(fitdata[3])),fitdata[2][fitdata[3].argmin()],0.04,np.mean(fitdata[3])]
		popt, pcov = curve_fit.curve_fit(Gaussian, fitdata[2], fitdata[3],p0=guess)
		ym = Gaussian(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Gaussian(fitdata[2],*popt)
		headers = ['Amplitdue','Center','sigma','Offset']
	if fittype == 'Lorentzian':
		if guess is None:
			guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[2])), 1, fitdata[2][fitdata[3].argmin()], 0.04, np.mean(fitdata[3])]
		popt, pcov = curve_fit.curve_fit(Lorentzian, fitdata[2], fitdata[3],p0=guess)
		ym = Lorentzian(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Lorentzian(fitdata[2],*popt)
		headers = ['Amplitdue','b','Center','sigma','Offset']
	if fittype == 'Sinc':
		if guess is None:
			guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[2])),(sorted(set(fitdata[2]))[1]+sorted(set(fitdata[2]))[-1])/2, (sorted(set(fitdata[2]))[1]-sorted(set(fitdata[2]))[-1])/2, np.mean(fitdata[3])]
		popt, pcov = curve_fit.curve_fit(Sinc, fitdata[2], fitdata[3],p0=guess)
		ym = Sinc(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Sinc(fitdata[2],*popt)
		headers= ['Amplitdue','Center','sigma','Offset']
	if fittype == 'Sinc2':
		if guess is None:
			guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[0])),(sorted(set(fitdata[3]))[1]+sorted(set(fitdata[3]))[0]), 4, np.mean(fitdata[3])]
		popt, pcov = curve_fit.curve_fit(Sinc2, fitdata[2], fitdata[3],p0=guess)
		ym = Sinc2(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Sinc2(fitdata[2],*popt)
		headers = ['Amplitdue','Center','sigma','Offset']
	if fittype == 'TrapFreq':
		if guess is None:
			guess = [10000, 0.05, 20  ,-2 , 100, -0.1]
		popt, pcov = curve_fit.curve_fit(TrapFreq, fitdata[2], fitdata[3],p0=guess)
		ym = TrapFreq(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - TrapFreq(fitdata[2],*popt)
		headers = ['Amplitude','b','l','Center','Offset','Linear Slope']
	if fittype == 'TrapFreq2':
		if guess is None:
			guess = [10000, 0.05, 20  ,-2 , 100]
		popt, pcov = curve_fit.curve_fit(TrapFreq2, fitdata[2], fitdata[3],p0=guess)
		ym = TrapFreq2(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - TrapFreq2(fitdata[2],*popt)
		headers=['Amplitude','b','l','Center','Offset']
	if fittype == 'RabiFreq':
		if guess is None:
			guess = [1,1,1,0]
		popt, pcov = curve_fit.curve_fit(RabiFreq, fitdata[2], fitdata[3],p0=guess)
		ym = RabiFreq(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - RabiFreq(fitdata[2],*popt)
		headers = ['Amplitdue','b','Center','Offset']
	if fittype == 'Parabola':
		if guess is None:
			guess = [-3000, 44.82, 3000]
		popt, pcov = curve_fit.curve_fit(Parabola, fitdata[2], fitdata[3],p0=guess)
		ym = Parabola(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		headers = ['Amplitude','Center','Offset']
	if fittype == 'Linear':
		if guess is None:
			guess = [(max(fitdata[3])-min(fitdata[3]))/(max(fitdata[2])-min(fitdata[2])),0]
		popt, pcov = curve_fit.curve_fit(Linear, fitdata[2], fitdata[3],p0=guess)
		ym = Linear(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Linear(fitdata[2],*popt)
		headers = ['Slope','offset']
	if fittype == 'Exponential':
		if guess is None:
			guess = [max(fitdata[3])-min(fitdata[3]), 1]
		popt, pcov = curve_fit.curve_fit(Exponential, fitdata[2], fitdata[3],p0=guess)
		ym = Exponential(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Exponential(fitdata[2],*popt)
		headers = ['Amplitude','sigma']
	if fittype == 'RabiLine':
		if guess is None:
			guess = [1, 1, 1, 1, 1, 1, 1, 0]
		popt, pcov = curve_fit.curve_fit(RabiLine, fitdata[2], fitdata[3],p0=guess)
		ym = RabiLine(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - RabiLine(fitdata[2],*popt)
		headers = ['b', 'l', 'm', 'A', 's', 'j','k','p']
	if fittype == 'ErfcFit':
		if guess is None:
			guess = [1, 1, 1, 0]
		popt, pcov = curve_fit.curve_fit(ErfcFit, fitdata[2], fitdata[3],p0=guess)
		ym = ErfcFit(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - ErfcFit(fitdata[2],*popt)
		headers = ['Amp', 'Center', 'b', 'Offset']
	if fittype == 'SinplusCos':
		if guess is None:
			guess = [1, 1, 1, 0]
		popt, pcov = curve_fit.curve_fit(SinplusCos, fitdata[2], fitdata[3],p0=guess)
		ym = SinplusCos(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - SinplusCos(fitdata[2],*popt)
		headers = ['Sin Amp', 'Cos Amp', 'Offset']
	if fittype == 'FixedSin':
		if guess is None:
			guess = [1, 1, 0]
		popt, pcov = curve_fit.curve_fit(FixedSin, fitdata[2], fitdata[3],p0=guess)
		ym = FixedSin(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - FixedSin(fitdata[2],*popt)
		headers = ['Amplitude','phase','offset', 'freq', 'period', 'delay']
	if fittype == 'Sqrt':
		if guess is None:
			guess = [0.01]
		popt, pcov = curve_fit.curve_fit(Sqrt, fitdata[2], fitdata[3],p0=guess, maxfev=5000)
		ym = Sqrt(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
		residuals = fitdata[3] - Sqrt(fitdata[2],*popt)
		headers = ['Amplitude']
# 	else:
# 		popt, pcov, ym, residuals = [0,0,0,0]
	
	return popt, pcov, ym, residuals, headers

#table with fitted values and errors 

def table(filename, datatype='raw', names=['freq','sum95'], avg=False, fittype='Linear',guess=None):
	popt, pcov, ym, residuals, headers = fitting_type(filename, datatype, names, avg, fittype, guess=guess)

	if fittype == 'FixedSin':
		freq = 0.01
		period = 1/freq
		delay = popt[1] % (3.141592654) /freq
		values = list([*popt, freq, period, delay])
		errors = np.sqrt(np.diag(pcov))
		errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
	elif fittype == 'Sin':
		freq = popt[1]
		period = 1/freq
		delay = popt[1] % (3.141592654) /freq
		values = list([*popt, period, delay])
		errors = np.sqrt(np.diag(pcov))
		errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
	elif fittype == 'Cos':
		freq = popt[1]
		period = 1/freq
		delay = popt[1] % (3.141592654) /freq
		values = list([*popt, period, delay])
		errors = np.sqrt(np.diag(pcov))
		errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
	else:
		values = list([*popt])
		errors = np.sqrt(np.diag(pcov))
		
		
	return tabulate([['Values', *values], ['Errors', *errors]], headers=headers)

def chisq(filename, datatype='raw',names=['freq','sum95'], avg=False, fittype='Cos', guess=None):
	popt, pcov, ym, residuals, headers = fitting_type(filename, datatype, names, avg, fittype, guess=guess)

	fitdata = data(filename, datatype, names)
# 	if avg is True:
# 		fitdata = avgdata_data(filename, datatype, names, avg)
# 	else:
# 		if datatype == 'raw':
# 			fitdata = data(filename, datatype, names)
# 		elif datatype == 'exclude':
# 			fitdata = data_exclude(filename, datatype, names)
# 		elif datatype == 'exclude multiple points':
# 			fitdata = data_exclude_points(filename, datatype, names)
# 		elif datatype == 'avg':
# 			fitdata = avgdata_data(filename, datatype, names)		
# 		else:
# 			fitdata = 'nothing'
	

	chisq = chisquare(f_obs=fitdata[2], f_exp=residuals)
		
	return chisq
	

#plotting the data and fitting to chosen function 

def plots(filename, datatype='raw', names=['freq','sum95'], avg=False, guess=None, fittype='Sin', labels='False'):
	"""
	Inputs: filename, header names - names=['',''], guess for fit (None is automated guess) [A, omega, p, C], fittype (Sin, Cos, Gaussian, Lorentzian, Sinc, Sinc2, TrapFreq, TrapFreq2, RabiFreq, Parabola, Linear, Exponential, RabiLine, ErfcFit, SinplusCos) 
	
	Returns: data plotted with chosen fit
	"""
	fig1 = plt.figure(0)
	if avg is True:
		fitdata = avgdata_data(filename, datatype, names, avg)
	else:
		if datatype == 'raw':
			fitdata = data(filename, datatype, names)
		elif datatype == 'exclude':
			fitdata = data_exclude(filename, datatype, names)
		elif datatype == 'exclude multiple points':
			fitdata = data_exclude_points(filename, datatype, names)
		elif datatype == 'avg':
			fitdata = avgdata_data(filename, datatype, names)		
		else:
			fitdata = 'nothing'
	plt.title(f"{fittype} fit for {datatype} data for {filename}")
	if labels == 'False':
		xlabel = f"{fitdata[0]}"
		ylabel = f"{fitdata[1]}"
	else:
		xlabel, ylabel = labels
# 	print(labels)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)		
	plt.plot(fitdata[2], fitdata[3], 'go')
	
	popt, pcov, ym, residuals, headers = fitting_type(filename, datatype, names, avg, fittype, guess=guess)

	print(table(filename, datatype, names, avg, fittype, guess=guess))
	print(f"chisquare is {chisq(filename, datatype, names, avg, fittype, guess)}")
	
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	
	return fig1


# residuals 

def residuals(filename,  datatype='raw', names=['delay time', 'sum95'], avg=False,  guess=None, fittype='Sin'):
	"""
	Inputs: filename, header names - names=['','']
	
	Returns: residuals plot 
	"""
	if avg is True:
		fitdata = avgdata_data(filename, datatype, names)
	else:
		if datatype == 'raw':
			fitdata = data(filename, datatype, names)
		elif datatype == 'exclude':
			fitdata = data_exclude(filename, datatype, names)
		elif datatype == 'exclude multiple points':
			fitdata = data_exclude_points(filename, datatype, names)
		else:
			fitdata = 'nothing'

	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	popt, pcov, ym, residuals, headers = fitting_type(filename, datatype, names, avg, fittype=fittype, guess=guess)
	
	fig2 = plt.figure(1)
# 	plt.plot(fitdata[2],fitdata[3]*0,'-')
	plt.plot(fitdata[2], residuals, 'g+')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel +" Residuals")
	
	print(np.sum(residuals**2))
	return fig2




def avgdata(filename, datatype, names, avg=False, guess=None, fittype='Gaussian'):
	fig1 = plt.figure(0)
	fitdata = data(filename, datatype, names, fittype)
	plt.title(f"{fittype} fit for Averaged Data in {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
    
	namex = data(filename, datatype, names)[0] 
	namey = data(filename, datatype, names)[1] #choosing x , y columns from .dat 
	x = data(filename, datatype, names)[2]
	y = data(filename, datatype, names)[3]
	data2 = pd.DataFrame({namex: x, namey: y}) 
    
	avgdata = data2.groupby([namex])[namey].mean()


	avgdata.plot( marker = '.', linestyle = 'none')
    
	popt, pcov, ym, residuals, headers = fitting_type(filename, datatype, names, avg, fittype=fittype, guess=guess)

	print(table(filename, datatype, names, fittype=fittype, guess=guess))

	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	
	return fig1
            