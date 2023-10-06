# -*- coding: utf-8 -*-
"""
2023-09-25
@author: Chip Lab

Fits script

"""
from analysisfunctions import * # includes numpy and constants
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os 
import scipy.optimize as curve_fit
from tabulate import tabulate # pip install tabulate
from collections import defaultdict
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



#plotting raw data with cos 
def plotcos(filename, names=['freq','fraction95'],guess=None, residualss=True, datatype='raw'):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear), datatype 
	
	Returns: cos fit, A*np.cos(omega*x - p) + C
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	print(fitdata)
	if datatype == 'raw':
		fitdata = data(filename, names)
	else:
		if datatype == 'exclude':
			fitdata = data_exclude(filename, names)
		else:
			if datatype == 'exclude multiple points':
				fitdata = data_exclude_points(filename, names)
	plt.title(f"Cos fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2], fitdata[3], 'go')
	if guess is None:	
		guess = [-0.2, 0, 10, 202]
	popt, pcov = curve_fit.curve_fit(Cos, fitdata[2], fitdata[3],p0=guess)
	ym = Cos(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	freq = popt[1]/2/3.14
	period = 1/freq
	delay = popt[2] % (3.141592654) /popt[1]
	values = list([*popt, freq, period, delay])
	errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
	print(tabulate([['Values', *values], ['Errors', *errors]], headers=['Amplitude','omega','phase','offset', 'freq', 'period', 'delay']))
	figures = [fig1]
	if residualss is True:
		figures.append(residuals(filename))
	plt.show(figures)


#plotting raw data with sin 
#guess=['Amplitude', 'Frequency','Width','Background']
def plotsin(filename, names=['freq','fraction95'],guess=None, errors=False, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear)
	
	Returns: sin fit, A*np.sin(omega*x - p) + C
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename,names)
	plt.title(f"Sin fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:	
		guess = [-0.2, 
		   0,10,202]
	popt, pcov = curve_fit.curve_fit(Sin, fitdata[2], fitdata[3],p0=guess)
	ym = Sin(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	freq = popt[1]/2/3.14
	period = 1/freq
	delay = popt[2] % (3.141592654) /popt[1]
	values = list([*popt, freq, period, delay])
	errors = np.concatenate((errors, [errors[1]/2/3.14, period * errors[1]/popt[1], delay * errors[2]/popt[2]]))
	print(tabulate([['Values', *values], ['Errors', *errors]], headers=['Amplitude','omega','phase','offset', 'freq', 'period', 'delay']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Sin(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)


#plotting raw data with gaussian 
#guess=['Amplitude', 'Frequency','Width','Background']
def plotgaussian(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: gaussian fit, A * np.exp(-(x-x0)**2/(2*sigma**2)) + C
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	plt.title(f"Gaussian fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:	
		guess = [-(max(fitdata[3])-min(fitdata[3])), 
		   fitdata[2][fitdata[3].argmin()],0.04,np.mean(fitdata[3])]
	popt, pcov = curve_fit.curve_fit(Gaussian, fitdata[2], fitdata[3],p0=guess)
	ym = Gaussian(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values', *popt], ['Errors', *errors]], 
				headers=['Amplitude','Frequency','Width','Background']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

#plotting raw data with linear function 
#guess=['Slope', 'Offset']
def plotlinear(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: linear fit, m*x + b 
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	plt.title(f"Linear fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [(max(fitdata[3])-min(fitdata[3]))/(max(fitdata[2])-min(fitdata[2])), 
		   fitdata[2][fitdata[3].argmin()]]
	popt, pcov = curve_fit.curve_fit(Linear, fitdata[2], fitdata[3],p0=guess)
	ym = Linear(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Slope', 'Offset']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

#plotting raw data with Lorentzian function 
#guess=['Amplitude', 'b**2' ,'Frequency', 'Width', 'Background']
def plotlorentzian(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: lorentzian fit, (A*b**2) /((x-x0)**2 + (sigma)**2) + C
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	plt.title(f"Lorentzian fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'bo')
	if guess is None:
		guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[2])), 1, 
		   fitdata[2][fitdata[3].argmin()], 0.04, np.mean(fitdata[3])]
	popt, pcov = curve_fit.curve_fit(Lorentzian, fitdata[2], fitdata[3],p0=guess)
	ym = Lorentzian(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'b**2' ,'Frequency', 'Width', 'Background']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)


#plotting raw data with Sinc function 

def plotsinc(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: sinc fit,  A*(np.sinc((x-x0) / sigma)) + C 
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	plt.title(f"Sinc fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[2])), 
		   (sorted(set(fitdata[2]))[1]+sorted(set(fitdata[2]))[-1])/2, (sorted(set(fitdata[2]))[1]-sorted(set(fitdata[2]))[-1])/2, np.mean(fitdata[3])]
		print(guess)
	popt, pcov = curve_fit.curve_fit(Sinc, fitdata[2], fitdata[3],p0=guess)
	ym = Sinc(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'center', 'Width', 'offset']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)
	

#plotting raw data with Sinc**2 function 

def plotsinc2(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: sinc**2 fit, A*(np.sinc((x-x0) / sigma))**2 + C
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	plt.title(f"Sinc**2 fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [(max(fitdata[3])-(sorted(set(fitdata[3]))[0])), 
		   (sorted(set(fitdata[3]))[1]+sorted(set(fitdata[3]))[0]), 4, np.mean(fitdata[3])]
	popt, pcov = curve_fit.curve_fit(Sinc2, fitdata[2], fitdata[3],p0=guess)
	ym = Sinc2(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'phase', 'Width', 'Background']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)


# plotting raw data with Trap Freq function 

def plottrapfreq(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: trap freq fit, A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C + D*x
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	# plot data
	fig1 = plt.figure(0)
	plt.title(f"Trap Freq fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim(-0.01, 0.2) # sets y axis limits
	# fit data
	for i in range(len(fitdata[2])) :
		if fitdata[2][i] == 0.7 :
 			fitdata[2][i] = 0 
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [10000, 0.05, 20  ,-2 , 100, -0.1] # 'Amplitude', 'tau', 'omega', 'phase', 'C', 'm'
					# where m is the slope of linear term 
	popt, pcov = curve_fit.curve_fit(TrapFreq, fitdata[2], fitdata[3],p0=guess)
	num = 200
	ym = TrapFreq(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'tau', 'omega', 'phase', 'Offset', 'Slope']))
	figures = [fig1]
	# plot residuals
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)
	
	

# plotting raw data with Trap Freq function 

def plottrapfreq2(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: trap freq fit without linear term, A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C 
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	# plot data
	fig1 = plt.figure(0)
	plt.title(f"Trap Freq fit no linear term for {filename}")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim(-0.01, 0.2) # sets y axis limits
	# fit data
	for i in range(len(fitdata[2])) :
		if fitdata[2][i] == 0.7 :
 			fitdata[2][i] = 0 
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [10000, 0.05, 20  ,-2 , 100] # 'Amplitude', 'tau', 'omega', 'phase', 'C', 'm'
					# where m is the slope of linear term 
	popt, pcov = curve_fit.curve_fit(TrapFreq2, fitdata[2], fitdata[3],p0=guess)
	num = 200
	ym = TrapFreq2(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=num),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'tau', 'omega', 'phase', 'Offset']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)
	

#plotting raw data with Rabi Freq function  


def plotrabifreq(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: rabi freq fit, A*(np.sin(b/2 * x - x0))**2 + C
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	plt.title(f"Rabi Freq fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
	# 	guess = [max(fitdata[3])-min(fitdata[3]), 1, min(fitdata[3]), 0]
		guess = [1,1,1,0]
	popt, pcov = curve_fit.curve_fit(RabiFreq, fitdata[2], fitdata[3],p0=guess)
	ym = RabiFreq(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'b', 'x0', 'C']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)


#plotting raw data with Parabola function 

def plotparabola(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: parabolic fit, A*(x - x0)**2 + C
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.title(f"Parabolic fit for {filename}")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [-3000, 44.82, 3000]
	popt, pcov = curve_fit.curve_fit(Parabola, fitdata[2], fitdata[3],p0=guess)
	ym = Parabola(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values', *popt], ['Errors', *errors]], 
				headers=['A', 'center', 'Offset']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

	

#plotting raw data with exponential function 

def plotexp(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: exponential fit  , A*np.exp(-x/sigma)
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	plt.title(f"Exponential fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [max(fitdata[3])-min(fitdata[3]), 1]
	popt, pcov = curve_fit.curve_fit(Expontial, fitdata[2], fitdata[3],p0=guess)
	ym = Expontial(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'sigma']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

#plotting raw data with Rabiline function 

def plotrabiline(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: rabiline fit, (b**2 / (l**2 + (x - m)**2 ) ) * (A * np.sin(np.sqrt(s**2 + (x - j)**2 ) * k)**2 + p )
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	plt.title(f"Rabi Line fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [1, 1, 1, 1, 1, 1, 1, 0]
	popt, pcov = curve_fit.curve_fit(RabiLine, fitdata[2], fitdata[3],p0=guess)
	ym = RabiLine(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['b', 'l', 'm','A','s','j','k','p']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)

#plotting raw data with Erfc function 

def ploterfc(filename, names=['freq','fraction95'], guess=None, residuals=False):
	"""
	Inputs: filename, header names, guess for fit (None is automated guess), residualss (true is have them appear) 
	
	Returns: erfc fit, A * math.erfc((x - x0) / b ) + C
	"""
	fig1 = plt.figure(0)
	fitdata = data(filename, names)
	plt.title(f"Erfc fit for {filename}")
	xlabel = f"{fitdata[0]}"
	ylabel = f"{fitdata[1]}"
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(fitdata[2],fitdata[3],'go')
	if guess is None:
		guess = [1, 1, 1, 0]
	popt, pcov = curve_fit.curve_fit(ErfcFit, fitdata[2], fitdata[3],p0=guess)
	ym = ErfcFit(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),*popt)
	plt.plot(np.linspace(max(fitdata[2]),min(fitdata[2]),num=200),ym)
	errors = np.sqrt(np.diag(pcov))
	print(tabulate([['Values',*popt], ['Errors',*errors]], 
				headers=['Amplitude', 'x0', 'sigma', 'Offset']))
	figures = [fig1]
	if residuals is True:
		residuals = fitdata[3] - Parabola(fitdata[2],*popt)
		fig2 = plt.figure(1)
		plt.plot(fitdata[2],fitdata[3]*0,'-')
		plt.plot(fitdata[2], residuals,'g+')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel +" Residuals")
		figures.append(fig2)
	plt.show(figures)
