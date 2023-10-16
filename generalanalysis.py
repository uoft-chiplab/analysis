# -*- coding: utf-8 -*-
"""
2023-09-25
@author: Chip Lab

Fits script

"""
from analysisfunctions import * # includes numpy and constants
import os 
from data import *
from plotting import *

#blah test  test test 

#plotting raw data with cos 

def plotcos(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='Cos'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, omega, p, C], residualss (true is have them appear), datatype 
	
	Returns: cos fit, A*np.cos(omega*x - p) + C
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='Cos')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='Cos'))
	plt.show(figures)


#plotting raw data with sin 

def plotsin(filename,datatype, names=['freq','fraction95'], autofind=True,  guess=None, residualss=False, fit=True, fittype='Sin'):
	"""
	Inputs: filename, header names  - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, omega, p, C], residualss (true is have them appear)
	
	Returns: sin fit, A*np.sin(omega*x - p) + C
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='Sin')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='Sin'))
	plt.show(figures)


#plotting raw data with gaussian 

def plotgaussian(filename, datatype, names=['freq','fraction95'], avg=False, autofind=True, guess=None, residualss=False, fit=True, fittype='Gaussian'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, x0, sigma, C], residualss (true is have them appear) 
	
	Returns: gaussian fit, A * np.exp(-(x-x0)**2/(2*sigma**2)) + C
	"""
	if fit is True :
		if avg is True :
			figures = [avgdata(filename, names, fittype='Gaussian')]
		else :
			figures = [plots(filename, datatype, names, guess, fittype='Gaussian')]
	if residualss is True:
		figures.append(residuals(filename, datatype, datatype, names, guess, fittype='Gaussian'))
	plt.show(figures)

#plotting raw data with linear function 

def plotlinear(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='Linear'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [m, b], residualss (true is have them appear) 
	
	Returns: linear fit, m*x + b 
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='Linear')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='Linear'))
	plt.show(figures)

#plotting raw data with Lorentzian function 
#guess=['Amplitude', 'b**2' ,'Frequency', 'Width', 'Background']
def plotlorentzian(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='Lorentzian'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, b, x0, sigma, C], residualss (true is have them appear) 
	
	Returns: lorentzian fit, (A*b**2) /((x-x0)**2 + (sigma)**2) + C
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='Lorentzian')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, fittype='Lorentzian'))
	plt.show(figures)


#plotting raw data with Sinc function 

def plotsinc(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='Sinc'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, x0, sigma, C], residualss (true is have them appear) 
	
	Returns: sinc fit,  A*(np.sinc((x-x0) / sigma)) + C 
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='Sinc')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='Sinc'))
	plt.show(figures)
	

#plotting raw data with Sinc**2 function 

def plotsinc2(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='Sinc2'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, x0, sigma, C], residualss (true is have them appear) 
	
	Returns: sinc**2 fit, A*(np.sinc((x-x0) / sigma))**2 + C
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='Sinc2')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='Sinc2'))
	plt.show(figures)


# plotting raw data with Trap Freq function 

def plottrapfreq(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='TrapFreq'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, b, l, x0, C, D], residualss (true is have them appear) 
	
	Returns: trap freq fit, A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C + D*x
	"""
	# plot data
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='TrapFreq')]
	# plot residuals
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='TrapFreq'))
	plt.show(figures)
	
	

# plotting raw data with Trap Freq function 

def plottrapfreq2(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='TrapFreq2'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, b, l, x0, C], residualss (true is have them appear) 
	
	Returns: trap freq fit without linear term, A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C 
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='TrapFreq2')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='TrapFreq2'))
	plt.show(figures)
	

#plotting raw data with Rabi Freq function  


def plotrabifreq(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='RabiFreq'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, b, x0, C], residualss (true is have them appear) 
	
	Returns: rabi freq fit, A*(np.sin(b/2 * x - x0))**2 + C
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='RabiFreq')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='RabiFreq'))
	plt.show(figures)


#plotting raw data with Parabola function 

def plotparabola(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='Parabola'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, x0, C], residualss (true is have them appear) 
	
	Returns: parabolic fit, A*(x - x0)**2 + C
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='Parabola')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='Parabola'))
	plt.show(figures)

	

#plotting raw data with exponential function 

def plotexp(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='Exponential'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, sigma], residualss (true is have them appear) 
	
	Returns: exponential fit  , A*np.exp(-x/sigma)
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='Exponential')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='Exponential'))
	plt.show(figures)

#plotting raw data with Rabiline function 

def plotrabiline(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='RabiLine'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [b, l, m, A, s, j, k, p], residualss (true is have them appear) 
	
	Returns: rabiline fit, (b**2 / (l**2 + (x - m)**2 ) ) * (A * np.sin(np.sqrt(s**2 + (x - j)**2 ) * k)**2 + p )
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, fittype='RabiLine')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, fittype='RabiLine'))
	plt.show(figures)

#plotting raw data with Erfc function 

def ploterfc(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='ErfcFit'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, x0, b, C], residualss (true is have them appear) 
	
	Returns: erfc fit, A * math.erfc((x - x0) / b ) + C
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, residualss,  fittype='ErfcFit')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, residualss,  fittype='ErfcFit'))
	plt.show(figures)
	
#plotting raw data with sin plus cos function

def plotsinpluscos(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, residualss=False, fit=True, fittype='SinplusCos'):
	"""
	Inputs: filename, header names - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [omega, A, B, C], residualss (true is have them appear) 
	
	Returns: sin + cos fit, A*np.sin(omega*t) + B*np.cos(omega*t) + C
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess, residualss,  fittype='SinplusCos')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names, guess, residualss,  fittype='SinplusCos'))
	plt.show(figures)

#plotting raw data with sin 
def plotfixedsin(filename, datatype, names=['freq','fraction95'], autofind=True, guess=None, errors=False, residualss=False, fit=True, fittype='FixedSin'):
	"""
	Inputs: filename, header names  - names=['',''], autofind (False is manually inputted path), guess for fit (None is automated guess) [A, omega, p, C], residualss (true is have them appear)
	
	Returns: sin fit, A*np.sin(omega*x - p) + C
	"""
	if fit is True :
		figures = [plots(filename, datatype, names, guess=None, fittype='FixedSin')]
	if residualss is True:
		figures.append(residuals(filename, datatype, names))
	plt.show(figures)