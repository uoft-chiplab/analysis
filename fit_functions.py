# -*- coding: utf-8 -*-
"""
2023-10-19
@author: Chip Lab

Fitting functions for general analysis scripts 
"""
# %%
import numpy as np
import math

def Linear(data):
	"""
	Returns:  m*x + b 
	"""
	guess = None
	param_names = ["m", "b"]
	
	def linear(x, m, b):
		return m*x + b 
	return linear, guess, param_names

def Parabola(data, guess=None):
	"""
	Returns:  A*(x - x0)**2 + C
	"""
	if guess is None:
		x_ofmax = data[np.abs(data[:,1]).argmax(),0]
		mean_y = data[:,1].mean()
		max_y = data[:,1].max()
		
		guess = [-(max_y-mean_y), 0, mean_y]
	param_names = ["A", "x0", "C"]
	
	def parabola(x, A, x0, C):
		return A*(x - x0)**2 + C
	return parabola, guess, param_names

def Quadratic(data, guess=None):
	"""
	Returns:  A*(x - x0)**2 + C
	"""
	if guess is None:
		x_ofmax = data[np.abs(data[:,1]).argmax(),0]
		mean_y = data[:,1].mean()
		max_y = data[:,1].max()
		
		guess = [-(max_y-mean_y), mean_y]
	param_names = ["A", "C"]
	
	def quadratic(x, A, C):
		return A*(x)**2 + C
	return quadratic, guess, param_names


def Sqrt(data):
	"""
	Returns:  A*np.sqrt(x - x0)
	"""
	min_y = data[:,1].min()
	max_y = data[:,1].max()
	
	param_names = ["A", "x0"]
	guess = [max_y-min_y, 0]
	
	def sqrt(x, A, x0):
		return A*np.sqrt(x-x0)
	return sqrt, guess, param_names

def Gaussian(data, guess=None):
	"""
	Returns:  A * np.exp(-(x-x0)**2/(2*sigma**2)) + C
	"""
	x_ofmin = data[np.abs(data[:,1]).argmin(),0]
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ["A", "x0", "sigma", "C"]
	guess = [-(max_y-mean_y), x_ofmin, (max_x-min_x)/2, mean_y]

	
	def gaussian(x, A, x0, sigma, C):
		return A * np.exp(-(x-x0)**2/(2*sigma**2)) + C
	return gaussian, guess, param_names

def Dimerlineshape(data, guess=None):
	x_ofmin = data[np.abs(data[:,1]).argmin(),0]
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ["A", "x0", "sigma", "C"]
	guess = [-(max_y-mean_y), x_ofmin, (max_x-min_x)/2, mean_y]

	def dimerlineshape(x, A, x0, sigma, C):
		# everything in MHz
		Gamma = A*np.sqrt(x-x0) * np.exp((-x+x0)/sigma)* np.heaviside(x - x0, 1) + C
		Gamma = np.nan_to_num(Gamma)
		return Gamma
	return dimerlineshape, guess, param_names

def DimerlineshapeZero(data, guess=None):
	x_ofmin = data[np.abs(data[:,1]).argmin(),0]
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ["A", "x0", "sigma"]
	guess = [-(max_y-mean_y), x_ofmin, (max_x-min_x)/2]

	def dimerlineshapezero(x, A, x0, sigma):
		# everything in MHz
		Gamma = A*np.sqrt(x-x0) * np.exp((-x+x0)/sigma)* np.heaviside(x - x0, 1) 
		Gamma = np.nan_to_num(Gamma)
		return Gamma
	return dimerlineshapezero, guess, param_names

def NegGaussian(data, guess=None):
	"""
	Returns:  A * np.exp(-(x-x0)**2/(2*sigma**2)) + C
	"""
	if guess is None:
		x_ofmin = data[np.abs(data[:,1]).argmin(),0]
		max_x = data[:,0].max()
		min_x = data[:,0].min()
		mean_y = data[:,1].mean()
		min_y = data[:,1].min()
		guess = [min_y-mean_y, x_ofmin, (max_x-min_x)/10, mean_y]
	
	param_names = ["A", "x0", "sigma", "C"]
	
	def gaussian(x, A, x0, sigma, C):
		return A * np.exp(-(x-x0)**2/(2*sigma**2)) + C
	return gaussian, guess, param_names

def Lorentzian(data, guess=None):
	"""
	Returns:  A/((x-x0)**2 + (sigma)**2) + C
	"""
	if guess is None:
		x_ofmax = data[np.abs(data[:,1]).argmax(),0]
		max_x = data[:,0].max()
		min_x = data[:,0].min()
		mean_y = data[:,1].mean()
		max_y = data[:,1].max()
		guess = [max_y-mean_y, x_ofmax, (max_x-min_x)/2, mean_y]
	
	param_names = ["A", "x0", "sigma", "C"]
	
	def lorentzian(x, A, x0, sigma, C):
		return A/((x-x0)**2 + (sigma)**2) + C
	return lorentzian, guess, param_names

def Sin(data, guess=None):
	"""
	Returns: A*np.sin(omega*x - phi) + C
	"""
	if guess is None:
		mean_y = data[:,1].mean()
		max_y = data[:,1].max()
		guess = [max_y-mean_y, 6*2.5, 0, mean_y]
		
	param_names = ["A", "omega", "phi", "C"]
	
	def sin(x, A, omega, phi, C):
		return A*np.sin(omega*x - phi) + C
	return sin, guess, param_names

def Sin2Decay(data, guess=None):
	"""
	Returns: A*np.exp(-x/tau)*np.sin(omega*x - phi)**2 + C
	"""
	if guess is None:
		mean_y = data[:,1].mean()
		max_y = data[:,1].max()
		guess = [max_y-mean_y, 1, 0, mean_y, 1]
		
	param_names = ["A", "omega", "phi", "C", "tau"]
	
	def sin2decay(x, A, omega, phi, C, tau):
		return A*np.exp(-x/tau)*np.sin(omega*x - phi) + C
	return sin2decay, guess, param_names

def Sinc(data, guess=None):
	"""
	Returns:   A*np.sinc((x-x0) / sigma) + C
	"""
	if guess is None:
		x_ofmax = data[np.abs(data[:,1]).argmax(),0]
		x_ofmin = data[np.abs(data[:,1]).argmin(),0]
		max_x = data[:,0].max()
		min_x = data[:,0].min()
		mean_y = data[:,1].mean()
		max_y = data[:,1].max()
		guess = [max_y-mean_y, x_ofmin, (max_x-min_x)/2, mean_y]
	
	param_names = ["A", "x0", "sigma", "C"]
	
	def sinc(x, A, x0, sigma, C):
		return A*(np.sinc((x-x0) / sigma)) + C
	return sinc, guess, param_names

def Sinc2(data, guess=None):
	"""
	Returns:   A*np.sinc((x-x0) / sigma))**2 + C
	"""
	if guess is None:
		x_ofmin = data[np.abs(data[:,1]).argmin(),0]
		x_ofmax = data[np.abs(data[:,1]).argmax(),0]
		max_x = data[:,0].max()
		min_x = data[:,0].min()
		mean_y = data[:,1].mean()
		max_y = data[:,1].max()
		guess = [max_y-mean_y, x_ofmax, (max_x-min_x)/2, mean_y]
	
	param_names = ["A", "x0", "sigma", "C"]
	guess = [max_y-mean_y, x_ofmax, (max_x-min_x)/2, mean_y]
	
	def sinc2(x, A, x0, sigma, C):
		return np.abs(A)*(np.sinc((x-x0) / sigma)**2) + C
	return sinc2, guess, param_names

def FixedSinc2(data):
	"""
	Returns:   A*np.sinc((x-x0) / sigma) + C
	"""
	x_ofmin = data[np.abs(data[:,1]).argmin(),0]
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ["A", "x0", "C"]
	guess = [max_y-mean_y, x_ofmax, mean_y]
	
	def fixedsinc2(x, A, x0, C):
# 		sigma=0.0380103777802075 avg value of sigmas from phaseshift using sinc2 np.absolute(plots(Sinc2)[4])
# 		sigma = 0.054895027020569725 #abs value of sigmas from above and avg np.average(np.absolute(plots(Sinc2)[4]))
		sigma = 0.04821735158888396		
		return A*(np.sinc((x-x0) / sigma)**2) + C
	return fixedsinc2, guess, param_names


def MinSinc2(data):
	"""
	Returns:   A*np.sinc((x-x0) / sigma) + C
	"""
	x_ofmin = data[np.abs(data[:,1]).argmin(),0]
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ["A", "x0", "sigma", "C"]
	guess = [max_y-mean_y, x_ofmin, (max_x-min_x)/2, mean_y]
	
	def minsinc2(x, A, x0, sigma, C):
		return A*(np.sinc((x-x0) / sigma)**2) + C
	return minsinc2, guess, param_names

def MinFixedSinc2(data):
	"""
	Returns:   A*np.sinc((x-x0) / sigma) + C
	"""
	x_ofmin = data[np.abs(data[:,1]).argmin(),0]
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ["A", "x0", "C"]
	guess = [max_y-mean_y, x_ofmin, mean_y]
	
	def fixedsinc2(x, A, x0, C):
# 		sigma=0.0380103777802075 avg value of sigmas from phaseshift using sinc2 np.absolute(plots(Sinc2)[4])
		sigma = 0.054895027020569725 #abs value of sigmas from above and avg np.average(np.absolute(plots(Sinc2)[4]))
		return A*(np.sinc((x-x0) / sigma)**2) + C
	return fixedsinc2, guess, param_names


def TrapFreq(data,guess=None):
	"""
	Returns:  A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C + D*x
	"""
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ['Amplitude','tau','omega','Center','Offset','Linear Slope']
	guess = [10000, 0.05, 20  ,-2 , 100, -0.1]
	
	def TrapFreq(x, A, b, l, x0, C, D):
		return A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C + D*x
# trap freq is then : 
# f*10**3/2/np.pi 
	return TrapFreq, guess, param_names

def TrapFreq2(data,guess=None):
	"""
	Returns: A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C 
	"""
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ['Amplitude','b','l','phase','Offset']
	if guess is None: guess = [5, 1e2, 2*3.14*0.2, 0, 100]
	
	def TrapFreq2(x, A, b, l, x0, C):
		return A*np.exp(-x/b)*(np.sin(l * x - x0)) +  C 
	return TrapFreq2, guess, param_names

def TrapFreqCD(data,guess=None):
	"""
	Returns: A*np.exp(-x/b)*(np.sin(f * x)) +  C 
	"""
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ['A','b','f','C']
	if guess is None: guess = [5, 1e2, 0.2, 100]
	
	def func(x, A, b, f, C):
		return A*np.exp(-x/b)*(np.sin((2*3.14159)*f * x)) +  C 
	return func, guess, param_names

def RabiFreq(data):
	"""
	Returns:  A*(np.sin(b/2 * x - x0))**2 + C
	"""
	param_names = ['Amplitude','b','Center','Offset']
	guess = [1,1,1,0]
	
	def RabiFreq(x, A, b, x0, C):
		return A*(np.sin(2*np.pi*b/2 * x - x0))**2 + C
	return RabiFreq, guess, param_names

def RabiFreqDecay(data):
	"""
	Returns:  A*np.exp(-t/tau)*(np.sin(b * x - x0))**2 + C
	"""
	param_names = ['Amplitude','b','tau','Center','offset']
	guess = [1,1,1,0,0]
	
	def RabiFreq(x, A, b, tau, x0, C):
		return A*np.exp(-x/tau)*(np.sin(b * x - x0))**2 + C
	return RabiFreq, guess, param_names

def Exponential(data):
	"""
	Returns: A*np.exp(-x/sigma)
	"""
	x_ofmax = data[np.abs(data[:,1]).argmax(),0]
	max_x = data[:,0].max()
	min_x = data[:,0].min()
	mean_y = data[:,1].mean()
	max_y = data[:,1].max()
	
	param_names = ['Amplitude','sigma']
	guess = [max_x - min_x, 1]
	
	def Exponential(x, A, sigma):
		return A*np.exp(-x/sigma)
	return Exponential, guess, param_names

def RabiLine(data):
	"""
	Returns:  (b**2 / (l**2 + (x - m)**2 ) ) * (A * np.sin(np.sqrt(s**2 + (x - j)**2 ) * k)**2 + p )
	"""
	param_names = ['b', 'l', 'm', 'A', 's', 'j','k','p']
	guess = [1, 1, 1, 1, 1, 1, 1, 0]
	
	def RabiLine(x, b, l, m, A, s, j, k, p): 
		return (b**2 / (l**2 + (x - m)**2 ) ) * (A * np.sin(np.sqrt(s**2 + (x - j)**2 ) * k)**2 + p )

	return RabiLine, guess, param_names


def ErfcFit(data):
	"""
	Returns:  A * math.erfc((x - x0) / b ) + C
	"""
	param_names =  ['Amp', 'Center', 'b', 'Offset']
	guess = [1, 1, 1, 0]
	
	def ErfcFit(x, A, x0, b, C):
		return A * math.erfc((x - x0) / b ) + C
	
	return ErfcFit, guess, param_names


def SinplusCos(data):
	"""
	Returns:  A*np.sin(omega*t) + B*np.cos(omega*t) + C
	"""
	param_names = ['Sin Amp', 'Cos Amp', 'Offset']
	guess = [1, 1, 1, 0]
	
	def SinplusCos(t, omega, A, B, C):
		return A*np.sin(omega*t) + B*np.cos(omega*t) + C

	return SinplusCos, guess, param_names

def FixedSin(data, f, guess=None):
	"""
	hard coded 10 kHz
	Returns: A*np.sin(0.0628*x - p) + C
	"""
	if guess is None:
		mean_y = data[:,1].mean()
		max_y = data[:,1].max()
		guess = [max_y-mean_y, 0, mean_y]
	
	param_names =  ['Amplitude','phase','offset']
	
	def FixedSin(t, A, p, C):
		omega = f * 2 * np.pi # 10 kHz
		return A*np.sin(omega*t - p) + C
	
	return FixedSin, guess, param_names

def FixedSin5kHz(data):
	"""
	hard coded 5 kHz
	Returns: A*np.sin(0.0314*x - p) + C
	"""
	param_names =  ['Amplitude','phase','offset']
	guess = [1, 1, 0]
	
	def FixedSin5kHz(t, A, p, C):
		omega = 0.005 * 2 * np.pi # 5 kHz
		return A*np.sin(omega*t - p) + C
	
	return FixedSin5kHz, guess, param_names

def FixedSin1kHz(data):
	"""
	hard coded 1 kHz
	Returns: A*np.sin(0.0314*x - p) + C
	"""
	param_names =  ['Amplitude','phase','offset']
	guess = [1, 1, 0]
	
	def FixedSin1kHz(t, A, p, C):
		omega = 1 * 2 * np.pi # 1 kHz
		return A*np.sin(omega*t - p) + C
	
	return FixedSin1kHz, guess, param_names

def FixedSin2kHz(data):
	"""
	hard coded 2 kHz
	Returns: A*np.sin(0.0314*x - p) + C
	"""
	param_names =  ['Amplitude','phase','offset']
	guess = [1, 1, 0]
	
	def FixedSin2kHz(t, A, p, C):
		omega = 2 * 2 * np.pi # 1 kHz
		return A*np.sin(omega*t - p) + C
	
	return FixedSin2kHz, guess, param_names
# %%
