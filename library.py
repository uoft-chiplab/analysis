# -*- coding: utf-8 -*-
"""
2023-09-25
@author: Chip Lab

Functions to call in analysis scripts
"""
# %%
import os
current_dir = os.path.dirname(__file__)

from scipy.constants import pi, hbar, h, c, k as kB
from scipy.integrate import trapz, simps, cumtrapz
from scipy.optimize import fsolve, curve_fit
import numpy as np

uatom = 1.660538921E-27
a0 = 5.2917721092E-11
uB = 9.27400915E-24
gS = 2.0023193043622
gJ = gS
mK = 39.96399848 * uatom
ahf = -h * 285.7308E6 # For groundstate 
gI = 0.000176490 # total nuclear g-factor

# plt settings
frame_size = 1.5
markers = ["o", "s", "^", "D", "h", "x", "o", "s", "^", "D", "h"]
	
plt_settings = {"axes.linewidth": frame_size,
				"lines.linewidth":2,
					 "font.size": 12,
					 "legend.fontsize": 10,
					 "legend.framealpha": 1.0,
					 "xtick.major.width": frame_size,
					 "xtick.minor.width": frame_size*0.75,
					 "xtick.direction":'in',
					 "xtick.major.size": 3.5*frame_size,
					 "xtick.minor.size": 2.0*frame_size,
					 "ytick.major.width": frame_size,
					 "ytick.minor.width": frame_size*0.75,
					 "ytick.major.size": 3.5*frame_size,
					 "ytick.minor.size": 2.0*frame_size,
					 "ytick.direction":'in',
					 "lines.linestyle":'',
					 "lines.marker":"o"}

# plot color and markers
colors = ["blue", "red", "green", "orange", 
		  "purple", "teal", "pink", "brown"]

	
tintshade=0.6

def tint_shade_color(color, amount=0.5):
    """
    Tints or shades the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
	
	From https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Examples:
    >> tint_shade_color('g', 0.3)
    >> tint_shade_color('#F034A3', 0.6)
    >> tint_shade_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def MonteCarlo_trapz(xs, ys, yserr, num_iter=1000):
	""" Computes trapz for list of data points (xs, ys+-yserr),
	and estimates std dev of result by sampling ys and yserr from 
	Gaussian distributions, num_iter (default 1000) times."""
# 	value = np.trapz(ys, x=xs)
	def rand_y(y, yerr, size):
		generator = np.random.default_rng()
		return generator.normal(loc=y, scale=yerr, size=num_iter)
	# array of lists of y values, sampled from Gaussians with centres y and widths yerr
	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, yerr in zip(ys, yserr)])
	values = np.array([np.trapz(ys_iter[:,i], x=xs) for i in range(num_iter)])
	distr_mean, distr_stdev = (np.mean(values), np.std(values))
	return values, distr_mean, distr_stdev


def MonteCarlo_interp_trapz(xs, ys, yserr, num_iter=1000):
	""" Computes trapz for interpolated list of data points (xs, ys+-yserr),
	and estimates std dev of result by sampling ys and yserr from 
	Gaussian distributions, num_iter (default 1000) times."""
# 	value = np.trapz(ys, x=xs)
	def rand_y(y, yerr, size):
		generator = np.random.default_rng()
		return generator.normal(loc=y, scale=yerr, size=num_iter)
	# array of lists of y values, sampled from Gaussians with centres y and widths yerr
	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, yerr in zip(ys, yserr)])
	
	# interpolation array for x, num_iter in size
	xs_interp = np.linspace(min(xs), max(xs), num_iter)
	
	# compute interpolation array for y, num_iter by num_iter in size
	ys_interp_iter = np.array([[np.interp(xi, xs, ys_iter[:,i]) for xi in xs_interp]
					  for i in range(num_iter)])
	
	# integrals using each interpolation set
	values = np.array([np.trapz(ys_interp_iter[i], x=xs_interp) for i in range(num_iter)])
	
	distr_mean, distr_stdev = (np.mean(values), np.std(values))
	return values, distr_mean, distr_stdev

def MonteCarlo_interp_extrap_trapz(xs, ys, yserr, xmax, 
								   fit_func, num_iter=1000):
	""" Computes trapz for interpolated list of data points (xs, ys+-yserr),
	which is extrapolated using fit_func out to xmax. Estimates std dev of 
	result by sampling ys and yserr from Gaussian distributions, and fitting
	to this sample, num_iter (default 1000) times."""
	def rand_y(y, yerr, size):
		generator = np.random.default_rng()
		return generator.normal(loc=y, scale=yerr, size=num_iter)
	# array of lists of y vals, from Gaussians with centres y and widths yerr
	ys_iter = np.array([rand_y(y, yerr, num_iter) for y, 
					 yerr in zip(ys, yserr)])
	
	fits = np.array([curve_fit(fit_func, xs, ys_iter[:,i]) \
					for i in range(num_iter)])
		
	popts = fits[:,0]
	pcovs = fits[:,1]
	
	# interpolation array for x, num_iter in size
	xs_interp = np.linspace(min(xs), max(xs), num_iter)
	# extrapolation array for x, num_iter in size
	xs_extrap = np.linspace(max(xs), xmax, num_iter)
	
	# compute interpolation array for y, num_iter by num_iter in size
	ys_interp_iter = np.array([[np.interp(xi, xs, ys_iter[:,i]) \
							 for xi in xs_interp] for i in range(num_iter)])
	
	# integrals using each interpolation set
	values = np.array([np.trapz(ys_interp_iter[i], x=xs_interp) \
					+ np.trapz(fit_func(xs_extrap, *popts[i]), x=xs_extrap) \
					for i in range(num_iter)])
	
	distr_mean, distr_stdev = (np.mean(values), np.std(values))
	return values, distr_mean, distr_stdev, popts, pcovs

def chi_sq(y, yfit, yerr, dof):
	return 1/dof * np.sum((np.array(y) - np.array(yfit))**2/(yerr**2))

def deBroglie(T):
	return h/np.sqrt(2*pi*mK*kB*T)

def deBroglie_kHz(T):
	return np.sqrt(hbar/(mK*T*1e3))

def EhfFieldInTesla(B, F, mF):
	term1 = -ahf/4 + gI * uB * mF * B
	term2 = (2*(gJ - gI)*uB *B /ahf/9)
	term3 = (-1)**(F-1/2) *9 *ahf/4 *np.sqrt(1+4*mF/9 * term2 + term2**2)
	return term1 + term3
	
def Ehf(B, F, mF):
	return EhfFieldInTesla(1E-4 *B, F, mF)

def FreqMHz(B, F1, mF1, F2, mF2):
  return 1E-6 *( Ehf(B, F1, mF1) - Ehf(B, F2, mF2))/h

def B_from_FreqMHz(freq, Bguess=202.1, qn=[9/2, -9/2, 9/2, -7/2]):
	return fsolve(lambda B: FreqMHz(B, *qn) + freq, Bguess)[0]

# def Gaussian(x, A, x0, sigma, C):
# 	return A * np.exp(-(x-x0)**2/(2*sigma**2)) + C

# def Lorentzian(x, A, x0, sigma, C):
# 	return A /((x-x0)**2 + (sigma/2)**2) + C

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


# %%
