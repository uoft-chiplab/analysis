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
				"axes.edgecolor":'black',
				"scatter.edgecolors":'black',
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
colors = ["blue", "orange", "green", "red", 
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

def OmegaRcalibration():
	"""
	Returns function that interpolates the recent calibration from 
	VVAtoVpp.txt which should be in the root of the analysis folder.
	Input of function is VVA, output is OmegaR in kHz.
	"""
	try: 
		VVAtoVppfile = os.path.join("VVAtoVpp.txt") # calibration file
	except:
		FileNotFoundError("VVAtoVpp.txt not found. Check CWD or that file exists.")
	VVAs, Vpps = np.loadtxt(VVAtoVppfile, unpack=True)
	VpptoOmegaR = 27.5833 # kHz
	OmegaR_interp = lambda x: VpptoOmegaR*np.interp(x, VVAs, Vpps)
	
	return OmegaR_interp

def ChipBlackman(x, a_n=[0.42659, 0.49656, 0.076849]):
	"""The ChipLab Blackman that exists in the pulse generation 
	MatLab script. Coefficients slightly differ from conventional.
	Defined as a pulse with length 1 starting at 0."""
	zero_func = lambda y: 0
	pulse_func = lambda y: a_n[0] - a_n[1]*np.cos(2*np.pi*y) \
		+ a_n[2]*np.cos(4*np.pi*y)
	return np.piecewise(x, [x<0, x>1, (x>=0) & (x<=1)], 
					 [zero_func, zero_func, pulse_func])

def ChipKaiser(x, a_n=[0.54]):
	"""The ChipLab Kaiser that exists in the pulse generation 
	MatLab script. Coefficients slightly differ from conventional.
	Defined as a pulse with length 1 starting at 0."""
	zero_func = lambda y: 0
	pulse_func = lambda y: a_n[0] - (1-a_n[0])*np.cos(2*np.pi*y)
	return np.piecewise(x, [x<0, x>1, (x>=0) & (x<=1)], 
					 [zero_func, zero_func, pulse_func])

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
