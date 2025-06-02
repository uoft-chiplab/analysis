# -*- coding: utf-8 -*-
"""
2023-09-25
@author: Chip Lab

Functions to call in analysis scripts
"""
import os
current_dir = os.path.dirname(__file__)

from scipy.constants import pi, hbar, h, k as kB
from scipy.integrate import simpson, cumulative_trapezoid
from scipy.optimize import fsolve
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mc
import colorsys

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
markers = ["o", "s", "^", "D", "h",  "o", "s", "^", "D", "h"]
	
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
				"lines.marker":"o",
				"lines.markeredgewidth": 2,
				"figure.dpi": 300}

paper_settings = {
				'font.size': 8,          # Base font size
				'axes.labelsize': 8,       # Axis label font size
				'axes.titlesize': 8,       # Title font size (if used)
				'xtick.labelsize': 7,      # Tick label font size (x-axis)
				'ytick.labelsize': 7,      # Tick label font size (y-axis)
				'legend.fontsize': 7,      # Legend font size
				'figure.dpi': 300,        # Publication-ready resolution
				'lines.linewidth': 1,      # Thinner lines for compactness
				"lines.linestyle":'',
				'axes.linewidth': 0.5,      # Thin axis spines
				'xtick.major.width': 0.5,    # Tick mark width
				'ytick.major.width': 0.5,
				'xtick.direction': 'in',     # Ticks pointing inward
				'ytick.direction': 'in',
				'xtick.major.size': 3,      # Shorter tick marks
				'ytick.major.size': 3,
				'font.family': 'sans-serif',
				# 'text.usetex': True,       # Use LaTeX for typesetting, needs local LaTeX install
				'axes.grid': False,       # No grid for PRL figures}
				}

# plot color and markers
colors = ["blue", "orange", "green", "red", 
		  "purple", "teal", "pink", "brown",
		  "khaki", "silver", "chocolate", "chartreuse"]

# matplotlib default colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

light_colors = []
dark_colors = []

# Maggie Wang colors
MW_colors = ['hotpink', 'cornflowerblue']
MW_light_colors = []
MW_dark_colors = []

tintshade = 0.6

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

for color in colors:
	light_colors.append(tint_shade_color(color, amount=1+tintshade))
	dark_colors.append(tint_shade_color(color, amount=1-tintshade))
	
for MW_color in MW_colors:
	MW_light_colors.append(tint_shade_color(MW_color, amount=1+tintshade/2))
	MW_dark_colors.append(tint_shade_color(MW_color, amount=1-tintshade/2))
	
styles = [{'color':dark_color, 'mec':dark_color, 'mfc':light_color,
					 'marker':marker} for dark_color, light_color, marker in \
						   zip(dark_colors, light_colors, markers)]
	
MW_styles = [{'color':dark_color, 'mec':dark_color, 'mfc':light_color,
					 'marker':marker} for dark_color, light_color, marker in \
						   zip(MW_dark_colors, MW_light_colors, markers)]
	
def generate_plt_styles(colors, markers=markers, ts=tintshade):
	""" Generates style dictionary for use in plt.plot and plt.errorbar """
	light_colors = [tint_shade_color(color, amount=1+ts) for color in colors]
	dark_colors = [tint_shade_color(color, amount=1-ts) for color in colors]
	styles = [{'color':dark_color, 'mec':dark_color, 'mfc':light_color,
					 'marker':marker} for dark_color, light_color, marker in \
						   zip(dark_colors, light_colors, markers)]
	return styles
	

def set_marker_color(color):
	"""
	Sets marker colors s.t. the face color is light and the edge color is like
	a la standard published plot schemes.
	"""
	light_color = tint_shade_color(color, amount=1+tintshade)
	dark_color = tint_shade_color(color, amount=1-tintshade)
	plt.rcParams.update({"lines.markeredgecolor": dark_color,
				   "lines.markerfacecolor": light_color,
				   "lines.color": dark_color})
	
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
	
def VVAtoVppInterpolation(file):
	"""Returns interpolation function based on VVA to Vpp file."""
	VVAs, Vpps = np.loadtxt(file, unpack=True)
	interp_func = lambda x: np.interp(x, VVAs, Vpps)
	return interp_func
	
def save_to_Excel(filename, df, sheet_name='Sheet1', mode='replace'):
	try: # to open save file, if it exists
		if mode == 'replace':
			# open file and write new df
			with pd.ExcelWriter(filename, mode='a', if_sheet_exists='overlay', \
					engine='openpyxl') as writer:
				print("Saving results to " + filename)
				df.to_excel(writer, index=False, sheet_name=sheet_name)
				
		if mode == 'overwrite':	
			existing_df = pd.read_excel(filename, sheet_name=sheet_name)
			raise ValueError(mode + " mode not implemented yet")
						
	except PermissionError:
		print("Can't write to Excel file " + filename + ".")
		print('Is the .xlsx file open?')
		print()
	except FileNotFoundError: # there is no save file
		print("Save file does not exist.")
		print("Creating file " + filename + " and writing header")
		df.to_excel(filename, index=False, sheet_name=sheet_name)
		 
def quotient_propagation(f, A, B, sA, sB, sAB):
	return f* (sA**2/A**2 + sB**2/B**2 - 2*sAB/A/B)**(1/2)


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
	integrated with simpsonons rule
	"""
	return [np.trapz(data[:,1], x=data[:,0]), 
		 cumulative_trapezoid(data[:,1], x=data[:,0])[-1],
		 simpson(data[:,1], x=data[:,0])]

def FirstMoment(data):
	"""
	integrated with simpsonons rule
	"""
	return [np.trapz(data[:,1]*data[:,0], x=data[:,0]), 
		 cumulative_trapezoid(data[:,1]*data[:,0], x=data[:,0])[-1],
		 simpson(data[:,1]*data[:,0], x=data[:,0])]

def tail3Dswave(w, C, gamma):
	return C*w**gamma

def guessACdimer(field):
	return -0.1145*field + 27.13 # MHz

def a97(B, B0=202.14, B0zero=209.07, abg=167.6*a0): 
	return abg * (1 - (B0zero - B0)/(B - B0));

def BlackmanFourier2(omega):
	A = 1060.9629086785837
	B = -3.5209670498557566
	C = 0.002744323946881455
	D = 6234.181826176155
	E = -197.39208802178717
	return np.abs((2 *np.sin(omega/2) * (A+B*omega**2 + C*omega**4))/ \
		(D*omega + E*omega**3 + omega**5))**2