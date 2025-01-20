# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:32:14 2025
Rough summary of dimer binding energies across field
determined experimentally, compared to theory.
Values are extracted from lab book end of October and start of November.
Measurements were originally performed to see if field affected balance
of spin loss. 
@author: coldatoms
"""

# paths

import os
import sys
# this is a hack to access modules in the parent directory
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
	sys.path.append(parent_dir)
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
data_path = os.path.join(proj_path, 'data')

from library import pi, h, hbar, mK, a0, plt_settings, styles, colors, FreqMHz
from data_helper import remove_indices_formatter
from data_class import Data
from rfcalibrations.Vpp_from_VVAfreq import Vpp_from_VVAfreq
from scipy.optimize import curve_fit
from warnings import catch_warnings, simplefilter
from cycler import Cycler, cycler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

Pickle = True

### ac binding energy theory
def a13(B):
	''' ac scattering length '''
	abg = 167.6*a0
	DeltaB = 7.2
	B0 = 224.2
	return abg*(1 - DeltaB/(B-B0))

re = 107*a0 # fixed, I think close to initial channel re

def EbMHz_full_sol(B, re):
	f = lambda x: 1 - 2/pi*np.arctan(pi*x*re/4) - 1/(x*a13(B))
	kappa = fsolve(f, 1e7)[0]
	Eb = -hbar**2 * kappa**2 / mK
	EbMHz = Eb / h / 1e6
	return EbMHz

def EbMHz_expansion_corr(B, re, order=1):
	if order == 1:
		kappa = 1/a13(B) * (1 + 1/2*re/a13(B))
	elif order == 2:
		kappa = 1/a13(B) * (1 + 1/2*re/a13(B) + 1/2*re**2/a13(B)**2)
	Eb = -hbar**2 * kappa**2 / mK
	EbMHz = Eb / h / 1e6
	return EbMHz

def EbMHz_naive(B):
	Eb = -hbar**2 / mK / a13(B)**2
	EbMHz = Eb / h / 1e6
	return EbMHz

Bs = np.linspace(200, 224, 100)
Ebs_full = [EbMHz_full_sol(B, re) for B in Bs]
Ebs_o1 = EbMHz_expansion_corr(Bs, re, 1)
Ebs_o2 = EbMHz_expansion_corr(Bs, re, 2)
Ebs_naive = EbMHz_naive(Bs)

Eb_df = pd.DataFrame(
	{
		'B':Bs,
		'Ebs_full':Ebs_full,
		'Ebs_o1':Ebs_o1,
		'Ebs_o2':Ebs_o2,
		'Ebs_naive':Ebs_naive
	}
)

### smattering of experimental data
files = ["2024-10-30_B_e",
		 "2024-10-30_D_e",
		 "2024-11-01_F_e",
		 "2024-11-01_H_e",
		 "on resonance"
		 ]
fields = np.array([209,
		  204,
		  207,
		  211,
		  202.14
		  ])

freqs = [45.441,
		 43.797,
		 44.773,
		 46.150,
		 43.248 # from memory
		 ] # center freq in MHz
ress = FreqMHz(fields, -9/2, -7/2, -9/2, -5/2)
Ebs = freqs-ress
e_Ebs = [0.002,
		   0.002,
		   0.002,
		   0.002,
		   0.002
		   ] # Gaussian fit error in MHz

ExpEb_df = pd.DataFrame(
	{
		'B':fields,
		'file':files,
		'freq':freqs,
		'res':ress,
		'Eb':Ebs,
		'e_Eb':e_Ebs
	}
)

if Pickle:
	Eb_df.to_pickle('./clockshift/analyzed_data/Ebs.pkl')
	ExpEb_df.to_pickle('./clockshift/analyzed_data/ExpEbs.pkl')

### PLOTTING
fig, ax = plt.subplots()
ax.errorbar(fields, Ebs, e_Ebs, **styles[0], label="Experiment")
ax.plot(Bs, Ebs_full, '-', label="full solution")
ax.plot(Bs, Ebs_o1, '-', label = "To order 1")
ax.plot(Bs, Ebs_o2, '-', label= "To order 2")
ax.plot(Bs, Ebs_naive, '-', label="1/a^2")
ax.legend()
ax.set(xlabel='B [G]', ylabel='Eb [MHz]')

plt.show()

