# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:19:19 2024

@author: coldatoms
"""
# paths
import os
import sys
import numpy as np
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.dirname(os.path.dirname(proj_path)))
if root not in sys.path:
	sys.path.insert(0, root)

from data_class import Data
from fit_functions import Sinc2, Parabola, Sin, Linear
import matplotlib.pyplot as plt
from scipy.stats import sem

plt.rcParams.update({"figure.figsize": [5,3.5]})

scan ='freq'
centre_freq = 45.89
FB_val = 7.1
file = "2026-08-07_H_e.dat"


if (scan == 'p1') :
	names = ['p1', 'fraction95']
	guess = [0.25, FB_val, 0.05, 0]
	fit_func=Sinc2
elif scan == 'VVA':
	names = ['VVA', 'fraction95']
	guess = [1,2.1,0.1]
	fit_func=Parabola

elif scan == 'freq':
	names = ['freq', 'fraction95']
	guess = [0.25, centre_freq, 0.01, 0]
	fit_func = Sinc2

elif scan == '5050':
	names = ['5050', 'fraction95']
	guess = None
	fit_func = Linear

elif scan == 'nopulses':
	fit_func = Sin
	names = ['VVA', 'c9']
	guess = [30000, 2,0,15000]
elif scan == 'grad':
	fit_func = Parabola
	names = ['grad', 'sum95']
	guess = None
else:
	names = ['57 freq', 'fraction95']
	guess = [9000, 45.96, 0.05, 0]
	fit_func=Sinc2

print("--------------------------")
run = Data(file)
run.fit(fit_func, names, guess=guess)

if scan == '5050':
	ff = np.mean(run.data['c5']/run.data['c9'])
	e_ff = sem(run.data['c5']/run.data['c9'])
	print(f"Fudge factor: {ff:.3f}({e_ff*1e3:.0f})")