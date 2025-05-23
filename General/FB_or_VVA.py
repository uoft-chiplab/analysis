# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:19:19 2024

@author: coldatoms
"""
# paths
import os
import sys
proj_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(proj_path)
if root not in sys.path:
	sys.path.insert(0, root)

from data_class import Data
from fit_functions import Sinc2, Parabola
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": [5,3.5]})

scan = 'VVA'
FB_val = 3.3
file = "2025-04-16_F_e.dat"

fit_func = Sinc2

if scan == 'FB':
	names = ['FB', 'fraction95']
	guess = [0.5, FB_val, 0.1, 0]
elif scan == 'VVA':
	names = ['VVA', 'fraction95']
	guess = [1, 2, 1, 0]
elif scan == 'grad':
	fit_func = Parabola
	names = ['grad', 'sum95']
	guess = None

print("--------------------------")
Data(file).fit(fit_func, names, guess=guess)