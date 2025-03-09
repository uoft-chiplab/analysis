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
from fit_functions import Sinc2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np

plt.rcParams.update({"figure.figsize": [5,3.5]})

FB = False
FB_val = 3.309
file = "2025-02-25_S_e.dat"

fit_func = Sinc2

if FB == True:
	names = ['FB', 'fraction95']
	guess = [0.5, FB_val, 0.1, 0]
else:
	names = ['VVA', 'fraction95']
	guess = [1, 2, 1, 0]

print("--------------------------")
Data(file).fit(fit_func, names, guess=guess)