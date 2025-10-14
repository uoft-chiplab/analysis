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
from fit_functions import ScatteringRate
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

plt.rcParams.update({"figure.figsize": [5,3.5]})

FM_val = 123
file = "2025-10-10_D_e.dat"

avg_counts = 6000
s0 = 0.123 * avg_counts/1000

fit_func = partial(ScatteringRate, s0=s0)

names = ['K FM', 'LiNfit']
guess = [80000, 123]
scan = names[0]

print("--------------------------")
Data(file).fit(fit_func, names, guess=guess)