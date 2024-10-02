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

plt.rcParams.update({"figure.figsize": [5,3.5]})

FB = True
guess = -2.88
files = ["2024-09-25_G_e.dat"]
fit_func = Sinc2

if FB == True:
	names = ['FB', 'fraction95']
	guess = [0.5, guess, 0.1, 0]
else:
	names = ['VVA', 'fraction95']
	guess = [0.1, guess, 1, 0]

for file in files:
	print("--------------------------")
	Data(file).fit(fit_func, names, guess=guess)
