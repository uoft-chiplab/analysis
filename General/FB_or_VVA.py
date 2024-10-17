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

FB = False
guess = 7.13
files = ["2024-10-02_I_e.dat"]
fit_func = Sinc2

if FB == True:
	names = ['FB', 'fraction95']
	guess = [0.5, guess, 0.1, 0]
else:
	names = ['VVA', 'fraction95']
	guess = [0.1, 2, 1, 0]

for file in files:
	print("--------------------------")
	Data(file).fit(fit_func, names, guess=guess)
